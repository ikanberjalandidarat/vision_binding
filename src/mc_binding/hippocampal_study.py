"""Real Minecraft collection -> frozen visual prediction -> spatial-activity report.
Collection is privileged scripted exploration, NOT a learned navigation policy.
"""
import argparse
import json
import math
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from .io import atomic_json,file_hash,source_hash
from .vla import Worker,observation_image
from .visual_readout import Encoder,geometry_key,save_tensor
from .binding_data import load_binding
from .approach import steering
from .hippocampal import SpatialMemory,odometry,revisit_mask


def route_points(reverse=False):
    # Open floor in front of the structures; twice through the same loop for revisits.
    points=[(x,z) for i,z in enumerate((2.,4.,6.,8.,10.)) for x in ((-6.,6.) if i%2==0 else (6.,-6.))]
    if reverse:points=list(reversed(points))
    return points*2


def collect(a,out):
    data,groups=load_binding(a.dataset)
    selected=[];seen=set()
    for fid,rs in sorted(groups.items()):
        geometry=geometry_key(rs)
        if geometry in seen:continue
        seen.add(geometry)
        record=next(r for r in rs if r['kind']=='full' and r['context']=='recipient')
        if min(o['bounds'][0][2] for o in record['objects'])<13:continue
        selected.append((fid,geometry,record))
        if len(selected)==a.scenes:break
    if len(selected)!=a.scenes or a.scenes<3:raise ValueError('Need >=3 distinct safe Minecraft geometries')
    entries=[];worker=None
    try:
        worker=Worker(a.render_python,out/'episodes',2)
        for k,(fid,geometry,record) in enumerate(selected):
            split='test' if k==len(selected)-1 else 'validation' if k==len(selected)-2 else 'train'
            for route in range(2):
                eid=f'mapping-{fid}-route{route}'
                obs=worker.request('reset',record=record,goal=['red','pillar'],seed=data['seed'],episode=eid,reference=str((Path(a.dataset)/record['image']).resolve()))
                points=route_points(bool(route));j=0;frames=[];actions=[]
                for step in range(a.steps+1):
                    pose=worker.request('mapping_pose')
                    if not(-18<pose['x']<18 and -5<pose['z']<15):raise RuntimeError('Exploration left safe floor bounds')
                    if step%a.stride==0 or step==a.steps or j==len(points):
                        frames.append(dict(frame=str(Path(obs['frame']).relative_to(out)),pose=pose,actions_since_previous=actions,sha256=file_hash(obs['frame'])))
                        actions=[]
                    if step==a.steps or j==len(points):break
                    distance,turn,forward=steering(pose,points[j])
                    if distance<.8:
                        j+=1
                        if j==len(points):continue
                        distance,turn,forward=steering(pose,points[j])
                    action=0 if forward else 1 if turn<0 else 2
                    obs=worker.request('step',index=action);actions.append(action)
                worker.request('finish',stopped=False)
                entry=dict(is_minecraft=True,frame_interval_ms=a.stride*2*50,episode=eid,family=fid,geometry=geometry,split=split,route=route,objects=record['objects'],frames=frames,waypoints_reached=j,waypoints_total=len(points),route_complete=j==len(points))
                entries.append(entry);atomic_json(out/'sequences.json',entries)
                print('Collected',eid,len(frames),'frames; route complete',entry['route_complete'],flush=True)
    finally:
        if worker:worker.close()
    return entries


def extract(a,out,entries):
    config=json.loads(Path(a.config).read_text());config['observation_mode']='neutral_bands_v1'
    encoder=Encoder(config);projection=None;result=[]
    from .perception_study import rasterize
    for entry in entries:
        values=[]
        for frame in entry['frames']:
            path=out/frame['frame']
            if file_hash(path)!=frame['sha256']:raise ValueError('Frame integrity failed')
            with Image.open(path) as image:im=observation_image(image,config)
            inp=encoder.inputs(im);_,h,w=map(int,inp[1][0].tolist())
            raster=rasterize(encoder.features(inp),h,w,int(encoder.visual.spatial_merge_size))
            v=torch.nn.functional.adaptive_avg_pool2d(raster.permute(2,0,1)[None],(2,4)).flatten().float().cpu()
            if projection is None:
                projection=torch.randn(len(v),64,generator=torch.Generator().manual_seed(991))/math.sqrt(len(v))
            values.append(v@projection)
        result.append(dict(episode=entry['episode'],x=torch.stack(values),motion=odometry([f['pose'] for f in entry['frames']])))
        print('Encoded',entry['episode'],flush=True)
    save_tensor(out/'features.pt',dict(sequences=result,projection=projection,config=config))
    return result


def fit(a,out,entries,data):
    torch.set_num_threads(4);torch.manual_seed(a.seed)
    indices=[i for i,e in enumerate(entries) if e['split']=='train' and e['route']==0]
    train_x=torch.cat([data[i]['x'] for i in indices]);mean=train_x.mean(0);scale=train_x.std(0).clamp_min(.1)
    xs=[(e['x']-mean)/scale for e in data]
    model=SpatialMemory();optimizer=torch.optim.Adam(model.parameters(),lr=.001);history=[]
    initial={k:v.detach().clone() for k,v in model.state_dict().items()}
    for epoch in range(a.epochs):
        losses=[]
        for i in torch.randperm(len(indices)).tolist():
            idx=indices[i];prediction,*_=model(xs[idx],data[idx]['motion'])
            loss=(prediction[1:]-xs[idx][1:]).square().mean()
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite prediction loss')
            optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();losses.append(float(loss))
        history.append(dict(epoch=epoch,loss=float(np.mean(losses))))
    save_tensor(out/'memory.pt',dict(state=model.state_dict(),initial=initial,mean=mean,scale=scale,seed=a.seed))
    atomic_json(out/'training.json',history)
    # Unit choices fixed using training episodes only, not aesthetically selected test units.
    with torch.no_grad():
        acts=[model(xs[i],data[i]['motion']) for i in indices]
        selected={name:torch.cat([r[k] for r in acts]).var(0).argsort(descending=True)[:4].tolist() for name,k in [('g',1),('p',2)]}
    rows=[];activations={}
    for i,entry in enumerate(entries):
        poses=[f['pose'] for f in entry['frames']];mask=revisit_mask(poses)
        partition=entry['split'] if entry['split']!='train' or entry['route']==0 else 'heldout_route'
        for condition in ('learned','zero_motion','shuffled_motion','untrained','last_frame'):
            motion=data[i]['motion'].clone()
            if condition=='zero_motion':motion.zero_()
            if condition=='shuffled_motion':
                motion[1:]=motion[1:][torch.randperm(len(motion)-1,generator=torch.Generator().manual_seed(551))]
            if condition=='untrained':
                net=SpatialMemory();net.load_state_dict(initial)
            else:net=model
            with torch.no_grad():
                pred,g,p,_=net(xs[i],motion)
                if condition=='last_frame':pred=torch.cat([torch.zeros_like(xs[i][:1]),xs[i][:-1]])
                error=(pred-xs[i]).square().mean(-1).numpy()
            rows.append(dict(episode=entry['episode'],split=partition,condition=condition,n=len(error)-1,mse=float(error[1:].mean()),revisits=int(mask.sum()),revisit_mse=float(error[mask].mean()) if mask.any() else None))
            if condition=='learned':activations[entry['episode']]=dict(g=g.numpy(),p=p.numpy(),prediction=pred.numpy(),target=xs[i].numpy())
    atomic_json(out/'metrics.json',rows)
    from .hippocampal_report import render
    coverage=render(out,entries,activations,selected)
    review=dict(data_kind='Minecraft' if all(e.get('is_minecraft',False) for e in entries) else 'NON-MINECRAFT synthetic fixture',scope='Odometry-assisted sensory prediction on scripted real Minecraft exploration; not learned navigation or a biological reproduction.',seed=a.seed,units=selected,metrics=rows,coverage=coverage,
        cell_claim='Candidate spatial units only. No place/grid-cell emergence claim; sparse paths and heading/visual confounds require dedicated tests.',
        train_episodes=[entries[i]['episode'] for i in indices],test_used_for_training=False)
    atomic_json(out/'review.json',review)


def main():
    p=argparse.ArgumentParser()
    for name in ('dataset','render-python','output'):p.add_argument('--'+name,required=True)
    p.add_argument('--config',default='configs/visual_readout.json');p.add_argument('--scenes',type=int,default=3)
    p.add_argument('--steps',type=int,default=120);p.add_argument('--stride',type=int,default=4)
    p.add_argument('--epochs',type=int,default=10);p.add_argument('--seed',type=int,default=731)
    a=p.parse_args()
    if min(a.steps,a.stride,a.epochs)<1 or a.steps<a.stride:raise ValueError('Invalid study size')
    out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=False)
    atomic_json(out/'manifest.json',dict(experiment='hippocampal_mapping_pilot_v1',is_minecraft=True,settings=vars(a),source_hash=source_hash(),dataset_manifest_sha256=file_hash(Path(a.dataset)/'manifest.json'),
        privileged='Telemetry drives scripted collection; measured world-axis displacement and yaw change enter memory. Absolute positions are plotting/scoring only.',
        objective='Predict current frozen Qwen features using past features and current motion before writing current frame; no manual labels.',
        visualization='Fixed allocentric axes, empirical activation maps. g/p are candidate model units, not established biological cells.'))
    atomic_json(out/'status.json',dict(state='running'))
    try:
        entries=collect(a,out);data=extract(a,out,entries);fit(a,out,entries,data)
        atomic_json(out/'status.json',dict(state='complete'))
    except BaseException as e:
        atomic_json(out/'status.json',dict(state='error',error=repr(e)));raise

if __name__=='__main__':main()
