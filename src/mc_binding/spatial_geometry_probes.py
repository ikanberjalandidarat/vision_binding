"""Exploratory within-world geometry probes; frozen cache, fixed hyperparameters.

Pose supervises the readouts, never the encoder. Two-route evaluation provides
no independent validation route. Seed repeats are not independent datasets.
"""
import argparse
import csv
import hashlib
import html
import json
from pathlib import Path
import numpy as np
from .io import atomic_json
from .spatial_video_dataset import load_diverse_v2
from .spatial_video_within_world import FEATURES


def pairs(y):
    i,j=np.triu_indices(len(y),1)
    return i,j,np.linalg.norm(y[i]-y[j],axis=1)


def scalar_fit(a,b):
    return max(0.,float(a@b/max(a@a,1e-12)))


def fit_probe(x,y,method,k=2,seed=0,steps=200):
    """All preprocessing/scale learned on training rows; scalar feature scaling."""
    import torch
    if len(x)<4 or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Need >=4 finite training rows')
    center=x.mean(0)
    scale=float(np.sqrt(np.mean((x-center)**2)))
    if scale<1e-10: raise ValueError('Degenerate training features')
    a=(x-center)/scale
    yc=y.mean(0)
    i,j,target=pairs(y)
    if target.max()<1e-8: raise ValueError('No training position variation')
    out=dict(center=center,scale=scale,method=method)
    if method=='mean_distance':
        return dict(out,constant=float(target.mean()))
    if method=='ridge':
        # Fixed alpha=1.0 in this training-only scalar-normalized feature basis.
        w=a.T@np.linalg.solve(a@a.T+np.eye(len(a)),y-yc)
        return dict(out,w=w,b=yc)
    if method=='raw':
        return dict(out,s=scalar_fit(np.linalg.norm(a[i]-a[j],axis=1),target))
    if k>x.shape[1]: raise ValueError('Projection wider than input')
    rng=np.random.default_rng(seed)
    q=np.linalg.qr(rng.normal(size=(x.shape[1],k)))[0]
    if method=='random_orth':
        d=np.linalg.norm((a[i]-a[j])@q,axis=1)
        return dict(out,w=q,s=scalar_fit(d,target))
    torch.manual_seed(seed)
    delta=torch.tensor(a[i]-a[j],dtype=torch.float64)
    # Target scale learned only from training positions; stabilizes optimization.
    unit=float(np.sqrt(np.mean(target**2)))
    truth=torch.tensor(target/unit,dtype=torch.float64)
    w=torch.nn.Parameter(torch.tensor(q,dtype=torch.float64))
    initial=scalar_fit(np.linalg.norm((a[i]-a[j])@q,axis=1),target/unit)
    logscale=torch.nn.Parameter(torch.tensor(np.log(max(initial,1e-8)),dtype=torch.float64))
    if method=='linear_metric':
        with torch.no_grad(): w.mul_(initial)
        parameters=[w]
    elif method=='orth': parameters=[w,logscale]
    else: raise ValueError(method)
    opt=torch.optim.Adam(parameters,lr=.01)
    for _ in range(steps):
        opt.zero_grad()
        d=torch.linalg.vector_norm(delta@w,dim=1)
        if method=='orth': d=d*logscale.exp()
        loss=((d-truth)**2).mean()
        if method=='linear_metric': loss=loss+1e-3*(w*w).sum()
        loss.backward(); opt.step()
        if method=='orth':
            with torch.no_grad():
                q,r=torch.linalg.qr(w,mode='reduced')
                signs=torch.where(torch.diag(r)<0,-1.,1.)
                w.copy_(q*signs)
    weight=w.detach().numpy()
    if not np.isfinite(weight).all(): raise ValueError('Nonfinite fitted projection')
    if method=='orth' and not np.allclose(weight.T@weight,np.eye(k),atol=1e-8):
        raise ValueError('Orthonormal constraint failed')
    return dict(out,w=weight,s=unit*(float(logscale.detach().exp()) if method=='orth' else 1.))


def project(model,x):
    a=(x-model['center'])/model['scale']
    if model['method']=='raw': return a*model['s']
    if model['method']=='mean_distance': return None
    if model['method']=='ridge': return a@model['w']+model['b']
    return (a@model['w'])*model['s']


def evaluate(model,train_x,train_y,test_x,test_y,heading):
    i,j,true=pairs(test_y)
    z=project(model,test_x)
    pred=np.full(len(i),model['constant']) if z is None else np.linalg.norm(z[i]-z[j],axis=1)
    yaw=np.abs((heading[i]-heading[j]+180)%360-180)
    if not np.isfinite(pred).all(): raise ValueError('Nonfinite test predictions')
    error=np.abs(pred-true)
    result={'test_pairs':len(i),'distance_mae_blocks':float(error.mean()),
            'distance_rmse_blocks':float(np.sqrt(np.mean((pred-true)**2)))}
    for name,mask in [('near',true<=2),('far',true>=6),('opposite_heading',yaw>=90)]:
        result[name+'_pairs']=int(mask.sum())
        result[name+'_mae_blocks']=float(error[mask].mean()) if mask.any() else None
    if model['method']=='ridge':
        result['coordinate_error_blocks']=float(np.linalg.norm(z-test_y,axis=1).mean())
    if z is not None:
        keys=project(model,train_x)
        choice=np.linalg.norm(z[:,None]-keys[None,:],axis=-1).argmin(1)
        spatial=np.linalg.norm(test_y[:,None]-train_y[None,:],axis=-1)
        result.update(retrieval_error_blocks=float(spatial[np.arange(len(z)),choice].mean()),
                      retrieval_oracle_blocks=float(spatial.min(1).mean()),
                      retrieval_random_blocks=float(spatial.mean()),
                      near_key_queries=int((spatial.min(1)<=2).sum()))
    return result,dict(pair_i=i,pair_j=j,true_distance=true,predicted_distance=pred,
                        test_projection=z if z is not None else np.empty((len(test_x),0)))


def load_cache(extraction,dataset):
    status=json.loads((extraction/'status.json').read_text())
    manifest=json.loads((extraction/'manifest.json').read_text())
    rows=json.loads((extraction/'clips.json').read_text())
    entries,provenance=load_diverse_v2(dataset)
    if status!={'state':'complete','clips':len(rows)} or len(rows)!=188:
        raise ValueError('Require complete audited 188-clip cache')
    if not manifest.get('frozen') or not manifest.get('raw_png_hashes_verified') or manifest.get('dataset_sha256')!=provenance['dataset_sha256']:
        raise ValueError('Frozen/verified dataset provenance mismatch')
    poses={(e['episode'],f['elapsed_ticks']):[f['pose']['x'],f['pose']['z']] for e in entries for f in e['frames']}
    data={key:[] for key in FEATURES}
    for r in rows:
        if not np.allclose(poses[(r['episode'],r['ticks'][-1])],r['position']):
            raise ValueError('Pose alignment failed')
        path=(extraction/r['cache']).resolve()
        if extraction.resolve() not in path.parents: raise ValueError('Unsafe cache path')
        with np.load(path,allow_pickle=False) as cache:
            for key in FEATURES:
                v=np.asarray(cache[key],dtype=float)
                if v.shape!=(1024,) or not np.isfinite(v).all(): raise ValueError('Invalid feature')
                data[key].append(v)
    return rows,{k:np.stack(v) for k,v in data.items()},manifest


def run(args):
    import torch
    torch.set_num_threads(2)
    args.output.mkdir(parents=True,exist_ok=False)
    try:
        atomic_json(args.output/'status.json',{'state':'running'})
        rows,features,manifest=load_cache(args.extraction,args.dataset)
        y=np.asarray([r['position'] for r in rows],float)
        world=np.array([r['geometry'] for r in rows]); route=np.array([r['route'] for r in rows])
        heading=np.asarray([r['heading'] for r in rows],float)
        results=[]
        coverage=[]
        specs=[('mean_distance',0),('raw',0),('ridge',2)]+[(m,k) for k in (2,8) for m in ('random_orth','orth','linear_metric')]
        worlds=sorted(set(world))
        if len(worlds)!=8: raise ValueError('Expected eight worlds')
        for env in worlds:
            left=np.flatnonzero((world==env)&(route==0));right=np.flatnonzero((world==env)&(route==1))
            d=np.linalg.norm(y[left,None]-y[right][None,:],axis=-1)
            h=np.abs((heading[left,None]-heading[right][None,:]+180)%360-180)
            coverage.append(dict(world=env,route0_clips=len(left),route1_clips=len(right),
                near_crossroute_pairs=int((d<=2).sum()),near_opposite_heading_pairs=int(((d<=2)&(h>=90)).sum()),
                far_same_heading_pairs=int(((d>=6)&(h<=45)).sum())))
            atomic_json(args.output/'coverage.json',coverage)
            for train_route in (0,1):
                tr=np.flatnonzero((world==env)&(route==train_route))
                te=np.flatnonzero((world==env)&(route==1-train_route))
                tr=np.array(sorted(tr,key=lambda i:rows[i]['ticks'][-1]))
                te=np.array(sorted(te,key=lambda i:rows[i]['ticks'][-1]))
                if min(len(tr),len(te))<4: raise ValueError('Too few clips per route')
                if set(rows[i]['episode'] for i in tr)&set(rows[i]['episode'] for i in te):
                    raise ValueError('Episode overlap')
                for name,x in features.items():
                    for control in ('true_positions','shifted_training_positions'):
                        train_y=y[tr] if control=='true_positions' else np.roll(y[tr],max(1,len(tr)//2),axis=0)
                        for method,k in specs:
                            seeds=args.seeds if method in ('random_orth','orth','linear_metric') else [0]
                            for seed in seeds:
                                model=fit_probe(x[tr],train_y,method,k,seed,args.steps)
                                scores,arrays=evaluate(model,x[tr],y[tr],x[te],y[te],heading[te])
                                tag=f'{env}-r{train_route}-{name}-{control}-{method}-k{k}-s{seed}'
                                arrays.update(test_row_indices=te,train_row_indices=tr)
                                np.savez_compressed(args.output/(tag+'.npz'),**arrays)
                                results.append(dict(world=env,train_route=train_route,test_route=1-train_route,
                                    representation=name,control=control,method=method,k=k,seed=seed,
                                    train_clips=len(tr),test_clips=len(te),artifact=tag+'.npz',**scores))
                    atomic_json(args.output/'progress.json',dict(world=env,train_route=train_route,representation=name,completed_fits=len(results)))
                    print(env,train_route,name,'fits',len(results),flush=True)
        report=dict(state='complete',runtime={'torch':torch.__version__,'numpy':np.__version__},protocol='within_world_geometry_fixed_v1',source_manifest=manifest,
            source_clips_sha256=hashlib.sha256((args.extraction/'clips.json').read_bytes()).hexdigest(),
            runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            config={'steps':args.steps,'seeds':args.seeds,'ridge_alpha':1.,'metric_regularization':.001,'learning_rate':.01,
                    'preprocessing':'Training feature mean and single scalar RMS; no per-channel whitening'},
            target='Ground-plane X,Z Euclidean geometry; elevation and geodesic distance are not tested',
            primary='Test-route pair distance MAE; every test pair has two unseen endpoint embeddings',
            secondary='Test-to-training-route nearest-neighbor retrieval in fitted space',
            limitations=['Exploratory: sparse clips and only two routes per world; no independent validation route',
                'Fixed settings, all layers and ranks reported; no best-test selection',
                '1024 input dimensions and very few training clips: strong overfitting risk, including orthonormal probes',
                'Metric and coordinate probes use pose supervision; encoder remains frozen',
                'Test pairs within a route are dependent; no IID pair confidence intervals',
                'Reverse folds and optimization seeds are not independent replicates',
                'Previously inspected data, not a new confirmatory test set',
                'Shifted training positions are a diagnostic control, not a significance test',
                'Ground-plane distance hides elevation in hills/mines',
                'No grid-cell, persistent-memory or navigation-utility claim'],results=results)
        atomic_json(args.output/'report.json',report)
        fields=list(dict.fromkeys(k for r in results for k in r))
        with (args.output/'results.csv').open('w') as stream:
            writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(results)
        summaries=[]
        for name in FEATURES:
            for method,k in specs:
                subsets={c:[r['distance_mae_blocks'] for r in results if r['representation']==name and r['method']==method and r['k']==k and r['control']==c] for c in ('true_positions','shifted_training_positions')}
                summaries.append(dict(representation=name,method=method,k=k,**{c:float(np.mean(v)) for c,v in subsets.items()}))
        atomic_json(args.output/'summary.json',{'scope':'Equal-weight world/fold/seed mean of distance MAE; no independent-replicate interpretation','rows':summaries})
        lookup={(r['representation'],r['method'],r['k']):r for r in summaries}
        comparisons=[]
        for layer in (3,11,23):
            for method,k in specs:
                vk=(f'video_layer{layer}',method,k);rk=(f'repeat_last_frame_layer{layer}',method,k)
                if vk in lookup and rk in lookup:
                    comparisons.append(dict(layer_zero_based=layer,method=method,k=k,video_minus_repeated_distance_mae=lookup[vk]['true_positions']-lookup[rk]['true_positions']))
        atomic_json(args.output/'temporal-comparison.json',{'interpretation':'Negative difference favors temporal video; descriptive, not a significance test','rows':comparisons})
        table=''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in r.values())+'</tr>' for r in summaries)
        (args.output/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Geometry probes</title><style>body{font:16px system-ui;margin:35px}td,th{padding:8px;border-bottom:1px solid #ddd}table{border-collapse:collapse}</style><h1>Within-world geometry probes — exploratory</h1><p>Lower distance MAE is better. Fixed settings; all eight worlds, both route directions. Pose supervises readouts only.</p><p>'+html.escape(' '.join(report['limitations']))+'</p><table><tr><th>Representation</th><th>Method</th><th>Rank</th><th>True-label MAE</th><th>Shifted-label MAE</th></tr>'+table+'</table>')
        atomic_json(args.output/'status.json',{'state':'complete','fits':len(results)})
    except BaseException as exc:
        atomic_json(args.output/'status.json',{'state':'error','error':repr(exc)});raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('extraction',type=Path);p.add_argument('dataset',type=Path);p.add_argument('output',type=Path)
    p.add_argument('--steps',type=int,default=200)
    p.add_argument('--seeds',type=int,nargs='+',default=[0,1,2])
    a=p.parse_args()
    if a.steps<1: p.error('steps must be positive')
    run(a)
