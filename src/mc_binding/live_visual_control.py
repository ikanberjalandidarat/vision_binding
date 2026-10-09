"""Live RGB visual servo: image proposals, frozen Qwen selection, optional memory.
No world coordinates or teacher actions enter select/steer. Arena-specific baseline.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from PIL import Image,ImageDraw
from .io import atomic_json,file_hash,source_hash
from .visual_readout import Encoder,Readout,geometry_key
from .perception_study import rasterize,features_for
from .tracking_rescue import FallbackMemory,balanced_panel
from .target_tracking import choose
from .vla import Worker,tokenize
from .rl_navigation import Agent


def proposals(image):
    """Goal-independent chromatic components. Arena prior, not learned localization."""
    a=np.asarray(image.convert('RGB'),dtype=np.float32)/255
    hi=a.max(2);lo=a.min(2)
    mask=(hi-lo>.22)&(hi>.25)&((hi-lo)/(hi+1e-6)>.45)
    # Fixed HUD/hand exclusions; do not use simulator visibility or goal color.
    mask[250:]=False;mask[195:,280:]=False
    visited=np.zeros(mask.shape,bool);regions=[]
    h,w=mask.shape
    for y,x in zip(*np.where(mask)):
        if visited[y,x]:continue
        stack=[(int(y),int(x))];visited[y,x]=True;points=[]
        while stack:
            r,c=stack.pop();points.append((r,c))
            for rr,cc in ((r-1,c),(r+1,c),(r,c-1),(r,c+1)):
                if 0<=rr<h and 0<=cc<w and mask[rr,cc] and not visited[rr,cc]:visited[rr,cc]=True;stack.append((rr,cc))
        if len(points)<35:continue
        yy,xx=zip(*points);regions.append(dict(bbox_raw=[min(xx),min(yy),max(xx)+1,max(yy)+1],pixels=len(points)))
    return sorted(regions,key=lambda r:r['bbox_raw'][0])


class Servo:
    def __init__(self,stop_height=170):self.stop_height=stop_height;self.near=0;self.misses=0
    def action(self,box):
        if box is None:
            self.near=0;self.misses+=1
            # No blind forward continuation. Bounded search, then explicit refusal.
            return (3,'search_exhausted') if self.misses>36 else (2,'search')
        self.misses=0
        x0,y0,x1,y1=box;offset=(x0+x1)/2-224
        if abs(offset)>18:
            self.near=0;return (2 if offset>0 else 1),'align'
        self.near=self.near+1 if y1-y0>=self.stop_height else 0
        if self.near>=2:return 3,'visual_arrival'
        if self.near:return 4,'verify_arrival'
        return 0,'approach'


def task_panel(demos,checkpoint):
    jobs=json.loads((Path(demos)/'episodes.json').read_text());groups=defaultdict(dict)
    for j in jobs:groups[j['family']][j['record']['record_id']]=j['record']
    splits=json.loads((Path(checkpoint).parent/'manifest.json').read_text())['splits']
    return balanced_panel([j for j in jobs if splits[geometry_key(list(groups[j['family']].values()))]=='validation'])


def run(args):
    torch.set_num_threads(4)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    jobs=task_panel(args.demos,args.checkpoint)
    if args.smoke:jobs=jobs[:1]+jobs[-1:]
    arms=['recognition','memory']+(['old_policy'] if args.policy_checkpoint else [])
    scene_seed=json.loads((Path(args.dataset)/'manifest.json').read_text())['seed']
    atomic_json(out/'manifest.json',dict(source_hash=source_hash(),checkpoint_sha256=file_hash(args.checkpoint),tasks=[dict(episode=j['episode'],family=j['family'],goal=j['goal']) for j in jobs],policy_inputs=['live RGB','structured goal'],arms=arms,old_policy_sha256=file_hash(args.policy_checkpoint) if args.policy_checkpoint else None,action_ticks={'recognition':2,'memory':2,'old_policy':4},scene_seed=scene_seed,max_steps=args.max_steps,stop_height=args.stop_height,stop_rule='Centered proposal >= stop_height for two observations, with a stationary verification step. Untuned heuristic, not metric depth.',localizer='goal-independent saturated RGB components; arena-specific prior, HUD exclusion',scope='New hand-designed visual servo with trained readout; not RL improvement, obstacle planning or teacher replay. Simulator metadata only builds scenes and scores completed episodes. No oracle boxes.',test_evaluated=False))
    atomic_json(out/'status.json',dict(state='running'));worker=None;totals=[]
    try:
        config=json.loads(Path(args.config).read_text());enc=Encoder(config)
        ck=torch.load(args.checkpoint,map_location='cpu',weights_only=True);readout=Readout(ck['width']);readout.load_state_dict(ck['state']);readout.eval()
        old_model=None
        if args.policy_checkpoint:
            old_ck=torch.load(args.policy_checkpoint,map_location='cpu',weights_only=True)
            old_model=Agent(old_ck['width'],old_ck['hidden']);old_model.load_state_dict(old_ck['state']);old_model.eval()
        for arm in arms:
            armout=out/arm;armout.mkdir();chunk=4 if arm=='old_policy' else 2;worker=Worker(args.render_python,armout/'episodes',chunk)
            for job in jobs:
                eid=job['episode'];memory=FallbackMemory(ck['threshold']);servo=Servo(args.stop_height);decisions=[];gif=[]
                obs=worker.request('reset',record=job['record'],goal=job['goal'],seed=scene_seed,episode=eid,reference=str((Path(args.dataset)/job['record']['image']).resolve()))
                stopped=False;state=None;previous=3
                for step in range(max(1,args.max_steps*2//chunk)):
                    image=Image.open(obs['frame']).convert('RGB');regions=proposals(image)
                    if regions or arm=='old_policy':
                        inp=enc.inputs(image);_,h,w=map(int,inp[1][0].tolist())
                        raster=rasterize(enc.features(inp),h,w,int(enc.visual.spatial_merge_size))
                        features=features_for(raster,regions,image.size,'8x16') if regions else torch.empty((0,ck['width']))
                        with torch.no_grad():scores=readout.goal_scores(features,job['goal'])
                    else:features=torch.empty((0,ck['width']));scores=torch.empty(0)
                    if arm=='memory':selected,source=memory.select(features,scores)
                    else:selected=choose(features,scores,ck['threshold']);source='recognition' if selected>=0 else 'unrecognized'
                    action,reason=servo.action(regions[selected]['bbox_raw'] if selected>=0 else None)
                    if arm=='old_policy':
                        pooled=torch.nn.functional.adaptive_avg_pool2d(raster.permute(2,0,1)[None],(8,16))[0].permute(1,2,0).reshape(128,-1)
                        with torch.no_grad():logits,state,_=old_model(pooled,tokenize(job['instruction']),previous,state)
                        action=int(logits.argmax());previous=action;reason='old_policy_greedy';selected=-1;source='old_policy (no region selector)'
                    decisions.append(dict(step=step,frame=obs['frame'],regions=regions,scores=scores.tolist(),selected=selected,source=source,action=action,reason=reason,memory=memory.diagnostics if arm=='memory' else None))
                    panel=Image.new('RGB',(448,340),'white');panel.paste(image,(0,60));d=ImageDraw.Draw(panel)
                    d.text((5,4),f'LIVE {arm} | Goal: {" ".join(job["goal"])} | step {step}',fill='black')
                    d.text((5,21),f'{source} | {reason} | action {action}',fill='black')
                    d.text((5,38),'0 forward / 1 left / 2 right / 3 stop / 4 wait',fill='black')
                    for i,r in enumerate(regions):
                        x0,y0,x1,y1=r['bbox_raw'];d.rectangle((x0,y0+60,x1,y1+60),outline='cyan' if selected==i else 'gray',width=3 if selected==i else 1)
                    gif.append(panel)
                    if action==3:stopped=True;break
                    obs=worker.request('wait') if action==4 else worker.request('step',index=action)
                # Evaluator-only telemetry is returned after all decisions are finished.
                result=worker.request('finish',stopped=stopped)
                # Post-episode scoring only; these values never influence actions.
                from .approach import waypoint
                import math
                telemetry=json.loads(Path(result['episode_file']).read_text())
                pose=telemetry['trajectory'][-1]['pose']
                wrong_distances=[math.hypot(float(pose['x'])-waypoint(o)[0],float(pose['z'])-waypoint(o)[1]) for o in telemetry['objects'] if (o['color'],o['type'])!=tuple(job['goal'])]
                result['stopped_at_wrong_waypoint']=bool(stopped and wrong_distances and min(wrong_distances)<=.8)
                closest=result['minimum_sampled_goal_distance']
                result['entered_then_left']=bool(closest is not None and closest<=.8 and result['distance']>.8)

                episode=armout/'episodes'/eid
                atomic_json(episode/'decisions.json',decisions)
                gif[0].save(episode/'selection-control.gif',save_all=True,append_images=gif[1:],duration=chunk*50,loop=0)
                totals.append(dict(arm=arm,episode=eid,goal=job['goal'],family=job['family'],stop_reason=reason,**result))
                atomic_json(out/'results.json',totals);print(arm,eid,result,flush=True)
            worker.close();worker=None
        summary=[dict(arm=a,episodes=len([r for r in totals if r['arm']==a]),successes=sum(r['success'] for r in totals if r['arm']==a),timeouts=sum(r['timeout'] for r in totals if r['arm']==a),by_presence=[dict(present=p,n=sum(r['goal_present']==p for r in totals if r['arm']==a),successes=sum(r['success'] and r['goal_present']==p for r in totals if r['arm']==a)) for p in (True,False)]) for a in arms]
        atomic_json(out/'summary.json',summary)
        body='<h1>LIVE visual selection controls movement</h1><p>Hand-designed image servo; frozen Qwen readout. No target coordinates in control. Compare actual arrival and stopping, not just plausible GIFs. Image-size stopping is an unvalidated heuristic. No obstacles/path planner.</p><pre>'+json.dumps(summary,indent=2)+'</pre>'
        for r in totals:
            path=f'{r["arm"]}/episodes/{r["episode"]}/selection-control.gif'
            body+=f'<h2>{r["arm"]} / {r["episode"]}: success {r["success"]}, {r["stop_reason"]}</h2><img src="{path}"><p>{out/path}</p><a href="{r["arm"]}/episodes/{r["episode"]}/report.html">Evaluator trajectory and coordinates</a>'
        (out/'report.html').write_text('<!doctype html><meta charset="utf-8"><style>body{font:17px system-ui;max-width:1100px;margin:auto}img{max-width:100%}</style>'+body)
        atomic_json(out/'status.json',dict(state='complete'))
    except BaseException as e:atomic_json(out/'status.json',dict(state='error',error=str(e)));raise
    finally:
        if worker:worker.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--demos',required=True);p.add_argument('--dataset',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--render-python',required=True);p.add_argument('--output',required=True);p.add_argument('--config',default='configs/visual_readout.json');p.add_argument('--max-steps',type=int,default=240);p.add_argument('--stop-height',type=int,default=170);p.add_argument('--policy-checkpoint');p.add_argument('--smoke',action='store_true');run(p.parse_args())
