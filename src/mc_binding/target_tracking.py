"""Offline Qwen selection/identity diagnostic on real recorded Minecraft sequences.
Geometry supplies candidate regions and evaluation only; this is not autonomous detection.
"""
import argparse
from collections import defaultdict
import html
import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageDraw
from .io import atomic_json, file_hash, source_hash
from .live_projection import calibrate, raycast
from .live_perception import sample_steps
from .visual_readout import Encoder, Readout, geometry_key
from .perception_study import rasterize, features_for
from .rl_pilot import distributed_tasks


def choose(features, scores, threshold, template=None, weight=.5):
    """No object IDs, poses, world attributes or true target are accepted here."""
    if len(scores)==0:return -1
    if not torch.isfinite(features).all() or not torch.isfinite(scores).all():
        raise ValueError('Nonfinite selection input')
    eligible=scores>=threshold
    if not eligible.any():return -1
    ranking=scores.clamp_min(1e-8).log()
    if template is not None:
        ranking=ranking+weight*torch.nn.functional.cosine_similarity(features,template[None],dim=1)
    return int(ranking.masked_fill(~eligible,-torch.inf).argmax())


def candidates(owner):
    """Geometry-only screen safety filters, not filtering by ground-truth color."""
    accepted=[];excluded=[]
    h,w=owner.shape
    for index in sorted(set(owner.ravel())-{-1}):
        ys,xs=np.where(owner==index);box=[int(xs.min()),int(ys.min()),int(xs.max()+1),int(ys.max()+1)]
        reasons=[]
        if len(xs)<40:reasons.append('small')
        if box[0]<=1 or box[1]<=1 or box[2]>=w-1 or box[3]>=h-1:reasons.append('clipped')
        if box[3]>195 or (box[1]<40 and box[2]>280):reasons.append('possible_HUD_or_hand')
        item=dict(index=int(index),bbox_raw=box,pixels=len(xs))
        if reasons:excluded.append(dict(item,reasons=reasons))
        else:accepted.append(item)
    # Never expose world object order as a stable tracking shortcut.
    accepted.sort(key=lambda o:(o['bbox_raw'][0],o['bbox_raw'][1]))
    return accepted,excluded


def prepare(rollout,checkpoint,output,count=16,interval=8,balanced=False):
    if count<2 or interval<1:raise ValueError('Need >=2 episodes and positive sampling interval')
    root=Path(rollout).resolve();out=Path(output).resolve();ck=Path(checkpoint).resolve()
    jobs=json.loads((root/'episodes.json').read_text())
    groups=defaultdict(dict)
    for j in jobs:groups[j['family']][j['record']['record_id']]=j['record']
    splits=json.loads((ck.parent/'manifest.json').read_text())['splits']
    eligible=[]
    for j in jobs:
        key=geometry_key(list(groups[j['family']].values()))
        if key not in splits:raise ValueError('Unknown readout geometry; refuse ambiguous split')
        if splits[key]=='validation':eligible.append(j)
    if balanced:
        from .tracking_rescue import balanced_panel
        selected=balanced_panel(eligible)
    else:selected=distributed_tasks(eligible,count)
    out.mkdir(parents=True,exist_ok=False);atomic_json(out/'status.json',dict(state='preparing'))
    rows=[];calibrations={}
    try:
        for job in selected:
            eid=job['episode'];ep=json.loads((root/'episodes'/eid/'episode.json').read_text())
            cal=calibrate(ep['objects'],ep['trajectory'][0]['pose']);calibrations[eid]=cal
            target=[i for i,o in enumerate(ep['objects']) if (o['color'],o['type'])==tuple(job['goal'])]
            if len(target)>1:raise ValueError('Ambiguous conjunction target')
            for step in sample_steps(len(ep['trajectory']),interval):
                t=ep['trajectory'][step];frame=root/'episodes'/eid/t['frame']
                with Image.open(frame) as im:
                    if im.size!=(448,280):raise ValueError('Unexpected recorded image size')
                owner=raycast(ep['objects'],t['pose'],cal['focal_pixels'],cal['eye_height'])
                regions,excluded=candidates(owner)
                rows.append(dict(episode=eid,family=job['family'],step=step,tick=t['tick'],frame=str(frame),sha256=file_hash(frame),goal=job['goal'],regions=regions,excluded=excluded,target_index=target[0] if target else None,world_present=bool(target),target_projected_pixels=int((owner==target[0]).sum()) if target else 0))
            print('Prepared',eid,flush=True)
        atomic_json(out/'frames.json',rows)
        atomic_json(out/'manifest.json',dict(rollout=str(root),checkpoint_sha256=file_hash(ck),source_hash=source_hash(),episodes=len(selected),balanced=balanced,interval=interval,calibrations=calibrations,split='readout-validation only; test untouched',privileged_regions=True,manual_review_required=False,scope='Recorded teacher trajectories, not new policy navigation. Projection candidates use world geometry; images supply selection and template features. Visibility approximation ignores arena/hand occlusion and view bob. Screen filters and exclusions are reported; no color-based QA.'))
        atomic_json(out/'status.json',dict(state='prepared',frames=len(rows)))
    except BaseException as e:
        atomic_json(out/'status.json',dict(state='error',error=str(e)));raise


def evaluate(audit,checkpoint,config,output):
    root=Path(audit);out=Path(output);ckpath=Path(checkpoint)
    manifest=json.loads((root/'manifest.json').read_text());frames=json.loads((root/'frames.json').read_text())
    if file_hash(ckpath)!=manifest['checkpoint_sha256']:raise ValueError('Readout checkpoint changed')
    for r in frames:
        if file_hash(r['frame'])!=r['sha256']:raise ValueError('Recorded frame changed')
    ck=torch.load(ckpath,map_location='cpu',weights_only=True)
    model=Readout(ck['width']);model.load_state_dict(ck['state']);model.eval()
    out.mkdir(parents=True,exist_ok=False);atomic_json(out/'status.json',dict(state='running'))
    atomic_json(out/'manifest.json',dict(manifest,config=config,memory_weight=.5,template='First model-selected region, frozen within episode; no oracle correction. Not recurrent training.'))
    try:
        enc=Encoder(config);templates={};results=[];gallery=defaultdict(list)
        for r in frames:
            with Image.open(r['frame']) as im:image=im.convert('RGB')
            if r['regions']:
                inp=enc.inputs(image);_,h,w=map(int,inp[1][0].tolist())
                raster=rasterize(enc.features(inp),h,w,int(enc.visual.spatial_merge_size))
                x=features_for(raster,r['regions'],image.size,'8x16')
                with torch.no_grad():scores=model.goal_scores(x,r['goal'])
            else:x=torch.empty((0,ck['width']));scores=torch.empty(0)
            panel=Image.new('RGB',(896,310),'white');pd=ImageDraw.Draw(panel)
            for column,method in enumerate(('frame_only','template_memory')):
                template=templates.get(r['episode']) if method=='template_memory' else None
                selected=choose(x,scores,ck['threshold'],template)
                if method=='template_memory' and template is None and selected>=0:
                    templates[r['episode']]=x[selected].detach().clone()
                indices=[o['index'] for o in r['regions']]
                selected_id=indices[selected] if selected>=0 else None
                target_eligible=r['target_index'] in indices if r['world_present'] else False
                # Out of view is not absence; excluded/occluded targets are not scored as absent.
                correct=(selected_id==r['target_index']) if target_eligible else (selected==-1 if not r['world_present'] else None)
                results.append(dict(episode=r['episode'],family=r['family'],step=r['step'],method=method,selected_index=selected_id,target_index=r['target_index'],world_present=r['world_present'],target_eligible=target_eligible,correct=correct,regions=len(indices),excluded=r['excluded'],scores=scores.tolist()))
                vis=image.copy();draw=ImageDraw.Draw(vis)
                for k,obj in enumerate(r['regions']):draw.rectangle(obj['bbox_raw'],outline='yellow' if k==selected else 'gray',width=3 if k==selected else 1)
                panel.paste(vis,(448*column,30));pd.text((448*column+4,4),f'{method} frame {r["step"]} selected={selected_id}',fill='black')
            gallery[r['episode']].append((r['tick'],panel))
            atomic_json(out/'results.json',results)
        episode_scores=[]
        for method in ('frame_only','template_memory'):
            for eid in gallery:
                rr=[r for r in results if r['method']==method and r['episode']==eid]
                eligible=[r for r in rr if r['target_eligible']]
                first=next((r for r in rr if r['selected_index'] is not None),None)
                episode_scores.append(dict(episode=eid,method=method,eligible_frames=len(eligible),correct_frames=sum(r['correct'] is True for r in eligible),first_selection_correct=(first['selected_index']==first['target_index'] if first and first['world_present'] else None),all_eligible_frames_correct=(all(r['correct'] for r in eligible) if eligible else None)))
        atomic_json(out/'episode-scores.json',episode_scores)
        summary=[]
        for method in ('frame_only','template_memory'):
            rr=[r for r in results if r['method']==method];scored=[r for r in rr if r['correct'] is not None]
            switches=0
            for eid in gallery:
                seq=[r for r in rr if r['episode']==eid]
                switches+=sum(a['selected_index']!=b['selected_index'] for a,b in zip(seq,seq[1:]) if a['selected_index'] is not None and b['selected_index'] is not None)
            summary.append(dict(method=method,frames=len(rr),eligible_present=sum(r['target_eligible'] for r in rr),correct_present=sum(r['correct'] is True and r['world_present'] for r in rr),absent_frames=sum(not r['world_present'] for r in rr),correct_absent=sum(r['correct'] is True and not r['world_present'] for r in rr),excluded_present=sum(r['world_present'] and not r['target_eligible'] for r in rr),identity_switches=switches,note='Switches may be corrections; fewer switches alone is not better. Correlated frames; conditional geometry-based scoring.'))
        atomic_json(out/'summary.json',summary)
        body='<h1>Visual target selection and identity memory</h1><p>Yellow is the predicted selection; gray boxes are privileged geometry proposals. These are recorded teacher movements, not movements controlled by either selector. No activation patches or manual annotations. A frozen feature template is an explicit baseline, not a learned tracker. Unknown visibility is excluded, not labeled world absence.</p><pre>'+html.escape(json.dumps(summary,indent=2))+'</pre>'
        for eid,panels in gallery.items():
            name=eid+'.gif';durations=[max(50,(b[0]-a[0])*50) for a,b in zip(panels,panels[1:])]+[500]
            panels[0][1].save(out/name,save_all=True,append_images=[p[1] for p in panels[1:]],duration=durations,loop=0)
            body+=f'<h2>{eid}</h2><img src="{name}" style="max-width:100%"><p><code>{html.escape(str((out/name).resolve()))}</code></p>'
        (out/'report.html').write_text('<!doctype html><meta charset="utf-8"><style>body{font:17px system-ui;max-width:1200px;margin:40px auto}code{overflow-wrap:anywhere}</style>'+body)
        atomic_json(out/'status.json',dict(state='complete'))
    except BaseException as e:
        atomic_json(out/'status.json',dict(state='error',error=str(e)));raise


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','evaluate']);p.add_argument('--rollout');p.add_argument('--audit');p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True);p.add_argument('--episodes',type=int,default=16);p.add_argument('--interval',type=int,default=8);p.add_argument('--config',default='configs/visual_readout.json');p.add_argument('--balanced',action='store_true');a=p.parse_args()
    if a.stage=='prepare':prepare(a.rollout,a.checkpoint,a.output,a.episodes,a.interval,a.balanced)
    else:evaluate(a.audit,a.checkpoint,json.loads(Path(a.config).read_text()),a.output)

if __name__=='__main__':main()
