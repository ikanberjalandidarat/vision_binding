#!/usr/bin/env python3
"""Recorded Minecraft evidence and action-attribution gallery; no generated scenes."""
import argparse
import html
import json
import math
from pathlib import Path
import sys
import numpy as np
from PIL import Image,ImageDraw
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from mc_binding.navigation_metrics import score_episode


def read(p):return json.loads(Path(p).read_text())

def main(suite):
    suite=Path(suite);out=suite/'showcase';out.mkdir(exist_ok=True)
    arms=['raw-episode','raw-timestep','neutral-episode','neutral-timestep'];allrows=[];examples=[]
    for arm in arms:
        base=suite/('rollout-'+arm);cases=[]
        for r in read(base/'results.json'):
            ep=read(base/'episodes'/r['episode']/'episode.json');metrics=score_episode(ep['trajectory'],ep['waypoint'],ep['stopped'])
            last=ep['trajectory'][-1]['pose'];near=min((math.hypot(last['x']-(o['bounds'][0][0]+o['bounds'][1][0])/2,last['z']-(o['bounds'][0][2]-2)),o['color'],o['type']) for o in ep['objects'])
            category='absent refused immediately' if not r['goal_present'] and metrics['success'] else 'absent but moved' if not r['goal_present'] else 'correct arrival' if metrics['success'] else 'immediate false stop' if metrics['immediate_stop'] else 'wrong object waypoint' if near[0]<=.8 else 'stopped away'
            decisions=read(base/('decisions-'+r['episode'][1:]+'.json'))['decisions']
            cams=np.array([d['action_gradcam'] for d in decisions]);att=np.array([d['cross_attention'] for d in decisions])
            if not np.isfinite(cams).all() or not np.isfinite(att).all():raise ValueError('Nonfinite attribution')
            rec=dict(arm=arm,episode=r['episode'],goal=r['goal'],present=r['goal_present'],category=category,metrics=metrics,
                     n_decisions=len(decisions),zero_cam_frames=int((cams.sum(1)==0).sum()),
                     scope='Grad-CAM at pooled 4x8 features for chosen action; recurrent history held fixed; per-frame normalization')
            allrows.append(rec);cases.append((rec,ep,decisions))
        wanted={'raw-episode':['immediate false stop'],'raw-timestep':['correct arrival','wrong object waypoint','absent but moved'],
                'neutral-episode':[],'neutral-timestep':['correct arrival','stopped away']}[arm]
        for label in wanted:
            candidate=next((x for x in cases if x[0]['category']==label),None)
            if candidate:examples.append(candidate)
    (out/'episode-analysis.json').write_text(json.dumps(allrows,indent=2))
    body='''<h1>From visual interventions to learned Minecraft navigation</h1><p>This gallery shows real recorded behavior and action-specific attribution. The four-arm experiment changes image preprocessing and loss weighting; <b>none of these movement GIFs uses V30/V31 activation patching</b>. Earlier frozen-model interventions and this trained policy are separate experiments.</p><h2>What we have established</h2><p>Teacher collection: 384/384 scripted demonstrations completed. Learned episode-weighted navigation stopped immediately for every present validation target. Timestep weighting improved raw-image arrivals to 13/56; gray-band inputs plus timestep weighting reached 5/56. Target choice remains unreliable. This is one seed, four families in one validation geometry group.</p><h2>Read the three views correctly</h2><ul><li><b>Movement:</b> what happened in Minecraft, including HUD and XYZ. GIF time follows simulator ticks, not inference wall-clock latency.</li><li><b>Policy attention:</b> the learned instruction query's weights over a 4×8 grid of pooled visual features. This is not a ViT head attention map.</li><li><b>Action Grad-CAM:</b> positive gradient-weighted feature contributions for the action selected at that frame. This is not a target detector or a patch mask. Memory from preceding steps is held fixed.</li></ul><p>Maps are independently rescaled to their own maximum in each frame. Equally bright frames need not have equally strong attribution. Bilinear smoothing makes 32 spatial values look more precise than they are. Blank Grad-CAM means no positive response under this calculation, not that vision was unused. A highlighted region can be correlated with an action without causing correct selection.</p><h2>Scoring correction</h2><p>The original evaluator counted any eventual stop on absent targets. Both timestep policies moved in all eight absent episodes. Under navigation_v2, absent success requires stopping before the first action. Old files remain unchanged; episode-analysis.json gives rescored results.</p>'''
    if suite.name=='vla-ablation-navigation-v1':
        body+='<h2>Visual review of the recorded GIFs</h2><p><b>raw-timestep / e00006:</b> the red-arch request succeeds. At the middle snapshot, policy attention is concentrated around the red arch; action Grad-CAM for FORWARD is much more diffuse, including ceiling and neighboring structures. At STOP, attention is low on the nearby structure/floor boundary. This does not identify an exclusive red-arch detector.</p><p><b>raw-timestep / e00007:</b> the blue-arch request follows the same discrete action sequence as the successful red-arch example and arrives at the red arch. Attention is again concentrated around that red arch at the middle snapshot. A visually plausible attention highlight therefore accompanies a wrong target.</p><p><b>raw-timestep / e00005:</b> the blue pillar is absent, but the policy moves toward the red pillar and eventually stops. Grad-CAM is broad, with responses on ceiling, floor and hand/HUD areas in some snapshots. Those features warrant input controls, but the map alone cannot tell us they caused the mistake.</p><p>Across all four arms, 1,220 of 5,694 recorded decision maps have zero positive Grad-CAM values. These frames should not be interpreted as a missing visual computation. Positive-only channel averaging can suppress relevant negative or cancelling contributions.</p>'
    body+='<table><tr><th>Arm</th><th>Correct present arrival</th><th>Immediate absent refusal</th></tr>'
    for arm in arms:
        a=[r for r in allrows if r['arm']==arm];present=[r for r in a if r['present']];absent=[r for r in a if not r['present']]
        body+=f'<tr><td>{arm}</td><td>{sum(r["metrics"]["success"] for r in present)}/{len(present)}</td><td>{sum(r["metrics"]["success"] for r in absent)}/{len(absent)}</td></tr>'
    body+='</table>'
    for rec,ep,ds in examples:
        arm=rec['arm'];eid=rec['episode'];base=suite/('rollout-'+arm);prefix=f'../rollout-{arm}';body+='<h2>'+html.escape(arm+' / '+eid+' — '+rec['category'])+'</h2><p>Instruction: Go to the '+html.escape(' '.join(rec['goal']))+'.</p>'
        body+='<div class="three">'+''.join('<figure><figcaption>'+label+'</figcaption><img loading="lazy" src="'+path+'"></figure>' for label,path in [('Raw movement',f'{prefix}/episodes/{eid}/movement.gif'),('Policy cross-attention',f'{prefix}/maps/{eid}/policy-attention.gif'),('Action Grad-CAM',f'{prefix}/maps/{eid}/action-gradcam.gif')])+'</div>'
        steps=sorted({0,len(ds)//2,len(ds)-1});canvas=Image.new('RGB',(448*3,305*len(steps)),'white');draw=ImageDraw.Draw(canvas)
        for row,step in enumerate(steps):
            d=ds[step];cam=np.asarray(d['action_gradcam']);r,c=divmod(int(cam.argmax()),8)
            paths=[base/'episodes'/eid/ep['trajectory'][step]['frame'],base/'maps'/eid/f'{step:04d}-attention.png',base/'maps'/eid/f'{step:04d}-cam.png']
            for col,(label,path) in enumerate(zip(('Raw','Attention','Grad-CAM'),paths)):
                draw.text((col*448+4,row*305+4),f'{label} step {step}: {d["action"]}',fill='black');canvas.paste(Image.open(path).convert('RGB'),(col*448,row*305+25))
        name=f'{arm}-{eid}-sequence.png';canvas.save(out/name);body+='<img class="sequence" src="'+name+'">'
        last=ds[-1];body+='<p>Recorded '+str(len(ds))+' decisions; final action '+last['action']+'. '
        if rec['present']:body+='Final requested-waypoint distance: '+str(round(rec['metrics']['distance'],2))+' blocks. '
        else:body+='Target absent; traveled '+str(round(rec['metrics']['travel_distance'],2))+' blocks. '
        body+='Zero-positive-CAM frames: '+str(rec['zero_cam_frames'])+'/'+str(len(ds))+'. The maps explain sensitivity of the selected action, not whether the intended object was selected correctly.</p>'
        body+=f'<p><a href="{prefix}/episodes/{eid}/report.html">Full movement and trajectory</a></p>'
    body+='<h2>Claim boundaries for presenting this work</h2><p>Supported: changing training weighting improved navigation in this pilot; some learned trajectories reach correct targets; startup stopping and wrong-target approaches are major failures. Unsupported: solved binding, a causal navigation head, general Minecraft competence, or reduced inference compute. To establish a mechanism, first verify unpatched object selection, then compare matched donor/self/random interventions and ablations on independently evaluated scenes.</p>'
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><style>body{max-width:1450px;margin:30px auto;padding:20px;font:17px/1.55 system-ui;color:#243447}.three{display:grid;grid-template-columns:repeat(3,1fr)}figure{margin:8px}img{max-width:100%}td,th{padding:10px;border-bottom:1px solid #ddd}.sequence{margin:15px 0}@media(max-width:800px){.three{grid-template-columns:1fr}}</style>'+body)
    print(out/'index.html')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('suite');main(p.parse_args().suite)
