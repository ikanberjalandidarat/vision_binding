"""Frozen-state cognitive-map diagnostics. Coordinates train evaluator readouts only."""
import json
import html
from pathlib import Path
import numpy as np
import torch
from .hippocampal import SpatialMemory
from .io import atomic_json


def ridge_readout(train_x,train_y,test_x):
    mean=train_x.mean(0);scale=train_x.std(0).clip(.01)
    a=np.column_stack(((train_x-mean)/scale,np.ones(len(train_x))))
    b=np.column_stack(((test_x-mean)/scale,np.ones(len(test_x))))
    penalty=np.eye(a.shape[1]);penalty[-1,-1]=0
    return b@np.linalg.solve(a.T@a+penalty,a.T@train_y)


def audit(root):
    root=Path(root);torch.set_num_threads(4)
    entries=json.loads((root/'sequences.json').read_text())
    cache=torch.load(root/'features.pt',map_location='cpu',weights_only=True)
    ck=torch.load(root/'memory.pt',map_location='cpu',weights_only=True)
    data={v['episode']:v for v in cache['sequences']}; states={};targets={};loop_rows=[]
    train=[e['episode'] for e in entries if e['split']=='train' and e['route'] in (0,1)]
    for e in entries:
        poses=[f['pose'] for f in e['frames']];targets[e['episode']]=np.array([[p['x'],p['z']] for p in poses])
        yaw=np.deg2rad([p['yaw'] for p in poses]);t=np.arange(len(poses))/max(1,len(poses)-1)
        states[e['episode']]={'time_heading':np.column_stack([t,t*t,np.sin(yaw),np.cos(yaw)])}
    with torch.no_grad():
        for label,key in [('learned','state'),('untrained','initial')]:
            net=SpatialMemory();net.load_state_dict(ck[key]);net.eval()
            for e in entries:
                c=data[e['episode']];x=(c['x']-ck['mean'])/ck['scale']
                _,g,p,_=net(x,c['motion'])
                states[e['episode']].update({label+'_g':g.numpy(),label+'_p':p.numpy()})
    readouts=[]
    for kind in states[entries[0]['episode']]:
        tx=np.concatenate([states[n][kind] for n in train]);ty=np.concatenate([targets[n] for n in train])
        for e in entries:
            name=e['episode'];pred=ridge_readout(tx,ty,states[name][kind])
            error=np.linalg.norm(pred-targets[name],axis=1)
            split=e['split'] if e['split']!='train' or e['route'] in (0,1) else 'heldout_route'
            readouts.append(dict(episode=name,split=split,representation=kind,mean_error_blocks=float(error.mean()),median_error_blocks=float(np.median(error))))
    # Closure stability at matched cardinal headings; normalized by training variation.
    for kind in ('learned_g','learned_p','untrained_g','untrained_p'):
        scale=np.concatenate([states[n][kind] for n in train]).std(0).clip(.01)
        for e in entries:
            refs={};a=states[e['episode']][kind]
            for t,f in enumerate(e['frames']):
                probe=f.get('probe')
                if not probe:continue
                yaw=probe['yaw']
                if probe['probe']=='initial':refs[yaw]=t
                elif yaw in refs:
                    ref=refs[yaw]
                    loop_rows.append(dict(episode=e['episode'],representation=kind,frame=t,reference_frame=ref,yaw=yaw,loop=probe['probe'],normalized_rms=float(np.sqrt(np.mean(((a[t]-a[ref])/scale)**2))),actual_position_gap=float(np.linalg.norm(targets[e['episode']][t]-targets[e['episode']][ref]))))
    provenance='Minecraft' if all(e.get('is_minecraft',False) for e in entries) else 'NON-MINECRAFT synthetic fixture'
    result=dict(data_kind=provenance,scope='Scripted exploration with privileged world-axis odometry; no learned navigation and no proven place/grid cells.',train_routes=[0,1],heldout_routes=[2,3],readout='Fixed ridge alpha=1, train geometries/routes 0/1 only. Coordinate labels used by offline evaluator, never memory objective. Time/heading baseline includes normalized episode time; descriptive control.',readouts=readouts,loop_returns=loop_rows,route_completion=[dict(episode=e['episode'],complete=e['route_complete']) for e in entries],limitations=['Single seed unless replicated. Shared floor/layout structure. Heading probes at one anchor, not uniform whole-map heading coverage.','Cross-route coordinate decoding tests accessible spatial information, not biological grid periodicity. No gridness claim.','Loop p changes may reflect different accumulated visual memory; they need not imply localization failure.'])
    atomic_json(root/'cognitive-review.json',result)
    rows=''
    for split in ('train','heldout_route','validation','test'):
        for kind in states[entries[0]['episode']]:
            vals=[r['mean_error_blocks'] for r in readouts if r['split']==split and r['representation']==kind]
            if vals:rows+=f'<tr><td>{split}</td><td>{kind}</td><td>{np.mean(vals):.3f}</td></tr>'
    gifs=''.join(f'<figure><img loading="lazy" width="100%" src="{e["episode"]}.gif"><figcaption>{html.escape(str(root/"analysis"/(e["episode"]+".gif")))}</figcaption></figure>' for e in entries if e['split']=='validation' and e['route'] in (0,2))
    (root/'analysis'/'cognitive-map.html').write_text('''<!doctype html><meta charset="utf-8"><title>Cognitive mapping in Minecraft</title><style>body{font:17px/1.6 system-ui;max-width:1100px;margin:40px auto;padding:20px}td,th{padding:8px}figcaption{overflow-wrap:anywhere;font-size:13px}</style><h1>Does movement-conditioned memory learn a cognitive map?</h1><p>'''+html.escape(provenance)+'''</p><p>Qwen visual features → motion-conditioned g → retrieval of past visual features → p → sensory prediction. The spatial memory learns from prediction, without location labels. Exploration is scripted; measured odometry is privileged.</p><p>Two route orders train the memory; two reversed/reordered routes test generalization. Every loop returns to a common anchor with four cardinal-heading probes. GIF axes use evaluator coordinates, not a map output by the model.</p><h2>Frozen location readout</h2><p>Lower error is better. Readouts use training-only coordinate supervision to test information in frozen states. Compare learned states with untrained states and time/heading. Report per-episode errors; correlated frames are not independent trials.</p><table><tr><th>Split</th><th>State</th><th>Mean error (blocks)</th></tr>'''+rows+'''</table><h2>Learned activity during exploration</h2><p>Unit selection uses training variance only. g/p are candidate model populations, not established biological cells. Loop-return distances and all episode results are in the compact JSON.</p>'''+gifs+'''<p><a href="../cognitive-review.json">Compact review JSON</a> · <a href="index.html">All activity GIFs</a></p><p>Evidence required next: reproducible location information across routes/headings and model seeds. This experiment does not establish grid periodicity, visual localization or learned navigation.</p>''')
