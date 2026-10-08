"""Audit original and explicit retry artifacts without merging or overwriting attempts."""
import argparse
from collections import Counter
import html
import json
import os
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from mc_binding.navigation_metrics import score_episode
from mc_binding.rl_pilot import approach_gate
from PIL import Image,ImageDraw


def read(p):return json.loads(p.read_text())


def main(suite,retry):
 root=Path(suite).resolve();retry=Path(retry).resolve();out=root/'analysis';out.mkdir(exist_ok=True)
 runs=[];sources={};rows_by_run={};audited=0
 for seed in (731,732):
  for chunk in (2,4):
   name=f'seed-{seed}-chunk-{chunk}';original=root/name;p=original
   if read(p/'status.json')['state']!='complete':
    p=retry/name
   assert read(p/'status.json')['state']=='complete'
   sources[name]=str(p);tr=read(p/'training.json');pr=read(p/'probes.json');va=read(p/'validation.json')
   assert len(tr)==128 and len(pr)==160 and len(va)==32
   assert approach_gate(pr)==read(p/'learning-gate.json')
   for row in tr+pr+va:
    ep=read(p/'episodes'/row['episode']/'episode.json')
    metrics=score_episode(ep['trajectory'],ep['waypoint'],ep['stopped'])
    assert metrics['success']==ep['success']==row['success']
    assert metrics['timeout']==ep['timeout']
    assert ep['goal_present']==row['goal_present']
    assert len(row['actions'])==row['steps']==len(row['decisions'])
    if row['episode'].startswith('validation'):
     assert ep['diagnostic_start'] is None and ep['training_start'] is None
     pose=ep['trajectory'][0]['pose'];assert abs(pose['x']-.5)<.1 and abs(pose['z']-.5)<.1
    row['_metrics']=metrics;audited+=1
   results=[]
   for checkpoint in (0,32,64,96,128):
    for mode in ('greedy','sampled'):
     rr=[r for r in pr if r['checkpoint']==checkpoint and r['evaluation_mode']==mode]
     present=[r for r in rr if r['goal_present']];absent=[r for r in rr if not r['goal_present']]
     assert len(present)==12 and len(absent)==4
     results.append(dict(checkpoint=checkpoint,mode=mode,present_success=sum(r['success'] for r in present),absent_success=sum(r['success'] for r in absent),entered_radius=sum(r['_metrics']['minimum_sampled_goal_distance']<=.8 for r in present),timeouts=sum(r['_metrics']['timeout'] for r in present),actions=dict(Counter(a for r in present for a in r['actions']))))
   original_results=[]
   for mode in ('greedy','sampled'):
    rr=[r for r in va if r['evaluation_mode']==mode]
    original_results.append(dict(mode=mode,present_success=sum(r['success'] and r['goal_present'] for r in rr),absent_success=sum(r['success'] and not r['goal_present'] for r in rr)))
   rows_by_run[name]=(tr,pr,va)
   runs.append(dict(name=name,seed=seed,chunk=chunk,source=str(p),passed=read(p/'learning-gate.json')['passed'],families=sorted({r['family'] for r in pr}),checkpoints=results,original_start=original_results))
 # Identical development panels, not independent scenes across modes/checkpoints/arms.
 signature=lambda rows:[(r['family'],r['instruction'],r['goal']) for r in rows if r['checkpoint']==0 and r['evaluation_mode']=='greedy']
 assert all(signature(rows_by_run[r['name']][1])==signature(next(iter(rows_by_run.values()))[1]) for r in runs)
 for seed in (731,732):
  a,b=[rows_by_run[f'seed-{seed}-chunk-{c}'][0] for c in (2,4)]
  assert [(r['family'],r['instruction']) for r in a]==[(r['family'],r['instruction']) for r in b]
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,ax=plt.subplots(figsize=(9,4.5))
 for r in runs:
  selected=[c for c in r['checkpoints'] if c['mode']=='sampled']
  ax.plot([c['checkpoint'] for c in selected],[c['present_success'] for c in selected],marker='o',label=r['name'])
 ax.set(xlabel='Training episodes (0 = untrained)',ylabel='Successful near-target approaches / 12',ylim=(-.2,12.5),xticks=[0,32,64,96,128],title='Sampled actions improve modestly with 4-tick movements')
 ax.legend(fontsize=8);ax.grid(alpha=.2);fig.tight_layout();fig.savefig(out/'checkpoint-success.png',dpi=160);plt.close(fig)
 body='''<h1>Approach benchmark: some sampled improvement, no reliable stopping</h1><p>The 4-tick policies improve near-target sampled success in both seeds: 2→5/12 and 2→4/12 at the final checkpoint. Both 2-tick runs end at their initialization counts. No arm passes the learning gate. All greedy present-target checkpoint success rates are zero.</p><p>Crucial distinction: seed 732, chunk 4 enters the success radius in all 12 final greedy present probes, but keeps turning and times out. Entering the radius is not scored as success without stopping. This condition demonstrates a stopping failure after approach; other arms also fail to approach.</p><img class="wide" src="checkpoint-success.png"><p>Same 16 development tasks from four held-out families, reused across checkpoints, modes and arms. Twelve present and four absent targets. Repeated evaluations are not independent scenes; sampled comparisons reuse random seeds. Near starts privilege target location, so these results do not establish binding.</p><table><tr><th>Run</th><th>Sampled present, initial→final</th><th>Final greedy: entered radius</th><th>Final greedy: successful stop</th><th>Original-start present, greedy / sampled</th></tr>'''
 for r in runs:
  initial=next(c for c in r['checkpoints'] if c['checkpoint']==0 and c['mode']=='sampled')
  sampled=next(c for c in r['checkpoints'] if c['checkpoint']==128 and c['mode']=='sampled')
  greedy=next(c for c in r['checkpoints'] if c['checkpoint']==128 and c['mode']=='greedy')
  body+=f'<tr><td>{r["name"]}</td><td>{initial["present_success"]}→{sampled["present_success"]}/12</td><td>{greedy["entered_radius"]}/12</td><td>{greedy["present_success"]}/12</td><td>'+ ' / '.join(str(x['present_success'])+'/12' for x in r['original_start'])+'</td></tr>'
 body+='''</table><p>Original-start navigation: zero present arrivals in every arm and both modes. The retry has 3/4 absent refusals in greedy original-start evaluation; other original-start absence results are zero. This does not compensate for its present-target failures.</p><h2>Recorded movement evidence</h2><p>Selected examples, not random samples. These are real Minecraft frames; GIFs retain the HUD and XYZ annotations. Playback follows simulation ticks, not inference latency. No activation patches, attention maps or Grad-CAM were generated in this runner.</p>'''
 examples=[('seed-732-chunk-4','probe-128-greedy-00000','Approaches the waypoint, then keeps turning'),('seed-731-chunk-4','probe-128-sampled-00000','Sampled near-target success'),('seed-731-chunk-2','probe-128-sampled-00000','Enters the radius, leaves it, then stops too far away'),('seed-732-chunk-2','probe-128-greedy-00000','Retry: stops immediately before approaching'),('seed-732-chunk-4','validation-00000','Same trained policy from the original start')]
 for i,(name,eid,title) in enumerate(examples):
  p=Path(sources[name]);ep=p/'episodes'/eid;data=read(ep/'episode.json');trace=data['trajectory'];row=next(r for phase in rows_by_run[name] for r in phase if r['episode']==eid)
  canvas=Image.new('RGB',(1344,310),'white');draw=ImageDraw.Draw(canvas)
  for j,k in enumerate((0,len(trace)//2,len(trace)-1)):
   with Image.open(ep/trace[k]['frame']) as im:canvas.paste(im.convert('RGB'),(448*j,30))
   draw.text((448*j+5,5),f'Recorded frame {k} | tick {trace[k]["tick"]}',fill='black')
  image=f'example-{i}.png';canvas.save(out/image)
  gif=ep/'movement.gif';rel=os.path.relpath(gif,out)
  body+=f'<h3>{html.escape(title)}</h3><p>{html.escape(row["instruction"])}. {name}/{eid}. Success: {data["success"]}; timeout: {data["timeout"]}. Closest sampled distance {data["minimum_sampled_goal_distance"]:.2f}; final {data["distance"]:.2f} blocks.</p><img class="wide" src="{image}"><figure><img src="{rel}"><figcaption>{html.escape(os.path.relpath(gif,root.parents[2]))}</figcaption></figure><a href="{os.path.relpath(ep/"report.html",out)}">Trajectory and episode report</a>'
 body+='''<h2>Next experiment: isolate stopping before increasing scope</h2><p>Retain these negative results. Test stop-versus-continue from controlled positions inside and outside the arrival radius, including absent targets, with an untrained baseline. Use simulator-generated rewards and fresh observations, no manual labels. Distinguish failure to learn on training scenes from failure to transfer to held-out geometry by evaluating both without optimizer updates.</p><p>A diagnostic automatic-arrival termination variant could isolate approach ability, but must be labeled privileged and must not replace explicit-stop success in the main benchmark. Once explicit stopping works, reconnect sustained movement and return to original-start binding tasks. Do not infer a Q/K/V mechanism from these outcomes.</p><p>Longer chunks change both movement and turning duration, as well as the maximum simulation-time budget at the fixed 96-decision limit. The observed gains cannot be attributed uniquely to movement persistence. No parameter search or new training was performed for this report.</p>'''
 body+=f'<h2>Audit and provenance</h2><p>{audited} episode outcomes recomputed from trajectories and reconciled with recorded success. Four learning gates recomputed; final validation starts verified; task panels matched. Retry seed 732/chunk 2 is read from a separate directory. The original failed attempt remains untouched and is excluded from performance counts.</p><ul>'
 for r in runs:body+='<li>'+html.escape(r['source'])+'</li>'
 body+='</ul>'
 (out/'findings.json').write_text(json.dumps(dict(runs=runs,audited_episodes=audited),indent=2))
 (out/'explained.html').write_text('<!doctype html><meta charset="utf-8"><title>Approach benchmark analysis</title><style>body{font:17px system-ui;line-height:1.55;max-width:1300px;margin:40px auto;padding:20px}td,th{padding:10px;border-bottom:1px solid #ccc;text-align:left}img{max-width:100%}.wide{width:100%}figcaption{font:12px monospace;overflow-wrap:anywhere}h2{margin-top:40px}figure{margin:15px 0}</style>'+body)
 print(out/'explained.html')

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('suite');p.add_argument('--retry',required=True);a=p.parse_args();main(a.suite,a.retry)
