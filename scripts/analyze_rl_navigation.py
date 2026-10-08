"""Summarize recorded RL learning and action collapse; never infer success from reward."""
import argparse,json,html,os
from collections import Counter
from pathlib import Path
from PIL import Image,ImageDraw

def main(root):
 root=Path(root).resolve();out=root/'analysis';out.mkdir(exist_ok=True);rows=[]
 read=lambda p:json.loads(p.read_text())
 for p in sorted(root.glob('seed-*-arm-*')):
  tr=read(p/'training.json');va=read(p/'validation.json');timeouts=0
  for v in va:
   e=read(p/'episodes'/v['episode']/'episode.json');timeouts+=bool(e['timeout'])
   if e['success']!=v['success']:raise ValueError('Outcome mismatch')
  row=dict(run=p.name,training_episodes=len(tr),training_present_successes=sum(x['success'] and x['goal_present'] for x in tr),training_absent_successes=sum(x['success'] and not x['goal_present'] for x in tr),mean_training_steps=sum(x['steps'] for x in tr)/len(tr),max_training_steps=max(x['steps'] for x in tr),validation_episodes=len(va),validation_successes=sum(x['success'] for x in va),timeouts=timeouts,validation_action_counts=dict(Counter(a for x in va for a in x['actions'])),unique_validation_sequences=len({tuple(x['actions']) for x in va}))
  rows.append(row)
 duplicates=[]
 for arm in range(3):
  def signatures(seed,phase):return [(x['family'],x['instruction'],x['actions']) for x in read(root/f'seed-{seed}-arm-{arm}'/(phase+'.json'))]
  duplicates.append(dict(arm=arm,training_actions_identical=signatures(731,'training')==signatures(732,'training')==signatures(733,'training'),validation_actions_identical=signatures(731,'validation')==signatures(732,'validation')==signatures(733,'validation')))
 body='<h1>RL v2: execution passed, navigation failed</h1><p>All nine runs completed, but all 576 validation episodes timed out. This total repeats the same 64 tasks; it is not 576 independent scenes. Encoder initialization reset the global seed, so these are not three independent replications per arm. A corrected RNG implementation is required.</p><table><tr><th>Run</th><th>Train success present/absent</th><th>Mean train decisions</th><th>Validation success</th><th>Timeouts</th></tr>'
 for r in rows:body+=f'<tr><td>{r["run"]}</td><td>{r["training_present_successes"]} / {r["training_absent_successes"]}</td><td>{r["mean_training_steps"]:.2f}</td><td>{r["validation_successes"]}/{r["validation_episodes"]}</td><td>{r["timeouts"]}</td></tr>'
 body+='</table><h2>Why this is not a binding result</h2><p>Training never reached a present target successfully. Stochastic training episodes ended after about 4–5 decisions on average; greedy validation usually repeated forward for 96 steps and never stopped. This training/evaluation behavior difference and inadequate successful exploration must be addressed before attributing failures to visual binding. Seed isolation is a separate implementation defect, not a complete explanation for failed learning.</p><p>The auxiliary predicts a global mean next-frame feature. It is not evidence of object-centric representation or binding. Scores cannot establish whether Qwen lacks relevant information.</p><h2>Representative recorded behavior</h2>'
 for arm in range(3):
  p=root/f'seed-731-arm-{arm}';eid='validation-00000';ep=p/'episodes'/eid;data=read(ep/'episode.json');t=data['trajectory'];canvas=Image.new('RGB',(1344,310),'white');draw=ImageDraw.Draw(canvas)
  for j,i in enumerate([0,len(t)//2,len(t)-1]):
   with Image.open(ep/t[i]['frame']) as im:canvas.paste(im.convert('RGB'),(448*j,30))
   draw.text((448*j+5,5),f'Arm {arm} | recorded step {i}',fill='black')
  name=f'arm-{arm}-sequence.png';canvas.save(out/name)
  gif=os.path.relpath(ep/'movement.gif',out)
  body+=f'<h3>Arm {arm}: {html.escape(str(data["goal"]))}</h3><img width="100%" src="{name}"><img width="448" src="{gif}"><p><code>{html.escape(str((ep/"movement.gif").relative_to(root.parent.parent.parent)))}</code></p><p>Final distance {data["distance"]:.2f}; closest sampled distance {data["minimum_sampled_goal_distance"]:.2f} blocks. Timed out: {data["timeout"]}.</p>'
 body+='<h2>Next experiment design</h2><p>First verify independently seeded policies and log action entropy, stop probability, and exploration depth. Then test an automatic near-target curriculum and matched stochastic/greedy evaluation in small pilots. Keep original-task evaluation fixed and distinguish curriculum rewards from original-task success. No more nine-run sweep until these checks pass; no manual object labels required.</p>'
 atomic=dict(runs=rows,seed_action_comparisons=duplicates)
 (out/'findings.json').write_text(json.dumps(atomic,indent=2));(out/'explained.html').write_text('<!doctype html><meta charset="utf-8"><style>body{font:17px system-ui;max-width:1300px;margin:40px auto}td,th{padding:10px;border-bottom:1px solid #ccc}code{overflow-wrap:anywhere}h2{margin-top:40px}</style>'+body)
 print(json.dumps(duplicates));print(out/'explained.html')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('suite');a=p.parse_args();main(a.suite)
