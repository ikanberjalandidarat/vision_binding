"""Audit recorded exploration pilots and build an evidence page; no model inference."""
import argparse
import html
import json
import math
import os
from collections import Counter
from pathlib import Path
from PIL import Image, ImageDraw


def read(path):
    return json.loads(path.read_text())


def analyze(root):
    root=Path(root).resolve();out=root/'analysis';out.mkdir(exist_ok=True)
    runs=[];all_rows={};audited=0
    for run in sorted(root.glob('seed-*')):
        assert read(run/'status.json')['state']=='complete', run
        training=read(run/'training.json');validation=read(run/'validation.json')
        manifest=read(run/'manifest.json')
        for row in training+validation:
            ep=read(run/'episodes'/row['episode']/'episode.json')
            assert ep['success']==row['success'] and ep['goal_present']==row['goal_present']
            assert row['steps']==len(row['actions'])==len(row['decisions'])
            assert all(math.isfinite(v) for d in row['decisions'] for v in d['probabilities'])
            assert all(abs(sum(d['probabilities'])-1)<1e-5 for d in row['decisions'])
            if row in validation:
                assert row['training_start'] is None and ep['training_start'] is None
                pose=ep['trajectory'][0]['pose']
                assert abs(pose['x']-.5)<.1 and abs(pose['z']-.5)<.1
            row['_metrics']=ep;audited+=1
        groups=[]
        for mode in ('greedy','sampled'):
            rows=[r for r in validation if r['evaluation_mode']==mode]
            groups.append(dict(mode=mode,n=len(rows),present=sum(r['goal_present'] for r in rows),
                present_success=sum(r['success'] and r['goal_present'] for r in rows),
                absent=sum(not r['goal_present'] for r in rows),absent_success=sum(r['success'] and not r['goal_present'] for r in rows),
                timeouts=sum(r['_metrics']['timeout'] for r in rows),mean_steps=sum(r['steps'] for r in rows)/len(rows),
                action_counts=dict(Counter(a for r in rows for a in r['actions'])),
                mean_initial_stop_probability=sum(r['decisions'][0]['probabilities'][3] for r in rows)/len(rows),
                mean_initial_entropy=sum(r['decisions'][0]['entropy'] for r in rows)/len(rows)))
        curriculum=[]
        for distance in (1.5,3.,6.,None):
            rows=[r for r in training if r['goal_present'] and (r['training_start'] or {}).get('distance')==distance]
            curriculum.append(dict(distance=distance,n=len(rows),successes=sum(r['success'] for r in rows)))
        runs.append(dict(run=run.name,seed=manifest['seed'],training_present_successes=sum(r['success'] and r['goal_present'] for r in training),curriculum=curriculum,validation=groups,
            validation_families=sorted({r['family'] for r in validation}),gradient_norm_range=[min(r['gradient_norm_before_clip'] for r in training),max(r['gradient_norm_before_clip'] for r in training)]))
        all_rows[run.name]=(training,validation)
    assert len(runs)==4
    comparisons=[]
    for arm in ('original','near-to-far'):
        left=all_rows[f'seed-731-{arm}'][0];right=all_rows[f'seed-732-{arm}'][0]
        assert [r['actions'] for r in left]!=[r['actions'] for r in right], 'Seed histories unexpectedly identical'
    for seed in (731,732):
        a,b=[all_rows[f'seed-{seed}-{arm}'] for arm in ('original','near-to-far')]
        key=lambda r:(r['family'],r['instruction'])
        assert list(map(key,a[0]))==list(map(key,b[0])), 'Training task schedules differ'
        assert list(map(key,a[1]))==list(map(key,b[1])), 'Evaluation task schedules differ'
        comparisons.append(dict(seed=seed,training_tasks_matched=True,evaluation_tasks_matched=True))
    body='''<h1>Exploration pilot: nearby successes, no transfer yet</h1>
<p>Four completed runs: two starting-position conditions × two seeds. Five present-target successes occurred during near-target training. Original-start validation had zero present-target arrivals in every condition and action-selection mode.</p>
<p><b>Scope:</b> the same 16 tasks from held-out family f0003 are repeated across runs and modes. These are 128 evaluations, not 128 independent scenes. Two aggregate successes were absent-target refusals from a policy that stopped immediately on every task.</p>
<h2>Validation from the original starting position</h2><table><tr><th>Run</th><th>Actions</th><th>Present success</th><th>Absent success</th><th>Timeouts</th><th>Mean decisions</th><th>Initial stop probability</th></tr>'''
    for run in runs:
        for g in run['validation']:
            body+=f'<tr><td>{run["run"]}</td><td>{g["mode"]}</td><td>{g["present_success"]}/{g["present"]}</td><td>{g["absent_success"]}/{g["absent"]}</td><td>{g["timeouts"]}/{g["n"]}</td><td>{g["mean_steps"]:.2f}</td><td>{g["mean_initial_stop_probability"]:.1%}</td></tr>'
    body+='</table><h2>What the curriculum actually achieved</h2><table><tr><th>Seed</th><th>1.5-block starts</th><th>3-block starts</th><th>6-block starts</th><th>Original starts during curriculum</th></tr>'
    for r in runs:
        if 'near-to-far' in r['run']:
            body+='<tr><td>'+str(r['seed'])+'</td>'+''.join(f'<td>{g["successes"]}/{g["n"]}</td>' for g in r['curriculum'])+'</tr>'
    body+='''</table><p>Present-target training episodes only. The near starts point toward the correct approach waypoint using simulator metadata. Their success demonstrates that approach plus stopping can occur, but does not establish learned selection among competing objects. Training episodes are sampled with the evolving policy, not held-out tests. No untrained-policy comparison was collected.</p>
<h2>Why more identical runs would not settle this</h2><p>Sampled actions usually stop before a long approach can happen. Greedy actions can repeat a slightly preferred turn or forward action even when probabilities are nearly uniform. High action entropy does not imply useful exploration. Gradients were finite and nonzero in every training episode; this checks gradient flow, not effective learning.</p>
<p>The curriculum advances on a fixed episode schedule, irrespective of success: nine near-start episodes per distance, then original starts. The five nearby successes did not establish mastery before progression. The original starts are approximately 18–22 blocks from the goal waypoint. Neither sampled nor greedy evaluation rescued present-target navigation.</p>
<p>Rewards are not success scores. Potential shaping adds distance-dependent terms, so positive reward can occur on a failed episode. Success here requires an explicit stop within 0.8 blocks of the waypoint; absent targets require stopping before movement.</p>
<h2>Recorded examples</h2><p>Selected examples illustrate failure modes, not a random sample. Sequences below are actual recorded start/middle/end frames. GIF playback uses simulation time, not model inference time. No V30/V31 patches, Grad-CAM or policy-attention maps were recorded by this RL runner. The near-target example visibly retains a tutorial popup; the displayed validation examples do not. This appearance difference is observed, but its effect is not isolated by this pilot.</p>'''
    examples=[('seed-731-near-to-far','train-00004','Near-target training success'),('seed-731-original','validation-00000','Greedy original policy: immediate stop'),('seed-731-near-to-far','validation-00001','Greedy curriculum policy: repeated left turns'),('seed-731-near-to-far','validation-00000','Greedy curriculum policy: forward without successful arrival'),('seed-731-near-to-far','validation-sampled-00000','Sampled curriculum policy: short motion then stop')]
    for index,(name,eid,title) in enumerate(examples):
        ep=root/name/'episodes'/eid;data=read(ep/'episode.json');trace=data['trajectory']
        canvas=Image.new('RGB',(1344,310),'white');draw=ImageDraw.Draw(canvas)
        for j,i in enumerate((0,len(trace)//2,len(trace)-1)):
            with Image.open(ep/trace[i]['frame']) as im:canvas.paste(im.convert('RGB'),(448*j,30))
            draw.text((448*j+5,5),f'Recorded frame {i}',fill='black')
        picture=f'example-{index}.png';canvas.save(out/picture)
        gif=ep/'movement.gif';path=os.path.relpath(gif,root.parents[2])
        row=next(r for phase in all_rows[name] for r in phase if r['episode']==eid)
        body+=f'<h3>{title}</h3><p>{html.escape(row["instruction"])} | {name}/{eid}. Success: {data["success"]}. Final distance: {data["distance"]:.2f} blocks; closest recorded distance: {data["minimum_sampled_goal_distance"]:.2f}. Actions: {html.escape(str(dict(Counter(row["actions"]))))} (0 forward, 1 left, 2 right, 3 stop).</p><img class="sequence" src="{picture}"><figure><img src="{os.path.relpath(gif,out)}"><figcaption>{html.escape(path)}</figcaption></figure><a href="{os.path.relpath(ep/"report.html",out)}">Episode trajectory and report</a>'
    body+='''<h2>Recommended next gate</h2><ol><li>Compare an untrained policy with trained checkpoints on the same near-target tasks and fixed evaluation sampling seeds. Establish improvement above chance before claiming learning.</li><li>Hold a simple near-target stage until it reliably learns approach and stop; evaluate periodically. Progress the distance only after a separate development check passes, with a strict budget cap.</li><li>Test a movement-persistence or action-duration variant while preserving the ability to stop immediately when absent. This directly tests premature termination, without supplying teacher actions.</li><li>Broaden evaluation across all held-out families after the small task works. Keep final test geometry untouched.</li></ol><p>No manual image labels are required. These are exploration and control diagnostics. Once a reliable policy selects targets, V30/V31 and head interventions can test which visual representations causally change the chosen destination. The current failures do not identify a binding circuit or show that Qwen lacks color/shape information.</p>'''
    body+=f'<p>Audit: {audited} episode outcomes reconciled with simulator files; original evaluation starts verified; task schedules matched across training-start conditions. Independent seeds produce different training action histories. Smoke is excluded from these results.</p>'
    findings=dict(runs=runs,audited_episodes=audited,matched_tasks=comparisons)
    (out/'findings.json').write_text(json.dumps(findings,indent=2))
    (out/'explained.html').write_text('<!doctype html><meta charset="utf-8"><title>RL exploration evidence</title><style>body{font:17px system-ui;line-height:1.55;max-width:1300px;margin:35px auto;padding:20px}td,th{padding:10px;border-bottom:1px solid #ccc;text-align:left}img{max-width:100%}.sequence{width:100%}figcaption{font:12px monospace;overflow-wrap:anywhere}h2{margin-top:45px}figure{margin:15px 0}</style>'+body)
    print(out/'explained.html')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('suite');analyze(p.parse_args().suite)
