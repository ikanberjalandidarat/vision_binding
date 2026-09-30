"""Audit destination choices and summarize actual recorded answers, not movement."""
import json
import html
from pathlib import Path
from mc_binding.io import digest, atomic_json
from mc_binding.destination import parse_side


def main():
    base=Path(__file__).resolve().parents[1]
    run=base/'runs/oscar/destination-replication-6868954'
    dataset=base/'runs/oscar/vision-captures-replication-24-6852730'
    manifest=json.loads((run/'manifest.json').read_text());data=json.loads((dataset/'manifest.json').read_text())
    assert manifest['dataset_hash']==digest(data)
    assert json.loads((run/'status.json').read_text())==dict(state='complete',rows=672)
    rows=[json.loads(s) for s in (run/'results.jsonl').read_text().splitlines()]
    saved=[r for p in (run/'families').glob('*.json') for r in json.loads(p.read_text())]
    assert len(rows)==len(saved)==len({r['trial_key'] for r in rows})==672
    assert {r['trial_key']:r for r in rows}=={r['trial_key']:r for r in saved}
    pairs={(r['scene_family_id'],r['context']):r for r in data['records'] if r['kind']=='pair'}
    clean={(r['family'],r['goal_color']):r for r in rows if r['condition']=='clean_recipient'}
    for r in rows:
        original=next(i for i,o in enumerate(pairs[r['family'],'recipient']['objects']) if o['color']==r['goal_color'])
        donor=next(i for i,o in enumerate(pairs[r['family'],'color_swap']['objects']) if o['color']==r['goal_color'])
        assert r['original_goal_side']==original and r['donor_goal_side']==donor
        assert r['choice_side']==parse_side(r['raw'])
        assert r['chose_original']==(r['choice_side']==original) and r['chose_donor']==(r['choice_side']==donor)
        if r['condition']=='self':assert r['raw']==clean[r['family'],r['goal_color']]['raw']
        if r['condition'].startswith('clean_'):assert r['chose_original'] if r['input_context']=='recipient' else r['chose_donor']
    counts=[]
    for st in ('layer30','layer31','group30-31'):
        for condition in ('both_objects','self','random','background'):
            a=[r for r in rows if r['patch_set']==st and r['condition']==condition];assert len(a)==48
            counts.append(dict(patch_set=st,condition=condition,n=48,donor_choices=sum(r['chose_donor'] for r in a),invalid=sum(r['choice_side'] is None for r in a)))
    failed=[r for r in rows if r['patch_set']=='layer30' and r['condition']=='both_objects' and not r['chose_donor']]
    out=run/'analysis';out.mkdir(exist_ok=True)
    atomic_json(out/'audit.json',dict(rows=672,clean_correct=96,exact_self_checks=144,counts=counts,v30_nontransfers=failed,dataset_hash=digest(data)))
    body=['<!doctype html><meta charset="utf-8"><title>Destination patch results</title><style>body{font:17px system-ui;max-width:1100px;margin:30px auto;line-height:1.6}img{width:48%}td,th{padding:10px;border-bottom:1px solid #ddd}table{border-collapse:collapse}</style>',
          '<h1>Do donor V activations change the chosen destination?</h1><p>All 672 rows verified. All 96 clean choices correct; 144 exact self-patches passed. These are recorded choices, not movement results.</p>',
          '<table><tr><th>ViT patch set</th><th>Condition</th><th>Donor-side choices</th></tr>']
    body += [f'<tr><td>{r["patch_set"]}</td><td>{r["condition"]}</td><td>{r["donor_choices"]}/48</td></tr>' for r in counts]
    body += ['</table><h2>Example: choose the yellow destination</h2><p>Original (left image): blue pillar LEFT, yellow stairs RIGHT. Donor (right image): yellow pillar LEFT, blue stairs RIGHT.</p>']
    for r in (pairs['f0000','recipient'],pairs['f0000','color_swap']):body.append(f'<img src="../../{dataset.name}/{r["image"]}">')
    body.append('<p>Question: “In this image, is the yellow structure on the left or the right? Answer left or right only.”</p><table><tr><th>Run</th><th>Image supplied</th><th>Recorded answer</th><th>Destination in original world</th></tr>')
    for condition,st in (('clean_recipient','baseline'),('self','group30-31'),('both_objects','group30-31')):
        r=next(r for r in rows if r['family']=='f0000' and r['goal_color']=='yellow' and r['condition']==condition and r['patch_set']==st)
        obj=pairs['f0000','recipient']['objects'][r['choice_side']]
        body.append(f'<tr><td>{condition}</td><td>Original</td><td>{html.escape(r["raw"])}</td><td>{obj["color"]} {obj["type"]}</td></tr>')
    body+=['</table><h2>Exact activation positions</h2><p>Both object regions are patched. Orange is the intervention mask, not attention or a destination detector. The original screenshot stays unchanged.</p><img src="../alignment/f0000/recipient-both.png"><img src="../alignment/f0000/color_swap-both.png">',
           '<p>V30+31 switched all 48 choices to the donor side. In the unchanged world this selects the wrong-color object; it is causal control of a decision, not improved task accuracy. Random and background controls did not switch choices.</p>',
           '<p>The direct prompt was selected on these same configurations. This is calibration evidence, not held-out generalization. Both objects were patched here, unlike the one-object color-report study; the fractions measure different outcomes.</p>',
           '<p>Next: replay clean, self and patched choices with the same known-coordinate controller. No model inference occurs during replay; actual arrival must be checked separately.</p>',
           '<p><a href="../report.html">All recorded trials</a> · <a href="audit.json">Audit and V30 exceptions</a> · <a href="../../../../docs/research-guide.html">Research guide</a></p>']
    (out/'explained.html').write_text('\n'.join(body))
    print(counts)
    print('V30 non-transfers:',[(r['family'],r['goal_color']) for r in failed])
if __name__=='__main__':main()
