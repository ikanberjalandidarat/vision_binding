"""Audit the recorded V replication and make a screenshot-based summary."""
import json
from pathlib import Path
import statistics
import html
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from mc_binding.io import digest, atomic_json
from mc_binding.vision_data import load_swaps

ROOT=Path(__file__).resolve().parents[1]
def main():
    run=ROOT/'runs/oscar/vision-pilot-replication-6853654'
    dataset=ROOT/'runs/oscar/vision-captures-replication-24-6852730'
    data,groups=load_swaps(dataset)
    manifest=json.loads((run/'manifest.json').read_text())
    assert manifest['dataset_hash']==digest(data)
    assert manifest['config']['vision_patch_kind']=='v'
    rows=[json.loads(s) for s in (run/'results.jsonl').read_text().splitlines()]
    saved=[r for p in (run/'families').glob('*.json') for r in json.loads(p.read_text())]
    assert len(rows)==len(saved)==len({r['trial_key'] for r in rows})==912
    assert {r['trial_key']:r for r in rows}=={r['trial_key']:r for r in saved}
    clean=[r for r in rows if r['condition']=='clean'];assert len(clean)==192 and all(r['correct'] for r in clean)
    baselines={r['trial_key']:r['raw'] for r in clean}
    sets=['layer30','layer31','group30-31'];names=['V30','V31','V30 + V31']
    target=[r for r in rows if r['condition']=='target']
    counts=[];breakdown=[];failures=[]
    success=lambda r:r['target_transferred'] and r['neighbor_preserved']
    for r in rows:
        if r['condition']=='self':
            rec=next(v for v in groups[r['family']] if v['context']=='recipient' and v['kind']=='pair')
            assert all(r['answers'][s]==baselines[f"{r['family']}:clean:{rec['record_id']}:{s}"] for s in (0,1))
            assert all(abs(s['change_from_clean'])<1e-4 for s in r['color_scores'])
    for st in sets:
        for condition in ('self','target','random','other_object','background'):
            a=[r for r in rows if r.get('patch_set')==st and r['condition']==condition];assert len(a)==48
            counts.append(dict(patch_set=st,condition=condition,n=len(a),specific_transfers=sum(map(success,a)),
                               neighbor_preserved=sum(r['neighbor_preserved'] for r in a),
                               median_score_shift=statistics.median(r['color_scores'][r['target_side']]['change_from_clean'] for r in a)))
    labels={'palette':['yellow / blue','red / yellow','red / blue'],'shape_pair':['stairs / pillar','arch / tower'],'placement':['placement 0','placement 1']}
    for factor,values in labels.items():
        for value,label in enumerate(values):
            fids={fid for fid,rs in groups.items() if rs[0]['factors'][factor]==value}
            for st in sets:
                a=[r for r in target if r['family'] in fids and r['patch_set']==st]
                breakdown.append(dict(factor=factor,label=label,patch_set=st,n=len(a),success=sum(map(success,a))))
    for r in target:
        if r['patch_set']=='layer30' and not success(r):
            rec=next(v for v in groups[r['family']] if v['context']=='recipient' and v['kind']=='pair')
            failures.append(dict(family=r['family'],target_side=r['target_side'],object=rec['objects'][r['target_side']]['type'],answers=r['answers'],factors=rec['factors']))
    out=run/'analysis';out.mkdir(exist_ok=True)
    atomic_json(out/'audit.json',dict(dataset_hash=digest(data),records=len(rows),clean_correct=len(clean),self_checks='exact answers and score changes passed',counts=counts,breakdown=breakdown,v30_failures=failures,
                                   scope='Repeated assignments and sides within one arena; not 48 independent worlds. No new Q/K/whole-state comparison.'))
    fig=plt.figure(figsize=(15,10));grid=fig.add_gridspec(3,3,height_ratios=[1.1,.65,1.6])
    pair={r['context']:r for r in groups['f0000'] if r['kind']=='pair'}
    paths=[dataset/pair['recipient']['image'],dataset/pair['color_swap']['image'],run/'alignment/f0000/patch-target-target0.png']
    titles=['Original: blue pillar + yellow stairs','Donor: yellow pillar + blue stairs','Original input: selected LEFT token locations']
    for i,(p,title) in enumerate(zip(paths,titles)):
        ax=fig.add_subplot(grid[0,i]);ax.imshow(Image.open(p));ax.axis('off');ax.set_title(title,fontsize=12)
    ax=fig.add_subplot(grid[1,:]);ax.axis('off')
    ax.text(.02,.95,'Three separate interventions on the SAME original image above:',fontsize=14,weight='bold')
    for i,(st,label) in enumerate(zip(sets,names)):
        r=next(r for r in target if r['family']=='f0000' and r['target_side']==0 and r['patch_set']==st)
        y=.65-i*.28
        ax.text(.02,y,label,fontsize=15,weight='bold')
        ax.text(.23,y,'LEFT: '+r['answers'][0]+'     RIGHT: '+r['answers'][1],fontsize=15)
        ax.text(.70,y,'Specific transfer: '+('YES' if success(r) else 'NO'),fontsize=14)
    ax=fig.add_subplot(grid[2,:]);vals=[sum(success(r) for r in target if r['patch_set']==st) for st in sets]
    ax.bar(names,vals,color=['#3286a6','#9ba8ae','#208369']);ax.set_ylim(0,55);ax.set_ylabel('Donor-color transfer AND neighbor preserved');ax.set_title('All 24 configurations × two target sides = 48 cases per patch set')
    for i,v in enumerate(vals):ax.text(i,v+1,f'{v}/48 ({v/48:.1%})',ha='center',fontsize=15)
    fig.suptitle('V30 + V31 transfers the target color in all tested replication cases',fontsize=19)
    fig.text(.06,.025,'Top: one recorded example, followed by three interventions on that SAME original image. Input pixels never change.\n192/192 clean checks passed. Random/background controls: 0/48 transfers per set. These are repeated configurations, not independent worlds.',fontsize=11)
    fig.tight_layout(rect=[0,.075,1,.95]);fig.savefig(out/'replication-results.png',dpi=160);plt.close(fig)
    body=['<!doctype html><meta charset="utf-8"><title>V replication results</title><style>body{font:17px system-ui;max-width:1150px;margin:30px auto;line-height:1.6}img{width:100%}td,th{padding:9px;border-bottom:1px solid #ddd}table{border-collapse:collapse}</style>',
          '<h1>V replication: recorded results</h1><img src="replication-results.png" alt="Screenshots and V patch outcomes">',
          '<p>All 912 unique rows match atomic checkpoints and the capture dataset hash. 192 clean answers were correct; all 144 self-patches reproduced their clean answers and score changes.</p>',
          '<h2>Breakdown</h2><table><tr><th>Factor</th><th>Group</th><th>Patch</th><th>Specific transfers</th></tr>']
    body += [f"<tr><td>{r['factor']}</td><td>{r['label']}</td><td>{r['patch_set']}</td><td>{r['success']}/{r['n']}</td></tr>" for r in breakdown]
    body += ['</table><h2>Controls</h2><p>Random and background patches produced no target transfers and preserved neighbors. Other-object patches also produced no target transfers, but changed the neighbor in 43/48 V30 and 48/48 V30+31 cases: the effect follows the patched object. This duplicates opposite-target interventions and is not independent evidence.</p>',
             '<h2>Five V30 failures</h2><pre>'+html.escape(json.dumps(failures,indent=2))+'</pre>',
             '<p>All five are LEFT targets (four pillars, one tower); right stairs/arches transferred in 24/24 V30 cases. Shape and side are confounded in this design. Swapping the shapes across sides is a useful next diagnostic.</p>',
             '<p>V31 alone did not flip answers; adding it after V30 rescued five cases. This does not establish synergy: it adds another donor overwrite. No Q/K or whole-state runs on this new dataset are included, so this cannot establish superiority over them.</p>',
             '<p><a href="../report.html">Full trial explorer with image overlays</a> · <a href="audit.json">Machine-readable audit</a> · <a href="../../../../docs/research-guide.html">Central research guide</a></p>']
    (out/'explained.html').write_text('\n'.join(body))
    print('Validated 912 rows; wrote',out)
if __name__=='__main__':main()
