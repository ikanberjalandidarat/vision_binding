"""Audit saved vision sweeps and plot descriptive outcomes; no inference or training."""
import argparse
import json
from collections import Counter
from pathlib import Path
import numpy as np
from mc_binding.io import digest, atomic_json
from mc_binding.vision_data import load_swaps
from mc_binding.scoring import parse


def analyze(run, dataset):
    run=Path(run)
    data, families=load_swaps(dataset)
    manifest=json.loads((run/'manifest.json').read_text())
    assert digest(data)==manifest['dataset_hash'], 'Dataset differs from recorded run'
    rows=[json.loads(line) for line in (run/'results.jsonl').read_text().splitlines()]
    keys=[r['trial_key'] for r in rows]
    assert len(keys)==len(set(keys)), 'Duplicate trial keys'
    saved=[r for f in (run/'families').glob('*.json') for r in json.loads(f.read_text())]
    assert len(saved)==len(rows) and {r['trial_key']:r for r in saved}=={r['trial_key']:r for r in rows}
    assert json.loads((run/'status.json').read_text())=={'state':'complete','rows':len(rows)}
    pairs={fid:{r['context']:r for r in rr if r['kind']=='pair'} for fid,rr in families.items()}
    base=[r for r in rows if r['condition']=='clean']
    patches=[r for r in rows if r['condition']!='clean']
    sets=[(l,) for l in manifest['config']['vision_layers']]+[tuple(g) for g in manifest['config'].get('vision_layer_groups',[])]
    cases=[(fid,side) for fid in sorted(families) for side in (0,1)]
    conditions=['self','target','random','other_object','background']
    index={(r['family'],r['target_side'],tuple(r['layer_set']),r['condition']):r for r in patches}
    assert len(index)==len(patches)==len(cases)*len(sets)*len(conditions)
    for fid,side in cases:
        for group in sets:
            for cond in conditions:
                r=index[fid,side,group,cond]
                rec=pairs[fid]['recipient']['objects']; donor=pairs[fid]['color_swap']['objects']
                parsed=[parse(a,task='color')['parsed'] for a in r['answers']]
                colors=[p['color'] if p else None for p in parsed]
                assert colors==r['parsed_colors']
                assert r['target_transferred']==(colors[side]==donor[side]['color'])
                assert r['neighbor_preserved']==(colors[1-side]==rec[1-side]['color'])
                if cond=='self':
                    for s in (0,1):
                        baseline=next(b for b in base if b['record_id']==pairs[fid]['recipient']['record_id'] and b['side']==s)
                        assert r['answers'][s]==baseline['raw']
    controls={c:{'n':len(rr),'target_transferred':sum(r['target_transferred'] for r in rr),
        'neighbor_preserved':sum(r['neighbor_preserved'] for r in rr),
        'specific_transfer':sum(r['target_transferred'] and r['neighbor_preserved'] for r in rr)}
        for c in conditions for rr in [[r for r in patches if r['condition']==c]]}
    matrix=np.array([[int(index[f,s,g,'target']['target_transferred'] and index[f,s,g,'target']['neighbor_preserved']) for f,s in cases] for g in sets])
    comparisons=[]
    for g in sets:
        if len(g)==1: continue
        outcomes=[]
        for f,s in cases:
            win=index[f,s,g,'target']['target_transferred']
            members=[index[f,s,(l,),'target']['target_transferred'] for l in g]
            outcomes.append((win, members))
        comparisons.append({'layers':g,'successes':sum(w for w,m in outcomes),
            'gains_over_last_layer':sum(w and not m[-1] for w,m in outcomes),
            'losses_vs_last_layer':sum(not w and m[-1] for w,m in outcomes),
            'successes_where_all_members_failed':sum(w and not any(m) for w,m in outcomes)})
    out=run/'analysis';out.mkdir(exist_ok=True)
    atomic_json(out/'audit.json',{'rows':len(rows),'clean_correct':sum(b['correct'] for b in base),
        'clean_n':len(base),'controls':controls,'groups':comparisons,
        'layer_counts':[{'layers':g,'successes':int(v.sum()),'n':len(cases)} for g,v in zip(sets,matrix)],
        'scope':'Descriptive calibration only. Four layouts include reverse donor/recipient pairs; no independent-scene confidence intervals.'})
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    labels=[]
    for f,s in cases:
        obj=pairs[f]['recipient']['objects'][s]
        labels.append(f'{f} {"left" if s==0 else "right"}\n{obj["color"]} {obj["type"]}')
    fig,ax=plt.subplots(figsize=(13,14))
    ax.imshow(matrix,cmap=ListedColormap(['#f1d8cf','#33846e']),vmin=0,vmax=1,aspect='auto')
    ax.set_xticks(range(len(cases)),labels,fontsize=9)
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(sets)),['Layer '+str(g[0]) if len(g)==1 else 'Group '+str(g[0])+'–'+str(g[-1]) for g in sets],fontsize=9)
    for i,row in enumerate(matrix):
        for j,v in enumerate(row):ax.text(j,i,'YES' if v else '—',ha='center',va='center',color='white' if v else '#66443a',fontsize=8)
        ax.text(len(cases)-.25,i,f'{row.sum()}/{len(cases)}',ha='left',va='center',fontsize=9)
    ax.axhline(len(manifest['config']['vision_layers'])-.5,color='white',linewidth=4)
    ax.set_title('Donor color transferred at target AND original neighbor color preserved\nFull vision sweep · single layers and simultaneous groups',pad=48,fontsize=15)
    fig.text(.08,.015,'YES = both behavioral checks pass. Eight cases reuse four counterbalanced layouts; these are not independent scenes.\nWhole block-output states were patched. No shape-binding, necessity, or unique color-layer claim.',fontsize=10)
    fig.subplots_adjust(left=.16,right=.92,top=.87,bottom=.06)
    fig.savefig(out/'layer-transfer.png',dpi=180)
    # PNG only: avoid generating unwanted PDF artifacts.
    plt.close(fig)
    print(json.dumps({'rows':len(rows),'controls':controls,'output':str(out)},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',required=True);parser.add_argument('--dataset',required=True)
    args=parser.parse_args();analyze(args.run,args.dataset)
