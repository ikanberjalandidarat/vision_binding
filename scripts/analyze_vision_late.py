"""Audit scored late-layer runs; illustrate measured probabilities with real frames."""
import argparse
import base64
import json
import math
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from mc_binding.io import digest, atomic_json
from mc_binding.vision_data import load_swaps


def analyze(run,dataset):
    root,ds=Path(run),Path(dataset)
    data,groups=load_swaps(ds)
    manifest=json.loads((root/'manifest.json').read_text())
    assert manifest['dataset_hash']==digest(data)
    rows=[json.loads(s) for s in (root/'results.jsonl').read_text().splitlines()]
    saved=[r for p in (root/'families').glob('*.json') for r in json.loads(p.read_text())]
    assert len(rows)==len(saved)==len({r['trial_key'] for r in rows})
    assert {r['trial_key']:r for r in rows}=={r['trial_key']:r for r in saved}
    assert json.loads((root/'status.json').read_text())=={'state':'complete','rows':len(rows)}
    trials=[r for r in rows if 'family' in r]
    for r in trials:
        assert len(r['color_scores'])==2
        for s in r['color_scores']:
            for c,v in s['probability'].items():
                assert math.isclose(v,math.exp(s['log_probability'][c]),rel_tol=1e-6)
            assert math.isclose(s['donor_minus_original_log_odds'],s['log_probability'][s['donor_color']]-s['log_probability'][s['original_color']],abs_tol=1e-8)
            assert math.isclose(s['change_from_clean'],s['donor_minus_original_log_odds']-s['clean_log_odds'],abs_tol=1e-8)
    sets=[(l,) for l in manifest['config']['vision_layers']]+[tuple(g) for g in manifest['config']['vision_layer_groups']]
    summary=[]
    for group in sets:
        rr=[r for r in trials if tuple(r['layer_set'])==group and r['condition']=='target']
        summary.append({'layers':group,'n':len(rr),'transfers':sum(r['target_transferred'] for r in rr),
            'neighbors_preserved':sum(r['neighbor_preserved'] for r in rr),
            'all_target_score_shifts_positive':all(r['color_scores'][r['target_side']]['change_from_clean']>0 for r in rr)})
    out=root/'analysis';out.mkdir(exist_ok=True)
    atomic_json(out/'score-audit.json',{'records':len(rows),'layer_counts':summary,
        'interpretation':'Descriptive first-answer-token color mass. Spelling variants are summed; generated token choice can disagree near a tie. Not full-answer probabilities.'})
    pair={r['context']:r for r in groups['f0000'] if r['kind']=='pair'}
    base=next(r for r in rows if r['condition']=='clean' and r['record_id']==pair['recipient']['record_id'] and r['side']==1)
    chosen=[next(r for r in trials if r['family']=='f0000' and r['target_side']==1 and r['condition']=='target' and r['layer_set']==g) for g in ([29],[30],[31],[29,31],[30,31])]
    scores=[base['color_scores']]+[r['color_scores'][1] for r in chosen]
    labels=['No patch','Layer 29','Layer 30','Layer 31','29 + 31','30 + 31']
    answers=[base['raw']]+[r['answers'][1] for r in chosen]
    fig=plt.figure(figsize=(14,9));grid=fig.add_gridspec(2,2,height_ratios=[1,1.6])
    for col,ctx,title in [(0,'recipient','Original recipient: RIGHT tower is red'),(1,'color_swap','Donor: RIGHT tower is blue')]:
        ax=fig.add_subplot(grid[0,col]);ax.imshow(Image.open(ds/pair[ctx]['image']));ax.axis('off');ax.set_title(title,fontsize=14)
        from matplotlib.patches import Rectangle
        x0,y0,x1,y1=pair[ctx]['objects'][1]['bbox'];ax.add_patch(Rectangle((x0,y0),x1-x0,y1-y0,fill=False,edgecolor='#ffc03b',linewidth=2))
    ax=fig.add_subplot(grid[1,:])
    for i,s in enumerate(scores):
        ax.bar(i-.18,100*s['probability']['red'],width=.35,color='#bf5551',label='Original color: red' if i==0 else None)
        ax.bar(i+.18,100*s['probability']['blue'],width=.35,color='#447db2',label='Donor color: blue' if i==0 else None)
        for dx,c in [(-.18,'red'),(.18,'blue')]:ax.text(i+dx,100*s['probability'][c]+1.5,f"{100*s['probability'][c]:.2f}%",ha='center',fontsize=10)
    ax.set_xticks(range(len(labels)),[a+'\nAnswer: '+b for a,b in zip(labels,answers)])
    ax.set_ylim(0,119);ax.set_ylabel('Probability mass at the first answer token (%)');ax.legend(loc='upper right')
    ax.set_title('Same right tower, same question: “What color is the rightmost structure?”',fontsize=15)
    fig.suptitle('An unchanged answer can hide a substantial shift toward the donor color',fontsize=18)
    fig.text(.06,.018,'Actual saved images and scores. Red/Red/leading-space spellings are summed (likewise blue).\nBars need not sum to 100%: other next tokens remain possible. The Minecraft input is never recolored.',fontsize=11)
    fig.tight_layout(rect=[0,.07,1,.95]);fig.savefig(out/'late-color-scores.png',dpi=170);plt.close(fig)
    html=['<!doctype html><meta charset="utf-8"><title>Late-layer scores explained</title><style>body{font:18px system-ui;max-width:1300px;margin:30px auto;padding:20px}img{width:100%}td,th{padding:10px;border:1px solid #ccc}table{border-collapse:collapse}</style><h1>Late-layer scores: what changed?</h1>',
        '<img alt="Actual Minecraft scene and measured color probabilities" src="data:image/png;base64,'+base64.b64encode((out/'late-color-scores.png').read_bytes()).decode()+'">',
        '<p>In this example the recipient tower is red, and the donor tower is blue. Layer 31 increases the probability of blue without making it the generated answer. This is a partial effect, not zero effect.</p><table><tr><th>Layers</th><th>Target transfers</th><th>Neighbors preserved</th><th>Target preference moved toward donor in every case?</th></tr>']
    for s in summary:html.append(f"<tr><td>{' + '.join(map(str,s['layers']))}</td><td>{s['transfers']}/{s['n']}</td><td>{s['neighbors_preserved']}/{s['n']}</td><td>{s['all_target_score_shifts_positive']}</td></tr>")
    html.append('</table><p>These are repeated measurements on the original four layouts, not new-scene validation. A successful answer transfer is an experimental effect, not improved accuracy. Near a tie, the largest individual token probability (used by greedy generation) may differ from the largest sum across spelling variants.</p>')
    (out/'explained.html').write_text('\n'.join(html))
    print(json.dumps(summary,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--dataset',required=True);a=p.parse_args();analyze(a.run,a.dataset)
