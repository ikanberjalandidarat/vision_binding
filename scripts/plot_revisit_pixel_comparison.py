"""Compare recorded native-feature and pixel controls; no fitting or selection."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path(__file__).resolve().parents[1]
v=json.loads((root/'output/revisit-retrieval-8977130/report.json').read_text())
p=json.loads((root/'runs/oscar/revisit-pixels-8977923/report.json').read_text())
assert p['state']=='complete'
for key in ('dataset_sha256','protocol_sha256'):assert p[key]==v[key]
out=root/'output/revisit-comparison-8977923';out.mkdir(parents=True,exist_ok=True)
rows=[]
for delta in (180,90,0):
    for r in [r for r in v['results']+p['results'] if r['heading_difference']==delta]:
        rows.append({k:r[k] for k in ('feature','heading_difference','queries','accuracy','mean_error_blocks','random_error_blocks','directions')})
fig,axes=plt.subplots(1,3,figsize=(13,4.8),sharey=True)
for ax,delta in zip(axes,(180,90,0)):
    for condition,label,color,offset in [('video','Video','#2376b7',-.18),('repeat_last_frame','Repeated endpoint','#d78327',.18)]:
        vals=[next(r['accuracy']*100 for r in rows if r['feature']==f'{condition}_layer{l}' and r['heading_difference']==delta) for l in (3,11,23)]
        ax.bar(np.arange(3)+offset,vals,width=.34,label=label,color=color)
        for x,y in zip(np.arange(3)+offset,vals):ax.text(x,y+1.7,f'{y:.1f}',ha='center',fontsize=9)
    pix=next(r['accuracy']*100 for r in p['results'] if r['heading_difference']==delta)
    ax.axhline(pix,color='#7249a3',linewidth=2,label=f'Pixels ({pix:.1f}%)')
    ax.axhline(6.25,color='gray',ls='--',label='Chance (6.25%)')
    ax.set_xticks(range(3));ax.set_xticklabels(['Block 4','Block 12','Block 24']);ax.set_ylim(-2,105)
    ax.set_title(f'{delta}° heading difference'+ (' · primary' if delta==180 else ''))
    ax.spines[['top','right']].set_visible(False);ax.legend(fontsize=8,loc='upper left',bbox_to_anchor=(0,.83 if delta==0 else 1))
axes[0].set_ylabel('Correct site retrieval (%)')
fig.suptitle('Pixels outperform V-JEPA at same-view retrieval; opposite-view recognition fails')
fig.text(.05,.015,'Same held-out routes and fixed UI masks • 16 site candidates • Both retrieval directions • No learned probe\nPixel baseline: endpoint RGB, 32×20, MSE; resolution specified after feature inspection. One world; correlated queries.',fontsize=9)
fig.tight_layout(rect=[0,.10,1,.95]);fig.savefig(out/'comparison.png',dpi=200);fig.savefig(out/'comparison.svg');plt.close(fig)
(out/'comparison.json').write_text(json.dumps(dict(dataset_sha256=p['dataset_sha256'],protocol_sha256=p['protocol_sha256'],results=rows,
    sources=['runs/oscar/revisit-pixels-8977923/report.json','output/revisit-retrieval-8977130/report.json'],
    limits=['The pixel control is 32x20 RGB MSE, not an exhaustive appearance baseline.', 'Pixel resolution was specified after feature inspection; no sweep was performed.', 'No causal explanation or absence of all spatial information is inferred from retrieval failures.']),indent=2)+'\n')
print(out)
