"""Audit real job artifacts and plot world-balanced geometry probe comparisons."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
source=ROOT/'runs/oscar/spatial-geometry-probes-8963648'
out=ROOT/'output/spatial-geometry-probes/results-8963648';out.mkdir(parents=True,exist_ok=True)
r=json.loads((source/'report.json').read_text());rows=r['results']
assert len(rows)==4032 and r['state']=='complete'
for row in rows:
    with np.load(source/row['artifact'],allow_pickle=False) as a:
        assert not set(a['train_row_indices'])&set(a['test_row_indices'])
        assert np.allclose(np.abs(a['predicted_distance']-a['true_distance']).mean(),row['distance_mae_blocks'])
worlds=sorted({v['world'] for v in rows})
specs=[('mean_distance',0,'Constant'),('raw',0,'Raw'),('ridge',2,'Coordinate'),('random_orth',2,'Random 2'),('orth',2,'Orth. 2'),('linear_metric',2,'Linear 2'),('random_orth',8,'Random 8'),('orth',8,'Orth. 8'),('linear_metric',8,'Linear 8')]

def values(layer,condition,method,k,control='true_positions',metric='distance_mae_blocks'):
    return np.array([np.mean([v[metric] for v in rows if v['world']==w and v['representation']==f'{condition}_layer{layer}' and v['method']==method and v['k']==k and v['control']==control]) for w in worlds])

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
records=[]
for layer in (3,11,23):
 for method,k,label in specs:
    v=values(layer,'video',method,k); rep=values(layer,'repeat_last_frame',method,k);shift=values(layer,'video',method,k,'shifted_training_positions');base=values(layer,'video','mean_distance',0)
    records.append(dict(block=layer+1,method=method,k=k,video_mae=float(v.mean()),repeated_mae=float(rep.mean()),shifted_mae=float(shift.mean()),video_better_than_repeated_worlds=int((v<rep).sum()),video_better_than_constant_worlds=int((v<base).sum()),true_better_than_shifted_worlds=int((v<shift).sum()),worlds={w:dict(video_mae=float(a),repeated_mae=float(b),shifted_mae=float(c),constant_mae=float(d)) for w,a,b,c,d in zip(worlds,v,rep,shift,base)}))
for control in (False,True):
 fig,axs=plt.subplots(1,3,figsize=(17,6),sharey=True)
 for ax,layer in zip(axs,(3,11,23)):
    x=np.arange(len(specs))
    first=[values(layer,'video',m,k).mean() for m,k,_ in specs]
    second=[values(layer,'video' if control else 'repeat_last_frame',m,k,'shifted_training_positions' if control else 'true_positions').mean() for m,k,_ in specs]
    ax.bar(x-.18,first,.36,label='Video · true labels' if control else 'Video',color='#2267AA')
    ax.bar(x+.18,second,.36,label='Video · shifted labels' if control else 'Repeated endpoint frame',color='#CF8733')
    ax.axhline(first[0],ls='--',color='#263545',lw=1)
    ax.set_title(f'Block {layer+1}');ax.set_xticks(x);ax.set_xticklabels([s[2] for s in specs],rotation=55,ha='right')
    ax.set_ylim(0,10);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
 axs[0].set_ylabel('Held-out distance MAE (blocks) ↓')
 axs[0].legend(fontsize=9,loc='upper left')
 title='Learned spatial probes show little separation from shifted-label controls' if control else 'Video helps, but learned probes do not beat the constant-distance baseline on average'
 fig.suptitle(title,fontsize=16,y=.97)
 fig.text(.03,.02,'Eight worlds equally weighted; seeds averaged within route direction, then route directions within world.\nConstant = training-route mean pair distance; dashed line = constant baseline. Coordinate bar measures distance error, not coordinate error.',fontsize=10,color='#4A5968')
 fig.subplots_adjust(left=.06,right=.99,top=.86,bottom=.28,wspace=.12)
 name='label-controls' if control else 'probe-comparison'
 fig.savefig(out/f'{name}.png',dpi=200);fig.savefig(out/f'{name}.svg');plt.close(fig)
fig,axs=plt.subplots(1,3,figsize=(17,6),sharey=True)
for ax,layer in zip(axs,(3,11,23)):
    baseline=values(layer,'video','mean_distance',0)
    matrix=np.stack([values(layer,'video',m,k)-baseline for m,k,_ in specs[1:]],axis=1)
    im=ax.imshow(matrix,cmap='RdBu_r',vmin=-6,vmax=6,aspect='auto')
    ax.set_title(f'Block {layer+1}');ax.set_yticks(range(8));ax.set_yticklabels(worlds)
    ax.set_xticks(range(8));ax.set_xticklabels([s[2] for s in specs[1:]],rotation=55,ha='right')
    for i in range(8):
     for j in range(8): ax.text(j,i,f'{matrix[i,j]:.1f}',ha='center',va='center',fontsize=8,color='white' if abs(matrix[i,j])>3 else 'black')
fig.suptitle('Per-world video error relative to the constant-distance baseline',fontsize=16)
fig.subplots_adjust(left=.07,right=.91,top=.87,bottom=.23,wspace=.12)
cax=fig.add_axes([.935,.24,.012,.60]);fig.colorbar(im,cax=cax,label='MAE difference (blocks); negative is better')
fig.text(.07,.035,'Descriptive paired differences; no independent-pair confidence intervals. Worlds each contain only 7–20 clips per route.',fontsize=11)
fig.savefig(out/'per-world.png',dpi=200);fig.savefig(out/'per-world.svg');plt.close(fig)
report={'job_id':8963648,'state':'reviewed','verified_artifacts':len(rows),'checks':['All artifact distance MAEs reproduce report values','All stored train/test row index sets are disjoint'],'aggregation':'Equal-world mean after within-world averaging over route directions and seeds','comparisons':records,'coverage':json.loads((source/'coverage.json').read_text()),'interpretation':['All tested video probe mean distance errors exceed the constant-distance baseline','Temporal improvement versus repeated frame does not establish successful spatial geometry recovery','Sparse opposite-heading revisits constrain place-identity conclusions','Training losses and convergence traces were not saved; cannot isolate optimization failure from data/generalization limitations']}
(out/'figure-data.json').write_text(json.dumps(report,indent=2)+'\n')
for name in ('probe-comparison','label-controls','per-world'):
 (out/f'{name}.json').write_text(json.dumps({'image':name+'.png','data':'figure-data.json','source_report':str(source/'report.json'),'job_id':8963648,'aggregation':report['aggregation'],'limitations':r['limitations']},indent=2)+'\n')
print('Verified',len(rows),'artifacts')
for layer in (3,11,23):
 print('block',layer+1,'ridge coordinate error',values(layer,'video','ridge',2,metric='coordinate_error_blocks').mean())
for v in records:
 if v['block']==12:print({k:val for k,val in v.items() if k!='worlds'})
