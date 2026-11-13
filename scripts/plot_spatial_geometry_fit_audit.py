"""Plot completed real training-fit audit; no fitting or checkpoint selection."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
source=ROOT/'runs/oscar/spatial-geometry-fit-audit-8967118'
out=ROOT/'output/spatial-geometry-probes/fit-audit-8967118';out.mkdir(parents=True,exist_ok=True)
r=json.loads((source/'report.json').read_text());rows=r['results']
assert r['state']=='complete' and len(rows)==1152
assert r['prior_reproduction_all_close'] and r['prior_reproduction_max_mae_difference']==0
steps=r['steps'];worlds=sorted({v['world'] for v in rows})

def get(layer,method,k,control='true_positions',condition='video'):
    return [v for v in rows if v['representation']==f'{condition}_layer{layer}' and v['control']==control and v['method']==method and v['k']==k]

def mean_at(subset,step,partition):
    return float(np.mean([np.mean([next(t for t in v['curve'] if t['step']==step)[partition]['distance_mae'] for v in subset if v['world']==w]) for w in worlds]))

summary=[]
for layer in (3,11,23):
 for method,k in [('mean_distance',0),('ridge',2),('orth',2),('orth',8),('linear_metric',2),('linear_metric',8)]:
  for condition in ('video','repeat_last_frame'):
   for control in ('true_positions','shifted_training_positions'):
    a=get(layer,method,k,control,condition)
    curve=[dict(step=s,train_mae=mean_at(a,s,'train'),heldout_mae=mean_at(a,s,'heldout')) for s in (steps if method in ('orth','linear_metric') else [0])]
    summary.append(dict(block=layer+1,method=method,k=k,condition=condition,control=control,curve=curve))
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
fig,axs=plt.subplots(2,3,figsize=(16,9),sharex=True,sharey=True)
for j,layer in enumerate((3,11,23)):
 for i,k in enumerate((2,8)):
    ax=axs[i,j]
    for method,color,label in [('orth','#2267AA','Orthonormal'),('linear_metric','#B5661E','Linear metric')]:
        a=get(layer,method,k)
        ax.plot(steps,[mean_at(a,s,'train') for s in steps],'--o',ms=3,color=color,label=label+' · train')
        ax.plot(steps,[mean_at(a,s,'heldout') for s in steps],'-o',ms=3,color=color,label=label+' · held-out')
    baseline=mean_at(get(layer,'mean_distance',0),0,'heldout')
    ax.axhline(baseline,color='#7A828A',ls=':',label='Constant · held-out')
    ax.axvline(200,color='#CCC',lw=1)
    ax.set_title(f'Block {layer+1} · {k} dimensions')
    ax.set_xscale('symlog',linthresh=10);ax.set_xticks([0,10,50,200,1000]);ax.set_xticklabels(['0','10','50','200','1000'])
    ax.grid(alpha=.15);ax.set_yscale('symlog',linthresh=.1);ax.set_ylim(0,80)
    ax.set_yticks([0,.1,1,5,10,50]);ax.set_yticklabels(['0','0.1','1','5','10','50'])
    if i==1:ax.set_xlabel('Optimization steps (compressed scale)')
    if j==0:ax.set_ylabel('Distance MAE (blocks; compressed scale) ↓')
axs[0,0].legend(fontsize=8,loc='upper right')
fig.suptitle('The probes fit training routes, but do not transfer to the held-out route',fontsize=18,y=.98)
fig.text(.06,.02,'Frozen V-JEPA video features · seed 0 · eight worlds equally weighted after averaging route directions.\nStep 200 reproduces the prior run exactly. Curves diagnose fit; no checkpoint is selected using held-out error.',fontsize=11)
fig.subplots_adjust(left=.07,right=.98,bottom=.13,top=.90,wspace=.15,hspace=.32)
fig.savefig(out/'training-vs-heldout.png',dpi=200);fig.savefig(out/'training-vs-heldout.svg');plt.close(fig)

fig,axs=plt.subplots(1,3,figsize=(16,6),sharey=True)
spec=[('ridge',2,'Coordinate'),('orth',2,'Orth. 2'),('orth',8,'Orth. 8'),('linear_metric',2,'Linear 2'),('linear_metric',8,'Linear 8')]
for ax,layer in zip(axs,(3,11,23)):
 x=np.arange(5)
 for offset,control,part,color,label in [(-.3,'true_positions','train','#2267AA','True · train'),(-.1,'true_positions','heldout','#7FAAD5','True · held-out'),(.1,'shifted_training_positions','train','#B5661E','Shifted · train'),(.3,'shifted_training_positions','heldout','#E5BC8E','Shifted · held-out')]:
    ax.bar(x+offset,[mean_at(get(layer,m,k,control),0 if m=='ridge' else 200,part) for m,k,_ in spec],.19,color=color,label=label)
 ax.axhline(5.035641533698811,color='#555',ls=':',lw=1)
 ax.set_title(f'Block {layer+1}');ax.set_xticks(x);ax.set_xticklabels([s[2] for s in spec],rotation=35,ha='right');ax.set_ylim(0,10);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
axs[0].set_ylabel('Pair-distance MAE (blocks) ↓');axs[0].legend(fontsize=8)
fig.suptitle('Low training error also occurs with shifted training labels',fontsize=18)
fig.text(.06,.025,'Video features · seed 0 · 200 steps (ridge is closed-form). Shifted training error uses shifted targets; held-out targets stay correct.\nDotted line: constant-distance held-out baseline. Pair counts and reverse folds are not independent replicates.',fontsize=10)
fig.subplots_adjust(left=.06,right=.99,top=.86,bottom=.23,wspace=.12)
fig.savefig(out/'training-label-controls.png',dpi=200);fig.savefig(out/'training-label-controls.svg');plt.close(fig)
controls=json.loads((source/'positive-controls.json').read_text())
report=dict(job=8967118,state='reviewed',aggregation='Seeds fixed to 0; mean over route directions within world, then equal mean of eight worlds',prior_reproduction_max_mae_difference=0.,positive_controls=controls,summary=summary,limits=r['limitations'])
(out/'figure-data.json').write_text(json.dumps(report,indent=2)+'\n')
for name in ('training-vs-heldout','training-label-controls'):
 (out/f'{name}.json').write_text(json.dumps(dict(image=name+'.png',data='figure-data.json',source_report=str(source/'report.json'),aggregation=report['aggregation'],limits=r['limitations']),indent=2)+'\n')
for m,k,_ in spec:
 a=get(11,m,k,'shifted_training_positions');s=0 if m=='ridge' else 200
 print('shifted block12',m,k,mean_at(a,s,'train'),mean_at(a,s,'heldout'))
print(out)
