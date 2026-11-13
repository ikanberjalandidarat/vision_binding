"""Render the implemented geometry-probe experiment; schematic, not results."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT=Path(__file__).resolve().parents[1]/'output/spatial-geometry-probes'
OUT.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','svg.fonttype':'none'})
fig=plt.figure(figsize=(16,9),facecolor='white')
ax=fig.add_axes([0,0,1,1]);ax.set_xlim(0,1);ax.set_ylim(0,1);ax.axis('off')
ink='#152A40';muted='#53677A';blue='#2267AA';orange='#AE6308';green='#16766C'

def txt(x,y,s,size=12,color=ink,weight='normal',ha='left',va='top'):
    ax.text(x,y,s,fontsize=size,color=color,weight=weight,ha=ha,va=va,linespacing=1.5)

def box(x,y,w,h,title,body,fill='#F3F7FC',edge='#B6C9DC',size=12):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.009,rounding_size=0.009',linewidth=1.2,edgecolor=edge,facecolor=fill))
    txt(x+.012,y+h-.016,title,14,weight='bold')
    txt(x+.012,y+h-.055,body,size)

def arrow(a,b,color=blue,dashed=False):
    ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=14,linewidth=1.7,color=color,linestyle='--' if dashed else '-'))

def line(xs,ys,color=blue):ax.plot(xs,ys,color=color,lw=1.5)

txt(.035,.965,'Does frozen V-JEPA contain an accessible spatial geometry?',25,weight='bold')
txt(.035,.914,'Within-world probing  •  Same cached features and route splits for every method',14,color=muted)

box(.035,.735,.19,.13,'Minecraft videos','8 worlds · 2 routes each\n188 endpoint-labeled clips',size=12)
box(.255,.735,.205,.13,'Matched visual inputs','16-frame video\n16 × repeated endpoint frame',size=11.5)
box(.49,.735,.21,.13,'Frozen V-JEPA 2 ViT-L','Official preprocessing + HUD mask\nEvaluation mode · no weight updates',fill='#E9F2FC',edge=blue,size=10.7)
box(.73,.735,.235,.13,'Native feature readout','Blocks 4 / 12 / 24 · 1,024 channels\nSpatial mean: final temporal tubelet',size=11.3)
for a,b in [((.225,.80),(.25,.80)),((.46,.80),(.485,.80)),((.70,.80),(.725,.80))]:arrow(a,b)
txt(.035,.702,'Encoder input: RGB only. Every clip ends at its labeled position; no persistent state between clips.',11.5,color=muted)

# Feature bus connects to the three heads; preprocessing is fit on train route only.
line([.847,.847,.373],[.735,.655,.655]);
for x in (.373,.615,.847):arrow((x,.655),(x,.603))
txt(.245,.679,'Train-route centering + one scalar RMS scale; no per-channel whitening',11,color=muted)

ax.add_patch(FancyBboxPatch((.245,.325),.72,.265,boxstyle='round,pad=0.009,rounding_size=0.009',facecolor='#FCFAF5',edgecolor='#D8BC90',linewidth=1))
txt(.26,.583,'TRAINABLE READOUTS  ·  positions supervise these probes only',11,color=orange,weight='bold')
box(.035,.365,.175,.175,'Training-route pose','Ground-plane X,Z\nCoordinates and pair distances\nNever an encoder input',fill='#FFF6E6',edge='#D8BC90',size=10.5)
arrow((.21,.45),(.245,.45),orange,True)
box(.26,.35,.225,.195,'1  Coordinate probe',r'$\hat{p}=hW+b$'+'\nRidge regression · 2 outputs\nPredict X,Z; derive distances\nFixed regularization',fill='white',size=11.5)
box(.502,.35,.225,.195,'2  Orthonormal subspace',r'$\hat{d}_{ij}=s\,\|(h_i-h_j)Q\|_2$'+'\n'+r'$Q^\top Q=I_k$'+' · learned scale s\nRanks k = 2, 8\nDistance supervision',fill='white',size=11.5)
box(.744,.35,.205,.195,'3  Linear metric',r'$\hat{d}_{ij}=\|(h_i-h_j)W\|_2$'+'\nGeneral linear projection\nRanks k = 2, 8\nDistance loss + regularization',fill='white',size=10.8)
for x in (.373,.615,.847):line([x,x],[.34,.285])
line([.373,.847],[.285,.285]);arrow((.615,.285),(.615,.25),green)
txt(.035,.32,'h = normalized frozen feature\np = ground-plane position\nOnly readouts are fitted',10.5,color=muted)
box(.035,.135,.93,.105,'Within each world: fit route 0 → test route 1; then reverse',
    'Primary: distance MAE between two held-out endpoints   |   Secondary: cross-route retrieval + coordinate error\nSame settings for all worlds and layers · Report heading/coverage confounds · Test poses used only for scoring',
    fill='#EDF8F4',edge='#85BBB0',size=11.7)

txt(.035,.105,'CONTROLS',10.5,color=muted,weight='bold')
txt(.13,.105,'Repeated frames  •  Raw distances  •  Random orthonormal projections  •  Mean distance  •  Shifted training positions',10.5,color=muted)
txt(.035,.055,'EXPLORATORY PILOT',10,color=orange,weight='bold')
txt(.19,.055,'Two sparse routes/world; no validation-route tuning. Seed repeats are not independent data. No place/grid-cell claim.',10,color=muted)
for ext in ('png','svg'):
    fig.savefig(OUT/f'architecture.{ext}',dpi=220,facecolor='white')
print(OUT/'architecture.png')
