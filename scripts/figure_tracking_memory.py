"""Publication-style explanatory figure from recorded frames; no invented activations."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch, FancyArrowPatch
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'runs/oscar/target-tracking-7176804'
OUT=RUN/'analysis'
frames=json.loads((RUN/'audit/frames.json').read_text())
results=json.loads((RUN/'evaluation/results.json').read_text())
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'svg.fonttype':'path'})
ink='#182b40';teal='#007e87';gold='#e4aa24';muted='#536579';red='#b74445'
fig=plt.figure(figsize=(17,12),facecolor='white')
fig.text(.04,.962,'Remembering a target as the camera moves',fontsize=25,weight='bold',color=ink)
fig.text(.04,.931,'Implemented baseline: frozen Qwen visual features + a fixed appearance template',fontsize=14,color=muted)
fig.text(.04,.890,'A   SELECT THE RED PILLAR, THEN RECOGNIZE IT IN LATER VIEWS',weight='bold',color=ink)
for j,step in enumerate([0,24,40]):
 row=next(r for r in frames if r['episode']=='e00048' and r['step']==step)
 pred=next(r for r in results if r['episode']=='e00048' and r['step']==step and r['method']=='template_memory')
 path=ROOT/'runs/oscar/vla-demos-7045999/episodes/e00048'/Path(row['frame']).name
 ax=fig.add_axes([.04+j*.315,.660,.292,.203]);ax.imshow(Image.open(path));ax.set_axis_off()
 for o in row['regions']:
  x0,y0,x1,y1=o['bbox_raw'];selected=o['index']==pred['selected_index']
  ax.add_patch(Rectangle((x0,y0),x1-x0,y1-y0,fill=False,edgecolor=gold if selected else '#c7d2dd',linewidth=2.5 if selected else 1))
 ax.set_title(['Acquire target • frame 0','Turn and reobserve • frame 24','Approach • frame 40'][j],fontsize=12,loc='left',color=ink,pad=7)
fig.text(.04,.643,'Gold = model-selected region. Gray = other accepted regions. Boxes come from simulator geometry, not a learned detector.',fontsize=11,color=muted)
fig.text(.04,.614,'Recorded episode e00048: the target shifts across the image and grows. The stored feature vector stays fixed.',fontsize=12,color=ink)
fig.text(.04,.569,'B   WHAT IS STORED AND HOW IT IS USED',weight='bold',color=ink)
boxed_text=[]
def box(x,y,w,h,title,body,color=teal):
 fig.patches.append(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.008',transform=fig.transFigure,facecolor='#f3f7fa',edgecolor=color,linewidth=1.3))
 title_text=fig.text(x+.012,y+h-.014,title,fontsize=12,weight='bold',color=color,va='top')
 body_text=fig.text(x+.012,y+h-.043,body,fontsize=10.5,color=ink,linespacing=1.5,va='top')
 boxed_text.extend([(title_text,(x,y,w,h)),(body_text,(x,y,w,h))])
def arrow(a,b):
 fig.patches.append(FancyArrowPatch(a,b,transform=fig.transFigure,arrowstyle='-|>',mutation_scale=16,color=muted,linewidth=1.4))
box(.04,.425,.275,.12,'1  Encode candidates','Qwen vision block 31 → 8×16 grid\nPool features within each candidate box.\nFeature vector: '+r'$z_{t,i}\in\mathbb{R}^{d}$')
box(.365,.425,.275,.12,'2  Store a template','Copy the first selected feature: '+r'$m=z_{t_0,i^*}$'+'\nA visual vector, not a crop or coordinate.\nKeep it fixed for the episode.')
box(.69,.425,.263,.12,'3  Match candidates','Current red-pillar score: '+r'$s_{t,i}$'+'\nEligible only if '+r'$s_{t,i}\geq0.05$'+'\nRank by semantics + template similarity.')
arrow((.326,.482),(.354,.482));arrow((.651,.482),(.679,.482))
fig.patches.append(FancyBboxPatch((.04,.300),.913,.106,boxstyle='round,pad=0.008',transform=fig.transFigure,facecolor='#f8fafc',edgecolor='#d4dce4',linewidth=1))
equation=fig.text(.058,.393,r'$i_t^*=\underset{i:\,s_{t,i}\geq0.05}{\mathrm{arg\,max}}\,[\log(s_{t,i})+0.5\,\mathrm{cos}(z_{t,i},m)]$',fontsize=17,color=ink,va='top')
boxed_text.append((equation,(.04,.300,.913,.106)))
fig.text(.058,.336,'No eligible candidate → no selection. Memory cannot override the semantic threshold.',fontsize=11,color=red,va='top')
fig.text(.058,.314,'Frame-only control: same score and threshold, without the cosine-similarity term.',fontsize=10.5,color=muted,va='top')
fig.text(.04,.269,'C   OBSERVED RESULT',weight='bold',color=ink)
fig.text(.53,.269,'D   WHY THIS MEMORY CANNOT RESCUE THE ERROR',weight='bold',color=ink)
ax=fig.add_axes([.04,.186,.425,.053]);left=0
for count,color,label in [(75,teal,'75 correct'),(1,red,''),(28,'#d4dce4','28 excluded')]:
 ax.barh([0],[count],left=left,color=color,height=.65)
 if label:ax.text(left+count/2,0,label,ha='center',va='center',fontsize=11,color='white' if count==75 else ink)
 left+=count
ax.set_xlim(0,104);ax.axis('off')
fig.text(.04,.173,'104 present-target frames = 75 correct + 1 abstention + 28 excluded',fontsize=10.5,color=ink)
fig.text(.04,.137,'Both methods: 75/76 eligible correct; identical predictions on all 108 frames.\nFour additional absent-target frames: 4/4 correct for both methods.',fontsize=11,color=ink,linespacing=1.6)
fig.text(.53,.223,'Blue-pillar example • e00049, frame 24',fontsize=12,weight='bold',color=ink)
fig.text(.53,.181,r'$s_{t,\mathrm{pillar}}=0.027\;<\;0.05\quad\Longrightarrow\quad\mathrm{rejected}$',fontsize=17,color=red)
fig.text(.53,.130,'The candidate is removed before memory can help.\nNo observed memory benefit ≠ evidence that memory is unnecessary.',fontsize=11,color=ink,linespacing=1.6)
fig.text(.04,.072,'Scope: 16 recorded teacher episodes, 4 validation families; correlated frames and privileged proposals. No learned navigation or Q/K/V patch in this run.',fontsize=10,color=muted)
fig.text(.04,.049,'Present-goal panel omits blue arch. Feature-vector contents and cosine scores were not logged; the equation shows the actual implementation, not measured activations.',fontsize=10,color=muted)
# Fail generation if any method-card text or main equation spills outside its box.
fig.canvas.draw()
renderer=fig.canvas.get_renderer()
for text, (x,y,w,h) in boxed_text:
 extent=text.get_window_extent(renderer).transformed(fig.transFigure.inverted())
 if extent.x0<x or extent.x1>x+w or extent.y0<y or extent.y1>y+h:
  raise RuntimeError('Text exceeds figure box: '+text.get_text())
OUT.mkdir(exist_ok=True)
for ext in ['png','svg']:fig.savefig(OUT/f'memory-baseline.{ext}',dpi=220,facecolor='white')
plt.close(fig)
print(OUT/'memory-baseline.png')
