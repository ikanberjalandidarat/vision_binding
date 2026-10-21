from pathlib import Path
import json, shutil, math
import numpy as np
from PIL import Image
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor, white
from reportlab.platypus import Paragraph
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.utils import ImageReader
ROOT=Path.cwd(); OUT=ROOT/'output/pdf/hippocampal-memory'; TMP=ROOT/'tmp/pdfs/hippocampal-deck'
run=ROOT/'runs/oscar/hippocampal-mapping-7246240'
audit=json.loads((run/'analysis/cache-audit.json').read_text())
entries=json.loads((run/'sequences.json').read_text())
for r in (0,1):
    source=Path(f'/Users/lianfei/Downloads/mapping-f0013-route{r}.gif')
    shutil.copy2(source,OUT/'assets'/source.name)
    im=Image.open(source)
    for t in (0,60,130,125):
        im.seek(t); frame=im.convert('RGB');frame.save(TMP/f'r{r}-{t}.png')
        frame.crop((20,95,468,375)).save(TMP/f'view-r{r}-{t}.png')
    im.seek(125);frame=im.convert('RGB')
    frame.crop((500,76,1210,350)).save(TMP/f'g-r{r}.png')
    frame.crop((500,360,1210,635)).save(TMP/f'p-r{r}.png')
# Scientific trajectory plot from recorded simulator telemetry.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axs=plt.subplots(1,2,figsize=(9,3),layout='constrained')
for r,ax in enumerate(axs):
    e=next(e for e in entries if e['family']=='f0013' and e['route']==r)
    xy=np.array([[f['pose']['x'],f['pose']['z']] for f in e['frames']])
    ax.plot(xy[:,0],xy[:,1],color=['#147d92','#bf6337'][r],lw=1.5)
    ax.scatter(*xy[0],c='black',s=30,label='start');ax.set(title=f'Validation route {r}',xlabel='World X (blocks)',ylabel='World Z (blocks)',xlim=(-8,8),ylim=(-1,12),aspect='equal');ax.grid(alpha=.2)
fig.savefig(TMP/'routes.png',dpi=200);plt.close(fig)
W,H=960,540;INK='#182d40';MUTED='#526575';ACC='#147d92';BG='#ffffff'
c=canvas.Canvas(str(OUT/'hippocampal-memory-research.pdf'),pagesize=(W,H));c.setTitle('Hippocampal-inspired memory in Minecraft');c.setAuthor('Minecraft Binding research')
page=0

def text(x,y,w,s,size=18,color=INK,bold=False):
    st=ParagraphStyle('s',fontName='Helvetica-Bold' if bold else 'Helvetica',fontSize=size,leading=size*1.32,textColor=HexColor(color))
    p=Paragraph(s,st);_,h=p.wrap(w,H);p.drawOn(c,x,H-y-h);return h

def title(s,subtitle=None):
    global page
    page+=1;c.setFillColor(white);c.rect(0,0,W,H,fill=1,stroke=0)
    text(42,30,876,s,30,bold=True)
    if subtitle:text(42,78,876,subtitle,14,MUTED)

def footer(source='Completed Minecraft pilot 7246240'):
    text(42,505,830,source,9,MUTED);text(906,502,30,str(page),11,MUTED);c.showPage()

def pic(path,x,y,w,h):
    with Image.open(path) as im:
        iw,ih=im.size
    scale=min(w/iw,h/ih);dw,dh=iw*scale,ih*scale
    c.drawImage(str(path),x+(w-dw)/2,H-y-dh,width=dw,height=dh,mask='auto')

def bullet(y,head,body,x=42,w=876):
    text(x,y,w,head,20,bold=True);text(x,y+32,w,body,17)

def table(y,headers,rows,widths):
    x0=42;yy=y
    for i,row in enumerate([headers]+rows):
        x=x0
        for val,w in zip(row,widths):text(x+8,yy,w-16,str(val),16 if i else 15,bold=i==0);x+=w
        yy+=43
        c.setStrokeColor(HexColor('#dbe3e8'));c.line(x0,H-yy+8,x0+sum(widths),H-yy+8)

# 1
text(44,65,865,'Hippocampal-inspired memory<br/>in Minecraft',43,bold=True)
text(46,205,780,'Does movement-conditioned visual memory learn spatial relationships that generalize across routes?',25)
text(46,340,760,'Completed pilot and offline activation audit',21,color=ACC,bold=True)
text(46,382,760,'Qwen visual features, associative retrieval and candidate g/p populations',17)
text(46,447,800,'Research update  |  October 2026',13,MUTED);page=1;footer('Evidence: run 7246240. New controlled experiment remains pending.')
# 2
title('Cognitive maps and associative memory')
bullet(115,'Behrens et al. (2018)','A cognitive map organizes relationships so an agent can make flexible inferences. Spatial navigation provides a concrete setting for studying this principle.')
bullet(225,'Whittington et al. (2022)','Recurrent movement-dependent position representations query past sensory experiences. The resulting model connects transformer retrieval with hippocampal models.')
bullet(350,'Our research question','After taking a different route, can the representation recover where the agent is and what it previously experienced there?')
footer('Sources: Behrens et al., Neuron (2018), doi:10.1016/j.neuron.2018.10.002; Whittington et al., arXiv:2112.04035')
# 3
title('Our implementation','A self-supervised predictive memory. Qwen stays frozen.')
table(117,['Component','Implementation'],[
 ['Visual input','Qwen2-VL-7B-Instruct vision features, projected to 64 dimensions'],
 ['Movement input','Measured world-axis displacement and heading change'],
 ['g state','32-dimensional GRU state updated from movement'],
 ['Retrieval','Current g queries previous g keys and recalls past visual features'],
 ['p state','32-dimensional tanh layer combining g and recalled vision'],
 ['Learning objective','Predict current features before writing them into memory']],[170,706])
text(42,413,876,'Our p layer is a learned conjunction. It differs from the original TEM outer-product construction. The g/p names denote candidate populations.',16,MUTED)
footer('Implementation: src/mc_binding/hippocampal.py; src/mc_binding/hippocampal_study.py')
#4
title('Completed exploration pilot','Six geometries, two scripted routes each, 1,697 recorded frames, one model seed (731)')
pic(TMP/'routes.png',32,108,565,300)
text(620,123,285,'Training',20,bold=True);text(620,155,285,'Route 0 in four geometries.<br/>100 epochs of visual prediction.',17)
text(620,230,285,'Evaluation',20,bold=True);text(620,262,285,'Route 1 in training geometries.<br/>Both routes in one validation and one test geometry.',17)
text(42,422,876,'All 12 routes completed. Cached features align with recorded poses. Exploration and odometry use simulator telemetry. The memory does not receive absolute coordinates.',17)
footer('Source: sequences.json and manifest.json. Plots show recorded validation geometry f0013.')
#5
title('g activity during exploration','Four units selected by activation variance on training episodes')
pic(TMP/'view-r0-125.png',42,130,270,170);pic(TMP/'g-r0.png',334,130,585,226)
text(42,325,270,'Validation route 0<br/>Frame 125',18,bold=True)
text(334,377,580,'Colors show accumulated mean activation in visited world bins. Gray cells are unvisited. The cursor marks the agent position.',17)
text(42,450,876,'GIF: assets/mapping-f0013-route0.gif',11,MUTED)
footer('The plotting code supplies the fixed world axes. These panels do not establish grid-cell periodicity.')
#6
title('p activity combines motion state with recalled vision','Four units selected using the same training-only rule')
pic(TMP/'view-r1-125.png',42,130,270,170);pic(TMP/'p-r1.png',334,130,585,226)
text(42,325,270,'Validation route 1<br/>Frame 125',18,bold=True)
text(334,377,580,'p depends on both movement history and retrieved sensory features. Its activity can change as the memory accumulates experience.',17)
text(42,450,876,'GIF: assets/mapping-f0013-route1.gif',11,MUTED)
footer('Candidate conjunction units. Our implementation does not reproduce the original TEM p construction.')
#7
title('The camera view changes along a route','Recorded Minecraft frames, validation route 0')
for i,t in enumerate((0,60,130)):
    x=42+i*296;pic(TMP/f'view-r0-{t}.png',x,135,280,175);text(x,325,280,f'Frame {t}',18,bold=True)
text(42,385,876,'The same world can generate different visual features as heading and position change. A sensory-prediction objective may therefore preserve view information as well as location.',20)
text(42,464,876,'GIF: assets/mapping-f0013-route0.gif',11,MUTED)
footer('PDF shows selected still frames. Open animations.html for the synchronized full activity replays.')
#8
title('Retrieval improves held-out sensory prediction','Mean normalized feature MSE across the two routes in the test geometry. Lower is better.')
conditions=[('Learned retrieval','learned'),('Past visual mean','past_mean'),('Empty retrieval','empty_retrieval'),('Untrained model','untrained')]
values=[np.mean([r['mse'] for r in audit['prediction'] if r['split']=='test' and r['condition']==key]) for _,key in conditions]
for i,((label,key),v) in enumerate(zip(conditions,values)):
    y=145+i*59;text(42,y+2,238,label,19)
    c.setFillColor(HexColor(ACC if i==0 else '#879ba9'));c.rect(295,H-y-28,float(v)*450,30,fill=1,stroke=0)
    text(310+float(v)*450,y,100,f'{v:.3f}',20,bold=True)
text(42,408,876,'Replacing retrieval at inference increases prediction error. These controls test the trained pathway, rather than separately retrained architectures.',18)
footer('Source: analysis/cache-audit.json. Two correlated trajectories in one test geometry, not independent trials.')
#9
title('Activity repeats most strongly along the same route','Descriptive medians across unit correlations and episodes or geometries')
table(130,['Comparison','g','p'],[
 ['Same-route, same-heading matched revisits','0.955','0.718'],
 ['Spatial maps across opposite routes','-0.181','-0.198']],[620,128,128])
text(42,280,876,'The two rows use different comparisons. The first pairs nearby positions with similar heading. The second correlates occupancy-averaged spatial maps across routes.',18)
text(42,370,876,'Route, heading and appearance change together. These results do not isolate the cause of the difference or rule out a joint position-and-heading representation.',20,bold=True)
footer('Source: cache-audit.json. Frames and units are correlated. These medians are not replication counts or significance tests.')
#10
title('What the evidence supports')
bullet(115,'Useful associative retrieval','Ablating retrieval harms sensory prediction. The learned memory contributes to the completed task.')
bullet(225,'A movement-conditioned representation','g receives motion during inference. Visual prediction errors train its parameters, but observations do not directly correct g online.')
bullet(350,'Spatial generalization remains unresolved','Colored fields alone do not identify place/grid cells. Population-level spatial information may coexist with heading and route dependence.')
footer('Architecture fact: hippocampal.py. Experimental evidence: offline audit of model seed 731.')
#11
title('Proposed cognitive-mapping experiment','Implemented, but no new experimental results are available')
bullet(115,'Repeated locations with varied movement histories','Four intersecting loops repeatedly return to a shared anchor. Each return samples four cardinal headings.')
bullet(225,'Separate learning and evaluation routes','Train on two route orders. Evaluate reversed and reordered routes and held-out geometries.')
bullet(335,'Measure accessible spatial information','Fit evaluator-only location readouts on frozen training representations. Compare learned g/p with untrained states and a time/heading baseline.')
text(42,453,876,'The next run remains pending after an environment startup failure. No new cognitive-mapping results are available.',14,MUTED)
footer('Design: docs/COGNITIVE_MAPPING.html. Location labels belong to the offline evaluator, not the memory objective.')
#12
title('Research conclusion and sources')
text(42,115,865,'We have evidence that movement-conditioned memory helps predict visual experience.',28,bold=True)
text(42,215,865,'The next scientific question is whether its spatial relationships transfer across routes and headings.',24,color=ACC)
text(42,315,870,'Behrens et al. (2018). What Is a Cognitive Map? Organizing Knowledge for Flexible Behavior.<br/>doi.org/10.1016/j.neuron.2018.10.002',14)
c.linkURL('https://doi.org/10.1016/j.neuron.2018.10.002',(42,H-365,915,H-315),relative=0)
text(42,379,870,'Whittington, Warren and Behrens (2022). Relating transformers to models and neural representations of the hippocampal formation.<br/>arxiv.org/abs/2112.04035',14)
c.linkURL('https://arxiv.org/abs/2112.04035',(42,H-435,915,H-379),relative=0)
text(42,450,870,'Local evidence: runs/oscar/hippocampal-mapping-7246240/analysis/cache-audit.json<br/>Companion: animations.html (GIF paths appear beneath each replay)',12,MUTED)
footer('Completed pilot 7246240. Scripted exploration with privileged odometry. Candidate g/p representations.')
c.save()
(OUT/'animations.html').write_text('''<!doctype html><meta charset="utf-8"><title>Hippocampal memory replays</title><style>body{font:19px/1.6 system-ui;max-width:1250px;margin:35px auto;padding:20px}img{width:100%}code{overflow-wrap:anywhere}</style><h1>Measured g/p activity during Minecraft exploration</h1><p>Completed pilot 7246240. Scripted movement and privileged odometry. The learned activity uses simulator coordinates for display. These are candidate model populations, not established biological place/grid cells.</p><p><a href="hippocampal-memory-research.pdf">Presentation PDF</a></p>'''+''.join(f'<h2>Validation geometry f0013, route {r}</h2><img src="assets/mapping-f0013-route{r}.gif"><p><code>assets/mapping-f0013-route{r}.gif</code></p>' for r in (0,1)))
print(OUT/'hippocampal-memory-research.pdf')
