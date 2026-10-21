"""Plain-language PDF and animated HTML slides from completed run 7246240."""
from pathlib import Path
import json, html
from PIL import Image
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor
from reportlab.platypus import Paragraph
from reportlab.lib.styles import ParagraphStyle
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/pdf/hippocampal-memory';OUT.mkdir(parents=True,exist_ok=True)
A=OUT/'assets'
for r in (0,1):
    with Image.open(A/f'mapping-f0013-route{r}.gif') as im:
        im.seek(125);im.convert('RGB').save(A/f'route{r}-still.png')
with Image.open(A/'mapping-f0013-route0.gif') as im:
    im.seek(0);im.convert('RGB').crop((20,95,468,375)).save(A/'first-view.png')
slides=[
 dict(title='What are we actually doing?',lines=['We recorded walks through Minecraft.', 'We trained a small model to predict the current image’s numbers using movement and earlier images.', 'We are testing whether that memory learns something about places.'],emphasis='The current experiment predicts visual features. It does not learn to walk.'),
 dict(title='First, we record a walk',gif='mapping-f0013-route0.gif',still='route0-still.png',lines=['A scripted controller moves the character.', 'We record camera images and actual movement.', 'The model learns from these recordings afterward.'],emphasis='This moving character is following a script, not a learned navigation policy.'),
 dict(title='Each image becomes 64 numbers',image='first-view.png',lines=['Qwen processes the camera image.', 'We pool its visual features and compress them to 64 numbers.', 'The numbers have no assigned labels such as “red pillar” or “distance”.'],emphasis='Qwen stays frozen. We train a separate, small memory model.'),
 dict(title='g is a list of 32 numbers',lines=['A small network receives the previous g and the measured change in X, Z and heading.', 'It calculates the next 32 numbers in g.', 'Training adjusts how this calculation works. We chose 32 as the model size.'],example='previous g + change in X, Z and heading\n\nGRU network calculates a new g',emphasis='g may encode location, heading or movement history. We have to test which.'),
 dict(title='What exactly do we keep in memory?',lines=['For each earlier frame, we keep two things:', '1. A key calculated from that frame’s g.', '2. The 64 visual numbers extracted from that image.'],example='Earlier frame 0: key 0 + visual list 0\nEarlier frame 1: key 1 + visual list 1\nEarlier frame 2: key 2 + visual list 2',emphasis='We store numerical lists, not object names or text descriptions.'),
 dict(title='“Retrieval” means mixing earlier lists',lines=['We compare the current g key with earlier keys.', 'Those comparisons give weights that add up to 1.', 'We multiply each earlier visual list by its weight, then add the lists.'],example='Illustrative weights, not measured results:\n\nrecalled list = 0.1 × list 0 + 0.7 × list 1 + 0.2 × list 2',emphasis='The recalled result is one list of 64 numbers. It can mix multiple earlier frames.'),
 dict(title='How do we make a prediction?',lines=['Join the 32 numbers in g with the 64 recalled numbers: 96 numbers total.', 'A learned layer converts these to 32 numbers called p.', 'Another learned layer converts p to 64 predicted visual numbers.'],example='96 input numbers\n32 intermediate numbers: p\n64 predicted visual numbers',emphasis='p is an intermediate calculation. Calling it p does not establish a biological place cell.'),
 dict(title='How does the model learn?',lines=['Compare the 64 predicted numbers with the current image’s actual 64 numbers.', 'Square each difference and average them. That is the prediction error.', 'Adjust the small model’s weights to reduce this error.'],example='error = average of (prediction − actual value)²',emphasis='The current visual list enters memory only after its prediction. Qwen does not change.'),
 dict(title='What do the colored maps mean?',gif='mapping-f0013-route1.gif',still='route1-still.png',lines=['“Unit 6” is entry 6 in a model’s internal list.', 'Red means higher numbers. Blue means lower numbers.', 'Each square shows the average number recorded at that world location.', 'Gray means unvisited. White marks the character.'],emphasis='Minecraft coordinates place the colors on the map. The model did not draw this map.'),
 dict(title='What evidence do we have?',result=True,lines=['We replayed the same test recordings through the same trained model.', 'Replacing its recalled information increased prediction error.'],emphasis='Memory helps this model predict visual features. This is not a navigation result.'),
 dict(title='Why is this related to cognitive mapping?',lines=['The goal is to learn relationships between places and experiences.', 'We follow the idea of using movement to query memories of what was seen before.', 'Our current result shows useful retrieval. It does not yet show reliable place or grid cells.'],emphasis='The key question: does the model recognize the same place after a different journey?',source='Behrens et al. (2018), doi:10.1016/j.neuron.2018.10.002. Whittington et al. (2022), arXiv:2112.04035.'),
 dict(title='What comes next?',lines=['Visit the same locations by different routes.', 'Turn to different headings while staying at the same place.', 'Test whether location information survives those changes.'],emphasis='This next experiment is proposed and implemented. Its new results are not available.',source='Completed evidence: pilot 7246240. Proposed protocol: docs/COGNITIVE_MAPPING.html.')
]
audit=json.loads((ROOT/'runs/oscar/hippocampal-mapping-7246240/analysis/cache-audit.json').read_text())
values=[]
for label,key in [('Normal learned retrieval','learned'),('Average of all earlier visual lists','past_mean'),('Replace retrieval with empty-memory vector','empty_retrieval')]:
    v=[r['mse'] for r in audit['prediction'] if r['split']=='test' and r['condition']==key]
    values.append((label,sum(v)/len(v)))
W,H=960,540;c=canvas.Canvas(str(OUT/'hippocampal-memory-explained.pdf'),pagesize=(W,H));c.setTitle('What we actually did: Minecraft visual memory')
def para(x,y,w,s,size=21,bold=False,color='#183044'):
    p=Paragraph(html.escape(s).replace('\n','<br/>'),ParagraphStyle('p',fontName='Helvetica-Bold' if bold else 'Helvetica',fontSize=size,leading=size*1.25,textColor=HexColor(color)))
    _,h=p.wrap(w,600);p.drawOn(c,x,H-y-h);return h
for n,s in enumerate(slides,1):
    para(40,28,880,s['title'],32,True)
    if s.get('gif'):
        c.drawImage(str(A/s['still']),40,105,width=520,height=337,preserveAspectRatio=True,anchor='c')
        y=116
        for line in s['lines']:y+=para(590,y,325,line,21)+22
        para(40,452,870,'GIF: assets/'+s['gif']+' (plays in explained-slides.html)',11,color='#516777')
    elif s.get('image'):
        c.drawImage(str(A/s['image']),40,190,width=400,height=250)
        y=118
        for line in s['lines']:y+=para(485,y,425,line,22)+24
    elif s.get('result'):
        y=108
        for line in s['lines']:y+=para(40,y,870,line,21)+14
        y=255
        for label,val in values:
            para(40,y,720,label,21);para(800,y,110,f'{val:.3f}',24,True);y+=48
        para(40,417,870,'Lower error is better. One test geometry, two routes, one trained model seed.',14,color='#516777')
    else:
        y=110
        for line in s['lines']:y+=para(40,y,875,line,22)+20
        if s.get('example'):
            para(50,max(y+4,305),860,s['example'],19,color='#147d92')
    para(40,477,875,s['emphasis'],15,True)
    para(40,519,850,s.get('source','Completed pilot 7246240. Scripted Minecraft movement. Frozen Qwen. Learned small memory model.'),8,color='#516777')
    para(915,515,30,str(n),11);c.showPage()
c.save()
parts=[]
for i,s in enumerate(slides):
    content='<div class="words">'+''.join('<p>'+html.escape(x)+'</p>' for x in s['lines'])+'</div>'
    if s.get('gif'):
        content='<div class="media"><img data-gif="assets/'+s['gif']+'" data-still="assets/'+s['still']+'" src="assets/'+s['gif']+'"><small>assets/'+s['gif']+'</small><button class="pause">Pause animation</button></div>'+content
    if s.get('image'):content='<div class="media"><img src="assets/'+s['image']+'"></div>'+content
    if s.get('example'):content+='<pre>'+html.escape(s['example'])+'</pre>'
    if s.get('result'):content+='<table>'+''.join(f'<tr><td>{label}</td><td>{v:.3f}</td></tr>' for label,v in values)+'</table><small>Lower is better. One test geometry, two routes, one seed. These are numerical errors, not percentages.</small>'
    parts.append(f'<section id="slide-{i+1}" class="slide'+(' visible' if i==0 else '')+'"><h1>'+html.escape(s['title'])+'</h1><div class="content '+('split' if s.get('gif') or s.get('image') else '')+'">'+content+'</div><p class="takeaway">'+html.escape(s['emphasis'])+'</p><small>'+html.escape(s.get('source','Completed pilot 7246240. Scripted movement with privileged odometry.'))+'</small></section>')
(OUT/'explained-slides.html').write_text('''<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Minecraft memory, explained</title><style>*{box-sizing:border-box}body{margin:0;background:#fff;color:#183044;font:clamp(17px,2vw,28px)/1.35 system-ui}.slide{display:none;min-height:90vh;padding:3vh 4vw 1vh}.slide.visible{display:block}h1{font-size:clamp(28px,3.4vw,46px);margin:0 0 3vh}.content{max-width:1400px;margin:auto}.split{display:grid;grid-template-columns:60% 36%;gap:4%}.words p{margin:0 0 2.5vh}.media img{max-width:100%;max-height:64vh;object-fit:contain}.media small{display:block;overflow-wrap:anywhere}small{font-size:clamp(10px,1vw,14px);color:#516777}pre{font:inherit;white-space:pre-wrap;color:#147d92;margin:2vh 0}table{width:100%;border-collapse:collapse}td{padding:1.7vh 0;border-bottom:1px solid #ddd}td:last-child{text-align:right;font-weight:bold}.takeaway{font-size:.8em;font-weight:650;margin-top:2vh}nav{display:flex;gap:18px;align-items:center;padding:10px 4vw;background:#edf3f5;font-size:16px}button{cursor:pointer;padding:8px 16px;border:1px solid #aaa;background:white;border-radius:4px}.media button{font-size:12px;padding:4px 8px}a{color:#147d92}@media(max-width:800px){.split{display:block}.media img{max-height:43vh}.slide{min-height:85vh}}@media print{nav,button{display:none}.slide{display:block;break-after:page}}</style>'''+''.join(parts)+'''<nav><button id="prev">Previous</button><span id="count"></span><button id="next">Next</button><button id="full">Fullscreen</button><a href="hippocampal-memory-explained.pdf">PDF version</a></nav><script>const ss=[...document.querySelectorAll('.slide')];let n=0;function go(i){n=Math.max(0,Math.min(ss.length-1,i));ss.forEach((s,j)=>s.classList.toggle('visible',n===j));document.querySelector('#count').textContent=`${n+1} / ${ss.length}`;history.replaceState(null,'','#slide-'+(n+1));}document.querySelector('#next').onclick=()=>go(n+1);document.querySelector('#prev').onclick=()=>go(n-1);document.querySelector('#full').onclick=()=>{if(!document.fullscreenElement)document.documentElement.requestFullscreen?.();else document.exitFullscreen?.();};document.addEventListener('keydown',e=>{if(['ArrowRight','PageDown',' '].includes(e.key)){e.preventDefault();go(n+1)}if(['ArrowLeft','PageUp'].includes(e.key)){e.preventDefault();go(n-1)}});document.querySelectorAll('.pause').forEach(b=>b.onclick=()=>{const im=b.parentElement.querySelector('img');const stopped=b.textContent.startsWith('Pause');im.src=stopped?im.dataset.still:im.dataset.gif;b.textContent=stopped?'Play animation':'Pause animation';});go(Number(location.hash.replace('#slide-',''))-1||0);</script>''')
print(OUT/'hippocampal-memory-explained.pdf')
