"""Scientific schematic of the recorded retrieval task; no synthetic result data."""
import json
import math
from pathlib import Path
from html import escape

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/revisit-heading-comparison';OUT.mkdir(parents=True,exist_ok=True)
source=ROOT/'output/revisit-comparison-8977923/comparison.json'
data=json.loads(source.read_text())
blue='#2463b4';orange='#cf6829';ink='#182c43';muted='#55677c'
parts=['<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="1000" viewBox="0 0 1600 1000">',
'<rect width="1600" height="1000" fill="#f5f7fa"/>']

def text(x,y,s,size=22,color=ink,weight='normal',anchor='start'):
    parts.append(f'<text x="{x}" y="{y}" font-family="Arial, Helvetica, sans-serif" font-size="{size}" fill="{color}" font-weight="{weight}" text-anchor="{anchor}">{escape(str(s))}</text>')

def rect(x,y,w,h,fill,stroke='none',radius=0):
    parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" fill="{fill}" stroke="{stroke}"/>')

def line(x,y,xx,yy,color,width=2,dash=''):
    parts.append(f'<line x1="{x}" y1="{y}" x2="{xx}" y2="{yy}" stroke="{color}" stroke-width="{width}"/ >'.replace("/ >", "/>"))

def mapview(x,y,angle,color):
    rect(x,y,192,192,'#f1f5f0','#d0dacd',10)
    # Same world position in both views. Sixteen points represent gallery sites.
    for i in range(4):
        for j in range(4):
            cx=x+30+i*44;cy=y+30+j*44
            parts.append(f'<circle cx="{cx}" cy="{cy}" r="4" fill="#aab7a8"/>')
    cx=x+74;cy=y+118
    parts.append(f'<circle cx="{cx}" cy="{cy}" r="12" fill="{color}"/>')
    parts.append(f'<circle cx="{cx}" cy="{cy}" r="8" fill="white"/>')
    a=math.radians(angle);dx,dy=math.sin(a),-math.cos(a)
    ex,ey=cx+60*dx,cy+60*dy
    sx,sy=cx+14*dx,cy+14*dy
    parts.append(f'<polygon points="{sx-3*dy},{sy+3*dx} {ex-3*dy},{ey+3*dx} {ex+3*dy},{ey-3*dx} {sx+3*dy},{sy-3*dx}" fill="{color}"/>')
    px,py=-dy,dx
    pts=f'{ex},{ey} {ex-16*dx+9*px},{ey-16*dy+9*py} {ex-16*dx-9*px},{ey-16*dy-9*py}'
    parts.append(f'<polygon points="{pts}" fill="{color}"/>')

text(52,63,'Does the same place match when the viewing direction changes?',36,weight='bold')
text(52,102,'One Minecraft world • Held-out routes 4 and 5 • Both retrieval directions • 16 candidate sites',23,muted)
rect(52,133,1496,102,'#e7eef7',radius=14)
text(76,171,'QUERY',17,blue,'bold');text(76,207,'16-frame clip on one route',24)
text(470,197,'→',35,muted)
text(535,171,'RETRIEVAL',17,blue,'bold');text(535,207,'Choose 1 of 16 sites on the other route',24)
text(1085,197,'→',35,muted)
text(1150,171,'SCORE',17,blue,'bold');text(1150,207,'Is it the same site?',24)
rows=[]
for index,(delta,title) in enumerate([(0,'Same direction'),(90,'Quarter turn'),(180,'Opposite direction')]):
    x=52+index*506
    rect(x,260,484,620,'white','#dce3eb',16)
    text(x+24,304,title,28,weight='bold');text(x+460,304,f'{delta}°',29,blue,'bold','end')
    text(x+24,341,'Same world position, compared across routes',18,muted)
    text(x+120,379,'Query route',19,blue,'bold','middle')
    text(x+360,379,'Other route',19,orange,'bold','middle')
    mapview(x+24,396,0,blue);mapview(x+264,396,delta,orange)
    text(x+242,504,'↔',30,muted,anchor='middle')
    text(x+242,620,'Correct gallery candidate illustrated above',18,muted,anchor='middle')
    text(x+242,649,'The test searches all 16 candidates.',18,muted,anchor='middle')
    line(x+24,671,x+460,671,'#dce3eb')
    pix=next(r for r in data['results'] if r['heading_difference']==delta and r['feature']=='masked_endpoint_rgb_32x20')
    video=[next(r for r in data['results'] if r['heading_difference']==delta and r['feature']==f'video_layer{l}') for l in (3,11,23)]
    vals=[r['accuracy']*100 for r in video]
    text(x+24,709,'CORRECT SITE RETRIEVAL',16,muted,'bold')
    text(x+24,750,'Pixel matching',22);text(x+460,750,f"{pix['accuracy']*100:.2f}%",30,'#70439c','bold','end')
    text(x+24,799,'V-JEPA video',22);text(x+460,799,f'{min(vals):.1f}–{max(vals):.1f}%',30,blue,'bold','end')
    text(x+24,846,'Video range: blocks 4, 12 and 24',18,muted)
    rows.append(dict(heading_difference_degrees=delta,label=title,queries=pix['queries'],pixel_matching=pix,video_layers=video,
        video_accuracy_range=[min(vals)/100,max(vals)/100]))
text(52,922,'Chance = 6.25%. No contrastive training here: frozen-feature cosine retrieval vs endpoint-pixel MSE.',22,weight='bold')
text(52,956,'Maps are schematics, not recorded trajectories. Arrows show relative viewing directions, not absolute Minecraft yaw.',19,muted)
text(52,984,'This comparison tests viewpoint robustness; it does not by itself establish a cognitive map. Repeated-frame controls are in the full report.',18,muted)
parts.append('</svg>')
(OUT/'heading-comparison.svg').write_text('\n'.join(parts))
(OUT/'heading-comparison.json').write_text(json.dumps(dict(title='Within-world place retrieval across headings',
    source=str(source.relative_to(ROOT)),dataset_sha256=data['dataset_sha256'],protocol_sha256=data['protocol_sha256'],
    schematic=True,not_contrastive_training=True,model='Frozen V-JEPA 2 ViT-L/16',
    query_gallery='Routes 4 and 5, both directions; one gallery item per site at the specified heading offset',
    candidates=16,chance_accuracy=.0625,rows=rows,
    notes=['90-degree results combine +90 and -90 offsets.', 'Site coordinates are used to score choices, not supplied to the encoder.',
           'Accuracy ranges show layer minima/maxima, not confidence intervals.', 'Pixel resolution 32x20 was specified after feature inspection; no sweep.',
           'Schematic arrows show relative headings. Correct candidate illustrated, not necessarily the actual retrieved candidate.']),indent=2)+'\n')
print(OUT)
