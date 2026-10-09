"""Scientific diagnostic figures from measured model activity, not generated imagery."""
import html
import json
import math
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from .hippocampal import rate_maps
from .io import atomic_json


def font(size):
    for path in ['/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf','/System/Library/Fonts/Supplemental/Arial.ttf']:
        if Path(path).exists():return ImageFont.truetype(path,size)
    return ImageFont.load_default()


def map_image(values,count,lo,hi,size=144):
    a=np.nan_to_num((values-lo)/max(hi-lo,1e-8),nan=0).clip(0,1)
    rgb=np.stack([255*a,110*(1-np.abs(2*a-1)),255*(1-a)],axis=-1).astype('uint8')
    rgb[count==0]=(45,45,45)
    return Image.fromarray(rgb[::-1]).resize((size,size),Image.Resampling.NEAREST)


def render(out,entries,activations,selected):
    out=Path(out);dest=out/'analysis';dest.mkdir()
    reports=[];coverage=[]
    for entry in entries:
        eid=entry['episode'];acts=activations[eid];poses=[f['pose'] for f in entry['frames']]
        xy=np.array([[p['x'],p['z']] for p in poses]);bounds=[-18.,18.,-5.,27.]
        _,counts=rate_maps(acts['g'],poses,bounds)
        coverage.append(dict(episode=eid,visited_bins=int((counts>0).sum()),total_bins=counts.size,
            bins_with_3_samples=int((counts>=3).sum()),frames=len(poses),route_complete=entry['route_complete'],
            gridness='Not estimated: this pilot does not establish adequate spatial/heading coverage or shuffle-controlled periodicity.'))
        limits={k:[float(acts[k][:,units].min()),float(acts[k][:,units].max())] for k,units in selected.items()}
        frames=[]
        for t in range(len(poses)):
            canvas=Image.new('RGB',(1220,790),'#f5f6f8');draw=ImageDraw.Draw(canvas)
            draw.text((20,12),('Minecraft spatial-memory pilot | ' if entry.get('is_minecraft',False) else 'NON-MINECRAFT synthetic fixture | ')+eid,font=font(20),fill='black')
            draw.text((20,43),'SCRIPTED EXPLORATION + PRIVILEGED ODOMETRY | learned g/p candidates, not proven grid/place cells',font=font(14),fill='#8a2500')
            with Image.open(out/entry['frames'][t]['frame']) as im:canvas.paste(im.convert('RGB'),(20,95))
            draw.text((20,72),'First-person recording (HUD preserved)',font=font(16),fill='black')
            pose=poses[t]
            draw.text((20,384),f"Frame {t}  X={pose['x']:.2f} Z={pose['z']:.2f} yaw={pose['yaw']:.1f}",font=font(16),fill='black')
            draw.text((20,410),'Coordinates below are evaluator ground truth.',font=font(14),fill='black')
            bx,by,bw,bh=35,468,410,275
            draw.rectangle((bx,by,bx+bw,by+bh),fill='white',outline='gray')
            def point(x,z):return bx+(x-bounds[0])/(bounds[1]-bounds[0])*bw,by+bh-(z-bounds[2])/(bounds[3]-bounds[2])*bh
            for obj in entry['objects']:
                low,high=obj['bounds'];a=point(low[0],high[2]);b=point(high[0],low[2]);draw.rectangle((*a,*b),fill=obj['color'])
            path=[point(*v) for v in xy[:t+1]]
            if len(path)>1:draw.line(path,fill='#777777',width=2)
            x,y=path[-1];yaw=math.radians(pose['yaw']);draw.ellipse((x-4,y-4,x+4,y+4),fill='black')
            draw.line((x,y,x-14*math.sin(yaw),y-14*math.cos(yaw)),fill='black',width=3)
            draw.text((20,445),'Fixed world map: +X right, +Z up; arrow = heading',font=font(14),fill='black')
            for row,key in enumerate(['g','p']):
                yy=100+row*285
                label='g: learned motion-state units' if key=='g' else 'p: learned retrieved sensory / spatial conjunction units'
                draw.text((500,yy-24),label,font=font(17),fill='black')
                maps,count=rate_maps(acts[key][:t+1],poses[:t+1],bounds)
                lo,hi=limits[key]
                for col,unit in enumerate(selected[key]):
                    xx=500+col*175
                    canvas.paste(map_image(maps[:,:,unit],count,lo,hi),(xx,yy+26))
                    draw.text((xx,yy),f'Unit {unit}',font=font(15),fill='black')
                    # Same fixed allocentric coordinates across every frame and every panel.
                    ux=(pose['x']-bounds[0])/(bounds[1]-bounds[0])*144
                    uz=144-(pose['z']-bounds[2])/(bounds[3]-bounds[2])*144
                    draw.ellipse((xx+ux-3,yy+26+uz-3,xx+ux+3,yy+26+uz+3),outline='white',width=2)
                    draw.text((xx,yy+178),f"now {acts[key][t,unit]:+.3f}",font=font(14),fill='black')
                draw.text((500,yy+210),f'Blue {lo:+.2f} to red {hi:+.2f}; gray = unvisited; running occupancy mean',font=font(13),fill='black')
            draw.text((500,675),'Unit selection: variance on training trajectories only.',font=font(15),fill='black')
            draw.text((500,701),'Color scale fixed per episode using its full recorded activity.',font=font(14),fill='black')
            draw.text((500,727),'A localized / periodic-looking map is not evidence of a biological cell.',font=font(14),fill='black')
            frames.append(canvas)
        gif=dest/(eid+'.gif');frames[0].save(gif,save_all=True,append_images=frames[1:],duration=entry.get('frame_interval_ms',180),loop=0)
        frames[-1].save(dest/(eid+'.png'))
        reports.append(f'<h2>{html.escape(eid)} — {entry["split"]}, route {entry["route"]}</h2><img loading="lazy" src="{gif.name}"><p class="path">{html.escape(str(gif))}</p>')
    provenance='Real Minecraft recording' if all(e.get('is_minecraft',False) for e in entries) else 'NON-MINECRAFT synthetic test fixture'
    (dest/'index.html').write_text('''<!doctype html><meta charset="utf-8"><title>Minecraft hippocampal-inspired memory</title>
<style>body{font:17px/1.6 system-ui;max-width:1250px;margin:30px auto;padding:20px}img{width:100%}.path{overflow-wrap:anywhere;font-family:monospace}</style>
<h1>Allocentric spatial-activity replay</h1><p>PROVENANCE; movement is scripted, not a learned policy. A self-supervised memory predicts frozen visual features using prior observations and privileged motion deltas. g and p denote candidate latent populations. No grid/place-cell emergence claim is established.</p>
<p>Maps accumulate occupancy-normalized activation over visited positions. Axes remain fixed when the camera turns. Dark gray means unobserved, not low activation. Color scales use the full episode for display only. Training-only unit selection avoids choosing attractive test maps. Prediction metrics and coverage are in review.json.</p>'''.replace('PROVENANCE',provenance)+''.join(reports))
    atomic_json(dest/'coverage.json',coverage)
    return coverage
