"""Dataset-only diverse world recording and Oscar review gallery. No model testing."""
import argparse
import bisect
import html
import json
import math
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from .io import atomic_json, digest, file_hash, source_hash
from .diverse_worlds import specification, NAMES


def font(size=16):
    for path in ['/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf','/System/Library/Fonts/Supplemental/Arial.ttf']:
        if Path(path).exists():return ImageFont.truetype(path,size)
    return ImageFont.load_default()


def review_frame(root,record,title,tick):
    with Image.open(root/record['frame']) as im:
        out=Image.new('RGB',(640,350),'white');out.paste(im.convert('RGB'),(0,35))
    d=ImageDraw.Draw(out);d.font=font(14);d.text((8,8),title,fill='black')
    p=record['pose'];d.text((8,320),f"t={tick/20:.1f}s X={p['x']:.1f} Y={p['y']:.1f} Z={p['z']:.1f} | scripted, 4x",fill='black')
    x=465+(p['x']+16)/32*160;y=300-(p['z']+16)/32*250
    d.rectangle((465,50,625,300),outline='gray');d.ellipse((x-4,y-4,x+4,y+4),fill='red')
    d.text((465,35),'World X / Z',fill='black')
    return out


def collect(a):
    from minestudio.simulator import MinecraftSim
    from minestudio.simulator.callbacks import CommandsCallback
    from .backends.minestudio_smoke import verify_pose,json_value
    spec=specification(a.index);out=Path(a.root)/spec['id'];out.mkdir(parents=True,exist_ok=False)
    atomic_json(out/'specification.json',spec)
    atomic_json(out/'manifest.json',dict(protocol='diverse_environments_v2',seed=731,source_hash=source_hash(),
        spec_sha256=digest(spec),rgb=[448,280],tick_ms=50,capture_every_ticks=8,action_chunk_ticks=2,
        routes=spec['routes'],settle_ticks=200,max_route_ticks=a.max_ticks,
        is_minecraft=True,review_status='awaiting_user_review',model_used=False,
        scope='Eight authored environment categories, including mazes and elevation; review required.'))
    atomic_json(out/'status.json',dict(state='running'));sim=None;entries=[]
    try:
        sim=MinecraftSim(action_type='env',obs_size=(448,280),render_size=(448,280),seed=731)
        obs,info=sim.reset()
        def step(action):
            nonlocal obs,info
            obs,_,done,truncated,info=sim.step(action)
            if done or truncated or ('error' in info):raise RuntimeError('Simulator terminated or returned error')
        obs,info=CommandsCallback(commands=['/gamemode creative','/forceload add -16 -16 16 16','/tp @p 0.5 200 0.5 0 0']).after_reset(sim,obs,info)
        for _ in range(200):step(sim.noop_action())
        for offset in range(0,len(spec['commands']),100):
            obs,info=CommandsCallback(commands=spec['commands'][offset:offset+100]).after_reset(sim,obs,info)
            for _ in range(4):step(sim.noop_action())
        for _ in range(200):step(sim.noop_action())
        verify_pose(info,spec['start'],0)
        for r in range(2):
            obs,info=CommandsCallback(commands=[spec['commands'][-1]]).after_reset(sim,obs,info)
            for _ in range(200):step(sim.noop_action())
            verify_pose(info,spec['start'],0)
            folder=out/f'route-{r}';folder.mkdir();frames=[];actions=[];j=0;points=spec['routes'][r];progress_tick=0
            # Initial full heading scan, then this world's terrain-aware route.
            for tick in range(0,a.max_ticks+1,2):
                pose=json_value(info['player_pos'])
                if not all(math.isfinite(float(pose[k])) for k in ('x','y','z','yaw')):raise ValueError('Nonfinite pose')
                if not(-15<pose['x']<15 and -15<pose['z']<15 and 199.5<pose['y']<211):raise ValueError('Pose outside world bounds')
                if tick%8==0:
                    pixels=np.asarray(obs['image'])
                    if pixels.shape!=(280,448,3) or pixels.std()<1:raise ValueError('Invalid RGB')
                    path=folder/f'frame-{tick:05d}.png';Image.fromarray(pixels).save(path)
                    frames.append(dict(frame=str(path.relative_to(out)),elapsed_ticks=tick,pose=pose,actions_since_previous=actions,sha256=file_hash(path)));actions=[]
                    atomic_json(folder/'partial.json',dict(frames=frames,waypoint=j))
                if tick>=72:
                    previous=j
                    while j<len(points) and math.hypot(pose['x']-points[j][0],pose['z']-points[j][2])<.45 and abs(pose['y']-points[j][1])<.6:j+=1
                    if j!=previous:progress_tick=tick
                    if j==len(points):break
                    if tick-progress_tick>400:raise RuntimeError(f'Stuck near waypoint {j}; inspect last recorded frames')
                    goal=points[j];dx,dz=goal[0]-pose['x'],goal[2]-pose['z']
                    turn=(math.degrees(math.atan2(-dx,dz))-pose['yaw']+180)%360-180
                    action=0 if abs(turn)<15 else 1 if turn<0 else 2
                else:action=2
                if tick==a.max_ticks:break
                command=sim.noop_action();command['forward']=int(action==0)
                command['camera']=np.array([0.,-5. if action==1 else 5. if action==2 else 0.],np.float32)
                jumping=bool(action==0 and tick>=72 and points[j][1]>pose['y']+.25)
                if 'jump' not in command:raise ValueError('Simulator lacks jump action')
                command['jump']=int(jumping)
                for _ in range(2):step(command)
                actions.append(dict(index=action,jump=jumping))
            entry=dict(episode=f'{spec["id"]}-route{r}',is_minecraft=True,route=r,frames=frames,
                       route_complete=j==len(points),waypoints_reached=j,waypoints_total=len(points),end_tick=tick)
            entries.append(entry);atomic_json(out/'sequences.json',entries)
            if not entry['route_complete']:raise RuntimeError('Incomplete route; inspect metadata; not an accepted recording')
            selected=frames[::4]
            images=[review_frame(out,f,f'{spec["id"]} {spec["name"]} | route {r}',f['elapsed_ticks']) for f in selected]
            images[0].save(out/f'route-{r}.gif',save_all=True,append_images=images[1:],duration=400,loop=0)
        atomic_json(out/'status.json',dict(state='complete',review_status='awaiting_user_review'))
    except BaseException as exc:
        atomic_json(out/'status.json',dict(state='error',error=repr(exc)));raise
    finally:
        if sim:sim.close()


def overview(spec,entries,path,elevation=False):
    out=Image.new('RGB',(800,850),'white');d=ImageDraw.Draw(out);d.font=font(16)
    d.text((25,12),f'{spec["id"]} {spec["name"]} | '+('ground elevation' if elevation else 'top-down build plan'),fill='black')
    d.text((25,40),'Authored specification + recorded paths; not an aerial screenshot.',fill='black')
    ox,oy,scale=70,90,20
    def xy(x,z):return ox+(x+16)*scale,oy+(17-z)*scale
    colors={'stone':'#747d87','leaves':'#286534','fence':'#b1a07b','brick_block':'#a25842','planks':'#b89160',
            'water':'#478cce','log':'#715034','bookshelf':'#aa6d3b','quartz_block':'#d2cbbb','stonebrick':'#747d87','sandstone':'#d3bd89','coal_ore':'#50585c','cobblestone':'#747d87','white_concrete':'#b8c2ce'}
    for b in spec['heights']:
        x,y=xy(b['x']+.5,b['z']+.5);h=b['height']
        color=(int(220-h*12),int(237-h*8),int(210-h*15)) if elevation else '#e1e9d4'
        d.rectangle((x-10,y-10,x+10,y+10),fill=color)
        if elevation and b['x']%4==0 and b['z']%4==0:d.text((x-5,y-8),str(h),fill='black')
    if not elevation:
        for b in spec['obstacles']:
            x,y=xy(b['x']+.5,b['z']+.5);d.rectangle((x-10,y-10,x+10,y+10),fill=colors.get(b['material'],'gray'))
    for i,e in enumerate(entries):
        pts=[xy(f['pose']['x'],f['pose']['z']) for f in e['frames']]
        if len(pts)>1:d.line(pts,fill=('#216cdd','#d13278')[i],width=2)
    x,y=xy(spec['start'][0],spec['start'][2]);d.ellipse((x-5,y-5,x+5,y+5),fill='black')
    d.text((25,770),'World +X right, +Z up. One square = one block. Black = start.',fill='black')
    d.text((25,797),'Blue / pink = measured routes. Roofs and tree canopies hidden.',fill='black')
    d.text((25,824),'Elevation numbers: floor height above Y=199. Review actual geometry in GIFs.',fill='black')
    out.save(path)


def gallery(a):
    import shutil
    root=Path(a.root);panels=[];summary=[]
    # Validate all eight before creating the review output.
    for i in range(8):
        folder=root/f'env-{i:02d}'
        if json.loads((folder/'status.json').read_text())['state']!='complete':raise ValueError(f'{folder}: incomplete')
        spec=json.loads((folder/'specification.json').read_text());entries=json.loads((folder/'sequences.json').read_text())
        if len(entries)!=2 or not all(e['route_complete'] for e in entries):raise ValueError('Incomplete recordings')
        panels.append((folder,spec,entries))
    out=root/'review';out.mkdir(exist_ok=False)
    for folder,spec,entries in panels:
        for r in range(2):shutil.copy2(folder/f'route-{r}.gif',out/f'{spec["id"]}-route-{r}.gif')
        overview(spec,entries,out/f'{spec["id"]}-topdown.png')
        overview(spec,entries,out/f'{spec["id"]}-elevation.png',True)
        summary.append(dict(id=spec['id'],name=spec['name'],frames=[len(e['frames']) for e in entries],notes=spec['notes']))
    end=max(es[0]['frames'][-1]['elapsed_ticks'] for _,_,es in panels);images=[]
    for tick in range(0,end+1,32):
        canvas=Image.new('RGB',(1280,1400),'white')
        for i,(folder,spec,es) in enumerate(panels):
            fs=es[0]['frames'];j=max(0,bisect.bisect_right([f['elapsed_ticks'] for f in fs],tick)-1)
            panel=review_frame(folder,fs[j],spec['id']+' '+spec['name'],fs[j]['elapsed_ticks'])
            canvas.paste(panel,((i%2)*640,(i//2)*350))
        images.append(canvas)
    images[0].save(out/'all-eight.gif',save_all=True,append_images=images[1:],duration=400,loop=0)
    atomic_json(out/'summary.json',dict(protocol='diverse_environments_v2',environments=summary,review_status='awaiting_user_review',model_testing='not_started'))
    body='<h1>Eight diverse Minecraft environments</h1><p>Authored worlds, actual simulator recordings. Review before accepting the dataset. No model testing.</p><img src="all-eight.gif"><p>GIF: review/all-eight.gif</p>'
    for row in summary:
        eid=row['id'];body+=f'<h2>{eid}: {html.escape(row["name"])}</h2><div class="pair"><div><img src="{eid}-topdown.png"><img src="{eid}-elevation.png"></div><div>'
        for r in range(2):body+=f'<img src="{eid}-route-{r}.gif"><p>GIF: review/{eid}-route-{r}.gif</p>'
        body+='</div></div>'
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><style>body{font:18px system-ui;max-width:1400px;margin:auto;padding:24px}img{max-width:100%}.pair{display:grid;grid-template-columns:1fr 1fr;gap:20px}@media(max-width:850px){.pair{display:block}}</style>'+body)
    print(out)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['collect','gallery','plans']);p.add_argument('--root',required=True);p.add_argument('--index',type=int,default=0);p.add_argument('--max-ticks',type=int,default=12000)
    a=p.parse_args()
    if a.mode=='plans':
        out=Path(a.root);out.mkdir(parents=True,exist_ok=False)
        for i in range(8):
            s=specification(i);atomic_json(out/f'{s["id"]}.json',s);overview(s,[],out/f'{s["id"]}.png');overview(s,[],out/f'{s["id"]}-elevation.png',True)
    else:(collect if a.mode=='collect' else gallery)(a)
if __name__=='__main__':main()
