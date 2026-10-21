"""Eight-environment Minecraft dataset review. No encoder or learned policy."""
import argparse
import html
import json
import math
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from .io import atomic_json, digest, file_hash, source_hash

NAMES = ['corner_towers', 'north_arch_gallery', 'staggered_columns', 'stepped_terraces',
         'side_alcoves', 'paired_gateways', 'alternating_spires', 'landmark_pavilions']


def specification(index):
    if index not in range(8):
        raise ValueError('Environment index must be 0..7')
    blocks = {}
    def cuboid(x0,x1,z0,z1,height,material):
        for x in range(x0,x1+1):
            for z in range(z0,z1+1):
                for y in range(200,200+height):blocks[x,y,z]=material
    def arch(x,z,width=3):
        cuboid(x,x,z,z,4,'red_sandstone')
        cuboid(x+width,x+width,z,z,4,'red_sandstone')
        for xx in range(x,x+width+1):blocks[xx,204,z]='red_sandstone'
    if index == 0:
        for x,z,h,m in [(-8,0,3,'red_wool'),(8,0,5,'blue_wool'),(-8,16,4,'yellow_wool'),(8,16,6,'green_wool')]:cuboid(x,x,z,z,h,m)
    elif index == 1:
        for x in (-8,-2,4):arch(x,16)
        cuboid(-8,-8,4,10,2,'stonebrick')
    elif index == 2:
        for x,z,h in [(-8,0,2),(-8,5,5),(-8,11,3),(8,3,4),(8,8,2),(8,16,5)]:cuboid(x,x,z,z,h,'quartz_block')
    elif index == 3:
        for x in range(-8,9):cuboid(x,x,16,17,1+(x+8)//4,'sandstone')
        for z in range(1,14,3):cuboid(-9,-8,z,z+1,2,'brick_block')
    elif index == 4:
        for x in (-9,8):
            for z in (1,6,11):
                cuboid(x,x+1,z,z,4,'brick_block');cuboid(x,x+1,z+3,z+3,4,'brick_block')
        cuboid(-2,2,17,17,3,'yellow_wool')
    elif index == 5:
        arch(-7,0,4);arch(3,16,4)
        cuboid(-9,-8,7,8,5,'blue_wool');cuboid(8,9,7,8,3,'red_wool')
    elif index == 6:
        for z in (0,4,8,12,16):
            cuboid(-8,-8,z,z,2+(z//4)%3,'yellow_wool')
            cuboid(8,8,z,z,5-(z//4)%3,'blue_wool')
    else:
        for x,z in [(-8,0),(6,16)]:
            for xx in (x,x+2):cuboid(xx,xx,z,z,4,'log')
            for xx in range(x,x+3):blocks[xx,204,z]='planks'
        cuboid(-9,-8,9,11,3,'green_wool');cuboid(8,9,5,6,5,'red_wool')
    assert all(not(-7<=x<=7 and 1<=z<=15) for x,y,z in blocks)
    commands = ['/gamerule sendCommandFeedback false','/gamemode creative',
      '/gamerule doDaylightCycle false','/gamerule doWeatherCycle false',
      '/gamerule doMobSpawning false','/time set 6000','/weather clear',
      '/fill -10 200 -2 10 210 18 minecraft:air',
      '/fill -10 199 -2 10 199 18 minecraft:stone',
      '/fill -10 200 -2 -10 207 18 minecraft:white_concrete',
      '/fill 10 200 -2 10 207 18 minecraft:white_concrete',
      '/fill -10 200 -2 10 207 -2 minecraft:white_concrete',
      '/fill -10 200 18 10 207 18 minecraft:white_concrete',
      '/fill -10 208 -2 10 208 18 minecraft:sea_lantern']
    commands += [f'/setblock {x} {y} {z} minecraft:{m}' for (x,y,z),m in sorted(blocks.items())]
    commands += ['/tp @p 0.5 200 0.5 0 0']
    return dict(id=f'env-{index:02d}',name=NAMES[index],commands=commands,
                landmarks=[dict(x=x,y=y,z=z,material=m) for (x,y,z),m in sorted(blocks.items())])


def route(reverse=False):
    points=[(x,z) for i,z in enumerate((2.,5.,8.,11.,14.)) for x in ((-6.,6.) if i%2==0 else (6.,-6.))]
    return list(reversed(points)) if reverse else points


def review_frame(root, record, title, tick):
    with Image.open(root/record['frame']) as im:
        out=Image.new('RGB',(640,350),'white');out.paste(im.convert('RGB'),(0,35))
    d=ImageDraw.Draw(out);d.text((10,10),title,fill='black')
    pose=record['pose'];d.text((10,320),f't={tick/20:.1f}s  X={pose["x"]:.2f} Z={pose["z"]:.2f} | SCRIPTED recording, 4x playback',fill='black')
    x=462+(pose['x']+10)/20*166;y=300-(pose['z']+2)/20*250
    d.rectangle((462,50,628,300),outline='gray');d.ellipse((x-3,y-3,x+3,y+3),fill='red')
    d.line((x,y,x-10*math.sin(math.radians(pose['yaw'])),y-10*math.cos(math.radians(pose['yaw']))),fill='black',width=2)
    return out


def collect(a):
    from minestudio.simulator import MinecraftSim
    from minestudio.simulator.callbacks import CommandsCallback
    from .backends.minestudio_smoke import verify_pose,json_value
    from .approach import steering
    spec=specification(a.index);out=Path(a.root)/spec['id'];out.mkdir(parents=True,exist_ok=False)
    atomic_json(out/'specification.json',spec)
    atomic_json(out/'manifest.json',dict(protocol='eight_environments_v1',seed=731,source_hash=source_hash(),
        spec_sha256=digest(spec),rgb=[448,280],tick_ms=50,capture_every_ticks=8,action_chunk_ticks=2,
        routes=[route(False),route(True)],settle_ticks=200,max_route_ticks=a.max_ticks,
        is_minecraft=True,review_status='awaiting_user_review',model_used=False,
        scope='Eight landmark layouts; shared open floor, walls, lighting and traversable region. Not eight maze topologies.'))
    atomic_json(out/'status.json',dict(state='running'));sim=None;entries=[]
    try:
        sim=MinecraftSim(action_type='env',obs_size=(448,280),render_size=(448,280),seed=731)
        obs,info=sim.reset()
        def step(action):
            nonlocal obs,info
            obs,_,done,truncated,info=sim.step(action)
            if done or truncated or ('error' in info):raise RuntimeError('Simulator terminated or returned error')
        obs,info=CommandsCallback(commands=['/gamemode creative','/forceload add -10 -2 10 18','/tp @p 0.5 200 0.5 0 0']).after_reset(sim,obs,info)
        for _ in range(200):step(sim.noop_action())
        obs,info=CommandsCallback(commands=spec['commands']).after_reset(sim,obs,info)
        for _ in range(200):step(sim.noop_action())
        verify_pose(info,[.5,200,.5],0)
        for r in range(2):
            obs,info=CommandsCallback(commands=['/tp @p 0.5 200 0.5 0 0']).after_reset(sim,obs,info)
            for _ in range(200):step(sim.noop_action())
            verify_pose(info,[.5,200,.5],0)
            folder=out/f'route-{r}';folder.mkdir();frames=[];actions=[];j=0;points=route(bool(r))
            # Initial full heading scan, then the same route specification in every world.
            for tick in range(0,a.max_ticks+1,2):
                pose=json_value(info['player_pos'])
                if not all(math.isfinite(float(pose[k])) for k in ('x','y','z','yaw')):raise ValueError('Nonfinite pose')
                if not(-9<pose['x']<9 and -1<pose['z']<17 and 199.5<pose['y']<200.5):raise ValueError('Pose outside arena')
                if tick%8==0:
                    pixels=np.asarray(obs['image'])
                    if pixels.shape!=(280,448,3) or pixels.std()<1:raise ValueError('Invalid RGB')
                    path=folder/f'frame-{tick:05d}.png';Image.fromarray(pixels).save(path)
                    frames.append(dict(frame=str(path.relative_to(out)),elapsed_ticks=tick,pose=pose,actions_since_previous=actions,sha256=file_hash(path)));actions=[]
                if tick>=72:
                    while j<len(points) and steering(pose,points[j])[0]<.8:j+=1
                    if j==len(points):break
                    _,turn,forward=steering(pose,points[j]);action=0 if forward else 1 if turn<0 else 2
                else:action=2
                if tick==a.max_ticks:break
                command=sim.noop_action();command['forward']=int(action==0);command['camera']=np.array([0.,-5. if action==1 else 5. if action==2 else 0.],np.float32)
                for _ in range(2):step(command)
                actions.append(action)
            entry=dict(episode=f'{spec["id"]}-route{r}',is_minecraft=True,route=r,frames=frames,
                       route_complete=j==len(points),waypoints_reached=j,waypoints_total=len(points))
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


def layout_map(spec, entries, path):
    """Allocentric projection of build specification, NOT an aerial render."""
    canvas=Image.new('RGB',(760,850),'white');d=ImageDraw.Draw(canvas)
    d.text((24,15),f'{spec["id"]}: {spec["name"]}',fill='black')
    d.text((24,38),'TOP-DOWN BUILD SPECIFICATION | +X right, +Z up | ceiling hidden',fill='black')
    colors={'red_wool':'#d34747','blue_wool':'#4477ce','yellow_wool':'#d7b340','green_wool':'#4a9855',
            'red_sandstone':'#bd754d','stonebrick':'#7c8790','quartz_block':'#d4d0bd','sandstone':'#d8bf87',
            'brick_block':'#9b5f55','log':'#76523b','planks':'#be9769'}
    cell=28;ox=86;oy=94
    def xy(x,z):return ox+(x+10.5)*cell,oy+(18.5-z)*cell
    d.rectangle((ox,oy,ox+21*cell,oy+21*cell),fill='#eceff2',outline='black',width=3)
    for x in range(-10,11):
        for z in range(-2,19):
            px,py=xy(x,z)
            if x in (-10,10) or z in (-2,18):d.rectangle((px-14,py-14,px+14,py+14),fill='#697580')
    top={}
    for b in spec['landmarks']:
        k=b['x'],b['z']
        if k not in top or b['y']>top[k]['y']:top[k]=b
    for (x,z),b in top.items():
        px,py=xy(x,z);d.rectangle((px-14,py-14,px+14,py+14),fill=colors.get(b['material'],'gray'),outline='white')
        d.text((px-5,py-6),str(b['y']-199),fill='black')
    for r,e in enumerate(entries):
        points=[xy(f['pose']['x'],f['pose']['z']) for f in e['frames']]
        if len(points)>1:d.line(points,fill=('#1a80be','#c33f96')[r],width=2)
    x,y=xy(.5,.5);d.ellipse((x-5,y-5,x+5,y+5),fill='black')
    for x in (-10,-5,0,5,10):
        px,_=xy(x,0);d.text((px-8,oy+21*cell+10),str(x),fill='black')
    for z in (-2,3,8,13,18):
        _,py=xy(0,z);d.text((ox-30,py-6),str(z),fill='black')
    d.text((ox,oy+21*cell+35),'X (blocks); Z on left. Numbers on landmarks = height in blocks.',fill='black')
    d.text((24,755),'Black dot: common start | Blue: recorded route 0 | Magenta: recorded route 1',fill='black')
    d.text((24,780),'All rooms: stone floor, white walls, sea-lantern ceiling at Y=208.',fill='black')
    d.text((24,805),'Planned geometry plus measured paths; inspect RGB to verify the actual build.',fill='black')
    canvas.save(path)


def gallery(a):
    root=Path(a.root);out=root/'review';out.mkdir(exist_ok=False)
    import shutil
    panels=[];summary=[]
    for i in range(8):
        folder=root/f'env-{i:02d}'
        if json.loads((folder/'status.json').read_text())['state']!='complete':raise ValueError(f'{folder}: incomplete')
        rows=json.loads((folder/'sequences.json').read_text())
        for r in range(2):shutil.copy2(folder/f'route-{r}.gif',out/f'env-{i:02d}-route-{r}.gif')
        layout_map(json.loads((folder/'specification.json').read_text()),rows,out/f'env-{i:02d}-topdown.png')
        panels.append((folder,rows[0]['frames']))
        summary.append(dict(id=f'env-{i:02d}',name=NAMES[i],frames=[len(e['frames']) for e in rows],complete=all(e['route_complete'] for e in rows)))
    # Shared simulation time, four times faster; completed panels hold their final frame.
    import bisect
    end=max(fs[-1]['elapsed_ticks'] for _,fs in panels);montage=[]
    for tick in range(0,end+1,32):
        canvas=Image.new('RGB',(1280,1400),'white')
        for i,(folder,fs) in enumerate(panels):
            idx=max(0,bisect.bisect_right([f['elapsed_ticks'] for f in fs],tick)-1)
            panel=review_frame(folder,fs[idx],f'env-{i:02d} {NAMES[i]} | route 0',fs[idx]['elapsed_ticks'])
            canvas.paste(panel,((i%2)*640,(i//2)*350))
        montage.append(canvas)
    montage[0].save(out/'all-eight.gif',save_all=True,append_images=montage[1:],duration=400,loop=0)
    atomic_json(out/'summary.json',dict(environments=summary,review_status='awaiting_user_review',model_testing='not_started'))
    body='<h1>Eight Minecraft environments — review before model testing</h1><p>Shared scale, lighting, start pose, camera, recording cadence and two scripted routes. New landmark layouts on an open floor. Automatic completion does not certify visual correctness.</p><img src="all-eight.gif"><p>GIF: review/all-eight.gif</p>'
    for row in summary:
        body+=f'<h2>{row["id"]}: {html.escape(row["name"])}</h2><div class=pair><figure><img src="{row["id"]}-topdown.png"><figcaption>Allocentric build specification + recorded paths; not an aerial render.</figcaption></figure><div>'
        for r in range(2):
            name=f'{row["id"]}-route-{r}.gif';body+=f'<img src="{name}"><p>GIF: review/{name}</p>'
        body+='</div></div>'
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><style>body{font:18px system-ui;max-width:1280px;margin:30px auto}img{max-width:100%}.pair{display:grid;grid-template-columns:1fr 1fr;gap:20px}figure{margin:0}@media(max-width:800px){.pair{grid-template-columns:1fr}}</style>'+body)
    print(out)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['collect','gallery']);p.add_argument('--root',required=True);p.add_argument('--index',type=int,default=0);p.add_argument('--max-ticks',type=int,default=5000)
    a=p.parse_args();(collect if a.mode=='collect' else gallery)(a)
if __name__=='__main__':main()
