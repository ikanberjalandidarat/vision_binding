"""Dense Minecraft revisit collection. Plans and synthetic tests are not rendering evidence."""
import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from .io import atomic_json, digest, file_hash, source_hash

PROTOCOL = 'within_world_revisit_pilot_v1'


def specification(route):
    if route not in range(6):
        raise ValueError('route must be 0..5')
    sites = [dict(id=i, x=x+.5, y=200., z=z+.5)
             for i, (x, z) in enumerate(( (x,z) for z in (-12,-4,4,12) for x in (-12,-4,4,12)))]
    rng = random.Random(9100+route)
    visits = [(i,h) for i in range(16) for h in (0,90,180,270)]
    rng.shuffle(visits)
    targets=[]
    for i,h in visits:
        s=sites[i]; angle=math.radians(h)
        # Approach along the requested heading; all interior terrain is clear.
        approach=[s['x']+4*math.sin(angle), 200., s['z']-4*math.cos(angle)]
        targets.append(dict(site=i, heading=h, approach=approach,
                            via=[rng.uniform(-18,18),200.,rng.uniform(-18,18)]))
    commands=['/gamemode creative','/gamerule doDaylightCycle false',
              '/gamerule doWeatherCycle false','/gamerule doMobSpawning false',
              '/gamerule randomTickSpeed 0','/time set 6000','/weather clear']
    # Every fill is below 32768 blocks. Clear the authored airspace in strips.
    for z in range(-24,25,7):
        commands.append(f'/fill -24 200 {z} 24 214 {min(z+6,24)} minecraft:air')
    commands.append('/fill -24 198 -24 24 199 24 minecraft:grass_block')
    for s in sites:
        x,z=int(math.floor(s['x'])),int(math.floor(s['z']))
        commands.append(f'/fill {x-1} 199 {z-1} {x+1} 199 {z+1} minecraft:sandstone')
        commands.append(f'/setblock {x} 199 {z} minecraft:stone_bricks')
    for x,z,block in [(-23,-23,'bricks'),(23,-23,'quartz_block'),(23,23,'oak_planks'),(-23,23,'stone_bricks')]:
        commands.append(f'/fill {x} 200 {z} {x} 205 {z} minecraft:{block}')
    commands.append('/tp @p 0.5 200 0.5 0 0')
    return dict(protocol=PROTOCOL,route=route,split=('train','validation','test')[route//2],
                sites=sites,targets=targets,commands=commands,start=[.5,200.,.5],
                seed=731,route_seed=9100+route,world_bounds=[-24,24],
                scope='Authored open field with repeated floor motifs and distant perimeter landmarks; ambiguity requires visual review.')


def angle_error(a,b):
    return (a-b+180)%360-180


def accepted(p,s,h):
    return (math.hypot(p['x']-s['x'],p['z']-s['z'])<=.75
            and abs(p['y']-s['y'])<.6 and abs(angle_error(p['yaw'],h))<=10)


def collect(a):
    from minestudio.simulator import MinecraftSim
    from minestudio.simulator.callbacks import CommandsCallback
    from .backends.minestudio_smoke import verify_pose, json_value
    spec=specification(a.route);out=Path(a.root)/f'route-{a.route}'
    out.mkdir(parents=True,exist_ok=False)
    atomic_json(out/'specification.json',spec)
    atomic_json(out/'manifest.json',dict(protocol=PROTOCOL,is_minecraft=True,source_hash=source_hash(),
        specification_sha256=digest(spec),capture_every_ticks=2,model_every_ticks=8,
        clip_frames=16,clip_span_ticks=120,rgb=[448,280],tick_seconds=.05,
        split=spec['split'],route=a.route,teleports_during_recording=0,model_run=False))
    atomic_json(out/'status.json',dict(state='running'));sim=None
    frames=[];visits=[];tick=0;pending=[]
    try:
        sim=MinecraftSim(action_type='env',obs_size=(448,280),render_size=(448,280),seed=731)
        obs,info=sim.reset()
        def step(action):
            nonlocal obs,info
            obs,_,done,truncated,info=sim.step(action)
            if done or truncated or 'error' in info:
                raise RuntimeError('Simulator terminated or returned error')
        obs,info=CommandsCallback(commands=['/gamemode creative','/tp @p 0.5 200 0.5 0 0']).after_reset(sim,obs,info)
        for _ in range(200):step(sim.noop_action())
        obs,info=CommandsCallback(commands=spec['commands']).after_reset(sim,obs,info)
        for _ in range(200):step(sim.noop_action())
        Image.fromarray(np.asarray(obs['image'])).save(out/'build-check.png')
        verify_pose(info,spec['start'],0)
        (out/'frames').mkdir()
        def record():
            nonlocal pending
            p=json_value(info['player_pos'])
            if not all(math.isfinite(float(p[k])) for k in ('x','y','z','yaw')):
                raise ValueError('Nonfinite pose')
            if not(-22<p['x']<22 and -22<p['z']<22 and 199.4<p['y']<200.6):
                raise ValueError('Pose left the flat walking area')
            pixels=np.asarray(obs['image'])
            if pixels.shape!=(280,448,3) or pixels.std()<1:raise ValueError('Invalid RGB')
            path=out/'frames'/f'{tick:06d}.png';Image.fromarray(pixels).save(path)
            frames.append(dict(frame=str(path.relative_to(out)),elapsed_ticks=tick,pose=p,
                               sha256=file_hash(path),actions_since_previous=pending));pending=[]
        def advance(forward=0,turn=0.):
            nonlocal tick
            cmd=sim.noop_action();cmd['forward']=forward;cmd['camera']=np.array([0.,turn],np.float32)
            for _ in range(2):step(cmd)
            tick+=2;pending.append(dict(ticks=2,forward=forward,camera=[0.,turn]))
            record()
            if tick>a.max_ticks:raise RuntimeError('Route tick budget exceeded')
        def walk(goal):
            begin=tick;best=float('inf');last_progress=tick
            while True:
                p=info['player_pos'];dx,dz=goal[0]-p['x'],goal[2]-p['z'];dist=math.hypot(dx,dz)
                if dist<.35:return
                if dist<best-.1:best=dist;last_progress=tick
                if tick-last_progress>400 or tick-begin>1400:raise RuntimeError('Stuck or over-budget walking segment')
                turn=angle_error(math.degrees(math.atan2(-dx,dz)),p['yaw'])
                advance(int(abs(turn)<12),float(np.clip(turn/2,-5,5)))
        record()
        # Exclude all setup teleports from recorded clips. Warm up 6 s in place.
        for _ in range(60):advance()
        for v,target in enumerate(spec['targets']):
            site=spec['sites'][target['site']]
            walk(target['via']);walk(target['approach']);walk([site['x'],200.,site['z']])
            begin=tick
            while abs(angle_error(target['heading'],info['player_pos']['yaw']))>3:
                if tick-begin>200:raise RuntimeError('Heading did not settle')
                advance(turn=float(np.clip(angle_error(target['heading'],info['player_pos']['yaw'])/2,-5,5)))
            while tick%8:advance()
            p=frames[-1]['pose']
            if not accepted(p,site,target['heading']):raise RuntimeError('Visit pose acceptance failed')
            visits.append(dict(visit=v,site=target['site'],target_heading=target['heading'],
                endpoint_tick=tick,pose=p,frame=frames[-1]['frame'],
                clip_ticks=list(range(tick-120,tick+1,8))))
            atomic_json(out/'visits.json',visits)
            atomic_json(out/'progress.json',dict(state='running',visits=len(visits),frames=len(frames),tick=tick))
            print(f'route {a.route}: {len(visits)}/64 visits',flush=True)
        atomic_json(out/'frames.json',frames)
        atomic_json(out/'status.json',dict(state='complete',visits=len(visits),frames=len(frames),review_status='awaiting_user_review'))
    except BaseException as exc:
        atomic_json(out/'frames.json',frames)
        atomic_json(out/'status.json',dict(state='error',error=repr(exc),visits=len(visits)));raise
    finally:
        if sim:sim.close()


def audit(a):
    root=Path(a.root);out=root/'review';out.mkdir(exist_ok=False)
    summary=[];html=['<!doctype html><meta charset="utf-8"><style>body{font:18px system-ui;max-width:1100px;margin:auto}img{max-width:100%}</style><h1>Dense Minecraft revisits — review required</h1><p>Actual recordings. Maps are measured paths on an authored plan, not aerial screenshots. GIFs: 4× playback. Floor motifs are candidates for visual ambiguity, not validated controls.</p>']
    endpoint_images={};route_visits={}
    for r in range(6):
        folder=root/f'route-{r}';spec=specification(r)
        if json.loads((folder/'status.json').read_text())['state']!='complete':raise ValueError(f'route {r} incomplete')
        if digest(json.loads((folder/'specification.json').read_text()))!=digest(spec):raise ValueError('Specification mismatch')
        fs=json.loads((folder/'frames.json').read_text());vs=json.loads((folder/'visits.json').read_text())
        if len(vs)!=64 or {(v['site'],v['target_heading']) for v in vs}!={(s,h) for s in range(16) for h in (0,90,180,270)}:raise ValueError('Missing site/headings')
        ticks={f['elapsed_ticks']:f for f in fs}
        if len(ticks)!=len(fs) or sorted(ticks)!=list(range(0,fs[-1]['elapsed_ticks']+1,2)):raise ValueError('Capture discontinuity')
        for f in fs:
            if file_hash(folder/f['frame'])!=f['sha256']:raise ValueError('Frame hash mismatch')
            with Image.open(folder/f['frame']) as im:
                if im.mode!='RGB' or im.size!=(448,280):raise ValueError('Invalid image dimensions/mode')
            p=f['pose']
            if not all(math.isfinite(float(p[k])) for k in ('x','y','z','yaw')):raise ValueError('Nonfinite recorded pose')
            if not(-22<p['x']<22 and -22<p['z']<22 and 199.4<p['y']<200.6):raise ValueError('Recorded pose out of bounds')
        max_step=max(math.hypot(b['pose']['x']-c['pose']['x'],b['pose']['z']-c['pose']['z']) for c,b in zip(fs,fs[1:]))
        if max_step>1:raise ValueError('Unexpected inter-frame jump')
        for v in vs:
            if not accepted(v['pose'],spec['sites'][v['site']],v['target_heading']):raise ValueError('Visit outside tolerances')
            if any(t not in ticks for t in v['clip_ticks']):raise ValueError('Missing model context')
            if v['pose']!=ticks[v['endpoint_tick']]['pose']:raise ValueError('Endpoint metadata mismatch')
        route_visits[r]=vs
        canvas=Image.new('RGB',(800,850),'white');d=ImageDraw.Draw(canvas)
        xy=lambda p:(400+15*p['x'],400-15*p['z'])
        d.line([xy(f['pose']) for f in fs],fill='steelblue',width=2)
        for s in spec['sites']:
            x,y=xy(s);d.ellipse((x-5,y-5,x+5,y+5),fill='red');d.text((x+6,y),str(s['id']),fill='black')
        d.text((20,800),f'Route {r}: {spec["split"]}; measured X/Z path; 15 px = 1 block; +Z up',fill='black')
        canvas.save(out/f'route-{r}-map.png')
        # Bounded-memory review: split long routes into <=600-frame GIF segments.
        sampled=fs[::4]
        for chunk in range(math.ceil(len(sampled)/600)):
            ims=[]
            for f in sampled[chunk*600:(chunk+1)*600]:
                with Image.open(folder/f['frame']) as im:ims.append(im.convert('RGB').resize((336,210)))
            ims[0].save(out/f'route-{r}-{chunk}.gif',save_all=True,append_images=ims[1:],duration=100,loop=0)
        sheet=Image.new('RGB',(4*224,16*156),'white');draw=ImageDraw.Draw(sheet)
        for v in vs:
            col=v['target_heading']//90;row=v['site']
            with Image.open(folder/v['frame']) as im:
                rgb=im.convert('RGB');sheet.paste(rgb.resize((224,140)),(col*224,row*156))
                # HUD-masked, low-resolution endpoint pixel control (not semantic equivalence).
                arr=np.asarray(rgb.resize((32,20)),dtype=float)[:17]/255
                endpoint_images[r,v['site'],v['target_heading']]=arr
            draw.text((col*224,row*156+140),f'site {row}, yaw {v["target_heading"]}',fill='black')
        sheet.save(out/f'route-{r}-endpoints.png')
        summary.append(dict(route=r,split=spec['split'],visits=64,frames=len(fs),max_step=max_step))
        html.append(f'<h2>Route {r} — {spec["split"]}</h2><img src="route-{r}-map.png"><details><summary>All site/heading endpoints</summary><img src="route-{r}-endpoints.png"></details>')
        html.extend(f'<img src="route-{r}-{c}.gif">' for c in range(math.ceil(len(sampled)/600)))
    pairs=[]
    for split,(a1,b1) in dict(train=(0,1),validation=(2,3),test=(4,5)).items():
        near=far=opposite=0
        for u in route_visits[a1]:
            for v in route_visits[b1]:
                dist=math.hypot(u['pose']['x']-v['pose']['x'],u['pose']['z']-v['pose']['z'])
                near+=dist<=2;far+=dist>=6
                opposite+=dist<=2 and abs(angle_error(u['pose']['yaw'],v['pose']['yaw']))>=90
        pairs.append(dict(split=split,near_pairs=near,far_pairs=far,near_opposite_heading_pairs=opposite))
    # Inspect ambiguity only on TRAIN route endpoints; do not select controls using test features.
    ambiguity=[];sites=specification(0)['sites']
    for i in range(16):
        for j in range(i+1,16):
            dist=math.hypot(sites[i]['x']-sites[j]['x'],sites[i]['z']-sites[j]['z'])
            if dist>=8:
                mse=np.mean([np.mean((endpoint_images[0,i,h]-endpoint_images[1,j,h])**2) for h in (0,90,180,270)])
                ambiguity.append(dict(sites=[i,j],distance=dist,endpoint_rgb_mse=float(mse)))
    atomic_json(out/'summary.json',dict(state='coverage_checked',protocol=PROTOCOL,routes=summary,
        visits=384,raw_hashes_verified=True,pair_coverage=pairs,
        ambiguity_candidates=sorted(ambiguity,key=lambda p:p['endpoint_rgb_mse']),
        ambiguity_gate='Manual image review required; pixel MSE is not proof of perceptual similarity',
        model_testing='not_started',review_status='awaiting_user_review'))
    (out/'index.html').write_text('\n'.join(html))
    print(out)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['collect','audit','plan'])
    p.add_argument('--root',required=True);p.add_argument('--route',type=int,default=0);p.add_argument('--max-ticks',type=int,default=100000)
    a=p.parse_args()
    if a.mode=='plan':
        out=Path(a.root);out.mkdir(parents=True,exist_ok=False)
        for r in range(6):atomic_json(out/f'route-{r}.json',specification(r))
    else:(collect if a.mode=='collect' else audit)(a)

if __name__=='__main__':main()
