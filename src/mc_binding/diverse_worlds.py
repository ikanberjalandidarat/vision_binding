"""Deterministic authored Minecraft worlds. Plans are not rendered evidence."""
from collections import deque
import math
import random

NAMES=['indoor_room','outdoor_courtyard','indoor_maze','outdoor_maze','forest_and_pond','mine_tunnels','hills','village']


def specification(index):
    if index not in range(8):raise ValueError('Index must be 0..7')
    heights={(x,z):0 for x in range(-16,17) for z in range(-16,17)}
    if index==6:
        heights={p:max(0,8-(abs(p[0]-3)+abs(p[1]-2))//3) for p in heights}
    cmds=[];obstacles={};cover=[]
    def fill(x0,x1,z0,z1,y0,y1,material,blocking=True):
        cmds.append(f'/fill {x0} {y0} {z0} {x1} {y1} {z1} minecraft:{material}')
        if blocking:
            for x in range(x0,x1+1):
                for z in range(z0,z1+1):obstacles[x,z]=material
    # Each fill remains below Minecraft's block-count limit.
    commands=['/gamerule sendCommandFeedback false','/gamemode creative','/gamerule doDaylightCycle false',
              '/gamerule doWeatherCycle false','/gamerule randomTickSpeed 0','/gamerule doMobSpawning false','/time set 6000','/weather clear',
              '/fill -16 194 -16 16 221 16 minecraft:air']
    ground='planks' if index==0 else 'stone' if index in (2,5) else 'grass_block'
    # Use the solid grass block for outdoor ground. minecraft:grass is a
    # non-solid plant in the simulator's Minecraft version.
    commands.append(f'/fill -16 194 -16 16 199 16 minecraft:{ground}')
    if index==6:
        # Add each height layer as contiguous row segments, preserving the
        # authored stepped hill while using far fewer simulator commands.
        for level in range(1,9):
            for z in range(-16,17):
                run_start=None
                for x in range(-16,18):
                    raised=x<=16 and heights[x,z]>=level
                    if raised and run_start is None:run_start=x
                    if not raised and run_start is not None:
                        commands.append(f'/fill {run_start} {199+level} {z} {x-1} {199+level} {z} minecraft:{ground}')
                        run_start=None
    if index in (0,2):
        for box in [(-16,16,-16,-16),(-16,16,16,16),(-16,-16,-16,16),(16,16,-16,16)]:fill(*box,200,207,'white_concrete')
        fill(-16,16,-16,16,208,208,'sea_lantern',False);cover.append('room ceiling at Y=208')
    else:
        # Visible safety fence, leaving open sky in outdoor scenes.
        for (x,z),h in heights.items():
            if abs(x)==16 or abs(z)==16:fill(x,x,z,z,200+h,201+h,'fence')
    if index==0:
        for x,z in [(-10,-8),(8,8),(-9,9)]:
            fill(x,x+3,z,z+2,200,201,'bookshelf')
        fill(8,10,-10,-8,200,203,'quartz_block')
    if index==1:
        for x,z in [(-10,-10),(9,-8),(-8,10),(10,10)]:fill(x,x+1,z,z+1,200,203,'stonebrick')
        fill(-3,3,6,7,200,200,'sandstone')
    if index in (2,3,5):
        # Perfect maze of 5x5 rooms; four-block center spacing, 3-wide passages.
        rng=random.Random(840+index);cells={(0,0)};stack=[(0,0)];edges=[]
        while stack:
            x,z=stack[-1];ns=[(x+dx,z+dz) for dx,dz in [(1,0),(-1,0),(0,1),(0,-1)] if 0<=x+dx<5 and 0<=z+dz<5 and (x+dx,z+dz) not in cells]
            if not ns:stack.pop();continue
            v=rng.choice(ns);edges.append(((x,z),v));cells.add(v);stack.append(v)
        open_cells=set()
        centers=lambda c:(c[0]*5-10,c[1]*5-10)
        for c in cells:
            x,z=centers(c)
            for dx in range(-1,2):
                for dz in range(-1,2):open_cells.add((x+dx,z+dz))
        for a,b in edges:
            x,z=centers(a);xx,zz=centers(b)
            for px in range(min(x,xx)-1,max(x,xx)+2):
                for pz in range(min(z,zz)-1,max(z,zz)+2):open_cells.add((px,pz))
        for x in range(-13,14):
            for z in range(-13,14):
                if (x,z) not in open_cells:fill(x,x,z,z,200,204,'stone' if index==5 else 'leaves' if index==3 else 'brick_block')
        if index==5:
            fill(-16,16,-16,16,205,207,'stone',False);cover.append('mine roof at Y=205; overview hides roof')
            for x,z in sorted(open_cells):
                if x%5==0 and z%5==0:
                    fill(x,x,z,z,205,205,'glowstone',False)
            for x,z in [(-12,-12),(12,12),(-12,12)]:fill(x,x,z,z,201,202,'coal_ore')
        start=(-10,-10)
    else:start=(-12,-12)
    if index==4:
        for x,z in [(-10,-6),(-5,-9),(4,-10),(10,-5),(-9,5),(-4,9),(4,8),(10,11),(8,2)]:
            fill(x,x,z,z,200,204,'log')
            fill(x-2,x+2,z-2,z+2,204,206,'leaves',False)
        fill(-2,2,-2,2,199,199,'water');cover.append('authored forest/pond, not vanilla world-generation')
    if index==6:
        for x,z in [(-9,9),(10,-9)]:
            h=heights[x,z];fill(x,x,z,z,200+h,203+h,'log');fill(x-1,x+1,z-1,z+1,203+h,204+h,'leaves',False)
    if index==7:
        for x,z in [(-11,-9),(4,-9),(-11,4),(4,4)]:
            fill(x,x+5,z,z+5,200,204,'planks')
            # Hollow houses and stepped roofs; routes remain on streets.
            fill(x+1,x+4,z+1,z+4,200,203,'air',False)
            fill(x-1,x+6,z-1,z+6,205,205,'brick_block',False)
            fill(x,x+5,z,z+5,206,206,'brick_block',False)
            fill(x+1,x+4,z+1,z+4,207,207,'brick_block',False)
            fill(x+2,x+3,z,z,201,202,'glass',False)
        fill(-1,1,-1,1,200,201,'cobblestone')
        for z in range(-14,15):commands.append(f'/fill -3 199 {z} -2 199 {z} minecraft:gravel')
    commands+=cmds
    walkable={p for p in heights if abs(p[0])<15 and abs(p[1])<15 and p not in obstacles}
    def bfs(a,b=None):
        q=deque([a]);prev={a:None}
        while q:
            p=q.popleft()
            if p==b:break
            for dx,dz in [(1,0),(0,1),(-1,0),(0,-1)]:
                v=(p[0]+dx,p[1]+dz)
                if v in walkable and v not in prev and abs(heights[v]-heights[p])<=1:prev[v]=p;q.append(v)
        if b is None:return prev
        if b not in prev:raise ValueError('Unreachable waypoint')
        path=[];p=b
        while p!=a:path.append(p);p=prev[p]
        return list(reversed(path))
    reached=bfs(start);anchors=[]
    for target in [(-10,10),(10,10),(10,-10),(0,0),(-10,0),(0,10)]:
        anchors.append(min(reached,key=lambda p:(abs(p[0]-target[0])+abs(p[1]-target[1]),p)))
    routes=[]
    for order in (anchors,list(reversed(anchors))):
        points=[start];current=start
        for goal in order+[start]:points+=bfs(current,goal);current=goal
        routes.append([[x+.5,200+heights[x,z],z+.5] for x,z in points])
    spawn=[start[0]+.5,200+heights[start],start[1]+.5]
    commands.append(f'/tp @p {spawn[0]} {spawn[1]} {spawn[2]} 0 0')
    return dict(id=f'env-{index:02d}',name=NAMES[index],commands=commands,start=spawn,routes=routes,
                heights=[dict(x=x,z=z,height=h) for (x,z),h in heights.items()],
                obstacles=[dict(x=x,z=z,material=m) for (x,z),m in obstacles.items()],
                notes=cover,authored=True,seed=731,bounds=[-16,16,-16,16])
