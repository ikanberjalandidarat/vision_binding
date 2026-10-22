"""Specification/route tests; no Minecraft rendering or success claim."""
from mc_binding.diverse_worlds import specification


def test_eight_categories_and_walkable_routes():
    specs=[specification(i) for i in range(8)]
    assert len({s['name'] for s in specs})==8
    for s in specs:
        blocked={(b['x'],b['z']) for b in s['obstacles']}
        heights={(b['x'],b['z']):b['height'] for b in s['heights']}
        for route in s['routes']:
            assert route[0]==s['start']==route[-1]
            for x,y,z in route:
                cell=(int(x-.5),int(z-.5))
                assert cell not in blocked
                assert y==200+heights[cell]
            for a,b in zip(route,route[1:]):
                assert abs(a[0]-b[0])+abs(a[2]-b[2])==1
                assert abs(a[1]-b[1])<=1
    assert max(p[1] for r in specs[6]['routes'] for p in r)-min(p[1] for r in specs[6]['routes'] for p in r)>=5


def test_fill_volumes_within_limit():
    for i in range(8):
        for cmd in specification(i)['commands']:
            if cmd.startswith('/fill '):
                v=list(map(int,cmd.split()[1:7]))
                assert (abs(v[3]-v[0])+1)*(abs(v[4]-v[1])+1)*(abs(v[5]-v[2])+1)<=32768


def test_constructed_floor_covers_every_cell_at_planned_height():
    for i in range(8):
        spec=specification(i)
        material='planks' if i==0 else 'stone' if i in (2,5) else 'grass'
        top={}
        for cmd in spec['commands']:
            if not cmd.startswith('/fill ') or not cmd.endswith(f'minecraft:{material}'):
                if top:break
                continue
            x0,y0,z0,x1,y1,z1=map(int,cmd.split()[1:7])
            for x in range(x0,x1+1):
                for z in range(z0,z1+1):top[x,z]=max(y1,top.get((x,z),-1))
        assert len(top)==33*33
        for cell in spec['heights']:
            assert top[cell['x'],cell['z']]==199+cell['height']
