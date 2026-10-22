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
