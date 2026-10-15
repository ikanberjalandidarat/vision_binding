"""Scripted intersecting Minecraft routes for a cognitive-map diagnostic.
All routes share an anchor and cardinal-heading probes. No policy is learned here.
"""
from .approach import steering


def route_plan(route):
    anchor=(0.,6.)
    # Each excursion closes at the anchor. Orders 2/3 are held out from fitting.
    loops=[[(0.,2.),(-6.,2.),(-6.,6.),anchor],
           [(6.,6.),(6.,10.),(0.,10.),anchor],
           [(-6.,6.),(-6.,10.),(0.,10.),anchor],
           [(0.,2.),(6.,2.),(6.,6.),anchor]]
    orders=[(0,1,2,3),(3,2,1,0),(1,3,0,2),(2,0,3,1)]
    if route not in range(4):raise ValueError('Route must be 0..3')
    points=[dict(x=anchor[0],z=anchor[1],yaw=None,probe='anchor-start')]
    def scan(label):
        for yaw in (0.,90.,180.,-90.):
            points.append(dict(x=anchor[0],z=anchor[1],yaw=yaw,probe=label))
    scan('initial')
    for repeat in range(2):
        for loop in orders[route]:
            path=loops[loop]
            if route>=2:path=list(reversed([anchor]+path[:-1]))+[anchor]
            for x,z in path:points.append(dict(x=x,z=z,yaw=None,probe=None))
            scan(f'loop-{repeat}-{loop}')
    return points


def control(pose,point):
    """Return reached/action. Probe turns use the same yaw convention as steering."""
    distance,turn,forward=steering(pose,(point['x'],point['z']))
    if distance<.8:
        if point['yaw'] is None:return True,None
        turn=(point['yaw']-pose['yaw']+180)%360-180
        if abs(turn)<=10:return True,None
        return False,1 if turn<0 else 2
    return False,0 if forward else 1 if turn<0 else 2
