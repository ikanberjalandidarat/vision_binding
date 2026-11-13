"""NON-MINECRAFT plan/pose fixtures; actual rendering remains an Oscar gate."""
import math
import pytest
from mc_binding.revisit_dataset import specification, accepted


def test_all_routes_have_all_site_heading_visits():
    signatures=[]
    for r in range(6):
        s=specification(r)
        assert s==specification(r)
        assert len(s['targets'])==64
        assert {(v['site'],v['heading']) for v in s['targets']}=={(i,h) for i in range(16) for h in (0,90,180,270)}
        assert s['split']==('train','validation','test')[r//2]
        for v in s['targets']:
            site=s['sites'][v['site']];dx=site['x']-v['approach'][0];dz=site['z']-v['approach'][2]
            assert math.hypot(dx,dz)==pytest.approx(4)
            assert (math.degrees(math.atan2(-dx,dz))-v['heading'])%360==pytest.approx(0)
            assert all(-22<c<22 for c in (v['via'][0],v['via'][2],v['approach'][0],v['approach'][2]))
        signatures.append([(v['site'],v['heading']) for v in s['targets']])
    assert len({str(s) for s in signatures})==6


def test_world_commands_are_identical_and_fill_sizes_valid():
    commands=specification(0)['commands']
    assert all(specification(r)['commands']==commands for r in range(6))
    for command in commands:
        parts=command.split()
        if parts[0]=='/fill':
            a=list(map(int,parts[1:4]));b=list(map(int,parts[4:7]))
            assert math.prod(abs(x-y)+1 for x,y in zip(a,b))<=32768


def test_actual_pose_gates():
    s=dict(x=.5,y=200,z=.5)
    assert accepted(dict(s,yaw=-90),s,270)
    assert not accepted(dict(s,yaw=20),s,0)
    assert not accepted(dict(s,x=1.5,yaw=0),s,0)
    assert not accepted(dict(s,y=84,yaw=0),s,0)
