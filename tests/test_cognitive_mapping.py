"""NON-MINECRAFT synthetic tests; simulator collection remains a separate gate."""
import numpy as np
from mc_binding.cognitive_routes import route_plan,control
from mc_binding.cognitive_audit import ridge_readout


def test_routes_close_and_balance_headings():
    plans=[route_plan(i) for i in range(4)]
    assert len({str(p) for p in plans})==4
    for p in plans:
        assert (p[-1]['x'],p[-1]['z'])==(0,6)
        for yaw in (0,90,180,-90):assert sum(v['yaw']==yaw for v in p)==9
        assert all(-18<v['x']<18 and -5<v['z']<15 for v in p)


def test_probe_controller_turns_without_advancing():
    point=dict(x=0,z=6,yaw=90)
    assert control(dict(x=0,z=6,yaw=0),point)==(False,2)
    assert control(dict(x=0,z=6,yaw=90),point)==(True,None)
    assert control(dict(x=0,z=6,yaw=180),point)==(False,1)


def test_readout_generalizes_linear_coordinate_code():
    rng=np.random.RandomState(2);train=rng.randn(100,4);test=rng.randn(20,4)
    target=train[:,:2]*3
    pred=ridge_readout(train,target,test)
    assert np.mean(np.linalg.norm(pred-test[:,:2]*3,axis=1))<.1


def test_synthetic_kinematics_reaches_every_probe_within_budget():
    # NON-MINECRAFT approximation, not a rendering/physics success claim.
    for route in range(4):
        plan=route_plan(route);j=0;probes=0;pose=dict(x=0.,z=0.,yaw=0.)
        for _ in range(2400):
            reached,action=control(pose,plan[j])
            if reached:
                probes+=int(plan[j]['yaw'] is not None);j+=1
                if j==len(plan):break
                continue
            if action==0:
                angle=np.deg2rad(pose['yaw']);pose['x']-=.5*np.sin(angle);pose['z']+=.5*np.cos(angle)
            else:pose['yaw']+=-10 if action==1 else 10
        assert j==len(plan) and probes==36


def test_non_minecraft_audit_writes_compact_report(tmp_path):
    import torch,json
    from mc_binding.hippocampal import SpatialMemory,odometry
    from mc_binding.cognitive_audit import audit
    net=SpatialMemory();entries=[];data=[]
    for split in ('train','validation','test'):
        for route in range(4):
            name=f'NON-MINECRAFT-{split}-{route}'
            poses=[dict(x=float(i%3),z=float(i//3),yaw=0) for i in range(9)]
            entries.append(dict(episode=name,split=split,route=route,is_minecraft=False,route_complete=True,frames=[dict(pose=p) for p in poses]))
            data.append(dict(episode=name,x=torch.randn(9,64),motion=odometry(poses)))
    (tmp_path/'sequences.json').write_text(json.dumps(entries));(tmp_path/'analysis').mkdir()
    torch.save(dict(sequences=data),tmp_path/'features.pt')
    torch.save(dict(state=net.state_dict(),initial=net.state_dict(),mean=torch.zeros(64),scale=torch.ones(64)),tmp_path/'memory.pt')
    audit(tmp_path)
    result=json.loads((tmp_path/'cognitive-review.json').read_text())
    assert result['data_kind']=='NON-MINECRAFT synthetic fixture'
    assert {r['split'] for r in result['readouts']}=={'train','heldout_route','validation','test'}
    assert (tmp_path/'analysis/cognitive-map.html').exists()


def test_collection_preserves_all_probe_labels_non_minecraft(tmp_path,monkeypatch):
    from argparse import Namespace
    from mc_binding import hippocampal_study as study
    groups={f'f{i}':[dict(kind='full',context='recipient',image='unused.png',objects=[dict(bounds=[[0,0,20],[1,1,21]])],geometry=str(i))] for i in range(3)}
    monkeypatch.setattr(study,'load_binding',lambda _:({'seed':1},groups))
    monkeypatch.setattr(study,'geometry_key',lambda rs:rs[0]['geometry'])
    class FakeWorker:
        def __init__(self,*args):
            self.frame=tmp_path/'NON-MINECRAFT-fixture.bin';self.frame.write_bytes(b'NON-MINECRAFT')
        def request(self,command,**kw):
            if command=='reset':self.pose=dict(x=0.,z=0.,yaw=0.)
            if command=='mapping_pose':return dict(self.pose)
            if command=='step':
                action=kw['index']
                if action==0:
                    angle=np.deg2rad(self.pose['yaw']);self.pose['x']-=.5*np.sin(angle);self.pose['z']+=.5*np.cos(angle)
                else:self.pose['yaw']+=-10 if action==1 else 10
            return dict(frame=str(self.frame))
        def close(self):pass
    monkeypatch.setattr(study,'Worker',FakeWorker)
    entries=study.collect(Namespace(dataset=tmp_path,render_python='unused',scenes=3,cognitive_routes=True,steps=2400,stride=4),tmp_path)
    for e in entries:
        # Production collector labels real Minecraft; this test injects synthetic worker only.
        assert e['route_complete']
        assert sum(f.get('probe') is not None for f in e['frames'])==36
        assert all(b['elapsed_ticks']>=a['elapsed_ticks'] for a,b in zip(e['frames'],e['frames'][1:]))
