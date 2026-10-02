"""Non-Minecraft toy tests: actual MineStudio/GPU checks remain required."""
import json
from pathlib import Path
import torch
import pytest
from PIL import Image
from mc_binding.vla import Policy,tokenize,spatial_pool,plan,train,rollout
from mc_binding.vla_env import teacher_action,action_dict
from mc_binding.binding_data import binding_contexts
from mc_binding.io import atomic_json,file_hash,digest
from mc_binding.visual_readout import save_tensor


def test_actions_language_memory_and_gradients():
    torch.manual_seed(4)
    model=Policy(8,16);x=torch.randn(32,8)
    a,h,w=model(x,tokenize('Go to the blue arch.'),3)
    b,_,_=model(x,tokenize('Go to the red pillar.'),3)
    assert a.shape==(4,) and h.shape==(1,16) and w.shape==(32,)
    assert not torch.equal(a,b)
    assert float(w.sum().detach())==pytest.approx(1.)
    a.sum().backward()
    assert model.visual[1].weight.grad is not None and model.words.weight.grad is not None
    assert teacher_action(dict(x=0,z=0,yaw=0),None)==3
    assert teacher_action(dict(x=0,z=0,yaw=0),[0,10])==0
    assert teacher_action(dict(x=0,z=0,yaw=0),[10,10])==1
    assert teacher_action(dict(x=0,z=0,yaw=0),[-10,10])==2
    class Sim:
        def noop_action(self):return dict(camera=[0,0],forward=0)
    assert action_dict(Sim(),1)['camera'][1]<0
    with pytest.raises(ValueError):action_dict(Sim(),4)


def test_grid_pool_preserves_locations():
    from mc_binding.vision_data import token_cells
    tokens=torch.tensor([[r*8+c] for r,c in token_cells(4,8,2)],dtype=torch.float32)
    assert torch.equal(spatial_pool(tokens,(4,8),2),torch.arange(32.)[:,None])
    assert len(plan({'f0000':binding_contexts(0,731)}))==16


def test_training_and_rollout_never_request_teacher(tmp_path,monkeypatch):
    import mc_binding.vla as vla
    torch.set_num_threads(1)
    groups={f'f{i:04d}':binding_contexts(i,731) for i in range(24)}
    for rs in groups.values():
        for j,r in enumerate(rs):r.update(record_id=str(j),image='fixture.png')
    data=dict(seed=731,is_minecraft=False,test_fixture=True)
    cache=tmp_path/'cache';cache.mkdir();rows=[]
    for i,(fid,rs) in enumerate(groups.items()):
        ep=f'e{i:05d}';p=cache/f'{ep}.pt'
        save_tensor(p,dict(features=torch.randn(4,32,8),actions=torch.arange(4)))
        rows.append(dict(episode=ep,instruction='Go to the blue arch.',family=fid,geometry=vla.geometry_key(rs),file=p.name,sha256=file_hash(p)))
    config=dict(seed=731,hidden=16,epochs=1,learning_rate=.001,action_chunk=2,max_decisions=2,evaluation_episodes=1)
    atomic_json(cache/'manifest.json',dict(config=config,demos_manifest={'dataset_hash':digest(data),'config':config}))
    atomic_json(cache/'index.json',rows);atomic_json(cache/'status.json',{'state':'complete'})
    out=tmp_path/'policy';train(cache,out,config)
    manifest=json.loads((out/'manifest.json').read_text())
    assert len(manifest['splits'])==6
    assert (out/'offline-test.json').exists()
    monkeypatch.setattr(vla,'load_binding',lambda path:(data,groups))
    class FakeEncoder:
        def __init__(self,config):self.visual=type('V',(),{'spatial_merge_size':2})()
        def inputs(self,image):return None,torch.tensor([[1,4,8]])
        def features(self,inputs):return torch.zeros(32,8)
    class FakeWorker:
        commands=[]
        def __init__(self,python,output,chunk):
            self.root=Path(output);self.root.mkdir(parents=True);self.frame=self.root/'frame.png';Image.new('RGB',(448,280)).save(self.frame)
        def request(self,command,**kwargs):
            self.commands.append(command)
            assert command!='teacher','Oracle teacher accessed during learned rollout'
            if command in ('reset','step'):return {'frame':str(self.frame)}
            if command=='finish':return dict(success=False,distance=10.,episode_file='toy',goal_present=True)
        def close(self):pass
    monkeypatch.setattr(vla,'Encoder',FakeEncoder);monkeypatch.setattr(vla,'Worker',FakeWorker)
    result=tmp_path/'rollout';rollout(tmp_path,out,result,'fake-python',config,True)
    assert json.loads((result/'status.json').read_text())['state']=='complete'
    assert (result/'decisions-00000.json').exists()


def test_worker_protocol_handles_buffered_noise_and_responses(tmp_path):
    import io,os
    from types import SimpleNamespace
    from mc_binding.vla import Worker
    rd,wr=os.pipe()
    worker=Worker.__new__(Worker);worker.root=tmp_path;worker.buffer=b'';worker.log=io.StringIO()
    stream=os.fdopen(rd,'rb',buffering=0)
    worker.p=SimpleNamespace(stdin=io.BytesIO(),stdout=stream)
    try:
        os.write(wr,b'simulator startup noise\n{"ok":true,"result":{"action":1}}\n{"ok":true,"result":{"success":false}}\n')
        assert worker.request('teacher')=={'action':1}
        assert worker.request('finish',stopped=True)=={'success':False}
        assert 'startup noise' in worker.log.getvalue()
    finally:
        stream.close();os.close(wr)
