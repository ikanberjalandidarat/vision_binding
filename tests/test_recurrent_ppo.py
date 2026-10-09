"""Synthetic non-Minecraft fixtures; real model/simulator remain gates."""
import pytest
import torch
from mc_binding.recurrent_ppo import gae,replay,clipped_loss,update
from mc_binding.rl_navigation import initialize_agent


def test_gae_terminal_vs_collection_boundary():
 a,r=gae([1.],[.5],[True],bootstrap=10.,gamma=.9,lam=1.)
 assert a.tolist()==[.5] and r.tolist()==[1.]
 a,r=gae([1.],[.5],[False],bootstrap=10.,gamma=.9,lam=1.)
 assert r.tolist()==[10.]
 _,r=gae([1.,2.],[0.,0.],[False,True],gamma=.5,lam=1.)
 assert r.tolist()==[2.,2.]
 # A terminal inside the input must block leakage from the next episode.
 _,r=gae([1.,100.],[0.,0.],[True,True],gamma=.5,lam=1.)
 assert r.tolist()==[1.,100.]


def test_ppo_clips_both_advantage_signs():
 old=torch.zeros(2);new=torch.tensor([2.,.5]).log().requires_grad_()
 loss,stats=clipped_loss(new,old,torch.tensor([1.,-1.]),torch.zeros(2),torch.zeros(2),torch.zeros(2))
 assert float(loss)==pytest.approx(-.2)
 loss.backward();assert torch.equal(new.grad,torch.zeros(2))
 assert stats['clip_fraction']==1


def test_recurrent_replay_and_binding_gradient():
 torch.set_num_threads(1);model=initialize_agent(6,731,binding=True)
 ep=dict(features=[torch.randn(128,6) for _ in range(3)],signals=[torch.rand(130) for _ in range(3)],instruction=torch.tensor([2,3]),previous=[3,0,1],actions=torch.tensor([0,1,2]))
 with torch.no_grad():old,values,_=replay(model,ep)
 ep['old_logps']=old;ep['advantages'],ep['returns']=gae([.1,.2,1.],values,[False,False,True])
 before=model.binding[0].weight.detach().clone()
 logs=update(model,torch.optim.Adam(model.parameters(),lr=3e-4),[ep],torch.Generator().manual_seed(1),epochs=2)
 assert logs[0]['replay_error']==0
 assert not torch.equal(before,model.binding[0].weight)
 # Old-policy data cannot silently be reused after changing the collection policy.
 ep['old_logps']=old+1
 with pytest.raises(RuntimeError,match='replay mismatch'):update(model,torch.optim.Adam(model.parameters()),[ep],torch.Generator())


@pytest.mark.parametrize('method',['ppo','grpo','ppo-icm','ppo-rnd'])
def test_runner_performs_live_ppo_update(tmp_path,monkeypatch,method):
 import json
 from argparse import Namespace
 from PIL import Image
 import mc_binding.ppo_navigation as p
 from mc_binding.visual_readout import Readout
 image=tmp_path/'non-minecraft.png';Image.new('RGB',(448,280),'red').save(image)
 ck=tmp_path/'readout.pt';torch.save(dict(width=6,state=Readout(6).state_dict()),ck)
 (tmp_path/'manifest.json').write_text(json.dumps(dict(splits={'t':'train','v':'validation'})))
 config=tmp_path/'config.json';config.write_text('{"readout_layer":31}')
 jobs=[dict(family=g,geometry=g,record=dict(record_id=g,image=image.name,objects=[dict(color='red',type='pillar')]),goal=['red','pillar'],instruction='Go to the red pillar.') for g in ('t','v')]
 monkeypatch.setattr(p,'load_binding',lambda _:({'seed':731},{}));monkeypatch.setattr(p,'plan',lambda _:jobs)
 monkeypatch.setattr(p,'distributed_tasks',lambda jobs,n:jobs[:n]);monkeypatch.setattr(p,'proposals',lambda im:[dict(bbox_raw=[0,0,28,35])])
 class Encoder:
  def __init__(self,_):self.visual=type('Visual',(),{'spatial_merge_size':1})()
  def inputs(self,_):return None,torch.tensor([[1,2,2]])
  def features(self,_):return torch.ones(4,6)
 requests=[]
 class Worker:
  def __init__(self,*a):pass
  def request(self,command,**kw):
   requests.append(command)
   if command=='reset':return dict(frame=str(image))
   assert command=='rl_step'
   return dict(reward=1.,terminal=True,success=True,observation=dict(frame=str(image)))
  def close(self):pass
 monkeypatch.setattr(p,'Encoder',Encoder);monkeypatch.setattr(p,'Worker',Worker)
 out=tmp_path/'out';p.run(Namespace(dataset=str(tmp_path),checkpoint=str(ck),output=str(out),render_python='unused',config=str(config),episodes=2,batch_episodes=2,max_steps=3,eval_episodes=1,epochs=2,seed=731,mode='binding',start_mode='original',method=method))
 assert json.loads((out/'status.json').read_text())['state']=='complete'
 logs=json.loads((out/'updates.json').read_text());assert len(logs)==1 and logs[0]['transitions']==2
 initial=torch.load(out/'initial.pt',weights_only=True);final=torch.load(out/'agent.pt',weights_only=True)
 assert not torch.equal(initial['state']['action.weight'],final['state']['action.weight'])
 assert len(json.loads((out/'initial.json').read_text()))==2
 assert len(json.loads((out/'final.json').read_text()))==2
 if method=='grpo':assert torch.equal(initial['state']['value.weight'],final['state']['value.weight'])
 for label in ('initial','final'):
  assert all(r['intrinsic_return']==0 for r in json.loads((out/(label+'.json')).read_text()))
 if method in ('ppo-icm','ppo-rnd'):
  assert final['curiosity']['count']<=2
 assert set(requests)=={'reset','rl_step'}
