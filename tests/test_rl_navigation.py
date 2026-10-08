import pytest
import torch
from mc_binding.rl_rewards import transition_reward,potential
from mc_binding.rl_navigation import returns,objective,Agent

def test_rewards():
 assert transition_reward(10,9,False,False)==-.01
 assert transition_reward(10,10,True,False)==-1
 assert transition_reward(None,None,True,True)==1
 # Discounted shaping telescopes, so ending early cannot create net shaping profit.
 g=.99
 a=transition_reward(10,8,False,False,'potential',g)+g*transition_reward(8,6,True,False,'potential',g)
 base=-.01-g
 assert a-base==pytest.approx(-potential(10))

def test_actor_critic_gradient():
 torch.set_num_threads(1);m=Agent(6);x=torch.randn(128,6);logits,state,_=m(x,torch.tensor([1,2]),3)
 d=torch.distributions.Categorical(logits=logits)
 loss=objective([d.log_prob(torch.tensor(0))],[m.value(state).squeeze()],[d.entropy()],[1.],.99,[],0)
 loss.backward();assert m.action.weight.grad.abs().sum()>0
 assert m.value.weight.grad.abs().sum()>0
 assert torch.allclose(returns([1,2],.5),torch.tensor([2.,2.]))


def test_rl_seed_survives_encoder_global_seed_reset():
 from mc_binding.rl_navigation import initialize_agent,rollout_generators
 torch.manual_seed(731)  # mimics Qwen constructor reset
 first=initialize_agent(6,732)
 torch.manual_seed(999)
 again=initialize_agent(6,732)
 other=initialize_agent(6,733)
 assert torch.equal(first.action.weight,again.action.weight)
 assert not torch.equal(first.action.weight,other.action.weight)
 s,a=rollout_generators(732);s2,a2=rollout_generators(732)
 torch.manual_seed(731)
 assert torch.equal(torch.randint(100,(20,),generator=s),torch.randint(100,(20,),generator=s2))
 assert torch.equal(torch.multinomial(torch.ones(4),20,replacement=True,generator=a),torch.multinomial(torch.ones(4),20,replacement=True,generator=a2))


def test_training_curriculum():
 from mc_binding.rl_pilot import training_start
 assert training_start('original',0,48) is None
 assert training_start('near-to-far',0,48)['distance']==1.5
 assert training_start('near-to-far',12,48)['distance']==3
 assert training_start('near-to-far',24,48)['distance']==6
 assert training_start('near-to-far',3,48) is None
 assert training_start('near-to-far',36,48) is None


def test_paired_evaluation_uses_original_starts(tmp_path,monkeypatch):
 """Synthetic non-Minecraft data: exercise orchestration, not model/render validity."""
 import json
 from PIL import Image
 import mc_binding.rl_navigation as rl
 image=tmp_path/'synthetic.png';Image.new('RGB',(8,8)).save(image)
 record={'image':image.name,'objects':[{'color':'red','type':'pillar'}]}
 jobs=[dict(geometry=g,family=g,record=record,goal=['red','pillar'],instruction='Go to the red pillar.') for g in ('train','validation')]
 monkeypatch.setattr(rl,'load_binding',lambda _:({'seed':731},{}))
 monkeypatch.setattr(rl,'plan',lambda _:jobs)
 monkeypatch.setattr(rl,'geometry_splits',lambda *_:{'train':'train','validation':'validation'})
 class Encoder:
  def __init__(self,_):self.visual=type('Visual',(),{'spatial_merge_size':1})()
  def inputs(self,_):return None,torch.tensor([[1,2,2]])
  def features(self,_):return torch.ones(4,6)
 monkeypatch.setattr(rl,'Encoder',Encoder)
 monkeypatch.setattr(rl,'rasterize',lambda *_:torch.ones(2,2,6))
 resets=[]
 class Worker:
  def __init__(self,*_):pass
  def request(self,command,**kwargs):
   if command=='reset':resets.append(kwargs);return {'frame':str(image)}
   return dict(reward=-1.,terminal=True,success=False)
  def close(self):pass
 monkeypatch.setattr(rl,'Worker',Worker)
 out=tmp_path/'run'
 rl.run(tmp_path,out,'unused',{},731,'potential',0,2,2,1,'near-to-far',True)
 assert all(r['training_start'] is not None for r in resets[:2])
 assert all(r['training_start'] is None for r in resets[2:])
 assert resets[2]['record']==resets[3]['record']
 rows=json.loads((out/'validation.json').read_text())
 assert [r['evaluation_mode'] for r in rows]==['greedy','sampled']
 assert all(len(r['decisions'][0]['probabilities'])==4 for r in rows)
 assert len(json.loads((out/'training.json').read_text()))==2
