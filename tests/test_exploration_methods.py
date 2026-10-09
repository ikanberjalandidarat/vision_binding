"""Synthetic non-Minecraft tests; curiosity is not navigation success."""
import torch
from mc_binding.intrinsic import Curiosity
from mc_binding.grpo import group_advantages


def test_grpo_ties_and_relative_rewards():
 assert torch.equal(group_advantages([0,0,0,0]),torch.zeros(4))
 a=group_advantages([0,0,1,0]);assert a[2]>0 and a[0]<0
 assert abs(float(a.mean()))<1e-6


def test_rnd_target_frozen_and_bonus_read_only():
 c=Curiosity('rnd',6,731);x=torch.randn(6);y=torch.randn(6)
 before={k:v.clone() for k,v in c.model.target.state_dict().items()}
 full={k:v.clone() for k,v in c.model.state_dict().items()}
 raw,bonus=c.bonus(x,0,y)
 assert 0<=bonus<=5 and c.count==0
 assert all(torch.equal(v,c.model.state_dict()[k]) for k,v in full.items())
 c.train_batch([(x,0,y,raw)])
 assert all(torch.equal(v,c.model.target.state_dict()[k]) for k,v in before.items())
 assert c.count==1
 assert any(not torch.equal(v,c.model.state_dict()[k]) for k,v in full.items())


def test_icm_updates_inverse_and_forward():
 c=Curiosity('icm',6,731)
 before={k:v.clone() for k,v in c.model.state_dict().items()};transitions=[]
 for action in range(3):
  x=torch.randn(6);y=torch.randn(6);raw,_=c.bonus(x,action,y);transitions.append((x,action,y,raw))
 c.train_batch(transitions)
 for prefix in ['encoder','inverse','forward_model']:
  assert any(not torch.equal(v,c.model.state_dict()[k]) for k,v in before.items() if k.startswith(prefix))
