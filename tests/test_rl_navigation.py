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
