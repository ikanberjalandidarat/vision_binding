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
