"""Non-Minecraft synthetic fixtures; no claim of simulator/model validation."""
from pathlib import Path
import pytest
from mc_binding.vla_env import Environment
from mc_binding.rl_rewards import transition_reward


def test_failed_discounted_returns_are_equal():
    # Telescoping potential and -.01 = -(1-gamma): failure lengths carry no group signal.
    for length in [1,4,32]:
        distances=[1.5-i*.01 for i in range(length+1)]
        total=sum(.99**i*transition_reward(distances[i],distances[i+1],i==length-1,False,'potential') for i in range(length))
        assert total==pytest.approx(-.925)


class Fake:
    episode=Path('train-00000')
    def __init__(self,distance): self.distance=distance; self.waits=0
    def reward_distance(self): return self.distance
    def wait(self): self.waits+=1;return {'frame':'non-Minecraft.png'}
    def finish(self,stopped): return {'success':stopped and (self.distance is None or self.distance<=.8)}


def test_recovery_keeps_action_and_bootstrap_alive():
    e=Fake(1.5)
    r=Environment.rl_step(e,3,recover_stop=True,mode='potential')
    assert r['terminal'] is False and r['recovered_stop'] and e.waits==1
    assert r['reward']==pytest.approx(-.1+.00075)


@pytest.mark.parametrize('distance',[None,.5,.8])
def test_valid_stop_remains_terminal(distance):
    e=Fake(distance)
    r=Environment.rl_step(e,3,recover_stop=True)
    assert r['terminal'] and r['success'] and e.waits==0


def test_deadline_and_evaluation_contract():
    e=Fake(1.5)
    r=Environment.rl_step(e,3,last=True,recover_stop=True)
    assert r['terminal'] and not r['success']
    e.episode=Path('final-greedy-0000')
    with pytest.raises(ValueError,match='training-only'): Environment.rl_step(e,3,recover_stop=True)


@pytest.mark.parametrize('name,training,diagnostic,valid',[
    ('initial-near-train-greedy-0000',None,{'distance':1},False),
    ('final-near-validation-greedy-0000',None,{'distance':1},False),
    ('probe-initial-near-train-greedy-0000',None,{'distance':1},True),
    ('probe-final-near-validation-greedy-0000',None,{'distance':1},True),
    ('probe-4-near-validation-greedy-0000',None,{'distance':1},True),
    ('probe-mixed',{'distance':1},{'distance':1},False),
    ('final-greedy-0000',None,None,True),
    ('final-greedy-0000',{'distance':1},None,False),
    ('train-0000',{'distance':1},None,True),
])
def test_reset_context(name,training,diagnostic,valid):
    from mc_binding.vla_env import validate_reset_context
    if valid: validate_reset_context(name,training,diagnostic)
    else:
        with pytest.raises(ValueError): validate_reset_context(name,training,diagnostic)
