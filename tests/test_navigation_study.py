"""Non-Minecraft fixtures: real GPU/simulator checks are separate gates."""
import importlib.util
import json
from pathlib import Path
import torch
import pytest
from mc_binding.navigation_metrics import score_episode
from mc_binding.vla import decision_weights,geometry_splits
from mc_binding.vla_selection import label_for,Selector,evaluate,train


def trace(*positions):return [dict(pose=dict(x=x,z=z)) for x,z in positions]

def test_strict_absence_and_arrival_metrics():
    assert score_episode(trace((0,0)),None,True)['success']
    r=score_episode(trace((0,0),(1,0),(0,0)),None,True)
    assert not r['success'] and r['legacy_success'] and r['travel_distance']==2
    # A turn-only action still violates immediate refusal.
    assert not score_episode(trace((0,0),(0,0)),None,True)['success']
    r=score_episode(trace((0,0),(9.5,0)),[10,0],True)
    assert r['success'] and r['distance']==.5
    assert not score_episode(trace((0,0),(9.5,0)),[10,0],False)['success']
    r=score_episode(trace((0,0),(10,0),(12,0)),[10,0],True)
    assert not r['success'] and r['minimum_sampled_goal_distance']==0


def test_fixed_split_and_declared_startup_turn_weighting():
    assert geometry_splits(range(6),dict(seed=731,split_seed=731))==geometry_splits(range(6),dict(seed=732,split_seed=731))
    actions=torch.tensor([1,1,0,3])
    assert decision_weights(actions,dict(startup_weight=5,turn_weight=2)).tolist()==[10,2,1,1]
    assert decision_weights(actions,{}).tolist()==[1,1,1,1]
    with pytest.raises(ValueError):decision_weights(actions,dict(turn_weight=-1))


def objects():
    return [dict(color='blue',type='pillar',bbox=[0,0,1,1]),dict(color='red',type='arch',bbox=[2,0,3,1])]

def test_region_labels_require_color_and_shape_and_controls():
    record={'objects':objects()}
    assert label_for(dict(record=record,goal=['red','arch']))==1
    assert label_for(dict(record=record,goal=['blue','arch']))==4
    torch.manual_seed(1);model=Selector(8,16)
    rows=[dict(episode='toy',family='toy',instruction='Go to the blue arch.',features=torch.randn(32,8),label=4,goal=['blue','arch'],objects=objects())]
    result=evaluate(model,rows)
    assert len(result['rows'])==4
    assert all(len(r['probabilities'])==5 and r['expected']==4 for r in result['rows'])


def load_script():
    spec=importlib.util.spec_from_file_location('navstudy',Path(__file__).parents[1]/'scripts/navigation_study.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def test_bulk_dag_fixed_seeds_cpu_training_gpu_rollout(tmp_path,capsys):
    m=load_script();arms,stages=m.graph([731,732,733]);assert len(stages)==17
    assert all(v['split_seed']==731 and not v['report_test_metrics'] for v in arms.values())
    plan=dict(output=str(tmp_path/'suite'),code_hash=m.code_hash(),arms=arms,stages=stages)
    m.schedule(plan,True);r=json.loads(capsys.readouterr().out)
    assert len(r['jobs'])==17 and not (tmp_path/'suite').exists()
    cmds=r['commands'];assert all('--partition=batch' in c for c in cmds if any('--job-name=nav-train-' in v for v in c))
    rollout=[c for c in cmds if '--gres=gpu:1' in c];assert len(rollout)==6
    assert 'afterany:' in next(v for v in rollout[1] if v.startswith('--dependency='))


def test_selector_training_keeps_test_unevaluated(tmp_path):
    from mc_binding.io import atomic_json,file_hash
    from mc_binding.visual_readout import save_tensor
    cache=tmp_path/'cache';cache.mkdir();demos=tmp_path/'demos';demos.mkdir();eps=[];index=[]
    for i in range(6):
        eid=f'e{i:05d}';eps.append(dict(episode=eid,record={'objects':objects()},goal=['red','arch']))
        p=cache/(eid+'.pt');save_tensor(p,dict(features=torch.randn(2,32,8),actions=torch.tensor([1,3])))
        index.append(dict(episode=eid,family='f'+str(i),geometry=str(i),instruction='Go to the red arch.',file=p.name,sha256=file_hash(p)))
    atomic_json(demos/'episodes.json',eps);atomic_json(cache/'manifest.json',{'episodes_hash':file_hash(demos/'episodes.json')});atomic_json(cache/'status.json',{'state':'complete'});atomic_json(cache/'index.json',index)
    train(cache,demos,tmp_path/'selector',dict(seed=731,split_seed=731,hidden=16,learning_rate=.001,selector_epochs=1))
    assert json.loads((tmp_path/'selector/status.json').read_text())['state']=='complete'
    assert not (tmp_path/'selector/test.json').exists()
    assert json.loads((tmp_path/'selector/validation.json').read_text())['summary'][0]['n']==1


def test_simulator_error_observations_are_not_scored():
    from mc_binding.vla_env import Environment
    class Sim:
        def step(self,a):return {},0,False,False,{'error':'connection lost'}
    env=Environment.__new__(Environment);env.sim=Sim()
    with pytest.raises(RuntimeError,match='fallback observation'):env._step({})


def test_retry_reuses_train_and_requires_fresh_preflight(tmp_path,monkeypatch,capsys):
    import hashlib
    from types import SimpleNamespace
    m=load_script();arms,stages=m.graph([731]);out=tmp_path/'study';out.mkdir()
    plan=dict(output=str(out),code_hash=m.code_hash(),arms=arms,stages=stages)
    folder=out/'train-timestep-s731';folder.mkdir();(folder/'status.json').write_text('{"state":"complete"}');(folder/'policy.pt').write_bytes(b'checkpoint');(folder/'manifest.json').write_text(json.dumps({'checkpoint_sha256':hashlib.sha256(b'checkpoint').hexdigest()}))
    failed=out/'rollout-timestep-s731';failed.mkdir();(failed/'status.json').write_text('{"state":"error"}')
    calls=[]
    def run(cmd,**kwargs):
        if cmd[0]=='sbatch':calls.append(cmd);return SimpleNamespace(stdout=str(100+len(calls)))
        return SimpleNamespace(stdout='')
    monkeypatch.setattr(m.subprocess,'run',run);monkeypatch.setenv('USER','fixture')
    m.schedule(plan,retry=True)
    assert (folder/'policy.pt').read_bytes()==b'checkpoint'
    assert not failed.exists()
    cmd=next(c for c in calls if '--job-name=nav-rollout-timestep-s731' in c)
    assert '--dependency=afterok:101' in cmd
    assert not any('--job-name=nav-train-timestep-s731' in c for c in calls)
    assert list((out/'attempts').glob('*/archived/rollout-timestep-s731/status.json'))
