"""Non-Minecraft fixtures; no Slurm, simulator, or GPU success implied."""
import importlib.util
import json
from pathlib import Path
import numpy as np
from PIL import Image
import pytest
import torch
from mc_binding.vla import observation_image,loss_scale,action_metrics,ACTIONS
from mc_binding.vla_diagnostic import checked_frame,summarize


def test_preprocessing_preserves_scene_and_does_not_mutate_raw():
    raw=Image.fromarray(np.random.default_rng(2).integers(0,256,(280,448,3),dtype=np.uint8))
    before=np.asarray(raw).copy();masked=np.asarray(observation_image(raw,{'observation_mode':'neutral_bands_v1'}))
    assert np.array_equal(np.asarray(raw),before)
    assert np.array_equal(masked[40:195],before[40:195])
    assert (masked[:40]==128).all() and (masked[195:]==128).all()
    assert np.array_equal(np.asarray(observation_image(raw,{})),before)
    with pytest.raises(ValueError):observation_image(raw,{'observation_mode':'typo'})


def test_timestep_weighting_matches_global_mean_gradient():
    x=torch.tensor(.2,requires_grad=True);losses=[(x-1).square()[None],(x+2).square().repeat(9)]
    expected=torch.cat(losses).mean()
    weighted=sum(v.mean()*loss_scale(len(v),5.,'timestep') for v in losses)/2
    assert float(weighted.detach())==pytest.approx(float(expected.detach()))
    assert torch.allclose(torch.autograd.grad(weighted,x,retain_graph=True)[0],torch.autograd.grad(expected,x)[0])
    assert loss_scale(1,5.,'episode')==1
    with pytest.raises(ValueError):loss_scale(1,5.,'bad')


def test_metrics_separate_startup_and_previous_action_history():
    class PreviousPolicy:
        def eval(self):pass
        def __call__(self,x,instruction,previous,state):
            logits=torch.zeros(4);logits[previous]=1
            return logits,None,None
    rows=[dict(instruction='Go to the red arch.',data={'features':torch.zeros(3,32,2),'actions':torch.tensor([0,0,3])})]
    a=action_metrics(PreviousPolicy(),rows,'teacher');b=action_metrics(PreviousPolicy(),rows,'predicted')
    assert a['first_steps']['accuracy']==0
    assert a['all_steps']['confusion'][0][0]==1
    assert b['all_steps']['confusion'][0][0]==0
    assert b['all_steps']['confusion'][3][3]==1


def load_launcher():
    spec=importlib.util.spec_from_file_location('vla_launcher',Path(__file__).parents[1]/'scripts/vla_ablation.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def test_submission_dependencies_and_partial_comparison(tmp_path,monkeypatch,capsys):
    from types import SimpleNamespace
    m=load_launcher();out=tmp_path/'suite'
    plan=dict(output=str(out),arms={},paths={},config={},scope='Non-Minecraft test fixture')
    monkeypatch.setattr(m,'build_plan',lambda args:plan)
    # Dry-run submits nothing and writes nothing.
    m.submit(SimpleNamespace(dry_run=True))
    data=json.loads(capsys.readouterr().out);cmds=data['commands'];assert len(cmds)==11
    assert not out.exists()
    by_key={c[c.index('--job-name='+next(v.split('=',1)[1] for v in c if v.startswith('--job-name=')))].split('=',1)[1]:c for c in cmds}
    neutral=by_key['vla-train-neutral-episode'];raw=by_key['vla-train-raw-episode']
    assert '--dependency=afterok:900001' in neutral
    assert not any(s.startswith('--dependency=') for s in raw)
    for key,c in by_key.items():
        assert '--kill-on-invalid-dep=yes' in c
        if 'rollout-' in key:assert any(s.startswith('--dependency=afterok:') for s in c)
    assert any('afterany:' in s for s in by_key['vla-rollout-raw-timestep'])
    assert '--partition=batch' in raw and not any(s.startswith('--gres=') for s in raw)
    out.mkdir();m.compare(plan)
    result=json.loads((out/'comparison.json').read_text())
    assert all(r['status']['state']=='missing or blocked' and 'summary' not in r for r in result['arms'])


def test_frame_integrity_and_diagnostic_false_stops(tmp_path):
    with pytest.raises(ValueError):checked_frame(tmp_path,{'frame':'../outside','sha256':'x'})
    rows=[dict(source='teacher',mode='raw',present=True,predicted='stop',correct=False),
          dict(source='teacher',mode='raw',present=False,predicted='stop',correct=True)]
    group=summarize(rows)[0]
    assert group['false_stops']==1 and group['absent_stops']==1 and group['correct']==1


def test_matched_diagnostic_runs_without_minecraft_or_gpu(tmp_path,monkeypatch):
    import mc_binding.vla_diagnostic as diagnostic
    from mc_binding.vla import Policy
    from mc_binding.io import atomic_json,file_hash
    from mc_binding.binding_data import FEATURES
    from mc_binding.visual_readout import save_tensor
    torch.set_num_threads(1)
    demos,rollout,policy,cache=[tmp_path/n for n in ('demos','rollout','policy','cache')]
    for p in (demos,rollout,policy,cache):p.mkdir();atomic_json(p/'status.json',{'state':'complete'})
    frame=demos/'frame.png';Image.new('RGB',(448,280),(10,20,30)).save(frame)
    eps=[]
    for i,goal in enumerate(FEATURES):
        eps.append(dict(record={'record_id':'000004'},goal=list(goal),samples=[{'frame':'frame.png','sha256':file_hash(frame),'action':i}],result={'goal_present':i!=3}))
    atomic_json(demos/'episodes.json',eps)
    config={'observation_mode':'raw'};fm={'config':config,'episodes_hash':file_hash(demos/'episodes.json')};atomic_json(cache/'manifest.json',fm)
    tensor=cache/'e00000.pt';save_tensor(tensor,{'features':torch.ones(2,32,8),'actions':torch.tensor([0,3])})
    atomic_json(cache/'index.json',[dict(file=tensor.name,sha256=file_hash(tensor),geometry='toy',instruction='Go to the blue arch.')])
    ck=policy/'policy.pt';save_tensor(ck,dict(state=Policy(8,16).state_dict(),width=8,hidden=16))
    pm=dict(checkpoint_sha256=file_hash(ck),feature_manifest=fm,feature_index_hash=file_hash(cache/'index.json'),splits={'toy':'test'})
    atomic_json(policy/'manifest.json',pm);atomic_json(rollout/'manifest.json',{'policy':pm})
    ep=rollout/'episodes/e00000';ep.mkdir(parents=True);Image.open(frame).save(ep/'frame-0000.png')
    atomic_json(ep/'episode.json',{'record_id':'000004','goal':list(FEATURES[0])})
    atomic_json(rollout/'results.json',[{'episode':'e00000'}])
    class FakeEncoder:
        def __init__(self,c):self.visual=type('V',(),{'spatial_merge_size':2})()
        def inputs(self,image):return None,torch.tensor([[1,4,8]])
        def features(self,inputs):return torch.ones(32,8)
    monkeypatch.setattr(diagnostic,'Encoder',FakeEncoder)
    out=tmp_path/'diagnostic';diagnostic.run(demos,rollout,policy,cache,out)
    rows=json.loads((out/'initial-decisions.json').read_text())
    assert len(rows)==16
    assert {tuple(r['goal']) for r in rows}==set(FEATURES)
    assert sum(not r['present'] for r in rows)==4
    assert json.loads((out/'status.json').read_text())['state']=='complete'
    assert (out/'report.html').exists()
