"""Offline scheduler-command tests; no real Slurm operations."""
import importlib.util
from pathlib import Path


def test_retry_dependencies_reuse_completed_training_and_remap_new_jobs():
    spec=importlib.util.spec_from_file_location('retry',Path(__file__).parents[1]/'scripts/retry_vla_ablation.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    old={'train-raw-episode':'10','features-neutral':'11','rollout-raw-episode':'12'}
    command=['sbatch','--dependency=afterok:10,afterany:11','worker']
    assert m.remap(command,old,{'train-raw-episode'},{'features-neutral':'21'})==['sbatch','--dependency=afterany:21','worker']
    assert m.remap(['sbatch','--dependency=afterok:10','worker'],old,{'train-raw-episode'},{})==['sbatch','worker']
    assert m.remap(['sbatch','--dependency=afterok:11','worker'],old,set(),{'features-neutral':'21'})==['sbatch','--dependency=afterok:21','worker']


def test_retry_archives_failures_preserves_policies_and_submits_nine(tmp_path,monkeypatch,capsys):
    import hashlib,json
    from types import SimpleNamespace
    root=Path(__file__).parents[1]
    spec=importlib.util.spec_from_file_location('retry2',root/'scripts/retry_vla_ablation.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    spec=importlib.util.spec_from_file_location('launch2',root/'scripts/vla_ablation.py');launch=importlib.util.module_from_spec(spec);spec.loader.exec_module(launch)
    suite=tmp_path/'suite';suite.mkdir()
    (suite/'plan.json').write_text(json.dumps(dict(output=str(suite),code_hash=launch.code_hash())))
    keys=['diagnostic','features-neutral','train-raw-episode','train-raw-timestep','train-neutral-episode','train-neutral-timestep','rollout-raw-episode','rollout-raw-timestep','rollout-neutral-episode','rollout-neutral-timestep','comparison']
    jobs={k:str(100+i) for i,k in enumerate(keys)};commands=[]
    for k in keys:
        cmd=['sbatch','--job-name=vla-'+k]
        if k.startswith('rollout-'):cmd+=['--dependency=afterok:'+jobs['train-'+k[8:]]]
        elif k.startswith('train-neutral'):cmd+=['--dependency=afterok:'+jobs['features-neutral']]
        elif k=='comparison':cmd+=['--dependency='+','.join('afterany:'+jobs[x] for x in keys[:-1])]
        commands.append(cmd+['worker'])
    (suite/'submission.json').write_text(json.dumps(dict(jobs=jobs,commands=commands)))
    for arm in ('raw-episode','raw-timestep'):
        p=suite/('policy-'+arm);p.mkdir();(p/'policy.pt').write_bytes(b'preserved')
        (p/'status.json').write_text('{"state":"complete"}')
        (p/'manifest.json').write_text(json.dumps({'checkpoint_sha256':hashlib.sha256(b'preserved').hexdigest()}))
    failed=suite/'features-neutral';failed.mkdir();(failed/'status.json').write_text('{"state":"error"}')
    (suite/'report.html').write_text('old evidence');submitted=[]
    def fake_run(cmd,**kwargs):
        if cmd[0]=='sbatch':submitted.append(cmd);return SimpleNamespace(stdout=str(200+len(submitted)))
        return SimpleNamespace(stdout='')
    monkeypatch.setenv('USER','fixture');monkeypatch.setattr(m.subprocess,'run',fake_run)
    m.retry(suite)
    assert len(submitted)==9
    assert not any('--job-name=vla-train-raw-episode' in c or '--job-name=vla-train-raw-timestep' in c for c in submitted)
    assert (suite/'policy-raw-episode/policy.pt').read_bytes()==b'preserved'
    assert len(list((suite/'retries').glob('*/failed-outputs/features-neutral/status.json')))==1
    assert len(list((suite/'retries').glob('*/failed-outputs/report.html')))==1
    assert not failed.exists()
