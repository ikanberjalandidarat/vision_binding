#!/usr/bin/env python3
"""Retry unfinished suite stages without changing its code hash or completed outputs."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess


def read(p):return json.loads(Path(p).read_text())

def remap(command, old_jobs, reused, new_jobs):
    reverse={v:k for k,v in old_jobs.items()};result=[]
    for arg in command:
        if not arg.startswith('--dependency='):
            result.append(arg);continue
        deps=[]
        for dep in arg.split('=',1)[1].split(','):
            kind,jid=dep.split(':');key=reverse[jid]
            if key not in reused:deps.append(kind+':'+new_jobs[key])
        if deps:result.append('--dependency='+','.join(deps))
    return result


def retry(suite,dry_run=False):
    suite=Path(suite).resolve();plan=read(suite/'plan.json');old=read(suite/'submission.json')
    repo=Path(__file__).resolve().parents[1]
    spec=importlib.util.spec_from_file_location('suite_launcher',repo/'scripts/vla_ablation.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    if plan['code_hash']!=module.code_hash():raise RuntimeError('Original suite code changed; refusing to mix implementations')
    if Path(plan['output']).resolve()!=suite:raise ValueError('Plan/output mismatch')
    templates={}
    for cmd in old['commands']:
        name=next(x.split('=',1)[1][4:] for x in cmd if x.startswith('--job-name=vla-'))
        templates[name]=cmd
    if set(templates)!=set(old['jobs']):raise ValueError('Incomplete original submission')
    reused=set();archive=[]
    for key in old['jobs']:
        if key=='comparison':continue
        folder=suite/('policy-'+key[6:] if key.startswith('train-') else key)
        status=read(folder/'status.json') if (folder/'status.json').exists() else {}
        if status.get('state')=='complete':
            if key.startswith('train-'):
                expected=read(folder/'manifest.json')['checkpoint_sha256']
                if hashlib.sha256((folder/'policy.pt').read_bytes()).hexdigest()!=expected:raise ValueError('Completed checkpoint hash mismatch')
            reused.add(key)
        elif folder.exists():archive.append(folder)
    if not dry_run:
        # Include previous partial retries; never archive while their jobs are active.
        known=set(old['jobs'].values())
        for p in (suite/'retries').glob('*/submission.json'):
            known.update(read(p).get('jobs',{}).values())
        active=subprocess.run(['squeue','--noheader','--user',os.environ['USER'],'--format=%A'],check=True,capture_output=True,text=True)
        if known.intersection(active.stdout.split()):raise RuntimeError('Suite still has active/queued jobs; wait before retrying')
        model=Path(os.environ.get('MC_BINDING_ENV',str(suite.parent.parent/'envs/model')))/'bin/python'
        render=Path(os.environ.get('MC_RENDER_ENV',str(suite.parent.parent/'envs/render')))/'bin/python'
        subprocess.run([str(model),'-I','-c','import bz2, torch, torchvision; from transformers import AutoProcessor, Qwen2VLForConditionalGeneration; print("Model preflight OK")'],check=True)
        subprocess.run([str(render),'-I','-c','import bz2; from minestudio.simulator import MinecraftSim; print("Render preflight OK")'],check=True)
        subprocess.run([str(render.parent/'java'),'-version'],check=True)
    attempt=suite/'retries'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    jobs={};commands=[]
    def record(extra=None):
        data=dict(reused=sorted(reused),jobs=jobs,commands=commands,archived=[p.name for p in archive])
        if extra:data.update(extra)
        tmp=attempt/'submission.tmp';tmp.write_text(json.dumps(data,indent=2));tmp.replace(attempt/'submission.json')
    if not dry_run:
        (attempt/'failed-outputs').mkdir(parents=True)
        # Keep original logs and previous comparison evidence.
        archive.extend(suite/n for n in ('comparison.json','comparison-status.json','report.html') if (suite/n).exists())
        record()
        for p in archive:p.rename(attempt/'failed-outputs'/p.name)
    try:
        for key in old['jobs']:
            if key in reused:continue
            cmd=remap(templates[key],old['jobs'],reused,jobs)
            if dry_run:jid=str(990000+len(jobs))
            else:
                result=subprocess.run(cmd,check=True,capture_output=True,text=True);jid=result.stdout.strip().split(';')[0]
                if not jid.isdigit():raise RuntimeError('Unexpected sbatch response: '+result.stdout)
            jobs[key]=jid;commands.append(cmd)
            if not dry_run:record()
    except BaseException as e:
        if not dry_run:record({'error':str(e)})
        raise
    print(json.dumps(dict(suite=str(suite),reused=sorted(reused),jobs=jobs,retry_record=str(attempt/'submission.json'),dry_run=dry_run),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('suite');p.add_argument('--dry-run',action='store_true');a=p.parse_args();retry(a.suite,a.dry_run)
