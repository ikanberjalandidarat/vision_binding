#!/usr/bin/env python3
"""One submission builds a dependency DAG; stdlib only on the login node."""
import argparse
import hashlib
import html
import json
import os
from pathlib import Path
import re
import subprocess
import sys

REPO=Path(__file__).resolve().parents[1]
ARMS={'raw-episode':('raw','episode'),'raw-timestep':('raw','timestep'),
      'neutral-episode':('neutral_bands_v1','episode'),'neutral-timestep':('neutral_bands_v1','timestep')}


def read(p):return json.loads(Path(p).read_text())


def write(p,data):
    p=Path(p);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(data,indent=2));tmp.replace(p)


def code_hash():
    h=hashlib.sha256()
    files=sorted((REPO/'src').rglob('*.py'))+[Path(__file__),REPO/'scripts/slurm/vla_ablation.sbatch']
    for p in files:h.update(str(p.relative_to(REPO)).encode());h.update(p.read_bytes())
    return h.hexdigest()


def build_plan(args):
    if not re.fullmatch(r'[A-Za-z0-9_-]+',args.tag):raise ValueError('Tag must use letters, numbers, _ or -')
    if not args.reviewed_captures:raise ValueError('Pass --reviewed-captures for the already reviewed binding dataset')
    runs=Path(args.runs).resolve();output=runs/f'vla-ablation-{args.tag}'
    if output.exists():raise ValueError('Suite output already exists; choose a new tag')
    paths={name:str(runs/getattr(args,name)) for name in ('dataset','demos','cache','policy','rollout')}
    for name,path in paths.items():
        if runs not in Path(path).resolve().parents:raise ValueError('Input must be under runs')
        if not Path(path).is_dir():raise ValueError(f'Missing {name}: {path}')
        if name!='dataset' and read(Path(path)/'status.json')['state']!='complete':raise ValueError(f'Incomplete {name}')
    fm=read(Path(paths['cache'])/'manifest.json');pm=read(Path(paths['policy'])/'manifest.json')
    if hashlib.sha256((Path(paths['demos'])/'episodes.json').read_bytes()).hexdigest()!=fm['episodes_hash']:raise ValueError('Cache was not extracted from these demos')
    if pm['feature_manifest']!=fm:raise ValueError('Original policy/cache mismatch')
    cfg=dict(pm['config'],evaluation_split='validation',evaluation_episodes=64)
    if fm['config'].get('observation_mode','raw')!='raw':raise ValueError('Expected original raw cache')
    return dict(schema='vla_ablation_v1',repo=str(REPO),output=str(output),paths=paths,code_hash=code_hash(),config=cfg,
                reviewed_captures=True,arms={a:dict(cfg,observation_mode=mode,loss_weighting=weight) for a,(mode,weight) in ARMS.items()},
                scope='Development 2x2 experiment on validation geometry. Test results from the original policy are diagnostic only. No automatic best-arm selection or guaranteed navigation success.')


def submit(args):
    plan=build_plan(args);out=Path(plan['output']);planpath=out/'plan.json';jobs={};commands=[]
    if not args.dry_run:
        out.mkdir();(out/'logs').mkdir();write(planpath,plan)
    def queue(key,stage,arm='none',dependencies=()):
        gpu=stage in ('diagnostic','features','rollout')
        cmd=['sbatch','--parsable','--kill-on-invalid-dep=yes',f'--job-name=vla-{key}',
             '--partition='+('gpu' if gpu else 'batch'),'--nodes=1','--cpus-per-task=4','--mem=48G','--time=03:00:00',
             '--chdir='+str(REPO),'--output='+str(out/'logs'/f'{key}-%j.out')]
        if gpu:cmd+=['--gres=gpu:1']
        if dependencies:cmd+=['--dependency='+','.join(f'{kind}:{jobs[job]}' for kind,job in dependencies)]
        cmd += [str(REPO/'scripts/slurm/vla_ablation.sbatch'),str(planpath),stage,arm]
        if args.dry_run:jid=str(900000+len(jobs))
        else:
            result=subprocess.run(cmd,check=True,text=True,capture_output=True);jid=result.stdout.strip().split(';')[0]
            if not jid.isdigit():raise RuntimeError(f'Unrecognized sbatch response: {result.stdout}')
        jobs[key]=jid;commands.append(cmd)
        if not args.dry_run:write(out/'submission.json',dict(jobs=jobs,commands=commands))
    try:
        queue('diagnostic','diagnostic');queue('features-neutral','features')
        for arm in ARMS:queue('train-'+arm,'train',arm,() if arm.startswith('raw') else (('afterok','features-neutral'),))
        previous=None
        for arm in ARMS:
            deps=[('afterok','train-'+arm)]
            if previous:deps.append(('afterany',previous))
            key='rollout-'+arm;queue(key,'rollout',arm,deps);previous=key
        queue('comparison','compare',dependencies=tuple(('afterany',key) for key in list(jobs)))
    except BaseException as e:
        if not args.dry_run:write(out/'submission-error.json',dict(error=str(e),submitted_jobs=jobs))
        raise
    print(json.dumps(dict(output=str(out),jobs=jobs,commands=commands if args.dry_run else 'See submission.json'),indent=2))


def compare(plan):
    out=Path(plan['output']);rows=[]
    for arm in ARMS:
        p=out/('rollout-'+arm);s=read(p/'status.json') if (p/'status.json').exists() else {'state':'missing or blocked'}
        row=dict(arm=arm,status=s)
        if s.get('state')=='complete':
            summary=read(p/'summary.json');results=read(p/'results.json');immediate=0;timeouts=0
            for r in results:
                ep=read(p/'episodes'/r['episode']/'episode.json')
                immediate+=bool(r['goal_present'] and ep['stopped'] and len(ep['trajectory'])==1)
                timeouts+=not ep['stopped']
            row.update(summary=summary,immediate_present_stops=immediate,timeouts=timeouts)
        policy=out/('policy-'+arm)
        if (policy/'offline-metrics.json').exists():row['offline_validation']=read(policy/'offline-metrics.json')['validation']
        rows.append(row)
    write(out/'comparison.json',dict(arms=rows,scope=plan['scope']))
    body='<h1>VLA appearance × training-weight comparison</h1><p>'+html.escape(plan['scope'])+'</p><p>Each arm uses the same seed, demonstrations, geometry split, architecture and epoch budget. Each chooses its checkpoint using its declared validation loss. All validation episodes are requested; rollout success is distinct from offline action accuracy. Raw HUD GIFs are preserved. Gray-band inputs also remove floor information, so their effect is not specific to tutorial pop-ups or hands.</p><p><a href="diagnostic/report.html">Matched teacher/rollout diagnostic</a></p>'
    body+='<table><tr><th>Arm</th><th>Status</th><th>Present arrival</th><th>Absent stop</th><th>Immediate present stops</th><th>Timeouts</th><th>Evidence</th></tr>'
    for r in rows:
        rates={x['goal_present']:str(x['successes'])+'/'+str(x['n']) for x in r.get('summary',{}).get('by_presence',[])}
        cells=[r['arm'],r['status']['state'],rates.get(True,'—'),rates.get(False,'—'),r.get('immediate_present_stops','—'),r.get('timeouts','—')]
        body+='<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in cells)+('<td><a href="rollout-'+r['arm']+'/report.html">Movement / maps</a></td>' if r['status']['state']=='complete' else '<td>No completed rollout</td>')+'</tr>'
    body+='</table>'
    for r in rows:
        if r['status']['state']!='complete':continue
        arm=r['arm'];base=out/('rollout-'+arm);results=read(base/'results.json');selected=[]
        for result in results:
            if not result['goal_present']:continue
            ep=read(base/'episodes'/result['episode']/'episode.json')
            if not selected or (len(selected)==1 and len(ep['trajectory'])>1 and result['episode']!=selected[0][0]['episode']):selected.append((result,ep))
            if len(selected)==2:break
        body+='<h2>'+arm+': recorded movement</h2>'
        for result,ep in selected:
            rel='rollout-'+arm+'/episodes/'+result['episode']+'/'
            body+='<p>'+html.escape(result['instruction'])+' — success='+str(result['success'])+'</p><img width=448 src="'+rel+'movement.gif"><img width=360 src="'+rel+'trajectory.png">'
        body+='<h3>Offline validation action metrics</h3><pre>'+html.escape(json.dumps(r.get('offline_validation',{}),indent=2))+'</pre>'
    body+='<h2>Interpretation</h2><p>Compare raw-timestep with raw-episode for the weighting change; neutral-episode with raw-episode for preprocessing; neutral-timestep measures the combination. A single seed and one validation geometry are development evidence, not a robust result. A missing/failed job is not zero navigation success. Inspect images and per-action/startup metrics before choosing a revision. Freeze any chosen revision before a separate full test evaluation.</p>'
    (out/'report.html').write_text('<!doctype html><meta charset="utf-8"><style>body{font:17px/1.5 system-ui;max-width:1100px;margin:35px auto;padding:15px}td,th{padding:10px;border-bottom:1px solid #ddd}</style>'+body)
    write(out/'comparison-status.json',dict(state='complete',completed_arms=sum(r['status']['state']=='complete' for r in rows),expected_arms=4))


def worker(args):
    plan=read(args.plan)
    if plan['code_hash']!=code_hash():raise RuntimeError('Code changed since submission; use a new suite tag after finishing edits')
    out=Path(plan['output']);p=plan['paths'];cfg=plan['config']
    if args.stage=='compare':compare(plan);return
    sys.path.insert(0,str(REPO/'src'))
    if args.stage=='diagnostic':
        from mc_binding.vla_diagnostic import run
        run(p['demos'],p['rollout'],p['policy'],p['cache'],out/'diagnostic');return
    from mc_binding.vla import featurize,train,rollout
    if args.stage=='features':featurize(p['demos'],out/'features-neutral',dict(cfg,observation_mode='neutral_bands_v1'));return
    cfg=plan['arms'][args.arm]
    if args.stage=='train':
        cache=p['cache'] if args.arm.startswith('raw') else out/'features-neutral'
        train(cache,out/('policy-'+args.arm),cfg)
    elif args.stage=='rollout':
        render=Path(os.environ['MC_RENDER_ENV'])/'bin/python'
        rollout(p['dataset'],out/('policy-'+args.arm),out/('rollout-'+args.arm),str(render),cfg,True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('submit');p.add_argument('--tag',required=True);p.add_argument('--runs',default=os.environ.get('MC_BINDING_SCRATCH',f'/oscar/scratch/{os.environ.get("USER", "")}/zef-binding')+'/runs')
    for name,default in dict(dataset='binding-captures-24-6961501',demos='vla-demos-7045999',cache='vla-features-7050533',policy='vla-policy-7052010',rollout='vla-rollout-7052970').items():p.add_argument('--'+name,default=default)
    p.add_argument('--reviewed-captures',action='store_true');p.add_argument('--dry-run',action='store_true')
    p=sub.add_parser('worker');p.add_argument('--plan',required=True);p.add_argument('--stage',choices=['diagnostic','features','train','rollout','compare'],required=True);p.add_argument('--arm',default='none')
    args=parser.parse_args();submit(args) if args.command=='submit' else worker(args)

if __name__=='__main__':main()
