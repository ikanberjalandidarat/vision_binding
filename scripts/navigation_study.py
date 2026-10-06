#!/usr/bin/env python3
"""Multi-seed navigation and startup selection; one submission with retries."""
import argparse,hashlib,json,os,re,subprocess,sys
from datetime import datetime,timezone
from pathlib import Path
REPO=Path(__file__).resolve().parents[1]

def read(p):return json.loads(Path(p).read_text())
def write(p,x):
    p=Path(p);q=p.with_suffix('.tmp');q.write_text(json.dumps(x,indent=2));q.replace(p)
def code_hash():
    h=hashlib.sha256()
    for p in sorted((REPO/'src').rglob('*.py'))+[Path(__file__),REPO/'scripts/slurm/navigation_study.sbatch']:
        h.update(str(p.relative_to(REPO)).encode());h.update(p.read_bytes())
    return h.hexdigest()

def graph(seeds):
    stages=[dict(key='preflight',stage='preflight',deps=[])]
    arms={}
    for seed in seeds:
        for method in ('timestep','startup-turn'):
            arm=f'{method}-s{seed}';arms[arm]=dict(seed=seed,split_seed=731,loss_weighting='timestep',observation_mode='raw',startup_weight=5. if method=='startup-turn' else 1.,turn_weight=2. if method=='startup-turn' else 1.,report_test_metrics=False,evaluation_split='validation',evaluation_episodes=64)
            stages.append(dict(key='train-'+arm,stage='train',arm=arm,deps=[('afterok','preflight')]))
        stages.append(dict(key=f'selector-s{seed}',stage='selector',seed=seed,deps=[('afterok','preflight')]))
    previous=None
    for arm in arms:
        deps=[('afterok','train-'+arm)]
        if previous:deps.append(('afterany',previous))
        previous='rollout-'+arm;stages.append(dict(key=previous,stage='rollout',arm=arm,deps=deps))
    stages.append(dict(key='compare',stage='compare',deps=[('afterany',s['key']) for s in stages]))
    return arms,stages

def build(args):
    if not args.reviewed_captures:raise ValueError('Require --reviewed-captures')
    if not re.fullmatch(r'[A-Za-z0-9_-]+',args.tag):raise ValueError('Invalid tag')
    if len(set(args.seeds))!=len(args.seeds) or not args.seeds:raise ValueError('Use distinct seeds')
    runs=Path(args.runs).resolve();out=runs/('navigation-study-'+args.tag)
    if out.exists():raise ValueError('Use a new tag or retry the existing suite')
    paths={k:str(runs/getattr(args,k)) for k in ('dataset','cache','demos')}
    for k,p in paths.items():
        if runs not in Path(p).resolve().parents or not Path(p).is_dir():raise ValueError('Invalid input '+p)
        if k!='dataset' and read(Path(p)/'status.json')['state']!='complete':raise ValueError('Incomplete input '+p)
    fm=read(Path(paths['cache'])/'manifest.json')
    if fm['config'].get('observation_mode','raw')!='raw':raise ValueError('Use original raw cache')
    if hashlib.sha256((Path(paths['demos'])/'episodes.json').read_bytes()).hexdigest()!=fm['episodes_hash']:raise ValueError('Cache/demo mismatch')
    dataset_hash=hashlib.sha256(json.dumps(read(Path(paths['dataset'])/'manifest.json'),sort_keys=True,separators=(',',':')).encode()).hexdigest()
    if dataset_hash!=fm['demos_manifest']['dataset_hash']:raise ValueError('Dataset/cache mismatch')
    cfg=dict(read(REPO/'configs/vla_pilot.json'),selector_epochs=80);arms,stages=graph(args.seeds)
    # Encoder settings must come from the actual cache rather than a drifting config.
    for k in ('model_id','revision','dtype','use_fast','min_pixels','max_pixels','readout_layer'):cfg[k]=fm['config'][k]
    return dict(output=str(out),paths=paths,config=cfg,arms=arms,stages=stages,code_hash=code_hash(),seeds=args.seeds,metric_version='navigation_v2',scope='Three-seed validation development study; fixed split seed 731; no held-out test optimization; no activation patching or on-policy corrections in this suite')

def schedule(plan,dry=False,retry=False):
    out=Path(plan['output']);reused=set();archive=[]
    if plan['code_hash']!=code_hash():raise RuntimeError('Code changed since submission; use a new study')
    if retry:
        if not dry:
            known=set()
            for p in (out/'attempts').glob('*/submission.json'):known.update(read(p)['jobs'].values())
            q=subprocess.run(['squeue','--noheader','--user',os.environ['USER'],'--format=%A'],check=True,capture_output=True,text=True)
            if known.intersection(q.stdout.split()):raise RuntimeError('Study still active; wait before retrying')
        for s in plan['stages']:
            folder=out/s['key'];status=read(folder/'status.json') if (folder/'status.json').exists() else {}
            if s['stage'] not in ('preflight','compare') and status.get('state')=='complete':
                if s['stage'] in ('train','selector'):
                    file='policy.pt' if s['stage']=='train' else 'selector.pt'
                    if hashlib.sha256((folder/file).read_bytes()).hexdigest()!=read(folder/'manifest.json')['checkpoint_sha256']:raise ValueError('Checkpoint changed')
                reused.add(s['key'])
            elif folder.exists():archive.append(folder)
    attempt=out/'attempts'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');jobs={};commands=[]
    if not dry:
        out.mkdir(exist_ok=True);(out/'logs').mkdir(exist_ok=True);(attempt/'archived').mkdir(parents=True)
        if not retry:write(out/'plan.json',plan)
        for folder in archive:folder.rename(attempt/'archived'/folder.name)
    def save(error=None):
        write(attempt/'submission.json',dict(jobs=jobs,commands=commands,reused=sorted(reused),error=error))
    try:
        for s in plan['stages']:
            if s['key'] in reused:continue
            gpu=s['stage']=='rollout';cmd=['sbatch','--parsable','--kill-on-invalid-dep=yes','--partition='+('gpu' if gpu else 'batch'),'--nodes=1','--cpus-per-task=4','--mem=48G','--time=03:00:00','--job-name=nav-'+s['key'],'--chdir='+str(REPO),'--output='+str(out/'logs'/(s['key']+'-%j.out'))]
            if gpu:cmd+=['--gres=gpu:1']
            deps=[kind+':'+jobs[key] for kind,key in s['deps'] if key not in reused]
            if retry and s['stage']=='rollout':deps.append('afterok:'+jobs['preflight'])
            if deps:cmd+=['--dependency='+','.join(deps)]
            cmd += [str(REPO/'scripts/slurm/navigation_study.sbatch'),str(out/'plan.json'),s['key']]
            if dry:jid=str(800000+len(jobs))
            else:
                r=subprocess.run(cmd,capture_output=True,text=True,check=True);jid=r.stdout.strip().split(';')[0]
                if not jid.isdigit():raise RuntimeError('Unexpected sbatch response: '+r.stdout)
            jobs[s['key']]=jid;commands.append(cmd)
            if not dry:save()
    except BaseException as e:
        if not dry:save(str(e))
        raise
    print(json.dumps(dict(output=str(out),reused=sorted(reused),jobs=jobs,commands=commands if dry else str(attempt/'submission.json')),indent=2))

def worker(plan,key):
    if plan['code_hash']!=code_hash():raise RuntimeError('Code hash changed')
    s=next(s for s in plan['stages'] if s['key']==key);out=Path(plan['output']);target=out/key;cfg=dict(plan['config']);paths=plan['paths']
    if s['stage']=='preflight':
        target.mkdir();write(target/'status.json',{'state':'running'})
        try:
            for name,imports in [('model','import bz2,torch,torchvision; from transformers import AutoProcessor,Qwen2VLForConditionalGeneration'),('render','import bz2; from minestudio.simulator import MinecraftSim')]:
                py=Path(os.environ['MC_BINDING_ENV' if name=='model' else 'MC_RENDER_ENV'])/'bin/python';subprocess.run([str(py),'-I','-c',imports],check=True)
            subprocess.run([os.environ['MC_RENDER_ENV']+'/bin/java','-version'],check=True);write(target/'status.json',{'state':'complete'})
        except BaseException as e:write(target/'status.json',{'state':'error','error':str(e)});raise
        return
    if s['stage']=='compare':compare(plan);return
    sys.path.insert(0,str(REPO/'src'))
    if s['stage']=='selector':
        from mc_binding.vla_selection import train
        train(paths['cache'],paths['demos'],target,dict(cfg,seed=s['seed'],split_seed=731));return
    cfg.update(plan['arms'][s['arm']])
    if s['stage']=='train':
        from mc_binding.vla import train
        train(paths['cache'],target,cfg)
    else:
        from mc_binding.vla import rollout
        rollout(paths['dataset'],out/('train-'+s['arm']),target,os.environ['MC_RENDER_ENV']+'/bin/python',cfg,True)

def compare(plan):
    import html
    out=Path(plan['output']);target=out/'compare';target.mkdir(exist_ok=True);rows=[];selectors=[]
    for arm in plan['arms']:
        p=out/('rollout-'+arm);r=dict(arm=arm,status=read(p/'status.json') if (p/'status.json').exists() else {'state':'missing/blocked'})
        if r['status']['state']=='complete':
            r['summary']=read(p/'summary.json');results=read(p/'results.json');r['immediate_present_stops']=sum(x.get('immediate_stop',False) and x['goal_present'] for x in results)
            r['timeouts']=sum(x.get('timeout',False) for x in results)
        rows.append(r)
    for seed in plan['seeds']:
        p=out/f'selector-s{seed}'
        selectors.append(dict(seed=seed,validation=read(p/'validation.json')['summary'] if (p/'status.json').exists() and read(p/'status.json')['state']=='complete' and (p/'validation.json').exists() else None))
    write(target/'summary.json',dict(navigation=rows,selection=selectors,scope=plan['scope']))
    body='<h1>Navigation and selection across seeds</h1><p>'+html.escape(plan['scope'])+'</p><p>Absent success now requires immediate refusal before any action. Arrival requires STOP within 0.8 blocks of the correct approach waypoint. Compare per-seed results; do not pool repeated scenes as independent samples.</p><table><tr><th>Arm</th><th>Status</th><th>Present arrival</th><th>Immediate absence refusal</th><th>False startup stops</th><th>Timeouts</th></tr>'
    for r in rows:
        rates={x['goal_present']:f'{x["successes"]}/{x["n"]}' for x in r.get('summary',{}).get('by_presence',[])}
        body+='<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in (r['arm'],r['status']['state'],rates.get(True,'—'),rates.get(False,'—'),r.get('immediate_present_stops','—'),r.get('timeouts','—')))+'</tr>'
    body+='</table><h2>Independent startup selector</h2><p>Predict one of four screen-order regions or absent, using cached RGB features and instruction. Coordinates/boxes are labels only. No-image/color-word/shape-word ablations keep the original labels; they measure dependency on inputs but are not ViT head interventions.</p><pre>'+html.escape(json.dumps(selectors,indent=2))+'</pre><h2>Movement and attribution reports</h2>'
    for r in rows:
        if r['status']['state']=='complete':body+=f'<p><a href="../rollout-{r["arm"]}/report.html">{r["arm"]}: Minecraft GIFs, trajectories, attention and action Grad-CAM</a></p>'
    body+='<p>Next gate: establish clean region selection and arrival across seeds before self/donor/random head interventions. Use disjoint training scenes for any future on-policy teacher corrections. A future test run must freeze the chosen revision first.</p>'
    (target/'report.html').write_text('<!doctype html><meta charset="utf-8"><style>body{font:17px system-ui;max-width:1100px;margin:30px auto}td,th{padding:10px;border-bottom:1px solid #ddd}pre{white-space:pre-wrap}</style>'+body)
    write(target/'status.json',dict(state='complete',completed_navigation=sum(r['status']['state']=='complete' for r in rows),expected_navigation=len(rows),completed_selectors=sum(s['validation'] is not None for s in selectors)))

def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('submit');p.add_argument('--tag',required=True);p.add_argument('--runs',default=f'/oscar/scratch/{os.environ.get("USER", "")}/zef-binding/runs');p.add_argument('--seeds',nargs='+',type=int,default=[731,732,733]);p.add_argument('--reviewed-captures',action='store_true');p.add_argument('--dry-run',action='store_true')
    for k,v in dict(dataset='binding-captures-24-6961501',demos='vla-demos-7045999',cache='vla-features-7050533').items():p.add_argument('--'+k,default=v)
    p=sub.add_parser('retry');p.add_argument('suite');p.add_argument('--dry-run',action='store_true')
    p=sub.add_parser('worker');p.add_argument('--plan',required=True);p.add_argument('--key',required=True)
    a=parser.parse_args()
    if a.command=='submit':schedule(build(a),a.dry_run)
    elif a.command=='retry':schedule(read(Path(a.suite)/'plan.json'),a.dry_run,True)
    else:worker(read(a.plan),a.key)

if __name__=='__main__':main()
