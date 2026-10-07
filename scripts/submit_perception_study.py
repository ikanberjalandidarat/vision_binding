"""Submit extraction, three CPU readouts, and a report. Never trains the vision backbone."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--tag',required=True);p.add_argument('--dataset',default='binding-captures-24-6961501');p.add_argument('--reviewed-captures',action='store_true');p.add_argument('--dry-run',action='store_true');a=p.parse_args()
    if not a.reviewed_captures:p.error('--reviewed-captures is required')
    if not re.fullmatch(r'[A-Za-z0-9_-]+',a.tag):p.error('Use a simple tag')
    repo=Path(__file__).resolve().parents[1];scratch=Path(os.environ.get('MC_BINDING_SCRATCH',f'/oscar/scratch/{os.environ["USER"]}/zef-binding'));root=scratch/'runs'/('perception-study-'+a.tag)
    dataset=scratch/'runs'/a.dataset
    if not (dataset/'manifest.json').is_file():p.error('Dataset manifest missing')
    if root.exists():p.error('Output already exists; use a new tag (preserve previous results)')
    if not a.dry_run:(root/'logs').mkdir(parents=True)
    jobs={};commands={}
    def submit(name,args,deps=(),gpu=False,after='afterok'):
        cmd=['sbatch','--parsable','--kill-on-invalid-dep=yes','--partition='+('gpu' if gpu else 'batch'),'--nodes=1','--cpus-per-task=4','--mem=48G','--time=02:00:00','--chdir='+str(repo),'--job-name=perception-'+name,'--output='+str(root/'logs'/(name+'-%j.out'))]
        if gpu:cmd+=['--gres=gpu:1']
        if deps:cmd+=['--dependency='+after+':'+':'.join(deps)]
        cmd+=[str(repo/'scripts/slurm/perception_study.sbatch')]+args;commands[name]=cmd
        if a.dry_run:jid=str(len(jobs)+1)
        else:
            jid=subprocess.run(cmd,check=True,capture_output=True,text=True).stdout.strip().split(';')[0]
            if not jid.isdigit():raise RuntimeError('Unexpected Slurm response '+jid)
        jobs[name]=jid
        if not a.dry_run:(root/'submission.json').write_text(json.dumps(dict(jobs=jobs,commands=commands),indent=2))
        return jid
    extraction=submit('extract',['extract','--dataset',str(dataset),'--output',str(root/'features'),'--reviewed-captures'],gpu=True)
    children=[submit(m,['train','--cache',str(root/'features'),'--output',str(root/m),'--mode',m],[extraction]) for m in ('4x8','8x16','native')]
    submit('report',['report','--output',str(root)],children,after='afterany')
    print(json.dumps(dict(output=str(root),jobs=jobs,commands=commands if a.dry_run else str(root/'submission.json')),indent=2))

if __name__=='__main__':main()
