"""Submit a real-simulator smoke gate, then nine bounded RL runs (two concurrent)."""
import argparse,json,os,re,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--tag',required=True);p.add_argument('--dry-run',action='store_true');a=p.parse_args()
if not re.fullmatch('[a-zA-Z0-9_-]+',a.tag):p.error('Simple tag required')
repo=Path(__file__).resolve().parents[1];scratch=Path(os.environ.get('MC_BINDING_SCRATCH',f'/oscar/scratch/{os.environ["USER"]}/zef-binding'));out=scratch/'runs'/('rl-navigation-'+a.tag)
if out.exists():p.error('Use a fresh tag; old results are preserved')
if not a.dry_run:(out/'logs').mkdir(parents=True)
base=['sbatch','--parsable','--chdir='+str(repo)]
script=str(repo/'scripts/slurm/rl_navigation.sbatch');jobs={};commands=[]
for stage in ('smoke','sweep'):
 cmd=base+['--output='+str(out/'logs'/(stage+'-%A_%a.out'))]
 if stage=='smoke':cmd+=['--time=01:00:00']
 else:cmd+=['--array=0-8%2','--dependency=afterok:'+jobs['smoke'],'--kill-on-invalid-dep=yes']
 cmd+=[script,str(out),stage];commands.append(cmd)
 jid=str(len(jobs)+1) if a.dry_run else subprocess.run(cmd,check=True,capture_output=True,text=True).stdout.strip().split(';')[0]
 if not jid.isdigit():raise RuntimeError('Unexpected submission response')
 jobs[stage]=jid
 if not a.dry_run:(out/'submission.json').write_text(json.dumps(dict(jobs=jobs,commands=commands),indent=2))
print(json.dumps(dict(output=str(out),jobs=jobs,commands=commands),indent=2))
