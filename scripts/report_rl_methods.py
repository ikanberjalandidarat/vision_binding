"""Compare completed method runs; missing arms never count as zero success."""
import argparse,html,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('suite',type=Path);p.add_argument('--expected',type=int,default=36);a=p.parse_args()
rows=[]
for run in sorted(a.suite.glob('seed-*')):
 status=json.loads((run/'status.json').read_text()) if (run/'status.json').exists() else {'state':'missing'}
 row=dict(run=run.name,state=status['state'])
 if status['state']=='complete':
  data=json.loads((run/'summary.json').read_text());meta=json.loads((run/'manifest.json').read_text());history=json.loads((run/'training.json').read_text());updates=json.loads((run/'updates.json').read_text())
  row.update(method=meta['settings'].get('method','ppo'),mode=meta['settings']['mode'],seed=meta['settings']['seed'],transitions=data['training_transitions'],results=data['results'],extrinsic_return=sum(r.get('extrinsic_return',r['return_sum']) for r in history),intrinsic_return=sum(r.get('intrinsic_return',0) for r in history),zero_variance_groups=sum(any(m.get('zero_variance',False) for m in u['minibatches']) for u in updates))
 rows.append(row)
complete=sum(r['state']=='complete' for r in rows)
(a.suite/'comparison-status.json').write_text(json.dumps(dict(state='complete' if complete==a.expected else 'incomplete',completed=complete,expected=a.expected),indent=2))
(a.suite/'comparison.json').write_text(json.dumps(rows,indent=2))
body=f'<h1>RL method comparison</h1><p>{complete}/{a.expected} completed. Same episode caps; actual transitions vary. Task success excludes intrinsic reward. Repeated seeds are not independent scenes.</p><table><tr><th>Run</th><th>State</th><th>Initial → final present (greedy)</th><th>Initial → final absent (greedy)</th><th>Train transitions</th><th>Task return / bonus</th><th>Tied GRPO groups</th></tr>'
for r in rows:
 cells=[f'<a href="{r["run"]}/report.html">{html.escape(r["run"])}</a>',html.escape(r['state'])]
 if r['state']=='complete':
  for present in (True,False):
   values=[]
   for phase in ('initial','final'):
    match=next(x for x in r['results'] if x['phase']==phase and x['action_mode']=='greedy' and x['present']==present)
    values.append(f'{match["successes"]}/{match["n"]}')
   cells.append(' → '.join(values))
  cells += [str(r['transitions']),f'{r["extrinsic_return"]:.2f} / {r["intrinsic_return"]:.2f}',str(r['zero_variance_groups'])]
 else:cells+=['—']*5
 body+='<tr>'+''.join('<td>'+c+'</td>' for c in cells)+'</tr>'
body+='</table><p>Full greedy and sampled breakdowns are in comparison.json. Linked reports contain actual movement GIFs and paths. Training return is not an arrival rate.</p>'
(a.suite/'comparison.html').write_text('<!doctype html><meta charset="utf-8"><style>body{font:16px system-ui;margin:30px}td,th{padding:12px;border-bottom:1px solid #ddd}</style>'+body)
print(a.suite/'comparison.html')
