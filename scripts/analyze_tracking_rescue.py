"""Summarize paired tracking evidence without rerunning Qwen."""
import json
from pathlib import Path
import html
from mc_binding.tracking_rescue import summarize, persistence

root=Path('runs/oscar/tracking-rescue-7184482')
out=root/'analysis';out.mkdir(exist_ok=True)
rows=json.loads((root/'evaluation/results.json').read_text())
assert summarize(rows)==json.loads((root/'evaluation/summary.json').read_text())
assert persistence(rows)==json.loads((root/'evaluation/persistence.json').read_text())
patches=json.loads((root/'evaluation/patch-results.json').read_text())
assert all(p['correct']==p['corrupt_correct'] for p in patches if p['control']=='self')
body='''<h1>Memory hurt selection; one donor-assisted rescue</h1>
<p>20 teacher episodes, 140 sampled observations. All four present goals occur in each of four validation families. Of 136 present observations, 100 pass the geometry filters; 36 are excluded. Four absent observations are static checks. Repeated observations are not independent scenes.</p>
<h2>Selection results</h2><table><tr><th>Condition</th><th>Frame-only</th><th>Original memory</th><th>Recovery memory</th></tr>'''
for stream,active,label in [('clean',False,'Clean: all eligible frames'),('band35',True,'35% band: disrupted frames only'),('band70',True,'70% band: disrupted frames only')]:
 body+='<tr><td>'+label+'</td>'
 for method in ['frame_only','gated_template','recovery_memory']:
  rr=[r for r in rows if r['stream']==stream and r['active']==active and r['method']==method and r['target_eligible']]
  body+=f'<td>{sum(r["correct"] for r in rr)}/{len(rr)}</td>'
 body+='</tr>'
body+='''</table><p>Recovery memory corrected zero frame-only errors. On clean frames it turned 54 correct selections into abstentions. Once acquired, the memory rule requires a similarity match even when semantic recognition is correct; a failed match blocks that correct answer. The saved results do not contain cosine similarities, so they cannot distinguish a failed absolute threshold from an ambiguous margin. Zero wrong-object streaks therefore do not mean successful tracking: this memory mostly loses targets.</p>
<h2>Only one corruption-induced failure</h2><p>At e00177, step 24, the blue-pillar score falls to 0.04955, just below the 0.05 threshold. The other 70% disrupted error was already wrong on the clean image. The 35% condition creates no eligible rescue failures.</p><table><tr><th>Patch</th><th>Band donor score</th><th>Outside-band control score</th></tr>'''
for layers in [[30],[31],[30,31]]:
 pp=[p for p in patches if p['rescue_eligible'] and p['layers']==layers]
 scores={p['control']:max(p['scores']) for p in pp}
 body+=f'<tr><td>V{html.escape(str(layers))}</td><td>{scores["clean_donor"]:.6f}</td><td>{scores["other_region"]:.6f}</td></tr>'
body+='''</table><p>V30 and V30+31 restore selection, while V31 alone does not. However, the V30 outside-band control also restores selection (0.05903). The joint patch outperforms its spatial control on this single failure, but one threshold-near case cannot establish generality or binding specificity. Self-patches preserve decisions; no scored correct corrupted choices were harmed.</p>
<h2>Read the sequence</h2><p>Yellow boxes show recovery-memory selection, not patched tokens. These GIFs do not show donor-patched predictions. At step 16 the memory resets in all streams; at step 24 strong corruption prevents reacquisition; step 28 reacquires. Motion is prerecorded teacher control.</p>'''
for eid in ['e00177','e00049','e00051']:
 path=root/'evaluation'/f'{eid}.gif'
 body+=f'<h3>{eid}</h3><img src="../evaluation/{eid}.gif"><p><code>{path}</code></p>'
body+='''<h2>Next bounded test</h2><ol><li>Keep a confident per-frame selection; consult memory only on semantic dropout. Log best cosine, runner-up margin, acquisition and reset causes. Test this rule first on clean sequences, where memory should not suppress working recognition.</li><li>Develop disruptions and thresholds on training families, then freeze them for held-out evaluation. Seek multiple clean-correct failures across goals; do not tune repeatedly on this validation panel.</li><li>Repeat V30/V31 rescue with score changes, matched spatial controls and independent failed episodes. Add disappearance and competing-object sequences before claiming memory safety.</li></ol><p>No new navigation or training improvement is established here.</p>'''
(out/'explained.html').write_text('<!doctype html><meta charset="utf-8"><title>Tracking rescue analysis</title><style>body{font:17px system-ui;line-height:1.6;max-width:1200px;margin:40px auto;padding:20px}td,th{padding:12px;border-bottom:1px solid #ddd}img{max-width:100%}code{overflow-wrap:anywhere}</style>'+body)
print(out/'explained.html')
