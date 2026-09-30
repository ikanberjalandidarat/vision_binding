"""Summarize saved no-patch destination calibration; no model execution."""
import html
import json
from pathlib import Path
from collections import Counter
from mc_binding.io import digest,atomic_json


def main():
    root=Path(__file__).resolve().parents[1]
    run=root/'runs/oscar/destination-diagnostic-6863843'
    dataset=root/'runs/oscar/vision-captures-replication-24-6852730'
    rows=json.loads((run/'diagnostics.json').read_text());manifest=json.loads((run/'manifest.json').read_text())
    data=json.loads((dataset/'manifest.json').read_text())
    assert manifest['dataset_hash']==digest(data)
    assert json.loads((run/'status.json').read_text())==dict(state='complete',rows=288)
    assert len(rows)==len({(r['family'],r['context'],r['mode'],r['prompt']) for r in rows})==288
    summary={}
    for mode in ('color_control','original_action','direct_spatial'):
        a=[r for r in rows if r['mode']==mode];assert len(a)==96
        assert all(r['correct']==(r['parsed']==r['expected']) for r in a)
        summary[mode]=dict(n=len(a),correct=sum(r['correct'] for r in a),answers=dict(Counter(str(r['parsed']) for r in a)))
    atomic_json(run/'audit.json',dict(dataset_hash=digest(data),counts=summary,scope='Prompt calibration on existing scenes; no patches or movement.'))
    parts=['<!doctype html><meta charset="utf-8"><title>Destination prompt diagnostic</title><style>body{font:17px system-ui;max-width:1050px;margin:30px auto;line-height:1.6}td,th{padding:10px;border-bottom:1px solid #ddd}img{width:48%}table{border-collapse:collapse}</style>',
           '<h1>Same images, different question wording</h1><p>No activation patches, training or movement in this diagnostic.</p><table><tr><th>Question type</th><th>Correct</th><th>Answers</th></tr>']
    for mode,s in summary.items():parts.append(f'<tr><td>{mode}</td><td>{s["correct"]}/{s["n"]}</td><td>{html.escape(str(s["answers"]))}</td></tr>')
    parts+=['</table><p>For side answers: 0 = LEFT, 1 = RIGHT. Both prompt modes had 48 LEFT and 48 RIGHT correct labels. The original prompt nearly always chose LEFT.</p>',
            '<h2>Recorded example: f0000</h2><p>Original (left image): blue pillar / yellow stairs. Donor (right image): yellow pillar / blue stairs.</p>']
    for name in ('000000','000003'):parts.append(f'<img src="../{dataset.name}/frames/{name}.png">')
    parts.append('<table><tr><th>Image</th><th>Prompt</th><th>Answer</th><th>Expected</th></tr>')
    for r in rows:
        if r['family']=='f0000':
            expected=r['expected'] if r['mode']=='color_control' else ('LEFT','RIGHT')[r['expected']]
            parts.append('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in (r['context'],r['prompt'],r['raw'],expected))+'</tr>')
    parts+=['</table><p>The direct prompt changed action wording, sentence structure and answer capitalization together. This establishes sensitivity to prompt formulation, not which individual change caused it. Selecting it on these scenes makes subsequent same-scene experiments exploratory; fresh scenes are needed for held-out validation.</p>',
            '<p>The original movement-choice gate remains intact. Next: use the explicit direct_spatial configuration, require correct unpatched choices again, and only then evaluate V patches. Movement replay is a separate stage.</p><a href="diagnostics.json">All 288 probes</a> · <a href="../../../docs/research-guide.html">Research guide</a>']
    (run/'report.html').write_text('\n'.join(parts))
    print(summary)
if __name__=='__main__':main()
