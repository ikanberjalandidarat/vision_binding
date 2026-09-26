"""Portable report of measured outputs; colored cells depict intervention scope."""
import base64
import html
import json
from pathlib import Path


def report(output):
    root = Path(output)
    esc = lambda x: html.escape(str(x))
    rows = [json.loads(line) for line in (root/'results.jsonl').read_text().splitlines()]
    body = ['<!doctype html><meta charset="utf-8"><title>Vision color-transfer pilot</title>',
        '<style>body{font:16px system-ui;max-width:1100px;margin:40px auto;padding:20px;color:#203044}table{border-collapse:collapse}td,th{padding:9px;border:1px solid #ccc}img{max-width:100%}figure{display:inline-block;margin:10px}code{background:#eee;padding:5px}</style>',
        '<h1>Spatial patches inside the vision transformer</h1>',
        '<p>Recipient image → vision blocks → visual merger → language decoder → color answer.</p>',
        '<p>At one selected vision block, copy donor hidden-state rows into recipient spatial-token rows, across all channels. All weights remain frozen. Each layer is tested separately. The question stays fixed between clean and patched runs.</p>',
        '<p>Orange cells show selected spatial tokens, not attention strengths. Masks use reviewed red/blue color evidence with ≥50% patch coverage; boundary tokens can include background. Both objects are queried in separate forward passes with the identical intervention.</p>',
        '<p>Target: donor rows at the target object. Other object: donor rows at the neighbor. Background: equal-count spatial control. Random: target-row noise matched to donor-delta norm before BF16 casting. Self: recipient rows copied back exactly.</p>',
        '<p>Exploratory fixed-camera color transfer; shape binding and camera invariance are not measured. Repeated sides and layouts are not independent scenes. The other-object control may contain a different token count.</p>']
    for family in sorted({r['family'] for r in rows if 'family' in r}):
        body.append('<h2>'+esc(family)+'</h2>')
        for path in sorted((root/'alignment'/family).glob('*.png')):
            src = base64.b64encode(path.read_bytes()).decode()
            body.append(f'<figure><figcaption>{esc(path.stem)}</figcaption><img src="data:image/png;base64,{src}"></figure>')
        body.append('<table><tr><th>Vision layer</th><th>Condition</th><th>Target side</th><th>Tokens</th><th>Left answer</th><th>Right answer</th><th>Target transferred</th><th>Neighbor preserved</th></tr>')
        for r in rows:
            if r.get('family')==family:
                fields=[r['layer'],r['condition'],['left','right'][r['target_side']],r['token_count'],*r['answers'],r['target_transferred'],r['neighbor_preserved']]
                body.append('<tr>'+''.join('<td>'+esc(v)+'</td>' for v in fields)+'</tr>')
        body.append('</table>')
    (root/'report.html').write_text('\n'.join(body))
