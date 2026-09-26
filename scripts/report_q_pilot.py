"""Build a portable illustrated HTML report from a Q pilot and its hashed captures."""
import argparse
import base64
import html
import json
from pathlib import Path
from mc_binding.capture_pairs import load_pilot
from mc_binding.io import digest, file_hash
from q_report_explainer import explain
from q_report_figure import patch_figures, figure_panel, network_panel


def build_report(run, dataset, output):
    run, dataset, output = Path(run), Path(dataset), Path(output)
    manifest = json.loads((run/'manifest.json').read_text())
    data = load_pilot(dataset)
    if manifest.get('schema_version') != 'q_color_pilot_v1' or digest(data) != manifest['dataset_hash']:
        raise ValueError('Run and capture manifest do not match')
    rows = [r for p in sorted((run/'families').glob('*.json')) for r in json.loads(p.read_text())]
    if len({r['trial_key'] for r in rows}) != len(rows):
        raise ValueError('Duplicate trial keys')
    images = {}
    for r in data['records']:
        path = dataset/r['image']
        if file_hash(path) != r['image_sha256']:
            raise ValueError('Image hash changed')
        images[r['image_sha256']] = 'data:image/png;base64,'+base64.b64encode(path.read_bytes()).decode()
    esc = lambda s: html.escape(str(s), quote=True)
    def picture(key, prompt):
        return f'<figure><img src="{images[key]}" alt="Saved model input"><figcaption>{esc(prompt)}</figcaption></figure>'
    interventions = [r for r in rows if r['condition'] in ('q', 'random_q')]
    def key(r):
        return r['scene_family_id'], r['side'], r['content_change'], r['address_flip']
    controls = {key(r): r for r in interventions if r['condition'] == 'random_q'}
    cards = []
    for r in interventions:
        if r['condition'] != 'q':
            continue
        c = controls[key(r)]
        primary = r['content_change'] and r['address_flip']
        target = r['recipient_objects'][r['side']]['color']
        selected = 1-r['side'] if r['address_flip'] else r['side']
        other = r['recipient_objects'][selected]['color']
        donor = r['donor_objects'][selected]['color']
        cards.append(f'''<article data-primary="{str(primary).lower()}">
<h2>{esc(r['scene_family_id'])} · recipient {'left' if r['side']==0 else 'right'} ·
{'disjoint' if r['content_change'] else 'same'} colors · {'opposite' if r['address_flip'] else 'same'} instruction</h2>
<div class="images">{picture(r['image_sha256'], 'Recipient: '+r['prompt'])}{picture(r['donor_image_sha256'], 'Donor: '+r['donor_prompt'])}</div>
<table><tr><th>Clean recipient answer</th><th>Recipient answer after donor-Q patch</th><th>Random Q answer</th></tr>
<tr><td>{esc(r['clean_raw'])}</td><td>{esc(r['raw'])}</td><td>{esc(c['raw'])}</td></tr></table>
<p>Original target: <b>{esc(target)}</b>. Recipient at donor-selected side: <b>{esc(other)}</b>.
Donor-selected color: <b>{esc(donor)}</b>.</p>
<p>Strict outcome: Q = {esc(r['scores']['strict']['outcome'])}; random Q = {esc(c['scores']['strict']['outcome'])}.</p>
<details><summary>Patch details</summary><pre>{esc(json.dumps({'trial_key':r['trial_key'], 'scope':r['patch_scope'], 'layers':[p['layer'] for p in r['patches']], 'recipient_image_sha256':r['image_sha256'], 'donor_image_sha256':r['donor_image_sha256']},indent=2))}</pre></details></article>''')
    primary_q = [r for r in interventions if r['condition']=='q' and r['address_flip'] and r['content_change']]
    primary_c = [controls[key(r)] for r in primary_q]
    count = lambda rs: sum(r['scores']['strict']['flags']['recipient_at_donor_address'] for r in rs)
    clean = [r for r in rows if r['condition']=='clean']
    self_rows = [r for r in rows if r['condition']=='self_q']
    config = manifest['config']
    figures = patch_figures(rows, manifest, images)
    explanation = network_panel() + figure_panel(figures) + explain(rows, manifest, images)
    page = f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Q color pilot — illustrated results</title><style>
body{{font:16px system-ui,sans-serif;background:#f3f5f7;color:#17212b;margin:0 auto;padding:28px;max-width:1080px}}
h1{{margin-bottom:8px}}h2{{font-size:18px}}article,header,section{{background:white;padding:22px;border-radius:12px;margin-bottom:20px}}
.images{{display:grid;grid-template-columns:1fr 1fr;gap:18px}}figure{{margin:0}}img{{width:100%;height:auto}}figcaption{{padding:10px 0;line-height:1.5}}
table{{width:100%;border-collapse:collapse}}th,td{{padding:12px;text-align:left;border:1px solid #ddd}}td{{font-size:22px;font-weight:600}}
pre{{white-space:pre-wrap;overflow-wrap:anywhere}}select{{padding:8px;font:inherit}}.note{{color:#495563}}@media(max-width:650px){{.images{{grid-template-columns:1fr}}body{{padding:12px}}}}
</style><header><h1>Color-only Q-patching pilot</h1>
<p>Frozen-model inference; no training. Images below are the actual cleaned inputs, embedded for offline viewing. Patches alter internal activations, not pixels.</p>
<p><b>Opposite instruction + disjoint colors:</b> Q transferred recipient selection on {count(primary_q)}/{len(primary_q)} probes;
random Q on {count(primary_c)}/{len(primary_c)}.</p>
<p>Clean color probes: {sum(r['correct'] for r in clean)}/{len(clean)}. Exact self-Q matches: {sum(r['exact_match'] for r in self_rows)}/{len(self_rows)}.</p>
<p class="note">These are repeated target sides/layouts, not independent scene samples. This diagnostic does not establish object identity transfer, localization, or color–type binding. No confidence intervals or generalization claims are warranted.</p>
<p>Model: {esc(config['model_id'])}; precision: {esc(config['dtype'])}; decoder layers: {esc(config['layers'])}.</p>
<p><a href="#how">How the intervention works: interactive walkthrough ↓</a></p></header>
{explanation}
<section><h2>Recorded trial cards</h2><p>Primary shows opposite-instruction/disjoint-color trials only. All contexts also shows the three control cells.</p><label>Show <select id="filter"><option value="primary">Primary diagnostic</option><option value="all">All donor contexts</option></select></label></section>
{''.join(cards)}<section><h2>Clean recognition answers</h2><pre>{esc(chr(10).join(f"{r['context']} / {r['kind']} / side {r['side']}: {r['raw']} (expected {r['expected_color']})" for r in clean))}</pre>
<p>Dataset hash: {esc(manifest['dataset_hash'])}. Only atomically completed configuration files are included.</p></section>
<script>const f=document.getElementById('filter');function show(){{document.querySelectorAll('article').forEach(a=>a.hidden=f.value==='primary'&&a.dataset.primary!=='true')}}f.addEventListener('change',show);show();</script></html>'''
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(page)
    for name, svg, side in figures:
        (output.parent/name).write_text(svg)
    return len(cards)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', required=True)
    p.add_argument('--dataset', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    print(f'{build_report(a.run, a.dataset, a.output)} illustrated contexts written to {a.output}')
