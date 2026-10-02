"""Build the local research map from saved results. No inference or training."""
import json
import html
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
RUNS=ROOT/'runs/oscar'

def read(p):return json.loads(p.read_text())
def link(run,file='report.html'):
    return f'../runs/oscar/{run}/{file}'
def image(run,file,caption):
    assert (RUNS/run/file).exists(), (run,file)
    return f'<figure><a href="{link(run,file)}"><img loading="lazy" src="{link(run,file)}" alt="{html.escape(caption)}"></a><figcaption>{html.escape(caption)} · <code>{run}/{file}</code></figcaption></figure>'

def build():
    trials=[];counts={};ledger=[]
    for kind in ('residual','q','k','v'):
        run=f'vision-late-{kind}-vision-captures-04'
        status=read(RUNS/run/'status.json');assert status['state']=='complete'
        manifest=read(RUNS/run/'manifest.json');assert manifest['config']['vision_patch_kind']==kind
        rows=[json.loads(s) for s in (RUNS/run/'results.jsonl').read_text().splitlines()]
        assert len(rows)==status['rows']
        selected=[r for r in rows if r['condition']=='target']
        counts[kind]=f"{sum(r['target_transferred'] and r['neighbor_preserved'] for r in selected)}/{len(selected)}"
        trials.extend({**r,'kind':kind,'run':run} for r in selected if r['family']=='f0000')
        ledger.append(dict(run=run,status=status,dataset_hash=manifest['dataset_hash'],counts=counts[kind]))
    assert len({r['dataset_hash'] for r in ledger})==1
    sweep=read(RUNS/'vision-sweep-01/summary.json')
    rows=[r for r in sweep['counts'] if r['condition']=='target']
    sweep_count=f"{sum(r['specific_transfers'] for r in rows)}/{sum(r['n'] for r in rows)}"
    pair=read(RUNS/'vision-captures-04/manifest.json')['records']
    pair={r['context']:r for r in pair if r['scene_family_id']=='f0000' and r['kind']=='pair'}
    stages=[
      ('q-color-pilot-01','1 · Decoder Q','Final prompt-token Q at decoder layers 21–27, simultaneously. Tests selection of the other recipient object; primary strict diagnostic: 2/2. Different task and metric from the vision experiments.'),
      ('vision-pilot-01','2 · Vision pilot','Whole block-output vectors at selected object-image tokens. ViT layers 0, 7, 15, 23, 31; one layout. A coarse depth check.'),
      ('vision-sweep-01','3 · Vision depth sweep',f'Whole block-output vectors again. All 32 single ViT layers plus eight groups of four. Four layouts; {sweep_count} specific transfers over target trials. This produced eli5-layer-results.png.'),
      ('vision-late-residual-vision-captures-04','4 · Late whole-state follow-up',f'Same block-output intervention; ViT layers 24–31 plus five groups. Adds donor-versus-original answer scores. {counts["residual"]} specific transfers.'),
      ('vision-late-q-vision-captures-04','5a · Vision Q',f'Q slice inside vision attention, at selected object tokens; same 13 late patch sets. {counts["q"]} specific transfers.'),
      ('vision-late-k-vision-captures-04','5b · Vision K',f'K slice inside vision attention. {counts["k"]} specific transfers.'),
      ('vision-late-v-vision-captures-04','5c · Vision V',f'V slice inside vision attention. {counts["v"]} specific transfers; concentrated in sets containing layer 30.'),
    ]
    cards=''.join(f'<article><h3>{title}</h3><p>{desc}</p><a href="{link(run)}">Open original report</a><p><code>{run}</code></p></article>' for run,title,desc in stages)
    inventory=[]
    for p in sorted(RUNS.iterdir()):
        if not p.is_dir():continue
        if p.name.startswith('render-smoke'):role='Renderer setup / debugging; not an intervention result'
        elif 'captures' in p.name and not p.name.startswith('vision-late'):role='Input screenshots + annotations; not model predictions'
        elif p.name=='recognition-baseline-01':role='Earlier numerically invalid model baseline; do not pool with BF16 results'
        elif p.name=='recognition-baseline-bf16-01':role='BF16 recognition calibration'
        elif p.name=='vision-projection-comparison':role='Derived plots comparing existing runs; no new model run'
        else:role='Intervention outputs; see experiment sequence above'
        target=next((f for f in ('report.html','manifest.json','summary.json','smoke.json') if (p/f).exists()),None)
        name=f'<a href="{link(p.name,target)}">{p.name}</a>' if target else p.name
        inventory.append(f'<tr><td>{name}</td><td>{role}</td></tr>')
    body='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Minecraft binding — research guide</title>
<style>
body{font:17px/1.6 system-ui;color:#203044;background:#f6f8fb;margin:0}main{max-width:1160px;margin:auto;padding:28px}h1,h2,h3{line-height:1.2}h2{margin-top:55px}a{color:#075a9c}nav{display:flex;gap:20px;flex-wrap:wrap}article,.box,figure{background:white;border:1px solid #d6dfe6;border-radius:12px;padding:20px;margin:14px 0}figure img{width:100%;height:auto}figcaption,small{font-size:14px;color:#526171}code{overflow-wrap:anywhere;font-size:13px}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:12px}.warning{border-left:6px solid #d99425;padding:16px;background:#fff6e5}svg{width:100%;height:auto;background:white;border-radius:12px}table{border-collapse:collapse;width:100%;background:white}td,th{text-align:left;padding:10px;border-bottom:1px solid #d6dfe6}select{font-size:17px;padding:8px;margin:5px}.pictures{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}.pictures img{width:100%}.answer{font-size:20px;font-weight:600} @media(max-width:700px){.pictures{grid-template-columns:1fr}} 
</style><main>
<h1>What did we patch in Minecraft?</h1>
<p>A guide to the recorded experiments, their pictures, and the next behavioral tests. Generated from local run files; no new inference performed.</p>
<p><a href="research-plan.html"><b>Research roadmap: binding → causal mechanisms → closed-loop movement → trained memory and active sensing</b></a></p>
<nav><a href="#map">Model map</a><a href="#block">Inside a vision block</a><a href="#history">Experiment sequence</a><a href="#figure">Your layer-results PNG</a><a href="#trial">Compare an actual trial</a><a href="#next">Next steps</a><a href="#folders">Folder index</a></nav>
<div class="warning"><b>Three intervention locations, one model.</b> First we patched Q in the language decoder. Then we patched entire vision-block output vectors. Then we separately patched Q, K, and V inside those vision blocks. “Whole state” is an intervention, not the untouched baseline.</div>
<h2 id="map">1. Two different stacks of layers</h2>
<svg viewBox="0 0 1120 310" role="img" aria-label="Screenshot enters vision transformer, then merger, then language decoder with question tokens, then answer">
<defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0 0 L8 4 L0 8Z" fill="#53677c"/></marker></defs>
<g font-family="system-ui" text-anchor="middle" font-size="18">
<rect x="15" y="105" width="145" height="80" rx="12" fill="#e6edf4"/><text x="87" y="138">Minecraft</text><text x="87" y="163">screenshot</text>
<rect x="200" y="75" width="250" height="140" rx="12" fill="#d9f1ec"/><text x="325" y="110" font-weight="bold">VISION TRANSFORMER</text><text x="325" y="145">Blocks 0 … 31</text><text x="325" y="177">Image-token vectors</text>
<rect x="490" y="105" width="135" height="80" rx="12" fill="#e6edf4"/><text x="557" y="150">Merger</text>
<rect x="665" y="75" width="250" height="140" rx="12" fill="#e9dff7"/><text x="790" y="110" font-weight="bold">LANGUAGE DECODER</text><text x="790" y="145">Its own layer numbering</text><text x="790" y="177">Visual + text tokens</text>
<rect x="965" y="105" width="140" height="80" rx="12" fill="#e6edf4"/><text x="1035" y="150">Answer</text>
<text x="325" y="260" fill="#137260">Whole output, or Q / K / V patches</text><text x="790" y="260" fill="#7952a3">Earlier experiment: final-token Q</text>
<text x="790" y="30">Question / chat-template tokens</text>
</g><g stroke="#53677c" stroke-width="3" marker-end="url(#arrow)"><path d="M160 145 H195"/><path d="M450 145 H485"/><path d="M625 145 H660"/><path d="M915 145 H960"/><path d="M790 37 V70"/></g></svg>
<p><b>ViT layer 21 and decoder layer 21 are different modules.</b> Within the ViT, however, “whole-state layer 30” and “V layer 30” refer to the same block, at different points inside it. All numbering is zero-based.</p>
<h2 id="block">2. Zoom into ONE vision block L</h2>
<svg viewBox="0 0 1120 450" role="img" aria-label="QKV patches are inside attention; whole-state patch is after attention and MLP residual additions">
<g font-family="system-ui" text-anchor="middle" font-size="18">
<rect x="25" y="165" width="150" height="80" rx="10" fill="#e6edf4"/><text x="100" y="197">Incoming image</text><text x="100" y="222">token vectors X</text>
<rect x="220" y="120" width="275" height="170" rx="10" fill="#fff0d7"/><text x="357" y="152">Normalize → fused projection</text><text x="357" y="190" font-size="26" font-weight="bold">Q     K     V</text><text x="357" y="226">RoPE on Q/K → attention</text><text x="357" y="258">→ output projection</text>
<rect x="535" y="165" width="125" height="80" rx="10" fill="#e6edf4"/><text x="597" y="198">Add X</text><text x="597" y="223">→ Y</text>
<rect x="700" y="165" width="150" height="80" rx="10" fill="#e6edf4"/><text x="775" y="198">Normalize</text><text x="775" y="223">→ MLP</text>
<rect x="895" y="165" width="200" height="80" rx="10" fill="#d9f1ec"/><text x="995" y="198">Add Y</text><text x="995" y="223">Block output H[L]</text>
<text x="357" y="62" fill="#a06108" font-weight="bold">Q/K/V-only intervention</text><text x="357" y="90">Selected rows, ONE projection slice</text>
<text x="905" y="330" fill="#137260" font-weight="bold">Whole-state intervention</text><text x="905" y="359">Selected rows, ALL output channels</text>
<text x="550" y="420">Both copy donor activations at the SAME layer into the recipient run. Neither changes weights or pixels.</text>
</g><g stroke="#53677c" stroke-width="3" fill="none" marker-end="url(#arrow)"><path d="M175 205 H215"/><path d="M495 205 H530"/><path d="M660 205 H695"/><path d="M850 205 H890"/><path d="M100 165 V110 H597 V160"/><path d="M597 245 V285 H995 V250"/><path d="M357 95 V115"/><path d="M995 320 V250"/></g></svg>
<p>“Whole” means the complete hidden vector <b>at selected spatial token positions</b>—not every image token, every layer, or the whole network. Groups such as 30+31 install replacements at both blocks during one forward pass.</p>
<h2 id="history">3. What we did, in order</h2><div class="cards">CARDS</div>
<p>The decoder experiment asked about transferring <b>which recipient object is selected</b>. The vision experiments ask whether the <b>donor color is reported at a selected image location</b>. Their success fractions measure different things and should not be ranked against each other.</p>
<h2 id="figure">4. What does eli5-layer-results.png show?</h2>
<div class="box"><b>Intervention: WHOLE VISION-BLOCK OUTPUT.</b><br>Each single-layer bar is a separate run patching object-token output vectors after one ViT block. Group bars patch several block outputs in the same run. The language decoder then answers left-color and right-color questions separately.<br><br>A success requires both: target answer equals donor color AND neighbor answer equals recipient color. It does not mean better accuracy on the unchanged screenshot.</div>
SWEEPFIG
<p><a href="../runs/oscar/vision-sweep-01/analysis/explained.html">Original sweep visual guide</a> · <a href="../runs/oscar/vision-sweep-01/manifest.json">Exact sweep configuration</a></p>
<h2>5. Same ViT blocks, different intervention points</h2>
COMPAREFIG
<p>Every cell uses eight cases (four layouts × two target sides). The late whole-state total and Q/K/V totals each use 13 patch sets × eight cases = 104. The full sweep has 40 patch sets × eight = 320. They do not have the same denominator.</p>
<p><b>Q/K zero means zero generated-answer transfers under these tests.</b> It does not mean zero internal change or that Q/K are unnecessary. V success concentrates near layer 30. Whole-output patching happens later in the block and replaces a different representation, so it is not equivalent to editing Q+K+V together.</p>
<h2 id="trial">6. Compare an actual recorded trial</h2>
<p>Choose intervention, layer set and target. All answers below are read from saved results, not hypothetical illustrations. This explorer uses layout f0000.</p>
<label>Patch <select id="kind"><option value="residual">Whole block output</option><option value="q">Vision Q</option><option value="k">Vision K</option><option value="v">Vision V</option></select></label>
<label>Layer set <select id="layers"></select></label><label>Target <select id="side"><option value="0">LEFT arch</option><option value="1">RIGHT tower</option></select></label>
<div class="pictures"><figure><figcaption>1. Original screenshot</figcaption><img id="recipient" alt="Original screenshot"><p>Actual colors: LEFT blue, RIGHT red.</p></figure><figure><figcaption>2. Donor screenshot</figcaption><img id="donor" alt="Donor screenshot"><p>Actual colors: LEFT red, RIGHT blue.</p></figure><figure><figcaption>3. Recipient patch positions</figcaption><img id="overlay" alt="Recorded spatial-token overlay"><p>Orange marks vector locations. Pixels fed to the model stay unchanged.</p></figure></div>
<div id="answers" class="box"></div><p id="evidence"></p>
<h2 id="next">7. Where we are now</h2>
REPLICATIONFIG
<p>The first yellow/blue stairs/pillar configuration passed local capture inspection. Full capture job <b>6852730</b> was reported RUNNING by the user on September 30, 2026. This is a dated observation, not a live job monitor. No full replication results have been included here.</p>
<ol><li>Review 144 captures from 24 configurations.</li><li>Run V30, V31 and V30+31 replication with clean recognition and self-patch gates.</li><li>Test LEFT/RIGHT destination decisions with BOTH object regions patched.</li><li>Replay decisions with a known-coordinate movement controller and record arrivals.</li><li>Later: frame-aligned camera-motion and occlusion-rescue experiments.</li></ol>
<p>All recorded studies here used frozen weights. Training, autonomous navigation, faster inference, and motion/occlusion rescue are not established results. <a href="../scripts/REPLICATION_BEHAVIOR.md">Execution guide and stage limitations</a>.</p>
<h2 id="folders">8. What each local folder is for</h2><table><tr><th>Folder / evidence</th><th>Role</th></tr>INVENTORY</table>
<h2>Interpretation rules</h2><ul><li>Observed color answers are outputs, not a reconstructed picture of the model’s perception.</li><li>Clean baseline = no patch. Whole-state = a real donor intervention.</li><li>Repeated layers, targets and reverse color assignments are not independent worlds.</li><li>Answer-score probabilities cover fixed first-token spellings, not the probability of an entire answer.</li><li>Strong color transfer does not by itself demonstrate shape binding, necessity, improved gameplay, or a unique color circuit.</li></ul>
<p>Evidence: each linked run’s manifest, status, results.jsonl and report. Implementation: <a href="../src/mc_binding/q_pilot.py">decoder experiment</a>, <a href="../src/mc_binding/vision_pilot.py">vision runner</a>, <a href="../src/mc_binding/vision_hooks.py">exact vision hooks</a>. Regenerate with <code>python3 scripts/build_research_guide.py</code>.</p>
<script>
const trials=TRIALDATA;
const select=document.getElementById('layers');
const sets=[...new Set(trials.map(r=>JSON.stringify(r.layer_set)))].map(JSON.parse).sort((a,b)=>(a.length>1)-(b.length>1)||a[0]-b[0]||a.length-b.length||a[a.length-1]-b[b.length-1]);
for(const s of sets){const o=document.createElement('option');o.value=JSON.stringify(s);o.textContent=s.join(' + ');select.append(o)}
select.value='[30]';
const el=id=>document.getElementById(id);
function update(){const r=trials.find(r=>r.kind===el('kind').value&&JSON.stringify(r.layer_set)===select.value&&r.target_side===Number(el('side').value));
el('recipient').src='../runs/oscar/vision-captures-04/'+r.recipient_image;
el('donor').src='../runs/oscar/vision-captures-04/'+r.donor_image;
el('overlay').src='../runs/oscar/'+r.run+'/alignment/f0000/patch-target-target'+r.target_side+'.png';
el('answers').replaceChildren();const p=document.createElement('p');p.className='answer';p.textContent='Recorded answers: LEFT = '+r.answers[0]+'; RIGHT = '+r.answers[1];el('answers').append(p);
const detail=document.createElement('p');detail.textContent='Target transferred: '+r.target_transferred+'. Neighbor kept original color: '+r.neighbor_preserved+'. '+r.token_count+' spatial tokens patched. '+(r.target_transferred&&r.neighbor_preserved?'Both criteria pass.':'Not counted as a specific transfer.');el('answers').append(detail);
const explanation=document.createElement('p');explanation.textContent=r.target_side===0?'Here success means LEFT answer red (donor arch), RIGHT answer red (original tower). The original arch is still blue in the screenshot.':'Here success means RIGHT answer blue (donor tower), LEFT answer blue (original arch). The original tower is still red in the screenshot.';el('answers').append(explanation);
const a=document.createElement('a');a.href='../runs/oscar/'+r.run+'/families/f0000-'+r.patch_set+'.json';a.textContent='Source: '+r.trial_key;el('evidence').replaceChildren(a);}
for(const id of ['kind','layers','side'])el(id).addEventListener('change',update);update();
</script></main></html>'''
    body=body.replace('CARDS',cards).replace('SWEEPFIG',image('vision-sweep-01','analysis/eli5-layer-results.png','Whole-state ViT sweep — not Q/K/V-only patches'))
    body=body.replace('COMPAREFIG',image('vision-projection-comparison','layer-comparison.png','Matched late-layer comparison: whole block output versus Q, K, V'))
    body=body.replace('REPLICATIONFIG',image('vision-captures-replication-1-6820294','contact_sheet.png','New replication capture smoke — real Minecraft screenshots, not model results'))
    body=body.replace('INVENTORY',''.join(inventory)).replace('TRIALDATA',json.dumps(trials).replace('<','\\u003c'))
    replication=RUNS/'vision-pilot-replication-6853654/analysis/audit.json'
    if replication.exists():
        audit=read(replication)
        results=[r for r in audit['counts'] if r['condition']=='target']
        result_text='; '.join(r['patch_set']+': '+str(r['specific_transfers'])+'/'+str(r['n']) for r in results)
        start=body.index('<p>The first yellow/blue')
        end=body.index('</p>',start)+4
        body=body[:start]+('<p><b>Replication complete and audited:</b> '+result_text+
            '. All 192 clean recognition checks passed. Full 24-configuration captures were reviewed locally. '
            '<a href="../runs/oscar/vision-pilot-replication-6853654/analysis/explained.html">Open replication breakdown and failure analysis</a>. '
            'See the later sections for destination choice and movement evidence.</p>'+
            image('vision-pilot-replication-6853654','analysis/replication-results.png','New-scene V replication: actual screenshots and recorded answers'))+body[end:]
    diagnostic=RUNS/'destination-diagnostic-6863843/report.html'
    if diagnostic.exists():
        body=body.replace('<h2 id="folders">', '<h2>Destination prompt calibration</h2><p>On the same 96 balanced side questions, the original action prompt scored 49/96 (95 LEFT answers), while the direct spatial prompt scored 96/96. Color controls also scored 96/96. This is prompt calibration, not held-out evaluation. <a href="../runs/oscar/destination-diagnostic-6863843/report.html">Images, exact prompts and observed answers</a>.</p><h2 id="folders">')
    destination=RUNS/'destination-replication-6868954/analysis/audit.json'
    if destination.exists():
        audit=read(destination)
        result='; '.join(r['patch_set']+': '+str(r['donor_choices'])+'/48' for r in audit['counts'] if r['condition']=='both_objects')
        body=body.replace('<h2 id="folders">','<h2>Destination interventions completed</h2><p>'+result+'. All 96 clean choices and 144 self-patches passed. These are offline decisions. <a href="../runs/oscar/destination-replication-6868954/analysis/explained.html">Actual images, patch masks and destination outcomes</a>.</p><h2 id="folders">')
    replay=RUNS/'destination-replay-6869428/blue-patched/report.html'
    if replay.exists():
        body=body.replace('<h2 id="folders">','<h2>Saved-choice movement replay</h2><p><a href="../runs/oscar/destination-replay-6869428/blue-patched/report.html">Inspect the recorded movement and trajectory</a>. This replay executes a saved destination choice using a known-coordinate controller. The model does not keep looking during movement. Moving region overlays are posthoc references, not continuous activation patches. <a href="research-plan.html">The new plan specifies binding tasks, closed-loop decisions, training and active sensing</a>.</p><h2 id="folders">')
    (ROOT/'docs/research-guide.html').write_text(body)
    print('Wrote docs/research-guide.html; linked local artifacts; comparison counts:',counts)
if __name__=='__main__':build()
