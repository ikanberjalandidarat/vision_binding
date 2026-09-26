"""Offline HTML teaching panels, using recorded trial values (no model calls)."""
import html
import json


def explain(rows, manifest, images):
    esc = lambda x: html.escape(str(x), quote=True)
    qrows = [r for r in rows if r['condition'] == 'q']
    if not qrows:
        return ''
    controls = {(r['scene_family_id'], r['side'], r['content_change'], r['address_flip']): r
                for r in rows if r['condition'] == 'random_q'}
    selves = {(r['scene_family_id'], r['side']): r for r in rows if r['condition']=='self_q'}
    cases = []
    for r in qrows:
        key = (r['scene_family_id'], r['side'], r['content_change'], r['address_flip'])
        c, s = controls[key], selves[key[:2]]
        selected = 1-r['side'] if r['address_flip'] else r['side']
        cases.append(dict(family=r['scene_family_id'], side=r['side'], changed=r['content_change'], flip=r['address_flip'],
            recipientImage=images[r['image_sha256']], donorImage=images[r['donor_image_sha256']],
            recipientPrompt=r['prompt'], donorPrompt=r['donor_prompt'], clean=r['clean_raw'],
            selfAnswer=s['raw'], q=r['raw'], random=c['raw'],
            originalColor=r['recipient_objects'][r['side']]['color'],
            selectedRecipientColor=r['recipient_objects'][selected]['color'],
            donorColor=r['donor_objects'][selected]['color'],
            positions=[{'layer':p['layer'], 'source':p['source_position'], 'destination':p['recipient_position'],
                        'width':p['shape'][-1], 'channels':len(p['channels'])} for p in r['patches']]))
    config = manifest['config']
    first = qrows[0]['patches'][0]
    text_cfg = manifest['model']['model_config'].get('text_config', manifest['model']['model_config'])
    nheads = text_cfg['num_attention_heads']
    dim = first['shape'][-1] // nheads
    layers = config['layers']
    family_options = ''.join(f'<option>{esc(f)}</option>' for f in sorted({r['scene_family_id'] for r in qrows}))
    payload = json.dumps(cases).replace('<', '\\u003c').replace('>', '\\u003e').replace('&', '\\u0026')
    body = '''
<style>
.explain h2{font-size:24px}.explain h3{font-size:18px;margin:8px 0}.explain p{line-height:1.6}
.explain .legend{display:flex;gap:12px;flex-wrap:wrap}.pill{border-radius:20px;padding:6px 12px;background:#e7eef4;font-size:14px}
.rec{border-top:5px solid #1973a3}.don{border-top:5px solid #8b4da9}.patched{border-top:5px solid #16826a}
.flow{display:grid;grid-template-columns:1fr 1fr 1fr;gap:16px;margin:20px 0}
.flow>div{padding:14px;background:#f4f7fa;border-radius:8px}.flow .formula{font-family:ui-monospace,monospace;padding:12px;background:white;border:1px solid #cbd7e1;overflow-wrap:anywhere}
.answer{font-size:24px;font-weight:700;color:#12614f}.explore-controls{display:flex;flex-wrap:wrap;gap:14px;padding:14px 0}.explore-controls label{display:grid;gap:6px;font-size:14px}
.matrix td{font-size:15px;font-weight:400}.matrix button{font:inherit;background:transparent;border:0;text-align:left;cursor:pointer;width:100%;padding:8px;color:inherit}.matrix .active{background:#e4f2ff;outline:2px solid #1973a3;outline-offset:-2px}.matrix .primary-cell{border:2px solid #8b4da9}
.matrix small{display:block;margin-top:8px;line-height:1.45}.matrix th{font-size:14px}
.tokenstrip{display:flex;gap:6px;flex-wrap:wrap;margin:12px 0}.tokenstrip span{padding:9px;background:#e9edf0;border-radius:5px}.tokenstrip .selected{background:#eee0fa;border:2px solid #8b4da9}
.diagram{width:100%;height:auto;background:#fafbfd;border-radius:10px}.explain .tiny{font-size:13px;color:#52616f}.explain pre{font-size:14px}
@media(max-width:760px){.flow{grid-template-columns:1fr}.matrix{display:block;overflow-x:auto}}
</style>
<section class="explain" id="how"><h2>What exactly did we donate?</h2>
<p><b>The donor supplies temporary internal Q activations.</b> It does not supply replacement pixels, model weights, an object crop, or an answer word. We run the same Qwen checkpoint on two different image–prompt contexts, then rerun the recipient with selected internal Q values replaced.</p>
<div class="legend"><span class="pill">Recipient = image/question being answered</span><span class="pill">Donor = image/question used to capture Q</span><span class="pill">Original = clean recipient, with no patch</span></div>
<h3>Explore one recorded intervention</h3>
<p>These controls replay saved results; they do not run or train a model. Start with opposite instructions and disjoint colors, then switch sides or intervention type.</p>
<div class="explore-controls">
<label>Configuration<select id="demo-family">__FAMILIES__</select></label>
<label>Recipient question<select id="demo-side"><option value="0">Leftmost structure</option><option value="1">Rightmost structure</option></select></label>
<label>Intervention<select id="demo-mode"><option value="q">Donor Q replacement</option><option value="clean">Clean / no patch</option><option value="self">Exact self-Q replacement</option><option value="random">Norm-matched random Q</option></select></label></div>
<div class="flow">
<div class="rec"><h3>1 · Original recipient</h3><img id="demo-rec-image" alt="Original recipient input"><p id="demo-rec-prompt"></p><div class="formula">Same checkpoint W → clean Qᴿ</div><p>Clean answer: <b id="demo-clean"></b></p></div>
<div class="don"><h3>2 · Donor context</h3><img id="demo-donor-image" alt="Donor context input"><p id="demo-donor-prompt"></p><div class="formula">Same checkpoint W → save Qᴰ</div><p>Donor-selected object’s color: <b id="demo-donor-color"></b>. This is an evaluation label, not the patch payload.</p></div>
<div class="patched"><h3>3 · Recipient inference again</h3><img id="demo-again-image" alt="Recipient input reused without pixel changes"><p id="demo-again-prompt"></p><div class="formula" id="demo-formula"></div><p>Recorded answer: <span class="answer" id="demo-answer"></span></p></div>
</div>
<p id="demo-reading" role="status" aria-live="polite"></p>
<p class="tiny">In clean and self modes, the displayed donor is only a reference context: its activations are not used. Random mode uses the donor–recipient difference only to set the perturbation magnitude.</p>
<h2>Primary diagnostic versus all donor contexts</h2>
<p>We cross two factors: does the donor ask for the same side or the opposite side, and does its image use the same colors or disjoint colors? “All donor contexts” includes all four cells below, including the primary cell. It does not mean a different model or a different patch location.</p>
<table class="matrix"><tr><th>Donor image</th><th>Same requested side</th><th>Opposite requested side</th></tr>
<tr><th>Same colors<br><small>Actually the same recipient image</small></th>
<td id="matrix-00"><button data-changed="false" data-flip="false"><b>Same-context control</b><small>Same image + same question. Q replacement should preserve the answer.</small></button></td>
<td id="matrix-01"><button data-changed="false" data-flip="true"><b>Instruction-change control</b><small>A switched answer can also match the donor-selected color, so selection and donor-answer transfer are confounded.</small></button></td></tr>
<tr><th>Disjoint colors<br><small>New donor structures/colors</small></th>
<td id="matrix-10"><button data-changed="true" data-flip="false"><b>Content-change control</b><small>Does the original recipient selection survive despite the different donor colors?</small></button></td>
<td id="matrix-11" class="primary-cell"><button data-changed="true" data-flip="true"><b>PRIMARY: opposite + disjoint</b><small>The other recipient color differs from the donor-selected color. This separates those two observed outcomes.</small></button></td></tr></table>
<p>Click a cell to change the example above. Each cell has both recipient target sides and both donor-Q and random-Q runs: <b>4 cells × 2 sides × 2 interventions = 16 intervention rows</b> per configuration. Separately: 8 clean color probes + 2 exact self-patches = 26 rows total.</p>
<p class="note">Disjoint donors also change structure templates; these are not perfectly matched silhouettes. The primary result can reflect left/right instruction transfer. It does not demonstrate an abstract object ID or persistence across camera views.</p>
<h2>Where inside Qwen did the patch happen?</h2>
<p>The vision encoder first turns the RGB image into visual embeddings. Patching occurs later, in the <b>language-model decoder’s self-attention</b>, at zero-based layers <b>__LAYERS__</b>, simultaneously in one recipient run. It is not a vision-encoder patch and not a layer sweep.</p>
<svg class="diagram" viewBox="0 0 1000 385" role="img" aria-labelledby="mechanism-title mechanism-desc">
<title id="mechanism-title">Donor Q replaces the final prompt token’s recipient Q before positional rotation</title>
<desc id="mechanism-desc">Recipient hidden states enter Q, K and V projections. Only Q at the final prompt position is overwritten by saved donor Q. Query and key positional rotation and attention follow. Keys and values have no direct patch.</desc>
<defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8" fill="#41576b"/></marker></defs>
<g font-family="system-ui,sans-serif" font-size="16" fill="#17212b">
<rect x="22" y="137" width="178" height="95" rx="10" fill="#dfedf6"/><text x="38" y="166">Recipient hidden states</text><text x="38" y="192">at decoder layer ℓ</text><text x="38" y="217" font-size="13">image + prompt context</text>
<rect x="249" y="70" width="130" height="48" rx="8" fill="#eee0fa"/><text x="268" y="100">q_proj(hᴿ)</text>
<rect x="249" y="161" width="130" height="48" rx="8" fill="#e9eef1"/><text x="268" y="191">k_proj(hᴿ)</text>
<rect x="249" y="264" width="130" height="48" rx="8" fill="#e9eef1"/><text x="268" y="294">v_proj(hᴿ)</text>
<path d="M200,157 L223,157 L223,94 L249,94 M200,185 L249,185 M200,215 L223,215 L223,288 L249,288" fill="none" stroke="#41576b" stroke-width="2" marker-end="url(#arrow)"/>
<rect x="419" y="62" width="192" height="62" rx="8" fill="#eee0fa" stroke="#8b4da9" stroke-width="2"/><text x="433" y="88">HOOK: replace Qᴿ[tᴿ]</text><text x="433" y="110">with saved Qᴰ[tᴰ]</text>
<rect x="419" y="8" width="192" height="33" rx="5" fill="#8b4da9"/><text x="434" y="31" fill="white">Donor capture at layer ℓ</text><path d="M515,41 L515,62" stroke="#8b4da9" stroke-width="3" marker-end="url(#arrow)"/>
<path d="M379,94 L419,94" stroke="#41576b" stroke-width="2" marker-end="url(#arrow)"/>
<rect x="653" y="72" width="118" height="145" rx="8" fill="#e8f3ee"/><text x="668" y="103">Reshape</text><text x="668" y="129">into heads</text><text x="668" y="165">Apply RoPE</text><text x="668" y="192" font-size="13">to Q and K</text>
<path d="M611,94 L653,94 M379,185 L653,185" stroke="#41576b" stroke-width="2" marker-end="url(#arrow)"/>
<rect x="813" y="116" width="164" height="111" rx="8" fill="#dfedf6"/><text x="831" y="145">Causal attention</text><text x="831" y="174">softmax(QKᵀ/√d)</text><text x="831" y="202">× V</text>
<path d="M771,146 L813,146 M379,288 L895,288 L895,227" fill="none" stroke="#41576b" stroke-width="2" marker-end="url(#arrow)"/>
<text x="422" y="175" font-size="13">K: no direct overwrite</text><text x="422" y="278" font-size="13">V: no direct overwrite</text>
<text x="22" y="345">Continue through output projection, residual stream, later decoder layers, then next-token prediction.</text>
<text x="22" y="371" font-size="13">Diagram is schematic. Earlier Q edits can indirectly change later hidden states, K and V; they are not frozen activations.</text>
</g></svg>
<h3>Which token? Which heads? What is the “original” vector?</h3>
<div class="tokenstrip"><span>visual tokens …</span><span>question tokens …</span><span>chat-template suffix …</span><span class="selected">FINAL INPUT TOKEN ← patch Q here</span><span>generated answer tokens → no new patches</span></div>
<p>The hook uses <code>input_ids.shape[1] − 1</code>: the last token of the fully formatted prompt, including the assistant-generation prefix. It is <b>not specifically the word “leftmost,” and not an image/object token</b>. The exact token’s text was not logged in this run; its index was.</p>
<p id="demo-position"></p>
<p>At each patched layer, the original activation is <code>Qᴿ = q_proj(hᴿ)</code>, an output of a learned linear projection (including its bias). The hook clones that output tensor and replaces only the selected token row/channels. The saved donor activation comes from a separate <b>clean donor forward pass</b>, at the same layer. Every layer receives its own captured vector, not one vector repeated seven times.</p>
<div class="flow"><div><h3>Original</h3><div class="formula">Q′ℓ[tᴿ,C] = Qᴿℓ[tᴿ,C]</div><p>No replacement. The recipient answers its original question.</p></div>
<div><h3>Donor replacement</h3><div class="formula">Q′ℓ[tᴿ,C] ← Qᴰℓ[tᴰ,C]</div><p>Literal replacement on C, not addition. Other token rows/channels pass through.</p></div>
<div><h3>Random control</h3><div class="formula">δ = Qᴰ − Qᴿ<br>Q′ = Qᴿ + ε · ‖δ‖ / ‖ε‖</div><p>Seeded Gaussian noise; matched norm on the actual selected channels, separately per layer. Norms are matched in float32 before BF16 casting.</p></div></div>
<p>The exact self-control uses the clean recipient capture instead of donor Q. Its output must equal the clean generated text before donor interventions proceed. No numerical Q values or attention maps were saved; this visualization shows the actual intervention specification and recorded outputs, not invented vector values or heatmaps.</p>
<h2>“Frozen model at inference” means weights do not learn</h2>
<div class="flow"><div><h3>Weights W stay fixed</h3><p>The same pretrained Qwen checkpoint is loaded once. The code uses <code>model.eval()</code> and <code>torch.inference_mode()</code>. There is no loss, backward pass, optimizer, or parameter update.</p></div>
<div><h3>Activations can change</h3><p>Each image/question creates temporary hidden states. The hook changes one part of that computation for this run. Changing an activation is an intervention; it is not fine-tuning.</p></div>
<div><h3>Prefill only</h3><p>Capture: donor forward with <code>use_cache=False</code>. Recipient generation: <code>use_cache=True</code>. Each patch fires on the initial full-input pass, then leaves subsequent token steps alone. Effects can still propagate through later layers, cached states, and generated text.</p></div></div>
<p class="note">Q vectors can be thought of as queries that contribute to attention over available keys. Calling them a pure “object address” would go beyond this experiment. The experiment measures what changes after replacing these contextual activations.</p>
</section>
<script type="application/json" id="demo-data">__DATA__</script>
<script>
(()=>{const cases=JSON.parse(document.getElementById('demo-data').textContent);let changed=true,flip=true;
const byId=id=>document.getElementById(id), text=(id,value)=>byId(id).textContent=value;
function update(){const side=Number(byId('demo-side').value), family=byId('demo-family').value,mode=byId('demo-mode').value;
const r=cases.find(x=>x.family===family&&x.side===side&&x.changed===changed&&x.flip===flip);if(!r)return;
byId('demo-rec-image').src=r.recipientImage;byId('demo-donor-image').src=r.donorImage;byId('demo-again-image').src=r.recipientImage;
text('demo-rec-prompt',r.recipientPrompt);text('demo-donor-prompt',r.donorPrompt);text('demo-again-prompt',r.recipientPrompt);
text('demo-clean',r.clean);text('demo-donor-color',r.donorColor);
const ans={q:r.q,clean:r.clean,self:r.selfAnswer,random:r.random}[mode];text('demo-answer',ans);
text('demo-formula',{q:'Overwrite final-token Q with donor Q at every selected layer.',clean:'No hook: retain the recipient Q.',self:'Overwrite final-token Q with clean recipient Q.',random:'Replace Q with clean recipient Q + norm-matched random perturbation.'}[mode]);
text('demo-reading','Recorded '+mode+' answer: “'+ans+'”. Original recipient target = '+r.originalColor+'; recipient at donor-selected side = '+r.selectedRecipientColor+'; donor-selected object = '+r.donorColor+'. '+(changed&&flip?'This is the primary diagnostic: the latter two colors are different.':'This is a control context; some outcome descriptions may coincide.'));
text('demo-position',r.positions.map(p=>'L'+p.layer+': donor index '+p.source+' → recipient index '+p.destination+'; '+p.channels+'/'+p.width+' Q channels.').join(' '));
document.querySelectorAll('.matrix td').forEach(td=>td.classList.remove('active'));byId('matrix-'+Number(changed)+Number(flip)).classList.add('active');}
['demo-side','demo-family','demo-mode'].forEach(id=>byId(id).addEventListener('change',update));
document.querySelectorAll('.matrix button').forEach(b=>b.addEventListener('click',()=>{changed=b.dataset.changed==='true';flip=b.dataset.flip==='true';update();}));update();})();
</script>
'''
    all_channels = all(len(p['channels'])==p['shape'][-1] for r in qrows for p in r['patches'])
    head_note = f"<p><b>This run:</b> {len(layers)} layers; {first['shape'][-1]:,} Q channels per token = {nheads} query heads × {dim} channels/head. {'All query heads were replaced; this is a broad multi-head intervention.' if all_channels else 'Only the logged channel subset was replaced.'} Layer and token indices are zero-based.</p>"
    body = body.replace('<h3>Which token? Which heads? What is the “original” vector?</h3>', '<h3>Which token? Which heads? What is the “original” vector?</h3>'+head_note)
    return body.replace('__FAMILIES__',family_options).replace('__LAYERS__',esc(', '.join(map(str,layers)))).replace('__DATA__',payload)
