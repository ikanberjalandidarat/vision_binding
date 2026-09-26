"""Three-run SVG figures of the recorded Q intervention (not attention heatmaps)."""
import html


def patch_figures(rows, manifest, images):
    figures = []
    clean = {(r['scene_family_id'], r['context'], r['side']): r for r in rows
             if r['condition'] == 'clean' and r['kind'] == 'pair'}
    for r in rows:
        if r['condition'] != 'q' or not (r['content_change'] and r['address_flip']):
            continue
        fid, side = r['scene_family_id'], r['side']
        donor = clean[fid, 'disjoint', 1-side]
        esc = lambda s: html.escape(str(s), quote=True)
        uid = f'patch-{fid}-{side}'
        pieces = [f'''<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" viewBox="0 0 1200 860" role="img" aria-labelledby="{uid}-title {uid}-desc">
<title id="{uid}-title">Original, donor and patched runs: recipient asks {'left' if side==0 else 'right'}</title>
<desc id="{uid}-desc">Actual input images and recorded answers. At decoder layers 21 through 27, all query heads at the final prompt token are replaced with the donor's captured Q. Matrix cells indicate patched head locations, not measured activation values.</desc>
<defs><marker id="{uid}-arrow" markerWidth="9" markerHeight="9" refX="8" refY="4.5" orient="auto"><path d="M0,0 L9,4.5 L0,9" fill="#89969c"/></marker><marker id="{uid}-orange" markerWidth="9" markerHeight="9" refX="8" refY="4.5" orient="auto"><path d="M0,0 L9,4.5 L0,9" fill="#d77a1d"/></marker></defs>
<rect width="1200" height="860" fill="white"/>
<rect x="14" y="64" width="1172" height="162" fill="#f5e5dc"/>
<rect x="14" y="226" width="1172" height="330" fill="#e8efdf"/>
<rect x="14" y="556" width="1172" height="236" fill="#fcf0d5"/>
<path d="M803,64 V792" stroke="#59656d" stroke-width="3"/>
<g font-family="Arial, sans-serif" fill="#27343c">
<text x="600" y="31" text-anchor="middle" font-size="24" font-weight="bold">What color is the {'leftmost' if side==0 else 'rightmost'} structure?</text>
<text x="600" y="53" text-anchor="middle" font-size="14">Read bottom → top. The patched run keeps the original recipient image and question.</text>''']
        def text(x, y, value, size=15, color='#27343c', anchor='middle', bold=False):
            return f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-size="{size}" fill="{color}" font-weight="{"bold" if bold else "normal"}">{esc(value)}</text>'
        columns = [(34, 'Original run', r['image_sha256'], r['prompt'], r['clean_raw'], 'Recipient Qᴿ', '#267b8c'),
                   (424, 'Counterfactual / donor run', r['donor_image_sha256'], r['donor_prompt'], donor['raw'], 'Captured donor Qᴰ', '#8252a4'),
                   (834, 'Patched recipient run', r['image_sha256'], r['prompt'], r['raw'], 'Replace Qᴿ ← Qᴰ', '#8252a4')]
        heads = manifest['model']['model_config'].get('text_config', manifest['model']['model_config'])['num_attention_heads']
        if heads != 28 or any(len(p['channels']) != p['shape'][-1] for p in r['patches']):
            raise ValueError('This figure layout requires the logged 28-head full-Q pilot')
        for col, (x, title, image_key, prompt, answer, vector_name, color) in enumerate(columns):
            cx=x+165
            pieces.append(text(cx, 91, 'RECORDED ANSWER', 12))
            pieces.append(f'<rect x="{x+40}" y="104" width="250" height="63" rx="4" fill="white" stroke="{color}" stroke-width="2"/>')
            pieces.append(text(cx, 146, answer, 31, color, bold=True))
            pieces.append(text(cx, 193, 'Same checkpoint · no weight updates', 13))
            pieces.append(f'<path d="M{cx},306 L{cx},205" stroke="#89969c" stroke-width="4" marker-end="url(#{uid}-arrow)"/>')
            border='#d77a1d' if col==2 else color
            pieces.append(f'<rect x="{x+6}" y="312" width="318" height="202" fill="white" stroke="{border}" stroke-width="{4 if col==2 else 2}"/>')
            pieces.append(text(cx,335,vector_name,18,border,bold=True))
            pieces.append(text(cx,354,'Final input token only · pre-RoPE',12))
            for j, p in enumerate(r['patches']):
                y=371+18*j
                pieces.append(text(x+43,y+9,'L'+str(p['layer']),11,anchor='end'))
                for h in range(heads):
                    pieces.append(f'<rect x="{x+52+h*9}" y="{y}" width="7" height="11" fill="{color}" opacity="0.85"/>')
            pieces.append(text(cx,536,'28 cells per row = 28 query heads × 128 channels',12))
            pieces.append(f'<path d="M{x+48},605 L{x+22},605 L{x+22},520" fill="none" stroke="#89969c" stroke-width="3" marker-end="url(#{uid}-arrow)"/>')
            request='rightmost' if 'rightmost' in prompt else 'leftmost'
            pieces.append(text(cx,578,'What color is the '+request+' structure?',15,bold=True))
            pieces.append(f'<rect x="{x+48}" y="590" width="234" height="31" rx="4" fill="white" stroke="{border}" stroke-width="2"/>')
            pos=r['patches'][0]['source_position' if col==1 else 'recipient_position']
            pieces.append(text(cx,611,f'… final prompt token · index {pos}',13))
            pieces.append(f'<path d="M{cx},638 L{cx},625" stroke="#89969c" stroke-width="3" marker-end="url(#{uid}-arrow)"/>')
            pieces.append(f'<image x="{x}" y="640" width="330" height="115" href="{images[image_key]}" xlink:href="{images[image_key]}"/>')
            pieces.append(text(cx,779,'Donor RGB' if col==1 else 'Recipient RGB — identical pixels',13))
            pieces.append(text(cx,819,title,20,bold=True))
        pieces.append(f'<path d="M589,309 C590,232 998,232 999,306" stroke="#d77a1d" stroke-width="5" fill="none" marker-end="url(#{uid}-orange)"/>')
        pieces.append('<rect x="642" y="237" width="324" height="43" rx="18" fill="#d77a1d"/>')
        pieces.append(text(804,264,'COPY Q: every shown layer + head',16,'white',bold=True))
        pieces.append(text(600,843,'Colored cells mark intervention locations, NOT vector magnitudes, attention strengths, or separate color/shape modules.',13))
        pieces.append('</g></svg>')
        figures.append((uid+'.svg', ''.join(pieces), side))
    return figures


def figure_panel(figures):
    blocks = []
    for i, (filename, svg, side) in enumerate(figures):
        blocks.append(f'<div class="patch-figure" data-case="{i}" {"hidden" if i else ""}>{svg}<p><a href="{html.escape(filename)}" download>Download this figure as SVG</a></p></div>')
    options=''.join(f'<option value="{i}">{html.escape(name[:-4])}: recipient asks {"left" if side==0 else "right"}</option>' for i,(name,svg,side) in enumerate(figures))
    return '''<section id="three-run-figure"><h2>Exactly what was patched: three-run figure</h2>
<p>This follows the layout of your example: input at the bottom, internal intervention in the middle, answer at the top. Here the intervention is <b>Q at the final prompt token across all query heads in layers 21–27</b>. We did not identify or patch separate “color heads” or “shape heads.”</p>
<label>Show recorded primary example <select id="patch-figure-select">'''+options+'''</select></label>
<div style="overflow-x:auto">'''+''.join(blocks)+'''</div>
<p>The purple donor cells are copied into the orange-outlined recipient Q rows. All seven rows are patched simultaneously. The recipient still receives its original RGB image and original question.</p>
<p><b>What stays fixed:</b> checkpoint weights and recipient input pixels/text. <b>What is replaced:</b> one Q token row per selected decoder layer, before RoPE. K and V are not directly overwritten, but downstream activations can change. The donor answer shown is from its separate clean baseline; the answer word itself is never injected.</p>
<p>Each cell represents one head’s 128 channels; its color indicates the source of those channels. Numerical vectors were not saved. These are intervention schematics grounded in the recorded metadata, not measured attention maps.</p>
</section><style>.patch-figure svg{width:100%;height:auto;min-width:850px;display:block}#three-run-figure select{max-width:100%}</style>
<script>document.getElementById('patch-figure-select').addEventListener('change',function(){document.querySelectorAll('.patch-figure').forEach(el=>el.hidden=el.dataset.case!==this.value)});</script>'''


def network_panel():
    return '''<section id="network-view"><h2>Neural-network view: the patch is in the LLM backbone</h2>
<p><b>Not the ViT.</b> Qwen first encodes the image into visual embeddings. Those embeddings and the question/chat-template tokens enter its language-model decoder. Our hooks target <code>model.language_model.layers[21…27].self_attn.q_proj</code> in this recorded run.</p>
<div style="overflow-x:auto"><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1140 850" style="width:100%;min-width:850px" role="img" aria-labelledby="nn-title nn-desc">
<title id="nn-title">Qwen vision encoder and LLM decoder with final-token Q replacement</title>
<desc id="nn-desc">An image is encoded by the vision transformer, then merged into visual embeddings alongside text tokens. Inside decoder layers 21 through 27, the Q projection computes a vector for every input token. The hook replaces only the final input token row with a same-layer donor Q vector before reshape, RoPE and attention. Neural units are schematic, not measured activations.</desc>
<defs><marker id="nn-arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8" fill="#466174"/></marker><marker id="nn-copy" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8" fill="#d77920"/></marker></defs>
<rect width="1140" height="850" rx="12" fill="#f8fafc"/>
<g font-family="Arial,sans-serif" font-size="16" fill="#263846">
<text x="30" y="34" font-size="23" font-weight="bold">A · Image and text enter different front ends, then meet in the decoder</text>
<rect x="25" y="66" width="100" height="65" rx="8" fill="#e0edf6"/><text x="49" y="95">RGB</text><text x="41" y="118">image</text>
<rect x="175" y="66" width="200" height="65" rx="8" fill="#e0edf6"/><text x="194" y="94">Vision transformer</text><text x="199" y="118" font-size="13">ViT · NO HOOK HERE</text>
<rect x="420" y="66" width="160" height="65" rx="8" fill="#e0edf6"/><text x="441" y="94">Patch merger</text><text x="431" y="118" font-size="13">visual embeddings</text>
<rect x="175" y="157" width="405" height="50" rx="8" fill="#e8edf1"/><text x="195" y="187">Question + chat template → text embeddings</text>
<rect x="650" y="68" width="260" height="139" rx="8" fill="#e6f0df" stroke="#466174"/><text x="683" y="98" font-weight="bold">LLM decoder · 28 layers</text><text x="682" y="126">Layers 0–20: no direct patch</text><rect x="670" y="143" width="219" height="44" fill="#fff1de" stroke="#d77920" stroke-width="3"/><text x="684" y="170">Layers 21–27: Q hooks</text>
<rect x="970" y="103" width="140" height="70" rx="8" fill="#e8edf1"/><text x="990" y="130">LM head</text><text x="984" y="154">→ next token</text>
<g fill="none" stroke="#466174" stroke-width="2" marker-end="url(#nn-arrow)"><path d="M125,98 H175"/><path d="M375,98 H420"/><path d="M580,98 H650"/><path d="M580,182 H650"/><path d="M910,138 H970"/></g>
<path d="M780,208 V245" stroke="#d77920" stroke-width="3" stroke-dasharray="6 4"/>
<text x="30" y="263" font-size="23" font-weight="bold">B · Zoom into one selected LLM decoder layer ℓ</text>
<text x="30" y="291" font-size="14">Prefill computes all input positions. “Final token” means the last formatted input token, not the final network layer.</text>
<rect x="28" y="325" width="230" height="264" rx="8" fill="white" stroke="#b7c8d4"/>
<text x="44" y="353" font-weight="bold">Token hidden states Hᴿ</text>
<text x="44" y="387" font-size="14">visual-token rows …</text><path d="M45,401 H239 M45,413 H239" stroke="#8eabbc" stroke-width="7"/>
<text x="44" y="450" font-size="14">text-token rows …</text><path d="M45,464 H239 M45,476 H239" stroke="#8eabbc" stroke-width="7"/>
<rect x="40" y="512" width="205" height="57" fill="#f0e5f6" stroke="#9056a8" stroke-width="2"/><text x="54" y="535" font-size="14">hᴿ at final input position</text><text x="54" y="556" font-size="14">tᴿ = 209 in this run</text>
<rect x="302" y="365" width="258" height="225" rx="8" fill="#edf3f8"/>
<text x="321" y="389" font-weight="bold">Learned Q projection</text><text x="321" y="412" font-size="14">q_proj(h) = Wq h + bq</text>
<g stroke="#a4b6c5" stroke-width="1.2" opacity="0.85">
<path d="M344,448 L512,443 M344,448 L512,474 M344,448 L512,505 M344,448 L512,536 M344,448 L512,567
M344,480 L512,443 M344,480 L512,474 M344,480 L512,505 M344,480 L512,536 M344,480 L512,567
M344,512 L512,443 M344,512 L512,474 M344,512 L512,505 M344,512 L512,536 M344,512 L512,567
M344,544 L512,443 M344,544 L512,474 M344,544 L512,505 M344,544 L512,536 M344,544 L512,567"/>
</g><g fill="#658da8" stroke="white" stroke-width="2"><circle cx="344" cy="448" r="9"/><circle cx="344" cy="480" r="9"/><circle cx="344" cy="512" r="9"/><circle cx="344" cy="544" r="9"/><circle cx="512" cy="443" r="9"/><circle cx="512" cy="474" r="9"/><circle cx="512" cy="505" r="9"/><circle cx="512" cy="536" r="9"/><circle cx="512" cy="567" r="9"/></g>
<path d="M258,530 H302" stroke="#466174" stroke-width="2" marker-end="url(#nn-arrow)"/>
<text x="310" y="610" font-size="12">A few schematic units shown; weights stay fixed.</text>
<rect x="612" y="325" width="260" height="264" rx="8" fill="white" stroke="#b7c8d4"/>
<text x="630" y="353" font-weight="bold">Q output: [1, T, 3584]</text><text x="630" y="387" font-size="14">Other token rows pass through</text>
<path d="M630,406 H850 M630,433 H850 M630,460 H850" stroke="#8eabbc" stroke-width="12"/>
<text x="630" y="495" font-size="14">Final token row → overwrite</text>
<rect x="625" y="513" width="231" height="58" fill="#f0e5f6" stroke="#d77920" stroke-width="4"/><text x="640" y="538" font-weight="bold">Q′[0, 209, :] ← Qᴰℓ[tᴰ]</text><text x="640" y="558" font-size="13">all 3,584 channels replaced</text>
<path d="M560,530 H612" stroke="#466174" stroke-width="2" marker-end="url(#nn-arrow)"/>
<rect x="920" y="334" width="191" height="114" rx="8" fill="#f0e5f6" stroke="#9056a8"/><text x="935" y="360" font-weight="bold">Separate donor pass</text><text x="935" y="385" font-size="14">Same layer’s q_proj</text><text x="935" y="408" font-size="14">Save final-row Qᴰℓ</text><text x="935" y="433" font-size="14">[1, 3584] per layer</text>
<path d="M1015,449 V540 H866" fill="none" stroke="#d77920" stroke-width="4" marker-end="url(#nn-copy)"/><text x="928" y="516" fill="#b46114" font-size="14" font-weight="bold">COPY ACTIVATIONS</text>
<path d="M741,590 V645" stroke="#466174" stroke-width="3" marker-end="url(#nn-arrow)"/>
<rect x="425" y="650" width="677" height="59" rx="8" fill="#e6f0df"/><text x="442" y="675">Reshape Q into 28 heads × 128 → apply RoPE → attention with K and V</text><text x="442" y="696" font-size="13">K/V receive no direct overwrite. Earlier interventions may indirectly change later K/V activations.</text>
<rect x="30" y="651" width="353" height="58" rx="8" fill="#e8edf1"/><text x="48" y="676" font-size="14">Hᴿ → separate learned K / V projections</text><text x="48" y="697" font-size="13">No patch hook on these branches</text><path d="M383,680 H425" stroke="#466174" stroke-width="2" marker-end="url(#nn-arrow)"/>
<text x="30" y="751" font-size="18" font-weight="bold">Seven distinct donor vectors, seven replacement sites, one patched recipient inference.</text>
<text x="30" y="780" font-size="15">Repeat this operation at layers 21, 22, 23, 24, 25, 26 and 27. Each layer uses its own saved donor vector.</text>
<text x="30" y="810" font-size="15">Hooks fire on prefill only. Later generated-token steps run without further replacement; effects can persist.</text>
<text x="30" y="836" font-size="13" fill="#576b79">Position 209 and 3,584 channels are from this recorded pilot. Units/edges above illustrate computation, not measured weights.</text>
</g></svg></div>
<p><b>Important:</b> the final input-token hidden state is contextual: it can carry information about the image and the instruction. We overwrite its <em>projected Q vector</em>, not its entire hidden state. We do not overwrite Q for every image token.</p>
<pre>For each ℓ in [21, 22, 23, 24, 25, 26, 27]:
    donor_Q[ℓ] = donor_layer[ℓ].self_attn.q_proj(...)[0, donor_last, :]
    # During recipient prefill, after this layer's q_proj runs:
    recipient_Q_output[0, recipient_last, :] = donor_Q[ℓ]
    # Continue with head reshape, RoPE and causal attention.</pre>
<p>The displayed module path and token index describe this run. “Final token” is determined after the chat template is applied; its exact text was not saved. The figure does not claim the vector is a pure location or object-identity representation.</p>
</section>'''
