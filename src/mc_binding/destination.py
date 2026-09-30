"""Frozen visual destination selection. Movement is a separate, audited replay."""
from contextlib import ExitStack
from pathlib import Path
import base64
import html
import json
import numpy as np
import torch
from PIL import Image
from .io import RunStore, atomic_json, digest, source_hash, environment
from .vision_data import load_swaps, regions, object_mask, overlay
from .vision_hooks import vision_backbone, capture_blocks, patch_block, replacement
from .vision_pilot import patch_sets


def parse_side(raw):
    value = raw.strip().lower().rstrip('.!')
    return {'left': 0, 'right': 1}.get(value)


def destination_prompt(color, mode='original_action'):
    if mode == 'direct_spatial':
        return f'In this image, is the {color} structure on the left or the right? Answer left or right only.'
    if mode != 'original_action':
        raise ValueError('Unknown destination prompt mode')
    return f'You must walk to the {color} structure. Which side is it on? Answer LEFT or RIGHT only.'


def destination(dataset, output, config, reviewed=False):
    if not reviewed:
        raise ValueError('Inspect captures first, then use --reviewed-captures')
    if config.get('vision_patch_kind') != 'v' or config.get('dtype') != 'bfloat16' or config.get('load_in_4bit') or not config.get('check_finite_scores') or not config.get('use_fast'):
        raise ValueError('Require finite-checked, unquantized BF16 V patches and fast processor')
    from importlib.metadata import version
    if version('transformers') != '4.55.0':
        raise ValueError('Token alignment requires transformers 4.55.0')
    prompt_mode = config.get('destination_prompt_mode', 'original_action')
    destination_prompt('blue', prompt_mode)  # Validate before loading the model.
    root, out = Path(dataset), Path(output)
    data, groups = load_swaps(root)
    store = RunStore(out, dict(experiment='destination_choice_v1', dataset_hash=digest(data), config=config,
                              source_hash=source_hash(), environment=environment(),
                              scope='Offline LEFT/RIGHT choice, not observed movement; both object regions patched'))
    from .models.qwen import Qwen
    model = Qwen(config)
    for p in model.model.parameters():
        p.requires_grad_(False)
    visual, path = vision_backbone(model.model)
    sites = [b.attn.qkv for b in visual.blocks]
    sets = patch_sets(config, len(sites))
    layers = sorted({l for _, group in sets for l in group})
    atomic_json(out/'model.json', {**model.manifest(), 'vision_path': path, 'weights_frozen': True})
    atomic_json(out/'status.json', {'state': 'running'})
    try:
        for fid, records in groups.items():
            if store.completed(fid):
                continue
            pairs = {r['context']: r for r in records if r['kind'] == 'pair'}
            images = {ctx: Image.open(root/r['image']).convert('RGB') for ctx,r in pairs.items()}
            goals = [o['color'] for o in pairs['recipient']['objects']]
            inputs = {(ctx,c): model.inputs(im, destination_prompt(c, prompt_mode)) for ctx,im in images.items() for c in goals}
            rows = []
            clean = {}
            def row(ctx, goal, raw, condition, name='baseline', layer_set=(), pos=(), patches=()):
                original = next(i for i,o in enumerate(pairs['recipient']['objects']) if o['color']==goal)
                donor = next(i for i,o in enumerate(pairs['color_swap']['objects']) if o['color']==goal)
                choice = parse_side(raw)
                return dict(trial_key=f'{fid}:{name}:{condition}:{goal}', family=fid, condition=condition,
                            patch_set=name, layer_set=list(layer_set), positions=list(pos), patches=list(patches),
                            goal_color=goal, prompt=destination_prompt(goal, prompt_mode), prompt_mode=prompt_mode, raw=raw, choice_side=choice,
                            original_goal_side=original, donor_goal_side=donor,
                            chose_original=choice==original, chose_donor=choice==donor,
                            input_context=ctx, record_id=pairs[ctx]['record_id'],
                            recipient_image=pairs['recipient']['image'], donor_image=pairs['color_swap']['image'])
            for ctx in pairs:
                for goal in goals:
                    raw = model.answer(inputs[ctx,goal]);clean[ctx,goal]=raw
                    rows.append(row(ctx,goal,raw,'clean_'+ctx))
            atomic_json(out/'diagnostics'/f'{fid}.json',rows)
            if not all(r['chose_original'] if r['input_context']=='recipient' else r['chose_donor'] for r in rows):
                raise RuntimeError(f'{fid}: clean destination gate failed; inspect diagnostics; do not interpret patches')
            grids = [inp['image_grid_thw'].tolist() for inp in inputs.values()]
            if any(g != grids[0] for g in grids) or len(grids[0]) != 1 or grids[0][0][0] != 1:
                raise ValueError('Expected aligned still-image grids')
            _,h,w = map(int,grids[0][0]);merge=int(visual.spatial_merge_size)
            if merge != int(model.processor.image_processor.merge_size):
                raise ValueError('Processor/model merge mismatch')
            selected,bg = regions(images['recipient'],pairs['recipient'],h,w,merge,config['mask_threshold'])
            for a,b in zip(pairs['recipient']['objects'],pairs['color_swap']['objects']):
                ma=object_mask(images['recipient'],a);mb=object_mask(images['color_swap'],b)
                if np.logical_and(ma,mb).sum()/np.logical_or(ma,mb).sum() < .9:
                    raise ValueError('Donor silhouette alignment failed')
            both=sorted(set(selected[0]+selected[1]))
            if len(bg)<len(both):
                raise ValueError('Insufficient background control')
            background=sorted(map(int,np.random.default_rng(config['seed']).choice(bg,len(both),replace=False)))
            audit=out/'alignment'/fid;audit.mkdir(parents=True,exist_ok=True)
            for ctx,im in images.items():
                overlay(im,both,h,w,merge,audit/f'{ctx}-both.png')
            atomic_json(audit/'tokens.json',dict(grid=grids[0],merge=merge,object_positions=selected,background_positions=background))
            caps={}
            for ctx in pairs:
                caps[ctx]={}
                with torch.inference_mode(),capture_blocks(sites,layers,caps[ctx],'v'):
                    model.model(**inputs[ctx,goals[0]],use_cache=False)
                if any(v.shape[0]!=h*w for v in caps[ctx].values()):
                    raise ValueError('Spatial activation count mismatch')
            for name,layer_set in sets:
                for condition in ('self','both_objects','random','background'):
                    pos=background if condition=='background' else both
                    values={};patches=[]
                    for l in layer_set:
                        values[l],norm=replacement(caps['recipient'][l],caps['color_swap'][l],pos,condition,config['seed']+l)
                        patches.append(dict(layer=l,module=f'{path}.blocks.{l}.attn.qkv:v',delta_norm=norm))
                    for goal in goals:
                        with ExitStack() as stack:
                            for l in layer_set:
                                stack.enter_context(patch_block(sites[l],pos,values[l],h*w,'v'))
                            raw=model.answer(inputs['recipient',goal])
                        if condition=='self' and raw!=clean['recipient',goal]:
                            raise RuntimeError('Exact self-patch destination gate failed')
                        rows.append(row('recipient',goal,raw,condition,name,layer_set,pos,patches))
                        atomic_json(out/'diagnostics'/f'{fid}.json',rows)
                        print(f'{fid} {name} {condition} go to {goal}: {raw}',flush=True)
            store.save(fid,rows);store.export()
        rows=store.export()
        destination_report(out,root,rows)
        atomic_json(out/'status.json',dict(state='complete',rows=len(rows)))
    except BaseException as exc:
        store.export()
        atomic_json(out/'status.json',dict(state='error',error=str(exc)))
        raise


def destination_report(out, root, rows):
    parts=['<!doctype html><meta charset="utf-8"><title>Destination choices</title>',
           '<style>body{font:16px system-ui;max-width:1100px;margin:30px auto}img{width:45%}td,th{padding:8px;border-bottom:1px solid #ddd}</style>',
           '<h1>Did a visual patch change the chosen destination?</h1><p>These are recorded model choices, not movement results. Both object regions receive donor V activations. Input pixels remain unchanged. LEFT/RIGHT refer to the displayed image.</p>']
    for fid in sorted({r['family'] for r in rows}):
        group=[r for r in rows if r['family']==fid]
        parts.append(f'<h2>{html.escape(fid)}</h2><p>Original (left image) and donor (right image)</p>')
        for key in ('recipient_image','donor_image'):
            encoded=base64.b64encode((root/group[0][key]).read_bytes()).decode()
            parts.append(f'<img src="data:image/png;base64,{encoded}">')
        parts.append('<table><tr><th>Trial</th><th>Go to</th><th>Answer</th><th>Original side</th><th>Donor side</th><th>Chose donor?</th></tr>')
        for r in group:
            cells=[r['trial_key'],r['goal_color'],r['raw'],('LEFT','RIGHT')[r['original_goal_side']],('LEFT','RIGHT')[r['donor_goal_side']],str(r['chose_donor'])]
            parts.append('<tr>'+''.join('<td>'+html.escape(c)+'</td>' for c in cells)+'</tr>')
        parts.append('</table>')
    (out/'report.html').write_text('\n'.join(parts))
