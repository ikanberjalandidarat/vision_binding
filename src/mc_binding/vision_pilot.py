"""Exploratory spatial vision-block patch sweep, with frozen language readout."""
import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from .vision_data import load_swaps, regions, overlay, object_mask
from .vision_hooks import vision_backbone, capture_blocks, patch_block, replacement
from .q_pilot import question
from .scoring import parse
from .io import RunStore, atomic_json, digest, source_hash, environment


def color(raw):
    parsed = parse(raw, task='color')['parsed']
    return parsed['color'] if parsed else None


def vision_pilot(dataset, output, config, reviewed=False):
    if not reviewed:
        raise ValueError('Inspect capture contact_sheet.png and raw frames, then use --reviewed-captures')
    if config.get('dtype') != 'bfloat16' or config.get('load_in_4bit') or not config.get('use_fast') or not config.get('check_finite_scores'):
        raise ValueError('Require BF16, unquantized, fast processor and finite-score checks')
    from importlib.metadata import version
    if version('transformers') != '4.55.0':
        raise ValueError('Vision token ordering is validated only for transformers 4.55.0')
    root, out = Path(dataset), Path(output)
    data, groups = load_swaps(root)
    store = RunStore(out, {'experiment': 'vision_block_color_swap_v1', 'dataset_hash': digest(data),
        'config': config, 'source_hash': source_hash(), 'environment': environment(),
        'scope': 'fixed-pose spatial color transfer; not proof of shape binding or independent worlds'})
    from .models.qwen import Qwen
    model = Qwen(config)
    visual, path = vision_backbone(model.model)
    layers = config['vision_layers']
    if not layers or len(set(layers)) != len(layers) or any(not isinstance(l,int) or l<0 or l>=len(visual.blocks) for l in layers):
        raise ValueError('Invalid vision layers')
    atomic_json(out/'model.json', {**model.manifest(), 'vision_path': path,
        'vision_depth': len(visual.blocks), 'hook_site': 'vision block output after attention and MLP residuals, before merger',
        'vision_layers': layers, 'weights_frozen': True})
    for p in model.model.parameters():
        p.requires_grad_(False)
    status = {'state': 'running'}
    atomic_json(out/'status.json', status)
    try:
        for fid, records in groups.items():
            if all(store.completed(f'{fid}-layer{layer:02d}') for layer in layers):
                continue
            pairs = {r['context']: r for r in records if r['kind']=='pair'}
            inputs, clean, images, rows = {}, {}, {}, []
            for record in records:
                with Image.open(root/record['image']) as im:
                    image = im.convert('RGB')
                for side, obj in enumerate(record['objects']):
                    prompt = question(side if record['kind']=='pair' else None)
                    inp = model.inputs(image, prompt)
                    raw = model.answer(inp)
                    rows.append({'trial_key': f'{fid}:clean:{record["record_id"]}:{side}', 'condition': 'clean',
                        'record_id': record['record_id'], 'side': side, 'prompt': prompt, 'raw': raw,
                        'expected': obj['color'], 'correct': color(raw)==obj['color']})
                    if record['kind']=='pair':
                        key = (record['context'], side)
                        inputs[key], clean[key] = inp, raw
                        images[record['context']] = image
            atomic_json(out/'diagnostics'/f'{fid}-baseline.json', rows)
            if not all(r['correct'] for r in rows):
                raise RuntimeError(f'{fid}: clean color gate failed; see diagnostics')
            if not store.completed(fid+'-baseline'):
                store.save(fid+'-baseline', rows)
            rec = pairs['recipient']
            grid = inputs['recipient',0]['image_grid_thw'].tolist()
            if len(grid)!=1 or grid[0][0]!=1 or any(inp['image_grid_thw'].tolist()!=grid for inp in inputs.values()):
                raise ValueError('Expected identical single-image grids')
            _, h, w = map(int, grid[0])
            merge = int(model.processor.image_processor.merge_size)
            if merge != int(visual.spatial_merge_size):
                raise ValueError('Processor/model merge mismatch')
            selected, background = regions(images['recipient'], rec, h, w, merge, config['mask_threshold'])
            for a,b in zip(rec['objects'], pairs['color_swap']['objects']):
                ma, mb = object_mask(images['recipient'],a), object_mask(images['color_swap'],b)
                iou = np.logical_and(ma,mb).sum()/np.logical_or(ma,mb).sum()
                if iou < .9:
                    raise ValueError('Color-swap silhouette alignment failed (mask IoU < 0.9)')
            audit = out/'alignment'/fid
            audit.mkdir(parents=True, exist_ok=True)
            for side, pos in enumerate(selected):
                for ctx, im in images.items():
                    overlay(im, pos, h,w,merge, audit/f'{ctx}-side{side}.png')
            atomic_json(audit/'tokens.json', {'grid_thw': grid, 'merge': merge, 'threshold': config['mask_threshold'],
                'ordering': 'merge block row,col then inner row,col', 'object_positions': selected,
                'objects': rec['objects'], 'background_candidates': background})
            caps = {}
            for ctx in ('recipient','color_swap'):
                caps[ctx] = {}
                with torch.inference_mode(), capture_blocks(visual.blocks, layers, caps[ctx]):
                    model.model(**inputs[ctx,0], use_cache=False)
                if any(v.shape[0]!=h*w for v in caps[ctx].values()):
                    raise ValueError('Activation count disagrees with spatial mapping')
            for layer in layers:
                unit = f'{fid}-layer{layer:02d}'
                if store.completed(unit):
                    continue
                rows = []
                # Gate all self replacements before donor effects in this layer.
                for condition in ('self','target','random','other_object','background'):
                    for target in (0,1):
                        n = len(selected[target])
                        if len(background)<n:
                            raise ValueError('Insufficient background control tokens')
                        rng = np.random.default_rng(config['seed']+target)
                        if condition == 'other_object':
                            pos = selected[1-target]
                        elif condition == 'background':
                            pos = sorted(map(int,rng.choice(background,n,replace=False)))
                        else:
                            pos = selected[target]
                        if layer == layers[0] and condition in ('target', 'other_object', 'background'):
                            overlay(images['recipient'], pos, h, w, merge, audit/f'patch-{condition}-target{target}.png')
                        seed = config['seed']+layer*10+target
                        value, delta = replacement(caps['recipient'][layer], caps['color_swap'][layer], pos, condition, seed)
                        answers = []
                        for side in (0,1):
                            with patch_block(visual.blocks[layer], pos, value, h*w):
                                raw = model.answer(inputs['recipient',side])
                            answers.append(raw)
                        if condition=='self' and any(answers[s]!=clean['recipient',s] for s in (0,1)):
                            raise RuntimeError(f'{fid} layer {layer}: exact self-patch failed')
                        row = {'trial_key': f'{unit}:{condition}:{target}', 'family': fid, 'layer': layer,
                            'condition': condition, 'target_side': target, 'positions': pos, 'token_count': len(pos),
                            'module': f'{path}.blocks.{layer}', 'all_channels': True,
                            'replacement_delta_norm_float32': delta,
                            'random_seed': seed if condition=='random' else None,
                            'answers': answers, 'parsed_colors': [color(a) for a in answers],
                            'target_transferred': color(answers[target])==pairs['color_swap']['objects'][target]['color'],
                            'neighbor_preserved': color(answers[1-target])==rec['objects'][1-target]['color'],
                            'recipient_image': rec['image'], 'donor_image': pairs['color_swap']['image']}
                        rows.append(row)
                        atomic_json(out/'diagnostics'/f'{unit}.json', rows)
                        print(f'{unit} {condition} target={target}: {answers}', flush=True)
                store.save(unit, rows)
                store.export()
        results = store.export()
        totals = []
        for layer in layers:
            for condition in ('target','random','other_object','background'):
                rows = [r for r in results if r.get('layer')==layer and r['condition']==condition]
                totals.append({'layer': layer,'condition': condition,'n':len(rows),
                    'specific_transfers':sum(r['target_transferred'] and r['neighbor_preserved'] for r in rows)})
        atomic_json(out/'summary.json', {'counts': totals, 'interpretation': 'Exploratory spatial color transfer. Repeated sides/layouts are not independent scenes; shape binding untested.'})
        from .vision_report import report
        report(out)
        status = {'state': 'complete', 'rows': len(results)}
    except BaseException as exc:
        store.export()
        status = {'state': 'error', 'error': str(exc)}
        raise
    finally:
        atomic_json(out/'status.json', status)
