"""Color-only Q-transfer diagnostic on reviewed recognition captures.

Repeated layouts/target sides are calibration probes, not independent scenes.
No V/ROI patching, shape scoring or localization inference is performed.
"""
from pathlib import Path
from collections import defaultdict
import torch
from PIL import Image
from .capture_pairs import load_pilot
from .interventions import random_replacement
from .io import RunStore, atomic_json, digest, environment, source_hash
from .scoring import parse, score


def question(side=None):
    ref = 'structure' if side is None else ('leftmost structure' if side == 0 else 'rightmost structure')
    return f'What color is the {ref}? Answer with color only.'


def make_specs(model, layers, recipient, donor, rlast, dlast, random_seed=None):
    specs, metadata = [], []
    for layer in layers:
        base = recipient[layer, 'q'][rlast:rlast+1].clone()
        value = donor[layer, 'q'][dlast:dlast+1].clone()
        if base.shape != value.shape or not torch.isfinite(base).all() or not torch.isfinite(value).all():
            raise RuntimeError('Incompatible or non-finite Q activations')
        ch = model.channels(layer, 'q')
        norm = float((value[:, ch]-base[:, ch]).norm())
        if random_seed is not None:
            value = random_replacement(base, value, ch, random_seed+layer)
        applied_norm = float((value[:, ch]-base[:, ch]).norm())
        if random_seed is not None and abs(applied_norm-norm) > 1e-5*max(1., norm):
            raise RuntimeError('Random Q norm mismatch')
        specs.append({'layer': layer, 'kind': 'q', 'positions': [rlast], 'channels': ch, 'value': value})
        metadata.append({'layer': layer, 'kind': 'q', 'source_position': dlast,
                         'recipient_position': rlast, 'channels': ch,
                         'shape': list(value.shape), 'donor_delta_norm_float32': norm,
                         'applied_delta_norm_float32': applied_norm,
                         'random_seed': None if random_seed is None else random_seed+layer})
    return specs, metadata


def run_group(root, records, model, config, diagnostic_path):
    pairs = {r['context']: r for r in records if r['kind'] == 'pair'}
    rec, dis = pairs['recipient'], pairs['disjoint']
    if {o['color'] for o in rec['objects']} & {o['color'] for o in dis['objects']}:
        raise ValueError('Color-only diagnostic requires disjoint donor colors')
    if any(len({o['color'] for o in p['objects']}) != 2 for p in pairs.values()):
        raise ValueError('Each pair requires two distinct colors')
    fid = rec['scene_family_id']
    rows, inputs, clean = [], {}, {}
    for record in records:
        with Image.open(root/record['image']) as source:
            image = source.convert('RGB')
        for side, obj in enumerate(record['objects']):
            prompt = question(side if record['kind'] == 'pair' else None)
            inp = model.inputs(image, prompt)
            raw = model.answer(inp)
            parsed = parse(raw, task='color')
            correct = parsed['parsed'] == {'color': obj['color']}
            rows.append({'trial_key': f'{fid}:clean:{record["record_id"]}:{side}',
                         'scene_family_id': fid, 'condition': 'clean', 'kind': record['kind'],
                         'context': record['context'], 'side': side, 'prompt': prompt,
                         'image_sha256': record['image_sha256'], 'expected_color': obj['color'],
                         'target_id': obj['object_id'], 'raw': raw, 'parsed': parsed, 'correct': correct})
            if record['kind'] == 'pair':
                key = (record['context'], side)
                inputs[key], clean[key] = inp, raw
    # All eight color probes must pass. Stop instead of silently selecting good cases.
    atomic_json(diagnostic_path, {'stage': 'baseline', 'rows': rows})
    if not all(r['correct'] for r in rows):
        raise RuntimeError(f'{fid}: color baseline gate failed; inspect {diagnostic_path}')
    requests = [(l, 'q') for l in config['layers']]
    caps = {key: model.capture(inp, requests) for key, inp in inputs.items()}
    for target in (0, 1):
        rkey = ('recipient', target)
        rinp = inputs[rkey]
        rlast = rinp['input_ids'].shape[1]-1
        specs, meta = make_specs(model, config['layers'], caps[rkey], caps[rkey], rlast, rlast)
        raw = model.answer(rinp, specs)
        rows.append({'trial_key': f'{fid}:self_q:{target}', 'scene_family_id': fid,
                     'condition': 'self_q', 'side': target, 'raw': raw,
                     'clean_raw': clean[rkey], 'exact_match': raw == clean[rkey], 'patches': meta})
        atomic_json(diagnostic_path, {'stage': 'self_patch', 'rows': rows})
        if raw != clean[rkey]:
            raise RuntimeError(f'{fid}: exact self-Q patch failed; inspect {diagnostic_path}')
    # Both self-patches pass before any donor effects are measured.
    for target in (0, 1):
        rkey = ('recipient', target)
        rinp = inputs[rkey]
        rlast = rinp['input_ids'].shape[1]-1
        for changed in (False, True):
            donor = dis if changed else rec
            for flip in (False, True):
                selected = 1-target if flip else target
                dkey = (donor['context'], selected)
                dlast = inputs[dkey]['input_ids'].shape[1]-1
                for condition in ('q', 'random_q'):
                    seed = int(digest([config['seed'], fid, target, changed, flip])[:8], 16)
                    specs, meta = make_specs(model, config['layers'], caps[rkey], caps[dkey], rlast, dlast,
                                             seed if condition == 'random_q' else None)
                    raw = model.answer(rinp, specs)
                    scores = {name: score(raw, rec['objects'], donor['objects'], target, selected, aliases, 'color')
                              for name, aliases in [('strict', None), ('synonym', config.get('aliases', {}))]}
                    rows.append({'trial_key': f'{fid}:{target}:{changed}:{flip}:{condition}',
                        'scene_family_id': fid, 'condition': condition, 'side': target,
                        'address_flip': flip, 'content_change': changed, 'raw': raw,
                        'prompt': question(target), 'donor_prompt': question(selected),
                        'image_sha256': rec['image_sha256'], 'donor_image_sha256': donor['image_sha256'],
                        'recipient_objects': rec['objects'], 'donor_objects': donor['objects'],
                        'target_id': rec['objects'][target]['object_id'],
                        'donor_target_id': donor['objects'][selected]['object_id'],
                        'clean_raw': clean[rkey], 'clean_correct': True, 'scores': scores,
                        'patch_scope': 'prefill_last_prompt_q', 'patches': meta})
                    atomic_json(diagnostic_path, {'stage': 'donor_patch', 'rows': rows})
                    print(f'{fid} side={target} changed={changed} flip={flip} {condition}: {raw!r}', flush=True)
    return rows


def summary(rows):
    result = {'scope': 'descriptive pilot only; repeated sides/layouts are not independent scenes',
              'clean_probes': sum(r['condition'] == 'clean' for r in rows),
              'self_patches': sum(r['condition'] == 'self_q' for r in rows), 'primary': []}
    for parser in ('strict', 'synonym'):
        paired = {}
        for condition in ('q', 'random_q'):
            selected = [r for r in rows if r['condition'] == condition and r['content_change'] and r['address_flip']]
            k = sum(r['scores'][parser]['flags']['recipient_at_donor_address'] for r in selected)
            result['primary'].append({'parser': parser, 'condition': condition, 'n_probes': len(selected),
                'recipient_selection_transfers': k, 'rate': k/len(selected) if selected else None,
                'donor_selected_color': sum(r['scores'][parser]['flags']['donor_at_donor_address'] for r in selected),
                'invalid': sum(r['scores'][parser]['status'] != 'valid' for r in selected)})
            paired[condition] = {(r['scene_family_id'], r['side']): int(r['scores'][parser]['flags']['recipient_at_donor_address']) for r in selected}
        if set(paired['q']) != set(paired['random_q']):
            raise ValueError('Missing paired random control')
        values = [paired['q'][key]-paired['random_q'][key] for key in paired['q']]
        result.setdefault('paired_differences', []).append({'parser': parser, 'n_probes': len(values),
            'q_minus_random_q': sum(values)/len(values) if values else None})
    return result


def q_pilot(dataset, output, config, reviewed=False, model=None):
    data = load_pilot(dataset)
    if not reviewed:
        raise ValueError('Review paired/isolated captures and screen order, then pass --reviewed-captures')
    if config.get('load_in_4bit') or config.get('dtype') != 'bfloat16' or config.get('use_fast') is not True:
        raise ValueError('Use the validated unquantized BF16/fast-processor config for this pilot')
    layers = config.get('layers', [])
    if not layers or len(set(layers)) != len(layers) or any(not isinstance(l, int) or l < 0 for l in layers):
        raise ValueError('Specify unique nonnegative decoder layers')
    config = {**config, 'task': 'color', 'check_finite_scores': True}
    if model is None:
        from .models.qwen import Qwen
        model = Qwen(config)
    store = RunStore(output, {'schema_version': 'q_color_pilot_v1', 'config': config,
        'dataset_hash': digest(data), 'environment': environment(), 'source_hash': source_hash(),
        'model': model.manifest(), 'is_minecraft': True, 'reviewed_captures': True,
        'scope': 'prefill-only Q color calibration; not an independent-world effect estimate',
        'random_control': 'one seeded draw per context; per-layer norm matched in float32 before model-dtype cast'})
    groups = defaultdict(list)
    for row in data['records']:
        groups[row['scene_family_id']].append(row)
    atomic_json(Path(output)/'status.json', {'state': 'running'})
    try:
        for fid, records in groups.items():
            if store.completed(fid):
                continue
            # Generated manifests use fNNNN; disallow unsafe checkpoint filenames.
            if not fid.startswith('f') or not fid[1:].isdigit():
                raise ValueError('Invalid pilot family ID')
            rows = run_group(Path(dataset), records, model, config, Path(output)/'diagnostics'/f'{fid}.json')
            store.save(fid, rows)
        rows = store.export()
        atomic_json(Path(output)/'summary.json', summary(rows))
        atomic_json(Path(output)/'status.json', {'state': 'complete'})
    except BaseException as exc:
        store.export()
        atomic_json(Path(output)/'status.json', {'state': 'error', 'error': str(exc)})
        raise
    return store
