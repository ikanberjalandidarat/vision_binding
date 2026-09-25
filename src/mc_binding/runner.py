import json
from pathlib import Path
from PIL import Image
from .dataset import validate
from .io import RunStore, atomic_json, digest, environment, source_hash
from .scenes import prompt
from .scoring import score


def run(dataset, output, config, patch=False, allow_fixture=False, model=None):
    data = validate(dataset)
    if not data['is_minecraft'] and not allow_fixture:
        raise ValueError('Fixture inference requires --allow-fixture; results are not Minecraft evidence')
    if model is None:
        from .models.qwen import Qwen
        model = Qwen(config)
    manifest = {'schema_version': '1.0', 'dataset_hash': digest(data), 'config': config, 'mode': 'patch' if patch else 'baseline', 'is_minecraft': data['is_minecraft'], 'environment': environment(), 'source_hash': source_hash(), 'model': model.manifest()}
    store = RunStore(output, manifest)
    atomic_json(Path(output)/'status.json', {'state': 'running'})
    try:
        for family in data['families']:
            if store.completed(family['family_id']):
                continue
            rows = run_family(Path(dataset), family, model, config, patch)
            from .schema import check
            for result in rows:
                check("trial", result)
            store.save(family['family_id'], rows)
        store.export()
        atomic_json(Path(output)/'status.json', {'state': 'complete'})
    except BaseException as exc:
        atomic_json(Path(output)/'status.json', {'state': 'interrupted' if isinstance(exc, KeyboardInterrupt) else 'error', 'error': str(exc)})
        store.export()
        raise
    return store


def run_family(root, family, model, config, patch):
    rec = family['contexts']['recipient']
    target = 0 if family['target_side'] == 'left' else 1
    task = config.get('task', 'pair')
    rprompt = prompt(family['target_side'], task)
    with Image.open(root/rec['image']) as im:
        rinp = model.inputs(im.convert('RGB'), rprompt)
    clean = model.answer(rinp)
    aliases = config.get('aliases', {})
    rows = []

    def row(condition, raw, changed=False, flip=False, donor=None, status='ok', reason=None, metadata=None):
        donor = rec if donor is None else donor
        selected = 1-target if flip else target
        scores, clean_correct = {}, {}
        for name, mapping in [('strict', None), ('synonym', aliases)]:
            scores[name] = score(raw, rec['objects'], donor['objects'], target, selected, mapping, task) if status == 'ok' else None
            clean_correct[name] = score(clean, rec['objects'], rec['objects'], target, target, mapping, task)['flags']['recipient_target']
        rows.append({'schema_version': '1.0', 'trial_key': f"{family['family_id']}:{condition}:{int(changed)}:{int(flip)}", 'scene_family_id': family['family_id'], 'condition': condition, 'address_flip': flip, 'content_change': changed, 'status': status, 'reason': reason, 'prompt': rprompt, 'donor_prompt': prompt(('left', 'right')[selected], task), 'target_id': rec['objects'][target]['object_id'], 'donor_target_id': donor['objects'][selected]['object_id'], 'recipient_objects': rec['objects'], 'donor_objects': donor['objects'], 'image_sha256': rec['image_sha256'], 'donor_image_sha256': donor['image_sha256'], 'raw': raw, 'scores': scores, 'clean_correct': clean_correct, 'patch_scope': 'prefill_only' if patch else 'none', 'patches': metadata or []})

    row('clean', clean)
    if not patch:
        return rows
    from .interventions import random_replacement
    requests = [(l, k) for l in config['layers'] for k in ('q', 'v')]
    rcaps = model.capture(rinp, requests)
    rpos, rlayout = model.roi(rinp, rec, target)
    last = rinp['input_ids'].shape[1]-1

    def specs_for(mode, dcaps, dinp, dpos, random=False, salt=0):
        specs = []
        for layer in config['layers']:
            for kind in ('q', 'v') if mode == 'qv' else (mode,):
                dst = [last] if kind == 'q' else rpos
                src = [dinp['input_ids'].shape[1]-1] if kind == 'q' else dpos
                value = dcaps[layer, kind][src].clone()
                ch = model.channels(layer, kind)
                if random:
                    value = random_replacement(rcaps[layer, kind][dst], value, ch, family['seed']+layer+salt+(1000 if kind == 'v' else 0))
                specs.append({'layer': layer, 'kind': kind, 'positions': dst, 'channels': ch, 'value': value})
        return specs

    # Abort before donor effects if any exact self replacement changes the answer.
    for mode in ('q', 'v', 'qv'):
        answer = model.answer(rinp, specs_for(mode, rcaps, rinp, rpos))
        if answer != clean:
            raise RuntimeError(f"Self-{mode} patch changed output for {family['family_id']}")
        row('self_'+mode, answer)
    for changed in (False, True):
        donor = family['contexts']['disjoint'] if changed else rec
        for flip in (False, True):
            selected = 1-target if flip else target
            with Image.open(root/donor['image']) as im:
                dinp = model.inputs(im.convert('RGB'), prompt(('left', 'right')[selected], task))
            dcaps = model.capture(dinp, requests)
            # Match physical ROI at the recipient's original side, as in notebook F.
            dpos, dlayout = model.roi(dinp, donor, target)
            compatible = rlayout == dlayout and len(dpos) == len(rpos)
            for mode in ('q', 'v', 'qv', 'random_q', 'random_qv'):
                base = mode.removeprefix('random_')
                if base in ('v', 'qv') and not compatible:
                    row(mode, '', changed, flip, donor, 'skipped', 'Incompatible donor/recipient ROI token counts or grids')
                    continue
                specs = specs_for(base, dcaps, dinp, dpos, mode.startswith('random_'), 10000*int(changed)+20000*int(flip))
                metadata = [{k: v for k, v in s.items() if k != 'value'} for s in specs]
                row(mode, model.answer(rinp, specs), changed, flip, donor, metadata=metadata)
    return rows
