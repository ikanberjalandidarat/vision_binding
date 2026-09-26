"""Saved-RGB recognition diagnostics, with no simulator or label input to Qwen."""
from pathlib import Path
from PIL import Image
from .capture_pairs import load_pilot
from .io import RunStore, atomic_json, digest, environment, source_hash
from .scoring import parse


def probes(row):
    for i, obj in enumerate(row['objects']):
        reference = 'structure' if row['kind'] == 'isolated' else ('leftmost structure' if i == 0 else 'rightmost structure')
        for task, name in [('color', 'color'), ('type', 'type'), ('pair', 'color and type')]:
            yield task, obj, f'What {name} is the {reference}? Answer with {name} only.'


def recognize(dataset, output, config, reviewed=False, model=None):
    data = load_pilot(dataset)
    if not reviewed:
        raise ValueError('Inspect contact_sheet.png and raw frames for correct labels/cleanup, then pass --reviewed-captures')
    if model is None:
        from .models.qwen import Qwen
        model = Qwen(config)
    store = RunStore(output, {'schema_version': 'recognition_run_v1', 'is_minecraft': True,
        'dataset_hash': digest(data), 'config': config, 'environment': environment(),
        'source_hash': source_hash(), 'model': model.manifest(), 'visual_review_confirmed': True,
        'scope': 'recognition calibration, not causal binding evidence'})
    atomic_json(Path(output)/'status.json', {'state': 'running'})
    try:
        for record in data['records']:
            if store.completed(record['record_id']):
                continue
            rows = []
            with Image.open(Path(dataset)/record['image']) as source:
                image = source.convert('RGB')
            for task, obj, prompt in probes(record):
                raw = model.answer(model.inputs(image, prompt))
                scores = {}
                for parser, aliases in [('strict', None), ('synonym', config.get('aliases', {}))]:
                    parsed = parse(raw, aliases, task)
                    expected = {k: obj[k] for k in (('color', 'type') if task == 'pair' else (task,))}
                    scores[parser] = {**parsed, 'correct': parsed['parsed'] == expected}
                rows.append({'trial_key': f"{record['record_id']}:{obj['object_id']}:{task}",
                    'record_id': record['record_id'], 'scene_family_id': record['scene_family_id'],
                    'context': record['context'], 'kind': record['kind'], 'task': task,
                    'target_id': obj['object_id'], 'expected': expected, 'prompt': prompt,
                    'image_sha256': record['image_sha256'], 'raw': raw, 'scores': scores})
            store.save(record['record_id'], rows)
        rows = store.export()
        summary = []
        for parser in ('strict', 'synonym'):
            for kind in ('pair', 'isolated'):
                for task in ('color', 'type', 'pair'):
                    selected = [r for r in rows if r['kind'] == kind and r['task'] == task]
                    n = len(selected)
                    k = sum(r['scores'][parser]['correct'] for r in selected)
                    summary.append({'parser': parser, 'kind': kind, 'task': task, 'n': n,
                        'correct': k, 'accuracy': k/n if n else None,
                        'invalid': sum(r['scores'][parser]['status'] != 'valid' for r in selected)})
        atomic_json(Path(output)/'summary.json', {'scope': 'descriptive calibration; repeated configurations are not independent trials', 'rates': summary})
        atomic_json(Path(output)/'status.json', {'state': 'complete'})
    except BaseException as exc:
        store.export()
        atomic_json(Path(output)/'status.json', {'state': 'error', 'error': str(exc)})
        raise
    return store
