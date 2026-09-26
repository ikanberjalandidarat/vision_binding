"""Interactive diagrams drawn from saved trial positions and verified capture images."""
import base64
import json
from pathlib import Path
from .io import digest, file_hash


def explainer(root, rows, dataset=None):
    manifest = json.loads((root/'manifest.json').read_text())
    candidates = [Path(dataset)] if dataset else [p.parent for p in root.parent.glob('*/manifest.json')]
    capture = None
    for path in candidates:
        data = json.loads((path/'manifest.json').read_text())
        if digest(data) == manifest['dataset_hash']:
            capture = (path, data)
            break
    if capture is None:
        if dataset:
            raise ValueError('Report dataset does not match recorded dataset hash')
        return '<p>Interactive image comparison requires the matching capture dataset beside this run, or --dataset.</p>'
    path, data = capture
    families = {}
    for row in data['records']:
        if row['kind'] != 'pair':
            continue
        image_path = (path/row['image']).resolve()
        if path.resolve() not in image_path.parents or file_hash(image_path) != row['image_sha256']:
            raise ValueError('Report image hash/path mismatch')
        families.setdefault(row['scene_family_id'], {})[row['context']] = {
            'image': 'data:image/png;base64,'+base64.b64encode(image_path.read_bytes()).decode(),
            'objects': row['objects']}
    for fid, family in families.items():
        family['tokens'] = json.loads((root/'alignment'/fid/'tokens.json').read_text())
    model = json.loads((root/'model.json').read_text())
    trials = []
    for row in rows:
        if 'family' not in row:
            continue
        group = row.get('layer_set', [row['layer']])
        trials.append({**row, 'layer_set': group,
                       'layer': row['layer'] if len(group)==1 else ' + '.join(map(str, group))})
    payload = {'families': families, 'rows': trials,
               'vision': model.get('model_config', {}).get('vision_config', {}),
               'layers': list(dict.fromkeys(r['layer'] for r in trials))}
    template = Path(__file__).with_name('vision_explainer.html').read_text()
    return template.replace('__TRIAL_DATA__', json.dumps(payload).replace('<','\\u003c'))
