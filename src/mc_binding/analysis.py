import json
import math
from collections import defaultdict
from pathlib import Path
import numpy as np
from .io import atomic_json


def wilson(k, n):
    if n == 0:
        return [None, None]
    z = 1.959963984540054
    p, d = k/n, 1+z*z/n
    c = (p+z*z/(2*n))/d
    h = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return [max(0, c-h), min(1, c+h)]


def analyze(run, output=None):
    run = Path(run)
    manifest = json.loads((run/'manifest.json').read_text())
    # Atomic family files are authoritative even following interruption.
    rows = [r for p in sorted((run/'families').glob('*.json')) for r in json.loads(p.read_text())]
    if len({r['trial_key'] for r in rows}) != len(rows):
        raise ValueError('Duplicate trial keys')
    result = {'is_minecraft': manifest['is_minecraft'], 'rates': [], 'comparisons': [], 'baseline': []}
    for parser in ('strict', 'synonym'):
        clean = [r for r in rows if r['condition'] == 'clean' and r['status'] == 'ok']
        k = sum(r['clean_correct'][parser] for r in clean)
        result['baseline'].append({'parser': parser, 'correct': k, 'n': len(clean), 'wilson95': wilson(k, len(clean))})
        for subset in ('all', 'clean_correct'):
            groups = defaultdict(list)
            for r in rows:
                if r['address_flip'] and r['content_change'] and (subset == 'all' or r['clean_correct'][parser]):
                    groups[r['condition']].append(r)
            for condition, group in groups.items():
                ok = [r for r in group if r['status'] == 'ok']
                k = sum(r['scores'][parser]['flags']['recipient_at_donor_address'] for r in ok)
                n = len(ok)
                if len({r['scene_family_id'] for r in ok}) != n:
                    raise ValueError('Wilson rates require one trial per independent family')
                result['rates'].append({'parser': parser, 'subset': subset, 'condition': condition, 'eligible': len(group), 'evaluated': n, 'skipped': len(group)-n, 'switches': k, 'rate': k/n if n else None, 'wilson95': wilson(k, n), 'invalid': sum(r['scores'][parser]['status'] != 'valid' for r in ok)})
            for control in ('random_q', 'random_qv', 'v', 'qv'):
                maps = [{r['scene_family_id']: int(r['scores'][parser]['flags']['recipient_at_donor_address']) for r in groups[c] if r['status'] == 'ok'} for c in ('q', control)]
                paired = sorted(set(maps[0]) & set(maps[1]))
                if not paired:
                    continue
                delta = np.array([maps[0][k]-maps[1][k] for k in paired])
                rng = np.random.default_rng(731)
                boot = [float(rng.choice(delta, len(delta), replace=True).mean()) for _ in range(5000)]
                result['comparisons'].append({'parser': parser, 'subset': subset, 'comparison': 'q minus '+control, 'primary': control == 'random_q', 'paired_families': len(paired), 'unpaired_families': len(set(maps[0]) | set(maps[1]))-len(paired), 'difference': float(delta.mean()), 'bootstrap95': np.quantile(boot, [.025, .975]).tolist()})
    atomic_json(output or run/'analysis.json', result)
    return result
