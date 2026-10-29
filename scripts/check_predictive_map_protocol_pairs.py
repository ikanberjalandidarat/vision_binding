#!/usr/bin/env python3
"""Check exported author distance helper on NON-MINECRAFT numerical fixtures.

Executes only the Latents class and Euclidean distance function extracted from
analysis.py, never notebook cells, imports, installation code or model code.
"""
import argparse
import ast
from dataclasses import dataclass
from functools import cached_property
import hashlib
import json
from pathlib import Path
from typing import Dict
import numpy as np


def check(source_file):
    report = json.loads(Path(source_file).read_text())
    item = report['python_sources']['predictive_coding/analysis.py']
    source = item['source']
    assert hashlib.sha256(source.encode()).hexdigest() == item['sha256']
    tree = ast.parse(source)
    selected = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef))
                and n.name in ('Latents', 'euclidean_distance')]
    assert len(selected) == 2
    namespace = dict(np=np, dataclass=dataclass, cached_property=cached_property,
                     Dict=Dict, DistFn=object)
    exec(compile(ast.Module(body=selected, type_ignores=[]), 'exported-analysis.py', 'exec'), namespace)
    Latents = namespace['Latents']
    # Deliberately artificial, unique coordinates/features. NOT Minecraft data.
    t = np.arange(200, dtype=float)
    positions = np.stack([t, t*t/100], axis=1)
    features = np.stack([t*t/73, np.sin(t), t/5], axis=1)
    ref = Latents(features, positions)
    ids = np.arange(ref.offset, ref.offset+ref.neighborhood)
    pairs = np.concatenate([np.stack([ids, np.roll(ids, shift-ref.neighborhood//2)], axis=1)
                            for shift in range(1, ref.L//2)])
    self_count = int(np.sum(pairs[:, 0] == pairs[:, 1]))
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]
    pd = np.linalg.norm(positions[pairs[:, 0]]-positions[pairs[:, 1]], axis=1)
    ld = np.linalg.norm(features[pairs[:, 0]]-features[pairs[:, 1]], axis=1)
    idx = np.argsort(pd)
    expected_position = pd[idx]
    expected_latent = ((ld-ld.min())/(ld.max()-ld.min()))[idx]
    actual = ref.distances
    np.testing.assert_allclose(actual['position'], expected_position, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(actual['latent'], expected_latent, rtol=1e-12, atol=1e-12)
    unique_ordered, counts = np.unique(pairs, axis=0, return_counts=True)
    unique_unordered = np.unique(np.sort(pairs, axis=1), axis=0)
    return {'state': 'complete', 'scope': 'NON-MINECRAFT numerical fixture; not a model or dataset result',
            'repo_commit': report['repo_commit'], 'analysis_sha256': item['sha256'],
            'defaults': {'L': ref.L, 'offset': ref.offset, 'neighborhood': ref.neighborhood},
            'selected_flat_indices_inclusive': [int(ids[0]), int(ids[-1])],
            'zero_self_entries_removed': self_count, 'retained_entries': len(pairs),
            'unique_ordered_pairs': len(unique_ordered), 'unique_unordered_pairs': len(unique_unordered),
            'ordered_pair_repeat_min_max': [int(counts.min()), int(counts.max())],
            'matches_author_distance_and_normalization_order': True,
            'real_data_gate': 'Compare position and latent nonzero masks before author helper. Independent zero removal can misalign pairs; do not silently change the protocol.',
            'inference_limit': 'Repeated entries are not independent evidence; fixture parity is not replication of a paper result.'}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    result = check(args.source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
