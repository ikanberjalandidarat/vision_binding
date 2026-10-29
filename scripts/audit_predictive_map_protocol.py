#!/usr/bin/env python3
"""Export pinned reference protocol as text/JSON; never import or execute it.

Reads small Git-tracked source files and notebook cell sources only. Git may
fetch missing source blobs in a partial clone. No models, data arrays, notebook
outputs, or installation cells are executed or changed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

REFERENCE_COMMIT = 'fed80b72925a4d3f3e9ec4b70f71ecaedec75c13'


def git(repo, *args):
    return subprocess.check_output(['git', '-C', str(repo), *args])


def audit(repo, output):
    commit = git(repo, 'rev-parse', 'HEAD').decode().strip()
    if commit != REFERENCE_COMMIT:
        raise ValueError(f'Expected {REFERENCE_COMMIT}; got {commit}')
    entries = []
    for line in git(repo, 'ls-tree', '-r', 'HEAD').decode().splitlines():
        metadata, path = line.split('\t', 1)
        mode, kind, oid = metadata.split()
        entries.append((mode, kind, oid, path))
    sources, notebooks, links = {}, {}, {}
    wanted_notebooks = {'notebooks/predictive_coding.ipynb',
                        'notebooks/autoencoding.ipynb', 'notebooks/circular.ipynb',
                        'notebooks/environment.ipynb'}
    for mode, kind, oid, path in entries:
        is_source = path.endswith('.py') and path.startswith(('predictive_coding/', 'notebooks/predictive_coding/'))
        is_link = mode == '120000' and path.startswith('notebooks/')
        if not (is_source or is_link or path in wanted_notebooks):
            continue
        blob = git(repo, 'show', f'{commit}:{path}')
        digest = hashlib.sha256(blob).hexdigest()
        if is_link:
            links[path] = {'target': blob.decode(), 'mode': mode, 'git_blob': oid}
        elif path in wanted_notebooks:
            notebook = json.loads(blob)
            cells = [{'index': i, 'type': cell['cell_type'],
                      'source': ''.join(cell.get('source', []))}
                     for i, cell in enumerate(notebook['cells'])]
            notebooks[path] = {'sha256': digest, 'cells': cells}
        else:
            sources[path] = {'sha256': digest, 'git_blob': oid, 'source': blob.decode()}
    missing = wanted_notebooks - notebooks.keys()
    if missing:
        raise ValueError(f'Missing required notebooks: {sorted(missing)}')
    report = {'state': 'complete', 'scope': 'Pinned source and notebook text only; no cells executed',
              'repo_commit': commit, 'notebook_links': links,
              'notebooks': notebooks, 'python_sources': sources}
    output.mkdir(parents=True, exist_ok=False)
    (output / 'protocol-source.json').write_text(json.dumps(report, indent=2)+'\n')
    lines = [f'Reference commit: {commit}', 'READ-ONLY SOURCE EXPORT; NO CELLS EXECUTED\n']
    for path, nb in notebooks.items():
        lines.append(f'NOTEBOOK: {path}')
        for cell in nb['cells']:
            lines.extend([f"CELL {cell['index']} ({cell['type']})", cell['source'], ''])
    (output / 'notebook-cells.txt').write_text('\n'.join(lines))
    print(json.dumps({'state': 'complete', 'output': str(output),
                      'notebooks': len(notebooks), 'python_sources': len(sources),
                      'notebook_links': links}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    audit(args.repo, args.output)
