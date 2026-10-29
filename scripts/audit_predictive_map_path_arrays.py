#!/usr/bin/env python3
"""Read-only path-array preflight. Never rescales data or runs a model."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def audit(weights):
    names = ['predictive-coder-path-images.npy', 'predictive-coder-path-positions.npy']
    missing = [name for name in names if not (weights/name).is_file()]
    if missing:
        return {'state': 'missing_inputs', 'missing': missing}
    files = {}
    for name in names:
        digest = hashlib.sha256()
        with (weights/name).open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024*1024), b''):
                digest.update(chunk)
        files[name] = {'bytes': (weights/name).stat().st_size, 'sha256': digest.hexdigest()}
    images, positions = [np.load(weights/name, mmap_mode='r', allow_pickle=False) for name in names]
    result = {'state': 'layout_check', 'files': files,
              'images_shape': list(images.shape), 'positions_shape': list(positions.shape),
              'images_dtype': str(images.dtype), 'positions_dtype': str(positions.dtype)}
    if images.ndim != 5 or images.shape[2:] != (3,64,64) or not 1 <= images.shape[1] <= 20:
        return {**result, 'state': 'unsupported_image_layout'}
    nframes = images.shape[0]*images.shape[1]
    if positions.shape[-1] != 3 or positions.size != nframes*3 or nframes < 190:
        return {**result, 'state': 'unaligned_positions'}
    p = np.asarray(positions).reshape(-1,3)[:, :2]
    finite = True
    lo, hi = float('inf'), float('-inf')
    for i in range(len(images)):
        a = np.asarray(images[i])
        finite = finite and bool(np.isfinite(a).all())
        lo, hi = min(lo, float(a.min())), max(hi, float(a.max()))
    indices = sorted({0,len(images)//4,len(images)//2,3*len(images)//4,len(images)-1})
    samples = np.stack([images[i] for i in indices])
    mean = np.array([121.6697,149.3242,154.9510],dtype=np.float32)/255
    std = np.array([40.7521,47.7267,103.2739],dtype=np.float32)/255
    rgb = samples*std[None,None,:,None,None]+mean[None,None,:,None,None]
    window=p[100:190]
    distances=np.linalg.norm(window[:,None,:]-window[None,:,:],axis=-1)
    return {**result, 'state': 'complete', 'frames': nframes,
            'all_images_and_positions_finite': finite and bool(np.isfinite(p).all()),
            'stored_range': [lo,hi], 'sample_sequences': indices,
            'diagnostic_inverse_environment_normalization_range': [float(rgb.min()),float(rgb.max())],
            'scope': 'Data audit only; inverse normalization is diagnostic, no scaling assumption or correction applied',
            'author_window_distinct_position_pairs': int(np.count_nonzero(distances)/2),
            'author_window_zero_distance_nonself_ordered_pairs': int(np.count_nonzero(distances==0)-90)}


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    report=audit(args.weights)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
