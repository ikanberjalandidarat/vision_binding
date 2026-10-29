#!/usr/bin/env python3
"""Run the pinned authors' path-distance notebook protocol, without notebook execution.

Keeps training-mode batch statistics, batch size 100, all timestep latents,
released path-array values, and the original distance/regression helpers.
This is descriptive retrospective mapping, not a held-out causal forecast test.
"""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
from run_predictive_map_reference import preflight

EXPECTED_HASHES = {
    'predictive_coding.ckpt': '356a2a7a51217030b5223e2b4f9f9a9e65d51717e042c03b7549afe205f156b0',
    'predictive-coder-path-images.npy': 'ae0327860c9dce5e208b1e42853877331ad5db20d7337d1cb34ff947a69bb4a8',
    'predictive-coder-path-positions.npy': '426cd4da80e699ddf46eac586c50d9a29b312389927f1fdf1896b16e03802cb8',
}


def validate_pair_masks(features, positions):
    """Gate independent author zero-removal; never silently replace its masks."""
    if len(features) != 90 or len(positions) != 90:
        raise ValueError('Expected the exact 90-point author window')
    fd = np.sqrt(np.sum((features[:, None]-features[None, :])**2, axis=-1))
    pd = np.sqrt(np.sum((positions[:, None]-positions[None, :])**2, axis=-1))
    if not np.isfinite(fd).all() or not np.isfinite(pd).all():
        raise ValueError('Nonfinite pair distances')
    if not np.array_equal(fd != 0, pd != 0):
        raise ValueError('Author independent zero-removal would misalign feature/position pairs')
    valid = fd[fd != 0]
    if not len(valid) or valid.max() == valid.min():
        raise ValueError('Degenerate latent normalization')
    return int(np.count_nonzero(pd)/2)


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    status = args.output/'status.json'
    try:
        write_json(status, {'state': 'running', 'stage': 'preflight'})
        paths, provenance = preflight(args.repo, args.weights, 'path')
        for name, expected in EXPECTED_HASHES.items():
            if provenance['official_files'][name]['sha256'] != expected:
                raise ValueError(f'Asset hash changed: {name}')
        import torch
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        sys.path.insert(0, str(args.repo.resolve()))
        from predictive_coding import PredictiveCoder
        from predictive_coding.analysis import generate_latents, Latents, LogRegression, regression_plot, qq_plot
        if not torch.cuda.is_available():
            raise RuntimeError('Real CUDA model run required')
        images = np.load(paths[1], mmap_mode='r', allow_pickle=False)
        positions = np.load(paths[2], allow_pickle=False)
        if images.shape != (8263,10,3,64,64) or positions.shape != (8263,10,3):
            raise ValueError('Path layouts differ from audited release')
        # The released notebook casts float16 arrays to float32 without rescaling.
        tensor = torch.from_numpy(np.array(images, dtype=np.float32)).to('cuda:0')
        model = PredictiveCoder(in_channels=3, out_channels=3, layers=[2,2,2,2], seq_len=20)
        model.load_state_dict(torch.load(paths[0], map_location='cpu', weights_only=True), strict=True)
        model.to('cuda:0')
        model.requires_grad_(False)
        # Explicitly preserve notebook default: no optimizer, but training-mode BN.
        model.train()
        write_json(status, {'state': 'running', 'stage': 'official_generate_latents', 'sequences': len(images)})
        print('Extracting all 8263 sequences in official batches of 100 (training-mode statistics)', flush=True)
        features = generate_latents(model, tensor).reshape(-1,128,8,8)
        del tensor
        torch.cuda.empty_cache()
        positions = positions.reshape(-1,3)[:, :2]
        if features.shape != (82630,128,8,8) or not np.isfinite(features).all():
            raise ValueError('Unexpected or nonfinite real latent output')
        distinct_pairs = validate_pair_masks(features[100:190].reshape(90,-1), positions[100:190])
        np.savez_compressed(args.output/'author-window.npz', latents=features[100:190],
                            positions=positions[100:190], flat_indices=np.arange(100,190))
        write_json(status, {'state': 'running', 'stage': 'official_distance_and_regression'})
        latent_set = Latents(features, positions)
        if (latent_set.L, latent_set.offset, latent_set.neighborhood) != (10000,100,90):
            raise ValueError('Official helper defaults changed')
        distances = latent_set.distances
        regression = LogRegression(1,1,latent_set,'Predictive Coding',plt.cm.tab10(0))
        regression.train(latent_set)
        prediction = regression.evaluate(distances['position'])
        if not np.isfinite(prediction).all():
            raise ValueError('Nonfinite regression fit')
        np.savez_compressed(args.output/'author-distances.npz', **distances)
        for name, plotter in [('distance-log-fit',regression_plot), ('distance-qq',qq_plot)]:
            fig = plotter([regression])
            fig.savefig(args.output/f'{name}.png', dpi=180, bbox_inches='tight')
            fig.savefig(args.output/f'{name}.pdf', bbox_inches='tight')
            plt.close(fig)
        report = {'state':'complete', **provenance,
                  'protocol':'pinned_released_notebook_path_distance',
                  'model':'authors PredictiveCoder; not V-JEPA',
                  'model_mode':'train, no_grad, no optimizer; batch statistics enabled',
                  'input_scale':'unchanged released path values cast float32',
                  'sequence_count':8263, 'sequence_length':10, 'feature_frames':len(features),
                  'batch_size':100, 'window_indices_inclusive':[100,189],
                  'distinct_unordered_pairs':distinct_pairs, 'distance_entries':len(prediction),
                  'alpha':float(regression.alpha), 'beta':float(regression.beta),
                  'weighted_fit_mse':float(np.mean((prediction-distances['latent'])**2/distances['position'])),
                  'physical_distance_range':[float(distances['position'].min()),float(distances['position'].max())],
                  'zero_masks_match':True,
                  'runtime':{'torch':torch.__version__, 'numpy':np.__version__, 'gpu':torch.cuda.get_device_name(0)},
                  'limits':['Repeated pairs are not independent observations',
                            'Matches released notebook code, not proof of original paper execution environment',
                            'Retrospective full-sequence features can depend on future frames and other batch members',
                            'No held-out spatial decoding, grid-cell or navigation claim',
                            'Earlier environment feature pass omitted; training-mode BN outputs use current batch statistics']}
        write_json(args.output/'report.json',report)
        write_json(status,{'state':'complete'})
        print(json.dumps(report,indent=2),flush=True)
    except BaseException as exc:
        write_json(status,{'state':'error','error':repr(exc)})
        raise


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',type=Path,required=True)
    parser.add_argument('--weights',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args())
