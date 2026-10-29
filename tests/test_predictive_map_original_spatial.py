"""NON-MINECRAFT synthetic geometry fixture; no renderer or pretrained model run."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_predictive_map_original_spatial import local_pairs


def test_local_pairs_separate_spatial_geometry_from_elapsed_time():
    theta = np.linspace(0, 4 * np.pi, 100, dtype=np.float32)
    position = np.column_stack((np.cos(theta), np.sin(theta))) * 10
    latent = position.copy()
    pixels = np.arange(100, dtype=np.float32)[:, None]
    metrics, _ = local_pairs(latent, pixels, position)
    assert metrics["pairs"] > 1000
    assert metrics["latent_vs_physical_spearman"] > .999
    assert metrics["rgb_vs_physical_spearman"] < .8
    assert metrics["by_temporal_separation"]["31to60"]["latent_spearman"] > .999


def test_path_checkpoint_survives_interrupted_replacement(tmp_path, monkeypatch):
    import pytest
    import run_predictive_map_original_spatial as spatial
    features = np.arange(12, dtype=np.float32).reshape(4, 3)
    states = features + 10
    pairs = (np.arange(4), np.arange(4) + 1, np.arange(4) + 2)
    path = tmp_path / 'path-000.npz'
    spatial.save_path(path, {'path_index': 0}, features, states, pairs)
    def fail(*args, **kwargs):
        raise OSError('NON-MINECRAFT simulated interrupted write')
    monkeypatch.setattr(spatial.np, 'savez_compressed', fail)
    with pytest.raises(OSError):
        spatial.save_path(path, {'path_index': 1}, features * 2, states, pairs)
    metrics, saved, saved_states, saved_pairs = spatial.load_path(path)
    assert metrics == {'path_index': 0}
    np.testing.assert_array_equal(saved, features)
    np.testing.assert_array_equal(saved_states, states)
    for actual, expected in zip(saved_pairs, pairs):
        np.testing.assert_array_equal(actual, expected)


def test_failure_status_and_resume_identity(tmp_path, monkeypatch):
    import argparse
    import json
    import pytest
    import run_predictive_map_original_spatial as spatial
    manifest, weights = tmp_path / 'manifest.json', tmp_path / 'fake-weights'
    manifest.write_text('{}')
    weights.write_text('NON-MINECRAFT fixture, not a checkpoint')
    args = argparse.Namespace(output=tmp_path / 'output', manifest=manifest,
                              checkpoint=weights, resume=False)
    def fail(args):
        spatial.progress(args.output, stage='loading_episode', episode='synthetic')
        raise RuntimeError('NON-MINECRAFT simulated failure')
    monkeypatch.setattr(spatial, 'run', fail)
    with pytest.raises(RuntimeError):
        spatial.execute(args)
    status = json.loads((args.output / 'status.json').read_text())
    assert status['state'] == 'error'
    assert status['stage'] == 'loading_episode'
    assert 'simulated failure' in status['traceback']
    args.resume = True
    monkeypatch.setattr(spatial, 'run', lambda args: None)
    spatial.execute(args)
    status = json.loads((args.output / 'status.json').read_text())
    assert status['state'] == 'complete'
    assert 'error' not in status and 'traceback' not in status
    manifest.write_text('{"changed": true}')
    with pytest.raises(ValueError, match='Resume requires identical'):
        spatial.execute(args)
