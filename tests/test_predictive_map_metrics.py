"""NON-MINECRAFT synthetic fixtures; no renderer or pretrained model runs."""

import numpy as np
import pytest

from mc_binding.predictive_map_metrics import (
    circular_positions, distance_summary, pair_distances, ridge_position_decode,
)


def test_wrapped_corridor_and_distance_ranks():
    p = np.array([[0., 0.], [0., 29.], [0., 15.]])
    wrapped = circular_positions(p)
    assert np.linalg.norm(wrapped[0] - wrapped[1]) < 2
    assert np.linalg.norm(wrapped[0] - wrapped[2]) > 9
    _, physical = pair_distances(wrapped, p, circular_period=30)
    assert physical.min() == pytest.approx(1)
    latent, physical = pair_distances(wrapped, wrapped)
    assert distance_summary(latent, physical)["spearman_latent_vs_physical"] == pytest.approx(1)


def test_heldout_position_decoder_uses_training_rows_only():
    x = np.arange(20., dtype=float).reshape(-1, 1)
    p = np.column_stack((x[:, 0], np.zeros(20)))
    baseline, _ = ridge_position_decode(x[:16], p[:16], x[16:], p[16:], alpha=.01)
    changed, _ = ridge_position_decode(x[:16], p[:16], x[16:], p[16:] + 100, alpha=.01)
    assert baseline["mean_error"] < 1
    assert changed["mean_error"] > 100
    assert baseline["train_rows"] == 16


def test_nonfinite_features_rejected():
    with pytest.raises(ValueError, match="finite"):
        pair_distances(np.array([[0.], [1.], [np.nan]]), np.arange(3.)[:, None])
