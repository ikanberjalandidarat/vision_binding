"""NON-MINECRAFT synthetic fixtures; no renderer or pretrained model runs."""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_predictive_map_reference import (
    ENV_MEAN, ENV_STD, environment_rgb, environment_scale_audit,
    restore_released_environment_scale, validate_saved_arrays,
)

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


def test_official_style_ten_frame_layout_is_accepted_without_model_run():
    # NON-MINECRAFT synthetic arrays: check shape logic only.
    images = np.zeros((6, 10, 3, 64, 64), dtype=np.float32)
    positions = np.zeros((6, 2), dtype=np.float32)
    result = validate_saved_arrays(images, positions, autoencoder=False)
    assert result == {"image_sequences": 6, "sequence_length": 10, "position_columns": 2}
    with pytest.raises(ValueError, match="align"):
        validate_saved_arrays(images, positions[:5], autoencoder=False)


def test_extra_255_scale_diagnostic_is_reversible():
    # NON-MINECRAFT synthetic RGB fixture: released-style over-scaled values.
    rgb = np.linspace(0, 1, 12, dtype=np.float32).reshape(3, 2, 2)
    released = (rgb / 255 - ENV_MEAN[:, None, None]) / ENV_STD[:, None, None]
    images = np.tile(released, (6, 10, 1, 1, 1))
    audit = environment_scale_audit(images)
    assert audit["rgb_max_near_one_over_255"]
    corrected = restore_released_environment_scale(images)
    np.testing.assert_allclose(environment_rgb(corrected[0, 0]), rgb, atol=2e-5)
