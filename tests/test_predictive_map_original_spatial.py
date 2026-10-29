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
