"""NON-MINECRAFT synthetic fixtures for within-world cached-feature scoring."""

import numpy as np
import pytest

from mc_binding.spatial_video_within_world import compare_world, heading_difference


def test_heading_wrap_and_same_place_across_views():
    assert heading_difference([350], [10])[0, 0] == 20
    # Three synthetic sites, revisited from the opposite heading. The middle
    # site differs in Y, so the evaluator must retain elevation.
    position = np.array([[0., 0., 0.], [0., 5., 0.], [10., 0., 0.]])
    features = np.eye(3)
    result = compare_world(features, features, position, position,
                           [0, 0, 0], [180, 180, 180])
    assert result["cross_view_queries"] == 3
    assert result["near_cross_view_pairs"] == 3
    assert result["hit_at_2_blocks"] == 1.0
    assert result["cross_view_hit_at_2_blocks"] == 1.0
    assert result["mean_excess_over_oracle"] == 0.0
    assert result["cross_view_near_vs_same_view_far_margin"] is None


def test_bad_matching_is_worse_than_oracle_and_nonfinite_is_rejected():
    position = np.array([[0., 0., 0.], [8., 0., 0.], [16., 0., 0.]])
    features = np.eye(3)
    bad = compare_world(features, features[[2, 0, 1]], position, position,
                        [0, 0, 0], [180, 180, 180])
    assert bad["hit_at_2_blocks"] == 0.0
    assert bad["mean_excess_over_oracle"] > 0
    with pytest.raises(ValueError, match="Nonfinite"):
        compare_world(features, np.full((3, 3), np.nan), position, position,
                      [0, 0, 0], [180, 180, 180])
