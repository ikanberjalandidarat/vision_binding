"""NON-MINECRAFT numerical fixtures for the real-data alignment gate."""
import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from run_predictive_map_author_distance import validate_pair_masks


def test_unique_pairs():
    t=np.arange(90,dtype=float)
    p=np.stack([t,t*t],axis=1)
    assert validate_pair_masks(p/100,p)==4005


def test_coordinate_duplicate_without_feature_duplicate_fails():
    t=np.arange(90,dtype=float)
    p=np.stack([t,t*t],axis=1)
    f=p.copy()
    p[1]=p[0]
    with pytest.raises(ValueError,match='misalign'):
        validate_pair_masks(f,p)


def test_equal_nonzero_feature_distances_fail_normalization():
    t=np.arange(90,dtype=float)
    p=np.stack([t,t*t],axis=1)
    with pytest.raises(ValueError,match='Degenerate'):
        validate_pair_masks(np.eye(90),p)
