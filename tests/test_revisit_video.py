"""NON-MINECRAFT array/metadata fixtures; no real rendering/model gate implied."""
import numpy as np
import pytest
from mc_binding.revisit_video import mask_ui, clip_ticks, protocol


def test_past_only_endpoint_clock():
    v=dict(endpoint_tick=200,clip_ticks=list(range(80,201,8)))
    assert len(clip_ticks(v))==16
    v['clip_ticks'][-1]=208
    with pytest.raises(ValueError):clip_ticks(v)


def test_fixed_mask_not_inplace_and_temporal_control_equivalence():
    x=np.zeros((280,448,3),dtype=np.uint8)
    y=mask_ui(x)
    assert not x.any()
    assert (y[:44]==127).all()
    assert (y[247:]==127).all()
    assert (y[200:,300:]==127).all()
    assert (y[140,224]==127).all()
    assert (y[100,100]==0).all()
    assert np.array_equal(mask_ui(x),y)
    assert len(protocol()['route_splits']['test'])==2
