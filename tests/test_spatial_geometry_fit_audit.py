"""NON-MINECRAFT fixtures; validate audit instrumentation, not model claims."""
import numpy as np
import torch
import pytest
from mc_binding.spatial_geometry_probes import fit_probe,project
from mc_binding.spatial_geometry_fit_audit import audit_fit,positive_controls

torch.set_num_threads(2)

@pytest.mark.parametrize('method',['orth','linear_metric'])
def test_observer_preserves_original_fit(method):
    rng=np.random.default_rng(10);x=rng.normal(size=(18,12));y=x[:,:2]*2
    original=fit_probe(x,y,method,2,steps=200)
    captured=[]
    traced=fit_probe(x,y,method,2,steps=200,callback=lambda step,m:captured.append((step,project(m,x))),checkpoints=(0,10,200))
    assert [s for s,_ in captured]==[0,10,200]
    np.testing.assert_array_equal(project(original,x),project(traced,x))
    np.testing.assert_array_equal(captured[-1][1],project(original,x))


def test_test_labels_cannot_change_training_trace():
    rng=np.random.default_rng(2);x=rng.normal(size=(24,8));y=x[:,:2]
    a=audit_fit(x[:16],y[:16],x[16:],y[16:],'orth',2,steps=10)
    b=audit_fit(x[:16],y[:16],x[16:],y[16:]*100,'orth',2,steps=10)
    assert [r['train'] for r in a]==[r['train'] for r in b]
    assert a[-1]['heldout']!=b[-1]['heldout']


def test_known_linear_embedding_can_be_recovered():
    c=positive_controls()
    assert 'NON-MINECRAFT' in c['label']
    for name,curve in c['curves'].items():
        assert curve[-1]['train']['distance_mae']<.15,name
        assert curve[-1]['heldout']['distance_mae']<.2,name
