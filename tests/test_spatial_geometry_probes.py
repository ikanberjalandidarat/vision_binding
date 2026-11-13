"""NON-MINECRAFT fixtures; these do not validate a real V-JEPA result."""
import numpy as np
import pytest
import torch
torch.set_num_threads(2)
from mc_binding.spatial_geometry_probes import fit_probe,project,evaluate


def fixture():
    r=np.random.default_rng(7)
    x=r.normal(size=(24,12)); y=x[:,:2]*3
    return x,y


def test_orthogonal_constraint_and_repeatability():
    x,y=fixture()
    a=fit_probe(x[:16],y[:16],'orth',2,4,20)
    b=fit_probe(x[:16],y[:16],'orth',2,4,20)
    np.testing.assert_allclose(a['w'].T@a['w'],np.eye(2),atol=1e-8)
    np.testing.assert_allclose(project(a,x),project(b,x))


@pytest.mark.parametrize('method',['raw','random_orth','orth','linear_metric','ridge','mean_distance'])
def test_evaluation_is_finite_and_does_not_refit(method):
    x,y=fixture()
    model=fit_probe(x[:16],y[:16],method,2,1,10)
    before={k:v.copy() if isinstance(v,np.ndarray) else v for k,v in model.items()}
    score,arrays=evaluate(model,x[:16],y[:16],x[16:],y[16:],np.zeros(8))
    assert score['test_pairs']==28
    assert np.isfinite(score['distance_mae_blocks'])
    assert len(arrays['true_distance'])==28
    evaluate(model,x[:16],y[:16],x[16:]+100,y[16:]+200,np.zeros(8))
    for k,v in before.items():
        if isinstance(v,np.ndarray): np.testing.assert_array_equal(v,model[k])
        else: assert v==model[k]


def test_raw_calibration_preserves_distance_ratios():
    x,y=fixture();model=fit_probe(x[:16],y[:16],'raw')
    z=project(model,x[16:])
    ratio=np.linalg.norm(z[0]-z[1])/np.linalg.norm(z[2]-z[3])
    expected=np.linalg.norm(x[16]-x[17])/np.linalg.norm(x[18]-x[19])
    assert ratio==pytest.approx(expected)
