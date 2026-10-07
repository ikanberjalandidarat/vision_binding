"""Synthetic unit data only; these tests do not validate Minecraft or a GPU model."""
import pytest
import torch
from mc_binding.perception_study import rasterize, region_weights, features_for, fit, evaluate
from mc_binding.vision_data import token_cells


def test_token_order():
    cells=token_cells(4,4,2)
    t=torch.tensor([[r*4+c] for r,c in cells],dtype=torch.float32)
    assert torch.equal(rasterize(t,4,4,2)[...,0],torch.arange(16).reshape(4,4))


def test_fractional_region():
    w=region_weights([49,0,51,100],(100,100),(1,2))
    assert torch.allclose(w,torch.tensor([[.5,.5]]))
    with pytest.raises(ValueError):region_weights([-1,0,2,2],(100,100),(4,8))


def test_native_pool_and_uniform_resolution():
    objects=[dict(bbox_raw=[0,0,2,4]),dict(bbox_raw=[2,0,4,4])]
    r=torch.arange(16,dtype=torch.float32).reshape(4,4,1)
    assert torch.allclose(features_for(r,objects,(4,4),'native')[:,0],torch.tensor([6.5,8.5]))
    for m in ('4x8','8x16','native'):
        assert torch.allclose(features_for(torch.ones(16,32,3),objects,(4,4),m),torch.ones(2,3))


def test_tiny_memorization_and_absent_conjunction():
    torch.set_num_threads(1)
    x=torch.tensor([[-1.,-1.],[-1.,1.],[1.,-1.],[1.,1.]])
    c=torch.tensor([0,0,1,1]);s=torch.tensor([0,1,0,1]);model=fit(x,c,s,500)
    objects=[dict(color='red',type='arch'),dict(color='blue',type='pillar')]
    rows=[dict(x=x[[0,3]],family='synthetic',record=dict(record_id='synthetic',objects=objects))]
    result=evaluate(model,rows,.5)
    assert all(r['correct'] for r in result['rows'])
    assert sum(r['expected']==-1 for r in result['rows'])==2
