"""Synthetic NON-MINECRAFT tests. Real rendering and Qwen remain separate gates."""
import numpy as np
import pytest
import torch
from mc_binding.hippocampal import SpatialMemory,odometry,rate_maps,revisit_mask


def test_causal_prediction_and_gradients():
    torch.manual_seed(1);model=SpatialMemory(6,8)
    x=torch.randn(12,6);motion=torch.randn(12,3)*.1
    before=model(x,motion)[0]
    changed=x.clone();changed[5:]=torch.randn_like(changed[5:])*10
    after=model(changed,motion)[0]
    assert torch.equal(before[:6],after[:6])  # current input not used to predict itself
    loss=(before[1:]-x[1:]).square().mean();loss.backward()
    assert model.motion.weight_hh.grad.abs().sum()>0
    assert model.key.weight.grad.abs().sum()>0
    assert torch.isfinite(loss)


def test_odometry_translation_invariant_and_yaw_wrap():
    poses=[dict(x=0,z=0,yaw=179),dict(x=1,z=2,yaw=-179)]
    translated=[dict(p,x=p['x']+100,z=p['z']-40) for p in poses]
    assert torch.equal(odometry(poses),odometry(translated))
    assert odometry(poses)[1].tolist()==pytest.approx([1,2,np.deg2rad(2)])


def test_allocentric_maps_preserve_unknown_and_average_occupancy():
    poses=[dict(x=0,z=0,yaw=0),dict(x=0,z=0,yaw=180),dict(x=1,z=1,yaw=0)]
    maps,count=rate_maps(np.array([[1],[3],[9]]),poses,[-1,3,-1,3],4)
    assert maps[1,1,0]==2 and count[1,1]==2
    assert np.isnan(maps[0,0,0])
    assert maps[2,2,0]==9


def test_revisits_require_gap_and_view():
    poses=[dict(x=0,z=0,yaw=0),dict(x=2,z=2,yaw=0),dict(x=0,z=0,yaw=180),dict(x=0,z=0,yaw=0)]
    assert revisit_mask(poses,min_gap=2).tolist()==[False,False,False,True]


def test_no_mapping_pose_on_navigation_worker():
    from mc_binding.vla_env import Environment
    from pathlib import Path
    class Fake: episode=Path('final-greedy-0000')
    with pytest.raises(ValueError,match='forbidden'):Environment.mapping_pose(Fake())


def test_synthetic_end_to_end_fit_and_report(tmp_path):
    """Only the fitting/reporting contract, using explicitly non-Minecraft images."""
    from argparse import Namespace
    from PIL import Image
    from mc_binding.hippocampal_study import fit
    from mc_binding.io import file_hash
    import json
    entries=[];data=[]
    im=tmp_path/'non-minecraft.png';Image.new('RGB',(448,280),'gray').save(im)
    torch.manual_seed(5)
    for k,split in enumerate(['train','validation','test']):
        for route in range(2):
            eid=f'non-minecraft-{k}-{route}'
            poses=[dict(x=float(i%4),z=float(i//4),yaw=0) for i in range(12)]
            entries.append(dict(episode=eid,split=split,route=route,objects=[],route_complete=False,
                frames=[dict(frame=im.name,pose=p,sha256=file_hash(im)) for p in poses]))
            data.append(dict(x=torch.randn(12,64),motion=odometry(poses)))
    fit(Namespace(seed=731,epochs=2),tmp_path,entries,data)
    review=json.loads((tmp_path/'review.json').read_text())
    assert review['train_episodes']==['non-minecraft-0-0']
    assert {r['split'] for r in review['metrics']}=={'train','heldout_route','validation','test'}
    assert (tmp_path/'analysis/non-minecraft-2-1.gif').exists()
    assert all(r['revisit_mse'] is None for r in review['metrics'])
