"""NON-MINECRAFT synthetic checks of offline audit logic."""
import importlib.util
from pathlib import Path
import numpy as np
import torch
from mc_binding.hippocampal import SpatialMemory
spec=importlib.util.spec_from_file_location('audit',Path(__file__).parents[1]/'scripts/analyze_hippocampal_cache.py')
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)


def test_constant_correlations_are_undefined():
    assert audit.correlations(np.ones((4,2)),np.ones((4,2))) == [None,None]
    x=np.arange(4)[:,None]
    assert audit.correlations(x,-x)==[-1.0]


def test_match_time_distance_heading_constraints():
    poses=[dict(x=0,z=0,yaw=0) for _ in range(14)]
    assert audit.matched_pairs(poses).tolist()==[[12,0],[13,0]]
    reverse=[dict(x=0,z=0,yaw=180)]
    assert len(audit.matched_pairs(poses,reverse,'same'))==0
    assert len(audit.matched_pairs(poses,reverse,'different'))==14
    assert len(audit.matched_pairs(poses,[dict(x=1,z=0,yaw=0)]))==0


def test_retrieval_intervention_has_no_current_or_future_leakage():
    torch.manual_seed(7); net=SpatialMemory(4,3);x=torch.randn(8,4);motion=torch.randn(8,3)
    g=net(x,motion)[1]
    before=audit.intervention(net,x,g,'past_mean')
    changed=x.clone();changed[4:]+=100
    after=audit.intervention(net,changed,g,'past_mean')
    assert torch.equal(before[:5],after[:5])
    assert not torch.equal(before[5:],after[5:])
