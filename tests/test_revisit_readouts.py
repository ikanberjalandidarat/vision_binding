"""NON-MINECRAFT synthetic coordinates test metric plumbing, not model ability."""
import numpy as np
from mc_binding.revisit_readouts import score, fit
from mc_binding.spatial_geometry_probes import project


def test_non_minecraft_perfect_coordinate_retrieval_and_units():
    rows=[];features=[]
    for route in range(6):
        for site in range(16):
            for h in (0,90,180,270):
                pos=[float(site%4)*4,float(site//4)*4]
                rows.append(dict(route=route,site=site,target_heading=h,position=pos))
                features.append(pos)
    x=np.array(features)
    for routes in ([2,3],[4,5]):
        for delta,count in ((0,128),(90,256),(180,128)):
            result=score(rows,x,routes,delta)
            assert result['queries']==count
            assert result['accuracy']==1
            assert result['distance_mae_blocks']==0
            assert result['mean_error_blocks']==0


def test_non_minecraft_ridge_training_only_transform():
    x=np.random.default_rng(4).normal(size=(20,4));y=x[:,:2]*3
    model=fit(x,y,('ridge',2,1.),1)
    before=project(model,x).copy()
    project(model,np.full((8,4),1e5))
    np.testing.assert_array_equal(project(model,x),before)
    assert np.mean(np.linalg.norm(before-y,axis=1))<1
