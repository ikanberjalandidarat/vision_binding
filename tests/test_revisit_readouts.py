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


def test_non_minecraft_balanced_loss_mass():
    from mc_binding.revisit_readouts import balanced_pair_weights
    rows=[dict(site=s,target_heading=h) for route in range(2) for s in range(3) for h in (0,90,180,270)]
    w=balanced_pair_weights(rows)
    i,j=np.triu_indices(len(rows),1)
    same=np.array([rows[a]['site']==rows[b]['site'] for a,b in zip(i,j)])
    different_heading=np.array([rows[a]['target_heading']!=rows[b]['target_heading'] for a,b in zip(i,j)])
    assert np.isclose(w[same&different_heading].sum(),.5)
    assert np.isclose(w[~same].sum(),.5)
    assert (w[same&~different_heading]==0).all()


def test_non_minecraft_constant_distance_baseline():
    rows=[dict(route=r,site=s,target_heading=h,position=[s%4,s//4]) for r in (4,5) for s in range(16) for h in (0,90,180,270)]
    x=np.array([r['position'] for r in rows])
    result=score(rows,x,[4,5],180,constant=0.)
    assert np.isclose(result['constant_distance_mae_blocks'],result['random_error_blocks'])


def test_non_minecraft_uniform_weights_preserve_original_loss():
    x=np.random.default_rng(7).normal(size=(12,4));y=x[:,:2]
    for method in ('orth','linear_metric'):
        a=fit(x,y,(method,2,None),5)
        b=fit(x,y,(method,2,None),5,np.ones(66))
        np.testing.assert_allclose(project(a,x),project(b,x),atol=1e-10)
