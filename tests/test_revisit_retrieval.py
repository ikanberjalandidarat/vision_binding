"""NON-MINECRAFT oracle vectors test retrieval logic only."""
import numpy as np
from mc_binding.revisit_retrieval import evaluate


def test_oracle_site_identity_both_directions_all_heading_contrasts():
    rows=[dict(route=r,site=s,target_heading=h,position=[s%4,s//4]) for r in (4,5) for s in range(16) for h in (0,90,180,270)]
    x=np.array([np.eye(16)[r['site']] for r in rows])
    for delta in (0,90,180):
        result=evaluate(rows,x,delta)
        assert result['accuracy']==1
        assert result['mean_error_blocks']==0
        assert result['queries']==(256 if delta==90 else 128)
        assert result['geometry']['distinct_unordered_site_pairs']==120
    assert evaluate(rows,x,180,pixel_mse=True)['accuracy']==1
    y=x.copy()
    for i,r in enumerate(rows):
        if r['route']==5:y[i]=np.eye(16)[(r['site']+1)%16]
    assert evaluate(rows,y,180)['accuracy']==0
