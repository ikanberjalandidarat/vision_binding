"""Synthetic annotation checks; no GPU or real visibility claims."""
import pytest
from mc_binding.live_perception import validate_annotation

def row(state='visible',box=None,reviewed=True):
 return dict(reviewed=reviewed,objects=[dict(object_id='synthetic',color='red',type='arch',visibility=state,bbox_raw=[1,2,10,20] if box is None else box)])

def test_review_required():
 with pytest.raises(ValueError):validate_annotation(row(reviewed=False),(448,280))

def test_visibility_and_bounds():
 assert len(validate_annotation(row(),(448,280)))==1
 assert validate_annotation(row('out_of_frame'),(448,280))==[]
 assert validate_annotation(row('uncertain'),(448,280))==[]
 with pytest.raises(ValueError):validate_annotation(row(box=[0,0,500,30]),(448,280))
 with pytest.raises(ValueError):validate_annotation(row('guessed'),(448,280))


def test_interval_sampling_keeps_endpoints_and_unique_middle():
 from mc_binding.live_perception import sample_steps
 assert sample_steps(1,8)==[0]
 assert sample_steps(17,8)==[0,8,16]
 assert sample_steps(18,8)==[0,8,9,16,17]
 assert sample_steps(10)==[0,5,9]
 with pytest.raises(ValueError):sample_steps(0,8)
 with pytest.raises(ValueError):sample_steps(10,-1)
