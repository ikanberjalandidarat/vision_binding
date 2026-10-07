"""Synthetic camera geometry, not Minecraft rendering validation."""
import numpy as np
from mc_binding.live_projection import basis,raycast,projected_box
POSE=dict(x=.5,y=0,z=0,yaw=0,pitch=0)

def test_camera_basis():
 r,d,f=basis(POSE)
 assert np.allclose(r,[-1,0,0])
 assert np.allclose(d,[0,-1,0])
 assert np.allclose(f,[0,0,1])

def test_depth_occlusion():
 objs=[dict(blocks=[[0,0,3]]),dict(blocks=[[0,0,6]])]
 m=raycast(objs,POSE,20,.5,(40,40))
 assert m[20,20]==0
 assert not (m==1).any()

def test_behind_camera():
 assert not (raycast([dict(blocks=[[0,0,-4]])],POSE,20,.5,(40,40))>=0).any()

def test_projection_horizontal_direction():
 a=projected_box(dict(blocks=[[2,0,10]]),POSE,200,.5)
 assert a[2]<224
