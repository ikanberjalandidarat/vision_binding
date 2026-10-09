"""Odometry-conditioned associative memory, inspired by TEM-t; not a reproduction.

'g' is a learned motion state; 'p' is a learned sensory/spatial conjunction.
Neither name establishes biological cell identity. No absolute pose enters forward.
"""
import numpy as np
import torch
from torch import nn


def odometry(poses):
    """World-axis displacement + yaw change; privileged measured odometry, not GPS."""
    a=np.array([[p['x'],p['z'],np.deg2rad(p['yaw'])] for p in poses],dtype=np.float64)
    if len(a)<2 or not np.isfinite(a).all():raise ValueError('Need finite pose sequence')
    d=np.zeros_like(a);d[1:]=np.diff(a,axis=0)
    d[:,2]=(d[:,2]+np.pi)%(2*np.pi)-np.pi
    return torch.tensor(d,dtype=torch.float32)


class SpatialMemory(nn.Module):
    def __init__(self,width=64,hidden=32):
        super().__init__()
        self.initial=nn.Parameter(torch.zeros(hidden))
        self.motion=nn.GRUCell(3,hidden)
        self.key=nn.Linear(hidden,hidden,bias=False)
        self.conjunction=nn.Linear(hidden+width,hidden)
        self.readout=nn.Linear(hidden,width)
        self.empty=nn.Parameter(torch.zeros(width))

    def forward(self,x,motion):
        if x.ndim!=2 or motion.shape!=(len(x),3):raise ValueError('Invalid sequence')
        g=self.initial;keys=[];pred=[];gs=[];ps=[];attentions=[]
        for t in range(len(x)):
            if t:g=self.motion(motion[t],g)
            key=torch.nn.functional.normalize(self.key(g),dim=-1)
            if keys:
                weights=torch.softmax(8*(torch.stack(keys)@key),dim=0)
                recalled=weights@x[:t]  # strictly past observations; no current/future leakage
            else:
                weights=x.new_empty(0);recalled=self.empty
            p=torch.tanh(self.conjunction(torch.cat([g,recalled])))
            pred.append(self.readout(p));gs.append(g);ps.append(p);attentions.append(weights)
            # Write key/value only AFTER making the current sensory prediction.
            keys.append(key)
        return torch.stack(pred),torch.stack(gs),torch.stack(ps),attentions


def revisit_mask(poses,min_gap=12,radius=.6,yaw_degrees=25):
    """Evaluator-only same-place-and-view test; never used to train representations."""
    xy=np.array([[p['x'],p['z']] for p in poses]);yaw=np.array([p['yaw'] for p in poses])
    result=np.zeros(len(poses),dtype=bool)
    for i in range(min_gap,len(poses)):
        dy=np.abs((yaw[:i-min_gap+1]-yaw[i]+180)%360-180)
        result[i]=np.any((np.linalg.norm(xy[:i-min_gap+1]-xy[i],axis=1)<radius)&(dy<yaw_degrees))
    return result


def rate_maps(activity,poses,bounds,bins=16):
    """Occupancy-normalized signed activations. Unvisited bins stay NaN, never zero."""
    a=np.asarray(activity);xy=np.array([[p['x'],p['z']] for p in poses]);x0,x1,z0,z1=bounds
    if a.ndim!=2 or len(a)!=len(xy) or x1<=x0 or z1<=z0:raise ValueError('Invalid map')
    ix=np.clip(((xy[:,0]-x0)/(x1-x0)*bins).astype(int),0,bins-1)
    iz=np.clip(((xy[:,1]-z0)/(z1-z0)*bins).astype(int),0,bins-1)
    count=np.zeros((bins,bins),int);total=np.zeros((bins,bins,a.shape[1]))
    for i,j,v in zip(iz,ix,a):count[i,j]+=1;total[i,j]+=v
    maps=np.full_like(total,np.nan)
    np.divide(total,count[...,None],out=maps,where=count[...,None]>0)
    return maps,count
