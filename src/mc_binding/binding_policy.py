"""Predicted target-map input to a trainable actor-critic; no oracle regions."""
import torch
from .vla import Policy
from .perception_study import region_weights


def target_signal(regions,scores,size,mode,generator=None):
    if mode not in ('binding','zero','shuffled'):raise ValueError('Unknown binding mode')
    if len(regions)!=len(scores) or not torch.isfinite(scores).all():raise ValueError('Invalid scores')
    grid=torch.zeros(8,16)
    for obj,score in zip(regions,scores):
        weights=region_weights(obj['bbox_raw'],size,(8,16))
        grid=torch.maximum(grid,weights/weights.max()*score.detach().cpu())
    if mode=='shuffled':
        if generator is None:raise ValueError('Explicit shuffle RNG required')
        grid=grid.flatten()[torch.randperm(128,generator=generator)].reshape(8,16)
    signal=torch.cat([grid.flatten(),torch.tensor([float(scores.max()) if len(scores) else 0.,float(not len(regions))])])
    return torch.zeros_like(signal) if mode=='zero' else signal


class BindingAgent(Policy):
    def __init__(self,width,hidden=64):
        super().__init__(width,hidden)
        self.position=torch.nn.Parameter(torch.randn(128,hidden)*.01)
        self.binding=torch.nn.Sequential(torch.nn.Linear(130,hidden),torch.nn.Tanh())
        self.memory=torch.nn.GRUCell(3*hidden+4,hidden)
        self.value=torch.nn.Linear(hidden,1)
        self.predict_next=torch.nn.Linear(hidden+4,width)
    def forward(self,features,instruction,previous,state=None,binding=None):
        if binding is None or binding.shape!=(130,):raise ValueError('Explicit target signal required')
        visual=self.visual(features)[None]+self.position[None]
        language=self.words(instruction).mean(0)[None,None]
        attended,weights=self.attention(language,visual,visual)
        prior=torch.nn.functional.one_hot(torch.tensor([previous],device=features.device),4).float()
        context=self.binding(binding.to(features.device))[None]
        state=self.memory(torch.cat([attended[:,0],language[:,0],context,prior],-1),state)
        return self.action(state)[0],state,weights[0,0]
