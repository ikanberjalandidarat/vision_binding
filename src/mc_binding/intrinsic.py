"""ICM/RND on frozen pooled Qwen features; training-only exploration bonuses."""
import torch
from torch import nn
from torch.nn import functional as F


class ICM(nn.Module):
    def __init__(self,width):
        super().__init__()
        self.encoder=nn.Sequential(nn.LayerNorm(width),nn.Linear(width,64),nn.Tanh())
        self.inverse=nn.Sequential(nn.Linear(128,128),nn.ReLU(),nn.Linear(128,4))
        self.forward_model=nn.Sequential(nn.Linear(68,128),nn.ReLU(),nn.Linear(128,64))
    def losses(self,x,action,y):
        z=self.encoder(x);next_z=self.encoder(y)
        prediction=self.forward_model(torch.cat([z,F.one_hot(action,4).float()],-1))
        error=.5*(prediction-next_z.detach()).square().mean(-1)
        inverse=F.cross_entropy(self.inverse(torch.cat([z,next_z],-1)),action)
        return error,.2*error.mean()+.8*inverse


class RND(nn.Module):
    def __init__(self,width):
        super().__init__()
        self.target=nn.Sequential(nn.LayerNorm(width),nn.Linear(width,128),nn.ReLU(),nn.Linear(128,64))
        self.predictor=nn.Sequential(nn.LayerNorm(width),nn.Linear(width,128),nn.ReLU(),nn.Linear(128,128),nn.ReLU(),nn.Linear(128,64))
        for p in self.target.parameters():p.requires_grad_(False)
    def losses(self,x,action,y):
        error=(self.predictor(y)-self.target(y).detach()).square().mean(-1)
        return error,error.mean()


class Curiosity:
    def __init__(self,kind,width,seed):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed);self.model={'icm':ICM,'rnd':RND}[kind](width)
        self.optimizer=torch.optim.Adam([p for p in self.model.parameters() if p.requires_grad],lr=3e-4)
        self.count=0;self.square_sum=0.
    def bonus(self,x,action,y):
        # No state mutation: held-out evaluation never calls this method.
        with torch.no_grad():error,_=self.model.losses(x[None],torch.tensor([action]),y[None])
        raw=float(error[0]);scale=max((self.square_sum/max(self.count,1))**.5,1e-4) if self.count else 1.
        return raw,min(raw/scale,5.)
    def train_batch(self,transitions):
        if not transitions:return dict(n=0)
        x=torch.stack([t[0] for t in transitions]);a=torch.tensor([t[1] for t in transitions]);y=torch.stack([t[2] for t in transitions])
        losses=[]
        for start in range(0,len(x),128):
            _,loss=self.model.losses(x[start:start+128],a[start:start+128],y[start:start+128])
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite curiosity loss')
            self.optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(self.model.parameters(),1.);self.optimizer.step();losses.append(float(loss.detach()))
        self.square_sum+=sum(t[3]**2 for t in transitions);self.count+=len(transitions)
        return dict(n=len(transitions),mean_loss=sum(losses)/len(losses),normalizer_count=self.count)
    def state_dict(self):
        return dict(model=self.model.state_dict(),optimizer=self.optimizer.state_dict(),count=self.count,square_sum=self.square_sum)
