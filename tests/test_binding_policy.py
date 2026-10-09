"""Synthetic non-Minecraft fixtures; no claims about model or rendering success."""
import torch
from mc_binding.binding_policy import target_signal,BindingAgent
from mc_binding.rl_navigation import initialize_agent,objective


def test_map_control_retains_scores_but_changes_location():
 regions=[dict(bbox_raw=[0,0,28,35])]
 signal=target_signal(regions,torch.tensor([.8]),(448,280),'binding')
 flipped=target_signal(regions,torch.tensor([.8]),(448,280),'shuffled',torch.Generator().manual_seed(12))
 assert signal.shape==(130,)
 assert signal[:128].argmax()==0 and not torch.equal(signal[:128],flipped[:128])
 assert torch.equal(signal[:128].sort().values,flipped[:128].sort().values)
 assert torch.equal(signal[128:],flipped[128:])
 assert not target_signal(regions,torch.tensor([.8]),(448,280),'zero').any()
 assert target_signal([],torch.empty(0),(448,280),'binding')[-1]==1


def test_rl_gradient_reaches_selection_branch_and_action():
 torch.set_num_threads(1)
 model=initialize_agent(6,731,binding=True)
 signal=target_signal([dict(bbox_raw=[0,0,30,40])],torch.tensor([.8]),(448,280),'binding')
 logits,state,_=model(torch.randn(128,6),torch.tensor([2,3]),3,binding=signal)
 dist=torch.distributions.Categorical(logits=logits)
 loss=objective([dist.log_prob(torch.tensor(0))],[model.value(state).squeeze()],[dist.entropy()],[1.],.99,[],0)
 loss.backward()
 assert model.binding[0].weight.grad.abs().sum()>0
 assert model.action.weight.grad.abs().sum()>0
 same=initialize_agent(6,731,binding=True)
 assert torch.equal(model.action.weight,same.action.weight)


def test_binding_rl_trains_and_keeps_initial_evaluation_separate(tmp_path,monkeypatch):
 import json
 from PIL import Image
 import mc_binding.rl_navigation as rl
 import mc_binding.live_visual_control as live
 from mc_binding.visual_readout import Readout
 image=tmp_path/'synthetic.png';Image.new('RGB',(448,280),'red').save(image)
 checkpoint=tmp_path/'readout.pt';frozen=Readout(6)
 torch.save(dict(width=6,state=frozen.state_dict(),threshold=.05),checkpoint)
 (tmp_path/'manifest.json').write_text(json.dumps(dict(splits={'train':'train','validation':'validation'})))
 record=dict(image=image.name,objects=[dict(color='red',type='pillar')])
 jobs=[dict(geometry=g,family=g,record=record,goal=['red','pillar'],instruction='Go to the red pillar.') for g in ('train','validation')]
 monkeypatch.setattr(rl,'load_binding',lambda _:({'seed':731},{}));monkeypatch.setattr(rl,'plan',lambda _:jobs)
 monkeypatch.setattr(rl,'geometry_splits',lambda *_:{'train':'train','validation':'validation'})
 monkeypatch.setattr(live,'proposals',lambda im:[dict(bbox_raw=[0,0,28,35])])
 class Encoder:
  def __init__(self,_):self.visual=type('Visual',(),{'spatial_merge_size':1})()
  def inputs(self,_):return None,torch.tensor([[1,2,2]])
  def features(self,_):return torch.ones(4,6)
 resets=[]
 class Worker:
  def __init__(self,*a):pass
  def request(self,cmd,**kw):
   if cmd=='reset':resets.append(kw);return dict(frame=str(image))
   assert cmd=='rl_step';return dict(reward=1.,terminal=True,success=True)
  def close(self):pass
 monkeypatch.setattr(rl,'Encoder',Encoder);monkeypatch.setattr(rl,'Worker',Worker)
 out=tmp_path/'run'
 rl.run(tmp_path,out,'unused',dict(readout_layer=31),731,'potential',0,2,2,1,paired_evaluation=True,binding_checkpoint=str(checkpoint),binding_mode='binding')
 initial=json.loads((out/'initial-evaluation.json').read_text());final=json.loads((out/'validation.json').read_text())
 assert len(initial)==len(final)==2
 assert all(r['phase'].startswith('initial-') for r in initial)
 assert all(r['phase'].startswith('validation') for r in final)
 assert len(final[0]['decisions'][0]['binding_signal'])==130
 ck=torch.load(out/'agent.pt',weights_only=True)
 assert ck['architecture']=='BindingAgent'
 original=rl.initialize_agent(6,731,binding=True)
 assert not torch.equal(ck['state']['binding.0.weight'],original.binding[0].weight)
 assert all(r['training_start'] is None for r in resets)
