"""Synthetic non-Minecraft fixtures; real model and projection remain external gates."""
import numpy as np
import torch
from mc_binding.target_tracking import choose,candidates


def test_template_changes_selection_without_ids():
 x=torch.tensor([[1.,0.],[0.,1.]])
 scores=torch.tensor([.55,.6]);template=x[0]
 assert choose(x,scores,.1)==1
 assert choose(x,scores,.1,template)==0
 assert choose(x.flip(0),scores.flip(0),.1,template)==1
 assert choose(x,scores,.9,template)==-1
 assert choose(torch.empty(0,2),torch.empty(0),.1)==-1


def test_geometry_candidates_filter_hud_and_sort_screen_order():
 mask=np.full((280,448),-1)
 mask[80:100,50:70]=7;mask[80:100,150:170]=2
 mask[220:240,100:120]=3
 a,e=candidates(mask)
 assert [o['index'] for o in a]==[7,2]
 assert e[0]['index']==3
 assert 'possible_HUD_or_hand' in e[0]['reasons']


def test_evaluation_does_not_call_out_of_view_absent(tmp_path,monkeypatch):
 import json
 from PIL import Image
 import mc_binding.target_tracking as t
 from mc_binding.io import file_hash
 image=tmp_path/'synthetic-non-minecraft.png';Image.new('RGB',(448,280)).save(image)
 ck=tmp_path/'readout.pt';ck.write_bytes(b'synthetic')
 audit=tmp_path/'audit';audit.mkdir()
 (audit/'manifest.json').write_text(json.dumps(dict(checkpoint_sha256=file_hash(ck))))
 base=dict(episode='synthetic',family='fixture',frame=str(image),sha256=file_hash(image),goal=['red','pillar'],excluded=[],target_index=0,world_present=True)
 frames=[dict(base,step=0,tick=0,regions=[dict(index=0,bbox_raw=[20,50,40,80])]),dict(base,step=1,tick=2,regions=[])]
 (audit/'frames.json').write_text(json.dumps(frames))
 monkeypatch.setattr(t.torch,'load',lambda *a,**k:dict(width=2,state={},threshold=.1))
 class Model:
  def __init__(self,*a):pass
  def load_state_dict(self,*a):pass
  def eval(self):pass
  def goal_scores(self,x,goal):return torch.ones(len(x))*.9
 class Encoder:
  def __init__(self,*a):self.visual=type('V',(),dict(spatial_merge_size=1))()
  def inputs(self,image):return None,torch.tensor([[1,2,2]])
  def features(self,x):return torch.ones(4,2)
 monkeypatch.setattr(t,'Readout',Model);monkeypatch.setattr(t,'Encoder',Encoder)
 monkeypatch.setattr(t,'rasterize',lambda *a:torch.ones(2,2,2))
 monkeypatch.setattr(t,'features_for',lambda *a:torch.ones(1,2))
 out=tmp_path/'results';t.evaluate(audit,ck,{},out)
 summary=json.loads((out/'summary.json').read_text())
 assert all(r['correct_present']==1 and r['excluded_present']==1 and r['absent_frames']==0 for r in summary)
 assert (out/'synthetic.gif').exists()
