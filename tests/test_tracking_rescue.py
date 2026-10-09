"""Synthetic non-Minecraft data; GPU model behavior remains an Oscar gate."""
import torch
import numpy as np
from PIL import Image
from mc_binding.tracking_rescue import RecoveryMemory,balanced_panel,disrupt,patch_positions,outcome,GOALS


def test_recovery_bridges_threshold_and_rejects_ambiguity():
 m=RecoveryMemory(.05)
 x=torch.eye(2)
 assert m.select(x,torch.tensor([.8,.1]))==(0,'acquired')
 assert m.select(x.flip(0),torch.tensor([.01,.02]))==(1,'matched')
 ambiguous=torch.tensor([[1.,0.],[1.,0.]])
 assert m.select(ambiguous,torch.ones(2))[0]==-1
 assert m.select(ambiguous,torch.ones(2))==(-1,'lost_reset')
 assert m.template is None
 assert m.select(x,torch.zeros(2))==(-1,'unacquired')


def test_balanced_panel_covers_all_goals():
 jobs=[]
 for family in ['a','b']:
  for i,goal in enumerate(GOALS):
   for present in [True,False]:
    jobs.append(dict(family=family,goal=goal,episode=f'{family}{i}{present}',record=dict(kind='full',objects=[dict(color=goal[0],type=goal[1])] if present else [])))
 panel=balanced_panel(jobs)
 assert len(panel)==12
 for goal in GOALS:
  assert sum(tuple(j['goal'])==goal and bool(j['record']['objects']) for j in panel)==2
  assert sum(tuple(j['goal'])==goal and not j['record']['objects'] for j in panel)==1


def test_disruption_and_patch_alignment():
 im=Image.new('RGB',(448,280),'red')
 clean,active,_=disrupt(im,0,.7);assert not active and np.array_equal(clean,im)
 changed,active,box=disrupt(im,2,.7);assert active
 assert np.array_equal(np.asarray(changed)[:80],np.asarray(im)[:80])
 assert not np.array_equal(changed,im)
 p,c=patch_positions(20,32,2,im.size,box)
 assert len(p)==len(c) and not set(p)&set(c)
 assert (p,c)==patch_positions(20,32,2,im.size,box)
 row=dict(regions=[dict(index=1)],world_present=True,target_index=0)
 assert outcome(row,0)['correct'] is None  # unavailable target is not world absence


def test_runner_self_patch_and_artifacts(tmp_path,monkeypatch):
 import json
 import mc_binding.tracking_rescue as t
 from mc_binding.io import file_hash
 from mc_binding.vision_hooks import capture_blocks
 from types import SimpleNamespace
 im=tmp_path/'non-minecraft.png';Image.new('RGB',(448,280),'red').save(im)
 ck=tmp_path/'readout.pt';ck.write_bytes(b'fixture')
 audit=tmp_path/'audit';audit.mkdir()
 (audit/'manifest.json').write_text(json.dumps(dict(balanced=True,checkpoint_sha256=file_hash(ck))))
 rows=[dict(episode='synthetic',family='fixture',step=i,tick=i*2,frame=str(im),sha256=file_hash(im),goal=['red','pillar'],world_present=True,target_index=0,regions=[dict(index=0,bbox_raw=[20,90,60,140])]) for i in range(4)]
 (audit/'frames.json').write_text(json.dumps(rows))
 class E:
  def __init__(self,config):
   self.visual=SimpleNamespace(spatial_merge_size=1,blocks=[SimpleNamespace(attn=SimpleNamespace(qkv=torch.nn.Identity())) for _ in range(32)])
  def inputs(self,image):return torch.ones(16,6)*float(np.asarray(image).mean()/255),torch.tensor([[1,4,4]])
  def run(self,inp):
   x=inp[0]
   for b in self.visual.blocks:x=b.attn.qkv(x)
   return x
  def features(self,inp):return self.run(inp)[:,4:]
 class M:
  def __init__(self,*args):pass
  def load_state_dict(self,*args):pass
  def eval(self):pass
  def goal_scores(self,x,goal):return x.mean(1)
 monkeypatch.setattr(t,'Encoder',E);monkeypatch.setattr(t,'Readout',M)
 monkeypatch.setattr(torch,'load',lambda *a,**kw:dict(width=2,state={},threshold=.05))
 out=tmp_path/'out';t.run(audit,ck,dict(readout_layer=31),out)
 assert json.loads((out/'status.json').read_text())['state']=='complete'
 patches=json.loads((out/'patch-results.json').read_text())
 assert len(patches)==36
 assert {p['control'] for p in patches}=={'self','clean_donor','other_region'}
 assert all(p['correct']==p['corrupt_correct'] for p in patches if p['control']=='self')
 assert (out/'synthetic.gif').exists()
