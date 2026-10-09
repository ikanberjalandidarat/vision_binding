"""Non-Minecraft synthetic fixtures; live renderer/model remain required gates."""
from PIL import Image,ImageDraw
from mc_binding.live_visual_control import proposals,Servo


def test_proposals_use_rgb_not_goal_or_world():
 im=Image.new('RGB',(448,280),'gray');d=ImageDraw.Draw(im)
 d.rectangle((50,60,70,120),fill='red');d.rectangle((220,60,250,120),fill='blue')
 d.rectangle((300,200,400,270),fill='yellow')
 boxes=[r['bbox_raw'] for r in proposals(im)]
 assert boxes==[[50,60,71,121],[220,60,251,121]]


def test_servo_reobserves_and_stops_without_forward_confirmation():
 s=Servo(170)
 assert s.action([300,60,340,120])==(2,'align')
 assert s.action([100,60,140,120])==(1,'align')
 assert s.action([214,60,234,120])==(0,'approach')
 assert s.action([214,20,234,220])==(4,'verify_arrival')
 assert s.action([214,20,234,220])==(3,'visual_arrival')
 assert s.action(None)==(2,'search')
 for _ in range(35):assert s.action(None)==(2,'search')
 assert s.action(None)==(3,'search_exhausted')


def test_loss_resets_arrival_confirmation():
 s=Servo(170);box=[214,20,234,220]
 assert s.action(box)[0]==4
 s.action(None)
 assert s.action(box)[0]==4


def test_live_runner_observes_after_actions(tmp_path,monkeypatch):
 import json
 import torch
 from types import SimpleNamespace
 import mc_binding.live_visual_control as m
 dataset=tmp_path/'data';dataset.mkdir();(dataset/'manifest.json').write_text('{"seed":731}')
 checkpoint=tmp_path/'ck.pt';checkpoint.write_bytes(b'non-Minecraft fixture')
 config=tmp_path/'config.json';config.write_text('{}')
 calls=[]
 class Worker:
  def __init__(self,python,root,chunk):self.root=root;root.mkdir();self.i=0
  def request(self,cmd,**kw):
   calls.append(cmd)
   if cmd=='reset':self.ep=self.root/kw['episode'];self.ep.mkdir();self.i=0
   elif cmd=='step':self.i+=1
   elif cmd=='wait':pass
   elif cmd=='finish':
    p=self.ep/'episode.json';p.write_text(json.dumps(dict(objects=[],trajectory=[dict(pose=dict(x=0,z=0))])))
    return dict(success=True,timeout=False,goal_present=True,episode_file=str(p),minimum_sampled_goal_distance=.2,distance=.2)
   else:raise AssertionError('Unexpected privileged request')
   # First observation asks a turn; subsequent observations trigger visual stop.
   im=Image.new('RGB',(448,280),'gray');d=ImageDraw.Draw(im)
   d.rectangle((300,20,330,210) if self.i==0 else (214,20,234,210),fill='red')
   p=self.ep/f'{len(calls)}.png';im.save(p);return dict(frame=str(p))
  def close(self):pass
 class Encoder:
  def __init__(self,*a):self.visual=SimpleNamespace(spatial_merge_size=1)
  def inputs(self,im):return None,torch.tensor([[1,4,4]])
  def features(self,inp):return torch.ones(16,2)
 class Readout:
  def __init__(self,*a):pass
  def load_state_dict(self,*a):pass
  def eval(self):pass
  def goal_scores(self,x,goal):return torch.ones(len(x))
 job=dict(episode='synthetic',family='non-Minecraft',goal=['red','pillar'],instruction='red pillar',record=dict(image='unused'))
 monkeypatch.setattr(m,'task_panel',lambda *a:[job]);monkeypatch.setattr(m,'Worker',Worker)
 monkeypatch.setattr(m,'Encoder',Encoder);monkeypatch.setattr(m,'Readout',Readout)
 monkeypatch.setattr(torch,'load',lambda *a,**kw:dict(width=2,state={},threshold=.05))
 out=tmp_path/'out';m.run(SimpleNamespace(output=str(out),demos='',checkpoint=str(checkpoint),dataset=str(dataset),smoke=False,policy_checkpoint=None,max_steps=5,stop_height=170,config=str(config),render_python='unused'))
 assert calls==['reset','step','wait','finish']*2
 assert json.loads((out/'status.json').read_text())['state']=='complete'
 assert (out/'memory/episodes/synthetic/selection-control.gif').exists()
