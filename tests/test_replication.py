"""Synthetic unit checks, not Minecraft or Qwen validation."""
from collections import Counter
import math
import pytest
from mc_binding.vision_data import swap_contexts
from mc_binding.approach import steering, waypoint


def test_replication_factorial_and_donor_geometry():
    factors=[];palettes=Counter();shapes=Counter()
    for index in range(24):
        rows=swap_contexts(index,731,'replication_v1')
        factors.append(tuple(rows[0]['factors'].values()))
        objects=rows[0]['objects']
        palettes[tuple(sorted(o['color'] for o in objects))]+=1
        shapes[tuple(o['type'] for o in objects)]+=1
        for a,b in zip(rows[:3],rows[3:]):
            for x,y in zip(a['objects'],b['objects']):
                assert x['blocks']==y['blocks'] and x['object_id']==y['object_id']
                assert x['color']!=y['color']
            for row in (a,b):
                expected={tuple(p):o['color'] for o in row['objects'] for p in o['blocks']}
                actual={tuple(map(int,s.split()[1:4])):s.split()[-1].removeprefix('minecraft:').removesuffix('_wool') for s in row['commands'] if s.startswith('/setblock ')}
                assert actual==expected
                assert all(-20<x<20 and 200<=y<205 and -5<z<25 for x,y,z in actual)
    assert len(set(factors))==24
    assert sorted(palettes.values())==[8,8,8]
    assert sorted(shapes.values())==[12,12]
    with pytest.raises(ValueError):swap_contexts(24,731,'replication_v1')


def test_controller_heading_and_stop():
    pose=dict(x=.5,z=.5,yaw=0)
    d,turn,forward=steering(pose,[.5,10])
    assert d==9.5 and turn==0 and forward==1
    assert steering(pose,[10,.5])[1:]==(-10.,0)
    assert steering(pose,[-10,.5])[1:]==(10.,0)
    assert steering(pose,[.5,.5])[2]==0
    assert steering(dict(x=0,z=0,yaw=179),[0,-10])[1]==1
    with pytest.raises(ValueError):steering(dict(x=math.nan,z=0,yaw=0),[0,0])
    assert waypoint({'bounds':[[-6,200,12],[-2,204,14]]})==[-4,10]


def test_extended_color_scores_and_strict_side_parser():
    torch=pytest.importorskip('torch')
    from mc_binding.vision_scores import score_colors
    from mc_binding.destination import parse_side
    class Tokenizer:
        def encode(self,text,add_special_tokens=False):
            return [{'red':0,'blue':1,'yellow':2}[text.strip().lower()]]
    result=score_colors(torch.zeros(4),Tokenizer(),['red','blue','yellow'])
    assert result['probability']['yellow']==pytest.approx(.25)
    assert result['token_ids']['yellow']==[2]
    assert parse_side('RIGHT.')==1 and parse_side('left')==0
    assert parse_side('left or right') is None


def test_destination_runner_controls_resume_and_gate(tmp_path,monkeypatch):
    torch=pytest.importorskip('torch')
    import importlib.metadata
    import json
    import numpy as np
    from PIL import Image
    from types import SimpleNamespace
    from mc_binding.capture_pairs import clean_frame
    from mc_binding.io import atomic_json,file_hash
    from mc_binding.vision_data import token_cells
    from mc_binding.destination import destination
    from mc_binding.models import qwen
    from test_recognition import frame_for
    root=tmp_path/'synthetic';(root/'frames').mkdir(parents=True);(root/'raw').mkdir()
    records=[]
    for i,ctx in enumerate(swap_contexts(0,731)):
        raw=frame_for(ctx['objects']);image,objects=clean_frame(raw,ctx['objects']);rid=f'{i:06d}'
        image.save(root/'frames'/f'{rid}.png');Image.fromarray(raw).save(root/'raw'/f'{rid}.png')
        records.append({**ctx,'objects':objects,'record_id':rid,'scene_family_id':'f0000',
                        'image':f'frames/{rid}.png','raw_image':f'raw/{rid}.png',
                        'image_sha256':file_hash(root/'frames'/f'{rid}.png'),'raw_sha256':file_hash(root/'raw'/f'{rid}.png'),
                        'actual_pose':dict(x=.5,y=200,z=.5,yaw=0,pitch=0)})
    # Loader contract is exercised with generated test pixels, never published as experimental data.
    atomic_json(root/'manifest.json',dict(schema_version='vision_swap_v1',state='captured_needs_visual_review',is_minecraft=True,records=records))
    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__();self.attn=torch.nn.Module();self.attn.qkv=torch.nn.Identity()
        def forward(self,x):return self.attn.qkv(x)
    class Network(torch.nn.Module):
        def __init__(self):
            super().__init__();self.visual=torch.nn.Module();self.visual.blocks=torch.nn.ModuleList([Block(),Block()]);self.visual.merger=torch.nn.Identity();self.visual.spatial_merge_size=2
        def forward(self,pixels,**kw):
            for block in self.visual.blocks:pixels=block(pixels)
            return pixels
    class Fake:
        calls=0
        def __init__(self,config):self.model=Network();self.processor=SimpleNamespace(image_processor=SimpleNamespace(merge_size=2))
        def manifest(self):return {'test_double':True}
        def inputs(self,image,prompt):
            a=np.asarray(image.resize((32,16))).copy()
            pixels=torch.tensor(np.array([a[r,c] for r,c in token_cells(16,32,2)]),dtype=torch.float32).repeat(1,3)
            goal='red' if 'red structure' in prompt else 'blue'
            colors=['red' if image.getpixel((x,70))[0]>image.getpixel((x,70))[2] else 'blue' for x in (110,310)]
            return dict(pixels=pixels,answer=('LEFT','RIGHT')[colors.index(goal)],image_grid_thw=torch.tensor([[1,16,32]]))
        def answer(self,inp):Fake.calls+=1;self.model(**inp);return inp['answer']
    monkeypatch.setattr(qwen,'Qwen',Fake)
    real=importlib.metadata.version
    monkeypatch.setattr(importlib.metadata,'version',lambda name:'4.55.0' if name=='transformers' else real(name))
    config=dict(dtype='bfloat16',load_in_4bit=False,use_fast=True,check_finite_scores=True,vision_patch_kind='v',vision_layers=[0,1],vision_layer_groups=[[0,1]],mask_threshold=.5,seed=731)
    out=tmp_path/'choices';destination(root,out,config,True)
    rows=[json.loads(s) for s in (out/'results.jsonl').read_text().splitlines()]
    assert len(rows)==28 and Fake.calls==28
    assert json.loads((out/'status.json').read_text())['state']=='complete'
    target=next(r for r in rows if r['condition']=='both_objects')
    bg=next(r for r in rows if r['condition']=='background')
    assert len(target['positions'])==len(bg['positions'])
    assert not set(target['positions'])&set(bg['positions'])
    assert 'data:image/png;base64,' in (out/'report.html').read_text()
    destination(root,out,config,True);assert Fake.calls==28
    monkeypatch.setattr(Fake,'answer',lambda self,inp:'invalid')
    failed=tmp_path/'failed'
    with pytest.raises(RuntimeError,match='clean destination gate failed'):destination(root,failed,config,True)
    assert json.loads((failed/'status.json').read_text())['state']=='error'
    assert not list((failed/'families').glob('*.json'))
