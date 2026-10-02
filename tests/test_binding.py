"""Synthetic/non-Minecraft unit tests; real rendering and GPU inference remain gates."""
import copy
from collections import Counter
import numpy as np
import pytest
import torch
from mc_binding.binding_data import binding_contexts, queries, FEATURES, parse_answer
from mc_binding.capture_pairs import clean_frame
from mc_binding.vision_hooks import patch_block
from mc_binding.binding_experiment import interventions, replacement_value


def test_binding_counterfactuals_balance_and_absence():
    slots=[Counter() for _ in range(4)]
    for i in range(24):
        rows=binding_contexts(i,731)
        assert len(rows)==12
        full={r['context']:r for r in rows if r['kind']=='full'}
        for j,o in enumerate(full['recipient']['objects']): slots[j][o['color'],o['type']]+=1
        for a,b in zip(full['recipient']['objects'],full['color_swap']['objects']):
            assert a['blocks']==b['blocks'] and a['object_id']==b['object_id']
            assert a['type']==b['type'] and a['color']!=b['color']
        for row in rows:
            xyz={tuple(map(int,c.split()[1:4])) for c in row['commands'] if c.startswith('/setblock ')}
            assert xyz=={tuple(p) for o in row['objects'] for p in o['blocks']}
            q=[q for q in queries(row) if q['task']=='binding']
            if row['kind']=='full': assert sorted(x['expected'] for x in q)==['1','2','3','4']
            if row['kind']=='absent':
                assert sum(x['expected']=='0' for x in q)==1
                assert len({o['color'] for o in row['objects']})==2
                assert len({o['type'] for o in row['objects']})==2
    assert all(dict(c)=={f:6 for f in FEATURES} for c in slots)
    assert parse_answer('2.','binding')=='2'
    assert parse_answer('1 or 2','binding') is None


def test_repeated_color_annotations_use_isolated_rois():
    raw=np.full((280,448,3),100,np.uint8)
    raw[90:140,40:70]=[180,10,10]
    raw[90:140,330:360]=[180,10,10]
    objs=[dict(color='red',annotation_roi_raw=[40,90,70,140]),dict(color='red',annotation_roi_raw=[330,90,360,140])]
    _,ann=clean_frame(raw,objs)
    assert [o['bbox'] for o in ann]==[[40,50,70,100],[330,50,360,100]]


def test_heads_strength_live_grouped_scope_and_cleanup():
    # Synthetic fused qkv: 3 projections, 4 heads, 2 channels/head.
    x=torch.arange(120,dtype=torch.float32).reshape(5,24)
    block=torch.nn.Identity()
    donor=torch.full((2,8),100.)
    with patch_block(block,[1,3],donor,5,'v',heads=[1,3],num_heads=4,alpha=.25): y=block(x)
    expected=x.clone()
    for row in (1,3):
        for col in (18,19,22,23): expected[row,col]=.75*x[row,col]+25
    assert torch.equal(y,expected)
    with patch_block(block,[1,3],donor,5,'v',heads=[1],num_heads=4,alpha=0): assert torch.equal(block(x),x)
    with pytest.raises(ValueError,match='Invalid projection heads'):
        with patch_block(block,[1],donor[:1],5,'v',heads=[4],num_heads=4): pass
    with pytest.raises(RuntimeError):
        with patch_block(block,[1],donor[:1],5,'v',heads=[0],num_heads=4):
            block(x);block(x)
    assert not block._forward_hooks


def test_selected_head_random_control_and_spec_validation():
    a=torch.zeros(5,8);b=torch.arange(40,dtype=torch.float32).reshape(5,8)
    v=replacement_value(a,b,[1,3],'random',731,[1],4)
    assert v[:,2:4].norm()==pytest.approx(float(b[[1,3]][:,2:4].norm()))
    assert not v[:,[0,1,4,5,6,7]].any()
    config=dict(vision_layers=[30,31],head_sets=[None,[0],[15]],alphas=[0,.5,1],conditions=['donor'])
    assert len(interventions(config,32,16))==24
    for bad in ({'alphas':[float('nan')]},{'head_sets':[[16]]},{'conditions':['bad']}):
        with pytest.raises(ValueError): interventions({**config,**bad},32,16)


def test_baseline_records_failures_patch_gate_and_resume(tmp_path,monkeypatch):
    import importlib.metadata
    import json
    from types import SimpleNamespace
    from PIL import Image
    from mc_binding import binding_experiment as be
    from mc_binding.models import qwen
    records=[]
    for i,row in enumerate(binding_contexts(0,731)):
        if row['kind']=='isolated': continue
        r=copy.deepcopy(row)
        r.update(record_id=f'{i:06d}',image=f'{i:06d}.png')
        Image.new('RGB',(448,155),'gray').save(tmp_path/r['image'])
        records.append(r)
    monkeypatch.setattr(be,'load_binding',lambda root:({'is_minecraft':False,'test_fixture':True},{'f0000':records}))
    real=importlib.metadata.version
    monkeypatch.setattr(importlib.metadata,'version',lambda n:'4.55.0' if n=='transformers' else real(n))
    class Fake:
        calls=0
        def __init__(self,config):
            self.model=torch.nn.Module()
            self.model.visual=SimpleNamespace(blocks=[SimpleNamespace(attn=SimpleNamespace(qkv=None)) for _ in range(32)],merger=True,config=SimpleNamespace(num_heads=16))
        def manifest(self): return {'is_test_double':True}
        def inputs(self,im,p): return p
        def answer(self,x): Fake.calls+=1;return 'invalid'
        def choice_scores(self,x): return {'log_probability':{str(i):-2. for i in range(5)}}
    monkeypatch.setattr(qwen,'Qwen',Fake)
    config=dict(dtype='bfloat16',load_in_4bit=False,use_fast=True,check_finite_scores=True,vision_layers=[30],seed=731)
    out=tmp_path/'run'
    be.binding_experiment(tmp_path,out,config,True,True)
    assert Fake.calls==44
    assert json.loads((out/'status.json').read_text())=={'state':'complete','rows':44}
    assert json.loads((out/'summary.json').read_text())['eligible_families']==[]
    assert not json.loads((out/'gates/f0000.json').read_text())['eligible']
    be.binding_experiment(tmp_path,out,config,True,True)
    assert Fake.calls==44
    assert 'data:image/png;base64,' in (out/'report.html').read_text()


def test_successful_runner_reverse_strength_and_head_patches(tmp_path,monkeypatch):
    """End-to-end runner plumbing with non-Minecraft RGB and a toy network."""
    import importlib.metadata
    import json
    from types import SimpleNamespace
    from PIL import Image
    from mc_binding import binding_experiment as be
    from mc_binding.models import qwen
    records=[]
    for i,row in enumerate(binding_contexts(0,731)):
        if row['kind']=='isolated': continue
        r=copy.deepcopy(row);r.update(record_id=f'{i:06d}',image=f'{i:06d}.png')
        rgb=np.full((155,448,3),100,np.uint8);rgb[0,0]=i
        for obj in r['objects']:
            slot=int(obj['object_id'][-1]);x=(20,100,280,360)[slot]
            obj['bbox']=[x,50,x+40,110]
            rgb[50:110,x:x+40]=[180,10,10] if obj['color']=='red' else [10,10,180]
        Image.fromarray(rgb).save(tmp_path/r['image']);records.append(r)
    monkeypatch.setattr(be,'load_binding',lambda root:({'is_minecraft':False,'test_fixture':True},{'f0000':records}))
    real=importlib.metadata.version
    monkeypatch.setattr(importlib.metadata,'version',lambda n:'4.55.0' if n=='transformers' else real(n))
    class Network(torch.nn.Module):
        def __init__(self):
            super().__init__();self.visual=torch.nn.Module();self.visual.blocks=torch.nn.ModuleList()
            for _ in range(2):
                block=torch.nn.Module();block.attn=torch.nn.Module();block.attn.qkv=torch.nn.Identity();block.attn.qkv.out_features=24
                self.visual.blocks.append(block)
            self.visual.merger=torch.nn.Identity();self.visual.config=SimpleNamespace(num_heads=4);self.visual.spatial_merge_size=2
        def forward(self,pixels,**kw):
            for b in self.visual.blocks: pixels=b.attn.qkv(pixels)
            return pixels
    class Fake:
        calls=0
        def __init__(self,config):
            self.model=Network();self.processor=SimpleNamespace(image_processor=SimpleNamespace(merge_size=2))
        def manifest(self): return {'test_double':True}
        def inputs(self,im,prompt):
            rid=f'{im.getpixel((0,0))[0]:06d}';r=next(r for r in records if r['record_id']==rid)
            q=next(q for q in queries(r) if q['prompt']==prompt)
            return dict(pixels=torch.full((16*64,24),float(int(rid))),image_grid_thw=torch.tensor([[1,16,64]]),answer=q['expected'])
        def answer(self,x):
            Fake.calls+=1;self.model(**x);return x['answer']
        def choice_scores(self,x):
            self.model(**x);return {'log_probability':{str(i):-2. for i in range(5)}}
    monkeypatch.setattr(qwen,'Qwen',Fake)
    config=dict(dtype='bfloat16',load_in_4bit=False,use_fast=True,check_finite_scores=True,
                vision_layers=[0],vision_layer_groups=[[0,1]],head_sets=[[1]],alphas=[0,.5,1],
                conditions=['donor','random','background','mean_ablation'],mask_threshold=.5,seed=731)
    out=tmp_path/'run'
    be.binding_experiment(tmp_path,out,config,True,True)
    rows=[json.loads(s) for s in (out/'results.jsonl').read_text().splitlines()]
    patched=[r for r in rows if r['condition']!='clean']
    assert len(patched)==2*13*2*4
    assert {r['context'] for r in patched}=={'recipient','color_swap'}
    assert all(r['expected']!=r['expected_donor'] for r in patched)
    assert json.loads((out/'status.json').read_text())['state']=='complete'
    assert json.loads((out/'summary.json').read_text())['eligible_families']==['f0000']
    assert (out/'alignment/f0000/recipient-objects.png').exists()
    previous=Fake.calls
    be.binding_experiment(tmp_path,out,config,True,True)
    assert Fake.calls==previous
