import copy
import numpy as np
import pytest
import torch
from PIL import Image
from mc_binding.vision_data import swap_contexts, token_cells, mask_coverage, regions
from mc_binding.vision_hooks import capture_blocks, patch_block, replacement


def test_swap_preserves_object_geometry_and_identity():
    for index in range(4):
        rows = swap_contexts(index, 731)
        for a,b in zip(rows[:3],rows[3:]):
            assert a['requested_camera']==b['requested_camera']
            assert a['kind']==b['kind']
            assert a['commands'][-1]==b['commands'][-1]
            for x,y in zip(a['objects'],b['objects']):
                assert {x['color'],y['color']}=={'red','blue'}
                assert {k:v for k,v in x.items() if k!='color'}=={k:v for k,v in y.items() if k!='color'}


def test_merge_grouped_order_and_mask_includes_no_arch_hole():
    assert token_cells(4,4,2)==[(0,0),(0,1),(1,0),(1,1),(0,2),(0,3),(1,2),(1,3),
                               (2,0),(2,1),(3,0),(3,1),(2,2),(2,3),(3,2),(3,3)]
    mask = np.zeros((40,40),bool)
    mask[10:20,20:30]=True
    values = mask_coverage(mask,4,4,2)
    assert np.flatnonzero(values).tolist()==[6]
    assert values[6]==1
    with pytest.raises(ValueError):
        token_cells(3,4,2)


def test_color_masks_exclude_ceiling_and_arch_hole():
    im = np.full((155,448,3),100,np.uint8)
    im[30:100,40:100]=[10,10,180]
    im[50:100,60:80]=100
    im[30:100,300:340]=[180,10,10]
    im[:20]=[10,10,180]  # same-color ceiling lies outside reviewed bbox
    rec={'objects':[{'color':'blue','bbox':[40,30,100,100]}, {'color':'red','bbox':[300,30,340,100]}]}
    sets,bg=regions(Image.fromarray(im),rec,  31,  56,1)
    cells=token_cells(31,56,1)
    assert all(cells[i][0]>=6 for i in sets[0])
    assert not(set(sets[0]) & set(bg))
    assert all(not(60 <= cells[i][1]*8 and (cells[i][1]+1)*8 <=80 and (cells[i][0]+.5)*5>=50) for i in sets[0])


def test_vision_hook_exact_scope_self_and_cleanup():
    block=torch.nn.Identity()
    x=torch.arange(24,dtype=torch.float32).reshape(6,4)
    values={}
    with capture_blocks([block],[0],values):
        assert torch.equal(block(x),x)
    assert torch.equal(values[0],x)
    with patch_block(block,[1,4],torch.zeros(2,4),6):
        changed=block(x)
    assert torch.equal(changed[[0,2,3,5]],x[[0,2,3,5]])
    assert not changed[[1,4]].any()
    assert torch.equal(block(x),x)
    with patch_block(block,[1,4],x[[1,4]],6):
        assert torch.equal(block(x),x)
    with pytest.raises(RuntimeError,match='did not fire'):
        with patch_block(block,[1],x[[1]],6):
            pass
    with pytest.raises(RuntimeError,match='exactly one'):
        with patch_block(block,[1],x[[1]],6):
            block(x)
            block(x)
    assert not block._forward_hooks


def test_random_delta_norm_and_finite_guard():
    a=torch.arange(24,dtype=torch.float32).reshape(6,4)
    b=a+3
    value,norm=replacement(a,b,[0,2],'random',731)
    assert norm==pytest.approx(float((b[[0,2]]-a[[0,2]]).norm()),rel=1e-6)
    assert not torch.equal(value,b[[0,2]])
    block=torch.nn.Identity()
    with pytest.raises(ValueError,match='Invalid replacement'):
        with patch_block(block,[1],torch.full((1,4),float('nan')),6):
            block(a)
    assert not block._forward_hooks


def test_runner_controls_checkpoint_resume_and_report(tmp_path, monkeypatch):
    import importlib.metadata
    import json
    from types import SimpleNamespace
    from mc_binding.capture_pairs import clean_frame
    from mc_binding.io import atomic_json, file_hash
    from mc_binding.vision_pilot import vision_pilot
    from mc_binding.models import qwen
    from test_recognition import frame_for
    root=tmp_path/'captures'
    (root/'frames').mkdir(parents=True)
    (root/'raw').mkdir()
    records=[]
    for index,ctx in enumerate(swap_contexts(0,731)):
        raw=frame_for(ctx['objects'])
        image,objects=clean_frame(raw,ctx['objects'])
        rid=f'{index:06d}'
        image.save(root/'frames'/f'{rid}.png')
        Image.fromarray(raw).save(root/'raw'/f'{rid}.png')
        records.append({**ctx,'objects':objects,'record_id':rid,'scene_family_id':'f0000',
            'image':f'frames/{rid}.png','raw_image':f'raw/{rid}.png',
            'image_sha256':file_hash(root/'frames'/f'{rid}.png'),
            'raw_sha256':file_hash(root/'raw'/f'{rid}.png'),
            'actual_pose':dict(x=.5,y=200,z=.5,yaw=0,pitch=0)})
    atomic_json(root/'manifest.json',{'schema_version':'vision_swap_v1','state':'captured_needs_visual_review','is_minecraft':True,'records':records})
    class Network(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.visual=torch.nn.Module()
            self.visual.blocks=torch.nn.ModuleList([torch.nn.Identity(), torch.nn.Identity()])
            self.visual.merger=torch.nn.Identity()
            self.visual.spatial_merge_size=2
        def forward(self, pixels, **kwargs):
            for block in self.visual.blocks:
                pixels = block(pixels)
            return pixels
    class Fake:
        calls=0
        def __init__(self,config):
            self.model=Network()
            self.processor=SimpleNamespace(image_processor=SimpleNamespace(merge_size=2))
        def manifest(self):
            return {'test_double':True}
        def inputs(self,image,prompt):
            a=np.asarray(image.resize((32,16))).copy()
            pixels=torch.tensor(np.array([a[r,c] for r,c in token_cells(16,32,2)]),dtype=torch.float32)
            colors=[]
            for x in (110,310):
                rgb=image.getpixel((x,70))
                if rgb[0]>rgb[2]: colors.append('red')
                elif rgb[2]>rgb[0]: colors.append('blue')
            answer=colors[0] if 'rightmost' not in prompt else colors[-1]
            return {'pixels':pixels,'answer':answer,'image_grid_thw':torch.tensor([[1,16,32]])}
        def answer(self,inp):
            Fake.calls+=1
            self.model(**inp)
            return inp['answer']
    monkeypatch.setattr(qwen,'Qwen',Fake)
    real_version=importlib.metadata.version
    monkeypatch.setattr(importlib.metadata,'version',lambda name:'4.55.0' if name=='transformers' else real_version(name))
    config=dict(dtype='bfloat16',load_in_4bit=False,use_fast=True,check_finite_scores=True,vision_layers=[0],mask_threshold=.5,seed=731)
    out=tmp_path/'run'
    vision_pilot(root,out,config,True)
    result=[json.loads(s) for s in (out/'results.jsonl').read_text().splitlines()]
    assert len(result)==18
    assert Fake.calls==28
    assert json.loads((out/'status.json').read_text())['state']=='complete'
    patches=[r for r in result if r['condition']!='clean']
    for side in (0,1):
        target=next(r for r in patches if r['condition']=='target' and r['target_side']==side)
        other=next(r for r in patches if r['condition']=='other_object' and r['target_side']==side)
        bg=next(r for r in patches if r['condition']=='background' and r['target_side']==side)
        assert not set(target['positions']) & set(other['positions'])
        assert not set(target['positions']) & set(bg['positions'])
        assert target['token_count']==bg['token_count']
    assert 'data:image/png;base64,' in (out/'report.html').read_text()
    vision_pilot(root,out,config,True)
    assert Fake.calls==28
    bad_config={**config,'seed':1}
    with pytest.raises(ValueError,match='Resume refused'):
        vision_pilot(root,out,bad_config,True)

    grouped=tmp_path/'grouped'
    group_config={**config, 'vision_layers': [], 'vision_layer_groups': [[0,1]]}
    vision_pilot(root,grouped,group_config,True)
    group_rows=json.loads((grouped/'families'/'f0000-group00-01.json').read_text())
    assert len(group_rows)==10
    assert all(r['layer_set']==[0,1] and r['layer'] is None and len(r['patches'])==2 for r in group_rows)
    assert '0 + 1' in (grouped/'report.html').read_text()
    before=Fake.calls
    vision_pilot(root,grouped,group_config,True)
    assert Fake.calls==before

    monkeypatch.setattr(Fake, 'answer', lambda self, inp: 'invalid')
    failed=tmp_path/'failed'
    with pytest.raises(RuntimeError,match='clean color gate failed'):
        vision_pilot(root,failed,config,True)
    assert json.loads((failed/'status.json').read_text())['state']=='error'
    assert not list((failed/'families').glob('*.json'))
    assert len(json.loads((failed/'diagnostics'/'f0000-baseline.json').read_text()))==8


def test_patch_set_validation():
    from mc_binding.vision_pilot import patch_sets
    sets=patch_sets({'vision_layers':[0,1], 'vision_layer_groups':[[0,1],[2,3]]},4)
    assert sets==[('layer00',[0]),('layer01',[1]),('group00-01',[0,1]),('group02-03',[2,3])]
    for config in ({'vision_layers':[0,0]}, {'vision_layers':[4]},
                   {'vision_layer_groups':[[1,0]]}, {'vision_layer_groups':[[]]},
                   {'vision_layers':[0], 'vision_layer_groups':[[0]]}):
        with pytest.raises(ValueError): patch_sets(config,4)


def test_simultaneous_hooks_use_each_layers_donor_and_cleanup():
    from contextlib import ExitStack
    blocks=torch.nn.Sequential(torch.nn.Identity(),torch.nn.Identity())
    x=torch.zeros(3,2)
    seen=[]
    observer=blocks[1].register_forward_pre_hook(lambda module,args:seen.append(args[0].clone()))
    with ExitStack() as stack:
        stack.enter_context(patch_block(blocks[0],[1],torch.full((1,2),10.),3))
        stack.enter_context(patch_block(blocks[1],[1],torch.full((1,2),20.),3))
        y=blocks(x)
    observer.remove()
    assert torch.equal(seen[0][1],torch.full((2,),10.))
    assert torch.equal(y[1],torch.full((2,),20.))
    assert torch.equal(y[[0,2]],x[[0,2]])
    assert all(not b._forward_hooks for b in blocks)
    with pytest.raises(ValueError):
        with ExitStack() as stack:
            stack.enter_context(patch_block(blocks[0],[1],torch.ones(1,2),3))
            stack.enter_context(patch_block(blocks[1],[1],torch.ones(1,3),3))
            blocks(x)
    assert all(not b._forward_hooks for b in blocks)
