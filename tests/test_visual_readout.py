"""Non-Minecraft synthetic tests of vision-readout math and provenance."""
import json
import numpy as np
import pytest
import torch
from PIL import Image
from mc_binding.binding_data import binding_contexts
from mc_binding.visual_readout import (split_groups, geometry_key,Readout,pool,gradcam,heatmap,
                                      expected,prediction,roi_tokens,extract,train,evaluate,Encoder)


def test_geometry_splits_group_counterfactual_duplicates():
    groups={f'f{i:04d}':binding_contexts(i,731) for i in range(24)}
    splits=split_groups(groups,731)
    assert len(set(geometry_key(v) for v in groups.values()))==6
    assert set(splits.values())=={'train','validation','test'}
    for a in groups:
        for b in groups:
            if geometry_key(groups[a])==geometry_key(groups[b]): assert splits[a]==splits[b]
    with pytest.raises(ValueError,match='three distinct'): split_groups({'f0000':groups['f0000']},731)


def test_bbox_masks_do_not_use_color_labels_and_cam_math(tmp_path):
    r={'objects':[{'bbox':[0,0,112,155],'color':'red'},{'bbox':[336,0,448,155],'color':'blue'}]}
    pos=roi_tokens(r,4,8,2)
    r['objects'][0]['color']='blue'
    assert roi_tokens(r,4,8,2)==pos
    torch.manual_seed(1);model=Readout(4);x=torch.randn(32,4)
    cam,value=gradcam(x,pos,model,('red','arch'),0)
    a=x.clone().requires_grad_(True);score=model.goal_scores(pool(a,pos),('red','arch'))[0]
    gradient=torch.autograd.grad(score,a)[0]
    assert torch.equal(cam,torch.relu((x*gradient.mean(0)).sum(-1)))
    assert not gradient[pos[1]].any() # privileged pooling limits source gradients
    heatmap(Image.new('RGB',(448,155)),cam,4,8,2,tmp_path/'cam.png')
    assert Image.open(tmp_path/'cam.png').size==(448,155)
    assert prediction(torch.tensor([.2,.3]),.5)==-1
    with pytest.raises(RuntimeError,match='forbidden'): Encoder._deny_decoder(None,None)


def test_synthetic_extract_train_evaluate_end_to_end(tmp_path,monkeypatch):
    import mc_binding.visual_readout as vr
    torch.set_num_threads(1)
    groups={};records=[]
    for i in range(24):
        rs=binding_contexts(i,731)
        for r in rs:
            rid=f'{len(records):06d}';r.update(record_id=rid,image=f'{rid}.png')
            for o in r['objects']:
                slot=int(o['object_id'][-1]);o['bbox']=[slot*112+20,50,slot*112+70,100]
            r['objects'].sort(key=lambda o:o['bbox'][0])
            im=Image.new('RGB',(448,155));im.putpixel((0,0),(len(records)//256,len(records)%256,0));im.save(tmp_path/r['image'])
            records.append(r)
        groups[f'f{i:04d}']=rs
    data={'records':records,'is_minecraft':False,'test_fixture':True}
    monkeypatch.setattr(vr,'load_binding',lambda p:(data,groups))
    class QKV(torch.nn.Module):
        def forward(self,x): return torch.cat([x,x,x],dim=-1)
    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__();self.attn=torch.nn.Module();self.attn.qkv=QKV();self.attn.num_heads=2
        def forward(self,x): return self.attn.qkv(x)[:,8:]
    class Fake:
        def __init__(self,config):
            self.visual=torch.nn.Module();self.visual.spatial_merge_size=2
            self.visual.blocks=torch.nn.ModuleList([Block(),Block()]);self.config=config
        def inputs(self,image):
            rgb=image.getpixel((0,0));return rgb[0]*256+rgb[1],torch.tensor([[1,16,64]])
        def run(self,inputs):
            r=records[inputs[0]];x=torch.zeros(1024,4)
            for o,indices in zip(r['objects'],roi_tokens(r,16,64,2)):
                x[indices,vr.COLORS.index(o['color'])]=2
                x[indices,2+vr.SHAPES.index(o['type'])]=2
            for block in self.visual.blocks: x=block(x)
            return x
        def features(self,inputs): return self.run(inputs)
    monkeypatch.setattr(vr,'Encoder',Fake)
    config=dict(seed=731,readout_layer=1,mask_threshold=.5,epochs=60,learning_rate=.05,weight_decay=.01,map_records=1,
                vision_layers=[0,1],vision_layer_groups=[[0,1]],head_sets=[None,[0]],alphas=[0,1],conditions=['donor'])
    cache=tmp_path/'cache';fit=tmp_path/'fit';out=tmp_path/'eval'
    extract(tmp_path,cache,config,True)
    train(cache,fit,config)
    assert json.loads((fit/'manifest.json').read_text())['independent_geometry_groups']==6
    summary=json.loads((fit/'summary.json').read_text())
    assert all(r['correct']==r['n'] for r in summary['test_binding'])
    evaluate(tmp_path,cache,fit,out,config,True)
    assert json.loads((out/'status.json').read_text())['state']=='complete'
    assert len(list(out.glob('*-cam.png')))==4
    assert (out/'report.html').exists()
    patched=tmp_path/'patch'
    evaluate(tmp_path,cache,fit,patched,config,True,True)
    ps=json.loads((patched/'patch-results.json').read_text())
    assert ps and any(r['choice']==r['expected_donor'] for r in ps if r['condition']=='donor' and r['alpha']==1)
    assert all(r['choice']==r['expected_original'] for r in ps if r['condition']=='self' or r['alpha']==0)
    assert len(list(patched.glob('*-cam.png')))==8
    # Corruption must be detected before evaluation.
    (fit/'readout.pt').write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='hash'): evaluate(tmp_path,cache,fit,tmp_path/'bad',config,True)
