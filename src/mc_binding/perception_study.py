"""Privileged region diagnostics on real binding capture RGB; no decoder or movement."""
import argparse
import copy
import html
import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageDraw
from .binding_data import load_binding, FEATURES
from .visual_readout import Encoder, Readout, COLORS, SHAPES, geometry_key, save_tensor
from .vision_data import token_cells
from .vla import geometry_splits
from .io import atomic_json, file_hash, digest, source_hash

MODES = ('4x8', '8x16', 'native')


def rasterize(tokens, h, w, merge):
    if tokens.ndim != 2 or len(tokens) != h*w or not torch.isfinite(tokens).all():
        raise ValueError('Invalid native tokens')
    raster = torch.empty(h, w, tokens.shape[-1], dtype=torch.float32)
    for i, (row, col) in enumerate(token_cells(h, w, merge)):
        raster[row, col] = tokens[i].detach().float().cpu()
    return raster


def region_weights(box, size, grid):
    """Fractional box/cell intersection; no color-mask selection or empty rounding."""
    width, height = size; h, w = grid
    x0,y0,x1,y1 = box
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ValueError('Invalid region coordinates')
    xs=torch.arange(w+1,dtype=torch.float32)*width/w
    ys=torch.arange(h+1,dtype=torch.float32)*height/h
    dx=(torch.minimum(xs[1:],torch.tensor(float(x1)))-torch.maximum(xs[:-1],torch.tensor(float(x0)))).clamp_min(0)
    dy=(torch.minimum(ys[1:],torch.tensor(float(y1)))-torch.maximum(ys[:-1],torch.tensor(float(y0)))).clamp_min(0)
    weights=dy[:,None]*dx[None,:]
    return weights/weights.sum()


def features_for(raster, objects, size, mode):
    if mode not in MODES: raise ValueError('Unknown resolution')
    if mode != 'native':
        h,w=map(int,mode.split('x'))
        raster=torch.nn.functional.adaptive_avg_pool2d(raster.permute(2,0,1)[None],(h,w))[0].permute(1,2,0)
    return torch.stack([(raster*region_weights(o['bbox_raw'],size,raster.shape[:2])[...,None]).sum((0,1)) for o in objects])


def extract(dataset, output, config, reviewed):
    if not reviewed: raise ValueError('Review real captures before extraction')
    root,out=Path(dataset),Path(output); data,groups=load_binding(root)
    out.mkdir(parents=True,exist_ok=False);atomic_json(out/'status.json',{'state':'running'})
    try:
        enc=Encoder(config);entries=[];audit=[]
        (out/'alignment').mkdir()
        for fid,records in groups.items():
            for r in records:
                if r['kind'] not in ('full','absent'): continue
                with Image.open(root/r['raw_image']) as im: image=im.convert('RGB')
                if image.size!=(448,280):raise ValueError('Expected original 448x280 RGB')
                # Capture preprocessing removes 40 pixels from the top, then the HUD.
                for o in r['objects']:
                    b=o['bbox'];raw=o['bbox_raw']
                    if raw != [b[0],b[1]+40,b[2],b[3]+40]:raise ValueError('Raw/cropped bbox mismatch')
                inputs=enc.inputs(image);_,h,w=map(int,inputs[1][0].tolist())
                raster=rasterize(enc.features(inputs),h,w,int(enc.visual.spatial_merge_size))
                tensors={m:features_for(raster,r['objects'],image.size,m) for m in MODES}
                p=out/(r['record_id']+'.pt');save_tensor(p,tensors)
                entries.append(dict(record=r,family=fid,geometry=geometry_key(records),file=p.name,sha256=file_hash(p),grid=[h,w]))
                # All evaluated records are reviewable, including image-to-label order.
                panel=Image.new('RGB',(1344,310),'white');d=ImageDraw.Draw(panel)
                for j,m in enumerate(MODES):
                    panel.paste(image,(448*j,30));gh,gw=(h,w) if m=='native' else map(int,m.split('x'))
                    for x in range(gw+1):d.line((448*j+x*448/gw,30,448*j+x*448/gw,310),fill='#777777')
                    for y in range(gh+1):d.line((448*j,30+y*280/gh,448*(j+1),30+y*280/gh),fill='#777777')
                    for n,o in enumerate(r['objects']):
                        x0,y0,x1,y1=o['bbox_raw'];d.rectangle((448*j+x0,y0+30,448*j+x1,y1+30),outline='yellow',width=2)
                        d.text((448*j+x0,y0+16),str(n),fill='yellow')
                    d.text((448*j+4,5),f'{r["record_id"]} {m} | privileged boxes',fill='black')
                panel.save(out/'alignment'/(r['record_id']+'.png'))
                audit.append(dict(record_id=r['record_id'],labels=[f'{i}: {o["color"]} {o["type"]}' for i,o in enumerate(r['objects'])]))
                print('Extracted',r['record_id'],flush=True)
        atomic_json(out/'index.json',entries)
        atomic_json(out/'manifest.json',dict(config=config,dataset_hash=digest(data),source_hash=source_hash(),input='raw capture RGB, HUD retained; not live rollout frames',privileged_regions=True,decoder_executed=False))
        body=''.join(f'<h3>{a["record_id"]}: {html.escape(", ".join(a["labels"]))}</h3><img width="100%" src="alignment/{a["record_id"]}.png">' for a in audit)
        (out/'alignment.html').write_text('<!doctype html><meta charset="utf-8"><h1>Raw image / box / grid audit</h1><p>Boxes are capture annotations, not predicted detections. Grid lines show pooling resolution, not attention. Native token ordering is reconstructed before pooling.</p>'+body)
        atomic_json(out/'status.json',dict(state='complete',records=len(entries)))
    except BaseException as e:
        atomic_json(out/'status.json',dict(state='error',error=str(e)));raise


def fit(x,c,s,epochs=300,seed=731):
    torch.manual_seed(seed);model=Readout(x.shape[1]);model.mean.copy_(x.mean(0));model.scale.copy_(x.std(0).clamp_min(.01))
    opt=torch.optim.AdamW(model.parameters(),lr=.01,weight_decay=.01)
    for _ in range(epochs):
        opt.zero_grad();a,b=model(x);loss=torch.nn.functional.cross_entropy(a,c)+torch.nn.functional.cross_entropy(b,s)
        if not torch.isfinite(loss):raise ValueError('Nonfinite readout loss')
        loss.backward();opt.step()
    return model.eval()


def evaluate(model, entries, threshold):
    result=[];correct_c=correct_s=n=0
    with torch.no_grad():
        for e in entries:
            objects=e['record']['objects'];a,b=model(e['x']);c=a.argmax(-1);s=b.argmax(-1)
            correct_c+=sum(int(c[i])==COLORS.index(o['color']) for i,o in enumerate(objects))
            correct_s+=sum(int(s[i])==SHAPES.index(o['type']) for i,o in enumerate(objects));n+=len(objects)
            for goal in FEATURES:
                scores=a.softmax(-1)[:,COLORS.index(goal[0])]*b.softmax(-1)[:,SHAPES.index(goal[1])]
                matches=[i for i,o in enumerate(objects) if (o['color'],o['type'])==goal]
                if len(matches)>1:raise ValueError('Ambiguous target')
                truth=matches[0] if matches else -1;pred=int(scores.argmax()) if float(scores.max())>=threshold else -1
                result.append(dict(record=e['record']['record_id'],family=e['family'],goal=goal,expected=truth,predicted=pred,correct=truth==pred,scores=scores.tolist()))
    return dict(attributes=dict(n=n,color_correct=correct_c,shape_correct=correct_s),selection=[dict(present=p,n=sum((r['expected']!=-1)==p for r in result),correct=sum(r['correct'] and (r['expected']!=-1)==p for r in result)) for p in (True,False)],rows=result)


def train(cache, output, mode, seed=731):
    root,out=Path(cache),Path(output)
    if json.loads((root/'status.json').read_text())['state']!='complete':raise ValueError('Incomplete features')
    entries=json.loads((root/'index.json').read_text());splits=geometry_splits([e['geometry'] for e in entries],dict(seed=731,split_seed=731))
    for e in entries:
        p=(root/e['file']).resolve()
        if root.resolve() not in p.parents or file_hash(p)!=e['sha256']:raise ValueError('Feature integrity mismatch')
        e['x']=torch.load(p,map_location='cpu',weights_only=True)[mode].float()
    tr=[e for e in entries if splits[e['geometry']]=='train'];val=[e for e in entries if splits[e['geometry']]=='validation']
    def batch(es):
        objects=[o for e in es for o in e['record']['objects']]
        return torch.cat([e['x'] for e in es]),torch.tensor([COLORS.index(o['color']) for o in objects]),torch.tensor([SHAPES.index(o['type']) for o in objects])
    out.mkdir(parents=True,exist_ok=False);atomic_json(out/'status.json',dict(state='running'))
    try:
        torch.set_num_threads(4)
        tiny=[next(e for e in tr if e['record']['kind']=='full')];tx,tc,ts=batch(tiny)
        tiny_model=fit(tx,tc,ts,500,seed);tiny_result=evaluate(tiny_model,tiny,.5)
        atomic_json(out/'tiny-memorization.json',tiny_result)
        passed=tiny_result['attributes']['color_correct']==len(tc) and tiny_result['attributes']['shape_correct']==len(ts)
        if not passed:raise RuntimeError('Tiny-set attribute memorization failed; inspect before interpreting validation')
        x,c,s=batch(tr);model=fit(x,c,s,300,seed)
        # Calibrate rejection on TRAIN only. Validation reports are not used for selection.
        best=(-1,0)
        for t in np.linspace(0,1,21):
            result=evaluate(model,tr,float(t));counts=result['selection']
            if any(v['n']==0 for v in counts):raise ValueError('Need present/absent training controls')
            score=sum(v['correct']/v['n'] for v in counts)/2
            if score>best[0]:best=(score,float(t))
        save_tensor(out/'readout.pt',dict(state=model.state_dict(),threshold=best[1],width=x.shape[1]))
        atomic_json(out/'train.json',evaluate(model,tr,best[1]));atomic_json(out/'validation.json',evaluate(model,val,best[1]))
        atomic_json(out/'manifest.json',dict(mode=mode,seed=seed,split_seed=731,splits=splits,threshold=best[1],feature_index_hash=file_hash(root/'index.json'),source_hash=source_hash(),privileged_regions=True,test_evaluated=False,epochs=300,threshold_fit='training only',scope='Factorized attribute readouts with oracle regions, not native binding or autonomous detection'))
        atomic_json(out/'status.json',dict(state='complete'))
    except BaseException as e:
        atomic_json(out/'status.json',dict(state='error',error=str(e)));raise


def report(root):
    root=Path(root);rows=[]
    for mode in MODES:
        p=root/mode;status=json.loads((p/'status.json').read_text()) if (p/'status.json').exists() else {'state':'missing'}
        rows.append(dict(mode=mode,status=status,validation=json.loads((p/'validation.json').read_text()) if status['state']=='complete' else None))
    atomic_json(root/'summary.json',rows)
    body='<h1>Perception diagnostics</h1><a href="features/alignment.html">Image/region/grid audit</a><p>Privileged boxes; raw captures, not live rollout inputs. Same fixed geometry split for all resolutions. No test-set evaluation. Success here is not autonomous detection or proof of a native binding circuit.</p>'
    for row in rows:body+='<h2>'+row['mode']+'</h2><pre>'+html.escape(json.dumps({k:v for k,v in row.items() if k!='validation'}|({'metrics':{k:v for k,v in row['validation'].items() if k!='rows'}} if row['validation'] else {}),indent=2))+'</pre>'
    (root/'report.html').write_text('<!doctype html><meta charset="utf-8">'+body)


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['extract','train','report']);p.add_argument('--dataset');p.add_argument('--cache');p.add_argument('--output',required=True);p.add_argument('--config',default='configs/visual_readout.json');p.add_argument('--mode',choices=MODES);p.add_argument('--reviewed-captures',action='store_true');a=p.parse_args()
    if a.stage=='extract':extract(a.dataset,a.output,json.loads(Path(a.config).read_text()),a.reviewed_captures)
    elif a.stage=='train':train(a.cache,a.output,a.mode)
    else:report(a.output)

if __name__=='__main__':main()
