"""Vision-only, privileged-region diagnostics. No text prompts or decoder forwards."""
import copy
import html
import json
from contextlib import ExitStack
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageDraw
from .binding_data import load_binding, FEATURES
from .io import atomic_json, digest, file_hash, source_hash, environment
from .vision_data import mask_coverage, token_cells, overlay
from .vision_hooks import vision_backbone, capture_blocks, patch_block
from .binding_experiment import interventions, replacement_value

COLORS=['red','blue']
SHAPES=['arch','pillar']


def geometry_key(records):
    r=next(r for r in records if r['kind']=='full' and r['context']=='recipient')
    # Colors, IDs and permutations are deliberately excluded: all reversals stay together.
    return digest(sorted((o['type'],sorted(map(tuple,o['blocks']))) for o in r['objects']))


def split_groups(groups,seed):
    keys=sorted({geometry_key(rs) for rs in groups.values()})
    if len(keys)<3:
        raise ValueError('Need at least three distinct shape/position geometries; capture 24 binding permutations first. One family is extraction smoke only.')
    np.random.default_rng(seed).shuffle(keys)
    n=max(1,len(keys)//5)
    assignment={k:'test' if i<n else 'validation' if i<2*n else 'train' for i,k in enumerate(keys)}
    return {fid:assignment[geometry_key(rs)] for fid,rs in groups.items()}


def roi_tokens(record,h,w,merge,threshold=.5):
    result=[]
    for o in record['objects']:
        m=np.zeros((155,448),bool);x0,y0,x1,y1=o['bbox'];m[y0:y1,x0:x1]=True
        result.append([i for i,x in enumerate(mask_coverage(m,h,w,merge)) if x>=threshold])
    if any(not s for s in result) or sum(map(len,result))!=len(set().union(*map(set,result))):
        raise ValueError('Empty/overlapping bbox token regions')
    return result


class Encoder:
    def __init__(self,config):
        from importlib.metadata import version
        if version('transformers')!='4.55.0': raise ValueError('Requires validated transformers 4.55.0')
        from .models.qwen import Qwen
        if config.get('dtype')!='bfloat16' or config.get('load_in_4bit'):
            raise ValueError('Vision extraction requires unquantized BF16')
        self.wrapper=Qwen(config)
        self.visual,self.path=vision_backbone(self.wrapper.model)
        for p in self.wrapper.model.parameters(): p.requires_grad_(False)
        self.config=config
        # Explicitly prevent accidental decoder execution.
        self.guard=self.wrapper.layers[0].register_forward_pre_hook(self._deny_decoder)
    @staticmethod
    def _deny_decoder(module,args): raise RuntimeError('Language decoder execution forbidden in vision-only pipeline')
    def inputs(self,image):
        p=next(self.visual.parameters())
        data=self.wrapper.processor.image_processor(images=[image],return_tensors='pt',
            min_pixels=self.config['min_pixels'],max_pixels=self.config['max_pixels'])
        grid=data['image_grid_thw'].to(p.device)
        if grid.shape!=(1,3) or int(grid[0,0])!=1: raise ValueError('Expected single still-image grid')
        return data['pixel_values'].to(device=p.device,dtype=p.dtype),grid
    def run(self,inputs):
        pixels,grid=inputs
        return self.visual(pixels,grid_thw=grid)
    def features(self,inputs):
        values={};layer=self.config['readout_layer']
        with torch.no_grad(),capture_blocks(self.visual.blocks,[layer],values): self.run(inputs)
        return values[layer]


def save_tensor(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.pending');torch.save(data,temp);temp.replace(path)


def extract(dataset,output,config,reviewed=False):
    if not reviewed: raise ValueError('Review captures first')
    root,out=Path(dataset),Path(output);data,groups=load_binding(root)
    out.mkdir(parents=True,exist_ok=True)
    manifest=dict(experiment='vision_region_features_v1',dataset_hash=digest(data),config=config,
        source_hash=source_hash(),environment=environment(),privileged_regions=True,
        mask='bbox area coverage, no color-dependent pixel selection',decoder_executed=False,
        loading='Full Qwen checkpoint loaded for compatibility; only visual encoder is executed')
    mp=out/'manifest.json'
    if mp.exists() and json.loads(mp.read_text())!=manifest: raise ValueError('Incompatible extraction resume')
    atomic_json(mp,manifest);atomic_json(out/'status.json',{'state':'running'})
    try:
        enc=Encoder(config);entries=[]
        for fid,records in groups.items():
            for r in records:
                path=out/'features'/f'{r["record_id"]}.pt'
                if not path.exists():
                    with Image.open(root/r['image']) as im: image=im.convert('RGB')
                    inputs=enc.inputs(image);_,h,w=map(int,inputs[1][0].tolist())
                    merge=int(enc.visual.spatial_merge_size)
                    tokens=enc.features(inputs)
                    if tokens.shape[0]!=h*w: raise ValueError('Feature-grid mismatch')
                    positions=roi_tokens(r,h,w,merge,config['mask_threshold'])
                    save_tensor(path,dict(tokens=tokens,positions=positions,grid=[h,w],merge=merge))
                entries.append(dict(record=r,family=fid,geometry=geometry_key(records),file=str(path.relative_to(out)),sha256=file_hash(path)))
                print('Extracted',r['record_id'],flush=True)
        atomic_json(out/'index.json',entries);atomic_json(out/'status.json',{'state':'complete','records':len(entries)})
    except BaseException as e:
        atomic_json(out/'status.json',{'state':'error','error':str(e)});raise


def load_features(root):
    root=Path(root)
    if json.loads((root/'status.json').read_text())['state']!='complete': raise ValueError('Incomplete feature cache')
    entries=json.loads((root/'index.json').read_text())
    for e in entries:
        p=(root/e['file']).resolve()
        if root.resolve() not in p.parents or file_hash(p)!=e['sha256']: raise ValueError('Feature hash/path mismatch')
        e['features']=torch.load(p,map_location='cpu',weights_only=True)
    return entries,json.loads((root/'manifest.json').read_text())


def pool(tokens,positions): return torch.stack([tokens[s].mean(0) for s in positions])


class Readout(torch.nn.Module):
    def __init__(self,width):
        super().__init__();self.register_buffer('mean',torch.zeros(width));self.register_buffer('scale',torch.ones(width))
        self.color=torch.nn.Linear(width,2);self.shape=torch.nn.Linear(width,2)
    def forward(self,x):
        z=(x-self.mean)/self.scale
        return self.color(z),self.shape(z)
    def goal_scores(self,x,goal):
        c,s=self(x)
        return c.softmax(-1)[:,COLORS.index(goal[0])]*s.softmax(-1)[:,SHAPES.index(goal[1])]


def expected(record,goal):
    matches=[i for i,o in enumerate(record['objects']) if (o['color'],o['type'])==tuple(goal)]
    if len(matches)>1: raise ValueError('Ambiguous goal')
    return matches[0] if matches else -1


def prediction(scores,threshold):
    return int(scores.argmax()) if float(scores.max())>=threshold else -1


def trials(model,entries,threshold):
    rows=[]
    with torch.no_grad():
        for e in entries:
            r=e['record'];f=e['features'];x=pool(f['tokens'],f['positions'])
            for goal in FEATURES:
                scores=model.goal_scores(x,goal);target=expected(r,goal)
                choice=prediction(scores,threshold)
                rows.append(dict(family=e['family'],record_id=r['record_id'],kind=r['kind'],goal=list(goal),
                    expected=target,choice=choice,correct=choice==target,target_present=target!=-1,scores=scores.tolist()))
    return rows


def counts(rows):
    return [dict(target_present=p,n=len(rs),correct=sum(r['correct'] for r in rs))
            for p in (True,False) for rs in [[r for r in rows if r['target_present']==p]]]


def train(cache,output,config):
    out=Path(output)
    if out.exists(): raise ValueError('Use a fresh training output directory')
    entries,manifest=load_features(cache)
    groups={}
    for e in entries: groups.setdefault(e['family'],[]).append(e['record'])
    splits=split_groups(groups,config['seed'])
    # Train on full scenes only; evaluation includes absent scenes. Isolates are excluded.
    train_rows=[e for e in entries if splits[e['family']]=='train' and e['record']['kind']=='full']
    val=[e for e in entries if splits[e['family']]=='validation' and e['record']['kind'] in ('full','absent')]
    test=[e for e in entries if splits[e['family']]=='test' and e['record']['kind'] in ('full','absent')]
    def batch(es):
        x=torch.cat([pool(e['features']['tokens'],e['features']['positions']) for e in es])
        objs=[o for e in es for o in e['record']['objects']]
        return x,torch.tensor([COLORS.index(o['color']) for o in objs]),torch.tensor([SHAPES.index(o['type']) for o in objs])
    x,c,s=batch(train_rows);vx,vc,vs=batch(val)
    if set(c.tolist())!={0,1} or set(s.tolist())!={0,1}: raise ValueError('Training split misses attribute classes')
    torch.manual_seed(config['seed']);model=Readout(x.shape[1])
    model.mean.copy_(x.mean(0));model.scale.copy_(x.std(0).clamp_min(.01))
    opt=torch.optim.AdamW(model.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'])
    best=float('inf');best_state=None;history=[]
    for step in range(config['epochs']):
        opt.zero_grad();a,b=model(x);loss=torch.nn.functional.cross_entropy(a,c)+torch.nn.functional.cross_entropy(b,s)
        if not torch.isfinite(loss): raise RuntimeError('Nonfinite training loss')
        loss.backward();opt.step()
        with torch.no_grad():
            a,b=model(vx);v=float(torch.nn.functional.cross_entropy(a,vc)+torch.nn.functional.cross_entropy(b,vs))
        history.append(dict(epoch=step,train_loss=float(loss.detach()),validation_loss=v))
        if v<best: best=v;best_state=copy.deepcopy(model.state_dict())
    if best_state is None: raise RuntimeError('No finite training checkpoint')
    model.load_state_dict(best_state);model.eval()
    threshold_trials=[]
    for t in np.linspace(0,1,21):
        rs=trials(model,val,float(t));cs=counts(rs)
        if any(k['n']==0 for k in cs): raise ValueError('Validation needs present and absent goals')
        balanced=sum(k['correct']/k['n'] for k in cs)/2
        threshold_trials.append((balanced,float(t)))
    # Ties choose the lower threshold; no test data used for this decision.
    threshold=max(threshold_trials,key=lambda z:(z[0],-z[1]))[1]
    out.mkdir(parents=True)
    save_tensor(out/'readout.pt',dict(state=best_state,width=x.shape[1],threshold=threshold))
    results=trials(model,test,threshold)
    atomic_json(out/'manifest.json',dict(experiment='linear_visual_readout_v1',feature_manifest=manifest,
        feature_index_hash=file_hash(Path(cache)/'index.json'),config=config,splits=splits,
        independent_geometry_groups=len(set(e['geometry'] for e in entries)),
        scope='Privileged bbox proposals; factorized color/shape classifier plus validation-calibrated rejection. Not proof of native binding.',
        checkpoint_sha256=file_hash(out/'readout.pt')))
    atomic_json(out/'history.json',history);atomic_json(out/'test-results.json',results)
    with torch.no_grad():
        tx,tc,ts=batch(test);a,b=model(tx)
        attributes={'n':len(tc),'color_correct':int((a.argmax(-1)==tc).sum()),'shape_correct':int((b.argmax(-1)==ts).sum())}
    atomic_json(out/'summary.json',dict(threshold=threshold,test_binding=counts(results),test_attributes=attributes,
        validation_threshold_curve=threshold_trials,geometry_warning='Same arena/camera, held-out shape-position patterns only; small number of geometry groups'))
    atomic_json(out/'status.json',{'state':'complete'})


def gradcam(tokens,positions,model,goal,region):
    # At the readout input, not a gradient through earlier vision layers.
    a=tokens.detach().clone().requires_grad_(True)
    score=model.goal_scores(pool(a,positions),goal)[region]
    grad=torch.autograd.grad(score,a)[0]
    weights=grad.mean(0)
    cam=torch.relu((a.detach()*weights).sum(-1))
    if not torch.isfinite(cam).all(): raise RuntimeError('Nonfinite CAM')
    return cam, float(score.detach())


def heatmap(image,values,h,w,merge,path):
    v=values.detach().cpu().numpy();m=np.zeros((h,w),np.float32)
    for value,(r,c) in zip(v,token_cells(h,w,merge)): m[r,c]=value
    maximum=float(m.max());m=m/maximum if maximum>0 else m
    heat=np.asarray(Image.fromarray(m).resize(image.size,Image.Resampling.BILINEAR))
    rgb=np.asarray(image.convert('RGB')).astype(float)
    alpha=.6*heat[...,None]
    rgb=rgb*(1-alpha)+np.array([255,50,0])*alpha
    Image.fromarray(rgb.clip(0,255).astype('uint8')).save(path)
    return maximum


def evaluate(dataset,cache,readout,output,config,reviewed=False,patch=False):
    if not reviewed: raise ValueError('Review captures first')
    root,out=Path(dataset),Path(output)
    if out.exists(): raise ValueError('Use a fresh evaluation directory')
    data,groups=load_binding(root);entries,fm=load_features(cache)
    rp=Path(readout);rm=json.loads((rp/'manifest.json').read_text())
    if digest(data)!=fm['dataset_hash'] or rm['feature_manifest']!=fm or rm['feature_index_hash']!=file_hash(Path(cache)/'index.json'):
        raise ValueError('Dataset/cache/readout provenance mismatch')
    if file_hash(rp/'readout.pt')!=rm['checkpoint_sha256']: raise ValueError('Readout hash mismatch')
    ck=torch.load(rp/'readout.pt',map_location='cpu',weights_only=True)
    model=Readout(ck['width']);model.load_state_dict(ck['state']);model.eval();threshold=ck['threshold']
    out.mkdir(parents=True);atomic_json(out/'status.json',{'state':'running'})
    atomic_json(out/'manifest.json',dict(config=config,readout_manifest=rm,source_hash=source_hash(),
        dataset_hash=digest(data),patch=patch,gradcam='Readout-input token Grad-CAM for fixed region goal score; not attention or causal evidence'))
    rows=[];figures=[]
    try:
        candidates=[e for e in entries if rm['splits'][e['family']]=='test' and e['record']['kind'] in ('full','absent')]
        rows=trials(model,candidates,threshold)
        for e in candidates[:config.get('map_records',8)]:
            r=e['record'];f=e['features'];h,w=f['grid']
            with Image.open(root/r['image']) as im: image=im.convert('RGB')
            for goal in FEATURES:
                score=model.goal_scores(pool(f['tokens'],f['positions']),goal).detach()
                region=int(score.argmax());choice=prediction(score,threshold)
                cam,value=gradcam(f['tokens'],f['positions'],model,goal,region)
                name=r['record_id']+'-'+'-'.join(goal)
                maximum=heatmap(image,cam,h,w,f['merge'],out/f'{name}-cam.png')
                # Region scores are an explicit localization readout, distinct from Grad-CAM.
                canvas=image.copy();draw=ImageDraw.Draw(canvas)
                for j,o in enumerate(r['objects']):
                    draw.rectangle(o['bbox'],outline='cyan' if j==region else 'white',width=2)
                    draw.text((o['bbox'][0],max(0,o['bbox'][1]-12)),f'{float(score[j]):.2f}',fill='white')
                canvas.save(out/f'{name}-regions.png')
                figures.append(dict(name=name,goal=goal,record=r['record_id'],choice=choice,region=region,
                    max_cam=maximum,score=value,expected=expected(r,goal),condition='clean',cam_values=cam.tolist()))
        if patch:
            enc=Encoder(fm['config']);sites=[b.attn.qkv for b in enc.visual.blocks]
            nh=int(enc.visual.blocks[0].attn.num_heads)
            specs=interventions(config,len(sites),nh)
            layers=sorted({l for s in specs for l in s['layers']})
            if config.get('vision_patch_kind','v')!='v' or max(layers)>fm['config']['readout_layer']:
                raise ValueError('This runner patches V at or before the readout layer')
            patches=[]
            patch_maps=0
            for fid in sorted({e['family'] for e in candidates}):
                pair={e['record']['context']:e for e in candidates if e['family']==fid and e['record']['kind']=='full'}
                inputs={};caps={}
                for ctx,e in pair.items():
                    with Image.open(root/e['record']['image']) as im: inputs[ctx]=enc.inputs(im.convert('RGB'))
                    caps[ctx]={}
                    with torch.no_grad(),capture_blocks(sites,layers,caps[ctx],'v'): enc.run(inputs[ctx])
                    live=enc.features(inputs[ctx]);cached=e['features']['tokens']
                    if live.shape!=cached.shape or not torch.allclose(live,cached,atol=.02,rtol=.01): raise ValueError('Live/cache feature disagreement')
                for ctx,e in pair.items():
                    donor='color_swap' if ctx=='recipient' else 'recipient'
                    f=e['features'];n=f['tokens'].shape[0];pos=sorted(set().union(*map(set,f['positions'])))
                    donor_e=pair[donor]
                    if f['positions']!=donor_e['features']['positions']: raise ValueError('Counterfactual token alignment mismatch')
                    bg=sorted(set(range(n))-set(pos))
                    if len(bg)<len(pos): raise ValueError('Insufficient background')
                    bg=sorted(map(int,np.random.default_rng(config['seed']).choice(bg,len(pos),False)))
                    clean_all=trials(model,[e],threshold);donor_all=trials(model,[donor_e],threshold)
                    eligible=all(r['correct'] for r in clean_all+donor_all)
                    for spec in specs:
                        if not eligible: continue
                        positions=bg if spec['condition']=='background' else pos
                        with ExitStack() as stack:
                            for l in spec['layers']:
                                value=replacement_value(caps[ctx][l],caps[donor][l],positions,spec['condition'],config['seed']+l,spec['heads'],nh)
                                stack.enter_context(patch_block(sites[l],positions,value,n,'v',heads=spec['heads'],num_heads=nh,alpha=spec['alpha']))
                            tokens=enc.features(inputs[ctx])
                        name=f'{fid}-{ctx}-{spec["name"]}'
                        overlay(Image.open(root/e['record']['image']).convert('RGB'),positions,*f['grid'],f['merge'],out/f'{name}-patch.png')
                        for goal in FEATURES:
                            scores=model.goal_scores(pool(tokens,f['positions']),goal).detach()
                            original_scores=model.goal_scores(pool(f['tokens'],f['positions']),goal).detach()
                            if spec['condition']=='self' or spec['alpha']==0:
                                if not torch.allclose(scores,original_scores,atol=1e-4,rtol=1e-4): raise RuntimeError('Identity score gate failed')
                            choice=prediction(scores,threshold)
                            patches.append(dict(family=fid,context=ctx,goal=goal,**spec,choice=choice,
                                expected_original=expected(e['record'],goal),expected_donor=expected(donor_e['record'],goal),
                                scores=scores.tolist(),score_change=(scores-original_scores).tolist(),positions=positions,
                                overlay=f'{name}-patch.png'))
                            if spec['condition']=='donor' and patch_maps<config.get('map_records',8)*4:
                                with Image.open(root/e['record']['image']) as im: image=im.convert('RGB')
                                region=int(scores.argmax());cam,value=gradcam(tokens,f['positions'],model,goal,region)
                                map_name=name+'-'+'-'.join(goal)
                                maximum=heatmap(image,cam,*f['grid'],f['merge'],out/f'{map_name}-cam.png')
                                draw=ImageDraw.Draw(image)
                                for j,o in enumerate(e['record']['objects']):
                                    draw.rectangle(o['bbox'],outline='cyan' if j==region else 'white',width=2)
                                    draw.text((o['bbox'][0],max(0,o['bbox'][1]-12)),f'{float(scores[j]):.2f}',fill='white')
                                image.save(out/f'{map_name}-regions.png')
                                figures.append(dict(name=map_name,goal=goal,record=e['record']['record_id'],choice=choice,
                                    region=region,max_cam=maximum,score=value,expected=expected(e['record'],goal),
                                    condition=spec['name'],cam_values=cam.tolist()))
                                patch_maps+=1
                        atomic_json(out/'patch-results.json',patches)
                    atomic_json(out/f'gate-{fid}-{ctx}.json',{'eligible':eligible})
            atomic_json(out/'patch-results.json',patches)
        atomic_json(out/'results.json',rows);atomic_json(out/'maps.json',figures)
        cards=''.join(f'<article><h3>{html.escape(str(f["goal"]))} · image {f["record"]} · {html.escape(f["condition"])}</h3><p>Choice: {f["choice"]}; expected in actual image: {f["expected"]}. Indices start at 0; −1 means absent. CAM explains candidate {f["region"]}, even when rejected.</p><img src="{f["name"]}-regions.png" alt="Privileged regions and goal scores"><img src="{f["name"]}-cam.png" alt="Readout-input Grad-CAM"></article>' for f in figures)
        if patch:
            cards+='<h2>Exact intervention positions and scores</h2><table><tr><th>Family / direction</th><th>Intervention / goal</th><th>Original / donor / predicted</th><th>Region scores</th></tr>'
            for p in patches:
                cards+=f'<tr><td>{p["family"]} / {p["context"]}</td><td><a href="{p["overlay"]}">{html.escape(p["name"]+" / "+str(p["goal"]))}</a></td><td>{p["expected_original"]} / {p["expected_donor"]} / {p["choice"]}</td><td>{html.escape(str(p["scores"]))}</td></tr>'
            cards+='</table>'
        (out/'report.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><title>Vision-only readout</title><style>body{font:17px system-ui;margin:30px}img{width:448px;max-width:100%}article{border-bottom:1px solid #ccc;padding:20px}</style><h1>Vision-only readout</h1><p>No text generation. Known object boxes are privileged proposals, not learned detections. Left: factorized goal scores per region. Right: token Grad-CAM at the trained readout input; per-map normalization, not attention or proof of causality. ROI pooling constrains the gradients. All-zero maps remain uncolored. Maps show a configured subset; results.json contains every held-out trial.</p>'+cards+'</html>')
        atomic_json(out/'summary.json',{'binding':counts(rows),'map_records':len(figures),'patch_trials':len(patches) if patch else 0})
        atomic_json(out/'status.json',{'state':'complete','rows':len(rows)})
    except BaseException as e:
        atomic_json(out/'status.json',{'state':'error','error':str(e)});raise
