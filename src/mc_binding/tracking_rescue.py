"""Balanced memory stress test and donor-assisted V30/V31 rescue diagnostic."""
import argparse
from collections import defaultdict
from contextlib import ExitStack
import html
import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageDraw
from .io import atomic_json,file_hash,source_hash
from .visual_readout import Encoder,Readout
from .perception_study import rasterize,features_for
from .vision_data import token_cells
from .vision_hooks import capture_blocks,patch_block
from .target_tracking import choose

GOALS=(('red','pillar'),('blue','pillar'),('red','arch'),('blue','arch'))


def balanced_panel(jobs):
    def present(j):return any((o['color'],o['type'])==tuple(j['goal']) for o in j['record']['objects'])
    families=sorted({j['family'] for j in jobs});selected=[]
    for family in families:
        for goal in GOALS:
            matches=[j for j in jobs if j['family']==family and tuple(j['goal'])==goal and present(j) and j['record']['kind']=='full']
            if not matches:raise ValueError('Missing present goal/family cell')
            selected.append(sorted(matches,key=lambda j:j['episode'])[0])
    # One absent episode per goal, drawn only from the eligible validation panel.
    for goal in GOALS:
        matches=[j for j in jobs if tuple(j['goal'])==goal and not present(j)]
        if not matches:raise ValueError('Missing absent goal cell')
        selected.append(sorted(matches,key=lambda j:j['episode'])[0])
    return selected


class RecoveryMemory:
    """Frozen self-acquired template; similarity can bridge semantic dropouts.
    No true object IDs, boxes, world state, or donor features enter this API.
    """
    def __init__(self,threshold,similarity=.9,margin=.05,max_misses=2):
        self.threshold=threshold;self.similarity=similarity;self.margin=margin;self.max_misses=max_misses
        self.template=None;self.misses=0
    def select(self,x,scores):
        if not torch.isfinite(x).all() or not torch.isfinite(scores).all():raise ValueError('Nonfinite inputs')
        if self.template is None:
            i=choose(x,scores,self.threshold)
            if i>=0:self.template=x[i].detach().clone()
            return i,'acquired' if i>=0 else 'unacquired'
        if len(x):
            similarities=torch.nn.functional.cosine_similarity(x,self.template[None],dim=1)
            values,order=similarities.sort(descending=True)
            gap=float(values[0]-values[1]) if len(values)>1 else float('inf')
            if float(values[0])>=self.similarity and gap>=self.margin:
                self.misses=0
                return int(order[0]),'matched'
        self.misses+=1
        if self.misses>=self.max_misses:
            self.template=None;self.misses=0
            return -1,'lost_reset'
        return -1,'lost'


def disrupt(image,ordinal,strength):
    active=ordinal%6 in (2,3)
    box=(0,80,image.width,180)
    out=image.copy()
    if active:
        patch=image.crop(box)
        patch=Image.blend(patch,Image.new('RGB',patch.size,(128,128,128)),strength)
        out.paste(patch,box[:2])
    return out,active,box


def patch_positions(h,w,merge,size,box):
    width,height=size;x0,y0,x1,y1=box
    positions=[i for i,(r,c) in enumerate(token_cells(h,w,merge)) if x0<=(c+.5)*width/w<x1 and y0<=(r+.5)*height/h<y1]
    other=[i for i in range(h*w) if i not in positions]
    if not positions or len(other)<len(positions):raise ValueError('Insufficient equal-count spatial controls')
    # Deterministic sampling independent of target identity or success.
    control=sorted(np.random.default_rng(731).choice(other,len(positions),replace=False).tolist())
    return positions,control


def outcome(row,selected):
    indices=[o['index'] for o in row['regions']]
    ident=indices[selected] if selected>=0 else None
    eligible=row['world_present'] and row['target_index'] in indices
    correct=ident==row['target_index'] if eligible else (ident is None if not row['world_present'] else None)
    return dict(selected_index=ident,target_eligible=eligible,correct=correct,
                wrong_object=bool(eligible and ident is not None and ident!=row['target_index']),
                false_presence=bool(not row['world_present'] and ident is not None))


def summarize(rows):
    groups=defaultdict(list)
    for row in rows:groups[(row['stream'],row['method'],row['active'])].append(row)
    output=[]
    for (stream,method,active),rr in sorted(groups.items()):
        output.append(dict(stream=stream,method=method,active=active,frames=len(rr),
            eligible_present=sum(r['target_eligible'] for r in rr),correct_present=sum(r['correct'] is True and r['world_present'] for r in rr),
            wrong_object=sum(r['wrong_object'] for r in rr),false_presence=sum(r['false_presence'] for r in rr),
            absent_frames=sum(not r['world_present'] for r in rr),excluded_present=sum(r['world_present'] and not r['target_eligible'] for r in rr),
            lost=sum(r['selected_index'] is None for r in rr),
            corrected_vs_frame=sum(r['correct'] is True and r['frame_correct'] is False for r in rr),
            harmed_vs_frame=sum(r['correct'] is False and r['frame_correct'] is True for r in rr)))
    return output


def persistence(rows):
    groups=defaultdict(list)
    for r in rows:groups[(r['episode'],r['stream'],r['method'])].append(r)
    output=[]
    for (episode,stream,method),rr in sorted(groups.items()):
        streak=longest=0
        for r in sorted(rr,key=lambda r:r['step']):
            streak=streak+1 if r['wrong_object'] else 0
            longest=max(longest,streak)
        output.append(dict(episode=episode,stream=stream,method=method,
            max_consecutive_wrong_samples=longest,
            wrong_samples=sum(r['wrong_object'] for r in rr),
            note='Consecutive sampled observations, not continuous duration; excluded frames break streaks.'))
    return output


def run(audit,checkpoint,config,output):
    root=Path(audit);out=Path(output);checkpoint=Path(checkpoint)
    meta=json.loads((root/'manifest.json').read_text());rows=json.loads((root/'frames.json').read_text())
    if not meta.get('balanced'):raise ValueError('Use balanced preparation')
    if config['readout_layer']!=31:raise ValueError('Requires block-31 readout for V30/V31 tests')
    if file_hash(checkpoint)!=meta['checkpoint_sha256']:raise ValueError('Changed readout checkpoint')
    for row in rows:
        if file_hash(row['frame'])!=row['sha256']:raise ValueError('Changed frame')
    out.mkdir(parents=True,exist_ok=False);atomic_json(out/'status.json',dict(state='running'))
    settings=dict(similarity=.9,margin=.05,max_misses=2)
    atomic_json(out/'manifest.json',dict(meta,config=config,source_hash=source_hash(),memory=settings,
        corruption=dict(type='gray horizontal image band',box=[0,80,448,180],strengths=[.35,.7],schedule='sample ordinals 2,3 modulo 6; clean acquisition and recovery'),
        scope='Fixed development hypotheses, no threshold tuning. Clean and corrupted share privileged candidate boxes. Memory receives no donor features. Rescue uses same-frame clean donor: privileged, not deployment. Other-region control is equal-count outside the corruption, not guaranteed background. No navigation or training.'))
    try:
        ck=torch.load(checkpoint,map_location='cpu',weights_only=True);model=Readout(ck['width']);model.load_state_dict(ck['state']);model.eval()
        enc=Encoder(config);sites=[b.attn.qkv for b in enc.visual.blocks];results=[];patches=[];by_episode=defaultdict(list)
        for r in rows:by_episode[r['episode']].append(r)
        def features(inputs,regions,size):
            _,h,w=map(int,inputs[1][0].tolist())
            tokens=enc.features(inputs)
            x=features_for(rasterize(tokens,h,w,int(enc.visual.spatial_merge_size)),regions,size,'8x16') if regions else torch.empty((0,ck['width']))
            with torch.no_grad():scores=model.goal_scores(x,goal) if len(x) else torch.empty(0)
            return x,scores
        def record(r,stream,method,active,selection,state,base):
            return dict(episode=r['episode'],step=r['step'],family=r['family'],goal=r['goal'],world_present=r['world_present'],stream=stream,method=method,active=active,state=state,frame_correct=base,**outcome(r,selection))
        gallery=[]
        for eid,sequence in by_episode.items():
            sequence.sort(key=lambda r:r['step'])
            memories={s:RecoveryMemory(ck['threshold'],**settings) for s in ('clean','band35','band70')}
            old={};panels=[]
            for ordinal,r in enumerate(sequence):
                goal=r['goal'];image=Image.open(r['frame']).convert('RGB');clean=enc.inputs(image)
                x0,s0=features(clean,r['regions'],image.size)
                clean_choice=choose(x0,s0,ck['threshold']);clean_correct=outcome(r,clean_choice)['correct']
                panel=Image.new('RGB',(1344,310),'white');draw=ImageDraw.Draw(panel)
                clean_caps=None
                for col,(stream,strength) in enumerate([('clean',0),('band35',.35),('band70',.7)]):
                    viewed,active,box=disrupt(image,ordinal,strength);active=active and stream!='clean'
                    inp=clean if not active else enc.inputs(viewed)
                    x,scores=(x0,s0) if not active else features(inp,r['regions'],image.size)
                    f=choose(x,scores,ck['threshold']);base=outcome(r,f)['correct']
                    for method in ('frame_only','gated_template','recovery_memory'):
                        state='stateless'
                        if method=='frame_only':selected=f
                        elif method=='gated_template':
                            selected=choose(x,scores,ck['threshold'],old.get(stream))
                            if stream not in old and selected>=0:old[stream]=x[selected].detach().clone()
                        else:selected,state=memories[stream].select(x,scores)
                        result=record(r,stream,method,active,selected,state,base)
                        result.update(scores=scores.tolist(),clean_correct=clean_correct)
                        results.append(result)
                        if method=='recovery_memory':
                            vis=viewed.copy();vd=ImageDraw.Draw(vis)
                            for k,obj in enumerate(r['regions']):vd.rectangle(obj['bbox_raw'],outline='yellow' if k==selected else 'gray',width=3 if k==selected else 1)
                            panel.paste(vis,(col*448,30));draw.text((col*448+4,5),f'{stream} | frame {r["step"]} | {state}',fill='black')
                    # Donor diagnostic uses frame-only selection: no memory confound.
                    if active and r['regions']:
                        if not torch.equal(clean[1],inp[1]):raise ValueError('Clean/corrupt grid mismatch')
                        _,h,w=map(int,inp[1][0].tolist());pos,other=patch_positions(h,w,int(enc.visual.spatial_merge_size),image.size,box)
                        if clean_caps is None:
                            clean_caps={}
                            with torch.no_grad(),capture_blocks(sites,[30,31],clean_caps,'v'):enc.run(clean)
                        recipient={}
                        with torch.no_grad(),capture_blocks(sites,[30,31],recipient,'v'):enc.run(inp)
                        for layers in ([30],[31],[30,31]):
                            for control in ('self','clean_donor','other_region'):
                                positions=other if control=='other_region' else pos
                                caps=recipient if control=='self' else clean_caps
                                with ExitStack() as stack:
                                    for layer in layers:stack.enter_context(patch_block(sites[layer],positions,caps[layer][positions],h*w,'v'))
                                    px,ps=features(inp,r['regions'],image.size)
                                chosen=choose(px,ps,ck['threshold'])
                                if control=='self' and (not torch.allclose(ps,scores,atol=1e-4,rtol=1e-4) or chosen!=f):raise RuntimeError('Self-patch identity failed')
                                patches.append(dict(episode=eid,step=r['step'],stream=stream,layers=layers,control=control,positions=positions,clean_correct=clean_correct,corrupt_correct=base,rescue_eligible=clean_correct is True and base is False,scores=ps.tolist(),**outcome(r,chosen)))
                panels.append((r['tick'],panel))
            durations=[max(50,(b[0]-a[0])*50) for a,b in zip(panels,panels[1:])]+[500]
            name=eid+'.gif';panels[0][1].save(out/name,save_all=True,append_images=[p[1] for p in panels[1:]],duration=durations,loop=0);gallery.append(name)
            atomic_json(out/'results.json',results);atomic_json(out/'patch-results.json',patches);print('Evaluated',eid,flush=True)
        summary=summarize(results);atomic_json(out/'summary.json',summary)
        atomic_json(out/'persistence.json',persistence(results))
        rescue=[]
        for stream in ('band35','band70'):
            for layers in ([30],[31],[30,31]):
                for control in ('self','clean_donor','other_region'):
                    rr=[r for r in patches if r['stream']==stream and r['layers']==layers and r['control']==control]
                    eligible=[r for r in rr if r['rescue_eligible']]
                    rescue.append(dict(stream=stream,layers=layers,control=control,trials=len(rr),rescue_eligible=len(eligible),rescued=sum(r['correct'] is True for r in eligible),harmed=sum(r['corrupt_correct'] is True and r['correct'] is False for r in rr)))
        atomic_json(out/'rescue-summary.json',rescue)
        body='<h1>Balanced tracking stress test</h1><p>Clean / band35 / band70 views with recovery-memory selection in yellow. These are offline teacher frames, not selector-controlled motion. Candidate regions are privileged. No manual labels or new training. Patch results are donor-assisted diagnostics, not deployable performance. Out-of-view targets remain excluded, not absent. Thresholds fixed before this run; repeated frames are correlated.</p><h2>Memory comparison</h2><pre>'+html.escape(json.dumps(summary,indent=2))+'</pre><h2>V rescue and controls</h2><pre>'+html.escape(json.dumps(rescue,indent=2))+'</pre>'
        body+='<p><a href="persistence.json">Wrong-object persistence by episode</a>. Absent-target teacher episodes contain only their initial frame: absence results are static checks, not disappearance or occlusion tests.</p>'
        for name in gallery:body+=f'<h3>{name}</h3><img style="max-width:100%" src="{name}"><p><code>{html.escape(str((out/name).resolve()))}</code></p>'
        (out/'report.html').write_text('<!doctype html><meta charset="utf-8"><style>body{font:16px system-ui;max-width:1350px;margin:30px auto}code{overflow-wrap:anywhere}</style>'+body)
        atomic_json(out/'status.json',dict(state='complete'))
    except BaseException as e:atomic_json(out/'status.json',dict(state='error',error=str(e)));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--audit',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True);p.add_argument('--config',default='configs/visual_readout.json');a=p.parse_args();run(a.audit,a.checkpoint,json.loads(Path(a.config).read_text()),a.output)
