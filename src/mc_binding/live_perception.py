"""Audit moving-frame regions before frozen readout evaluation. No inferred visibility."""
import argparse
import html
import json
from pathlib import Path
import torch
from PIL import Image
from .io import atomic_json, file_hash
from .visual_readout import Encoder, Readout, COLORS, SHAPES
from .perception_study import rasterize, features_for, evaluate


def validate_annotation(row, size):
    if not row.get('reviewed'):raise ValueError('Every sampled frame requires explicit review')
    ids=set();visible=[]
    for obj in row['objects']:
        if obj['object_id'] in ids:raise ValueError('Duplicate object ID')
        ids.add(obj['object_id'])
        if obj['color'] not in COLORS or obj['type'] not in SHAPES:raise ValueError('Unknown attribute')
        state=obj.get('visibility')
        if state not in ('visible','occluded','out_of_frame','uncertain'):raise ValueError('Specify visibility for each object')
        if state=='visible':
            b=obj.get('bbox_raw');w,h=size
            if not isinstance(b,list) or len(b)!=4 or not all(isinstance(x,(int,float)) for x in b):raise ValueError('Visible object needs a pixel box')
            if not (0<=b[0]<b[2]<=w and 0<=b[1]<b[3]<=h):raise ValueError('Invalid box')
            visible.append(obj)
    return visible


def sample_steps(length, interval=0):
    if length < 1 or interval < 0: raise ValueError('Invalid sampling parameters')
    return sorted({0, length//2, length-1} | (set(range(0,length,interval)) if interval else set()))


def prepare(rollout, output, count, interval=0):
    root,out=Path(rollout).resolve(),Path(output)
    if count<1:raise ValueError('Positive episode count required')
    if json.loads((root/'status.json').read_text())['state']!='complete':raise ValueError('Incomplete rollout')
    out.mkdir(parents=True,exist_ok=False);(out/'frames').mkdir();rows=[]
    for result in json.loads((root/'results.json').read_text())[:count]:
        eid=result['episode'];ep=json.loads((root/'episodes'/eid/'episode.json').read_text());trace=ep['trajectory']
        for step in sample_steps(len(trace), interval):
            src=root/'episodes'/eid/trace[step]['frame'];name=f'{eid}-{step:04d}.png'
            with Image.open(src) as image:image.convert('RGB').save(out/'frames'/name)
            rows.append(dict(id=f'{eid}-{step:04d}',episode=eid,family=result['family'],step=step,phase='start' if step==0 else 'end' if step==len(trace)-1 else 'middle' if step==len(trace)//2 else 'interval',image='frames/'+name,image_sha256=file_hash(out/'frames'/name),goal=result['goal'],pose=trace[step]['pose'],reviewed=False,objects=[dict(object_id=o['object_id'],color=o['color'],type=o['type'],visibility='uncertain',bbox_raw=None) for o in ep['objects']]))
    data=dict(schema='live_regions_v1',rollout=str(root),rollout_manifest_sha256=file_hash(root/'manifest.json'),sampling=dict(episodes_requested=count,interval_decisions=interval,mandatory='unique start/middle/end',scope='first N recorded episodes; correlated development frames, not random independent scenes'),frames=rows)
    atomic_json(out/'annotations.json',data)
    template=Path(__file__).with_name('live_review.html').read_text()
    (out/'review.html').write_text(template.replace('DATA',json.dumps(data).replace('<','\\u003c')))



def run(audit, checkpoint, config, output):
    root,out=Path(audit),Path(output);data=json.loads((root/'annotations.json').read_text())
    if data.get('schema')!='live_regions_v1':raise ValueError('Unknown annotations')
    # Complete review and integrity preflight before loading the GPU model.
    for r in data['frames']:
        p=(root/r['image']).resolve()
        if root.resolve() not in p.parents or file_hash(p)!=r['image_sha256']:raise ValueError('Frame hash/path mismatch')
        with Image.open(p) as im:validate_annotation(r,im.size)
    checkpoint=Path(checkpoint);manifest=json.loads((checkpoint.parent/'manifest.json').read_text())
    if manifest['mode']!='8x16':raise ValueError('Use the 8x16 checkpoint')
    ck=torch.load(checkpoint,map_location='cpu',weights_only=True);model=Readout(ck['width']);model.load_state_dict(ck['state']);model.eval()
    out.mkdir(parents=True,exist_ok=False);atomic_json(out/'status.json',dict(state='running'))
    try:
        enc=Encoder(config);results=[]
        for r in data['frames']:
            with Image.open(root/r['image']) as im:im=im.convert('RGB')
            objects=validate_annotation(r,im.size)
            if not objects:
                results.append(dict(id=r['id'],phase=r['phase'],excluded='no visible object regions; no detector is evaluated'));continue
            inputs=enc.inputs(im);_,h,w=map(int,inputs[1][0].tolist());raster=rasterize(enc.features(inputs),h,w,int(enc.visual.spatial_merge_size))
            x=features_for(raster,objects,im.size,'8x16')
            scores=evaluate(model,[dict(x=x,record=dict(record_id=r['id'],objects=objects),family=r['family'])],ck['threshold'])
            ambiguous=any(o['visibility'] in ('occluded','uncertain') for o in r['objects'])
            if ambiguous:scores['selection']=[];scores['rows']=[]
            results.append(dict(id=r['id'],phase=r['phase'],selection_excluded=ambiguous,**scores))
        atomic_json(out/'results.json',results)
        phases=[]
        for phase in ('start','middle','end','interval'):
            rs=[r for r in results if r['phase']==phase and 'attributes' in r]
            phases.append(dict(phase=phase,frames=len(rs),attributes={k:sum(r['attributes'][k] for r in rs) for k in ('n','color_correct','shape_correct')},selection=[dict(present=p,n=sum(v['n'] for r in rs for v in r['selection'] if v['present']==p),correct=sum(v['correct'] for r in rs for v in r['selection'] if v['present']==p)) for p in (True,False)]))
        atomic_json(out/'summary.json',phases)
        atomic_json(out/'manifest.json',dict(annotations_sha256=file_hash(root/'annotations.json'),checkpoint_sha256=file_hash(checkpoint),config=config,scope='Human-audited visible regions; absent means absent from visible annotated regions, not absent in world. No navigation or patching. Correlated frames; development sample.'))
        (out/'report.html').write_text('<!doctype html><meta charset="utf-8"><h1>Live-frame readouts</h1><p>Known visible regions, not autonomous detection. Visibility exclusions apply. No retraining or threshold adjustment.</p><pre>'+html.escape(json.dumps(phases,indent=2))+'</pre>')
        atomic_json(out/'status.json',dict(state='complete'))
    except BaseException as e:atomic_json(out/'status.json',dict(state='error',error=str(e)));raise


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','evaluate']);p.add_argument('--rollout');p.add_argument('--audit');p.add_argument('--checkpoint');p.add_argument('--config',default='configs/visual_readout.json');p.add_argument('--output',required=True);p.add_argument('--episodes',type=int,default=8);p.add_argument('--interval',type=int,default=0);a=p.parse_args()
    if a.stage=='prepare':prepare(a.rollout,a.output,a.episodes,a.interval)
    else:run(a.audit,a.checkpoint,json.loads(Path(a.config).read_text()),a.output)

if __name__=='__main__':main()
