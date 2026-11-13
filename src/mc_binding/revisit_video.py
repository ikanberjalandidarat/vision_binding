"""Validate measured revisit clips before frozen extraction. No model fitting."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
from .io import atomic_json, digest, file_hash
from .revisit_dataset import specification, accepted

CONFIG = Path(__file__).resolve().parents[2]/'configs/revisit_evaluation_v1.json'


def protocol():
    return json.loads(CONFIG.read_text())


def mask_ui(pixels):
    p=protocol();out=pixels.copy();h,w=out.shape[:2]
    for x0,y0,x1,y1 in p['mask_rectangles_xyxy_fraction']:
        out[int(y0*h):int(np.ceil(y1*h)),int(x0*w):int(np.ceil(x1*w))]=p['mask_fill_rgb']
    return out


def clip_ticks(visit):
    end=visit['endpoint_tick'];ticks=list(range(end-120,end+1,8))
    if end<120 or visit['clip_ticks']!=ticks:
        raise ValueError('Clip must contain exactly 16 past/endpoint frames at eight-tick cadence')
    return ticks


def load_revisits(root, verify_pixels=False, preview=None):
    root=Path(root).resolve();windows=[];sources={};checked=set()
    for r in range(6):
        folder=root/f'route-{r}';spec=specification(r)
        for name in ['manifest.json','specification.json','frames.json','visits.json','status.json']:
            sources[f'route-{r}/{name}']=file_hash(folder/name)
        manifest=json.loads((folder/'manifest.json').read_text())
        if not manifest.get('is_minecraft') or manifest.get('teleports_during_recording')!=0:
            raise ValueError('Require real continuous Minecraft recording')
        if digest(json.loads((folder/'specification.json').read_text()))!=digest(spec):raise ValueError('Unexpected specification')
        if json.loads((folder/'status.json').read_text())['state']!='complete':raise ValueError('Incomplete route')
        fs=json.loads((folder/'frames.json').read_text());vs=json.loads((folder/'visits.json').read_text())
        frames={f['elapsed_ticks']:f for f in fs}
        if len(frames)!=len(fs) or sorted(frames)!=list(range(0,max(frames)+1,2)):raise ValueError('Frame clock discontinuity')
        if len(vs)!=64 or {(v['site'],v['target_heading']) for v in vs}!={(i,h) for i in range(16) for h in (0,90,180,270)}:
            raise ValueError('Incomplete site/heading coverage')
        for v in vs:
            ticks=clip_ticks(v);selected=[]
            for t in ticks:
                f=dict(frames[t]);path=(folder/f['frame']).resolve()
                if folder not in path.parents:raise ValueError('Frame escapes route directory')
                if verify_pixels and path not in checked:
                    if file_hash(path)!=f['sha256']:raise ValueError('Raw frame hash mismatch')
                    with Image.open(path) as im:
                        if im.mode!='RGB' or im.size!=(448,280):raise ValueError('Unexpected raw RGB')
                    checked.add(path)
                f['frame']=str(path.relative_to(root));selected.append(f)
            p=selected[-1]['pose']
            if p!=v['pose'] or v['frame']!=frames[ticks[-1]]['frame']:raise ValueError('Endpoint metadata mismatch')
            if not accepted(p,spec['sites'][v['site']],v['target_heading']):raise ValueError('Visit pose outside acceptance')
            if not all(np.isfinite([f['pose'][k] for k in ('x','y','z','yaw')]).all() for f in selected):raise ValueError('Nonfinite clip pose')
            windows.append(dict(episode=f'route-{r}',geometry='revisit-world',route=r,split=spec['split'],
                visit=v['visit'],site=v['site'],target_heading=v['target_heading'],ticks=ticks,frames=selected,
                position=[p['x'],p['z']],heading=p['yaw']))
    provenance=dict(dataset_sha256=digest(sources),metadata_hashes=sources,protocol_sha256=digest(protocol()))
    if preview:
        # Training-only montage of the exact model clip endpoints after UI masking.
        canvas=Image.new('RGB',(4*224,4*140))
        for i,w in enumerate(windows[:16]):
            with Image.open(root/w['frames'][-1]['frame']) as im:
                canvas.paste(Image.fromarray(mask_ui(np.asarray(im))).resize((224,140)),((i%4)*224,(i//4)*140))
        canvas.save(preview)
    return windows,provenance


def verify(root,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    atomic_json(out/'status.json',dict(state='running'))
    try:
        windows,provenance=load_revisits(root,True,out/'masked-training-endpoints.png')
        paths={s:{f['frame'] for w in windows if w['split']==s for f in w['frames']} for s in ('train','validation','test')}
        if any(paths[a]&paths[b] for a,b in [('train','validation'),('train','test'),('validation','test')]):raise ValueError('Cross-split frame path reuse')
        # Hash-identical content can occur at distinct observations; report, not silently delete.
        hashes={s:{f['sha256'] for w in windows if w['split']==s for f in w['frames']} for s in paths}
        similarities=[]
        for a,b in [(0,1),(2,3),(4,5)]:
            wa={(w['site'],w['target_heading']):w for w in windows if w['route']==a}
            wb={(w['site'],w['target_heading']):w for w in windows if w['route']==b}
            diffs=[]
            for key in wa:
                pa=np.array([[f['pose']['x'],f['pose']['z']] for f in wa[key]['frames']])
                pb=np.array([[f['pose']['x'],f['pose']['z']] for f in wb[key]['frames']])
                diffs.append(np.linalg.norm(pa-pb,axis=1))
            diffs=np.array(diffs)
            similarities.append(dict(routes=[a,b],mean_xz_separation_by_clip_frame=diffs.mean(0).tolist(),
                fraction_aligned_pose_pairs_under_075=float((diffs<.75).mean()),
                note='Matched site/heading context diagnostic, not proof of independent recent views'))
        atomic_json(out/'protocol.json',protocol())
        atomic_json(out/'clips.json',windows)
        atomic_json(out/'report.json',dict(state='complete',clips=len(windows),clips_per_split={s:sum(w['split']==s for w in windows) for s in paths},
            raw_png_hashes_verified=True,verified_unique_frames=sum(len(v) for v in paths.values()),
            **provenance,shared_frame_paths_across_splits=0,
            shared_content_hashes_across_splits={a+'_'+b:len(hashes[a]&hashes[b]) for a,b in [('train','validation'),('train','test'),('validation','test')]},
            context_diagnostics=similarities,mask_preview='masked-training-endpoints.png',
            model_run=False,scope='Selected model frames rehashed; source metadata bound to protocol. Tutorial region is masked on every frame, irrespective of presence.'))
        atomic_json(out/'status.json',dict(state='complete',clips=len(windows)))
    except BaseException as e:
        atomic_json(out/'status.json',dict(state='error',error=repr(e)));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('dataset');p.add_argument('output');a=p.parse_args();verify(a.dataset,a.output)
