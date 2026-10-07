"""Privileged ray-cast proposals from recorded blocks and poses, not model detections."""
import argparse
import json
import math
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from .io import atomic_json, file_hash


def basis(pose):
    yaw,pitch=np.radians([pose['yaw'],pose.get('pitch',0)])
    right=np.array([-np.cos(yaw),0,-np.sin(yaw)])
    forward=np.array([-np.sin(yaw)*np.cos(pitch),-np.sin(pitch),np.cos(yaw)*np.cos(pitch)])
    down=np.cross(forward,right)
    return right,down,forward


def projected_box(obj,pose,focal,eye):
    right,down,forward=basis(pose);origin=np.array([pose['x'],pose['y']+eye,pose['z']])
    corners=np.array([np.array(b)+[dx,dy,dz] for b in obj['blocks'] for dx in (0,1) for dy in (0,1) for dz in (0,1)])-origin
    z=corners@forward
    if np.any(z<=.01):raise ValueError('Calibration object behind camera')
    u=224+focal*(corners@right)/z;v=140+focal*(corners@down)/z
    return np.array([u.min(),v.min(),u.max(),v.max()])


def calibrate(objects,pose):
    # Fit only initial annotation extents, never model predictions or evaluation labels.
    def loss(f,e):return np.mean([(projected_box(o,pose,f,e)-np.array(o['bbox_raw']))**2 for o in objects])
    f,e=200.,1.62
    for fs,es in [(120, .6),(10,.1),(1,.02)]:
        candidates=[(loss(ff,ee),ff,ee) for ff in np.linspace(max(30,f-fs),f+fs,31) for ee in np.linspace(max(.5,e-es),e+es,21)]
        err,f,e=min(candidates)
    if math.sqrt(err)>3:raise ValueError('Initial projection calibration exceeds 3 pixel RMS')
    return dict(focal_pixels=f,eye_height=e,initial_rms_pixels=math.sqrt(err),vertical_fov_degrees=math.degrees(2*math.atan(140/f)))


def raycast(objects,pose,focal,eye,size=(448,280)):
    w,h=size;right,down,forward=basis(pose);origin=np.array([pose['x'],pose['y']+eye,pose['z']])
    yy,xx=np.mgrid[:h,:w];rays=forward+((xx+.5-w/2)/focal)[...,None]*right+((yy+.5-h/2)/focal)[...,None]*down
    depth=np.full((h,w),np.inf);owner=np.full((h,w),-1,dtype=int)
    for i,obj in enumerate(objects):
        for block in obj['blocks']:
            near=np.full((h,w),-np.inf);far=np.full((h,w),np.inf)
            for k in range(3):
                d=rays[...,k];parallel=np.abs(d)<1e-10
                safe=np.where(parallel,1,d);a=(block[k]-origin[k])/safe;b=(block[k]+1-origin[k])/safe
                low=np.minimum(a,b);high=np.maximum(a,b)
                inside=block[k]<=origin[k]<=block[k]+1
                low=np.where(parallel,-np.inf if inside else np.inf,low);high=np.where(parallel,np.inf if inside else -np.inf,high)
                near=np.maximum(near,low);far=np.minimum(far,high)
            hit=(far>=np.maximum(near,.01))&(near>.01)&(near<depth)
            depth[hit]=near[hit];owner[hit]=i
    return owner


def propose(audit,output):
    root,out=Path(audit),Path(output);data=json.loads((root/'annotations.json').read_text());rollout=Path(data['rollout'])
    out.mkdir(parents=True,exist_ok=False);(out/'frames').mkdir();(out/'overlays').mkdir();calibrations={};fit_cache={};panels=[]
    for r in data['frames']:
        ep=json.loads((rollout/'episodes'/r['episode']/'episode.json').read_text());objects=ep['objects']
        if r['episode'] not in calibrations:
            key=json.dumps(dict(objects=[dict(blocks=o['blocks'],bbox=o['bbox_raw']) for o in objects],pose=ep['trajectory'][0]['pose']),sort_keys=True)
            if key not in fit_cache:fit_cache[key]=calibrate(objects,ep['trajectory'][0]['pose'])
            calibrations[r['episode']]=fit_cache[key]
        cal=calibrations[r['episode']]
        src=root/r['image']
        if file_hash(src)!=r['image_sha256']:raise ValueError('Frame integrity failed')
        im=Image.open(src).convert('RGB');im.save(out/r['image']);r['image_sha256']=file_hash(out/r['image'])
        owner=raycast(objects,r['pose'],cal['focal_pixels'],cal['eye_height']);pixels=np.asarray(im).astype(float);draw=ImageDraw.Draw(im)
        for i,o in enumerate(r['objects']):
            mask=owner==i;ys,xs=np.where(mask);o['visibility']='uncertain';o['bbox_raw']=None
            if len(xs):
                box=[int(xs.min()),int(ys.min()),int(xs.max()+1),int(ys.max()+1)]
                # Image agreement detects hand/HUD overlap and some unmodeled occlusion.
                rgb=pixels[mask];channel=0 if o['color']=='red' else 2
                agreement=float(((rgb[:,channel]>rgb[:,1]*1.3)&(rgb[:,channel]>rgb[:,2-channel]*1.3)&(rgb[:,channel]>35)).mean())
                o['bbox_raw']=box;o['proposal']=dict(method='block raycast',pixels=len(xs),color_agreement=agreement)
                if agreement>=.65 and len(xs)>=20:o['visibility']='visible'
                draw.rectangle(box,outline='cyan' if o['visibility']=='visible' else 'orange',width=2);draw.text((box[0],max(0,box[1]-10)),str(i)+' '+o['color']+' '+o['type'],fill='yellow')
            else:
                alone=raycast([objects[i]],r['pose'],cal['focal_pixels'],cal['eye_height'])
                o['visibility']='occluded' if (alone==0).any() else 'out_of_frame'
                o['proposal']=dict(method='block raycast',pixels=0,note='Classified by isolated raycast; unmodeled occluders remain a limitation')
        r['reviewed']=False
        r['review_flags']=[]
        for o in r['objects']:
            if o['visibility'] in ('uncertain','occluded'):r['review_flags'].append(o['object_id']+': '+o['visibility'])
            box=o.get('bbox_raw')
            if box and (box[0]==0 or box[1]==0 or box[2]==448 or box[3]==280):r['review_flags'].append(o['object_id']+': edge clipping')
            if box and (box[3]>195 or (box[1]<40 and box[2]>280)):r['review_flags'].append(o['object_id']+': potential hand/HUD overlay')
        print('Proposed',r['id'],flush=True)
        im.save(out/'overlays'/(r['id']+'.png'));panels.append((r['id'],im.copy()))
    data['projection']=dict(calibrations=calibrations,limitations='Objects only depth test; arena, hand, HUD and view bob not modeled. Color agreement is privileged QA, not an independent learned detector. No automatic reviewed flags.')
    atomic_json(out/'annotations.json',data)
    template=Path(__file__).with_name('live_review.html').read_text();(out/'review.html').write_text(template.replace('DATA',json.dumps(data).replace('<','\\u003c')))
    pages=[]
    for offset in range(0,len(panels),24):
        subset=panels[offset:offset+24];sheet=Image.new('RGB',(448*3,305*math.ceil(len(subset)/3)),'white');d=ImageDraw.Draw(sheet)
        for i,(name,im) in enumerate(subset):
            x=(i%3)*448;y=(i//3)*305;sheet.paste(im,(x,y+25));d.text((x+5,y+5),name,fill='black')
        name=f'contact-sheet-{offset//24:03d}.png';sheet.save(out/name);pages.append(name)
    flagged=[dict(id=r['id'],flags=r['review_flags']) for r in data['frames'] if r['review_flags']]
    atomic_json(out/'review-summary.json',dict(frames=len(data['frames']),episodes=len(calibrations),flagged_frames=len(flagged),flagged=flagged,reviewed_frames=0))
    gallery=''.join(f'<h2>Page {i+1}</h2><img loading="lazy" width="100%" src="{name}">' for i,name in enumerate(pages))
    (out/'projection.html').write_text('<!doctype html><meta charset="utf-8"><h1>Automatic region proposals</h1><p>All proposals remain unreviewed. Flags prioritize uncertain, clipped and possible hand/HUD-overlap cases; unflagged does not mean verified. Use Next flagged in the review interface. No model attention or patching.</p><a href="review.html">Review/correct proposals</a><p>'+str(len(data['frames']))+' frames; '+str(len(flagged))+' flagged.</p>'+gallery)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--audit',required=True);p.add_argument('--output',required=True);a=p.parse_args();propose(a.audit,a.output)
