"""Post-hoc visual references, never evidence of ongoing model inference/patching."""
import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from mc_binding.capture_pairs import color_mask,CROP
from mc_binding.vision_data import token_cells
from mc_binding.io import digest,file_hash,atomic_json


def visible_region(raw,color):
    mask=color_mask(raw,color).copy()
    # Conservative exclusions for this 448x280 first-person renderer.
    # Missing portions are intentionally NOT inferred through the hand/HUD.
    mask[-24:]=False
    mask[:35,290:]=False
    mask[180:,290:]=False
    seen=np.zeros(mask.shape,bool);best=[]
    for y,x in zip(*np.where(mask)):
        if seen[y,x]:continue
        stack=[(int(y),int(x))];seen[y,x]=True;component=[]
        while stack:
            yy,xx=stack.pop();component.append((yy,xx))
            for ny,nx in ((yy-1,xx),(yy+1,xx),(yy,xx-1),(yy,xx+1)):
                if 0<=ny<mask.shape[0] and 0<=nx<mask.shape[1] and mask[ny,nx] and not seen[ny,nx]:
                    seen[ny,nx]=True;stack.append((ny,nx))
        if len(component)>len(best):best=component
    out=np.zeros(mask.shape,bool)
    if len(best)<30:return out,None
    ys,xs=zip(*best);out[ys,xs]=True
    return out,[min(xs),min(ys),max(xs)+1,max(ys)+1]


def annotate(root,decisions):
    d=json.loads((root/'approach.json').read_text());trial=d['trial']
    manifest=json.loads((decisions/'manifest.json').read_text())
    assert digest(manifest)==d['decision_manifest_hash']
    assert file_hash(decisions/'results.jsonl')==d['decision_results_sha256']
    recorded=[json.loads(l) for l in (decisions/'results.jsonl').read_text().splitlines()]
    assert next(r for r in recorded if r['trial_key']==trial['trial_key'])==trial
    alignment=json.loads((decisions/'alignment'/trial['family']/'tokens.json').read_text())
    _,h,w=alignment['grid'][0];cells=token_cells(h,w,alignment['merge'])
    frames=[r for r in d['trajectory'] if 'frame' in r];animation=[];audit=[]
    condition=trial['condition'];edited=bool(trial['positions'])
    label={'clean_recipient':'CLEAN: no activation patch','self':'SELF: original activations copied back','both_objects':'DONOR: V30+31 copied at decision time'}[condition]
    for index,r in enumerate(frames):
        raw=Image.open(root/r['frame']).convert('RGB');rgba=raw.convert('RGBA');layer=Image.new('RGBA',raw.size)
        pen=ImageDraw.Draw(layer);regions=[]
        for side,obj in enumerate(d['objects']):
            mask,box=visible_region(np.asarray(raw),obj['color'])
            regions.append(dict(side=side,color=obj['color'],bbox=box,visible_pixels=int(mask.sum())))
            if index>0 and edited:
                tint=np.zeros((raw.height,raw.width,4),np.uint8);tint[mask]=[255,140,0,65]
                layer=Image.alpha_composite(layer,Image.fromarray(tint,'RGBA'));pen=ImageDraw.Draw(layer)
            if box and side==trial['choice_side']:
                pen.rectangle(box,outline=(0,255,255,255),width=2)
        if index==0:
            for pos in trial['positions']:
                rr,cc=cells[pos]
                pen.rectangle((cc*448/w,CROP[1]+rr*155/h,(cc+1)*448/w,CROP[1]+(rr+1)*155/h),fill=(255,140,0,75),outline=(255,140,0,255))
        image=Image.alpha_composite(rgba,layer).convert('RGB')
        canvas=Image.new('RGB',(448,400),'#152333');canvas.paste(image,(0,120));draw=ImageDraw.Draw(canvas)
        pose=r['pose'];obj=d['objects'][trial['choice_side']]
        lines=[label,'NO LIVE PATCHING during this recorded walk',
               'Orange: exact initial token grid' if index==0 and edited else ('Orange: heuristic object reference, NOT token grid' if edited else 'No orange: clean decision'),
               'Cyan: chosen object reference, NOT model attention',
               f"Goal {trial['goal_color']} | chosen {obj['color']} {obj['type']} | step {r['tick']}",
               f"X {pose['x']:.2f}  Y {pose['y']:.2f}  Z {pose['z']:.2f}",
               f"Requested-waypoint distance: {r['distance_to_requested']:.2f}"]
        for i,line in enumerate(lines):draw.text((5,3+16*i),line,fill='white')
        animation.append(canvas);audit.append(dict(tick=r['tick'],regions=regions,exact_patch_grid=index==0 and edited))
        if index in (0,len(frames)//2,len(frames)-1):canvas.save(root/f'regions-step-{r["tick"]:04d}.png')
    durations=[max(20,50*(b['tick']-a['tick'])) for a,b in zip(frames,frames[1:])]+[1200]
    animation[0].save(root/'movement-regions.gif',save_all=True,append_images=animation[1:],duration=durations,loop=0,optimize=False)
    atomic_json(root/'region-overlay.json',dict(method='posthoc_largest_color_component_v1',limitations='Not model detection, attention, optical flow or live patch positions. Hand, tutorial and HUD areas excluded. Regions can be partial or absent.',source_trial=trial['trial_key'],frames=audit))
    report=root/'report.html';text=report.read_text();start='<!-- moving-region-reference -->';end='<!-- /moving-region-reference -->'
    if start in text:text=text[:text.index(start)]+text[text.index(end)+len(end):]
    section=start+'<h2>Annotated movement: where are the previously edited objects?</h2><img src="movement-regions.gif" alt="Movement with exact initial patch grid and heuristic moving-object overlays"><p><b>No activations were patched while walking.</b> Orange at step 0 shows the exact recorded decision-time token positions. Later orange shading follows visible object-color regions as a post-hoc visual reference; it is not an exact reprojected token grid. Cyan identifies the chosen object using the same heuristic, not a model detector. Self runs copied original values; donor runs copied donor values. HUD and hand areas are excluded, so masks may be incomplete. Raw frames and the original GIF are unchanged.</p><p><a href="region-overlay.json">Per-frame overlay method and bounds</a></p>'+end
    report.write_text(text.replace('<h1>Minecraft approach recording</h1>','<h1>Minecraft approach recording</h1>'+section))


def main():
    root=Path(__file__).resolve().parents[1]/'runs/oscar'
    replay=root/'destination-replay-6869428';decisions=root/'destination-replication-6868954'
    for p in sorted(replay.iterdir()):
        if (p/'approach.json').exists():annotate(p,decisions);print(p.name,flush=True)
if __name__=='__main__':main()
