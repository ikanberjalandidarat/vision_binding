"""Render saved decisions; no inference or invented rescue outcomes."""
import argparse
import html
import json
from collections import defaultdict
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
from .tracking_rescue import disrupt


def build(audit,evaluation,local_demos=None):
    audit=Path(audit);evaluation=Path(evaluation);out=evaluation/'stories';out.mkdir(exist_ok=True)
    rows=json.loads((audit/'frames.json').read_text())
    results=json.loads((evaluation/'results.json').read_text())
    patches=json.loads((evaluation/'patch-results.json').read_text())
    lookup={(r['episode'],r['step'],r['stream'],r['method']):r for r in results}
    plook={(r['episode'],r['step'],tuple(r['layers']),r['control']):r for r in patches if r['stream']=='band70'}
    groups=defaultdict(list)
    for r in rows:groups[r['episode']].append(r)
    font=ImageFont.load_default()
    body='<h1>Recognition, memory and donor interventions</h1><p>Goal scores are color-times-shape readout scores, not calibrated success probabilities. R labels identify geometry proposals, not predicted attribute names. Cyan: recognition; orange: memory; magenta: donor-patched selection. No box means abstention. Gray boxes are candidate regions. The red band outline marks the image disruption, not an exact activation mask. Movement is prerecorded teacher control. Clean donors are privileged. Patched panels retain the corrupted pixels: only internal V activations were replaced. These are goal-match scores, not independent color/shape classifications.</p>'
    for eid,rr in groups.items():
        rr.sort(key=lambda r:r['step'])
        for layers in [(30,),(31,),(30,31)]:
            frames=[]
            for ordinal,r in enumerate(rr):
                path=Path(r['frame'])
                if not path.exists() and local_demos:path=Path(local_demos)/'episodes'/eid/path.name
                image=Image.open(path).convert('RGB');view,active,band=disrupt(image,ordinal,.7)
                baseline=lookup[(eid,r['step'],'band70','frame_only')]
                method='fallback_memory' if (eid,r['step'],'band70','fallback_memory') in lookup else 'recovery_memory'
                memory=lookup[(eid,r['step'],'band70',method)]
                entries=[('Recognition only',baseline,'cyan'),('Fallback memory' if method=='fallback_memory' else 'OLD recovery memory',memory,'orange'),('Clean donor V'+ '+'.join(map(str,layers)),plook.get((eid,r['step'],layers,'clean_donor')),'magenta'),('Outside-band donor control',plook.get((eid,r['step'],layers,'other_region')),'magenta')]
                canvas=Image.new('RGB',(896,800),'#f5f7fa');draw=ImageDraw.Draw(canvas)
                for col,(title,decision,color) in enumerate(entries):
                    x=(col%2)*448;y=(col//2)*400
                    draw.text((x+8,y+6),title,fill='black',font=font)
                    draw.text((x+8,y+22),f'Goal: {" ".join(r["goal"])} | step {r["step"]} | 70% band: {active}',fill='black',font=font)
                    vis=view.copy();vd=ImageDraw.Draw(vis)
                    if active:vd.rectangle(band,outline='red',width=1)
                    for k,region in enumerate(r['regions']):
                        selected=decision is not None and decision['selected_index']==region['index']
                        box=region['bbox_raw'];vd.rectangle(box,outline=color if selected else '#aaaaaa',width=3 if selected else 1)
                        label=f'R{k}'
                        if decision is not None:label+=f' {decision["scores"][k]:.5f}'
                        vd.text((box[0],max(0,box[1]-12)),label,fill='white',stroke_width=1,stroke_fill='black',font=font)
                    canvas.paste(vis,(x,y+45))
                    if decision is None:lines=['Patch not run on this frame.','No patched prediction is inferred.']
                    else:
                        selected=decision['selected_index'];choice=next((f'R{k}' for k,o in enumerate(r['regions']) if o['index']==selected),'ABSTAIN')
                        correctness={True:'correct',False:'incorrect',None:'excluded'}[decision['correct']]
                        lines=[f'Choice: {choice} | evaluation: {correctness}']
                        if col<2:lines.append('Source: '+decision.get('state','stateless'))
                        else:lines.append('RESCUED' if decision.get('rescue_eligible') and decision['correct'] is True else 'No correction of a scored error')
                        lines.append('Threshold: 0.05 | scores printed at each candidate')
                    for n,line in enumerate(lines):draw.text((x+8,y+332+n*17),line,fill='black',font=font)
                frames.append(canvas)
            name=eid+'-V'+'-'.join(map(str,layers))+'.gif'
            durations=[max(50,(b['tick']-a['tick'])*50) for a,b in zip(rr,rr[1:])]+[700]
            frames[0].save(out/name,save_all=True,append_images=frames[1:],duration=durations,loop=0)
            body+=f'<h2>{eid} / V{html.escape(str(layers))}</h2><img loading="lazy" src="{name}"><p><code>{html.escape(str(out/name))}</code></p>'
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><style>body{font:17px system-ui;max-width:1000px;margin:30px auto}img{max-width:100%}code{overflow-wrap:anywhere}</style>'+body)
    return out/'index.html'

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--audit',required=True);p.add_argument('--evaluation',required=True);p.add_argument('--local-demos');a=p.parse_args();print(build(a.audit,a.evaluation,a.local_demos))
