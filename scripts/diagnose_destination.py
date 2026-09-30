"""No-patch prompt calibration; failures are measured rather than gated away."""
import argparse
import hashlib
import json
from pathlib import Path
from PIL import Image
from mc_binding.io import atomic_json, digest, environment, source_hash
from mc_binding.vision_data import load_swaps


def queries(objects):
    for side,obj in enumerate(objects):
        yield 'color_control', f"What color is the {('leftmost','rightmost')[side]} structure? Answer with color only.", obj['color'], 'color'
    for side,obj in enumerate(objects):
        color=obj['color']
        yield 'original_action', f'You must walk to the {color} structure. Which side is it on? Answer LEFT or RIGHT only.', side, 'side'
        yield 'direct_spatial', f'In this image, is the {color} structure on the left or the right? Answer left or right only.', side, 'side'


def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--output',required=True)
    p.add_argument('--config',default='configs/vision_replication_v.json');a=p.parse_args()
    root,out=Path(a.dataset),Path(a.output)
    data,groups=load_swaps(root);config=json.loads(Path(a.config).read_text())
    out.mkdir(parents=True,exist_ok=False)
    atomic_json(out/'manifest.json',dict(experiment='destination_prompt_calibration_v1',dataset_hash=digest(data),config=config,
        source_hash=source_hash(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),environment=environment(),
        scope='No patches, no training. All scenes are calibration for prompt selection; fresh held-out scenes needed after selecting a prompt.'))
    rows=[];atomic_json(out/'status.json',dict(state='running'))
    try:
        from mc_binding.models.qwen import Qwen
        from mc_binding.destination import parse_side
        from mc_binding.scoring import parse
        model=Qwen(config)
        for fid,records in groups.items():
            for r in records:
                if r['kind']!='pair':continue
                with Image.open(root/r['image']) as im:image=im.convert('RGB')
                for mode,prompt,expected,task in queries(r['objects']):
                    inp=model.inputs(image,prompt)
                    # Fingerprints help check actual input changes, not only printed prompts.
                    hashes={k:hashlib.sha256(inp[k].detach().float().cpu().numpy().tobytes()).hexdigest() for k in ('input_ids','pixel_values')}
                    raw=model.answer(inp)
                    parsed=parse_side(raw) if task=='side' else (parse(raw,task='color')['parsed'] or {}).get('color')
                    row=dict(family=fid,context=r['context'],image=r['image'],mode=mode,prompt=prompt,expected=expected,
                             raw=raw,parsed=parsed,correct=parsed==expected,input_hashes=hashes)
                    rows.append(row);atomic_json(out/'diagnostics.json',rows)
                    print(f'{fid} {r["context"]} {mode}: {raw!r}; expected={expected}; correct={row["correct"]}',flush=True)
        summary={}
        for mode in ('color_control','original_action','direct_spatial'):
            selected=[r for r in rows if r['mode']==mode]
            summary[mode]=dict(n=len(selected),correct=sum(r['correct'] for r in selected),invalid=sum(r['parsed'] is None for r in selected),
                               left_answers=sum(r['parsed']==0 for r in selected) if mode!='color_control' else None)
        atomic_json(out/'summary.json',summary);atomic_json(out/'status.json',dict(state='complete',rows=len(rows)))
    except BaseException as exc:
        atomic_json(out/'status.json',dict(state='error',error=str(exc),rows=len(rows)));raise
if __name__=='__main__':main()
