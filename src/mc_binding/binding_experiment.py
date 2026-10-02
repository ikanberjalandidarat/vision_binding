"""Binding baseline and gated, reversible vision-head interventions. Frozen model."""
import base64
import html
from contextlib import ExitStack
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from .binding_data import load_binding, queries, parse_answer
from .vision_data import regions, object_mask, overlay
from .vision_hooks import capture_blocks, patch_block, vision_backbone
from .vision_pilot import patch_sets
from .io import RunStore, atomic_json, digest, environment, source_hash


def interventions(config, depth, nheads):
    sets=patch_sets(config,depth)
    heads=config.get('head_sets',[None])
    alphas=config.get('alphas',[1.])
    conditions=config.get('conditions',['donor','random','background'])
    if not heads or len({None if h is None else tuple(h) for h in heads})!=len(heads):
        raise ValueError('Empty or duplicate head sets')
    for hs in heads:
        if hs is not None and (not hs or hs!=sorted(set(hs)) or any(type(h) is not int or not 0<=h<nheads for h in hs)):
            raise ValueError('Invalid head sets')
    if not alphas or len(set(alphas))!=len(alphas) or any(type(a) not in (int,float) or not 0<=a<=1 for a in alphas):
        raise ValueError('Invalid replacement strengths')
    if not conditions or len(set(conditions))!=len(conditions) or any(c not in ('donor','random','background','mean_ablation') for c in conditions):
        raise ValueError('Invalid conditions')
    result=[]
    for name,layers in sets:
        for hs in heads:
            label=name+('-all' if hs is None else '-heads'+'-'.join(map(str,hs)))
            result.append(dict(name=label+'-self',layers=layers,heads=hs,alpha=1.,condition='self'))
            for alpha in alphas:
                for condition in conditions:
                    result.append(dict(name=f'{label}-{condition}-a{alpha:g}',layers=layers,heads=hs,alpha=alpha,condition=condition))
    return result


def replacement_value(original, donor, positions, condition, seed, heads, nheads):
    """Noise norm matches donor change within selected channels, not all heads."""
    value=donor[positions].clone()
    if condition=='self': return original[positions].clone()
    if condition=='mean_ablation': return original.mean(0,keepdim=True).expand(len(positions),-1).clone()
    if condition=='random':
        value=original[positions].clone()
        d=value.shape[1]//nheads
        cols=list(range(value.shape[1])) if heads is None else [h*d+c for h in heads for c in range(d)]
        noise=torch.randn((len(positions),len(cols)),generator=torch.Generator().manual_seed(seed))
        delta=donor[positions][:,cols]-value[:,cols]
        value[:,cols]+=noise*(delta.norm()/noise.norm().clamp_min(1e-12))
    return value


def binding_experiment(dataset, output, config, reviewed=False, patch=False):
    if not reviewed: raise ValueError('Inspect captures before --reviewed-captures')
    if config.get('dtype')!='bfloat16' or config.get('load_in_4bit') or not config.get('use_fast') or not config.get('check_finite_scores'):
        raise ValueError('Require unquantized BF16, fast processor and finite checks')
    from importlib.metadata import version
    if version('transformers')!='4.55.0': raise ValueError('Binding hooks require transformers 4.55.0')
    root,out=Path(dataset),Path(output)
    data,groups=load_binding(root)
    limit=config.get('family_limit',len(groups))
    if type(limit) is not int or not 1<=limit<=len(groups): raise ValueError('family_limit exceeds dataset or is invalid')
    chosen=list(groups)[:limit]
    store=RunStore(out,dict(experiment='binding_patch_v1' if patch else 'binding_baseline_v1',
        config=config,dataset_hash=digest(data),source_hash=source_hash(),environment=environment(),
        families=chosen,scope='development four-object conjunction selection; frozen weights; no movement or training'))
    status={'state':'running'}
    atomic_json(out/'status.json',status)
    try:
        from .models.qwen import Qwen
        model=Qwen(config)
        for p in model.model.parameters(): p.requires_grad_(False)
        visual,path=vision_backbone(model.model)
        nheads=int(visual.config.num_heads) if hasattr(visual,'config') else int(visual.blocks[0].attn.num_heads)
        specs=interventions(config,len(visual.blocks),nheads) if patch else []
        sites=[b.attn.qkv for b in visual.blocks] if patch else []
        layers=sorted({l for s in specs for l in s['layers']})
        kind=config.get('vision_patch_kind','v')
        if kind not in ('q','k','v'): raise ValueError('Binding interventions support q/k/v projections')
        directions=config.get('directions',['recipient','color_swap'])
        if not directions or len(set(directions))!=len(directions) or any(d not in ('recipient','color_swap') for d in directions): raise ValueError('Invalid directions')
        atomic_json(out/'model.json',{**model.manifest(),'vision_path':path,'vision_heads':nheads,
            'patch_kind':kind,'head_indexing':'qkv reshape(tokens,3,num_heads,head_dim); contiguous head channels within projection',
            'alpha_reference':'live recipient activation at each patched site','specs':specs})
        for fid in chosen:
            records=[r for r in groups[fid] if r['kind']!='isolated']
            full={r['context']:r for r in records if r['kind']=='full'}
            images={r['record_id']:Image.open(root/r['image']).convert('RGB') for r in records}
            cache={}
            def inp(r,q):
                key=r['record_id'],q['query_id']
                if key not in cache: cache[key]=model.inputs(images[r['record_id']],q['prompt'])
                return cache[key]
            baseline_id=fid+'-baseline'
            if not store.completed(baseline_id):
                rows=[]
                for r in records:
                    for q in queries(r):
                        x=inp(r,q)
                        raw=model.answer(x)
                        scores=model.choice_scores(x) if q['task']=='binding' else None
                        rows.append(dict(trial_key=f'{fid}:{r["record_id"]}:{q["query_id"]}',family=fid,
                            condition='clean',context=r['context'],record_id=r['record_id'],image=r['image'],kind=r['kind'],
                            **q,raw=raw,parsed=parse_answer(raw,q['task']),correct=parse_answer(raw,q['task'])==q['expected'],scores=scores))
                        atomic_json(out/'diagnostics'/f'{baseline_id}.json',rows)
                store.save(baseline_id,rows)
                store.export()
            import json
            baseline=json.loads((out/'families'/f'{baseline_id}.json').read_text())
            eligible=all(r['correct'] for r in baseline)
            atomic_json(out/'gates'/f'{fid}.json',dict(eligible=eligible,correct=sum(r['correct'] for r in baseline),n=len(baseline),
                criterion='all clean conjunction, absent-target, color and shape queries correct in both contexts',
                action='patch' if eligible and patch else 'baseline_only' if not patch else 'skip_patches'))
            if not patch or not eligible:
                print(f'{fid}: clean {sum(r["correct"] for r in baseline)}/{len(baseline)}; patches {"disabled" if not patch else "skipped"}',flush=True)
                continue
            clean={(r['record_id'],r['query_id']):r for r in baseline}
            images_full={ctx:images[r['record_id']] for ctx,r in full.items()}
            first=queries(full['recipient'])[0]
            grid=inp(full['recipient'],first)['image_grid_thw'].tolist()
            if len(grid)!=1 or grid[0][0]!=1: raise ValueError('Expected still image grid')
            _,h,w=grid[0];merge=int(model.processor.image_processor.merge_size)
            if merge!=int(visual.spatial_merge_size): raise ValueError('Merge mismatch')
            selected,bg=regions(images_full['recipient'],full['recipient'],h,w,merge,config['mask_threshold'])
            pos=sorted(set().union(*map(set,selected)))
            if len(bg)<len(pos): raise ValueError('Insufficient background tokens')
            background=sorted(map(int,np.random.default_rng(config['seed']).choice(bg,len(pos),replace=False)))
            for a,b in zip(full['recipient']['objects'],full['color_swap']['objects']):
                ma,mb=object_mask(images_full['recipient'],a),object_mask(images_full['color_swap'],b)
                if np.logical_and(ma,mb).sum()/np.logical_or(ma,mb).sum()<.9: raise ValueError('Donor silhouette mismatch')
            audit=out/'alignment'/fid;audit.mkdir(parents=True,exist_ok=True)
            for ctx in full:
                overlay(images_full[ctx],pos,h,w,merge,audit/f'{ctx}-objects.png')
                overlay(images_full[ctx],background,h,w,merge,audit/f'{ctx}-background.png')
            atomic_json(audit/'tokens.json',dict(grid=grid,merge=merge,objects=selected,background=background))
            caps={}
            for ctx,r in full.items():
                x=inp(r,queries(r)[0])
                if x['image_grid_thw'].tolist()!=grid: raise ValueError('Grid mismatch')
                caps[ctx]={}
                with torch.inference_mode(),capture_blocks(sites,layers,caps[ctx],kind): model.model(**x,use_cache=False)
                if any(v.shape!=(h*w,int(sites[l].out_features)//3) for l,v in caps[ctx].items()): raise ValueError('Projection layout mismatch')
            for ctx in directions:
                donor='color_swap' if ctx=='recipient' else 'recipient'
                recipient=full[ctx]
                donor_queries={q['query_id']:q for q in queries(full[donor])}
                for spec in specs:
                    unit=f'{fid}-{ctx}-{spec["name"]}'
                    if store.completed(unit): continue
                    positions=background if spec['condition']=='background' else pos
                    values={l:replacement_value(caps[ctx][l],caps[donor][l],positions,spec['condition'],config['seed']+l,spec['heads'],nheads) for l in spec['layers']}
                    def hooks():
                        stack=ExitStack()
                        try:
                            for l in spec['layers']:
                                stack.enter_context(patch_block(sites[l],positions,values[l],h*w,kind,
                                    heads=spec['heads'],num_heads=nheads,alpha=spec['alpha']))
                        except BaseException:
                            stack.close();raise
                        return stack
                    rows=[]
                    for q in queries(recipient):
                        if q['task']!='binding': continue
                        x=inp(recipient,q)
                        with hooks(): raw=model.answer(x)
                        with hooks(): scores=model.choice_scores(x)
                        ref=clean[recipient['record_id'],q['query_id']]
                        if spec['condition']=='self' or spec['alpha']==0:
                            if raw!=ref['raw'] or any(abs(scores['log_probability'][k]-ref['scores']['log_probability'][k])>1e-4 for k in scores['log_probability']):
                                raise RuntimeError('Self/alpha-zero identity gate failed')
                        expected_donor=donor_queries[q['query_id']]['expected']
                        margin=scores['log_probability'][expected_donor]-scores['log_probability'][q['expected']]
                        old=ref['scores']['log_probability']
                        rows.append(dict(trial_key=unit+':'+q['query_id'],family=fid,context=ctx,**q,
                            **spec,vision_patch_kind=kind,positions=positions,raw=raw,parsed=parse_answer(raw,'binding'),
                            expected_donor=expected_donor,chose_donor=parse_answer(raw,'binding')==expected_donor,
                            chose_original=parse_answer(raw,'binding')==q['expected'],scores=scores,
                            donor_minus_original=margin,margin_change=margin-(old[expected_donor]-old[q['expected']]),
                            recipient_image=recipient['image'],donor_image=full[donor]['image']))
                    store.save(unit,rows);store.export()
                    print(f'{unit}: {[r["raw"] for r in rows]}',flush=True)
        results=store.export()
        clean_rows=[r for r in results if r['condition']=='clean']
        counts=[]
        for task in ('binding','color','shape'):
            for kind_scene in ('full','absent'):
                rs=[r for r in clean_rows if r['task']==task and r['kind']==kind_scene]
                counts.append(dict(task=task,kind=kind_scene,n=len(rs),correct=sum(r['correct'] for r in rs)))
        atomic_json(out/'summary.json',dict(clean=counts,total_rows=len(results),
            eligible_families=[fid for fid in chosen if json.loads((out/'gates'/f'{fid}.json').read_text())['eligible']],
            interpretation='Development binding task; repeated queries/contexts are not independent scenes. Patches only on fully baseline-correct families.'))
        report(out,root,results)
        status={'state':'complete','rows':len(results)}
    except BaseException as exc:
        store.export();status={'state':'error','error':str(exc)};raise
    finally: atomic_json(out/'status.json',status)


def report(out,root,rows):
    def image(path): return 'data:image/png;base64,'+base64.b64encode(path.read_bytes()).decode()
    figures=[]
    for r in rows:
        if r['condition']=='clean' and r['task']=='binding' and r['query_id']=='red-pillar':
            figures.append(f'<figure><img src="{image(root/r["image"])}"><figcaption>{html.escape(r["family"]+" / "+r["context"]+" / "+r["kind"])}</figcaption></figure>')
    for p in sorted((out/'alignment').glob('*/*.png')):
        figures.append(f'<figure><img src="{image(p)}"><figcaption>Actual patched token positions: {html.escape(str(p.relative_to(out)))}</figcaption></figure>')
    lines=[]
    for r in rows:
        values=[r['family'],r['context'],r.get('name','clean '+r.get('kind','')),r['query_id'],r['expected'],r.get('expected_donor','—'),r['raw'],r.get('correct',r.get('chose_donor')),r.get('margin_change','—')]
        lines.append('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in values)+'</tr>')
    (out/'report.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><title>Binding experiment</title><style>body{font:16px system-ui;margin:30px;color:#234}img{max-width:448px;width:100%}figure{display:inline-block}td,th{padding:7px;border-bottom:1px solid #ddd}table{border-collapse:collapse}</style><h1>Binding experiment</h1><p>Real Minecraft inputs. Digits are current screen order from left (1) to right (4); 0 means absent. No pixels are replaced. Orange masks show activation-token sites. Self/alpha-zero are identity gates; mean ablation is a disruption control, not a donor intervention. Reverse directions swap input and donor roles. Every clean failure remains in this report; patches use only eligible families.</p>'+''.join(figures)+'<table><tr><th>Family</th><th>Input</th><th>Intervention</th><th>Question</th><th>Original correct</th><th>Donor correct</th><th>Answer</th><th>Clean correct / chose donor</th><th>Score-margin change</th></tr>'+''.join(lines)+'</table></html>')
