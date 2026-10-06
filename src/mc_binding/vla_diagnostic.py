"""Matched initial-frame and offline-history diagnostics. No simulator or training."""
import json
import html
from pathlib import Path
import torch
from PIL import Image
from .io import atomic_json,file_hash,source_hash
from .vla import Policy,Encoder,tokenize,observation_image,spatial_pool,action_metrics,ACTIONS
from .binding_data import FEATURES


def read(path):return json.loads(Path(path).read_text())


def checked_frame(root, sample):
    root=Path(root).resolve();p=(root/sample['frame']).resolve()
    if root not in p.parents or file_hash(p)!=sample['sha256']:raise ValueError('Teacher frame integrity mismatch')
    return p


def summarize(rows):
    groups=[]
    for source in ('teacher','rollout'):
        for mode in ('raw','neutral_bands_v1'):
            selected=[r for r in rows if r['source']==source and r['mode']==mode]
            groups.append(dict(source=source,mode=mode,n=len(selected),correct=sum(r['correct'] for r in selected),
                present=sum(r['present'] for r in selected),false_stops=sum(r['present'] and r['predicted']=='stop' for r in selected),
                absent=sum(not r['present'] for r in selected),absent_stops=sum(not r['present'] and r['predicted']=='stop' for r in selected)))
    return groups


def paired_metrics(rows):
    pairs=[]
    for mode in ('raw','neutral_bands_v1'):
        teacher={(r['episode'],tuple(r['goal'])):r for r in rows if r['source']=='teacher' and r['mode']==mode}
        matched=[(teacher[(r['episode'],tuple(r['goal']))],r) for r in rows if r['source']=='rollout' and r['mode']==mode]
        pairs.append(dict(mode=mode,n=len(matched),action_flips=sum(a['predicted']!=b['predicted'] for a,b in matched),
            teacher_correct_rollout_wrong=sum(a['correct'] and not b['correct'] for a,b in matched),
            mean_stop_probability_change=sum(b['probabilities'][3]-a['probabilities'][3] for a,b in matched)/len(matched) if matched else None))
    sensitivity=[]
    keys=sorted({(r['episode'],r['source'],r['mode']) for r in rows})
    for key in keys:
        group=[r for r in rows if (r['episode'],r['source'],r['mode'])==key]
        sensitivity.append(dict(episode=key[0],source=key[1],mode=key[2],distinct_predicted_actions=len({r['predicted'] for r in group}),
                                distinct_expected_actions=len({r['expected'] for r in group}),correct=sum(r['correct'] for r in group),n=len(group)))
    return dict(teacher_to_rollout=pairs,instruction_sensitivity=sensitivity)


def run(demos,rollout,policy,cache,output):
    root,rr,pp,cc,out=map(Path,(demos,rollout,policy,cache,output))
    for p in (root,rr,pp,cc):
        if read(p/'status.json')['state']!='complete':raise ValueError(f'Incomplete input {p}')
    pm=read(pp/'manifest.json');fm=read(cc/'manifest.json');rm=read(rr/'manifest.json')
    if file_hash(pp/'policy.pt')!=pm['checkpoint_sha256'] or rm['policy']['checkpoint_sha256']!=pm['checkpoint_sha256']:raise ValueError('Policy mismatch')
    if file_hash(root/'episodes.json')!=fm['episodes_hash'] or file_hash(cc/'index.json')!=pm['feature_index_hash']:raise ValueError('Feature/demonstration provenance mismatch')
    if pm['feature_manifest']!=fm:raise ValueError('Policy trained with different feature manifest')
    out.mkdir(parents=True,exist_ok=False);atomic_json(out/'status.json',{'state':'running'})
    try:
        torch.set_num_threads(4)
        ck=torch.load(pp/'policy.pt',map_location='cpu',weights_only=True);model=Policy(ck['width'],ck['hidden']);model.load_state_dict(ck['state']);model.eval()
        config=fm['config']
        if config.get('observation_mode','raw')!='raw':raise ValueError('This diagnostic expects the original raw-image policy')
        enc=Encoder(config);eps=read(root/'episodes.json');by_record={}
        for ep in eps:by_record.setdefault(ep['record']['record_id'],{})[tuple(ep['goal'])]=ep
        rows=[];panels=[]
        for result in read(rr/'results.json'):
            eid=result['episode']
            if not eid.startswith('e') or not eid[1:].isdigit():raise ValueError('Invalid episode ID')
            episode=read(rr/'episodes'/eid/'episode.json');matches=by_record[episode['record_id']]
            original=matches[tuple(episode['goal'])]
            paths={'teacher':checked_frame(root,original['samples'][0]),'rollout':rr/'episodes'/eid/'frame-0000.png'}
            images=[]
            for source,path in paths.items():
                for mode in ('raw','neutral_bands_v1'):
                    with Image.open(path) as im:image=observation_image(im,dict(config,observation_mode=mode))
                    saved=f'{eid}-{source}-{mode}.png';image.save(out/saved);images.append((source+' / '+mode,saved))
                    inputs=enc.inputs(image);_,h,w=map(int,inputs[1][0].tolist());features=spatial_pool(enc.features(inputs),(h,w),int(enc.visual.spatial_merge_size))
                    for goal in FEATURES:
                        expected=matches[tuple(goal)];present=expected['result']['goal_present'];truth=int(expected['samples'][0]['action'])
                        instruction=f'Go to the {goal[0]} {goal[1]}.'
                        with torch.no_grad():logits,_,_=model(features,tokenize(instruction),3,None)
                        if not torch.isfinite(logits).all():raise RuntimeError('Nonfinite policy logits')
                        pred=int(logits.argmax())
                        rows.append(dict(episode=eid,record_id=episode['record_id'],original_goal=episode['goal'],goal=list(goal),instruction=instruction,
                            source=source,mode=mode,present=present,expected=ACTIONS[truth],predicted=ACTIONS[pred],correct=pred==truth,
                            probabilities=logits.softmax(0).tolist(),frame_sha256=file_hash(path),image=saved))
            panels.append((eid,images))
            atomic_json(out/'initial-decisions.json',rows);print('Matched',eid,flush=True)
        cached=[]
        for r in read(cc/'index.json'):
            if pm['splits'][r['geometry']]!='test':continue
            p=(cc/r['file']).resolve()
            if cc.resolve() not in p.parents or file_hash(p)!=r['sha256']:raise ValueError('Cached features changed')
            cached.append(dict(r,data=torch.load(p,map_location='cpu',weights_only=True)))
        metrics={mode:action_metrics(model,cached,mode) for mode in ('teacher','predicted')}
        atomic_json(out/'offline-metrics.json',metrics)
        summary=dict(groups=summarize(rows),paired=paired_metrics(rows),scope='Repeated startup frames and instruction swaps are paired diagnostics, not independent scenes. Masking the original raw-trained policy is out-of-distribution; neither saliency nor this diagnostic proves a visual binding mechanism.')
        atomic_json(out/'summary.json',summary)
        atomic_json(out/'manifest.json',dict(source_hash=source_hash(),policy_sha256=pm['checkpoint_sha256'],demos_hash=fm['episodes_hash'],feature_index_hash=pm['feature_index_hash'],actions=ACTIONS))
        body='<h1>Matched-input diagnostic</h1><p>'+summary['scope']+'</p><p>Each instruction is evaluated with fresh recurrent state and the same initial previous-action sentinel (stop). Expected first actions come from matched teacher episodes. Masked bands remove top 40 and bottom 85 rows, including useful floor pixels; this is a broad appearance ablation.</p>'
        body+='<h2>Aggregate initial decisions</h2><pre>'+html.escape(json.dumps(summary['groups'],indent=2))+'</pre>'
        body+='<h2>Matched appearance changes and instruction sensitivity</h2><pre>'+html.escape(json.dumps(summary['paired'],indent=2))+'</pre>'
        body+='<h2>Offline teacher versus predicted action history</h2><pre>'+html.escape(json.dumps(metrics,indent=2))+'</pre>'
        for eid,images in panels:
            body+='<h2>'+eid+'</h2><div class="images">'+''.join('<figure><figcaption>'+label+'</figcaption><img src="'+name+'"></figure>' for label,name in images)+'</div>'
            body+='<table><tr><th>Source</th><th>Input</th><th>Goal</th><th>Teacher action</th><th>Policy action</th><th>P(stop)</th></tr>'
            for r in rows:
                if r['episode']==eid:body+='<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in (r['source'],r['mode'],' '.join(r['goal']),r['expected'],r['predicted'],round(r['probabilities'][3],3)))+'</tr>'
            body+='</table>'
        (out/'report.html').write_text('<!doctype html><meta charset="utf-8"><style>body{font:16px system-ui;max-width:1150px;margin:30px auto}.images{display:grid;grid-template-columns:1fr 1fr}figure{margin:10px}img{width:100%}td,th{padding:6px;border-bottom:1px solid #ddd}pre{white-space:pre-wrap}</style>'+body)
        atomic_json(out/'status.json',{'state':'complete','initial_decisions':len(rows),'offline_episodes':len(cached)})
    except BaseException as e:
        atomic_json(out/'status.json',{'state':'error','error':str(e)});raise
