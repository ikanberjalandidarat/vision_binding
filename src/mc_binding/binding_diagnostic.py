"""Fixed prompt/budget calibration; never changes baseline scores or patch gates."""
import base64
import html
from pathlib import Path
from PIL import Image
from .binding_data import load_binding, queries, parse_answer
from .io import RunStore, atomic_json, digest, environment, source_hash


def diagnostic_queries(record):
    rows=[]
    if record['kind']=='isolated':
        obj=record['objects'][0]
        for variant,prompt in [('isolated_shape','What shape is the colored structure? Answer arch or pillar only.'),
                               ('isolated_description','Is the colored structure an arch with an opening or a solid upright pillar? Answer arch or pillar only.')]:
            rows.append(dict(query_id=variant,variant=variant,task='shape',expected=obj['type'],prompt=prompt,max_new_tokens=64))
        rows.append(dict(query_id='isolated_color',variant='isolated_color',task='color',expected=obj['color'],prompt='What color is the colored structure? Answer red or blue only.',max_new_tokens=64))
        return rows
    for q in queries(record):
        for budget in (8,64):
            rows.append({**q,'query_id':q['query_id']+f'-original-{budget}','variant':f'original_{budget}','max_new_tokens':budget})
        if q['task']=='binding':
            prompt=f"Which structure is the {q['goal_color']} {q['goal_shape']}? Number the visible structures from left to right (1 to {len(record['objects'])}). Reply only with its number, or 0 if it is absent."
            rows.append({**q,'query_id':q['query_id']+'-existence','task':'existence','variant':'existence',
                         'expected':'no' if q['expected']=='0' else 'yes','max_new_tokens':64,
                         'prompt':f"Is there a {q['goal_color']} {q['goal_shape']} in this image? Answer yes or no only."})
        else:
            number=int(q['query_id'].split('-')[-1])
            ordinal=('first','second','third','fourth')[number-1]
            choices='arch or pillar' if q['task']=='shape' else 'red or blue'
            prompt=f'What {q["task"]} is the {ordinal} colored structure from the left? Reply {choices} only.'
        rows.append({**q,'query_id':q['query_id']+'-concise','variant':'concise_64','max_new_tokens':64,'prompt':prompt})
    return rows


def parsed(raw,task):
    if task=='existence':
        v=raw.strip().lower().rstrip('.!')
        return v if v in ('yes','no') else None
    return parse_answer(raw,task)


def summarize(rows):
    keys=sorted({(r['kind'],r['task'],r['variant'],r['target_present']) for r in rows})
    return [dict(kind=k,task=t,variant=v,target_present=p,n=len(rs),correct=sum(r['correct'] for r in rs),
                 invalid=sum(r['parsed'] is None for r in rs))
            for k,t,v,p in keys
            for rs in [[r for r in rows if (r['kind'],r['task'],r['variant'],r['target_present'])==(k,t,v,p)]]]


def run(dataset,output,config,reviewed=False):
    if not reviewed: raise ValueError('Review captures before --reviewed-captures')
    if config.get('dtype')!='bfloat16' or config.get('load_in_4bit') or not config.get('check_finite_scores'):
        raise ValueError('Require unquantized BF16 and finite checks')
    root,out=Path(dataset),Path(output)
    data,groups=load_binding(root)
    store=RunStore(out,dict(experiment='binding_diagnostic_v1',dataset_hash=digest(data),config=config,
                           source_hash=source_hash(),environment=environment(),scope='prompt calibration; no interventions, training or gate changes'))
    atomic_json(out/'status.json',{'state':'running'})
    try:
        from .models.qwen import Qwen
        model=Qwen(dict(config))
        atomic_json(out/'model.json',model.manifest())
        for fid,records in groups.items():
            for record in records:
                unit=fid+'-'+record['record_id']
                if store.completed(unit): continue
                with Image.open(root/record['image']) as im: image=im.convert('RGB')
                rows=[]
                for q in diagnostic_queries(record):
                    model.config['max_new_tokens']=q['max_new_tokens']
                    raw=model.answer(model.inputs(image,q['prompt']))
                    answer=parsed(raw,q['task'])
                    present='not_applicable'
                    if q['task'] in ('binding','existence'):
                        present='absent' if q['expected'] in ('0','no') else 'present'
                    rows.append(dict(**q,trial_key=unit+':'+q['query_id'],family=fid,record_id=record['record_id'],
                                     context=record['context'],kind=record['kind'],image=record['image'],raw=raw,parsed=answer,
                                     correct=answer==q['expected'],target_present=present))
                    atomic_json(out/'diagnostics'/f'{unit}.json',rows)
                store.save(unit,rows);store.export()
                print(f'{unit}: {sum(r["correct"] for r in rows)}/{len(rows)} strict answers correct',flush=True)
        rows=store.export()
        atomic_json(out/'summary.json',{'counts':summarize(rows),'rows':len(rows),'scope':'Calibration only; original scores and patch gates unchanged'})
        pictures=[]
        for record in data['records']:
            uri='data:image/png;base64,'+base64.b64encode((root/record['image']).read_bytes()).decode()
            pictures.append(f'<figure><img src="{uri}" alt="Minecraft capture"><figcaption>{html.escape(record["record_id"]+" / "+record["kind"]+" / "+record["context"])}</figcaption></figure>')
        table=[]
        for r in rows:
            table.append('<tr>'+''.join('<td>'+html.escape(str(r[k]))+'</td>' for k in ('record_id','variant','task','target_present','prompt','expected','raw','correct'))+'</tr>')
        (out/'report.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><title>Binding diagnostic</title><style>body{font:16px system-ui;margin:30px}figure{display:inline-block}img{width:448px;max-width:100%}td,th{padding:8px;border-bottom:1px solid #ccc}table{border-collapse:collapse}</style><h1>Binding diagnostic: calibration only</h1><p>Original prompts at 8 versus 64 tokens isolate the generation-budget change. Concise prompts use 64 tokens. Existence questions remove ordinal counting. Isolated images remove neighboring objects. Scores remain strict; raw explanatory answers are shown for inspection, not silently counted as correct. No patch gates or original results were changed.</p>'+''.join(pictures)+'<table><tr><th>Image</th><th>Variant</th><th>Task</th><th>Target present?</th><th>Prompt</th><th>Expected</th><th>Raw answer</th><th>Strict correct</th></tr>'+''.join(table)+'</table></html>')
        atomic_json(out/'status.json',{'state':'complete','rows':len(rows)})
    except BaseException as exc:
        store.export();atomic_json(out/'status.json',{'state':'error','error':str(exc)});raise
