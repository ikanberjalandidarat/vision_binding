"""Startup region selection from RGB features and instruction, without movement."""
import json
from pathlib import Path
import torch
from .vla import Policy,tokenize,geometry_splits,VOCAB
from .visual_readout import save_tensor
from .io import atomic_json,file_hash,source_hash


def label_for(ep):
    objects=sorted(ep['record']['objects'],key=lambda o:o['bbox'][0])
    matches=[i for i,o in enumerate(objects) if [o['color'],o['type']]==ep['goal']]
    if len(matches)>1:raise ValueError('Ambiguous target')
    return matches[0] if matches else 4


class Selector(Policy):
    def __init__(self,width,hidden=64):
        super().__init__(width,hidden);self.action=torch.nn.Linear(hidden,5)


def forward(model,row,condition):
    x=row['features'];words=tokenize(row['instruction'])
    if condition=='no_image':x=torch.zeros_like(x)
    elif condition in ('no_color','no_shape'):
        removed=('red','blue') if condition=='no_color' else ('arch','pillar')
        words=words.clone()
        for word in removed:words[words==VOCAB.index(word)]=1
    elif condition!='clean':raise ValueError('Unknown selector control')
    return model(x,words,3,None)[0]


def evaluate(model,rows):
    output=[];summary=[]
    model.eval()
    with torch.no_grad():
        for condition in ('clean','no_image','no_color','no_shape'):
            start=len(output)
            for r in rows:
                logits=forward(model,r,condition)
                if not torch.isfinite(logits).all():raise RuntimeError('Nonfinite selector logits')
                pred=int(logits.argmax());truth=r['label'];objects=r['objects']
                outcome='correct' if pred==truth else 'false_absent' if pred==4 else 'missing_slot' if pred>=len(objects) else 'false_present' if truth==4 else 'wrong_region'
                if outcome=='wrong_region':
                    o=objects[pred];goal=r['goal']
                    outcome='same_color_wrong_shape' if o['color']==goal[0] else 'same_shape_wrong_color' if o['type']==goal[1] else 'wrong_both'
                output.append(dict(episode=r['episode'],family=r['family'],condition=condition,expected=truth,predicted=pred,correct=pred==truth,outcome=outcome,probabilities=logits.softmax(0).tolist()))
            rs=output[start:];summary.append(dict(condition=condition,n=len(rs),correct=sum(r['correct'] for r in rs),present_n=sum(r['expected']!=4 for r in rs),present_correct=sum(r['correct'] and r['expected']!=4 for r in rs),absent_n=sum(r['expected']==4 for r in rs),absent_correct=sum(r['correct'] and r['expected']==4 for r in rs)))
    return dict(rows=output,summary=summary,scope='Known object boxes define training/evaluation labels only. Policy input is pooled visual features and instruction. Input ablations are distribution changes, not head-level causal proof.')


def train(cache,demos,output,config):
    root=Path(cache);out=Path(output)
    fm=json.loads((root/'manifest.json').read_text());dm=Path(demos)
    if json.loads((root/'status.json').read_text())['state']!='complete':raise ValueError('Incomplete cache')
    if file_hash(dm/'episodes.json')!=fm['episodes_hash']:raise ValueError('Demonstration/cache mismatch')
    eps={r['episode']:r for r in json.loads((dm/'episodes.json').read_text())};rows=[]
    for r in json.loads((root/'index.json').read_text()):
        p=(root/r['file']).resolve()
        if root.resolve() not in p.parents or file_hash(p)!=r['sha256']:raise ValueError('Cache integrity failed')
        ep=eps[r['episode']];tensor=torch.load(p,map_location='cpu',weights_only=True)
        rows.append(dict(r,features=tensor['features'][0],label=label_for(ep),goal=ep['goal'],objects=sorted(ep['record']['objects'],key=lambda o:o['bbox'][0])))
    splits=geometry_splits([r['geometry'] for r in rows],config);tr=[r for r in rows if splits[r['geometry']]=='train'];val=[r for r in rows if splits[r['geometry']]=='validation']
    if not tr or not val:raise ValueError('Missing split')
    out.mkdir(parents=True,exist_ok=False);atomic_json(out/'status.json',{'state':'running'})
    try:
        torch.set_num_threads(4);torch.manual_seed(config['seed']);model=Selector(rows[0]['features'].shape[-1],config['hidden'])
        opt=torch.optim.AdamW(model.parameters(),lr=config['learning_rate'],weight_decay=.01);best=float('inf');state=None;history=[]
        def loss(r):return torch.nn.functional.cross_entropy(forward(model,r,'clean')[None],torch.tensor([r['label']]))
        for epoch in range(config.get('selector_epochs',80)):
            model.train();order=torch.randperm(len(tr));total=0
            for i in order:
                opt.zero_grad();value=loss(tr[int(i)])
                if not torch.isfinite(value):raise RuntimeError('Nonfinite selection loss')
                value.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();total+=float(value.detach())
            model.eval()
            with torch.no_grad():v=sum(float(loss(r)) for r in val)/len(val)
            history.append(dict(epoch=epoch,train_loss=total/len(tr),validation_loss=v))
            if v<best:best=v;state={k:x.detach().clone() for k,x in model.state_dict().items()}
        if state is None:raise RuntimeError('No selector checkpoint')
        model.load_state_dict(state);save_tensor(out/'selector.pt',dict(state=state,width=rows[0]['features'].shape[-1],hidden=config['hidden']))
        atomic_json(out/'validation.json',evaluate(model,val));atomic_json(out/'history.json',history)
        atomic_json(out/'manifest.json',dict(config=config,splits=splits,source_hash=source_hash(),feature_manifest=fm,checkpoint_sha256=file_hash(out/'selector.pt'),labels=['leftmost','second','third','fourth','absent'],scope='Independent startup selector, not used to drive these navigation policies; test set remains unevaluated'))
        atomic_json(out/'status.json',{'state':'complete'})
    except BaseException as e:atomic_json(out/'status.json',{'state':'error','error':str(e)});raise
