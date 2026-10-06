"""Small controlled-language VLA: frozen vision, learned text/attention/GRU action policy."""
import json
import os
import select
import subprocess
import time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from .io import atomic_json,digest,file_hash,source_hash,environment
from .binding_data import load_binding,FEATURES
from .visual_readout import Encoder,save_tensor,split_groups,geometry_key
from .vision_data import token_cells

ACTIONS=['forward','turn_left','turn_right','stop']
VOCAB=['<pad>','<unk>','go','to','the','red','blue','arch','pillar','walk','approach','find']


def tokenize(text):
    import re
    words=re.findall(r'[a-z]+',text.lower())
    if not words:raise ValueError('Empty instruction')
    return torch.tensor([VOCAB.index(w) if w in VOCAB else 1 for w in words],dtype=torch.long)


def observation_image(image, config):
    """Versioned policy input transform; recorded raw HUD images remain intact."""
    mode=config.get('observation_mode','raw')
    image=image.convert('RGB').copy()
    if image.size!=(448,280):raise ValueError('Expected raw 448x280 observation')
    if mode=='neutral_bands_v1':
        from PIL import ImageDraw
        draw=ImageDraw.Draw(image)
        draw.rectangle((0,0,447,39),fill=(128,128,128))
        draw.rectangle((0,195,447,279),fill=(128,128,128))
    elif mode!='raw':raise ValueError('Unknown observation_mode')
    return image


def loss_scale(length, mean_length, weighting):
    if weighting=='episode':return 1.
    if weighting=='timestep':return length/mean_length
    raise ValueError('loss_weighting must be episode or timestep')


def decision_weights(actions, config):
    """Predeclared startup/turn weighting, independent of validation labels."""
    weights=torch.ones(len(actions),dtype=torch.float32,device=actions.device)
    startup=float(config.get('startup_weight',1.));turn=float(config.get('turn_weight',1.))
    if not np.isfinite(startup+turn) or min(startup,turn)<=0:raise ValueError('Positive finite decision weights required')
    weights[(actions==1)|(actions==2)]*=turn
    weights[0]*=startup
    return weights


def geometry_splits(keys, config):
    keys=sorted(set(keys))
    if len(keys)<3:raise ValueError('Need >=3 geometry groups')
    np.random.default_rng(config.get('split_seed',config['seed'])).shuffle(keys);n=max(1,len(keys)//5)
    return {k:'test' if i<n else 'validation' if i<2*n else 'train' for i,k in enumerate(keys)}


def action_metrics(model, rows, previous_mode='teacher'):
    """Offline sequence evaluation, including startup and minority action metrics."""
    if previous_mode not in ('teacher','predicted'):raise ValueError('Unknown previous mode')
    confusion=np.zeros((4,4),dtype=int);first=np.zeros((4,4),dtype=int)
    model.eval()
    with torch.no_grad():
        for r in rows:
            state=None;previous=3;instruction=tokenize(r['instruction'])
            for t,(x,a) in enumerate(zip(r['data']['features'],r['data']['actions'])):
                logits,state,_=model(x,instruction,previous,state);pred=int(logits.argmax());truth=int(a)
                confusion[truth,pred]+=1
                if t==0:first[truth,pred]+=1
                previous=truth if previous_mode=='teacher' else pred
    def summary(c):
        total=int(c.sum())
        return dict(n=total,accuracy=float(c.trace()/total) if total else None,
                    confusion=c.tolist(),recall={a:float(c[i,i]/c[i].sum()) if c[i].sum() else None for i,a in enumerate(ACTIONS)})
    return dict(previous_mode=previous_mode,all_steps=summary(confusion),first_steps=summary(first),
                scope='Recorded teacher images; predicted previous actions are not a simulator rollout')


def action_cam(model,features,instruction,previous,state,action):
    """Grad-CAM at pooled vision features for one action logit, history fixed."""
    x=features.detach().clone().requires_grad_(True)
    logits,_,_=model(x,instruction,previous,None if state is None else state.detach())
    grad=torch.autograd.grad(logits[action],x)[0]
    cam=torch.relu((x.detach()*grad.mean(0)).sum(-1))
    if not torch.isfinite(cam).all():raise RuntimeError('Nonfinite action CAM')
    return cam


def spatial_pool(tokens,grid,merge):
    h,w=grid
    if tokens.shape[0]!=h*w:raise ValueError('Token grid mismatch')
    # Coarse spatial grid retains location without privileged boxes.
    raster=torch.empty(h,w,tokens.shape[1],dtype=tokens.dtype)
    for i,(r,c) in enumerate(token_cells(h,w,merge)):raster[r,c]=tokens[i]
    return torch.nn.functional.adaptive_avg_pool2d(raster.permute(2,0,1)[None],(4,8))[0].permute(1,2,0).reshape(32,-1)


class Policy(torch.nn.Module):
    def __init__(self,width,hidden=64):
        super().__init__();self.hidden=hidden
        self.visual=torch.nn.Sequential(torch.nn.LayerNorm(width),torch.nn.Linear(width,hidden))
        self.position=torch.nn.Parameter(torch.randn(32,hidden)*.01)
        self.words=torch.nn.Embedding(len(VOCAB),hidden)
        self.attention=torch.nn.MultiheadAttention(hidden,4,batch_first=True)
        self.memory=torch.nn.GRUCell(2*hidden+4,hidden)
        self.action=torch.nn.Linear(hidden,4)
    def forward(self,features,instruction,previous,state=None):
        visual=self.visual(features)[None]+self.position[None]
        language=self.words(instruction).mean(0)[None,None]
        attended,weights=self.attention(language,visual,visual)
        prior=torch.nn.functional.one_hot(torch.tensor([previous],device=features.device),4).float()
        state=self.memory(torch.cat([attended[:,0],language[:,0],prior],-1),state)
        return self.action(state)[0],state,weights[0,0]


class Worker:
    def __init__(self,python,output,chunk):
        root=Path(output).resolve();root.parent.mkdir(parents=True,exist_ok=True)
        self.root=root;self.log=open(str(root)+'-worker.log','a')
        env=dict(os.environ);env['JAVA_HOME']=str(Path(python).parent.parent)
        env['PATH']=str(Path(python).parent)+os.pathsep+env.get('PATH','')
        self.buffer=b''
        self.p=subprocess.Popen(['xvfb-run','-a',python,'-m','mc_binding.vla_env','--output',str(root),'--chunk',str(chunk)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.log,bufsize=0,env=env,start_new_session=True)
    def request(self,command,**kw):
        self.p.stdin.write((json.dumps(dict(command=command,**kw))+'\n').encode());self.p.stdin.flush()
        deadline=time.monotonic()+600
        while True:
            while b'\n' not in self.buffer:
                if not select.select([self.p.stdout],[],[],max(0,deadline-time.monotonic()))[0]:raise TimeoutError('MineStudio worker timeout; inspect worker log')
                chunk=os.read(self.p.stdout.fileno(),65536)
                if not chunk:raise RuntimeError('Worker exited; inspect worker log')
                self.buffer+=chunk
            raw,self.buffer=self.buffer.split(b'\n',1);line=raw.decode('utf-8',errors='replace')
            try:r=json.loads(line)
            except json.JSONDecodeError:self.log.write(line);continue
            if not isinstance(r,dict) or 'ok' not in r:continue
            if not r['ok']:raise RuntimeError(r['error'])
            result=r['result']
            if isinstance(result,dict) and 'frame' in result and self.root not in Path(result['frame']).resolve().parents:
                raise RuntimeError('Worker frame outside output root')
            return result
    def close(self):
        import signal
        try:
            if self.p.poll() is None:
                try:self.p.stdin.write(b'{"command":"close"}\n');self.p.stdin.flush()
                except (BrokenPipeError,OSError):pass
        finally:
            try:self.p.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(self.p.pid,signal.SIGTERM)
                try:self.p.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(self.p.pid,signal.SIGKILL);self.p.wait()
            self.log.close()


def plan(groups,n=None):
    jobs=[]
    for fid,rs in list(groups.items())[:n]:
        for r in rs:
            if r['kind'] not in ('full','absent'):continue
            for goal in FEATURES:
                jobs.append(dict(family=fid,record=r,goal=list(goal),instruction=f'Go to the {goal[0]} {goal[1]}.',geometry=geometry_key(rs)))
    return jobs


def collect(dataset,output,render_python,config,reviewed=False,n=1):
    if not reviewed:raise ValueError('Review captures first')
    data,groups=load_binding(dataset)
    if not 1<=n<=len(groups):raise ValueError('Invalid family count')
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    atomic_json(out/'manifest.json',dict(experiment='vla_demos_v1',dataset_hash=digest(data),config=config,source_hash=source_hash(),environment=environment(),teacher='privileged waypoint demonstrator; no RL',policy_inputs=['RGB','instruction','previous_action']))
    atomic_json(out/'status.json',{'state':'running'});worker=None;episodes=[]
    try:
        worker=Worker(render_python,out/'episodes',config['action_chunk'])
        for j,job in enumerate(plan(groups,n)):
            obs=worker.request('reset',record=job['record'],goal=job['goal'],seed=data['seed'],episode=f'e{j:05d}',reference=str((Path(dataset)/job['record']['image']).resolve()))
            samples=[];stopped=False
            for step in range(config['max_decisions']):
                action=worker.request('teacher')['action']
                samples.append(dict(frame=str(Path(obs['frame']).relative_to(out.resolve())),sha256=file_hash(obs['frame']),action=action))
                if action==3:stopped=True;break
                obs=worker.request('step',index=action)
            result=worker.request('finish',stopped=stopped)
            episodes.append(dict(episode=f'e{j:05d}',**job,samples=samples,result=result))
            atomic_json(out/'episodes.json',episodes)
            print(f'e{j:05d} teacher success={result["success"]} samples={len(samples)}',flush=True)
        atomic_json(out/'status.json',{'state':'complete','episodes':len(episodes),'successful':sum(e['result']['success'] for e in episodes)})
    except BaseException as e:
        atomic_json(out/'status.json',{'state':'error','error':str(e)});raise
    finally:
        if worker:worker.close()


def featurize(demos,output,config):
    root,out=Path(demos),Path(output)
    if json.loads((root/'status.json').read_text())['state']!='complete':raise ValueError('Incomplete demonstrations')
    episodes=json.loads((root/'episodes.json').read_text());out.mkdir(parents=True,exist_ok=False)
    atomic_json(out/'status.json',{'state':'running'})
    atomic_json(out/'manifest.json',dict(experiment='vla_features_v1',config=config,demos_manifest=json.loads((root/'manifest.json').read_text()),episodes_hash=file_hash(root/'episodes.json'),source_hash=source_hash(),environment=environment()))
    try:
        enc=Encoder(config);index=[]
        for ep in episodes:
            if not ep['result']['success']:continue
            values=[]
            for s in ep['samples']:
                p=(root/s['frame']).resolve()
                if root.resolve() not in p.parents or file_hash(p)!=s['sha256']:raise ValueError('Frame provenance mismatch')
                with Image.open(p) as im:inputs=enc.inputs(observation_image(im,config))
                _,h,w=map(int,inputs[1][0].tolist())
                values.append(spatial_pool(enc.features(inputs),(h,w),int(enc.visual.spatial_merge_size)))
            p=out/f'{ep["episode"]}.pt';save_tensor(p,dict(features=torch.stack(values),actions=torch.tensor([s['action'] for s in ep['samples']])))
            index.append(dict(episode=ep['episode'],instruction=ep['instruction'],family=ep['family'],geometry=ep['geometry'],file=p.name,sha256=file_hash(p)))
            atomic_json(out/'index.json',index)
            print('Encoded',ep['episode'],flush=True)
        atomic_json(out/'status.json',{'state':'complete','episodes':len(index)})
    except BaseException as e:
        atomic_json(out/'status.json',{'state':'error','error':str(e)});raise


def train(cache,output,config):
    root,out=Path(cache),Path(output)
    if out.exists():raise ValueError('Training output already exists')
    if json.loads((root/'status.json').read_text())['state']!='complete':raise ValueError('Incomplete cache')
    fm=json.loads((root/'manifest.json').read_text())
    if fm['demos_manifest']['config']['action_chunk']!=config['action_chunk']:
        raise ValueError('Training action chunk differs from demonstrations')
    rows=json.loads((root/'index.json').read_text());keys=sorted({r['geometry'] for r in rows})
    if len(keys)<3:raise ValueError('Need >=3 geometry groups; one-family demonstrations are smoke only')
    splits=geometry_splits(keys,config)
    for r in rows:
        p=(root/r['file']).resolve()
        if root.resolve() not in p.parents or file_hash(p)!=r['sha256']:raise ValueError('Feature hash mismatch')
        r['data']=torch.load(p,map_location='cpu',weights_only=True)
    torch.manual_seed(config['seed']);torch.set_num_threads(4)
    width=rows[0]['data']['features'].shape[-1];model=Policy(width,config['hidden'])
    optimizer=torch.optim.AdamW(model.parameters(),lr=config['learning_rate'],weight_decay=.01)
    weighting=config.get('loss_weighting','episode')
    if config.get('observation_mode','raw')!=fm['config'].get('observation_mode','raw'):raise ValueError('Feature preprocessing mismatch')
    train_rows=[r for r in rows if splits[r['geometry']]=='train'];val=[r for r in rows if splits[r['geometry']]=='validation']
    if set(torch.cat([r['data']['actions'] for r in train_rows]).tolist())!=set(range(4)):raise ValueError('Training demonstrations miss action classes')
    mean_length=float(np.mean([len(r['data']['actions']) for r in train_rows]));loss_scale(1,mean_length,weighting)
    best=float('inf');best_state=None;history=[]
    def episode_loss(r):
        state=None;prev=3;losses=[];correct=0;instruction=tokenize(r['instruction'])
        for x,a in zip(r['data']['features'],r['data']['actions']):
            logits,state,_=model(x,instruction,prev,state)
            losses.append(torch.nn.functional.cross_entropy(logits[None],a[None]));correct+=int(logits.argmax()==a);prev=int(a)
        return (torch.stack(losses)*decision_weights(r['data']['actions'],config)).mean(),correct,len(losses)
    for epoch in range(config['epochs']):
        model.train();order=torch.randperm(len(train_rows)).tolist();total=0
        for idx in order:
            optimizer.zero_grad();loss,_,_=episode_loss(train_rows[idx])
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss')
            (loss*loss_scale(len(train_rows[idx]['data']['actions']),mean_length,weighting)).backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();total+=float(loss.detach())
        model.eval()
        with torch.no_grad():
            vals=[(float(episode_loss(r)[0]),len(r['data']['actions'])) for r in val]
            v=sum(a*(n if weighting=='timestep' else 1) for a,n in vals)/sum(n if weighting=='timestep' else 1 for _,n in vals)
        history.append(dict(epoch=epoch,train_loss=total/len(order),validation_loss=v))
        if v<best:
            best=v;best_state={k:v.detach().clone() for k,v in model.state_dict().items()}
        print('Epoch',epoch,'validation loss',v,flush=True)
    if best_state is None:raise RuntimeError('No checkpoint')
    model.load_state_dict(best_state);model.eval();results=[]
    with torch.no_grad():
        for r in rows:
            if config.get('report_test_metrics',True) and splits[r['geometry']]=='test':
                loss,correct,count=episode_loss(r);results.append(dict(episode=r['episode'],correct=correct,n=count,loss=float(loss)))
    out.mkdir(parents=True)
    save_tensor(out/'policy.pt',dict(state=best_state,width=width,hidden=config['hidden']))
    atomic_json(out/'manifest.json',dict(experiment='minecraft_vla_bc_v1',config=config,feature_manifest=fm,feature_index_hash=file_hash(root/'index.json'),splits=splits,vocab=VOCAB,actions=ACTIONS,checkpoint_sha256=file_hash(out/'policy.pt'),scope='Controlled-language behavior cloning. Offline teacher-forced accuracy is not navigation success.'))
    atomic_json(out/'offline-metrics.json',{split:action_metrics(model,[r for r in rows if splits[r['geometry']]==split]) for split in (('validation','test') if config.get('report_test_metrics',True) else ('validation',))});
    atomic_json(out/'history.json',history);atomic_json(out/'offline-test.json',results);atomic_json(out/'status.json',{'state':'complete'})


def rollout(dataset,policy,output,render_python,config,reviewed=False):
    if not reviewed:raise ValueError('Review captures first')
    data,groups=load_binding(dataset);p=Path(policy);pm=json.loads((p/'manifest.json').read_text())
    if digest(data)!=pm['feature_manifest']['demos_manifest']['dataset_hash']:raise ValueError('Dataset/policy mismatch')
    if file_hash(p/'policy.pt')!=pm['checkpoint_sha256']:raise ValueError('Policy hash mismatch')
    encoder_config=pm['feature_manifest']['config']
    if config['action_chunk']!=pm['config']['action_chunk']:raise ValueError('Action-chunk mismatch')
    ck=torch.load(p/'policy.pt',map_location='cpu',weights_only=True)
    model=Policy(ck['width'],ck['hidden']);model.load_state_dict(ck['state']);model.eval()
    evaluation_split=config.get('evaluation_split','test')
    if evaluation_split not in ('validation','test'):raise ValueError('Invalid evaluation split')
    jobs=[j for j in plan(groups) if pm['splits'].get(j['geometry'])==evaluation_split]
    if not jobs:raise ValueError('No held-out jobs')
    out=Path(output);out.mkdir(parents=True,exist_ok=False);worker=None;results=[]
    atomic_json(out/'manifest.json',dict(policy=pm,config=config,source_hash=source_hash(),scope='Closed-loop RGB + instruction policy; no teacher requests or coordinate inputs'))
    atomic_json(out/'status.json',{'state':'running'})
    try:
        enc=Encoder(encoder_config);worker=Worker(render_python,out/'episodes',config['action_chunk'])
        for j,job in enumerate(jobs[:config['evaluation_episodes']]):
            obs=worker.request('reset',record=job['record'],goal=job['goal'],seed=data['seed'],episode=f'e{j:05d}',reference=str((Path(dataset)/job['record']['image']).resolve()))
            state=None;previous=3;decisions=[];stopped=False;instruction=tokenize(job['instruction'])
            maps=out/'maps'/f'e{j:05d}';maps.mkdir(parents=True)
            attention_frames=[];cam_frames=[]
            for step in range(config['max_decisions']):
                started=time.monotonic()
                with Image.open(obs['frame']) as image:raw=image.convert('RGB')
                policy_image=observation_image(raw,encoder_config)
                inputs=enc.inputs(policy_image)
                _,h,w=map(int,inputs[1][0].tolist());features=spatial_pool(enc.features(inputs),(h,w),int(enc.visual.spatial_merge_size))
                prior_state=state;prior_action=previous
                with torch.no_grad():logits,state,attention=model(features,instruction,previous,state)
                action=int(logits.argmax());previous=action
                cam=action_cam(model,features,instruction,prior_action,prior_state,action)
                from .visual_readout import heatmap
                heatmap(policy_image,attention,4,8,1,maps/f'{step:04d}-attention.png')
                heatmap(policy_image,cam,4,8,1,maps/f'{step:04d}-cam.png')
                for path,frames in [(maps/f'{step:04d}-attention.png',attention_frames),(maps/f'{step:04d}-cam.png',cam_frames)]:
                    with Image.open(path) as im:frames.append(im.convert('RGB'))
                decisions.append(dict(frame=obs['frame'],action=ACTIONS[action],probabilities=logits.softmax(0).tolist(),cross_attention=attention.tolist(),action_gradcam=cam.tolist(),inference_seconds=time.monotonic()-started))
                if action==3:stopped=True;break
                obs=worker.request('step',index=action)
            result=worker.request('finish',stopped=stopped)
            for frames,name in [(attention_frames,'policy-attention.gif'),(cam_frames,'action-gradcam.gif')]:
                frames[0].save(maps/name,save_all=True,append_images=frames[1:],duration=config['action_chunk']*50,loop=0)
            atomic_json(out/f'decisions-{j:05d}.json',dict(instruction=job['instruction'],decisions=decisions))
            results.append(dict(episode=f'e{j:05d}',family=job['family'],goal=job['goal'],instruction=job['instruction'],**result))
            atomic_json(out/'results.json',results)
        by_presence=[dict(goal_present=present,n=len(rs),successes=sum(r['success'] for r in rs)) for present in (True,False) for rs in [[r for r in results if r['goal_present']==present]]]
        atomic_json(out/'summary.json',dict(metric_version='navigation_v2',evaluation_split=evaluation_split,families=sorted({r['family'] for r in results}),episodes=len(results),successes=sum(r['success'] for r in results),by_presence=by_presence,scope='Held-out geometry within one arena; not general Minecraft autonomy'))
        links=''.join(f'<li>{r["instruction"]}: success={r["success"]} <a href="episodes/{r["episode"]}/report.html">GIF + trajectory</a> · <a href="maps/{r["episode"]}/policy-attention.gif">Policy cross-attention</a> · <a href="maps/{r["episode"]}/action-gradcam.gif">Action Grad-CAM</a></li>' for r in results)
        (out/'report.html').write_text('<!doctype html><meta charset="utf-8"><h1>Closed-loop VLA pilot</h1><p>Learned actions from new RGB observations and instructions. No privileged coordinates are supplied to the policy. Known state is used only for evaluation. Policy cross-attention is distinct from Grad-CAM for the selected action logit at pooled visual features. Neither is an activation intervention. Maps are separately normalized and GIF timing is simulator time, not measured real-time speed.</p><ul>'+links+'</ul>')
        atomic_json(out/'status.json',{'state':'complete'})
    except BaseException as e:
        atomic_json(out/'status.json',{'state':'error','error':str(e)});raise
    finally:
        if worker:worker.close()
