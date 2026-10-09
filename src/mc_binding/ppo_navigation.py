"""Live recurrent PPO with frozen Qwen and predicted binding-map ablations."""
import argparse
import json
from pathlib import Path
import torch
from PIL import Image
from .binding_data import load_binding
from .binding_policy import target_signal
from .live_visual_control import proposals
from .perception_study import rasterize,features_for
from .visual_readout import Encoder,Readout,save_tensor
from .vla import Worker,plan,tokenize
from .rl_navigation import initialize_agent
from .rl_pilot import training_start,distributed_tasks
from .recurrent_ppo import gae,update
from .io import atomic_json,file_hash,digest,source_hash


def run(a):
    a.method=getattr(a,'method','ppo');a.intrinsic_coef=getattr(a,'intrinsic_coef',.001)
    if a.method not in ('ppo','grpo','ppo-icm','ppo-rnd') or not 0<=a.intrinsic_coef<=1:raise ValueError('Invalid method or bonus scale')
    if a.method=='grpo' and (a.batch_episodes<2 or a.episodes%a.batch_episodes):raise ValueError('GRPO requires complete groups of >=2 episodes')
    if min(a.episodes,a.batch_episodes,a.max_steps,a.eval_episodes,a.epochs)<1:raise ValueError('Positive budgets required')
    torch.set_num_threads(4)
    data,groups=load_binding(a.dataset);jobs=plan(groups)
    ckpath=Path(a.checkpoint);meta=json.loads((ckpath.parent/'manifest.json').read_text());splits=meta['splits']
    if any(j['geometry'] not in splits for j in jobs):raise ValueError('Unknown selector split')
    training=[j for j in jobs if splits[j['geometry']]=='train']
    validation=[j for j in jobs if splits[j['geometry']]=='validation']
    if not training or a.eval_episodes>len(validation):raise ValueError('Insufficient tasks')
    validation=distributed_tasks(validation,a.eval_episodes)
    config=json.loads(Path(a.config).read_text())
    if config.get('readout_layer')!=31:raise ValueError('Readout requires block 31')
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False);worker=None
    atomic_json(out/'status.json',dict(state='running'))
    atomic_json(out/'manifest.json',dict(algorithm=a.method,variant='recurrent_full_episode_v2',intrinsic_spec='ICM or RND on mean pooled frozen Qwen features; RMS scaled, clipped at 5, training only. Combined episodic value target; not a paper reproduction.',grpo_spec='Same task/start per batch, group-standardized discounted task returns; equal trajectory weight, no critic loss or reference KL, entropy .01. Not token GRPO.',settings=vars(a),config=config,splits=splits,dataset_hash=digest(data),selector_hash=file_hash(ckpath),source_hash=source_hash(),
        gamma=.99,gae_lambda=.95,clip=.2,learning_rate=3e-4,target_kl=.02,entropy_coef=.01,value_coef=.5,gradient_clip=.5,minibatch_episodes=2,
        policy_inputs=['RGB features','instruction','predicted map + match score + no-proposal flag','previous action','GRU state'],
        selector_provenance='Frozen supervised simulator-label readout; no new manual labels, not end-to-end self-supervision.',
        episode_end='Stop or fixed task deadline is true terminal with zero bootstrap and terminal task reward. Batches contain whole episodes; collection never cuts an episode. No time-limit bootstrapping for this finite-horizon task.',
        recurrent_update='Replay full stored observation/action sequences from zero hidden state for each optimization pass. Shuffle episodes, never timesteps.',
        budget='Matched episode caps; actual transitions reported and can differ with stopping. Not a fixed-transition-budget comparison.',
        control='All modes compute image proposals and frozen readout. Zero drops the signal; shuffled independently permutes map locations each frame.',
        validation_tasks=[dict(family=j['family'],record_id=j['record']['record_id'],goal=j['goal']) for j in validation],test_evaluated=False))
    try:
        enc=Encoder(config);frozen=torch.load(ckpath,map_location='cpu',weights_only=True)
        readout=Readout(frozen['width']);readout.load_state_dict(frozen['state']);readout.eval()
        for p in readout.parameters():p.requires_grad_(False)
        model=initialize_agent(frozen['width'],a.seed,binding=True)
        if a.method=='grpo':
            for p in model.value.parameters():p.requires_grad_(False)
        optimizer=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=3e-4)
        curiosity=None
        if a.method in ('ppo-icm','ppo-rnd'):
            from .intrinsic import Curiosity
            curiosity=Curiosity(a.method.split('-')[1],frozen['width'],a.seed+8001)
        scene_rng=torch.Generator().manual_seed(a.seed+1001);action_rng=torch.Generator().manual_seed(a.seed+2001)
        update_rng=torch.Generator().manual_seed(a.seed+3001);start_rng=torch.Generator().manual_seed(a.seed+4001)
        worker=Worker(a.render_python,out/'episodes',2)
        def features(path,goal,map_rng):
            with Image.open(path) as im:image=im.convert('RGB')
            inp=enc.inputs(image);_,h,w=map(int,inp[1][0].tolist())
            raster=rasterize(enc.features(inp),h,w,int(enc.visual.spatial_merge_size))
            x=torch.nn.functional.adaptive_avg_pool2d(raster.permute(2,0,1)[None],(8,16))[0].permute(1,2,0).reshape(128,-1)
            regions=proposals(image)
            with torch.no_grad():scores=readout.goal_scores(features_for(raster,regions,image.size,'8x16'),goal) if regions else torch.empty(0)
            return x.detach(),target_signal(regions,scores,image.size,a.mode,map_rng).detach()
        def episode(job,eid,rng,sampled,start=None,map_seed=0):
            obs=worker.request('reset',record=job['record'],goal=job['goal'],seed=data['seed'],episode=eid,reference=str((Path(a.dataset)/job['record']['image']).resolve()),training_start=start)
            state=None;previous=3;map_rng=torch.Generator().manual_seed(map_seed)
            trajectory=dict(features=[],signals=[],previous=[],actions=[],old_logps=[],values=[],rewards=[],terminated=[],instruction=tokenize(job['instruction']))
            decisions=[];model.eval();prefetched=None;intrinsic_transitions=[];extrinsic_total=intrinsic_total=0.
            for step in range(a.max_steps):
                x,signal=prefetched if prefetched is not None else features(obs['frame'],job['goal'],map_rng)
                prefetched=None
                with torch.no_grad():
                    logits,state,_=model(x,trajectory['instruction'],previous,state,binding=signal)
                    dist=torch.distributions.Categorical(logits=logits)
                    action=int(torch.multinomial(dist.probs,1,generator=rng)) if sampled else int(logits.argmax())
                    value=float(model.value(state).reshape(()));logp=float(dist.log_prob(torch.tensor(action)))
                response=worker.request('rl_step',index=action,last=step==a.max_steps-1,mode='potential',gamma=.99)
                extrinsic=float(response['reward']);bonus=raw_bonus=0.
                if curiosity is not None and eid.startswith('train-') and action!=3:
                    next_x,next_signal=features(response['observation']['frame'],job['goal'],map_rng)
                    raw_bonus,bonus=curiosity.bonus(x.mean(0),action,next_x.mean(0))
                    intrinsic_transitions.append((x.mean(0),action,next_x.mean(0),raw_bonus))
                    prefetched=(next_x,next_signal)
                response['reward']=extrinsic+a.intrinsic_coef*bonus
                extrinsic_total+=extrinsic;intrinsic_total+=a.intrinsic_coef*bonus
                for key,item in [('features',x),('signals',signal),('previous',previous),('actions',action),('old_logps',logp),('values',value),('rewards',response['reward']),('terminated',response['terminal'])]:trajectory[key].append(item)
                decisions.append(dict(step=step,frame=obs['frame'],action=action,probabilities=dist.probs.tolist(),old_logp=logp,value=value,binding_signal=signal.tolist(),reward=response['reward'],extrinsic_reward=extrinsic,intrinsic_raw=raw_bonus,intrinsic_bonus=a.intrinsic_coef*bonus,terminal=response['terminal']))
                if response['terminal']:break
                previous=action;obs=response['observation']
            if not trajectory['terminated'][-1]:raise RuntimeError('Worker failed task deadline terminal contract')
            trajectory['actions']=torch.tensor(trajectory['actions']);trajectory['old_logps']=torch.tensor(trajectory['old_logps'])
            trajectory['advantages'],trajectory['returns']=gae(trajectory['rewards'],trajectory['values'],trajectory['terminated'])
            summary=dict(episode=eid,family=job['family'],goal=job['goal'],record_id=job['record']['record_id'],success=response['success'],goal_present=any((o['color'],o['type'])==tuple(job['goal']) for o in job['record']['objects']),steps=len(decisions),return_sum=sum(trajectory['rewards']),extrinsic_return=extrinsic_total,intrinsic_return=intrinsic_total,stop=action==3,timeout=action!=3,training_start=start)
            atomic_json(out/(eid+'-decisions.json'),decisions)
            trajectory['intrinsic_transitions']=intrinsic_transitions
            return trajectory,summary
        def evaluate(label):
            rows=[]
            for sampled in (False,True):
                for i,job in enumerate(validation):
                    rng=torch.Generator().manual_seed(a.seed+5001+i)
                    _,row=episode(job,f'{label}-'+('sampled' if sampled else 'greedy')+f'-{i:04d}',rng,sampled,map_seed=a.seed+6001+i)
                    row['action_mode']='sampled' if sampled else 'greedy';rows.append(row)
                    atomic_json(out/(label+'.json'),rows)
            return rows
        save_tensor(out/'initial.pt',dict(state=model.state_dict(),width=frozen['width'],hidden=64,architecture='BindingAgent',mode=a.mode))
        initial=evaluate('initial');history=[];updates=[];transitions=0
        for batch_start in range(0,a.episodes,a.batch_episodes):
            batch=[];group_job=None;group_start=None
            for i in range(batch_start,min(a.episodes,batch_start+a.batch_episodes)):
                # Every batch uses one task/start across all methods, for matched group scheduling.
                if group_job is None:
                    group_job=training[int(torch.randint(len(training),(1,),generator=scene_rng))]
                    group_start=training_start(a.start_mode,batch_start,a.episodes)
                    if group_start is not None:group_start=dict(group_start,anchor_index=int(torch.randint(len(group_job['record']['objects']),(1,),generator=start_rng)))
                job=group_job;start=group_start
                trajectory,row=episode(job,f'train-{i:05d}',action_rng,True,start,a.seed+7001+batch_start)
                batch.append(trajectory);history.append(row);transitions+=row['steps'];atomic_json(out/'training.json',history)
            model.train()
            if a.method=='grpo':
                from .grpo import update as group_update
                stats=group_update(model,optimizer,batch,update_rng,epochs=a.epochs)
            else:stats=update(model,optimizer,batch,update_rng,epochs=a.epochs)
            intrinsic_stats=curiosity.train_batch([t for e in batch for t in e['intrinsic_transitions']]) if curiosity else None
            updates.append(dict(episodes_completed=len(history),transitions=transitions,minibatches=stats,intrinsic=intrinsic_stats));atomic_json(out/'updates.json',updates)
            save_tensor(out/'agent.pt',dict(state=model.state_dict(),optimizer=optimizer.state_dict(),width=frozen['width'],hidden=64,architecture='BindingAgent',mode=a.mode,method=a.method,curiosity=curiosity.state_dict() if curiosity else None,episodes_completed=len(history),transitions=transitions))
            print(a.method+' update',len(updates),'episodes',len(history),'transitions',transitions,flush=True)
        final=evaluate('final')
        summary=[]
        for label,rows in [('initial',initial),('final',final)]:
            for mode in ('greedy','sampled'):
                for present in (True,False):
                    rr=[r for r in rows if r['action_mode']==mode and r['goal_present']==present]
                    summary.append(dict(phase=label,action_mode=mode,present=present,n=len(rr),successes=sum(r['success'] for r in rr),timeouts=sum(r['timeout'] for r in rr)))
        atomic_json(out/'summary.json',dict(results=summary,training_transitions=transitions,training_episodes=len(history)))
        body='<h1>Live recurrent '+a.method+'</h1><p>Actual learned-policy movement, not teacher replay. Binding readout and Qwen frozen. Completion does not imply learning.</p><pre>'+json.dumps(summary,indent=2)+'</pre>'
        for row in final:
            path='episodes/'+row['episode']+'/movement.gif'
            body+=f'<h2>{row["episode"]}: {row["goal"]}, success {row["success"]}</h2><img loading="lazy" src="{path}"><p>{out/path}</p><a href="episodes/{row["episode"]}/report.html">Trajectory and evaluator outcome</a>'
        (out/'report.html').write_text('<!doctype html><meta charset="utf-8"><style>body{font:17px system-ui;max-width:1000px;margin:auto}img{max-width:100%}</style>'+body)
        atomic_json(out/'status.json',dict(state='complete'))
    except BaseException as e:atomic_json(out/'status.json',dict(state='error',error=str(e)));raise
    finally:
        if worker:worker.close()

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('dataset','checkpoint','render-python','output'):p.add_argument('--'+name,required=True)
    p.add_argument('--config',default='configs/visual_readout.json');p.add_argument('--mode',choices=['zero','binding','shuffled'],default='binding');p.add_argument('--seed',type=int,default=731)
    p.add_argument('--episodes',type=int,default=96);p.add_argument('--batch-episodes',type=int,default=4);p.add_argument('--max-steps',type=int,default=96);p.add_argument('--eval-episodes',type=int,default=64);p.add_argument('--epochs',type=int,default=4);p.add_argument('--start-mode',choices=['original','near-to-far'],default='near-to-far')
    p.add_argument('--method',choices=['ppo','grpo','ppo-icm','ppo-rnd'],default='ppo');p.add_argument('--intrinsic-coef',type=float,default=.001)
    run(p.parse_args())
