"""On-policy episodic actor-critic with optional self-supervised next-feature prediction.
Frozen vision encoder; no teacher requests, boxes, coordinates or reward in policy inputs.
"""
import argparse
import json
from pathlib import Path
import torch
from PIL import Image
from .rl_pilot import training_start, evaluation_summary
from .binding_data import load_binding
from .vla import Policy, Worker, plan, tokenize, geometry_splits
from .visual_readout import Encoder, save_tensor
from .perception_study import rasterize
from .io import atomic_json, digest, source_hash


class Agent(Policy):
    def __init__(self,width,hidden=64):
        super().__init__(width,hidden)
        self.position=torch.nn.Parameter(torch.randn(128,hidden)*.01)
        self.value=torch.nn.Linear(hidden,1)
        self.predict_next=torch.nn.Linear(hidden+4,width)


def initialize_agent(width, seed):
    # Encoder construction may reset the global RNG. Policy initialization is isolated.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return Agent(width)


def rollout_generators(seed):
    # Scene scheduling and exploration must not depend on encoder/global RNG calls.
    return (torch.Generator().manual_seed(seed),
            torch.Generator().manual_seed(seed+1000003))


def returns(rewards,gamma):
    total=0.;values=[]
    for r in reversed(rewards):total=r+gamma*total;values.append(total)
    return torch.tensor(list(reversed(values)),dtype=torch.float32)


def objective(logps,values,entropies,rewards,gamma,auxiliary,weight):
    target=returns(rewards,gamma);v=torch.stack(values).reshape(-1)
    advantage=target-v.detach()
    actor=-(torch.stack(logps)*advantage).mean()
    critic=.5*(v-target).square().mean()
    entropy=torch.stack(entropies).mean()
    aux=torch.stack(auxiliary).mean() if auxiliary else v.sum()*0
    return actor+critic-.01*entropy+weight*aux


def run(dataset,output,render,config,seed,mode,aux_weight,episodes,max_steps,eval_episodes,start_mode="original",paired_evaluation=False):
    torch.set_num_threads(4);scene_rng,action_rng=rollout_generators(seed)
    evaluation_rng=torch.Generator().manual_seed(seed+2000003)
    start_rng=torch.Generator().manual_seed(seed+3000003)
    training_start(start_mode,0,episodes)
    data,groups=load_binding(dataset);jobs=plan(groups)
    splits=geometry_splits([j['geometry'] for j in jobs],dict(seed=731,split_seed=731))
    training=[j for j in jobs if splits[j['geometry']]=='train'];validation=[j for j in jobs if splits[j['geometry']]=='validation']
    if aux_weight<0 or episodes<1 or max_steps<1 or not 1<=eval_episodes<=len(validation):raise ValueError('Invalid run size')
    out=Path(output);out.mkdir(parents=True,exist_ok=False);worker=None
    atomic_json(out/'status.json',dict(state='running'))
    atomic_json(out/'manifest.json',dict(algorithm='episodic_on_policy_actor_critic',seed=seed,split_seed=731,splits=splits,reward=mode,aux_weight=aux_weight,episodes=episodes,max_steps=max_steps,eval_episodes=eval_episodes,gamma=.99,config=config,dataset_hash=digest(data),source_hash=source_hash(),rng_version='isolated_policy_scene_action_v2',start_mode=start_mode,paired_evaluation=paired_evaluation,action_order=['forward','turn_left','turn_right','stop'],curriculum='Training only; goal-relative spawn uses privileged state. Absent goals use a random object anchor. Original-start evaluation.',policy_inputs=['RGB frozen 8x16 features','instruction','previous action','recurrent memory'],privileged_reward=True,teacher_actions=False,test_evaluated=False,scope='Controlled arena. Scratch policy, frozen encoder. Auxiliary target is next observation feature, not object labels. Not general Minecraft autonomy.'))
    try:
        enc=Encoder(config);worker=Worker(render,out/'episodes',2)
        def feature(path):
            with Image.open(path) as im:inp=enc.inputs(im.convert('RGB'))
            _,h,w=map(int,inp[1][0].tolist());r=rasterize(enc.features(inp),h,w,int(enc.visual.spatial_merge_size))
            return torch.nn.functional.adaptive_avg_pool2d(r.permute(2,0,1)[None],(8,16))[0].permute(1,2,0).reshape(128,-1)
        model=None;opt=None;history=[];evaluation=[]
        phases=[('train',episodes),('validation',eval_episodes)]
        if paired_evaluation:phases.append(('validation-sampled',eval_episodes))
        for phase,n in phases:
            for episode in range(n):
                job=training[int(torch.randint(len(training),(1,),generator=scene_rng))] if phase=='train' else validation[episode]
                eid=f'{phase}-{episode:05d}'
                start=training_start(start_mode,episode,episodes) if phase=='train' else None
                if start is not None:
                    start=dict(start,anchor_index=int(torch.randint(len(job['record']['objects']),(1,),generator=start_rng)))
                obs=worker.request('reset',training_start=start,record=job['record'],goal=job['goal'],seed=data['seed'],episode=eid,reference=str((Path(dataset)/job['record']['image']).resolve()))
                x=feature(obs['frame'])
                if model is None:
                    model=initialize_agent(x.shape[-1],seed);opt=torch.optim.AdamW(model.parameters(),lr=3e-4)
                model.train(phase=='train');state=None;previous=3
                logps=[];values=[];entropies=[];rewards=[];auxiliary=[];actions=[];decisions=[]
                for step in range(max_steps):
                    with torch.set_grad_enabled(phase=='train'):
                        logits,state,_=model(x,tokenize(job['instruction']),previous,state)
                        distribution=torch.distributions.Categorical(logits=logits)
                        action=int(torch.multinomial(distribution.probs,1,generator=action_rng)) if phase=='train' else int(torch.multinomial(distribution.probs,1,generator=evaluation_rng)) if phase=='validation-sampled' else int(logits.argmax())
                        if phase=='train':
                            logps.append(distribution.log_prob(torch.tensor(action)));values.append(model.value(state).squeeze());entropies.append(distribution.entropy())
                    decisions.append(dict(step=step,action=action,probabilities=distribution.probs.detach().tolist(),entropy=float(distribution.entropy().detach())))
                    response=worker.request('rl_step',index=action,last=step==max_steps-1,mode=mode,gamma=.99)
                    rewards.append(response['reward']);actions.append(action)
                    if response['terminal']:break
                    next_x=feature(response['observation']['frame'])
                    if phase=='train' and aux_weight:
                        onehot=torch.nn.functional.one_hot(torch.tensor([action]),4).float()
                        prediction=model.predict_next(torch.cat([state,onehot],-1)).squeeze(0)
                        target=torch.nn.functional.normalize(next_x.mean(0),dim=0).detach()
                        auxiliary.append((torch.nn.functional.normalize(prediction,dim=0)-target).square().sum())
                    x=next_x;previous=action
                row=dict(episode=eid,evaluation_mode="sampled" if phase=="validation-sampled" else "greedy" if phase=="validation" else "training",training_start=start,decisions=decisions,family=job['family'],instruction=job['instruction'],goal=job['goal'],goal_present=any((o['color'],o['type'])==tuple(job['goal']) for o in job['record']['objects']),success=response['success'],steps=len(actions),actions=actions,rewards=rewards,total_reward=sum(rewards))
                if phase=='train':
                    loss=objective(logps,values,entropies,rewards,.99,auxiliary,aux_weight)
                    if not torch.isfinite(loss):raise RuntimeError('Nonfinite RL loss')
                    opt.zero_grad();loss.backward();gradient_norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();row['gradient_norm_before_clip']=float(gradient_norm)
                    row['loss']=float(loss.detach());history.append(row);atomic_json(out/'training.json',history)
                    save_tensor(out/'agent.pt',dict(state=model.state_dict(),width=x.shape[-1],hidden=64,episodes_completed=episode+1))
                else:evaluation.append(row);atomic_json(out/'validation.json',evaluation)
                print(phase,episode,'success',row['success'],'reward',row['total_reward'],flush=True)
        atomic_json(out/'summary.json',dict(validation_episodes=len(evaluation),successes=sum(r['success'] for r in evaluation),by_presence=[dict(present=p,n=sum(r['goal_present']==p for r in evaluation),successes=sum(r['success'] and r['goal_present']==p for r in evaluation)) for p in (True,False)],by_action_selection={m:evaluation_summary([r for r in evaluation if r['evaluation_mode']==m]) for m in ('greedy','sampled')},note='Development evaluation on fixed tasks; inspect manifests and trajectories. Completion does not imply learning.'))
        atomic_json(out/'status.json',dict(state='complete'))
    except BaseException as e:atomic_json(out/'status.json',dict(state='error',error=str(e)));raise
    finally:
        if worker:worker.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--output',required=True);p.add_argument('--render-python',required=True);p.add_argument('--config',default='configs/visual_readout.json');p.add_argument('--seed',type=int,default=731);p.add_argument('--reward',choices=['sparse','potential'],default='sparse');p.add_argument('--aux-weight',type=float,default=0);p.add_argument('--episodes',type=int,default=96);p.add_argument('--max-steps',type=int,default=96);p.add_argument('--eval-episodes',type=int,default=64);p.add_argument('--start-mode',choices=['original','near-to-far'],default='original');p.add_argument('--paired-evaluation',action='store_true');a=p.parse_args()
    run(a.dataset,a.output,a.render_python,json.loads(Path(a.config).read_text()),a.seed,a.reward,a.aux_weight,a.episodes,a.max_steps,a.eval_episodes,a.start_mode,a.paired_evaluation)
