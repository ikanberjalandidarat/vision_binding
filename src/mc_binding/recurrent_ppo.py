"""Full-episode recurrent PPO. Sequence order is preserved inside minibatches."""
import torch


def gae(rewards,values,terminated,bootstrap=0.,gamma=.99,lam=.95):
    """True terminations zero bootstrap; a nonterminal buffer end uses bootstrap."""
    rewards=torch.as_tensor(rewards,dtype=torch.float32)
    values=torch.as_tensor(values,dtype=torch.float32)
    terminated=torch.as_tensor(terminated,dtype=torch.bool)
    if not (rewards.ndim==1 and rewards.shape==values.shape==terminated.shape) or not len(rewards):raise ValueError('Invalid trajectory')
    if not torch.isfinite(rewards).all() or not torch.isfinite(values).all():raise ValueError('Nonfinite rollout')
    if not 0<=lam<=1 or not 0<gamma<=1 or not torch.isfinite(torch.tensor(bootstrap)):raise ValueError('Invalid GAE setting')
    advantage=torch.zeros_like(rewards);carry=0.;next_value=float(bootstrap)
    for t in reversed(range(len(rewards))):
        continuation=float(not terminated[t])
        delta=rewards[t]+gamma*next_value*continuation-values[t]
        carry=delta+gamma*lam*continuation*carry
        advantage[t]=carry;next_value=values[t]
    return advantage,advantage+values


def replay(model,episode):
    state=None;logps=[];values=[];entropy=[]
    for t,x in enumerate(episode['features']):
        logits,state,_=model(x,episode['instruction'],episode['previous'][t],state,binding=episode['signals'][t])
        distribution=torch.distributions.Categorical(logits=logits)
        logps.append(distribution.log_prob(episode['actions'][t]))
        values.append(model.value(state).reshape(()));entropy.append(distribution.entropy())
    return torch.stack(logps),torch.stack(values),torch.stack(entropy)


def clipped_loss(logps,old_logps,advantages,values,returns,entropy,clip=.2,value_coef=.5,entropy_coef=.01):
    logratio=logps-old_logps;ratio=logratio.exp()
    actor=-torch.minimum(ratio*advantages,ratio.clamp(1-clip,1+clip)*advantages).mean()
    critic=(values-returns).square().mean()
    loss=actor+value_coef*critic-entropy_coef*entropy.mean()
    stats=dict(actor=float(actor.detach()),critic=float(critic.detach()),entropy=float(entropy.mean().detach()),
               approx_kl=float(((ratio-1)-logratio).mean().detach()),clip_fraction=float(((ratio-1).abs()>clip).float().mean().detach()))
    return loss,stats


def update(model,optimizer,episodes,generator,epochs=4,minibatch_episodes=2,target_kl=.02):
    if not episodes or epochs<1 or minibatch_episodes<1:raise ValueError('Empty PPO batch')
    # Collection uses a fixed policy for the entire batch; verify exact recurrent replay.
    with torch.no_grad():
        error=max(float((replay(model,e)[0]-e['old_logps']).abs().max()) for e in episodes)
    if error>1e-4:raise RuntimeError(f'On-policy replay mismatch: {error}')
    all_adv=torch.cat([e['advantages'] for e in episodes]);mean=all_adv.mean();std=all_adv.std(unbiased=False).clamp_min(1e-8)
    logs=[];stopped=False
    for epoch in range(epochs):
        order=torch.randperm(len(episodes),generator=generator).tolist()
        for start in range(0,len(order),minibatch_episodes):
            selected=[episodes[i] for i in order[start:start+minibatch_episodes]]
            forward=[replay(model,e) for e in selected]
            logps,values,entropy=[torch.cat([p[k] for p in forward]) for k in range(3)]
            old=torch.cat([e['old_logps'] for e in selected]);ret=torch.cat([e['returns'] for e in selected])
            adv=(torch.cat([e['advantages'] for e in selected])-mean)/std
            loss,stats=clipped_loss(logps,old,adv,values,ret,entropy)
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite PPO loss')
            stats.update(epoch=epoch,replay_error=error,steps=len(logps),loss=float(loss.detach()))
            if stats['approx_kl']>target_kl:
                stats['skipped_for_kl']=True;logs.append(stats);stopped=True;break
            optimizer.zero_grad();loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),.5)
            if not torch.isfinite(norm):raise RuntimeError('Nonfinite PPO gradient')
            optimizer.step();stats.update(gradient_norm=float(norm),skipped_for_kl=False);logs.append(stats)
        if stopped:break
    return logs
