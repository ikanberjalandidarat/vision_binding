"""Recurrent trajectory GRPO adaptation: group-relative returns, no critic loss.
No frozen-reference KL; clipped behavior ratios and KL early stop (beta_ref=0).
"""
import torch
from .recurrent_ppo import replay,clipped_loss


def group_advantages(returns):
    r=torch.as_tensor(returns,dtype=torch.float32)
    if r.ndim!=1 or len(r)<2 or not torch.isfinite(r).all():raise ValueError('Invalid GRPO group')
    return (r-r.mean())/r.std(unbiased=False).clamp_min(1e-8)


def update(model,optimizer,episodes,generator,epochs=4,target_kl=.02):
    if len(episodes)<2:raise ValueError('GRPO needs at least two episodes from same task/start')
    returns=[sum(.99**t*r for t,r in enumerate(e['rewards'])) for e in episodes]
    advantages=group_advantages(returns)
    with torch.no_grad():error=max(float((replay(model,e)[0]-e['old_logps']).abs().max()) for e in episodes)
    if error>1e-4:raise RuntimeError('GRPO on-policy replay mismatch')
    logs=[]
    for epoch in range(epochs):
        optimizer.zero_grad();stats=[]
        # Precheck KL on the whole group before applying any update.
        with torch.no_grad():
            kl=sum(float(((lp-e['old_logps']).exp()-1-(lp-e['old_logps'])).mean()) for e in episodes for lp,_,_ in [replay(model,e)])/len(episodes)
        if kl>target_kl:logs.append(dict(epoch=epoch,approx_kl=kl,skipped_for_kl=True));break
        for i,e in enumerate(episodes):
            lp,v,entropy=replay(model,e)
            loss,info=clipped_loss(lp,e['old_logps'],torch.full_like(lp,float(advantages[i])),v,torch.zeros_like(v),entropy,value_coef=0.)
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite GRPO loss')
            # Equal episode weight: long rollouts must not count as extra group members.
            (loss/len(episodes)).backward();stats.append(info)
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),.5)
        if not torch.isfinite(norm):raise RuntimeError('Nonfinite GRPO gradient')
        optimizer.step()
        logs.append(dict(epoch=epoch,approx_kl=kl,gradient_norm=float(norm),replay_error=error,group_returns=returns,group_advantages=advantages.tolist(),zero_variance=bool(torch.as_tensor(returns).std(unbiased=False)<1e-8),clip_fraction=sum(s['clip_fraction'] for s in stats)/len(stats),actor=sum(s['actor'] for s in stats)/len(stats),entropy=sum(s['entropy'] for s in stats)/len(stats),skipped_for_kl=False))
    return logs
