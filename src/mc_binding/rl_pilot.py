"""Training-only curriculum configuration; simulator metadata never enters policy inputs."""
def training_start(mode, episode, episodes):
    if mode == 'original':
        return None
    if mode != 'near-to-far':
        raise ValueError('Unknown starting-position curriculum')
    # Every fourth episode and the final quarter retain the original start.
    progress = episode / max(episodes, 1)
    if episode % 4 == 3 or progress >= .75:
        return None
    return dict(distance=1.5 if progress < .25 else 3. if progress < .5 else 6.)


def evaluation_summary(rows):
    return dict(episodes=len(rows), successes=sum(bool(r['success']) for r in rows),
                by_presence=[dict(present=p, n=sum(r['goal_present']==p for r in rows),
                    successes=sum(bool(r['success']) and r['goal_present']==p for r in rows))
                    for p in (True, False)])


def checkpoint_schedule(episodes, interval):
    """Compare initialization and fixed checkpoints; no automatic difficulty increase."""
    if episodes < 1 or interval < 1:
        raise ValueError('Positive training budget and interval required')
    schedule=[]
    completed=0
    while True:
        schedule.extend([(f'probe-{completed}-greedy', 0, completed),
                         (f'probe-{completed}-sampled', 0, completed)])
        if completed == episodes:
            break
        count=min(interval,episodes-completed)
        schedule.append(('train',count,completed))
        completed+=count
    return schedule


def distributed_tasks(jobs, count):
    """Spread a development panel across families, retaining absent-goal cases."""
    from collections import defaultdict
    if not 2 <= count <= len(jobs):
        raise ValueError('Development panel needs at least two tasks')
    buckets=defaultdict(list)
    for job in jobs:
        present=any((o['color'],o['type'])==tuple(job['goal']) for o in job['record']['objects'])
        buckets[(present,job['family'])].append(job)
    def ordered(present):
        lists=[v[:] for (p,_),v in sorted(buckets.items()) if p==present]
        result=[]
        while any(lists):
            for bucket in lists:
                if bucket:result.append(bucket.pop(0))
        return result
    absent,present=ordered(False),ordered(True)
    na=min(len(absent),max(1,count//4))
    selected=present[:count-na]+absent[:na]
    if len(selected)!=count:raise ValueError('Insufficient present/absent tasks for panel')
    return selected


def approach_gate(rows, threshold=.8, improvement=.2):
    """Development gate only; never a generalization or binding claim."""
    checkpoints=sorted({r['checkpoint'] for r in rows if r.get('diagnostic_near_start')})
    results=[]
    for checkpoint in checkpoints:
        rates={}
        for mode in ('greedy','sampled'):
            panel=[r for r in rows if r.get('diagnostic_near_start') and r['checkpoint']==checkpoint and r['evaluation_mode']==mode]
            rates[mode]={}
            for present in (True,False):
                subset=[r for r in panel if r['goal_present']==present]
                rates[mode][str(present)]=sum(r['success'] for r in subset)/len(subset) if subset else None
        results.append(dict(checkpoint=checkpoint,rates=rates))
    passed=False
    if len(results)>=3 and results[0]['checkpoint']==0 and all(results[0]['rates'][m]['True'] is not None for m in ('greedy','sampled')):
        baseline=results[0]['rates']
        def acceptable(result):
            return all(result['rates'][m][p] is not None and result['rates'][m][p]>=threshold
                       for m in ('greedy','sampled') for p in ('True','False')) and all(
                       result['rates'][m]['True']-baseline[m]['True']>=improvement for m in ('greedy','sampled'))
        passed=all(acceptable(r) for r in results[-2:])
    return dict(passed=passed,checkpoints=results,criterion='Last two checkpoints: >=80% present and absent success in greedy and sampled development evaluation, and >=20 percentage point present gain over initialization. Requires replication across seeds.',scope='Privileged near-target development diagnostic; no binding or original-start reliability claim')
