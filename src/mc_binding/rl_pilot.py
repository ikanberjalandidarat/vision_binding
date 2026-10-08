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
