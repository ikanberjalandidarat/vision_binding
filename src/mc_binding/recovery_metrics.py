"""Small JSON-only learning diagnostics; scores do not certify general autonomy."""
def panel(rows):
    result=[]
    for mode in ('greedy','sampled'):
        for present in (True,False):
            r=[x for x in rows if x['action_mode']==mode and x['goal_present']==present]
            result.append(dict(action_mode=mode,present=present,n=len(r),successes=sum(x['success'] for x in r),
                immediate_stops=sum(x['stop'] and x['steps']==1 for x in r),timeouts=sum(x['timeout'] for x in r),
                mean_steps=sum(x['steps'] for x in r)/len(r) if r else None))
    return result


def review(initial,final,history,probes,budget):
    transitions=sum(x['steps'] for x in history)
    return dict(budget_target=budget,transitions=transitions,budget_met=transitions>=budget if budget else None,
        episodes=len(history),movement_and_wait_decisions=transitions-sum(x['stop'] for x in history),
        recovered_stop_decisions=sum(x.get('recovered_stops',0) for x in history),
        training_present_successes=sum(x['success'] and x['goal_present'] for x in history),
        initial_original_start=panel(initial),final_original_start=panel(final),
        near_probes=[dict(checkpoint=p['checkpoint'],split=p['split'],scores=panel(p['rows'])) for p in probes],
        interpretation='Near-start probes are privileged diagnostics, not original-start navigation. Training-only stop recovery changes the MDP. Compare unrestricted evaluation; do not rank arms with unmet budgets. Validation is development data, not a final test.')
