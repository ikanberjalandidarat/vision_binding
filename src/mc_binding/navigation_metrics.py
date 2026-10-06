"""Versioned evaluator-only navigation metrics; no state supplied to policies."""
import math


def score_episode(trace, goal, stopped):
    if not trace:raise ValueError('Empty trajectory')
    points=[(float(t['pose']['x']),float(t['pose']['z'])) for t in trace]
    if not all(math.isfinite(v) for p in points for v in p):raise ValueError('Nonfinite trajectory')
    travel=sum(math.dist(a,b) for a,b in zip(points,points[1:]))
    max_displacement=max(math.dist(points[0],p) for p in points)
    distance=None if goal is None else math.dist(points[-1],goal)
    minimum=None if goal is None else min(math.dist(p,goal) for p in points)
    # This arena task prescribes abstaining before any action when the target is absent.
    immediate=bool(stopped and len(trace)==1)
    success=bool(stopped and (immediate if goal is None else distance<=.8))
    return dict(metric_version='navigation_v2',success=success,distance=distance,
                minimum_sampled_goal_distance=minimum,travel_distance=travel,max_displacement=max_displacement,
                immediate_stop=immediate,absent_immediate_refusal=bool(goal is None and immediate),
                timeout=not stopped,legacy_success=bool(stopped and (goal is None or distance<=.8)))
