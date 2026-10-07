"""Simulator-only reward calculation. Never passed as policy input."""
import math


def potential(distance):
    return 0. if distance is None else -min(float(distance),40.)/20.


def transition_reward(before, after, terminal, success, mode='sparse', gamma=.99):
    if mode not in ('sparse','potential'):raise ValueError('Unknown reward mode')
    if not 0<gamma<=1:raise ValueError('Invalid discount')
    if any(d is not None and (not math.isfinite(d) or d<0) for d in (before,after)):raise ValueError('Invalid distance')
    reward=(1. if success else -1.) if terminal else -.01
    # Terminal potential zero prevents rewarding early stopping from shaped distance.
    if mode=='potential':reward+=gamma*(0. if terminal else potential(after))-potential(before)
    return float(reward)
