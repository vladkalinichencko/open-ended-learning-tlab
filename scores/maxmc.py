"""Upstream-score ACCEL: max Monte-Carlo return как прокси regret."""

from jaxued.utils import max_mc

def score(config, dones, values, max_returns, advantages, rewards=None, prior_p=None):
    return max_mc(dones, values, max_returns)
