"""learnability * max(PVL, 0): уровень на границе и с сигналом обучения."""

import jax.numpy as jnp
from jaxued.utils import positive_value_loss

from teacher_stats import success_rate

def score(config, dones, values, max_returns, advantages, rewards=None, prior_p=None):
    p = success_rate(dones, rewards)
    if prior_p is not None:
        p = jnp.where(prior_p < 0, p,
                      (1 - config["p_decay"]) * prior_p + config["p_decay"] * p)
    result = p * (1 - p) * jnp.maximum(positive_value_loss(dones, advantages), 0.0)
    return jnp.where(jnp.ones_like(p) > 0, result, -jnp.inf)
