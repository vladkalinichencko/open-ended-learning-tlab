"""p(1-p) по доле решённых эпизодов: максимум на границе способностей студента.

Максимум на p=0.5, то есть не на самых сложных уровнях, а на тех, где студенту есть
чему учиться. Это прямой ответ на то, чем плохи прокси regret'а: MaxMC коррелирует с
длиной пути на -0.45, то есть меряет сложность (см. NOTES).
"""

import jax.numpy as jnp

from teacher_stats import success_rate

def score(config, dones, values, max_returns, advantages, rewards=None, prior_p=None):
    p = success_rate(dones, rewards)
    if prior_p is not None:
        # p с одного захода почти всегда 0 или 1, а p(1-p) на двоичной величине
        # тождественно ноль — измерено, см. NOTES. Копим долю по всем визитам.
        p = jnp.where(prior_p < 0, p,
                      (1 - config["p_decay"]) * prior_p + config["p_decay"] * p)
    result = p * (1 - p)
    return jnp.where(jnp.ones_like(p) > 0, result, -jnp.inf)
