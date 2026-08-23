"""Доля решённых эпизодов на уровне: общая для teacher и score-функций."""

import jax.numpy as jnp
from jaxued.utils import accumulate_rollout_stats


def success_rate(dones, rewards):
    """Доля эпизодов на уровне, закончившихся достижением цели."""
    solved, _, episodes = accumulate_rollout_stats(dones, (rewards > 0).astype(jnp.float32),
                                                   time_average=False)
    return jnp.where(episodes > 0, solved, 0.0)
