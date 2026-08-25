"""Upstream-score PLR: положительная ошибка предсказания value-функции."""

from jaxued.utils import positive_value_loss

def score(config, dones, values, max_returns, advantages, rewards=None, prior_p=None):
    return positive_value_loss(dones, advantages)
