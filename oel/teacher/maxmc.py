import jax.numpy as jnp

from jaxued.utils import max_mc

from oel.methods import create_transition_predictor, update_transition_predictor

def setup(config, run_dir=None, save_json=None):
    transition = None
    if config.get("diagnostic_transition"):
        transition = create_transition_predictor(config["seed"])
    return None, None, transition

def step(ctx):
    max_returns = ctx.max_returns
    if ctx.branch == "replay":
        previous = ctx.level_sampler.get_levels_extra(ctx.sampler, ctx.level_indices)["max_return"]
        max_returns = jnp.maximum(previous, max_returns)

    scores = max_mc(ctx.dones, ctx.values, max_returns)
    method_metrics = {}
    transition = ctx.transition

    if transition is not None:
        transition, transition_loss, _, _, _ = update_transition_predictor(
            transition, ctx.batch[0], ctx.batch[1]
        )
        method_metrics["transition_loss"] = transition_loss

    return {
        "scores": scores,
        "method_metrics": method_metrics,
        "predictor": ctx.predictor,
        "transition": transition,
        "max_returns": max_returns,
        "mean_regret_diff": jnp.asarray(0.0),
    }
