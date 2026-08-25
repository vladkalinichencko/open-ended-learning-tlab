import jax.numpy as jnp

from jaxued.utils import positive_value_loss

from oel.methods import create_transition_predictor, update_transition_predictor

def setup(config, run_dir=None, save_json=None):
    return None, None, create_transition_predictor(config["seed"])

def step(ctx):
    pvl = positive_value_loss(ctx.dones, ctx.advantages)
    transition, transition_loss, transition_scores, _, _ = update_transition_predictor(
        ctx.transition, ctx.batch[0], ctx.batch[1]
    )
    scores = pvl + ctx.config["transition_weight"] * transition_scores
    mean_regret_diff = jnp.asarray(0.0)

    if ctx.branch == "replay":
        previous = ctx.level_sampler.get_levels_extra(ctx.sampler, ctx.level_indices)["difficulty"]
        seen = ctx.replayed_before[ctx.level_indices]
        mean_regret_diff = jnp.where(seen, previous - scores, 0).mean()

    return {
        "scores": scores,
        "method_metrics": {
            "transition_loss": transition_loss,
            "pvl_score": pvl.mean(),
            "mean_regret_diff": mean_regret_diff,
            "colearnability_bonus": (
                mean_regret_diff
                if ctx.config["score"] == "traced_colearn"
                else jnp.asarray(0.0)
            ),
            "curriculum_score": scores.mean(),
        },
        "predictor": ctx.predictor,
        "transition": transition,
        "max_returns": ctx.max_returns,
        "mean_regret_diff": mean_regret_diff,
    }
