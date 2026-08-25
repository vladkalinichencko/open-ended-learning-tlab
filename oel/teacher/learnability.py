import jax.numpy as jnp

from oel.methods import create_predictor, predict_success, update_predictor

def setup(config, run_dir=None, save_json=None):
    predictor, encode = create_predictor(
        config["score"], config["seed"], config["predictor_lr"]
    )
    return predictor, encode, None

def step(ctx):
    inputs = ctx.encode(ctx.levels)
    probability = predict_success(ctx.predictor, inputs)
    scores = probability * (1 - probability)
    predictor, predictor_loss = update_predictor(ctx.predictor, inputs, ctx.signals["success"])
    return {
        "scores": scores,
        "method_metrics": {
            "predicted_success": probability.mean(),
            "predictor_loss": predictor_loss,
        },
        "predictor": predictor,
        "transition": ctx.transition,
        "max_returns": ctx.max_returns,
        "mean_regret_diff": jnp.asarray(0.0),
    }
