import jax.numpy as jnp

from oel.methods import ResNetPredictor

def setup(config, run_dir, save_json):
    predictor = ResNetPredictor(config["seed"], config["predictor_lr"])
    config["teacher_device"] = str(predictor.device)
    save_json(run_dir / "config.json", config)
    print(__import__("json").dumps({"teacher_device": config["teacher_device"]}), flush=True)
    return predictor, None, None

def step(ctx):
    probability, predictor_loss = ctx.predictor.predict_and_update(ctx.levels, ctx.signals["success"])
    scores = probability * (1 - probability)
    return {
        "scores": scores,
        "method_metrics": {
            "predicted_success": probability.mean(),
            "predictor_loss": predictor_loss,
        },
        "predictor": ctx.predictor,
        "transition": ctx.transition,
        "max_returns": ctx.max_returns,
        "mean_regret_diff": jnp.asarray(0.0),
    }
