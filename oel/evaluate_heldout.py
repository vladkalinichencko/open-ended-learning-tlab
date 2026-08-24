"""Held-out evaluation and raw rollout logging for saved policies."""

import argparse
import base64
import io
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from flax import serialization

from jaxued.environments import Maze
from jaxued.environments.maze import Level, make_level_generator
from jaxued.wrappers import AutoReplayWrapper

from .methods import create_transition_predictor
from .training import base, create_student


ATTEMPTS = 5


def load_student(run_dir: Path):
    config = json.loads((run_dir / "config.json").read_text())
    env = Maze(13, 13, agent_view_size=config["agent_view_size"], normalize_obs=True)
    sample_level = make_level_generator(13, 13, config["n_walls"])
    student = create_student(config, AutoReplayWrapper(env), sample_level)
    params = serialization.from_bytes(student.params, (run_dir / "checkpoint.msgpack").read_bytes())
    return config, env, student.replace(params=params)


def collect(env, student, levels, seed: int, zero_memory: bool) -> dict:
    count = levels.wall_map.shape[0]
    reset_key, rollout_key = jax.random.split(jax.random.PRNGKey(seed))
    observations, states = jax.vmap(env.reset_to_level, in_axes=(0, 0, None))(
        jax.random.split(reset_key, count), levels, env.default_params
    )

    def step(carry, _):
        rng, memory, observation, state, done = carry
        rng, action_key, env_key = jax.random.split(rng, 3)
        if zero_memory:
            memory = base.ActorCritic.initialize_carry((count,))
        next_memory, policy, value = student.apply_fn(
            student.params,
            jax.tree_util.tree_map(lambda x: x[None], (observation, done)),
            memory,
        )
        action = policy.sample(seed=action_key).squeeze(0)
        next_observation, next_state, reward, next_done, _ = jax.vmap(
            env.step, in_axes=(0, 0, 0, None)
        )(jax.random.split(env_key, count), state, action, env.default_params)
        output = {
            "observation": observation.image,
            "position": state.agent_pos,
            "action": action,
            "probabilities": jax.nn.softmax(policy.logits_parameter()).squeeze(0),
            "reward": reward,
            "done": next_done,
            "value": value.squeeze(0),
            "cell": next_memory[0],
            "hidden": next_memory[1],
        }
        return (rng, next_memory, next_observation, next_state, next_done), output

    initial = (
        rollout_key,
        base.ActorCritic.initialize_carry((count,)),
        observations,
        states,
        jnp.zeros(count, dtype=bool),
    )
    _, trajectory = jax.lax.scan(step, initial, None, env.default_params.max_steps_in_episode)
    return jax.device_get(trajectory)


def summarize(trajectory: dict, names: list[str]) -> dict:
    dones = np.asarray(trajectory["done"])
    rewards = np.asarray(trajectory["reward"])
    length = np.where(dones.any(axis=0), dones.argmax(axis=0) + 1, len(dones))
    mask = np.arange(len(dones))[:, None] < length
    returns = (rewards * mask).sum(axis=0).reshape(ATTEMPTS, len(names))
    lengths = length.reshape(ATTEMPTS, len(names))
    return {
        name: {
            "solve_rate": float((returns[:, index] > 0).mean()),
            "returns": returns[:, index].tolist(),
            "episode_lengths": lengths[:, index].tolist(),
        }
        for index, name in enumerate(names)
    }


def evaluate(run_dir: Path, output_dir: Path) -> dict:
    config, env, student = load_student(run_dir)
    names = config["eval_levels"]
    levels = Level.load_prefabs(names)
    levels = jax.tree_util.tree_map(
        lambda x: jnp.tile(x, (ATTEMPTS,) + (1,) * (x.ndim - 1)), levels
    )
    normal = collect(env, student, levels, config["seed"] + 30_000, False)
    zero_memory = collect(env, student, levels, config["seed"] + 30_000, True)
    result = {
        "run": config["name"],
        "seed": config["seed"],
        "attempts": ATTEMPTS,
        "levels": names,
        "normal": summarize(normal, names),
        "zero_memory": summarize(zero_memory, names),
    }

    arrays = {
        f"normal_{key}": np.asarray(value)
        for key, value in normal.items()
    } | {
        f"zero_memory_{key}": np.asarray(value)
        for key, value in zero_memory.items()
    }
    teacher_path = run_dir / "teacher.msgpack"
    if teacher_path.exists():
        transition = create_transition_predictor(config["seed"])
        transition = transition.replace(
            params=serialization.from_bytes(transition.params, teacher_path.read_bytes())
        )
        predictions = transition.apply_fn(
            transition.params,
            jnp.asarray(normal["observation"][:-1]),
            jnp.asarray(normal["action"][:-1]),
        )
        targets = np.asarray(normal["observation"][1:])
        arrays["transition_predictions"] = np.asarray(predictions)
        arrays["transition_targets"] = targets
        arrays["transition_l1"] = np.abs(np.asarray(predictions) - targets).mean(axis=(2, 3, 4))

    destination = output_dir / config["name"]
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "heldout.json").write_text(json.dumps(result, indent=2))
    np.savez_compressed(destination / "heldout_rollouts.npz", **arrays)
    return result


def write_report(results: list[dict], output: Path) -> None:
    names = results[0]["levels"]
    x = np.arange(len(names))
    width = 0.8 / len(results)
    fig, axis = plt.subplots(figsize=(12, 5.5))
    for index, result in enumerate(results):
        solve = [result["normal"][name]["solve_rate"] for name in names]
        axis.bar(x + (index - (len(results) - 1) / 2) * width, solve, width, label=result["run"])
    axis.set(xticks=x, xticklabels=names, ylabel="Solve rate", ylim=(0, 1))
    axis.tick_params(axis="x", rotation=30)
    axis.legend(fontsize=8)
    fig.tight_layout()
    image = io.BytesIO()
    fig.savefig(image, format="png", dpi=150)
    plt.close(fig)
    rows = "".join(
        "<tr><td>" + name + "</td>" + "".join(
            f'<td>{result["normal"][name]["solve_rate"]:.2f}</td>' for result in results
        ) + "</tr>"
        for name in names
    )
    headers = "".join(f"<th>{result['run']}</th>" for result in results)
    output.write_text(f'''<!doctype html><meta charset="utf-8"><title>A100 held-out evaluation</title>
<style>body{{font:15px system-ui;max-width:1100px;margin:32px auto;color:#222}}img{{width:100%}}
table{{border-collapse:collapse}}td,th{{padding:6px 10px;border:1px solid #ddd;text-align:right}}</style>
<h1>Held-out evaluation, seed 0</h1>
<p>Пять стохастических попыток на каждом из восьми уровней. Это короткий A100 triage на 5000 updates, а не финальный прогон задания.</p>
<img alt="held-out solve rate by level" src="data:image/png;base64,{base64.b64encode(image.getvalue()).decode()}">
<table><tr><th>level</th>{headers}</tr>{rows}</table>''')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("runs", nargs="+", type=Path)
    args = parser.parse_args()
    results = [evaluate(run, args.output) for run in args.runs]
    write_report(results, args.output / "heldout_comparison.html")


if __name__ == "__main__":
    main()
