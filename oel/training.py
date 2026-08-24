"""Общий PPO-путь и точный уменьшенный SFL."""

import json
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import serialization
from flax.training.train_state import TrainState

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ext" / "jaxued" / "src"))
sys.path.insert(0, str(ROOT / "ext" / "jaxued" / "examples"))

import maze_plr as base  # noqa: E402
from jaxued.environments import Maze  # noqa: E402
from jaxued.environments.maze import make_level_generator, make_level_mutator_minimax  # noqa: E402
from jaxued.level_sampler import LevelSampler  # noqa: E402
from jaxued.utils import accumulate_rollout_stats, compute_max_returns, max_mc, positive_value_loss  # noqa: E402
from jaxued.wrappers import AutoReplayWrapper  # noqa: E402

from .diagnostics import save_json, write_report  # noqa: E402
from .methods import (  # noqa: E402
    ResNetPredictor,
    create_predictor,
    create_transition_predictor,
    predict_success,
    predict_transitions,
    update_predictor,
    update_transition_predictor,
)


def create_student(config: dict, env, sample_level) -> TrainState:
    rng = jax.random.PRNGKey(config["seed"])
    obs, _ = env.reset_to_level(rng, sample_level(rng), env.default_params)
    obs = jax.tree_util.tree_map(
        lambda x: jnp.repeat(
            jnp.repeat(x[None], config["num_train_envs"], 0)[None], config["num_steps"], 0
        ),
        obs,
    )
    network = base.ActorCritic(env.action_space(env.default_params).n)
    params = network.init(
        rng,
        (obs, jnp.zeros((config["num_steps"], config["num_train_envs"]))),
        base.ActorCritic.initialize_carry((config["num_train_envs"],)),
    )

    def learning_rate(step):
        updates = step // (config["num_minibatches"] * config["epoch_ppo"])
        return config["lr"] * (1 - updates / config["num_updates"])

    optimizer = optax.chain(
        optax.clip_by_global_norm(config["max_grad_norm"]),
        optax.adam(learning_rate, eps=1e-5),
    )
    return TrainState.create(apply_fn=network.apply, params=params, tx=optimizer)


def make_train_functions(config: dict, env):
    @jax.jit
    def collect_rollout(rng, student, levels):
        rng, reset_rng, rollout_rng = jax.random.split(rng, 3)
        observations, states = jax.vmap(env.reset_to_level, in_axes=(0, 0, None))(
            jax.random.split(reset_rng, config["num_train_envs"]), levels, env.default_params
        )
        carry = base.ActorCritic.initialize_carry((config["num_train_envs"],))
        (rollout_rng, student, _, _, _, last_value), trajectory = base.sample_trajectories_rnn(
            rollout_rng,
            env,
            env.default_params,
            student,
            carry,
            observations,
            states,
            config["num_train_envs"],
            config["num_steps"],
        )
        obs, actions, rewards, dones, log_probs, values, _ = trajectory
        advantages, targets = base.compute_gae(
            config["gamma"], config["gae_lambda"], last_value, values, rewards, dones
        )
        success, _, episodes = accumulate_rollout_stats(
            dones, (rewards > 0).astype(jnp.float32), time_average=False
        )
        batch = (obs, actions, dones, log_probs, values, targets, advantages)
        return rollout_rng, batch, {
            "reward": rewards.mean(),
            "train_solve_rate": success.mean(),
            "episodes": episodes.sum(),
        }, {"rewards": rewards, "success": success}

    def update(rng, student, batch, update_grad):
        (rng, student), losses = base.update_actor_critic_rnn(
            rng,
            student,
            base.ActorCritic.initialize_carry((config["num_train_envs"],)),
            batch,
            config["num_train_envs"],
            config["num_steps"],
            config["num_minibatches"],
            config["epoch_ppo"],
            config["clip_eps"],
            config["entropy_coeff"],
            config["critic_coeff"],
            update_grad=update_grad,
        )
        loss, (value_loss, policy_loss, entropy) = jax.tree_util.tree_map(jnp.mean, losses)
        return rng, student, {
            "loss": loss,
            "value_loss": value_loss,
            "policy_loss": policy_loss,
            "entropy": entropy,
        }

    ppo_update = jax.jit(lambda rng, student, batch: update(rng, student, batch, True))
    ppo_evaluate = jax.jit(lambda rng, student, batch: update(rng, student, batch, False))
    return collect_rollout, ppo_update, ppo_evaluate


def make_search(config: dict, eval_env, sample_level):
    batch_size = config["sfl_search_batch_size"]
    attempts = config["sfl_attempts"]

    @jax.jit
    def search_batch(rng, student):
        rng, level_rng, reset_rng, rollout_rng = jax.random.split(rng, 4)
        levels = jax.vmap(sample_level)(jax.random.split(level_rng, batch_size))
        repeated = jax.tree_util.tree_map(lambda x: jnp.repeat(x, attempts, axis=0), levels)
        observations, states = jax.vmap(eval_env.reset_to_level, in_axes=(0, 0, None))(
            jax.random.split(reset_rng, batch_size * attempts), repeated, eval_env.default_params
        )
        _, rewards, lengths = base.evaluate_rnn(
            rollout_rng,
            eval_env,
            eval_env.default_params,
            student,
            base.ActorCritic.initialize_carry((batch_size * attempts,)),
            observations,
            states,
            eval_env.default_params.max_steps_in_episode,
        )
        mask = jnp.arange(rewards.shape[0])[:, None] < lengths
        solved = ((rewards * mask).sum(axis=0) > 0).reshape(batch_size, attempts)
        probability = solved.mean(axis=1)
        return rng, levels, probability

    return search_batch


def search_frontier(config: dict, rng, student, search_batch):
    levels, probabilities = [], []
    batches = config["sfl_num_levels"] // config["sfl_search_batch_size"]
    for _ in range(batches):
        rng, batch_levels, batch_probability = search_batch(rng, student)
        levels.append(batch_levels)
        probabilities.append(batch_probability)
    levels = jax.tree_util.tree_map(lambda *xs: jnp.concatenate(xs), *levels)
    probabilities = jnp.concatenate(probabilities)
    score = probabilities * (1 - probabilities)
    selected = jnp.argsort(score)[-config["sfl_buffer_size"] :]
    return rng, jax.tree_util.tree_map(lambda x: x[selected], levels), probabilities[selected]


def make_evaluate(config: dict, eval_env, sample_level):
    levels = jax.vmap(sample_level)(
        jax.random.split(jax.random.PRNGKey(config["validation_seed"]), config["validation_size"])
    )
    attempts = config["eval_num_attempts"]
    repeated = jax.tree_util.tree_map(lambda x: jnp.tile(x, (attempts,) + (1,) * (x.ndim - 1)), levels)
    count = config["validation_size"]

    @jax.jit
    def evaluate(rng, student):
        rng, reset_rng, rollout_rng = jax.random.split(rng, 3)
        observations, states = jax.vmap(eval_env.reset_to_level, in_axes=(0, 0, None))(
            jax.random.split(reset_rng, count * attempts), repeated, eval_env.default_params
        )
        _, rewards, lengths = base.evaluate_rnn(
            rollout_rng,
            eval_env,
            eval_env.default_params,
            student,
            base.ActorCritic.initialize_carry((count * attempts,)),
            observations,
            states,
            eval_env.default_params.max_steps_in_episode,
        )
        mask = jnp.arange(rewards.shape[0])[:, None] < lengths
        solved = ((rewards * mask).sum(axis=0) > 0).reshape(attempts, count)
        returns = (rewards * mask).sum(axis=0).reshape(attempts, count)
        return rng, solved.mean(axis=0), returns.mean(axis=0)

    return evaluate


def diagnostic_rollout(config: dict, eval_env, sample_level, student) -> dict:
    level_key = jax.random.split(
        jax.random.PRNGKey(config["validation_seed"]), config["validation_size"]
    )[0]
    level = sample_level(level_key)

    def run(reset_memory: bool) -> dict:
        reset_key, action_key = jax.random.split(jax.random.PRNGKey(config["seed"] + 20000))
        observations, states = jax.vmap(eval_env.reset_to_level, in_axes=(0, 0, None))(
            reset_key[None], jax.tree_util.tree_map(lambda x: jnp.asarray(x)[None], level), eval_env.default_params
        )

        def step(carry, _):
            rng, memory, observation, state, done = carry
            rng, choose_key, env_key = jax.random.split(rng, 3)
            if reset_memory:
                memory = base.ActorCritic.initialize_carry((1,))
            next_memory, policy, value = student.apply_fn(
                student.params,
                jax.tree_util.tree_map(lambda x: x[None], (observation, done)),
                memory,
            )
            action = policy.sample(seed=choose_key).squeeze(0)
            next_observation, next_state, reward, next_done, _ = jax.vmap(
                eval_env.step, in_axes=(0, 0, 0, None)
            )(env_key[None], state, action, eval_env.default_params)
            output = {
                "position": state.agent_pos[0],
                "observation": observation.image[0],
                "action": action[0],
                "probabilities": jax.nn.softmax(policy.logits_parameter()).squeeze(0).squeeze(0),
                "reward": reward[0],
                "done": next_done[0],
                "value": value.squeeze(),
                "cell": next_memory[0][0],
                "hidden": next_memory[1][0],
            }
            return (rng, next_memory, next_observation, next_state, next_done), output

        initial = (
            action_key,
            base.ActorCritic.initialize_carry((1,)),
            observations,
            states,
            jnp.zeros(1, dtype=bool),
        )
        _, trajectory = jax.lax.scan(step, initial, None, eval_env.default_params.max_steps_in_episode)
        trajectory = jax.device_get(trajectory)
        dones = np.asarray(trajectory["done"])
        length = int(np.argmax(dones) + 1) if dones.any() else len(dones)
        return {
            "positions": np.asarray(trajectory["position"][:length]).tolist(),
            "observations": np.asarray(trajectory["observation"][:length]).tolist(),
            "actions": np.asarray(trajectory["action"][:length]).tolist(),
            "action_probabilities": np.asarray(trajectory["probabilities"][:length]).tolist(),
            "rewards": np.asarray(trajectory["reward"][:length]).tolist(),
            "dones": np.asarray(trajectory["done"][:length]).tolist(),
            "values": np.asarray(trajectory["value"][:length]).tolist(),
            "cell": np.asarray(trajectory["cell"][:length]).tolist(),
            "hidden": np.asarray(trajectory["hidden"][:length]).tolist(),
        }

    return {
        "wall_map": np.asarray(level.wall_map).tolist(),
        "agent_pos": np.asarray(level.agent_pos).tolist(),
        "goal_pos": np.asarray(level.goal_pos).tolist(),
        "normal": run(False),
        "zero_memory": run(True),
    }


def rebuild_report(run_dir: Path) -> None:
    config = json.loads((run_dir / "config.json").read_text())
    eval_env = Maze(13, 13, agent_view_size=config["agent_view_size"], normalize_obs=True)
    sample_level = make_level_generator(13, 13, config["n_walls"])
    student = create_student(config, AutoReplayWrapper(eval_env), sample_level)
    checkpoint = (run_dir / "checkpoint.msgpack").read_bytes()
    student = student.replace(params=serialization.from_bytes(student.params, checkpoint))
    rollout = diagnostic_rollout(config, eval_env, sample_level, student)
    save_json(run_dir / "rollout.json", rollout)
    records = json.loads((run_dir / "metrics.json").read_text())
    events = json.loads((run_dir / "timeline.json").read_text())
    write_report(run_dir, config, records, events, rollout)


def run_sfl(config: dict) -> Path:
    if config["sfl_num_levels"] % config["sfl_search_batch_size"]:
        raise ValueError("sfl_num_levels must be divisible by sfl_search_batch_size")
    replay_count = round(config["num_train_envs"] * config["sfl_replay_fraction"])
    if replay_count + (config["num_train_envs"] - replay_count) != config["num_train_envs"]:
        raise ValueError("SFL batch size mismatch")

    config = dict(config)
    config["device"] = str(jax.devices()[0])
    run_dir = ROOT / "runs" / config["name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    save_json(run_dir / "config.json", config)
    print(json.dumps({k: config[k] for k in ("name", "device", "seed", "num_train_envs", "num_steps")}), flush=True)

    eval_env = Maze(13, 13, agent_view_size=config["agent_view_size"], normalize_obs=True)
    env = AutoReplayWrapper(eval_env)
    sample_level = make_level_generator(13, 13, config["n_walls"])
    student = create_student(config, env, sample_level)
    collect_rollout, ppo_update, _ = make_train_functions(config, env)
    search_batch = make_search(config, eval_env, sample_level)
    evaluate = make_evaluate(config, eval_env, sample_level)
    rng = jax.random.PRNGKey(config["seed"])
    records, events, frontier = [], [], None
    teacher_env_steps = 0

    for update in range(config["num_updates"]):
        if update % config["sfl_search_interval"] == 0:
            started = time.perf_counter()
            rng, frontier, frontier_probability = search_frontier(config, rng, student, search_batch)
            jax.block_until_ready(frontier_probability)
            teacher_env_steps += (
                config["sfl_num_levels"]
                * config["sfl_attempts"]
                * eval_env.default_params.max_steps_in_episode
            )
            finished = time.perf_counter()
            events.append({"phase": "search", "update": update, "start": started, "end": finished})

        rng, replay_rng, fresh_rng = jax.random.split(rng, 3)
        chosen = jax.random.randint(replay_rng, (replay_count,), 0, config["sfl_buffer_size"])
        replay_levels = jax.tree_util.tree_map(lambda x: x[chosen], frontier)
        fresh_count = config["num_train_envs"] - replay_count
        fresh_levels = jax.vmap(sample_level)(jax.random.split(fresh_rng, fresh_count))
        levels = jax.tree_util.tree_map(lambda a, b: jnp.concatenate((a, b)), replay_levels, fresh_levels)

        started = time.perf_counter()
        rng, batch, rollout_metrics, _ = collect_rollout(rng, student, levels)
        jax.block_until_ready(batch)
        rolled_out = time.perf_counter()
        rng, student, ppo_metrics = ppo_update(rng, student, batch)
        jax.block_until_ready(student.params)
        trained = time.perf_counter()
        train_metrics = {**rollout_metrics, **ppo_metrics}
        events.extend([
            {"phase": "rollout", "update": update + 1, "start": started, "end": rolled_out},
            {"phase": "ppo", "update": update + 1, "start": rolled_out, "end": trained},
        ])

        if (update + 1) % config["eval_freq"] == 0 or update + 1 == config["num_updates"]:
            eval_started = time.perf_counter()
            rng, solve_rates, returns = evaluate(rng, student)
            jax.block_until_ready(solve_rates)
            eval_finished = time.perf_counter()
            events.append({"phase": "eval", "update": update + 1, "start": eval_started, "end": eval_finished})
            train_env_steps = (update + 1) * config["num_train_envs"] * config["num_steps"]
            row = {
                "update": update + 1,
                "train_env_steps": train_env_steps,
                "teacher_env_steps": teacher_env_steps,
                "total_env_steps": train_env_steps + teacher_env_steps,
                "validation_solve_mean": float(solve_rates.mean()),
                "validation_solve_by_level": np.asarray(solve_rates, float).tolist(),
                "validation_return_by_level": np.asarray(returns, float).tolist(),
                "frontier_probability_mean": float(frontier_probability.mean()),
                **{key: float(value) for key, value in train_metrics.items()},
            }
            records.append(row)
            save_json(run_dir / "metrics.json", records)
            write_report(run_dir, config, records, events)
            print(json.dumps(row), flush=True)

    (run_dir / "checkpoint.msgpack").write_bytes(serialization.to_bytes(student.params))
    rollout = diagnostic_rollout(config, eval_env, sample_level, student)
    save_json(run_dir / "rollout.json", rollout)
    save_json(run_dir / "timeline.json", events)
    write_report(run_dir, config, records, events, rollout)
    return run_dir


def run_accel(config: dict) -> Path:
    config = dict(config)
    config["device"] = str(jax.devices()[0])
    run_dir = ROOT / "runs" / config["name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    save_json(run_dir / "config.json", config)
    print(json.dumps({k: config[k] for k in ("name", "device", "seed", "num_train_envs", "num_steps", "score")}), flush=True)

    eval_env = Maze(13, 13, agent_view_size=config["agent_view_size"], normalize_obs=True)
    env = AutoReplayWrapper(eval_env)
    sample_level = make_level_generator(13, 13, config["n_walls"])
    mutate_level = make_level_mutator_minimax(100)
    student = create_student(config, env, sample_level)
    collect_rollout, ppo_update, ppo_evaluate = make_train_functions(config, env)
    evaluate = make_evaluate(config, eval_env, sample_level)
    level_sampler = LevelSampler(
        capacity=config["level_buffer_capacity"],
        replay_prob=config["replay_prob"],
        staleness_coeff=config["staleness_coeff"],
        minimum_fill_ratio=config["minimum_fill_ratio"],
        prioritization="rank",
        prioritization_params={"temperature": config["temperature"]},
        duplicate_check=True,
    )
    placeholder = sample_level(jax.random.PRNGKey(0))
    sampler = level_sampler.initialize(placeholder, {"max_return": -jnp.inf, "difficulty": -jnp.inf})
    sample_replay = jax.jit(lambda state, key: level_sampler.sample_replay_levels(
        state, key, config["num_train_envs"]
    ))
    insert = jax.jit(level_sampler.insert_batch)
    update_buffer = jax.jit(level_sampler.update_batch)
    mutate = jax.jit(lambda key, levels: jax.vmap(mutate_level, (0, 0, None))(
        jax.random.split(key, config["num_train_envs"]), levels, config["num_edits"]
    ))
    predictor = encode = transition = None
    if config["score"] in ("fixed", "cnn"):
        predictor, encode = create_predictor(
            config["score"], config["seed"], config["predictor_lr"]
        )
    elif config["score"] == "resnet":
        predictor = ResNetPredictor(config["seed"], config["predictor_lr"])
        config["teacher_device"] = str(predictor.device)
        save_json(run_dir / "config.json", config)
        print(json.dumps({"teacher_device": config["teacher_device"]}), flush=True)
    elif config["score"] in ("traced", "traced_colearn") or config.get("diagnostic_transition"):
        transition = create_transition_predictor(config["seed"])

    rng = jax.random.PRNGKey(config["seed"])
    records, events = [], []
    last_replay_levels = None
    previous_replay_mask = jnp.zeros(config["level_buffer_capacity"], dtype=bool)
    replayed_before = jnp.zeros(config["level_buffer_capacity"], dtype=bool)
    branch_counts = {"new": 0, "replay": 0, "mutation": 0}

    for update in range(config["num_updates"]):
        rng, branch_rng, level_rng = jax.random.split(rng, 3)
        should_replay = bool(level_sampler.sample_replay_decision(sampler, branch_rng))
        if last_replay_levels is not None:
            branch = "mutation"
            levels = mutate(level_rng, last_replay_levels)
            level_indices = None
            last_replay_levels = None
        elif should_replay:
            branch = "replay"
            sampler, (level_indices, levels) = sample_replay(sampler, level_rng)
            last_replay_levels = levels
        else:
            branch = "new"
            levels = jax.vmap(sample_level)(jax.random.split(level_rng, config["num_train_envs"]))
            level_indices = None
        branch_counts[branch] += 1

        started = time.perf_counter()
        rng, batch, rollout_metrics, signals = collect_rollout(rng, student, levels)
        jax.block_until_ready(batch)
        rolled_out = time.perf_counter()
        dones, values, advantages = batch[2], batch[4], batch[6]
        max_returns = compute_max_returns(dones, signals["rewards"])
        method_metrics = {}
        if config["score"] == "maxmc":
            if branch == "replay":
                previous = level_sampler.get_levels_extra(sampler, level_indices)["max_return"]
                max_returns = jnp.maximum(previous, max_returns)
            scores = max_mc(dones, values, max_returns)
            if transition is not None:
                transition, transition_loss, _, _, _ = update_transition_predictor(
                    transition, batch[0], batch[1]
                )
                method_metrics["transition_loss"] = transition_loss
        elif transition is not None:
            pvl = positive_value_loss(dones, advantages)
            transition, transition_loss, transition_scores, _, _ = update_transition_predictor(
                transition, batch[0], batch[1]
            )
            scores = pvl + config["transition_weight"] * transition_scores
            mean_regret_diff = jnp.asarray(0.0)
            if branch == "replay":
                previous = level_sampler.get_levels_extra(sampler, level_indices)["difficulty"]
                seen = replayed_before[level_indices]
                mean_regret_diff = jnp.where(seen, previous - scores, 0).mean()
            method_metrics = {
                "transition_loss": transition_loss,
                "pvl_score": pvl.mean(),
                "mean_regret_diff": mean_regret_diff,
                "colearnability_bonus": (
                    mean_regret_diff if config["score"] == "traced_colearn" else jnp.asarray(0.0)
                ),
                "curriculum_score": scores.mean(),
            }
        elif config["score"] == "resnet":
            probability, predictor_loss = predictor.predict_and_update(levels, signals["success"])
            scores = probability * (1 - probability)
            method_metrics = {
                "predicted_success": probability.mean(),
                "predictor_loss": predictor_loss,
            }
        else:
            inputs = encode(levels)
            probability = predict_success(predictor, inputs)
            scores = probability * (1 - probability)
            predictor, predictor_loss = update_predictor(predictor, inputs, signals["success"])
            method_metrics = {
                "predicted_success": probability.mean(),
                "predictor_loss": predictor_loss,
            }

        extras = {"max_return": max_returns, "difficulty": scores}
        if branch == "replay":
            sampler = update_buffer(sampler, level_indices, scores, extras)
            if config["score"] == "traced_colearn":
                sampler = {
                    **sampler,
                    "scores": sampler["scores"]
                    + previous_replay_mask * config["colearnability_weight"] * mean_regret_diff,
                }
            if transition is not None:
                current_mask = jnp.zeros(config["level_buffer_capacity"], dtype=bool).at[level_indices].set(True)
                replayed_before = replayed_before | current_mask
                previous_replay_mask = current_mask
            rng, student, ppo_metrics = ppo_update(rng, student, batch)
        else:
            sampler, _ = insert(sampler, levels, scores, extras)
            rng, student, ppo_metrics = ppo_evaluate(rng, student, batch)
        jax.block_until_ready(student.params)
        trained = time.perf_counter()
        events.extend([
            {"phase": "rollout", "update": update + 1, "start": started, "end": rolled_out},
            {"phase": "ppo", "update": update + 1, "start": rolled_out, "end": trained},
        ])

        if (update + 1) % config["eval_freq"] == 0 or update + 1 == config["num_updates"]:
            eval_started = time.perf_counter()
            rng, solve_rates, returns = evaluate(rng, student)
            jax.block_until_ready(solve_rates)
            eval_finished = time.perf_counter()
            events.append({"phase": "eval", "update": update + 1, "start": eval_started, "end": eval_finished})
            metrics = {**rollout_metrics, **ppo_metrics, **method_metrics}
            train_env_steps = (update + 1) * config["num_train_envs"] * config["num_steps"]
            row = {
                "update": update + 1,
                "train_env_steps": train_env_steps,
                "teacher_env_steps": 0,
                "total_env_steps": train_env_steps,
                "validation_solve_mean": float(solve_rates.mean()),
                "validation_solve_by_level": np.asarray(solve_rates, float).tolist(),
                "validation_return_by_level": np.asarray(returns, float).tolist(),
                "buffer_size": int(sampler["size"]),
                "branch_counts": dict(branch_counts),
                **{key: float(value) for key, value in metrics.items()},
            }
            records.append(row)
            save_json(run_dir / "metrics.json", records)
            write_report(run_dir, config, records, events)
            print(json.dumps(row), flush=True)

    (run_dir / "checkpoint.msgpack").write_bytes(serialization.to_bytes(student.params))
    rollout = diagnostic_rollout(config, eval_env, sample_level, student)
    if transition is not None:
        (run_dir / "teacher.msgpack").write_bytes(serialization.to_bytes(transition.params))
        predictions = predict_transitions(transition, batch[0], batch[1])
        targets = batch[0].image[1:]
        rollout["transition"] = {
            "observations": np.asarray(batch[0].image[:-1, 0]).tolist(),
            "actions": np.asarray(batch[1][:-1, 0]).tolist(),
            "predictions": np.asarray(predictions[:, 0]).tolist(),
            "targets": np.asarray(targets[:, 0]).tolist(),
        }
    save_json(run_dir / "rollout.json", rollout)
    save_json(run_dir / "timeline.json", events)
    write_report(run_dir, config, records, events, rollout)
    return run_dir
