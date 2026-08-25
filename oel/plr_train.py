"""Общий PLR/ACCEL train loop; train_step/create_train_state — копии из teacher/sfl."""

import json
import os
import sys
import time
from pathlib import Path
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
import wandb
from jaxued.environments import Maze, MazeRenderer
from jaxued.environments.maze import Level, make_level_generator, make_level_mutator_minimax
from jaxued.level_sampler import LevelSampler
from jaxued.utils import compute_max_returns
from jaxued.wrappers import AutoReplayWrapper

import chex

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ext" / "jaxued" / "src"))
sys.path.insert(0, str(ROOT / "ext" / "jaxued" / "examples"))

from rl_core import (
    ActorCritic,
    TrainState,
    UpdateState,
    compute_gae,
    evaluate_rnn,
    sample_trajectories_rnn,
    setup_checkpointing,
    train_state_to_log_dict,
    update_actor_critic_rnn,
)

def main(config, variant, project="JAXUED_TEST"):
    tags = []
    if not config["exploratory_grad_updates"]:
        tags.append("robust")
    if config["use_accel"]:
        tags.append("ACCEL")
    else:
        tags.append("PLR")
    wandb.init(config=config, project=project, group=config["run_name"], tags=tags)
    config = wandb.config

    wandb.define_metric("num_updates")
    wandb.define_metric("num_env_steps")
    wandb.define_metric("solve_rate/*", step_metric="num_updates")
    wandb.define_metric("level_sampler/*", step_metric="num_updates")
    wandb.define_metric("agent/*", step_metric="num_updates")
    wandb.define_metric("return/*", step_metric="num_updates")
    wandb.define_metric("eval_ep_lengths/*", step_metric="num_updates")

    def log_eval(stats, train_state_info):
        print(f"Logging update: {stats['update_count']}")
        env_steps = stats["update_count"] * config["num_train_envs"] * config["num_steps"]
        log_dict = {
            "num_updates": stats["update_count"],
            "num_env_steps": env_steps,
            "sps": env_steps / stats["time_delta"],
        }
        solve_rates = stats["eval_solve_rates"]
        returns = stats["eval_returns"]
        log_dict.update(
            {f"solve_rate/{name}": solve_rate for name, solve_rate in zip(config["eval_levels"], solve_rates)}
        )
        log_dict.update({"solve_rate/mean": solve_rates.mean()})
        log_dict.update({f"return/{name}": ret for name, ret in zip(config["eval_levels"], returns)})
        log_dict.update({"return/mean": returns.mean()})
        log_dict.update({"eval_ep_lengths/mean": stats["eval_ep_lengths"].mean()})
        log_dict.update(train_state_info["log"])
        log_dict.update(
            {
                "images/highest_scoring_level": wandb.Image(
                    np.array(stats["highest_scoring_level"]), caption="Highest scoring level"
                )
            }
        )
        log_dict.update(
            {
                "images/highest_weighted_level": wandb.Image(
                    np.array(stats["highest_weighted_level"]), caption="Highest weighted level"
                )
            }
        )
        for branch in ("dr", "replay", "mutation"):
            if train_state_info["info"][f"num_{branch}_updates"] > 0:
                log_dict.update(
                    {
                        f"images/{branch}_levels": [
                            wandb.Image(np.array(image)) for image in stats[f"{branch}_levels"]
                        ]
                    }
                )
        for index, level_name in enumerate(config["eval_levels"]):
            frames, episode_length = stats["eval_animation"][0][:, index], stats["eval_animation"][1][index]
            frames = np.array(frames[:episode_length])
            log_dict.update({f"animations/{level_name}": wandb.Video(frames, fps=4)})
        wandb.log(log_dict)

    env = Maze(max_height=13, max_width=13, agent_view_size=config["agent_view_size"], normalize_obs=True)
    eval_env = env
    sample_random_level = make_level_generator(env.max_height, env.max_width, config["n_walls"])
    env_renderer = MazeRenderer(env, tile_size=8)
    env = AutoReplayWrapper(env)
    env_params = env.default_params
    mutate_level = make_level_mutator_minimax(100)

    level_sampler = LevelSampler(
        capacity=config["level_buffer_capacity"],
        replay_prob=config["replay_prob"],
        staleness_coeff=config["staleness_coeff"],
        minimum_fill_ratio=config["minimum_fill_ratio"],
        prioritization=config["prioritization"],
        prioritization_params={"temperature": config["temperature"], "k": config["topk_k"]},
        duplicate_check=config["buffer_duplicate_check"],
    )

    create_train_state, train_step, sfl_search = variant.setup(
        config,
        env,
        eval_env,
        env_params,
        sample_random_level,
        level_sampler,
        mutate_level,
        jax,
        jnp,
        chex,
        Tuple,
        TrainState,
        UpdateState,
        ActorCritic,
        optax,
        compute_gae,
        sample_trajectories_rnn,
        update_actor_critic_rnn,
        evaluate_rnn,
        compute_max_returns,
    )

    def eval(rng: chex.PRNGKey, train_state: TrainState):
        rng, rng_reset = jax.random.split(rng)
        levels = Level.load_prefabs(config["eval_levels"])
        num_levels = len(config["eval_levels"])
        init_obs, init_env_state = jax.vmap(eval_env.reset_to_level, in_axes=(0, 0, None))(
            jax.random.split(rng_reset, num_levels), levels, env_params
        )
        states, rewards, episode_lengths = evaluate_rnn(
            rng,
            eval_env,
            env_params,
            train_state,
            ActorCritic.initialize_carry((num_levels,)),
            init_obs,
            init_env_state,
            env_params.max_steps_in_episode,
        )
        mask = jnp.arange(env_params.max_steps_in_episode)[..., None] < episode_lengths
        cum_rewards = (rewards * mask).sum(axis=0)
        return states, cum_rewards, episode_lengths

    @jax.jit
    def train_and_eval_step(runner_state, _):
        (rng, train_state), metrics = jax.lax.scan(train_step, runner_state, None, config["eval_freq"])
        rng, rng_eval = jax.random.split(rng)
        states, cum_rewards, episode_lengths = jax.vmap(eval, (0, None))(
            jax.random.split(rng_eval, config["eval_num_attempts"]), train_state
        )
        eval_solve_rates = jnp.where(cum_rewards > 0, 1.0, 0.0).mean(axis=0)
        eval_returns = cum_rewards.mean(axis=0)
        states, episode_lengths = jax.tree_util.tree_map(lambda x: x[0], (states, episode_lengths))
        images = jax.vmap(jax.vmap(env_renderer.render_state, (0, None)), (0, None))(
            states, env_params
        )
        frames = images.transpose(0, 1, 4, 2, 3)
        metrics["update_count"] = (
            train_state.num_dr_updates + train_state.num_replay_updates + train_state.num_mutation_updates
        )
        metrics["eval_returns"] = eval_returns
        metrics["eval_solve_rates"] = eval_solve_rates
        metrics["eval_ep_lengths"] = episode_lengths
        metrics["eval_animation"] = (frames, episode_lengths)
        metrics["dr_levels"] = jax.vmap(env_renderer.render_level, (0, None))(
            train_state.dr_last_level_batch, env_params
        )
        metrics["replay_levels"] = jax.vmap(env_renderer.render_level, (0, None))(
            train_state.replay_last_level_batch, env_params
        )
        metrics["mutation_levels"] = jax.vmap(env_renderer.render_level, (0, None))(
            train_state.mutation_last_level_batch, env_params
        )
        highest_scoring_level = level_sampler.get_levels(
            train_state.sampler, train_state.sampler["scores"].argmax()
        )
        highest_weighted_level = level_sampler.get_levels(
            train_state.sampler, level_sampler.level_weights(train_state.sampler).argmax()
        )
        metrics["highest_scoring_level"] = env_renderer.render_level(highest_scoring_level, env_params)
        metrics["highest_weighted_level"] = env_renderer.render_level(highest_weighted_level, env_params)
        return (rng, train_state), metrics

    def eval_checkpoint(og_config):
        rng_init, rng_eval = jax.random.split(jax.random.PRNGKey(10000))

        def load(rng_init, checkpoint_directory: str):
            with open(os.path.join(checkpoint_directory, "config.json")) as handle:
                loaded_config = json.load(handle)
            checkpoint_manager = ocp.CheckpointManager(
                os.path.join(os.getcwd(), checkpoint_directory, "models"),
                item_handlers=ocp.StandardCheckpointHandler(),
            )
            train_state_og: TrainState = create_train_state(rng_init)
            step = (
                checkpoint_manager.latest_step()
                if og_config["checkpoint_to_eval"] == -1
                else og_config["checkpoint_to_eval"]
            )
            loaded_checkpoint = checkpoint_manager.restore(step)
            params = loaded_checkpoint["params"]
            train_state = train_state_og.replace(params=params)
            return train_state, loaded_config

        train_state, loaded_config = load(rng_init, og_config["checkpoint_directory"])
        states, cum_rewards, episode_lengths = jax.vmap(eval, (0, None))(
            jax.random.split(rng_eval, og_config["eval_num_attempts"]), train_state
        )
        save_loc = og_config["checkpoint_directory"].replace("checkpoints", "results")
        os.makedirs(save_loc, exist_ok=True)
        np.savez_compressed(
            os.path.join(save_loc, "results.npz"),
            states=np.asarray(states),
            cum_rewards=np.asarray(cum_rewards),
            episode_lengths=np.asarray(episode_lengths),
            levels=loaded_config["eval_levels"],
        )
        return states, cum_rewards, episode_lengths

    if config["mode"] == "eval":
        return eval_checkpoint(config)

    rng = jax.random.PRNGKey(config["seed"])
    rng_init, rng_train = jax.random.split(rng)
    train_state = create_train_state(rng_init)
    runner_state = (rng_train, train_state)

    if config["checkpoint_save_interval"] > 0:
        checkpoint_manager = setup_checkpointing(config, train_state, env, env_params)

    if sfl_search is None:
        for eval_step in range(config["num_updates"] // config["eval_freq"]):
            start_time = time.time()
            runner_state, metrics = train_and_eval_step(runner_state, None)
            metrics["time_delta"] = time.time() - start_time
            log_eval(metrics, train_state_to_log_dict(runner_state[1], level_sampler))
            if config["checkpoint_save_interval"] > 0:
                checkpoint_manager.save(eval_step, args=ocp.args.StandardSave(runner_state[1]))
                checkpoint_manager.wait_until_finished()
    else:
        every = max(1, config["sfl_period"] // config["eval_freq"])
        for eval_step in range(config["num_updates"] // config["eval_freq"]):
            start_time = time.time()
            if eval_step % every == 0:
                rng_search, runner_state = jax.random.split(runner_state[0])[0], runner_state
                new_state, p = sfl_search(rng_search, runner_state[1])
                runner_state = (runner_state[0], new_state)
                print(
                    f"поиск: решаемых {(p > 0).mean():.3f}, обучаемых {((p > 0) & (p < 1)).mean():.3f}, "
                    f"лучший p(1-p) {(p * (1 - p)).max():.3f}",
                    flush=True,
                )
            runner_state, metrics = train_and_eval_step(runner_state, None)
            metrics["time_delta"] = time.time() - start_time
            log_eval(metrics, train_state_to_log_dict(runner_state[1], level_sampler))
            if config["checkpoint_save_interval"] > 0:
                checkpoint_manager.save(eval_step, args=ocp.args.StandardSave(runner_state[1]))
                checkpoint_manager.wait_until_finished()
    return runner_state[1]
