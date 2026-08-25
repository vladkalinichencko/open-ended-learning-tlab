"""Копия teacher.py: create_train_state, train_step."""

import scores
from teacher_stats import success_rate

def compute_score(config, dones, values, max_returns, advantages, rewards=None, prior_p=None):
    return scores.SCORES[config["score_function"]](
        config, dones, values, max_returns, advantages, rewards, prior_p)

def setup(
    config, env, eval_env, env_params, sample_random_level, level_sampler, mutate_level,
    jax, jnp, chex, Tuple, TrainState, UpdateState, ActorCritic, optax,
    compute_gae, sample_trajectories_rnn, update_actor_critic_rnn, evaluate_rnn,
    compute_max_returns,
):
        @jax.jit
        def create_train_state(rng) -> TrainState:
            # Creates the train state
            def linear_schedule(count):
                frac = (
                    1.0
                    - (count // (config["num_minibatches"] * config["epoch_ppo"]))
                    / config["num_updates"]
                )
                return config["lr"] * frac
            obs, _ = env.reset_to_level(rng, sample_random_level(rng), env_params)
            obs = jax.tree_util.tree_map(
                lambda x: jnp.repeat(jnp.repeat(x[None, ...], config["num_train_envs"], axis=0)[None, ...], 256, axis=0),
                obs,
            )
            init_x = (obs, jnp.zeros((256, config["num_train_envs"])))
            network = ActorCritic(env.action_space(env_params).n)
            network_params = network.init(rng, init_x, ActorCritic.initialize_carry((config["num_train_envs"],)))
            tx = optax.chain(
                optax.clip_by_global_norm(config["max_grad_norm"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
                # optax.adam(learning_rate=config["lr"], eps=1e-5),
            )
            pholder_level = sample_random_level(jax.random.PRNGKey(0))
            sampler = level_sampler.initialize(pholder_level, {"max_return": -jnp.inf, "p_ema": 0.0})
            pholder_level_batch = jax.tree_util.tree_map(lambda x: jnp.array([x]).repeat(config["num_train_envs"], axis=0), pholder_level)
            return TrainState.create(
                apply_fn=network.apply,
                params=network_params,
                tx=tx,
                sampler=sampler,
                update_state=0,
                num_dr_updates=0,
                num_replay_updates=0,
                num_mutation_updates=0,
                dr_last_level_batch=pholder_level_batch,
                replay_last_level_batch=pholder_level_batch,
                mutation_last_level_batch=pholder_level_batch,
            )

        def train_step(carry: Tuple[chex.PRNGKey, TrainState], _):
            """
                This is the main training loop. It basically calls either `on_new_levels`, `on_replay_levels`, or `on_mutate_levels` at every step.
            """
            def on_new_levels(rng: chex.PRNGKey, train_state: TrainState):
                """
                    Samples new (randomly-generated) levels and evaluates the policy on these. It also then adds the levels to the level buffer if they have high-enough scores.
                    The agent is updated on these trajectories iff `config["exploratory_grad_updates"]` is True.
                """
                sampler = train_state.sampler

                # Reset
                rng, rng_levels, rng_reset = jax.random.split(rng, 3)
                new_levels = jax.vmap(sample_random_level)(jax.random.split(rng_levels, config["num_train_envs"]))
                init_obs, init_env_state = jax.vmap(env.reset_to_level, in_axes=(0, 0, None))(jax.random.split(rng_reset, config["num_train_envs"]), new_levels, env_params)
                # Rollout
                (
                    (rng, train_state, hstate, last_obs, last_env_state, last_value),
                    (obs, actions, rewards, dones, log_probs, values, info),
                ) = sample_trajectories_rnn(
                    rng,
                    env,
                    env_params,
                    train_state,
                    ActorCritic.initialize_carry((config["num_train_envs"],)),
                    init_obs,
                    init_env_state,
                    config["num_train_envs"],
                    config["num_steps"],
                )
                advantages, targets = compute_gae(config["gamma"], config["gae_lambda"], last_value, values, rewards, dones)
                max_returns = compute_max_returns(dones, rewards)
                p_now = success_rate(dones, rewards)
                scores = compute_score(config, dones, values, max_returns, advantages, rewards,
                                       -jnp.ones_like(p_now))
                sampler, _ = level_sampler.insert_batch(sampler, new_levels, scores,
                                                        {"max_return": max_returns, "p_ema": p_now})

                # Update: train_state only modified if exploratory_grad_updates is on
                (rng, train_state), losses = update_actor_critic_rnn(
                    rng,
                    train_state,
                    ActorCritic.initialize_carry((config["num_train_envs"],)),
                    (obs, actions, dones, log_probs, values, targets, advantages),
                    config["num_train_envs"],
                    config["num_steps"],
                    config["num_minibatches"],
                    config["epoch_ppo"],
                    config["clip_eps"],
                    config["entropy_coeff"],
                    config["critic_coeff"],
                    update_grad=config["exploratory_grad_updates"],
                )

                metrics = {
                    "losses": jax.tree_util.tree_map(lambda x: x.mean(), losses),
                    "mean_num_blocks": new_levels.wall_map.sum() / config["num_train_envs"],
                }

                train_state = train_state.replace(
                    sampler=sampler,
                    update_state=UpdateState.DR,
                    num_dr_updates=train_state.num_dr_updates + 1,
                    dr_last_level_batch=new_levels,
                )
                return (rng, train_state), metrics

            def on_replay_levels(rng: chex.PRNGKey, train_state: TrainState):
                """
                    This samples levels from the level buffer, and updates the policy on them.
                """
                sampler = train_state.sampler

                # Collect trajectories on replay levels
                rng, rng_levels, rng_reset = jax.random.split(rng, 3)
                sampler, (level_inds, levels) = level_sampler.sample_replay_levels(sampler, rng_levels, config["num_train_envs"])
                init_obs, init_env_state = jax.vmap(env.reset_to_level, in_axes=(0, 0, None))(jax.random.split(rng_reset, config["num_train_envs"]), levels, env_params)
                (
                    (rng, train_state, hstate, last_obs, last_env_state, last_value),
                    (obs, actions, rewards, dones, log_probs, values, info),
                ) = sample_trajectories_rnn(
                    rng,
                    env,
                    env_params,
                    train_state,
                    ActorCritic.initialize_carry((config["num_train_envs"],)),
                    init_obs,
                    init_env_state,
                    config["num_train_envs"],
                    config["num_steps"],
                )
                advantages, targets = compute_gae(config["gamma"], config["gae_lambda"], last_value, values, rewards, dones)
                max_returns = jnp.maximum(level_sampler.get_levels_extra(sampler, level_inds)["max_return"], compute_max_returns(dones, rewards))
                prior_p = level_sampler.get_levels_extra(sampler, level_inds)["p_ema"]
                scores = compute_score(config, dones, values, max_returns, advantages, rewards,
                                       prior_p)
                p_now = (1 - config["p_decay"]) * prior_p + config["p_decay"] * success_rate(dones, rewards)
                sampler = level_sampler.update_batch(sampler, level_inds, scores,
                                                     {"max_return": max_returns, "p_ema": p_now})

                # Update the policy using trajectories collected from replay levels
                (rng, train_state), losses = update_actor_critic_rnn(
                    rng,
                    train_state,
                    ActorCritic.initialize_carry((config["num_train_envs"],)),
                    (obs, actions, dones, log_probs, values, targets, advantages),
                    config["num_train_envs"],
                    config["num_steps"],
                    config["num_minibatches"],
                    config["epoch_ppo"],
                    config["clip_eps"],
                    config["entropy_coeff"],
                    config["critic_coeff"],
                    update_grad=True,
                )

                metrics = {
                    "losses": jax.tree_util.tree_map(lambda x: x.mean(), losses),
                    "mean_num_blocks": levels.wall_map.sum() / config["num_train_envs"],
                }

                train_state = train_state.replace(
                    sampler=sampler,
                    update_state=UpdateState.REPLAY,
                    num_replay_updates=train_state.num_replay_updates + 1,
                    replay_last_level_batch=levels,
                )
                return (rng, train_state), metrics

            def on_mutate_levels(rng: chex.PRNGKey, train_state: TrainState):
                """
                    This mutates the previous batch of replay levels and potentially adds them to the level buffer.
                    This also updates the policy iff `config["exploratory_grad_updates"]` is True.
                """
                sampler = train_state.sampler
                rng, rng_mutate, rng_reset = jax.random.split(rng, 3)

                # mutate
                parent_levels = train_state.replay_last_level_batch
                child_levels = jax.vmap(mutate_level, (0, 0, None))(jax.random.split(rng_mutate, config["num_train_envs"]), parent_levels, config["num_edits"])
                init_obs, init_env_state = jax.vmap(env.reset_to_level, in_axes=(0, 0, None))(jax.random.split(rng_reset, config["num_train_envs"]), child_levels, env_params)

                # rollout
                (
                    (rng, train_state, hstate, last_obs, last_env_state, last_value),
                    (obs, actions, rewards, dones, log_probs, values, info),
                ) = sample_trajectories_rnn(
                    rng,
                    env,
                    env_params,
                    train_state,
                    ActorCritic.initialize_carry((config["num_train_envs"],)),
                    init_obs,
                    init_env_state,
                    config["num_train_envs"],
                    config["num_steps"],
                )
                advantages, targets = compute_gae(config["gamma"], config["gae_lambda"], last_value, values, rewards, dones)
                max_returns = compute_max_returns(dones, rewards)
                p_now = success_rate(dones, rewards)
                scores = compute_score(config, dones, values, max_returns, advantages, rewards,
                                       -jnp.ones_like(p_now))
                sampler, _ = level_sampler.insert_batch(sampler, child_levels, scores,
                                                        {"max_return": max_returns, "p_ema": p_now})

                # Update: train_state only modified if exploratory_grad_updates is on
                (rng, train_state), losses = update_actor_critic_rnn(
                    rng,
                    train_state,
                    ActorCritic.initialize_carry((config["num_train_envs"],)),
                    (obs, actions, dones, log_probs, values, targets, advantages),
                    config["num_train_envs"],
                    config["num_steps"],
                    config["num_minibatches"],
                    config["epoch_ppo"],
                    config["clip_eps"],
                    config["entropy_coeff"],
                    config["critic_coeff"],
                    update_grad=config["exploratory_grad_updates"],
                )

                metrics = {
                    "losses": jax.tree_util.tree_map(lambda x: x.mean(), losses),
                    "mean_num_blocks": child_levels.wall_map.sum() / config["num_train_envs"],
                }

                train_state = train_state.replace(
                    sampler=sampler,
                    update_state=UpdateState.DR,
                    num_mutation_updates=train_state.num_mutation_updates + 1,
                    mutation_last_level_batch=child_levels,
                )
                return (rng, train_state), metrics

            rng, train_state = carry
            rng, rng_replay = jax.random.split(rng)

            # The train step makes a decision on which branch to take, either on_new, on_replay or on_mutate.
            # on_mutate is only called if the replay branch has been taken before (as it uses `train_state.update_state`).
            if config["use_accel"]:
                s = train_state.update_state
                branch = (1 - s) * level_sampler.sample_replay_decision(train_state.sampler, rng_replay) + 2 * s
            else:
                branch = level_sampler.sample_replay_decision(train_state.sampler, rng_replay).astype(int)

            return jax.lax.switch(
                branch,
                [
                    on_new_levels,
                    on_replay_levels,
                    on_mutate_levels,
                ],
                rng, train_state
            )
        return create_train_state, train_step, locals().get("sfl_search")
