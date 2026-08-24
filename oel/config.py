"""Фиксированные конфигурации локальных экспериментов."""

BASE = {
    "seed": 0,
    "num_updates": 500,
    "num_steps": 256,
    "num_train_envs": 32,
    "num_minibatches": 1,
    "epoch_ppo": 5,
    "lr": 1e-4,
    "max_grad_norm": 0.5,
    "gamma": 0.995,
    "gae_lambda": 0.98,
    "clip_eps": 0.2,
    "entropy_coeff": 1e-3,
    "critic_coeff": 0.5,
    "agent_view_size": 5,
    "n_walls": 25,
    "eval_freq": 25,
    "eval_num_attempts": 10,
    "validation_size": 64,
    "validation_seed": 10000,
    "eval_levels": [
        "SixteenRooms",
        "SixteenRooms2",
        "Labyrinth",
        "LabyrinthFlipped",
        "Labyrinth2",
        "StandardMaze",
        "StandardMaze2",
        "StandardMaze3",
    ],
}

SFL_MAC = {
    **BASE,
    "name": "sfl_mac_seed0",
    "method": "sfl",
    "sfl_num_levels": 128,
    "sfl_buffer_size": 64,
    "sfl_search_interval": 50,
    "sfl_attempts": 5,
    "sfl_search_batch_size": 32,
    "sfl_replay_fraction": 0.5,
}

SFL_SMOKE = {
    **SFL_MAC,
    "name": "sfl_smoke_seed0",
    "num_updates": 4,
    "eval_freq": 1,
    "eval_num_attempts": 2,
    "validation_size": 8,
    "sfl_num_levels": 8,
    "sfl_buffer_size": 4,
    "sfl_search_interval": 1,
    "sfl_attempts": 2,
    "sfl_search_batch_size": 4,
}

ACCEL = {
    **BASE,
    "method": "accel",
    "level_buffer_capacity": 4000,
    "replay_prob": 0.8,
    "staleness_coeff": 0.3,
    "temperature": 0.3,
    "minimum_fill_ratio": 0.5,
    "num_edits": 5,
    "predictor_lr": 1e-3,
}

ACCEL_MAXMC_MAC = {**ACCEL, "name": "accel_maxmc_mac_seed0", "score": "maxmc"}
ACCEL_FIXED_MAC = {**ACCEL, "name": "accel_fixed_predictor_mac_seed0", "score": "fixed"}
ACCEL_CNN_MAC = {**ACCEL, "name": "accel_cnn_predictor_mac_seed0", "score": "cnn"}
ACCEL_RESNET_MAC = {**ACCEL, "name": "accel_resnet_predictor_mac_seed0", "score": "resnet"}
ACCEL_TRACED_MAC = {
    **ACCEL,
    "name": "accel_traced_mac_seed0",
    "score": "traced",
    "transition_weight": 1.0,
}
ACCEL_TRACED_COLEARN_MAC = {
    **ACCEL_TRACED_MAC,
    "name": "accel_traced_colearn_mac_seed0",
    "score": "traced_colearn",
    "colearnability_weight": 1.0,
}

ACCEL_FIXED_SMOKE = {
    **ACCEL_FIXED_MAC,
    "name": "accel_fixed_predictor_smoke_seed0",
    "num_updates": 4,
    "eval_freq": 1,
    "eval_num_attempts": 2,
    "validation_size": 8,
    "level_buffer_capacity": 64,
    "minimum_fill_ratio": 0.5,
}

ACCEL_CNN_SMOKE = {**ACCEL_FIXED_SMOKE, "name": "accel_cnn_predictor_smoke_seed0", "score": "cnn"}
ACCEL_RESNET_SMOKE = {**ACCEL_FIXED_SMOKE, "name": "accel_resnet_predictor_smoke_seed0", "score": "resnet"}
ACCEL_MAXMC_SMOKE = {**ACCEL_FIXED_SMOKE, "name": "accel_maxmc_smoke_seed0", "score": "maxmc"}
ACCEL_TRACED_SMOKE = {
    **ACCEL_TRACED_MAC,
    "name": "accel_traced_smoke_seed0",
    "num_updates": 6,
    "eval_freq": 1,
    "eval_num_attempts": 2,
    "validation_size": 8,
    "level_buffer_capacity": 64,
    "minimum_fill_ratio": 0.5,
    "replay_prob": 1.0,
}
ACCEL_TRACED_COLEARN_SMOKE = {
    **ACCEL_TRACED_SMOKE,
    "name": "accel_traced_colearn_smoke_seed0",
    "score": "traced_colearn",
    "colearnability_weight": 1.0,
}

FULL = {
    "num_updates": 30_000,
    "eval_freq": 500,
    "eval_num_attempts": 30,
}

ACCEL_MAXMC_A100 = {
    **ACCEL_MAXMC_MAC,
    **FULL,
    "name": "accel_maxmc_a100_seed0",
    "diagnostic_transition": True,
}
ACCEL_FIXED_A100 = {
    **ACCEL_FIXED_MAC,
    **FULL,
    "name": "accel_fixed_predictor_a100_seed0",
}
ACCEL_CNN_A100 = {
    **ACCEL_CNN_MAC,
    **FULL,
    "name": "accel_cnn_predictor_a100_seed0",
}
SFL_A100 = {
    **SFL_MAC,
    **FULL,
    "name": "sfl_a100_seed0",
}

A100_RUNS = (ACCEL_MAXMC_A100, ACCEL_FIXED_A100, ACCEL_CNN_A100, SFL_A100)
