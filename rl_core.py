"""Тонкий re-export PPO-ядра из jaxued.examples.maze_plr.
"""

import sys
import pathlib

_ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT / "ext" / "jaxued" / "src"))
sys.path.insert(0, str(_ROOT / "ext" / "jaxued" / "examples"))

from maze_plr import (
    UpdateState,
    TrainState,
    compute_gae,
    sample_trajectories_rnn,
    evaluate_rnn,
    update_actor_critic_rnn,
    ActorCritic,
    setup_checkpointing,
    train_state_to_log_dict,
)

