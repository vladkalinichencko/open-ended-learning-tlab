"""PLR с фазой поиска обучаемых уровней; запуск: python sfl.py."""

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "ext" / "jaxued" / "src"))
sys.path.insert(0, str(ROOT / "ext" / "jaxued" / "examples"))

from oel.plr_train import main
from oel.plr_variants import sfl as variant

def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=str, default="JAXUED_TEST")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--checkpoint_directory", type=str, default=None)
    parser.add_argument("--checkpoint_to_eval", type=int, default=-1)
    parser.add_argument("--checkpoint_save_interval", type=int, default=2)
    parser.add_argument("--max_number_of_checkpoints", type=int, default=60)
    parser.add_argument("--eval_freq", type=int, default=250)
    parser.add_argument("--eval_num_attempts", type=int, default=10)
    parser.add_argument(
        "--eval_levels",
        nargs="+",
        default=[
            "SixteenRooms",
            "SixteenRooms2",
            "Labyrinth",
            "LabyrinthFlipped",
            "Labyrinth2",
            "StandardMaze",
            "StandardMaze2",
            "StandardMaze3",
        ],
    )
    group = parser.add_argument_group("Training params")
    group.add_argument("--lr", type=float, default=1e-4)
    group.add_argument("--max_grad_norm", type=float, default=0.5)
    mut_group = group.add_mutually_exclusive_group()
    mut_group.add_argument("--num_updates", type=int, default=30000)
    mut_group.add_argument("--num_env_steps", type=int, default=None)
    group.add_argument("--num_steps", type=int, default=256)
    group.add_argument("--num_train_envs", type=int, default=32)
    group.add_argument("--num_minibatches", type=int, default=1)
    group.add_argument("--gamma", type=float, default=0.995)
    group.add_argument("--epoch_ppo", type=int, default=5)
    group.add_argument("--clip_eps", type=float, default=0.2)
    group.add_argument("--gae_lambda", type=float, default=0.98)
    group.add_argument("--entropy_coeff", type=float, default=1e-3)
    group.add_argument("--critic_coeff", type=float, default=0.5)
    group.add_argument("--score_function", type=str, default="MaxMC", choices=["MaxMC", "pvl"])
    group.add_argument("--exploratory_grad_updates", action=argparse.BooleanOptionalAction, default=False)
    group.add_argument("--level_buffer_capacity", type=int, default=4000)
    group.add_argument("--replay_prob", type=float, default=0.8)
    group.add_argument("--staleness_coeff", type=float, default=0.3)
    group.add_argument("--temperature", type=float, default=0.3)
    group.add_argument("--topk_k", type=int, default=4)
    group.add_argument("--minimum_fill_ratio", type=float, default=0.5)
    group.add_argument("--prioritization", type=str, default="rank", choices=["rank", "topk"])
    group.add_argument("--buffer_duplicate_check", action=argparse.BooleanOptionalAction, default=True)
    group.add_argument(
        "--sfl_num_levels",
        type=int,
        default=512,
        help="сколько случайных уровней перебирать за фазу поиска (в статье 5000)",
    )
    group.add_argument(
        "--sfl_attempts",
        type=int,
        default=8,
        help="сколько независимых эпизодов на уровень: p — это доля, по одному эпизоду её не измерить",
    )
    group.add_argument(
        "--sfl_period",
        type=int,
        default=250,
        help="как часто пересобирать буфер, в апдейтах (в статье 50)",
    )
    group.add_argument("--use_accel", action=argparse.BooleanOptionalAction, default=False)
    group.add_argument("--num_edits", type=int, default=5)
    group.add_argument("--agent_view_size", type=int, default=5)
    group.add_argument("--n_walls", type=int, default=25)
    return parser

if __name__ == "__main__":
    parser = build_parser()
    config = vars(parser.parse_args())
    if config["num_env_steps"] is not None:
        config["num_updates"] = config["num_env_steps"] // (
            config["num_train_envs"] * config["num_steps"]
        )
    config["group_name"] = "".join(
        str(config[key])
        for key in sorted(action.dest for action in parser._action_groups[2]._group_actions)
    )
    if config["mode"] == "eval":
        os.environ["WANDB_MODE"] = "disabled"
    main(config, variant, project=config["project"])
