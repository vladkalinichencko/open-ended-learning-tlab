"""Export an oel policy in the checkpoint layout expected by JaxUED eval."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "ext" / "jaxued" / "src"))

import jax
import orbax.checkpoint as ocp
from flax import serialization
from jaxued.environments import Maze
from jaxued.environments.maze import make_level_generator
from jaxued.wrappers import AutoReplayWrapper

from oel.training import create_student


def export_checkpoint(run_dir: Path, destination: Path) -> None:
    if destination.exists():
        raise FileExistsError(destination)

    config = json.loads((run_dir / "config.json").read_text())
    env = Maze(13, 13, agent_view_size=config["agent_view_size"], normalize_obs=True)
    student = create_student(
        config,
        AutoReplayWrapper(env),
        make_level_generator(13, 13, config["n_walls"]),
    )
    params = serialization.from_bytes(
        student.params,
        (run_dir / "checkpoint.msgpack").read_bytes(),
    )
    if jax.tree_util.tree_structure(params) != jax.tree_util.tree_structure(student.params):
        raise ValueError("checkpoint does not match the original ActorCritic")
    for loaded, expected in zip(
        jax.tree_util.tree_leaves(params),
        jax.tree_util.tree_leaves(student.params),
    ):
        if loaded.shape != expected.shape:
            raise ValueError(f"parameter shape {loaded.shape} != {expected.shape}")

    destination.mkdir(parents=True)
    config["exported_from"] = str(run_dir)
    (destination / "config.json").write_text(json.dumps(config, indent=2))

    manager = ocp.CheckpointManager(
        (destination / "models").resolve(),
        item_handlers=ocp.StandardCheckpointHandler(),
    )
    manager.save(0, args=ocp.args.StandardSave({"params": params}))
    manager.wait_until_finished()
    restored = manager.restore(0)["params"]
    if jax.tree_util.tree_structure(restored) != jax.tree_util.tree_structure(params):
        raise ValueError("Orbax restore changed the parameter tree")

    print(destination)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    export_checkpoint(args.run_dir, args.destination)
