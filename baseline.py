"""Launcher for the three JaxUED baselines with the student config left untouched.

Runs the upstream scripts from ext/jaxued, but with cwd = this project, so that
checkpoints land in ./checkpoints/<run_name>/<seed> and eval results in ./results/.

Logging goes to a local MLflow store (sqlite:///mlflow.db): wandb_shim/ is put first
on PYTHONPATH, so upstream `import wandb` picks up our shim instead of the real one.

    python baseline.py --method accel --seed 0
    python baseline.py --method accel --seed 0 --mode eval
    mlflow ui --backend-store-uri sqlite:///mlflow.db
"""

import argparse
import os
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).parent.resolve()
JAXUED = ROOT / "ext" / "jaxued"
SHIM = ROOT / "wandb_shim"

SCRIPT = {"dr": "examples/maze_dr.py", "plr": "examples/maze_plr.py", "accel": "examples/maze_plr.py"}
FLAGS = {"dr": [], "plr": [], "accel": ["--use_accel"]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--method", choices=SCRIPT, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--mode", choices=["train", "eval"], default="train")
    p.add_argument("--project", default="tlab-ued")
    p.add_argument("--run-name", default=None)
    args, rest = p.parse_known_args()  # rest goes upstream verbatim (teacher side only)

    run_name = args.run_name or args.method
    cmd = [sys.executable, str(JAXUED / SCRIPT[args.method]),
           "--seed", str(args.seed), "--run_name", run_name, "--project", args.project]

    if args.mode == "train":
        cmd += FLAGS[args.method] + ["--checkpoint_save_interval", "17"]
    else:
        cmd += ["--mode", "eval",
                "--checkpoint_directory", f"checkpoints/{run_name}/{args.seed}",
                "--checkpoint_to_eval", "-1"]

    cmd += rest
    print(" ".join(cmd))

    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(SHIM), env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
    sys.exit(subprocess.call(cmd, cwd=ROOT, env=env))


if __name__ == "__main__":
    main()
