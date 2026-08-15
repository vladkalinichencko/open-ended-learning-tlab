"""Launcher for the three JaxUED baselines with the student config left untouched.

Runs the upstream scripts from ext/jaxued, but with cwd = this project, so that
checkpoints land in ./checkpoints/<run_name>/<seed> and eval results in ./results/.

    python baseline.py --method accel --seed 0
    python baseline.py --method accel --seed 0 --mode eval
"""

import argparse
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).parent.resolve()
JAXUED = ROOT / "ext" / "jaxued"

SCRIPT = {"dr": "examples/maze_dr.py", "plr": "examples/maze_plr.py", "accel": "examples/maze_plr.py"}
FLAGS = {"dr": [], "plr": [], "accel": ["--use_accel"]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--method", choices=SCRIPT, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--mode", choices=["train", "eval"], default="train")
    p.add_argument("--project", default="tlab-ued")
    p.add_argument("--run-name", default=None)
    p.add_argument("rest", nargs="*", help="extra flags forwarded upstream (teacher side only)")
    args = p.parse_args()

    run_name = args.run_name or args.method
    cmd = [sys.executable, str(JAXUED / SCRIPT[args.method]),
           "--seed", str(args.seed), "--run_name", run_name, "--project", args.project]

    if args.mode == "train":
        cmd += FLAGS[args.method] + ["--checkpoint_save_interval", "17"]
    else:
        cmd += ["--mode", "eval",
                "--checkpoint_directory", f"checkpoints/{run_name}/{args.seed}",
                "--checkpoint_to_eval", "-1"]

    cmd += args.rest
    print(" ".join(cmd))
    sys.exit(subprocess.call(cmd, cwd=ROOT))


if __name__ == "__main__":
    main()
