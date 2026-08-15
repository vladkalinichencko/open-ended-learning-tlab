"""Aggregate `results/<run>/<seed>/results.npz` into a solve-rate table over seeds.

    python solve_rate.py results/dr results/plr results/accel results/ours --plot
"""

import argparse
import pathlib

import numpy as np


def load(run_dir):
    """-> (levels, solve_rates of shape [n_seeds, n_levels])."""
    levels, rates = None, []
    for f in sorted(pathlib.Path(run_dir).glob("*/results.npz")):
        d = np.load(f, allow_pickle=True)
        levels = [str(x) for x in d["levels"]]
        rates.append((d["cum_rewards"] > 0).mean(axis=0))  # (n_levels,)
    if not rates:
        raise SystemExit(f"no results.npz under {run_dir}")
    return levels, np.stack(rates)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+")
    p.add_argument("--plot", action="store_true")
    args = p.parse_args()

    table = {pathlib.Path(r).name: load(r) for r in args.runs}
    levels = next(iter(table.values()))[0]

    head = f"{'level':<18}" + "".join(f"{n:>18}" for n in table)
    print(head)
    for i, lvl in enumerate(levels):
        line = f"{lvl:<18}"
        for _, rates in table.values():
            line += f"{rates[:, i].mean():>12.2f} ±{rates[:, i].std():.2f}"
        print(line)
    line = f"{'MEAN':<18}"
    for _, rates in table.values():
        m = rates.mean(axis=1)
        line += f"{m.mean():>12.2f} ±{m.std():.2f}"
    print("-" * len(head))
    print(line)
    print(f"\nseeds: " + ", ".join(f"{n}={len(r)}" for n, (_, r) in table.items()))

    if args.plot:
        import matplotlib.pyplot as plt

        x = np.arange(len(levels))
        w = 0.8 / len(table)
        fig, ax = plt.subplots(figsize=(10, 4))
        for k, (name, (_, rates)) in enumerate(table.items()):
            ax.bar(x + k * w, rates.mean(0), w, yerr=rates.std(0), label=name, capsize=2)
        ax.set_xticks(x + 0.4 - w / 2, levels, rotation=30, ha="right")
        ax.set_ylabel("solve rate")
        ax.legend()
        fig.tight_layout()
        out = pathlib.Path("runs") / "solve_rate.png"
        fig.savefig(out, dpi=150)
        print(f"-> {out}")


if __name__ == "__main__":
    main()
