"""Final A100 teacher comparison: generated validation against human held-out maps."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

METHODS = [
    ("MaxMC", 0.994, 0.100, "#64748b"),
    ("Fixed", 0.942, 0.150, "#0f766e"),
    ("SFL", 0.975, 0.125, "#c2410c"),
    ("CNN", 0.952, 0.375, "#1d4ed8"),
]
LEVELS = [
    ("StandardMaze3", 1.0), ("SixteenRooms", 0.8), ("Labyrinth2", 0.8),
    ("StandardMaze", 0.4), ("Labyrinth", 0.0), ("LabyrinthFlipped", 0.0),
]


def main():
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))

    positions = range(len(METHODS))
    axes[0].bar([p - 0.2 for p in positions], [m[1] for m in METHODS], width=0.4,
                color="#cbd5e1", label="сгенерированная validation")
    axes[0].bar([p + 0.2 for p in positions], [m[2] for m in METHODS], width=0.4,
                color=[m[3] for m in METHODS], label="человеческие held-out карты")
    for position, method in zip(positions, METHODS):
        axes[0].text(position - 0.2, method[1], f"{method[1]:.3f}", ha="center",
                     va="bottom", fontsize=9, color="#475569")
        axes[0].text(position + 0.2, method[2], f"{method[2]:.3f}", ha="center",
                     va="bottom", fontsize=9)
    axes[0].set_xticks(list(positions))
    axes[0].set_xticklabels([m[0] for m in METHODS])
    axes[0].set_ylabel("solve rate")
    axes[0].set_ylim(0, 1.12)
    axes[0].set_title("Один сид, 5000 updates, пять попыток на уровень")
    axes[0].legend(frameon=False, fontsize=9, loc="upper left")
    axes[0].grid(alpha=0.25, axis="y")

    names = [name for name, _ in LEVELS]
    values = [value for _, value in LEVELS]
    colors = ["#1d4ed8" if value > 0 else "#e2e8f0" for value in values]
    bars = axes[1].barh(names[::-1], values[::-1], color=colors[::-1])
    for bar, value in zip(bars, values[::-1]):
        axes[1].text(value + 0.02, bar.get_y() + bar.get_height() / 2,
                     f"{value:.1f}", va="center", fontsize=9)
    axes[1].set_xlim(0, 1.15)
    axes[1].set_xlabel("solve rate CNN")
    axes[1].set_title("Held-out карты по одной: четыре из восьми ненулевые")
    axes[1].grid(alpha=0.25, axis="x")

    figure.tight_layout()
    out = Path("assets/teacher-comparison.png")
    figure.savefig(out, dpi=160)
    print(out)


if __name__ == "__main__":
    main()
