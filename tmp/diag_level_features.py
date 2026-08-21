"""На каких уровнях студент справляется, а на каких нет — и чем эти уровни отличаются.

Не «какой solve rate», а «что в уровне делает его непроходимым для агента». Считаем
структурные признаки каждого dev-уровня и сопоставляем с измеренным solve rate.

Признаки выбраны так, чтобы каждый отвечал на конкретный вопрос про агента, который
видит только 5x5 вокруг себя:

  длина кратчайшего пути   сколько шагов надо продержаться без обратной связи
  тупики                   сколько раз можно уйти не туда и потерять время
  развилки                 сколько раз надо принимать решение
  длина коридора           самый длинный участок пути без единой развилки:
                           столько шагов агент идёт вслепую, не имея выбора,
                           и настолько же дорого стоит одна ошибка на входе
  извилистость             отношение пути к прямой линии: насколько нельзя идти "на глаз"
  плотность стен           общая заполненность

    python tmp/diag_level_features.py
"""

import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import levels  # noqa: E402

DEV = ["SixteenRooms", "SixteenRooms2", "Labyrinth", "LabyrinthFlipped",
       "Labyrinth2", "StandardMaze", "StandardMaze2", "StandardMaze3"]

# измерено на живом прогоне ACCEL, апдейт 7750, 10 попыток на уровень
SOLVE = {"SixteenRooms": 0.50, "SixteenRooms2": 0.50, "Labyrinth": 0.00,
         "LabyrinthFlipped": 0.00, "Labyrinth2": 0.00, "StandardMaze": 0.00,
         "StandardMaze2": 0.10, "StandardMaze3": 0.00}


def features(name):
    return levels.features(*levels.from_prefab(name))


def main():
    rows = {}
    for name in DEV:
        rows[name] = features(name)

    keys = list(next(iter(rows.values())))
    print(f"{'уровень':<18}{'solve':>7}" + "".join(f"{k:>18}" for k in keys))
    for name in sorted(DEV, key=lambda n: -SOLVE[n]):
        line = f"{name:<18}{SOLVE[name]:>7.2f}"
        for k in keys:
            line += f"{rows[name][k]:>18}"
        print(line)

    solved = [n for n in DEV if SOLVE[n] > 0]
    failed = [n for n in DEV if SOLVE[n] == 0]
    print(f"\nрешает ({len(solved)}) против не решает ({len(failed)}):")
    print(f"{'признак':>20}{'решает':>12}{'не решает':>12}{'разрыв':>10}")
    gaps = {}
    for k in keys:
        a = np.mean([rows[n][k] for n in solved])
        b = np.mean([rows[n][k] for n in failed])
        gaps[k] = (a, b, b - a)
        print(f"{k:>20}{a:>12.2f}{b:>12.2f}{b - a:>+10.2f}")

    pathlib.Path("tmp/level_features.json").write_text(
        json.dumps({"признаки": rows, "solve": SOLVE,
                    "разрывы": {k: list(map(float, v)) for k, v in gaps.items()}},
                   ensure_ascii=False, indent=2))
    print("\n-> tmp/level_features.json")
    draw()


def draw():
    """Картинка: восемь dev-уровней с кратчайшим путём, solve rate и коридором."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 4, figsize=(16, 8.5))
    order = sorted(DEV, key=lambda n: -SOLVE[n])
    for ax, name in zip(axes.ravel(), order):
        walls, agent, goal = levels.from_prefab(name)
        _, path = levels.shortest_path(walls, agent, goal)
        f = levels.features(walls, agent, goal)

        ax.imshow(walls, cmap="Greys", interpolation="nearest")
        if path:
            ax.plot([c for _, c in path], [r for r, _ in path], color="tab:orange", lw=2)
        ax.scatter([agent[1]], [agent[0]], c="tab:blue", s=70, marker="o", zorder=3)
        ax.scatter([goal[1]], [goal[0]], c="tab:green", s=90, marker="*", zorder=3)
        colour = "tab:green" if SOLVE[name] > 0 else "tab:red"
        ax.set_title(f"{name}\nsolve {SOLVE[name]:.2f} | коридор {f['длина коридора']} | "
                     f"развилок {f['развилок']}", color=colour, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Что агент решает (зелёное) и что нет (красное). "
                 "Оранжевое — кратчайший путь, синее — старт, звезда — цель", fontsize=12)
    fig.tight_layout()
    fig.savefig("tmp/dev_levels.png", dpi=130)
    print("-> tmp/dev_levels.png")





if __name__ == "__main__":
    main()
