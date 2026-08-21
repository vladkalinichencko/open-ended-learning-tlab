"""Диагностика teacher'а: что происходит с курикулумом по ходу обучения.

Тянет метрики живого прогона из MLflow и строит четыре среза:

1. Solve rate по каждому dev-уровню отдельно (не среднее): какие уровни студент
   осваивает первыми, какие не осваивает вообще.
2. Буфер уровней: размер, средний и максимальный score, взвешенный score.
   Видно, набирается ли буфер и расходится ли оценка сложности.
3. Сам сигнал score во времени. Если он шумит в начале, teacher первые тысячи
   апдейтов работает по шуму — это ровно та слабость, в которую стоит бить.
4. Соотношение среднего и максимального score: если они не расходятся, буфер
   не отличает уровни друг от друга.

    python tmp/diag_teacher.py --run accel
"""

import argparse
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow


def history(client, run_id, key):
    pts = client.get_metric_history(run_id, key)
    return [p.step for p in pts], [p.value for p in pts]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="accel")
    p.add_argument("--experiment", default="tlab-ued")
    args = p.parse_args()

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    runs = mlflow.search_runs(experiment_names=[args.experiment])
    row = runs[runs["tags.mlflow.runName"] == args.run].iloc[0]
    run_id = row["run_id"]
    client = mlflow.tracking.MlflowClient()

    levels = sorted(c.replace("metrics.solve_rate/", "")
                    for c in runs.columns if c.startswith("metrics.solve_rate/"))
    print(f"прогон {args.run}: {row['status']}, уровней {len(levels)}")

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    ax = axes[0][0]
    for lvl in levels:
        steps, vals = history(client, run_id, f"solve_rate/{lvl}")
        if steps:
            ax.plot(steps, vals, marker="o", ms=3, label=lvl)
    ax.set_title("solve rate по каждому dev-уровню")
    ax.set_xlabel("апдейт"); ax.set_ylabel("доля решённых"); ax.set_ylim(-.02, 1.02)
    ax.legend(fontsize=6, ncol=2); ax.grid(alpha=.3)

    ax = axes[0][1]
    for key in ("level_sampler/size", "level_sampler/episode_count"):
        steps, vals = history(client, run_id, key)
        if steps:
            ax.plot(steps, vals, marker="o", ms=3, label=key.split("/")[-1])
    ax.set_title("буфер уровней"); ax.set_xlabel("апдейт")
    ax.legend(fontsize=8); ax.grid(alpha=.3)

    ax = axes[1][0]
    for key in ("level_sampler/mean_score", "level_sampler/max_score",
                "level_sampler/weighted_score"):
        steps, vals = history(client, run_id, key)
        if steps:
            ax.plot(steps, vals, marker="o", ms=3, label=key.split("/")[-1])
    ax.set_title("score-функция во времени"); ax.set_xlabel("апдейт")
    ax.legend(fontsize=8); ax.grid(alpha=.3)

    ax = axes[1][1]
    s_mean = history(client, run_id, "level_sampler/mean_score")
    s_max = history(client, run_id, "level_sampler/max_score")
    if s_mean[0] and s_max[0]:
        n = min(len(s_mean[1]), len(s_max[1]))
        spread = [mx - mn for mx, mn in zip(s_max[1][:n], s_mean[1][:n])]
        ax.plot(s_mean[0][:n], spread, marker="o", ms=3, color="tab:red")
        ax.set_title("разброс score (max - mean): различает ли буфер уровни")
        ax.set_xlabel("апдейт"); ax.grid(alpha=.3)

    fig.suptitle(f"UED, прогон {args.run}: диагностика teacher'а")
    fig.tight_layout()
    out = pathlib.Path("tmp/teacher_diag.png")
    fig.savefig(out, dpi=140)

    print("\nпоследние значения:")
    for key in ("level_sampler/size", "level_sampler/mean_score",
                "level_sampler/max_score", "level_sampler/weighted_score"):
        steps, vals = history(client, run_id, key)
        if vals:
            print(f"  {key:<32} {vals[-1]:.4f}  (апдейт {steps[-1]})")
    print("\nsolve rate по уровням:")
    for lvl in levels:
        steps, vals = history(client, run_id, f"solve_rate/{lvl}")
        if vals:
            print(f"  {lvl:<20} {vals[-1]:.2f}")
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
