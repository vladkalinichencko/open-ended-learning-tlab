"""Build one self-contained report from the A100 triage artifacts."""

import base64
import io
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .hierarchy import merge_curves, overview_tree
from .training import ROOT


A100 = ROOT / "runs" / "a100_clearml"
RUNS = [
    ("ACCEL MaxMC", A100 / "ba25cc166eb2498b9a034bccb1d7cffc" / "accel_maxmc_a100_triage_seed0"),
    ("ACCEL fixed predictor", A100 / "8cbcabf23781446d97b2d08d500e76af" / "accel_fixed_predictor_a100_triage_seed0"),
    ("ACCEL CNN predictor", A100 / "8cbcabf23781446d97b2d08d500e76af" / "accel_cnn_predictor_a100_triage_seed0"),
    ("SFL", A100 / "8cbcabf23781446d97b2d08d500e76af" / "sfl_a100_triage_seed0"),
]
HELDOUT = A100 / "heldout_seed0_5k"
HIERARCHY = HELDOUT / "accel_maxmc_failure_hierarchy"


def image(fig) -> str:
    buffer = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buffer, format="png", dpi=150)
    plt.close(fig)
    return base64.b64encode(buffer.getvalue()).decode()


def training_figure(records: dict[str, list[dict]]) -> str:
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    fields = [
        ("validation_solve_mean", "Generated validation solve rate"),
        ("train_solve_rate", "Current training batch solve rate"),
        ("reward", "Current training batch mean reward"),
        ("entropy", "Policy entropy"),
    ]
    for axis, (field, title) in zip(axes.flat, fields):
        for name, rows in records.items():
            x = np.asarray([row["total_env_steps"] for row in rows]) / 1_000_000
            axis.plot(x, [row[field] for row in rows], marker="o", ms=3, label=name)
        axis.set(title=title, xlabel="Total environment interactions, millions")
        axis.grid(alpha=0.25)
    axes[0, 0].set_ylim(0, 1)
    axes[0, 1].set_ylim(0, 1)
    axes[0, 0].legend(fontsize=8)
    return image(fig)


def loss_figure(records: dict[str, list[dict]]) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for name, rows in records.items():
        x = np.asarray([row["total_env_steps"] for row in rows]) / 1_000_000
        axes[0].plot(x, [row["policy_loss"] for row in rows], marker="o", ms=3, label=name)
        axes[1].plot(x, [row["value_loss"] for row in rows], marker="o", ms=3, label=name)
    axes[0].set(title="Policy loss", xlabel="Total environment interactions, millions")
    axes[1].set(title="Value loss", xlabel="Total environment interactions, millions")
    for axis in axes:
        axis.grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    return image(fig)


def timing_figure(events: dict[str, list[dict]]) -> str:
    phases = ["search", "rollout", "ppo", "eval"]
    colors = ["#7c3aed", "#0ea5e9", "#f97316", "#10b981"]
    totals = {
        name: [sum(event["end"] - event["start"] for event in rows if event["phase"] == phase) for phase in phases]
        for name, rows in events.items()
    }
    fig, axis = plt.subplots(figsize=(11, 3.5))
    left = np.zeros(len(totals))
    for index, phase in enumerate(phases):
        values = np.asarray([totals[name][index] for name in totals])
        axis.barh(list(totals), values, left=left, label=phase, color=colors[index])
        left += values
    axis.set(xlabel="Recorded phase time, seconds")
    axis.legend(ncol=4)
    return image(fig)


def heldout_figure(results: dict[str, dict]) -> str:
    levels = next(iter(results.values()))["levels"]
    x = np.arange(len(levels))
    width = 0.8 / len(results)
    fig, axis = plt.subplots(figsize=(12, 5.5))
    for index, (name, result) in enumerate(results.items()):
        solve = [result["normal"][level]["solve_rate"] for level in levels]
        axis.bar(x + (index - 1.5) * width, solve, width, label=name)
    axis.set(xticks=x, xticklabels=levels, ylabel="Solve rate", ylim=(0, 1))
    axis.tick_params(axis="x", rotation=30)
    axis.legend(fontsize=8)
    return image(fig)


def teacher_figure(records: dict[str, list[dict]]) -> str:
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    specs = [
        ("ACCEL MaxMC", "transition_loss", "Diagnostic transition loss"),
        ("ACCEL fixed predictor", "predictor_loss", "Fixed predictor loss"),
        ("ACCEL CNN predictor", "predictor_loss", "CNN predictor loss"),
        ("SFL", "frontier_probability_mean", "SFL frontier success probability"),
    ]
    for axis, (name, field, title) in zip(axes.flat, specs):
        rows = records[name]
        axis.plot([row["update"] for row in rows], [row[field] for row in rows], marker="o")
        axis.set(title=title, xlabel="Update")
        axis.grid(alpha=0.25)
    return image(fig)


def main() -> None:
    records = {name: json.loads((path / "metrics.json").read_text()) for name, path in RUNS}
    events = {name: json.loads((path / "timeline.json").read_text()) for name, path in RUNS}
    heldout = {
        name: json.loads((HELDOUT / path.name / "heldout.json").read_text())
        for name, path in RUNS
    }
    hierarchy = np.load(HIERARCHY / "hierarchy.npz")
    trees = {length: hierarchy[f"linkage_{length}"] for length in range(10, 81, 10)}

    result_rows = []
    memory_rows = []
    for name, path in RUNS:
        final = records[name][-1]
        normal = np.mean([row["solve_rate"] for row in heldout[name]["normal"].values()])
        zero = np.mean([row["solve_rate"] for row in heldout[name]["zero_memory"].values()])
        result_rows.append(
            f"<tr><td>{name}</td><td>{final['total_env_steps']:,}</td>"
            f"<td>{final['validation_solve_mean']:.3f}</td><td>{normal:.3f}</td>"
            f'<td><a href="{path.relative_to(ROOT)}/report.html">run report</a></td></tr>'
        )
        memory_rows.append(f"<tr><td>{name}</td><td>{normal:.3f}</td><td>{zero:.3f}</td></tr>")

    levels = next(iter(heldout.values()))["levels"]
    heldout_rows = "".join(
        "<tr><td>" + level + "</td>" + "".join(
            f'<td>{heldout[name]["normal"][level]["solve_rate"]:.1f}</td>' for name, _ in RUNS
        ) + "</tr>"
        for level in levels
    )
    hierarchy_sections = "".join(
        f'''<details><summary>{length} frames</summary>
<img alt="hierarchy for {length}-frame contexts" src="data:image/png;base64,{overview_tree(trees[length])}"></details>'''
        for length in range(10, 81, 10)
    )
    headers = "".join(f"<th>{name}</th>" for name, _ in RUNS)

    (ROOT / "DIAGNOSTICS.html").write_text(f'''<!doctype html><meta charset="utf-8"><title>Open-ended learning diagnostics</title>
<style>
body{{font:15px system-ui;max-width:1120px;margin:36px auto;color:#222;line-height:1.45}}
h1,h2{{margin-top:36px}}img{{width:100%}}table{{border-collapse:collapse;width:100%}}
td,th{{padding:7px 10px;border:1px solid #ddd;text-align:right}}td:first-child,th:first-child{{text-align:left}}
.note{{background:#f5f5f5;padding:12px 16px;border-left:4px solid #777}}details{{border-top:1px solid #ddd;padding:10px 0}}
summary{{cursor:pointer;font-weight:600}}code{{background:#f1f1f1;padding:2px 4px}}
</style>
<h1>Open-ended learning: A100 triage and diagnostics</h1>
<p class="note">Один seed, 5000 updates и пять попыток на held-out уровень. Это отсев методов, а не финальное сравнение задания на 30 000 updates и нескольких seeds.</p>

<h2>Что обучалось</h2>
<p>Во всех четырёх запусках student одинаков: PPO + LSTM, 32 параллельные среды по 256 шагов. Меняется только teacher.</p>
<ul>
<li><b>ACCEL MaxMC.</b> Новые и мутированные уровни попадают в buffer по разнице между лучшим наблюдавшимся return и текущей value estimate. PPO обновляет student только на replay. Transition predictor обучается рядом, но не влияет на score или student.</li>
<li><b>ACCEL fixed predictor.</b> Teacher получает семь признаков карты, предсказывает вероятность успеха и ставит score <code>q(1-q)</code>. Поэтому buffer предпочитает уровни около границы решаемости.</li>
<li><b>ACCEL CNN predictor.</b> Механика та же, но predictor видит стены, старт и цель как три канала 13×13. Он может различать расположение клеток, которое теряется в семи числах.</li>
<li><b>SFL.</b> Каждые 50 updates teacher проверяет 128 случайных уровней по пять раз, оставляет 64 лучших по <code>p(1-p)</code>, затем собирает training batch поровну из frontier buffer и новых уровней. Поэтому SFL потратил 16 млн дополнительных search interactions.</li>
</ul>

<h2>Краткий результат</h2>
<table><tr><th>method</th><th>total interactions</th><th>generated validation</th><th>held-out mean</th><th>artifacts</th></tr>{''.join(result_rows)}</table>
<p>Все методы почти насыщают случайный validation-набор, но на человеческих картах расходятся. В этом коротком запуске CNN получил 0.375 held-out solve rate; остальные остались между 0.100 и 0.150. Значит локальный validation заметно проще dev-набора. Одного seed и пяти попыток недостаточно, чтобы объявлять CNN победителем.</p>

<h2>Динамика обучения</h2>
<img alt="training and validation dynamics" src="data:image/png;base64,{training_figure(records)}">
<p>Training solve, reward и losses относятся к последнему batch перед eval, поэтому они показывают ход оптимизации, а не итоговое качество метода. По оси X учтены и training, и SFL search interactions.</p>
<img alt="policy and value loss" src="data:image/png;base64,{loss_figure(records)}">

<h2>Что происходило внутри teacher</h2>
<img alt="teacher signals" src="data:image/png;base64,{teacher_figure(records)}">
<p>Predictor loss показывает обучение вспомогательной модели, но сам по себе не доказывает хороший curriculum. У SFL средняя вероятность frontier к концу близка к единице, то есть уменьшенный поиск часто возвращает уже простые уровни.</p>
<img alt="recorded phase time" src="data:image/png;base64,{timing_figure(events)}">

<h2>Held-out уровни</h2>
<img alt="held-out solve rate" src="data:image/png;base64,{heldout_figure(heldout)}">
<table><tr><th>level</th>{headers}</tr>{heldout_rows}</table>
<p>CNN выиграл среднее за счёт Labyrinth2, StandardMaze, StandardMaze3 и SixteenRooms. Это наблюдение по пяти попыткам, а не устойчивый вывод о типе карт.</p>

<h2>Causal zero-memory ablation</h2>
<table><tr><th>method</th><th>normal</th><th>zero LSTM memory every step</th></tr>{''.join(memory_rows)}</table>
<p>При обнулении hidden и cell почти все успешные эпизоды исчезают. На этих роллаутах политика использует память, но таблица не объясняет, какую информацию она хранит.</p>

<h2>Failure-context hierarchy для ACCEL MaxMC</h2>
<p>Transition predictor кодирует перекрывающиеся окна 10–80 кадров с шагом 10. Ни одно дерево не разрезано на готовые кластеры. Сначала смотрим расстояния слияния, потом выбираем линию разреза.</p>
<img alt="hierarchy merge distances" src="data:image/png;base64,{merge_curves(trees)}">
{hierarchy_sections}
<p><a href="{HIERARCHY.relative_to(ROOT)}/contexts.json">Context metadata</a>, <a href="{HIERARCHY.relative_to(ROOT)}/hierarchy.npz">embeddings and linkage matrices</a>, <a href="{HIERARCHY.relative_to(ROOT)}/report.html">separate hierarchy report with full trees</a>.</p>

<h2>Что пока не доказано</h2>
<p>Эти прогоны не воспроизводят полный бюджет задания, не дают разброс по seeds и не сравниваются с полными DR/PLR baseline. Они показывают, что pipeline работает, CNN-вариант заслуживает полного запуска, а validation-набор нельзя использовать как замену held-out оценке.</p>
''')
    print(ROOT / "DIAGNOSTICS.html")


if __name__ == "__main__":
    main()
