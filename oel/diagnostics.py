"""JSON и один автономный HTML для каждого запуска."""

import base64
import io
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/oel-matplotlib")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def save_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _figure(records: list[dict]) -> str:
    steps = np.asarray([row["total_env_steps"] for row in records]) / 1_000_000
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    axes[0].plot(steps, [row["validation_solve_mean"] for row in records], label="validation solve rate")
    axes[0].plot(steps, [row["train_solve_rate"] for row in records], label="train solve rate")
    if "predicted_success" in records[0]:
        axes[0].plot(steps, [row["predicted_success"] for row in records], label="predicted success")
    if "frontier_probability_mean" in records[0]:
        axes[0].plot(steps, [row["frontier_probability_mean"] for row in records], label="frontier success")
    axes[0].set_ylim(0, 1)
    axes[0].legend()
    axes[0].grid(alpha=0.25)
    axes[1].plot(steps, [row["reward"] for row in records], label="mean step reward")
    axes[1].legend()
    axes[1].grid(alpha=0.25)
    axes[2].plot(steps, [row["policy_loss"] for row in records], label="policy loss")
    axes[2].plot(steps, [row["value_loss"] for row in records], label="value loss")
    if "predictor_loss" in records[0]:
        axes[2].plot(steps, [row["predictor_loss"] for row in records], label="predictor loss")
    for key in ("transition_loss", "pvl_score", "curriculum_score", "colearnability_bonus"):
        if key in records[0]:
            axes[2].plot(steps, [row[key] for row in records], label=key.replace("_", " "))
    axes[2].set_xlabel("Total environment interactions, millions")
    axes[2].legend()
    axes[2].grid(alpha=0.25)
    fig.tight_layout()
    out = io.BytesIO()
    fig.savefig(out, format="png", dpi=150)
    plt.close(fig)
    return base64.b64encode(out.getvalue()).decode()


def _timeline(events: list[dict]) -> str:
    if not events:
        return ""
    start = min(event["start"] for event in events)
    end = max(event["end"] for event in events)
    phases = ("search", "rollout", "ppo", "eval")
    left, width, row_height = 75, 850, 24
    colors = {"search": "#7c3aed", "rollout": "#0ea5e9", "ppo": "#f97316", "eval": "#10b981"}
    duration = max(end - start, 1e-9)
    marks = []
    for row, phase in enumerate(phases):
        y = 8 + row * row_height
        marks.append(f'<text x="0" y="{y + 12}" font-size="11">{phase}</text>')
        marks.append(f'<line x1="{left}" x2="{left + width}" y1="{y + 7}" y2="{y + 7}" stroke="#e5e7eb"/>')
    for event in events:
        x = left + width * (event["start"] - start) / duration
        w = max(1, width * (event["end"] - event["start"]) / duration)
        y = 8 + phases.index(event["phase"]) * row_height
        marks.append(
            f'<rect x="{x:.1f}" y="{y}" width="{w:.1f}" height="14" '
            f'fill="{colors[event["phase"]]}"><title>{event["phase"]}, update {event["update"]}</title></rect>'
        )
    for index in range(5):
        x = left + width * index / 4
        elapsed = duration * index / 4
        marks.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="104" y2="108" stroke="#6b7280"/>')
        marks.append(f'<text x="{x:.1f}" y="121" text-anchor="middle" font-size="10">{elapsed / 60:.1f} min</text>')
    return f'<svg viewBox="0 0 940 126" role="img">{"".join(marks)}</svg>'


def _parallel_update(events: list[dict], num_envs: int) -> str:
    rollout = next((event for event in reversed(events) if event["phase"] == "rollout"), None)
    ppo = next((event for event in reversed(events) if event["phase"] == "ppo"), None)
    if rollout is None or ppo is None:
        return ""
    update = rollout["update"]
    selected = [event for event in events if event["update"] == update and event["phase"] in ("rollout", "ppo", "eval")]
    start, end, left, width = selected[0]["start"], selected[-1]["end"], 105, 820
    duration = max(end - start, 1e-9)
    colors = {"rollout": "#0ea5e9", "ppo": "#f97316", "eval": "#10b981"}
    labels = {"rollout": f"rollout, {num_envs} env", "ppo": "shared PPO", "eval": "eval"}
    marks = [f'<text x="0" y="31" font-size="11">update {update}</text>']
    for event in selected:
        x = left + width * (event["start"] - start) / duration
        w = width * (event["end"] - event["start"]) / duration
        elapsed = event["end"] - event["start"]
        marks.append(
            f'<rect x="{x:.1f}" y="10" width="{w:.1f}" height="30" fill="{colors[event["phase"]]}">'
            f'<title>{labels[event["phase"]]}, {elapsed:.3f} s</title></rect>'
        )
        if w >= 58:
            marks.append(
                f'<text x="{x + w / 2:.1f}" y="29" text-anchor="middle" font-size="10" fill="white">'
                f'{labels[event["phase"]]} · {elapsed:.2f}s</text>'
            )
    marks.append(
        f'<text x="{left}" y="58" font-size="10" fill="#4b5563">'
        f'Все {num_envs} сред выполняются внутри одного векторизованного rollout.</text>'
    )
    return f'<svg viewBox="0 0 940 66" role="img">{"".join(marks)}</svg>'


def _rollout_figure(rollout: dict) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    axes[0].imshow(np.asarray(rollout["wall_map"]), cmap="Greys")
    for name, color in (("normal", "tab:blue"), ("zero_memory", "tab:orange")):
        path = np.asarray(rollout[name]["positions"])
        axes[0].plot(path[:, 0], path[:, 1], color=color, label=name)
    axes[0].scatter(*rollout["goal_pos"], marker="*", s=100, color="tab:green")
    axes[0].legend()
    axes[0].set_title("Один validation-уровень")
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    for name, color in (("normal", "tab:blue"), ("zero_memory", "tab:orange")):
        hidden = np.asarray(rollout[name]["hidden"])
        cell = np.asarray(rollout[name]["cell"])
        axes[1].plot(np.linalg.norm(hidden, axis=-1), color=color, label=f"{name}: hidden")
        axes[1].plot(np.linalg.norm(cell, axis=-1), color=color, linestyle="--", label=f"{name}: cell")
    axes[1].set_title("Состояние LSTM")
    axes[1].set_xlabel("step")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    out = io.BytesIO()
    fig.savefig(out, format="png", dpi=150)
    plt.close(fig)
    return base64.b64encode(out.getvalue()).decode()


def _transition_figure(transition: dict) -> str:
    predictions = np.asarray(transition["predictions"])
    targets = np.asarray(transition["targets"])
    errors = np.abs(predictions - targets).mean(axis=(1, 2, 3))
    chosen = np.linspace(0, len(errors) - 1, 4, dtype=int)
    fig, axes = plt.subplots(3, 4, figsize=(10, 7))
    for column, step in enumerate(chosen):
        axes[0, column].imshow(targets[step])
        axes[0, column].set_title(f"target, step {step}")
        axes[1, column].imshow(predictions[step])
        axes[1, column].set_title(f"prediction, step {step}")
        axes[0, column].axis("off")
        axes[1, column].axis("off")
    axes[2, 0].plot(errors)
    axes[2, 0].set(xlabel="step", ylabel="L1 error")
    for axis in axes[2, 1:]:
        axis.remove()
    fig.tight_layout()
    out = io.BytesIO()
    fig.savefig(out, format="png", dpi=150)
    plt.close(fig)
    return base64.b64encode(out.getvalue()).decode()


def write_report(run_dir: Path, config: dict, records: list[dict], events: list[dict], rollout: dict | None = None) -> None:
    rows = "".join(
        f"<tr><td>{row['update']}</td><td>{row['train_env_steps']:,}</td><td>{row['teacher_env_steps']:,}</td><td>{row['total_env_steps']:,}</td>"
        f"<td>{row['validation_solve_mean']:.3f}</td><td>{row['train_solve_rate']:.3f}</td>"
        f"<td>{row['reward']:.5f}</td><td>{row['policy_loss']:.5f}</td><td>{row['value_loss']:.5f}</td></tr>"
        for row in records
    )
    rollout_html = ""
    if rollout:
        rollout_html = (
            f'<h2>Rollout и causal zero-memory ablation</h2><img alt="rollout paths and LSTM state" '
            f'src="data:image/png;base64,{_rollout_figure(rollout)}"><p><a href="rollout.json">Исходные шаги, '
            'observations, actions, probabilities, rewards, dones, value, hidden и cell</a>.</p>'
        )
        if "transition" in rollout:
            rollout_html += (
                '<h2>Transition predictor на реальном rollout</h2><img alt="predicted and actual next observations" '
                f'src="data:image/png;base64,{_transition_figure(rollout["transition"])}">'
            )
    html = f"""<!doctype html><meta charset="utf-8"><title>{config['name']}</title>
<style>body{{font:15px system-ui;max-width:1050px;margin:32px auto;color:#222}}img,svg{{width:100%}}
table{{border-collapse:collapse}}td,th{{padding:6px 12px;border:1px solid #ddd;text-align:right}}
code{{background:#f4f4f4;padding:2px 4px}}</style>
<h1>{config['name']}</h1>
<p>Device: <code>{config['device']}</code>. Seed: {config['seed']}. Batch: {config['num_train_envs']} environments × {config['num_steps']} steps.</p>
<img alt="training dynamics" src="data:image/png;base64,{_figure(records)}">
<h2>Время поиска, rollout, PPO и eval</h2>{_timeline(events)}
<h2>Параллельные среды в последнем update</h2>{_parallel_update(events, config['num_train_envs'])}
{rollout_html}
<h2>Оценки</h2><table><tr><th>update</th><th>training steps</th><th>teacher steps</th><th>total steps</th><th>validation solve</th><th>train solve</th><th>reward</th><th>policy loss</th><th>value loss</th></tr>{rows}</table>
<p>Исходные значения: <a href="metrics.json">metrics.json</a>. Конфигурация: <a href="config.json">config.json</a>.</p>"""
    (run_dir / "report.html").write_text(html, encoding="utf-8")


def write_comparison_report(run_dirs: list[Path], output: Path) -> None:
    runs = []
    fig, axis = plt.subplots(figsize=(10, 5.5))
    for run_dir in run_dirs:
        config = json.loads((run_dir / "config.json").read_text())
        records = json.loads((run_dir / "metrics.json").read_text())
        steps = np.asarray([row["total_env_steps"] for row in records]) / 1_000_000
        solve = np.asarray([row["validation_solve_mean"] for row in records])
        axis.plot(steps, solve, label=config["name"])
        runs.append((config["name"], records[-1]["total_env_steps"], solve[-1], solve.max(), run_dir))
    axis.set(xlabel="Total environment interactions, millions", ylabel="Validation solve rate", ylim=(0, 1))
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    image = io.BytesIO()
    fig.savefig(image, format="png", dpi=150)
    plt.close(fig)
    rows = "".join(
        f'<tr><td>{name}</td><td>{steps:,}</td><td>{final:.3f}</td><td>{peak:.3f}</td>'
        f'<td><a href="{run_dir.name}/report.html">run</a></td></tr>'
        for name, steps, final, peak, run_dir in runs
    )
    html = f'''<!doctype html><meta charset="utf-8"><title>Mac comparison, seed 0</title>
<style>body{{font:15px system-ui;max-width:1050px;margin:32px auto;color:#222}}img{{width:100%}}
table{{border-collapse:collapse}}td,th{{padding:6px 12px;border:1px solid #ddd;text-align:right}}</style>
<h1>Предварительное сравнение на Mac, seed 0</h1>
<p>Кривые построены по одному и тому же validation-набору из 64 сгенерированных уровней. Один seed, поэтому доверительных полос нет.</p>
<img alt="validation solve dynamics" src="data:image/png;base64,{base64.b64encode(image.getvalue()).decode()}">
<table><tr><th>method</th><th>total interactions</th><th>final solve</th><th>peak solve</th><th>artifacts</th></tr>{rows}</table>'''
    output.write_text(html, encoding="utf-8")
