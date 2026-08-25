"""Hierarchical view of held-out sequences using the trained transition model."""

import argparse
import base64
import io
import json
from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from flax import serialization
from scipy.cluster.hierarchy import dendrogram, linkage

from .methods import create_transition_predictor

LENGTHS = range(10, 81, 10)
STRIDE = 10

def episode_lengths(dones: np.ndarray) -> np.ndarray:
    return np.where(dones.any(axis=0), dones.argmax(axis=0) + 1, len(dones))

def windows(data: np.lib.npyio.NpzFile, length: int, level_names: list[str]):
    observations = data["normal_observation"]
    actions = data["normal_action"]
    lengths = episode_lengths(data["normal_done"])
    solved = (data["normal_reward"] * (np.arange(len(observations))[:, None] < lengths)).sum(0) > 0
    obs, act, metadata = [], [], []
    for rollout, episode_length in enumerate(lengths):
        for start in range(0, int(episode_length) - length + 1, STRIDE):
            obs.append(observations[start : start + length, rollout])
            act.append(actions[start : start + length, rollout])
            metadata.append({
                "context": len(metadata),
                "level": level_names[rollout % len(level_names)],
                "attempt": rollout // len(level_names),
                "start": start,
                "end": start + length,
                "episode_solved": bool(solved[rollout]),
            })
    return np.stack(obs, axis=1), np.stack(act, axis=1), metadata

def embed(transition, observations: np.ndarray, actions: np.ndarray) -> np.ndarray:
    chunks = []
    for start in range(0, observations.shape[1], 128):
        end = start + 128
        _, intermediates = transition.apply_fn(
            transition.params,
            jnp.asarray(observations[:, start:end]),
            jnp.asarray(actions[:, start:end]),
            capture_intermediates=True,
            mutable=["intermediates"],
        )
        chunks.append(np.asarray(intermediates["intermediates"]["RNN_0"]["__call__"][0][-1]))
    embeddings = np.concatenate(chunks)
    if not np.isfinite(embeddings).all():
        raise ValueError("transition embeddings contain non-finite values")
    return embeddings

def full_tree(tree: np.ndarray, path: Path) -> None:
    leaves = tree.shape[0] + 1
    fig, axis = plt.subplots(figsize=(max(18, leaves * 0.045), 7))
    dendrogram(
        tree,
        ax=axis,
        color_threshold=0,
        above_threshold_color="#222",
        leaf_rotation=90,
        leaf_font_size=4,
    )
    axis.set(ylabel="Cosine distance", xlabel="Context index; metadata in contexts.json")
    fig.tight_layout()
    fig.savefig(path, format="svg")
    plt.close(fig)

def overview_tree(tree: np.ndarray) -> str:
    fig, axis = plt.subplots(figsize=(11, 4.5))
    dendrogram(
        tree,
        ax=axis,
        truncate_mode="lastp",
        p=60,
        show_leaf_counts=True,
        color_threshold=0,
        above_threshold_color="#222",
    )
    axis.set(ylabel="Cosine distance", xlabel="Top 60 branches; number in brackets is hidden leaves")
    fig.tight_layout()
    image = io.BytesIO()
    fig.savefig(image, format="png", dpi=140)
    plt.close(fig)
    return base64.b64encode(image.getvalue()).decode()

def merge_curves(trees: dict[int, np.ndarray]) -> str:
    fig, axis = plt.subplots(figsize=(10, 5.5))
    for length, tree in trees.items():
        distances = tree[:, 2]
        axis.plot(np.arange(len(distances), 0, -1), distances, label=f"{length} frames")
    axis.set(xlabel="Groups remaining before merge", ylabel="Cosine distance", xscale="log")
    axis.grid(alpha=0.25)
    axis.legend(ncol=2)
    fig.tight_layout()
    image = io.BytesIO()
    fig.savefig(image, format="png", dpi=150)
    plt.close(fig)
    return base64.b64encode(image.getvalue()).decode()

def build(run_dir: Path, rollout_path: Path, output_dir: Path) -> None:
    config = json.loads((run_dir / "config.json").read_text())
    transition = create_transition_predictor(config["seed"])
    transition = transition.replace(
        params=serialization.from_bytes(transition.params, (run_dir / "teacher.msgpack").read_bytes())
    )
    data = np.load(rollout_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    contexts, trees, arrays, sections = {}, {}, {}, []
    for length in LENGTHS:
        observations, actions, metadata = windows(data, length, config["eval_levels"])
        embeddings = embed(transition, observations, actions)
        tree = linkage(embeddings, method="average", metric="cosine")
        contexts[str(length)] = metadata
        trees[length] = tree
        arrays[f"embedding_{length}"] = embeddings
        arrays[f"linkage_{length}"] = tree
        full_name = f"tree_{length}.svg"
        full_tree(tree, output_dir / full_name)
        failed = sum(not row["episode_solved"] for row in metadata)
        sections.append(f'''<details><summary>{length} frames: {len(metadata)} contexts, {failed} from failed episodes</summary>
<img alt="hierarchical tree for {length}-frame contexts" src="data:image/png;base64,{overview_tree(tree)}">
<p><a href="{full_name}">Full tree with context indices</a></p></details>''')

    (output_dir / "contexts.json").write_text(json.dumps(contexts, indent=2))
    np.savez_compressed(output_dir / "hierarchy.npz", **arrays)
    (output_dir / "report.html").write_text(f'''<!doctype html><meta charset="utf-8"><title>Failure-context hierarchy</title>
<style>body{{font:15px system-ui;max-width:1100px;margin:32px auto;color:#222}}img{{width:100%}}
details{{border-top:1px solid #ddd;padding:12px 0}}summary{{cursor:pointer;font-weight:600}}</style>
<h1>Failure-context hierarchy</h1>
<p>Окна 10–80 кадров с шагом 10. Embedding берётся из последнего recurrent-state обученного transition predictor. Деревья построены average linkage по cosine distance и нигде не разрезаны.</p>
<img alt="merge distances for every context length" src="data:image/png;base64,{merge_curves(trees)}">
{''.join(sections)}
<p><a href="contexts.json">Context metadata</a>. <a href="hierarchy.npz">Embeddings and full linkage matrices</a>.</p>''')

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("rollouts", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    build(args.run, args.rollouts, args.output)

if __name__ == "__main__":
    main()
