"""Что учитель на самом деле держит в буфере — по чекпойнтам, то есть во времени.

ACCEL мутирует уровни с высоким score, и вопрос «что конкретно он строит» до сих пор
был без ответа. Здесь буфер разбирается по тем же структурным осям, по которым мы
меряли dev-уровни (levels.py), и взвешивается по настоящей вероятности переигрывания
PLR — то есть показывает не «какие уровни лежат», а «какие уровни студент видит».

Плюс главный вопрос задания: корреляция score с каждым признаком. Score-функция
должна находить уровни на границе способностей студента; если она коррелирует, скажем,
только с длиной пути, то она меряет сложность, а не обучаемость.

    python diag_buffer.py checkpoints/full_accel_s0/0/models
"""

import argparse
import json
import pathlib

import numpy as np
import orbax.checkpoint as ocp

import levels

DEV = ["SixteenRooms", "SixteenRooms2", "Labyrinth", "LabyrinthFlipped",
       "Labyrinth2", "StandardMaze", "StandardMaze2", "StandardMaze3"]

def replay_weights(sampler, temperature=0.3, staleness_coeff=0.3):
    """Ровно формула jaxued: (1-c)·вес по рангу score + c·вес по устареванию."""
    size = int(sampler["size"])
    live = np.arange(len(sampler["scores"])) < size
    s = np.where(live, np.asarray(sampler["scores"]), -np.inf)
    ranks = np.empty(len(s), dtype=float)
    ranks[np.argsort(-s, kind="stable")] = np.arange(1, len(s) + 1)
    w_s = np.where(live, 1 / ranks, 0.0) ** (1 / temperature)
    stale = np.where(live, int(sampler["episode_count"]) - np.asarray(sampler["timestamps"]), 0.0)
    w_s, w_c = w_s / max(w_s.sum(), 1e-12), stale / max(stale.sum(), 1e-12)
    return live, (1 - staleness_coeff) * w_s + staleness_coeff * w_c

def buffer_stats(sampler):
    live, w = replay_weights(sampler)
    idx = np.flatnonzero(live)
    rows = [levels.features(*levels.from_buffer(sampler, int(i))) for i in idx]
    scores = np.asarray(sampler["scores"])[idx]
    wl = w[idx]
    keys = list(rows[0])
    out = {"size": int(sampler["size"]), "score_mean": float(np.mean(scores)), "признаки": {}}
    for k in keys:
        v = np.array([r[k] for r in rows], dtype=float)
        ok = np.isfinite(scores) & np.isfinite(v)
        out["признаки"][k] = {
            "равномерно": float(v.mean()),
            "по переигрыванию": float((v * wl).sum() / wl.sum()),
            "p90": float(np.percentile(v, 90)),
            "max": float(v.max()),
            "корреляция со score": float(np.corrcoef(v[ok], scores[ok])[0, 1])
                                   if ok.sum() > 2 and v[ok].std() > 0 else None,
        }
    return out

def main():
    p = argparse.ArgumentParser()
    p.add_argument("models", help="checkpoints/<run>/<seed>/models")
    p.add_argument("--every", type=int, default=1, help="брать каждый N-й чекпойнт")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    mgr = ocp.CheckpointManager(pathlib.Path(args.models).resolve(), ocp.PyTreeCheckpointer())
    out = {"dev": {n: levels.features(*levels.from_prefab(n)) for n in DEV}, "checkpoints": []}
    for s in mgr.all_steps()[::args.every]:
        st = mgr.restore(s)
        row = {"checkpoint": int(s), "step": int(st["step"]), **buffer_stats(st["sampler"])}
        out["checkpoints"].append(row)
        f = row["признаки"]
        print(f"шаг {row['step']:>7} буфер {row['size']:>5}  "
              f"коридор {f['длина коридора']['по переигрыванию']:6.2f} "
              f"(r со score {f['длина коридора']['корреляция со score']:+.2f})  "
              f"путь {f['путь']['по переигрыванию']:6.2f} "
              f"(r {f['путь']['корреляция со score']:+.2f})  "
              f"развилок {f['развилок']['по переигрыванию']:6.1f}", flush=True)

    path = pathlib.Path(args.out or "runs/buffer.json")
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"-> {path}")

if __name__ == "__main__":
    main()
