"""Механика UED одной страницей: схема цикла, буфер с настоящими числами, разбор score.

Данные берутся из чекпойнта живого прогона ACCEL, а не выдумываются: буфер, его score,
устаревание и сами лабиринты. Ползунки пересчитывают вероятность переигрывания ровно по
формуле jaxued, так что страницу можно крутить и видеть, что от чего зависит.
"""

import json
import pathlib

TEMPLATE = pathlib.Path(__file__).with_name("explainer.html").read_text()

data = json.loads(pathlib.Path("runs/buffer_slice.json").read_text())
import sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import levels as L

data["dev"] = json.loads(pathlib.Path("tmp/level_features.json").read_text())
data["dev"]["grids"] = {}
for name in data["dev"]["признаки"]:
    walls, agent, goal = L.from_prefab(name)
    _, path = L.shortest_path(walls, agent, goal)
    data["dev"]["grids"][name] = {"walls": walls.astype(int).tolist(),
                                  "agent": list(agent), "goal": list(goal),
                                  "path": [list(x) for x in path]}
out = pathlib.Path("runs/ued_explainer.html")
out.write_text(TEMPLATE.replace("__DATA__", json.dumps(data, ensure_ascii=False)))
print(f"-> {out}  ({out.stat().st_size // 1024} КБ)")
