"""Structural features of a maze level, as pure functions of its grid.

The point of these is the question "what makes a level hard for *this* agent", which
is not the same as "how many walls". The agent sees 5x5 around itself, so what costs
it is the stretch of path where it has no choice and no feedback — hence the corridor
feature, which turned out to separate solved from unsolved dev levels better than
anything else (see NOTES).

Used both for the eight hand-made dev levels and for the levels the teacher keeps in
its buffer, so the two are measured on the same axes.
"""

import collections

import numpy as np


def neighbours(r, c, walls):
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        rr, cc = r + dr, c + dc
        if 0 <= rr < walls.shape[0] and 0 <= cc < walls.shape[1] and not walls[rr, cc]:
            yield rr, cc


def shortest_path(walls, start, goal):
    """BFS: length and the path itself; (-1, []) when the goal is walled off."""
    prev, seen = {}, {start}
    q = collections.deque([start])
    while q:
        cur = q.popleft()
        if cur == goal:
            path = [cur]
            while path[-1] != start:
                path.append(prev[path[-1]])
            return len(path) - 1, path[::-1]
        for nxt in neighbours(*cur, walls):
            if nxt not in seen:
                seen.add(nxt)
                prev[nxt] = cur
                q.append(nxt)
    return -1, []


def features(walls, agent, goal):
    walls = np.asarray(walls)
    cells = [(r, c) for r in range(walls.shape[0]) for c in range(walls.shape[1])
             if not walls[r, c]]
    deg = {cell: sum(1 for _ in neighbours(*cell, walls)) for cell in cells}
    dist, path = shortest_path(walls, agent, goal)
    straight = abs(agent[0] - goal[0]) + abs(agent[1] - goal[1])

    longest, run = 0, 0  # самый длинный участок пути без развилок
    for cell in path:
        run = 0 if deg.get(cell, 0) >= 3 else run + 1
        longest = max(longest, run)

    return {"путь": dist,
            "прямая": straight,
            "извилистость": round(dist / max(straight, 1), 2),
            "тупиков": sum(1 for d in deg.values() if d == 1),
            "развилок": sum(1 for d in deg.values() if d >= 3),
            "плотность стен": round(float(walls.mean()), 3),
            "длина коридора": longest}


def from_prefab(name):
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).parent / "ext/jaxued/src"))
    from jaxued.environments.maze.level import Level, prefabs
    lvl = Level.from_str(prefabs[name])
    return (np.asarray(lvl.wall_map),
            (int(lvl.agent_pos[1]), int(lvl.agent_pos[0])),
            (int(lvl.goal_pos[1]), int(lvl.goal_pos[0])))


def from_buffer(sampler, i):
    """i-th level of a restored PLR buffer (checkpoints/<run>/<seed>/models)."""
    lv = sampler["levels"]
    return (np.asarray(lv["wall_map"][i]),
            (int(lv["agent_pos"][i][1]), int(lv["agent_pos"][i][0])),
            (int(lv["goal_pos"][i][1]), int(lv["goal_pos"][i][0])))
