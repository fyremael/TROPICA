from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import heapq
import itertools


@dataclass(frozen=True)
class PState:
    x: int
    y: int
    a_seen: bool
    b_seen: bool
    d_seen: bool


class GridLTLPlanner:
    """Dijkstra planner on grid × task flags.

    Constraints:
      1. Visit A before B.
      2. Avoid H until D.
      3. Reach G after A, B, D are satisfied.
    """

    DIRS = [(0, -1), (1, 0), (0, 1), (-1, 0)]
    DIRNAME = {(0, -1): "N", (1, 0): "E", (0, 1): "S", (-1, 0): "W"}

    def __init__(self, width=16, height=12):
        self.width = width
        self.height = height
        self.S = (1, 9)
        self.A = (3, 2)
        self.B = (13, 2)
        self.D = (8, 9)
        self.G = (14, 10)
        self.obstacles = set((x, 6) for x in range(2, 14))
        self.obstacles.discard((5, 6))
        self.obstacles.discard((10, 6))
        self.hazard = set((x, 4) for x in range(5, 12))

    def in_bounds(self, xy):
        x, y = xy
        return 0 <= x < self.width and 0 <= y < self.height

    def step_allowed(self, ps: PState, mv: tuple[int, int]) -> Optional[PState]:
        nx, ny = ps.x + mv[0], ps.y + mv[1]
        if not self.in_bounds((nx, ny)) or (nx, ny) in self.obstacles:
            return None
        a_seen = ps.a_seen or (nx, ny) == self.A
        d_seen = ps.d_seen or (nx, ny) == self.D
        if not a_seen and (nx, ny) == self.B:
            return None
        if not d_seen and (nx, ny) in self.hazard:
            return None
        b_seen = ps.b_seen or ((nx, ny) == self.B and a_seen)
        return PState(nx, ny, a_seen, b_seen, d_seen)

    def accepting(self, ps: PState) -> bool:
        return (ps.x, ps.y) == self.G and ps.a_seen and ps.b_seen and ps.d_seen

    def plan(self):
        start = PState(self.S[0], self.S[1], False, False, False)
        counter = itertools.count()
        pq = [(0, next(counter), start)]
        parent = {start: (None, (0, 0))}
        dist = {start: 0}
        goal = None
        while pq:
            g, _, u = heapq.heappop(pq)
            if g != dist[u]:
                continue
            if self.accepting(u):
                goal = u
                break
            for mv in self.DIRS:
                v = self.step_allowed(u, mv)
                if v is None:
                    continue
                ng = g + 1
                if v not in dist or ng < dist[v]:
                    dist[v] = ng
                    parent[v] = (u, mv)
                    heapq.heappush(pq, (ng, next(counter), v))
        if goal is None:
            return [], parent, -1
        path = []
        cur = goal
        while cur is not None:
            path.append(cur)
            cur = parent[cur][0]
        path.reverse()
        return path, parent, dist[goal]

    def audit(self, path: list[PState]):
        a_seen = False
        d_seen = False
        ok_pre_a = True
        ok_pre_d = True
        for ps in path:
            if (ps.x, ps.y) == self.A:
                a_seen = True
            if (ps.x, ps.y) == self.D:
                d_seen = True
            if (ps.x, ps.y) == self.B and not a_seen:
                ok_pre_a = False
            if (ps.x, ps.y) in self.hazard and not d_seen:
                ok_pre_d = False
        return {
            "visited_A": any((ps.x, ps.y) == self.A for ps in path),
            "visited_B": any((ps.x, ps.y) == self.B for ps in path),
            "visited_D": any((ps.x, ps.y) == self.D for ps in path),
            "reached_G": any((ps.x, ps.y) == self.G for ps in path),
            "A_before_B": ok_pre_a,
            "avoid_H_until_D": ok_pre_d,
            "accepting": bool(path) and self.accepting(path[-1]),
        }
