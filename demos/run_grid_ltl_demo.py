from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


from cdsd.planners.grid_ltl import GridLTLPlanner


def render_ascii(planner: GridLTLPlanner, path):
    path_cells = {(ps.x, ps.y) for ps in path}
    grid = [[" ." for _ in range(planner.width)] for _ in range(planner.height)]
    for x, y in planner.obstacles:
        grid[y][x] = "##"
    for x, y in planner.hazard:
        if grid[y][x] != "##":
            grid[y][x] = "~~"
    labels = {planner.S: "S ", planner.A: "A ", planner.B: "B ", planner.D: "D ", planner.G: "G "}
    for xy, ch in labels.items():
        x, y = xy
        grid[y][x] = ch
    for x, y in path_cells:
        if (x, y) in labels or (x, y) in planner.obstacles:
            continue
        grid[y][x] = "* " if grid[y][x] != "~~" else "*~"
    return "\n".join("".join(row) for row in grid)


if __name__ == "__main__":
    planner = GridLTLPlanner()
    path, parent, cost = planner.plan()
    print("Optimal route length:", cost)
    print(render_ascii(planner, path))
    print("Audit:", planner.audit(path))
