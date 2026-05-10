from cdsd.planners.grid_ltl import GridLTLPlanner


def test_grid_ltl_audit_and_cost():
    planner = GridLTLPlanner()
    path, parent, cost = planner.plan()
    audit = planner.audit(path)
    assert cost > 0
    assert all(audit.values())
