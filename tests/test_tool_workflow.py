from cdsd.planners.tool_workflow import ToolWorkflowGuard, ToolWorkflowPlanner


def test_tool_workflow_graph_masks_transitions():
    graph = {"START": ["SEARCH", "ASK"], "SEARCH": ["SUMMARIZE"], "ASK": ["SUMMARIZE"], "SUMMARIZE": ["DONE"]}
    planner = ToolWorkflowPlanner(graph, "START")
    guard = ToolWorkflowGuard(graph)
    state = planner.initial_state()
    assert planner.step(state).winners == {"SEARCH", "ASK"}
    state = guard.update(state, "SEARCH")
    assert guard.mask([], state).allowed == frozenset(["SUMMARIZE"])
    state = guard.update(state, "SUMMARIZE")
    state = guard.update(state, "DONE")
    assert guard.mask([], state).allowed == frozenset(["<eos>"])
