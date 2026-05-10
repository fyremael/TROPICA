from __future__ import annotations

from dataclasses import dataclass

from cdsd.decoder import PlannerOutput
from cdsd.masks import EmptySupportError, SupportMask


@dataclass(frozen=True)
class ToolWorkflowState:
    node: str
    step: int = 0


class ToolWorkflowPlanner:
    def __init__(self, graph: dict[str, list[str]], start: str, terminal: str = "DONE"):
        self.graph = {k: tuple(v) for k, v in graph.items()}
        self.start = start
        self.terminal = terminal

    def initial_state(self) -> ToolWorkflowState:
        return ToolWorkflowState(self.start)

    def step(self, state: ToolWorkflowState) -> PlannerOutput:
        winners = set(self.graph.get(state.node, ()))
        if state.node == self.terminal:
            winners = {"<eos>"}
        return PlannerOutput(
            plan_mask=SupportMask.from_iter(winners),
            winners=winners,
            margin=0.0 if len(winners) > 1 else float("inf"),
            control_features={"node": state.node, "step": state.step, "choices": sorted(winners)},
            trace={"node": state.node, "step": state.step},
        )


class ToolWorkflowGuard:
    def __init__(self, graph: dict[str, list[str]], terminal: str = "DONE"):
        self.graph = {k: tuple(v) for k, v in graph.items()}
        self.terminal = terminal

    def mask(self, prefix: list[str], state: ToolWorkflowState) -> SupportMask:
        if state.node == self.terminal:
            return SupportMask.from_iter(["<eos>"])
        return SupportMask.from_iter(self.graph.get(state.node, ()))

    def update(self, state: ToolWorkflowState, token: str) -> ToolWorkflowState:
        allowed = set(self.mask([], state).allowed)
        if token not in allowed:
            raise EmptySupportError(f"Tool transition {state.node!r} -> {token!r} is illegal")
        if token == "<eos>":
            return state
        return ToolWorkflowState(token, state.step + 1)
