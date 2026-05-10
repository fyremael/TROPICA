from __future__ import annotations

from dataclasses import dataclass

from cdsd.decoder import PlannerOutput
from cdsd.masks import SupportMask

OPEN = "("
CLOSE = ")"
EOS = "<eos>"


@dataclass
class DyckState:
    balance: int = 0
    step: int = 0
    horizon: int = 48


class DyckPlanner:
    """Multi-winner planner for Dyck-1.

    Early/mid sequence: exposes OPEN and CLOSE when both are feasible.
    Finish window: forces CLOSE until balanced, then EOS.
    """

    def __init__(self, horizon: int = 48, max_balance: int = 64, finish_slack: int = 8):
        self.horizon = horizon
        self.max_balance = max_balance
        self.finish_slack = finish_slack

    def step(self, state: DyckState) -> PlannerOutput:
        r = max(0, self.horizon - state.step)
        winners: set[str] = set()
        if state.step == 0 and state.balance == 0:
            winners.add(OPEN)
        elif r <= self.finish_slack:
            if state.balance == 0:
                winners.add(EOS)
            else:
                winners.add(CLOSE)
        else:
            if state.balance > 0:
                winners.add(CLOSE)
            if state.balance + 1 <= self.max_balance:
                winners.add(OPEN)
        return PlannerOutput(
            plan_mask=SupportMask.from_iter(winners),
            winners=winners,
            margin=0.0 if len(winners) > 1 else float("inf"),
            control_features={"balance": state.balance, "step": state.step, "remaining": r, "winners": sorted(winners)},
            trace={"balance": state.balance, "step": state.step, "remaining": r},
        )
