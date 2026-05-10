from __future__ import annotations

from cdsd.masks import SupportMask
from cdsd.planners.dyck import DyckState, OPEN, CLOSE, EOS


class DyckGuard:
    def mask(self, prefix: list[str], state: DyckState) -> SupportMask:
        allowed = {OPEN} if state.balance == 0 else {OPEN, CLOSE}
        if state.balance == 0 and prefix:
            allowed.add(EOS)
        return SupportMask.from_iter(allowed)

    def update(self, state: DyckState, token: str) -> DyckState:
        bal = state.balance
        if token == OPEN:
            bal += 1
        elif token == CLOSE:
            bal -= 1
            if bal < 0:
                raise AssertionError("Guard allowed an illegal close; this should be impossible.")
        return DyckState(balance=bal, step=state.step + 1, horizon=state.horizon)


def is_valid_dyck(seq: list[str]) -> bool:
    bal = 0
    for tok in seq:
        if tok == EOS:
            break
        if tok == OPEN:
            bal += 1
        elif tok == CLOSE:
            bal -= 1
        if bal < 0:
            return False
    return bal == 0 and seq and seq[-1] == EOS
