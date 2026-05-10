from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol
import random

from .masks import SupportMask, intersect_masks, masked_softmax_sample


class Planner(Protocol):
    def step(self, state: Any) -> "PlannerOutput": ...


class Guard(Protocol):
    def mask(self, prefix: list[str], state: Any) -> SupportMask: ...
    def update(self, state: Any, token: str) -> Any: ...


class Policy(Protocol):
    def mask(self, prefix: list[str], state: Any) -> SupportMask | None: ...


class Generator(Protocol):
    def logits(self, prefix: list[str], state: Any, control: dict[str, Any] | None = None) -> dict[str, float]: ...


@dataclass
class PlannerOutput:
    plan_mask: SupportMask
    winners: set[str]
    margin: float = float("inf")
    control_features: dict[str, Any] = field(default_factory=dict)
    trace: dict[str, Any] = field(default_factory=dict)


@dataclass
class DecodeTrace:
    token: str
    winners: list[str]
    final_support: list[str]
    margin: float
    planner_trace: dict[str, Any]


class NullPolicy:
    def __init__(self, vocab: set[str]):
        self.vocab = vocab

    def mask(self, prefix: list[str], state: Any) -> SupportMask | None:
        return SupportMask.from_iter(self.vocab)


class SupportDecoder:
    """Planner-guided support decoder.

    This is the external authority. Internal ControlDelta modules can improve
    logits, but correctness comes from this mask intersection.
    """

    def __init__(self, planner: Planner, guard: Guard, generator: Generator, policy: Policy | None = None, rng: random.Random | None = None):
        self.planner = planner
        self.guard = guard
        self.generator = generator
        self.policy = policy
        self.rng = rng or random.Random(0)

    def step(self, prefix: list[str], state: Any, temperature: float = 1.0) -> tuple[str, Any, DecodeTrace]:
        pout = self.planner.step(state)
        guard_mask = self.guard.mask(prefix, state)
        masks = [pout.plan_mask, guard_mask]
        if self.policy is not None:
            pmask = self.policy.mask(prefix, state)
            if pmask is not None:
                masks.append(pmask)
        final_mask = intersect_masks(*masks)
        final_mask.assert_nonempty()
        logits = self.generator.logits(prefix, state, control=pout.control_features)
        token = masked_softmax_sample(logits, final_mask, temperature=temperature, rng=self.rng)
        new_state = self.guard.update(state, token)
        trace = DecodeTrace(
            token=token,
            winners=sorted(pout.winners),
            final_support=sorted(final_mask.allowed),
            margin=pout.margin,
            planner_trace=pout.trace,
        )
        return token, new_state, trace
