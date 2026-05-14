from __future__ import annotations

from typing import Any
import random

from .contracts import (
    DecodeTrace,
    Generator,
    Guard,
    Planner,
    PlannerOutput,
    Policy,
    ensure_guard_allows,
    ensure_selected_in_support,
    support_items,
    validate_intersection,
)
from .masks import SupportMask, intersect_masks, masked_softmax_sample


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
        policy_mask = None
        if self.policy is not None:
            policy_mask = self.policy.mask(prefix, state)
            if policy_mask is not None:
                masks.append(policy_mask)
        final_mask = intersect_masks(*masks)
        validate_intersection(final_mask, *masks)
        final_mask.assert_nonempty()
        logits = self.generator.logits(prefix, state, control=pout.control_features)
        token = masked_softmax_sample(logits, final_mask, temperature=temperature, rng=self.rng)
        ensure_selected_in_support(token, final_mask)
        ensure_guard_allows(token, guard_mask)
        new_state = self.guard.update(state, token)
        trace = DecodeTrace(
            token=token,
            winners=sorted(pout.winners),
            final_support=sorted(final_mask.allowed),
            margin=pout.margin,
            planner_trace=pout.trace,
            planner_support=support_items(pout.plan_mask) or [],
            guard_support=support_items(guard_mask) or [],
            policy_support=support_items(policy_mask),
            selected_score=logits.get(token),
            selected_was_allowed=token in final_mask.allowed,
        )
        return token, new_state, trace
