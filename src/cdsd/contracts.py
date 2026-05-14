from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol


TRACE_SCHEMA_VERSION = 1


class SupportContractError(RuntimeError):
    """Base error for planner/guard/policy support contract violations."""


class EmptySupportViolation(SupportContractError):
    """Raised when intersected support is empty and decoding must fail closed."""


class IllegalSelectionError(SupportContractError):
    """Raised when a selected token/action is outside final support."""


class IllegalTransitionError(SupportContractError):
    """Raised when a guard transition is attempted outside guard support."""


class StaleStateError(SupportContractError):
    """Raised when a trace or update is inconsistent with its source state."""


class SupportMaskLike(Protocol):
    allowed: frozenset[str]


class Planner(Protocol):
    def step(self, state: Any) -> "PlannerOutput": ...


class Guard(Protocol):
    def mask(self, prefix: list[str], state: Any) -> SupportMaskLike: ...
    def update(self, state: Any, token: str) -> Any: ...


class Policy(Protocol):
    def mask(self, prefix: list[str], state: Any) -> SupportMaskLike | None: ...


class Generator(Protocol):
    def logits(self, prefix: list[str], state: Any, control: dict[str, Any] | None = None) -> dict[str, float]: ...


@dataclass
class PlannerOutput:
    plan_mask: SupportMaskLike
    winners: set[str]
    margin: float = float("inf")
    control_features: dict[str, Any] = field(default_factory=dict)
    trace: dict[str, Any] = field(default_factory=dict)


@dataclass
class UnifiedTraceEvent:
    """Versioned, JSON-safe trace event shared by support surfaces."""

    family: str
    scenario: str
    step: int
    planner_support: list[str]
    guard_support: list[str]
    policy_support: list[str] | None
    final_support: list[str]
    selected: str | None
    state_summary: dict[str, Any] = field(default_factory=dict)
    selected_score: float | None = None
    selected_was_allowed: bool = False
    accepting: bool = False
    failure_reason: str | None = None
    planner_trace: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        record = {
            "schema_version": TRACE_SCHEMA_VERSION,
            "trace_type": "support_event",
            "family": self.family,
            "scenario": self.scenario,
            "step": self.step,
            "state_summary": self.state_summary,
            "planner_support": list(self.planner_support),
            "guard_support": list(self.guard_support),
            "policy_support": None if self.policy_support is None else list(self.policy_support),
            "final_support": list(self.final_support),
            "selected": self.selected,
            "selected_score": self.selected_score,
            "selected_was_allowed": self.selected_was_allowed,
            "accepting": self.accepting,
            "failure_reason": self.failure_reason,
            "planner_trace": self.planner_trace,
        }
        json.dumps(record, ensure_ascii=False, allow_nan=False)
        return record


@dataclass
class DecodeTrace:
    token: str
    winners: list[str]
    final_support: list[str]
    margin: float
    planner_trace: dict[str, Any]
    planner_support: list[str] = field(default_factory=list)
    guard_support: list[str] = field(default_factory=list)
    policy_support: list[str] | None = None
    selected_score: float | None = None
    selected_was_allowed: bool = True
    accepting: bool = False
    failure_reason: str | None = None

    def to_unified_event(
        self,
        *,
        family: str,
        scenario: str,
        step: int,
        state_summary: Mapping[str, Any] | None = None,
    ) -> UnifiedTraceEvent:
        return UnifiedTraceEvent(
            family=family,
            scenario=scenario,
            step=step,
            state_summary=dict(state_summary or {}),
            planner_support=list(self.planner_support),
            guard_support=list(self.guard_support),
            policy_support=None if self.policy_support is None else list(self.policy_support),
            final_support=list(self.final_support),
            selected=self.token,
            selected_score=self.selected_score,
            selected_was_allowed=self.selected_was_allowed,
            accepting=self.accepting,
            failure_reason=self.failure_reason,
            planner_trace=dict(self.planner_trace),
        )


def support_items(mask: SupportMaskLike | None) -> list[str] | None:
    if mask is None:
        return None
    return sorted(mask.allowed)


def intersection_support(*masks: SupportMaskLike) -> set[str]:
    if not masks:
        raise ValueError("intersection_support requires at least one mask")
    out = set(masks[0].allowed)
    for mask in masks[1:]:
        out &= set(mask.allowed)
    return out


def validate_intersection(final_support: SupportMaskLike, *inputs: SupportMaskLike) -> None:
    expected = intersection_support(*inputs)
    actual = set(final_support.allowed)
    if actual != expected:
        raise SupportContractError(
            f"Final support must be the exact intersection of planner/guard/policy support: "
            f"expected={sorted(expected)!r} actual={sorted(actual)!r}"
        )


def ensure_nonempty_support(support: SupportMaskLike, *, context: str = "final support") -> None:
    if not support.allowed:
        raise EmptySupportViolation(f"{context} is empty. Fail closed instead of sampling.")


def ensure_selected_in_support(selected: str, support: SupportMaskLike, *, context: str = "selected token") -> None:
    if selected not in support.allowed:
        raise IllegalSelectionError(f"{context} {selected!r} is outside final support {sorted(support.allowed)!r}")


def ensure_guard_allows(selected: str, guard_support: SupportMaskLike) -> None:
    if selected not in guard_support.allowed:
        raise IllegalTransitionError(f"Guard rejected transition for token {selected!r}")
