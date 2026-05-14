from __future__ import annotations

import json
import random

import pytest

import cdsd
from cdsd.contracts import (
    IllegalSelectionError,
    IllegalTransitionError,
    UnifiedTraceEvent,
    ensure_guard_allows,
    ensure_selected_in_support,
    validate_intersection,
)
from cdsd.decoder import SupportDecoder
from cdsd.guards.dyck import DyckGuard
from cdsd.masks import EmptySupportError, SupportMask, intersect_masks
from cdsd.planners.dyck import CLOSE, EOS, OPEN, DyckPlanner, DyckState


class FixedGenerator:
    def logits(self, prefix, state, control=None):
        return {OPEN: 0.0, CLOSE: 0.0, EOS: 0.0, "ILLEGAL": 10_000.0}


class EmptyPolicy:
    def mask(self, prefix, state):
        return SupportMask.from_iter([])


def test_contract_exports_are_public() -> None:
    assert cdsd.SupportContractError is not None
    assert cdsd.UnifiedTraceEvent is UnifiedTraceEvent
    assert cdsd.PlannerOutput is not None


def test_planner_guard_policy_intersection_is_exact() -> None:
    planner = SupportMask.from_iter(["A", "B", "C"])
    guard = SupportMask.from_iter(["B", "C", "D"])
    policy = SupportMask.from_iter(["C", "D"])
    final = intersect_masks(planner, guard, policy)

    validate_intersection(final, planner, guard, policy)
    assert final.allowed == frozenset(["C"])


def test_bad_intersection_rejected() -> None:
    planner = SupportMask.from_iter(["A", "B"])
    guard = SupportMask.from_iter(["B"])
    bad_final = SupportMask.from_iter(["A", "B"])

    with pytest.raises(cdsd.SupportContractError):
        validate_intersection(bad_final, planner, guard)


def test_empty_support_is_typed_contract_violation() -> None:
    decoder = SupportDecoder(DyckPlanner(horizon=8), DyckGuard(), FixedGenerator(), policy=EmptyPolicy())

    with pytest.raises(EmptySupportError) as exc:
        decoder.step([], DyckState(horizon=8))

    assert isinstance(exc.value, cdsd.EmptySupportViolation)


def test_illegal_selection_and_guard_transition_are_typed() -> None:
    support = SupportMask.from_iter(["B"])

    with pytest.raises(IllegalSelectionError):
        ensure_selected_in_support("A", support)

    with pytest.raises(IllegalTransitionError):
        ensure_guard_allows("A", support)


def test_decoder_trace_records_contract_supports() -> None:
    decoder = SupportDecoder(DyckPlanner(horizon=4, finish_slack=3), DyckGuard(), FixedGenerator(), rng=random.Random(0))
    token, _, trace = decoder.step([], DyckState(horizon=4), temperature=1e-9)

    assert token == OPEN
    assert trace.planner_support == [OPEN]
    assert trace.guard_support == [OPEN]
    assert trace.final_support == [OPEN]
    assert trace.selected_was_allowed
    assert trace.selected_score == 0.0


def test_unified_trace_event_is_json_safe_and_complete() -> None:
    event = UnifiedTraceEvent(
        family="unit",
        scenario="contract",
        step=3,
        state_summary={"phase": "x"},
        planner_support=["A", "B"],
        guard_support=["B"],
        policy_support=None,
        final_support=["B"],
        selected="B",
        selected_score=0.5,
        selected_was_allowed=True,
        accepting=True,
        planner_trace={"margin": 1.0},
    )

    payload = event.to_dict()

    assert payload["schema_version"] == 1
    assert payload["trace_type"] == "support_event"
    assert payload["selected"] == "B"
    assert payload["final_support"] == ["B"]
    assert json.loads(json.dumps(payload))["accepting"] is True
