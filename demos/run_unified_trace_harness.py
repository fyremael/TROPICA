from __future__ import annotations

import csv
import json
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cdsd.contracts import UnifiedTraceEvent, ensure_selected_in_support, validate_intersection
from cdsd.control_delta_block import ControlDeltaBlock
from cdsd.decoder import SupportDecoder
from cdsd.guards.dyck import DyckGuard
from cdsd.masks import EmptySupportError, SupportMask, intersect_masks
from cdsd.planners.dyck import CLOSE, EOS, OPEN, DyckPlanner, DyckState
from cdsd.planners.grid_ltl import GridLTLPlanner
from cdsd.planners.json_schema import JSONSchemaGuard, JSONSchemaPlanner, JSONSchemaSpec, JSONSchemaState, render_json_tokens
from cdsd.planners.tool_workflow import ToolWorkflowGuard, ToolWorkflowPlanner
from cdsd.tokenizer_compiler import ByteTokenizer, TokenPrefixAutomaton


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = Path(os.environ.get("CDSD_ARTIFACT_DIR", ROOT / "artifacts"))
OUT_CSV = ARTIFACT_DIR / "unified_trace_summary.csv"
OUT_MD = ARTIFACT_DIR / "unified_trace_summary.md"
OUT_TRACE = ARTIFACT_DIR / "unified_traces.jsonl"


@dataclass
class UnifiedTraceResult:
    family: str
    cases: int
    failures: int
    trace_events: int
    negative_controls: int
    duration_ms: float
    notes: str


class ScriptedGenerator:
    def __init__(self, target: list[str], illegal: list[str] | None = None):
        self.target = target
        self.illegal = illegal or []

    def logits(self, prefix, state, control=None):
        index = len(prefix)
        scores = {token: 0.0 for token in self.target + self.illegal}
        if index < len(self.target):
            scores[self.target[index]] = 100.0
        for token in self.illegal:
            scores[token] = 1000.0
        return scores


def trace_record(
    *,
    family: str,
    scenario: str,
    events: list[dict[str, object]],
    accepted: bool,
    value: str | None,
    error: str | None = None,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "trace_type": "unified_support",
        "family": family,
        "scenario": {"provider": "support-contract", "suite": scenario, "family": family},
        "accepted": accepted,
        "value": value,
        "parsed": None,
        "emitted_token_ids": [],
        "emitted_text": value,
        "steps": len(events),
        "events": events,
        "error": error,
    }


def run_scripted_decoder_trace(
    *,
    family: str,
    scenario: str,
    planner,
    guard,
    initial_state,
    target: list[str],
    state_summary,
) -> tuple[dict[str, object], int]:
    decoder = SupportDecoder(planner, guard, ScriptedGenerator(target, illegal=["}", "WRONG", "<bad>"]), rng=random.Random(7))
    prefix: list[str] = []
    state = initial_state
    events = []
    failures = 0
    for step in range(len(target)):
        before = state
        token, state, trace = decoder.step(prefix, state, temperature=1e-9)
        prefix.append(token)
        if token != target[step] or token not in trace.final_support:
            failures += 1
        trace.accepting = step == len(target) - 1
        events.append(
            trace.to_unified_event(
                family=family,
                scenario=scenario,
                step=step,
                state_summary=state_summary(before),
            ).to_dict()
        )
    return trace_record(family=family, scenario=scenario, events=events, accepted=failures == 0, value=" ".join(prefix)), failures


def dyck_trace() -> tuple[dict[str, object], int]:
    return run_scripted_decoder_trace(
        family="dyck",
        scenario="balanced parentheses support",
        planner=DyckPlanner(horizon=4, finish_slack=3),
        guard=DyckGuard(),
        initial_state=DyckState(horizon=4),
        target=[OPEN, CLOSE, EOS],
        state_summary=lambda state: asdict(state),
    )


def json_schema_trace() -> tuple[dict[str, object], int]:
    spec = JSONSchemaSpec.enum_object({"mode": ["fast", "safe"], "target": ["docs", "ci"]})
    target = ["{", "mode", ":", "safe", ",", "target", ":", "ci", "}", JSONSchemaPlanner.EOS]
    record, failures = run_scripted_decoder_trace(
        family="json_schema",
        scenario="finite enum object",
        planner=JSONSchemaPlanner(spec),
        guard=JSONSchemaGuard(spec),
        initial_state=JSONSchemaState(),
        target=target,
        state_summary=lambda state: {"phase": state.phase, "current_key": state.current_key, "emitted": list(state.emitted)},
    )
    record["value"] = render_json_tokens(target)
    record["emitted_text"] = record["value"]
    return record, failures


def workflow_trace() -> tuple[dict[str, object], int]:
    graph = {"START": ["SEARCH", "ASK"], "SEARCH": ["SUMMARIZE"], "ASK": ["SUMMARIZE"], "SUMMARIZE": ["DONE"]}
    planner = ToolWorkflowPlanner(graph, "START")
    return run_scripted_decoder_trace(
        family="workflow",
        scenario="tool route support",
        planner=planner,
        guard=ToolWorkflowGuard(graph),
        initial_state=planner.initial_state(),
        target=["SEARCH", "SUMMARIZE", "DONE", "<eos>"],
        state_summary=lambda state: {"node": state.node, "step": state.step},
    )


def grid_trace() -> tuple[dict[str, object], int]:
    planner = GridLTLPlanner()
    path, _, cost = planner.plan()
    failures = 0
    events = []
    for step, (current, nxt) in enumerate(zip(path, path[1:])):
        if step >= 12:
            break
        allowed = []
        selected = None
        for move in planner.DIRS:
            candidate = planner.step_allowed(current, move)
            if candidate is None:
                continue
            name = planner.DIRNAME[move]
            allowed.append(name)
            if candidate == nxt:
                selected = name
        if selected is None or selected not in allowed:
            failures += 1
        events.append(
            UnifiedTraceEvent(
                family="grid",
                scenario="ltl route prefix",
                step=step,
                state_summary={"x": current.x, "y": current.y, "a_seen": current.a_seen, "b_seen": current.b_seen, "d_seen": current.d_seen},
                planner_support=sorted(allowed),
                guard_support=sorted(allowed),
                policy_support=None,
                final_support=sorted(allowed),
                selected=selected,
                selected_score=0.0,
                selected_was_allowed=selected in allowed if selected is not None else False,
                accepting=planner.accepting(nxt),
                planner_trace={"cost": cost},
            ).to_dict()
        )
    return trace_record(family="grid", scenario="ltl route prefix", events=events, accepted=failures == 0, value=f"cost={cost}"), failures


def tokenizer_trace() -> tuple[dict[str, object], int]:
    tokenizer = ByteTokenizer()
    automaton = TokenPrefixAutomaton(tokenizer, ["a", "ab", "abc", "snow"])
    state = automaton.initial_state()
    events = []
    failures = 0
    for step, token_id in enumerate(tokenizer.encode("abc")):
        allowed = sorted(automaton.allowed_token_ids(state))
        selected = str(token_id)
        if token_id not in allowed:
            failures += 1
        state = automaton.update(state, token_id)
        support = [str(tok) for tok in allowed]
        events.append(
            UnifiedTraceEvent(
                family="tokenizer",
                scenario="byte prefix automaton",
                step=step,
                state_summary={"node": state.node, "emitted": list(state.emitted)},
                planner_support=support,
                guard_support=support,
                policy_support=None,
                final_support=support,
                selected=selected,
                selected_score=0.0,
                selected_was_allowed=token_id in allowed,
                accepting=automaton.is_accepting(state),
                planner_trace={"complete_value": state.complete_value},
            ).to_dict()
        )
    if not automaton.is_accepting(state) or state.complete_value != "abc":
        failures += 1
    return trace_record(family="tokenizer", scenario="byte prefix automaton", events=events, accepted=failures == 0, value=state.complete_value), failures


def control_delta_trace() -> tuple[dict[str, object], int]:
    torch.manual_seed(31)
    block = ControlDeltaBlock(d_in=4, d_mem=6, vocab_size=5)
    x = torch.randn(1, 4, 4)
    out = block(x, chunk_size=1)
    failures = 0
    values = [value for value in out.values() if torch.is_tensor(value)]
    if not all(torch.isfinite(value).all().item() for value in values):
        failures += 1
    logits = out["logit_bias"][0]
    events = []
    for step in range(logits.shape[0]):
        row = logits[step]
        selected = int(torch.argmax(row).item())
        support = [str(idx) for idx in range(row.shape[0])]
        events.append(
            UnifiedTraceEvent(
                family="control_delta",
                scenario="finite recurrent control lane",
                step=step,
                state_summary={"time": step, "memory_norm": round(float(out["memory"].norm().item()), 6)},
                planner_support=support,
                guard_support=support,
                policy_support=support,
                final_support=support,
                selected=str(selected),
                selected_score=float(row[selected].item()),
                selected_was_allowed=str(selected) in support,
                accepting=step == logits.shape[0] - 1,
                planner_trace={"summary_norm": round(float(out["summary"][0, step].norm().item()), 6)},
            ).to_dict()
        )
    return trace_record(family="control_delta", scenario="finite recurrent control lane", events=events, accepted=failures == 0, value="finite"), failures


def contract_negative_controls() -> tuple[dict[str, object], int, int]:
    failures = 0
    controls = 0
    planner = SupportMask.from_iter(["A", "B"])
    guard = SupportMask.from_iter(["B", "C"])
    final = intersect_masks(planner, guard)
    try:
        validate_intersection(final, planner, guard)
    except Exception:
        failures += 1
    controls += 1
    try:
        ensure_selected_in_support("A", final)
        failures += 1
    except Exception:
        pass
    controls += 1
    try:
        bad_final = SupportMask.from_iter(["A", "B"])
        validate_intersection(bad_final, planner, guard)
        failures += 1
    except Exception:
        pass
    controls += 1
    guard_obj = JSONSchemaGuard(JSONSchemaSpec.enum_object({"k": ["v"]}))
    try:
        guard_obj.update(JSONSchemaState(), "WRONG")
        failures += 1
    except EmptySupportError:
        pass
    controls += 1
    event = UnifiedTraceEvent(
        family="contract",
        scenario="negative controls",
        step=0,
        state_summary={"control_count": controls},
        planner_support=sorted(planner.allowed),
        guard_support=sorted(guard.allowed),
        policy_support=None,
        final_support=sorted(final.allowed),
        selected="A",
        selected_score=1000.0,
        selected_was_allowed=False,
        accepting=False,
        failure_reason="illegal selection rejected",
        planner_trace={"expected_intersection": sorted(final.allowed)},
    ).to_dict()
    return trace_record(family="contract", scenario="negative controls", events=[event], accepted=failures == 0, value=None, error="illegal controls rejected"), failures, controls


def timed_family(family: str, work) -> tuple[UnifiedTraceResult, list[dict[str, object]]]:
    start = time.perf_counter()
    negative_controls = 0
    if family == "contract":
        record, failures, negative_controls = work()
    else:
        record, failures = work()
    duration_ms = (time.perf_counter() - start) * 1000.0
    events = int(record["steps"])
    result = UnifiedTraceResult(
        family=family,
        cases=max(1, events + negative_controls),
        failures=failures,
        trace_events=events,
        negative_controls=negative_controls,
        duration_ms=duration_ms,
        notes=str(record["scenario"]["suite"]),
    )
    return result, [record]


def write_results(results: list[UnifiedTraceResult], traces: list[dict[str, object]]) -> None:
    OUT_CSV.parent.mkdir(exist_ok=True)
    fields = ["Family", "Cases", "Failures", "TraceEvents", "NegativeControls", "DurationMs", "Notes"]
    with OUT_CSV.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "Family": result.family,
                    "Cases": result.cases,
                    "Failures": result.failures,
                    "TraceEvents": result.trace_events,
                    "NegativeControls": result.negative_controls,
                    "DurationMs": f"{result.duration_ms:.3f}",
                    "Notes": result.notes,
                }
            )
    with OUT_MD.open("w", encoding="utf-8") as fh:
        fh.write("| Family | Cases | Failures | TraceEvents | NegativeControls | DurationMs | Notes |\n")
        fh.write("| --- | ---: | ---: | ---: | ---: | ---: | --- |\n")
        for result in results:
            fh.write(
                f"| {result.family} | {result.cases} | {result.failures} | {result.trace_events} | "
                f"{result.negative_controls} | {result.duration_ms:.3f} | {result.notes} |\n"
            )
    with OUT_TRACE.open("w", encoding="utf-8") as fh:
        for trace in traces:
            fh.write(json.dumps(trace, ensure_ascii=False, allow_nan=False, sort_keys=True) + "\n")


if __name__ == "__main__":
    checks = [
        ("dyck", dyck_trace),
        ("json_schema", json_schema_trace),
        ("workflow", workflow_trace),
        ("grid", grid_trace),
        ("tokenizer", tokenizer_trace),
        ("control_delta", control_delta_trace),
        ("contract", contract_negative_controls),
    ]
    results = []
    traces = []
    for family, work in checks:
        result, records = timed_family(family, work)
        results.append(result)
        traces.extend(records)
    write_results(results, traces)
    for result in results:
        print(
            f"{result.family}: cases={result.cases} failures={result.failures} trace_events={result.trace_events} "
            f"negative_controls={result.negative_controls} duration_ms={result.duration_ms:.3f} notes={result.notes}"
        )
    print(f"Wrote {OUT_CSV}, {OUT_MD}, and {OUT_TRACE}")
    if any(result.failures for result in results):
        raise SystemExit(1)
