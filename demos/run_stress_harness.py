from __future__ import annotations

import csv
import json
import os
import random
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cdsd.control_delta_block import ControlDeltaBlock
from cdsd.decoder import SupportDecoder
from cdsd.guards.dyck import DyckGuard, is_valid_dyck
from cdsd.masks import EmptySupportError, SupportMask
from cdsd.planners.dyck import CLOSE, EOS, OPEN, DyckPlanner, DyckState
from cdsd.planners.grid_ltl import GridLTLPlanner
from cdsd.planners.json_schema import JSONSchemaGuard, JSONSchemaPlanner, JSONSchemaSpec, JSONSchemaState, render_json_tokens
from cdsd.planners.tool_workflow import ToolWorkflowGuard, ToolWorkflowPlanner
from cdsd.tokenizer_compiler import ByteTokenizer, TokenPrefixAutomaton, WordPieceTokenizer


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = Path(os.environ.get("CDSD_ARTIFACT_DIR", ROOT / "artifacts"))
OUT_CSV = ARTIFACT_DIR / "stress_summary.csv"
OUT_MD = ARTIFACT_DIR / "stress_summary.md"


@dataclass
class StressResult:
    domain: str
    cases: int
    failures: int
    duration_ms: float
    notes: str


class AdversarialDyckGenerator:
    def __init__(self, seed: int):
        self.rng = random.Random(seed)

    def logits(self, prefix, state, control=None):
        return {
            OPEN: 25.0 if state.balance >= 8 else self.rng.uniform(-1.0, 1.0),
            CLOSE: 50.0 if state.balance == 0 else self.rng.uniform(-1.0, 1.0),
            EOS: 40.0 if state.balance > 0 else self.rng.uniform(-1.0, 1.0),
        }


class EmptyPolicy:
    def mask(self, prefix, state):
        return SupportMask.from_iter([])


def timed(domain: str, fn) -> StressResult:
    start = time.perf_counter()
    cases, failures, notes = fn()
    return StressResult(domain, cases, failures, (time.perf_counter() - start) * 1000.0, notes)


def stress_dyck() -> tuple[int, int, str]:
    failures = 0
    cases = 0
    lengths = []
    support_sizes = []
    for horizon in [2, 4, 8, 16, 32, 64, 128, 256]:
        finish_slack = min(8, max(1, horizon - 1))
        for seed in range(250):
            cases += 1
            planner = DyckPlanner(horizon=horizon, max_balance=32, finish_slack=finish_slack)
            guard = DyckGuard()
            decoder = SupportDecoder(planner, guard, AdversarialDyckGenerator(seed), rng=random.Random(seed))
            prefix = []
            state = DyckState(horizon=horizon)
            try:
                for _ in range(horizon + 96):
                    token, state, trace = decoder.step(prefix, state)
                    if token not in trace.final_support:
                        failures += 1
                        break
                    support_sizes.append(len(trace.final_support))
                    prefix.append(token)
                    if token == EOS:
                        break
                if not is_valid_dyck(prefix):
                    failures += 1
                lengths.append(len(prefix))
            except Exception:
                failures += 1
    note = f"horizons=2..256 seeds=250 mean_len={statistics.fmean(lengths):.2f} mean_support={statistics.fmean(support_sizes):.2f}"
    return cases, failures, note


def stress_empty_support() -> tuple[int, int, str]:
    cases = 100
    failures = 0
    for seed in range(cases):
        decoder = SupportDecoder(
            DyckPlanner(horizon=16, finish_slack=4),
            DyckGuard(),
            AdversarialDyckGenerator(seed),
            policy=EmptyPolicy(),
            rng=random.Random(seed),
        )
        try:
            decoder.step([], DyckState(horizon=16))
            failures += 1
        except EmptySupportError:
            pass
    return cases, failures, "typed EmptySupportError required for contradictory policy"


def stress_json_schema() -> tuple[int, int, str]:
    rng = random.Random(17)
    cases = 300
    failures = 0
    max_props = 0
    for case in range(cases):
        prop_count = rng.randint(1, 7)
        max_props = max(max_props, prop_count)
        props = {}
        for idx in range(prop_count):
            key = f"k{idx}_{case}"
            enum_count = rng.randint(2, 5)
            props[key] = [f"v{idx}_{j}" for j in range(enum_count - 1)] + [f"space {idx} {case}"]
        spec = JSONSchemaSpec.enum_object(props)
        planner = JSONSchemaPlanner(spec)
        guard = JSONSchemaGuard(spec)
        state = JSONSchemaState()
        tokens: list[str] = []
        try:
            for _ in range(prop_count * 5 + 4):
                allowed = planner.step(state).plan_mask & guard.mask(tokens, state)
                allowed.assert_nonempty()
                token = rng.choice(sorted(allowed.allowed))
                tokens.append(token)
                state = guard.update(state, token)
                if token == JSONSchemaPlanner.EOS:
                    break
            parsed = json.loads(render_json_tokens(tokens))
            if set(parsed) != set(props):
                failures += 1
            for key, value in parsed.items():
                if value not in props[key]:
                    failures += 1
        except Exception:
            failures += 1
    return cases, failures, f"random enum objects max_props={max_props}"


def random_literal(rng: random.Random) -> str:
    alphabet = ["a", "b", "c", " ", '"', "_", "-", "0", "1", "snow", "☃", "é", "\t"]
    return "".join(rng.choice(alphabet) for _ in range(rng.randint(1, 8)))


def stress_tokenizer() -> tuple[int, int, str]:
    rng = random.Random(29)
    failures = 0
    cases = 0
    vocab = {'"': 1, "snow": 2, " ": 3, "☃": 4, "ab": 5, "a": 6, "b": 7, "_": 8, "-": 9, "\t": 10}
    tokenizers = [ByteTokenizer(), WordPieceTokenizer(vocab)]
    for tokenizer in tokenizers:
        for _ in range(150):
            literals = sorted({random_literal(rng) for _ in range(12)})
            automaton = TokenPrefixAutomaton(tokenizer, literals)
            for literal in literals:
                cases += 1
                state = automaton.initial_state()
                ids = tokenizer.encode(literal)
                for token_id in ids:
                    if token_id not in automaton.allowed_token_ids(state):
                        failures += 1
                        break
                    state = automaton.update(state, token_id)
                if not automaton.is_accepting(state) or state.complete_value != literal:
                    failures += 1
                if tokenizer.decode(list(state.emitted)) != literal:
                    failures += 1
    return cases, failures, "byte + wordpiece shared-prefix unicode literals"


def stress_workflow() -> tuple[int, int, str]:
    rng = random.Random(41)
    cases = 300
    failures = 0
    for case in range(cases):
        nodes = [f"N{i}" for i in range(rng.randint(3, 12))]
        graph: dict[str, list[str]] = {}
        for idx, node in enumerate(nodes):
            later = nodes[idx + 1 :]
            choices = rng.sample(later, rng.randint(1, min(3, len(later)))) if later else []
            if rng.random() < 0.35 or not choices:
                choices.append("DONE")
            graph[node] = sorted(set(choices))
        planner = ToolWorkflowPlanner(graph, nodes[0])
        guard = ToolWorkflowGuard(graph)
        state = planner.initial_state()
        try:
            for _ in range(len(nodes) + 2):
                allowed = planner.step(state).plan_mask & guard.mask([], state)
                allowed.assert_nonempty()
                token = rng.choice(sorted(allowed.allowed))
                state = guard.update(state, token)
                if token == "DONE":
                    eos = guard.mask([], state)
                    if eos.allowed != frozenset(["<eos>"]):
                        failures += 1
                    break
            else:
                failures += 1
        except Exception:
            failures += 1
    return cases, failures, "random DAG workflows with terminal EOS"


def stress_grid() -> tuple[int, int, str]:
    cases = 50
    failures = 0
    costs = []
    for _ in range(cases):
        planner = GridLTLPlanner()
        path, _, cost = planner.plan()
        audit = planner.audit(path)
        costs.append(cost)
        if cost <= 0 or not all(audit.values()):
            failures += 1
    return cases, failures, f"repeated Dijkstra audit cost={statistics.fmean(costs):.1f}"


def stress_control_delta() -> tuple[int, int, str]:
    torch.manual_seed(53)
    cases = 80
    failures = 0
    for idx in range(cases):
        batch = 1 + idx % 5
        time_steps = 1 + idx % 17
        d_in = 3 + idx % 7
        d_mem = 4 + idx % 9
        vocab = 2 + idx % 11
        block = ControlDeltaBlock(d_in, d_mem, vocab, channel_decay=idx % 2 == 0)
        base = torch.randn(batch, time_steps, d_in, requires_grad=True)
        x = base * (1 + idx % 5)
        out = block(x)
        chunked = block(x, chunk_size=1 + idx % 4)
        values = [v for v in out.values() if torch.is_tensor(v)]
        if not all(torch.isfinite(v).all().item() for v in values):
            failures += 1
        if not torch.allclose(out["memory"], chunked["memory"], atol=1e-6):
            failures += 1
        loss = out["logit_bias"].sum() + out["winner_logits"].sum() + out["phase_logits"].sum() + out["margin_logits"].sum()
        loss.backward()
        if base.grad is None or not torch.isfinite(base.grad).all().item():
            failures += 1
    return cases, failures, "random shapes, chunk equivalence, finite heads"


def write_results(results: list[StressResult]) -> None:
    OUT_CSV.parent.mkdir(exist_ok=True)
    with OUT_CSV.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["Domain", "Cases", "Failures", "DurationMs", "Notes"])
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "Domain": result.domain,
                    "Cases": result.cases,
                    "Failures": result.failures,
                    "DurationMs": f"{result.duration_ms:.3f}",
                    "Notes": result.notes,
                }
            )
    with OUT_MD.open("w", encoding="utf-8") as fh:
        fh.write("| Domain | Cases | Failures | DurationMs | Notes |\n")
        fh.write("| --- | ---: | ---: | ---: | --- |\n")
        for result in results:
            fh.write(f"| {result.domain} | {result.cases} | {result.failures} | {result.duration_ms:.3f} | {result.notes} |\n")


if __name__ == "__main__":
    checks = [
        ("Dyck adversarial decode", stress_dyck),
        ("Empty support contract", stress_empty_support),
        ("JSON schema subset", stress_json_schema),
        ("Tokenizer automata", stress_tokenizer),
        ("Tool workflow graph", stress_workflow),
        ("Grid LTL planner", stress_grid),
        ("ControlDelta numerics", stress_control_delta),
    ]
    results = [timed(name, fn) for name, fn in checks]
    write_results(results)
    for result in results:
        print(f"{result.domain}: cases={result.cases} failures={result.failures} duration_ms={result.duration_ms:.3f} notes={result.notes}")
    print(f"Wrote {OUT_CSV} and {OUT_MD}")
    if any(result.failures for result in results):
        raise SystemExit(1)
