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
from cdsd.planners.dyck import CLOSE, EOS, OPEN, DyckPlanner, DyckState
from cdsd.planners.json_schema import JSONSchemaGuard, JSONSchemaPlanner, JSONSchemaSpec, JSONSchemaState, render_json_tokens
from cdsd.planners.tool_workflow import ToolWorkflowGuard, ToolWorkflowPlanner
from cdsd.tokenizer_compiler import ByteTokenizer, TokenPrefixAutomaton, WordPieceTokenizer


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = Path(os.environ.get("CDSD_ARTIFACT_DIR", ROOT / "artifacts"))
OUT_CSV = ARTIFACT_DIR / "scale_summary.csv"
OUT_MD = ARTIFACT_DIR / "scale_summary.md"


@dataclass
class ScaleRow:
    track: str
    size: int
    cases: int
    failures: int
    duration_ms: float
    throughput: float
    primary_metric: float
    notes: str


class HostileGenerator:
    def __init__(self, seed: int):
        self.rng = random.Random(seed)

    def logits(self, prefix, state, control=None):
        return {
            OPEN: 30.0 if state.balance >= 12 else self.rng.uniform(-2.0, 2.0),
            CLOSE: 80.0 if state.balance == 0 else self.rng.uniform(-2.0, 2.0),
            EOS: 60.0 if state.balance > 0 else self.rng.uniform(-2.0, 2.0),
        }


def timed_row(track: str, size: int, cases: int, work, notes: str) -> ScaleRow:
    start = time.perf_counter()
    failures, primary_metric = work()
    duration_ms = (time.perf_counter() - start) * 1000.0
    throughput = (cases / duration_ms * 1000.0) if duration_ms > 0 else 0.0
    return ScaleRow(track, size, cases, failures, duration_ms, throughput, primary_metric, notes)


def dyck_rows() -> list[ScaleRow]:
    rows: list[ScaleRow] = []
    for horizon in [16, 32, 64, 128, 256, 512, 1024]:
        seeds = 96 if horizon <= 256 else 48

        def work(horizon=horizon, seeds=seeds):
            failures = 0
            lengths = []
            for seed in range(seeds):
                planner = DyckPlanner(horizon=horizon, max_balance=64, finish_slack=min(12, max(1, horizon // 4)))
                guard = DyckGuard()
                decoder = SupportDecoder(planner, guard, HostileGenerator(seed), rng=random.Random(seed))
                prefix = []
                state = DyckState(horizon=horizon)
                try:
                    for _ in range(horizon + 128):
                        tok, state, trace = decoder.step(prefix, state)
                        if tok not in trace.final_support:
                            failures += 1
                            break
                        prefix.append(tok)
                        if tok == EOS:
                            break
                    if not is_valid_dyck(prefix):
                        failures += 1
                    lengths.append(len(prefix))
                except Exception:
                    failures += 1
            return failures, statistics.fmean(lengths)

        rows.append(timed_row("Dyck horizon", horizon, seeds, work, "mean emitted tokens"))
    return rows


def json_rows() -> list[ScaleRow]:
    rows: list[ScaleRow] = []
    rng = random.Random(101)
    for props_count in [1, 4, 8, 16, 32, 64]:
        cases = 24 if props_count <= 16 else 12

        def work(props_count=props_count, cases=cases):
            failures = 0
            emitted_counts = []
            for case in range(cases):
                props = {
                    f"k{i}_{case}": [f"v{i}_{j}" for j in range(4)] + [f"spaced value {i} {case}"]
                    for i in range(props_count)
                }
                spec = JSONSchemaSpec.enum_object(props)
                planner = JSONSchemaPlanner(spec)
                guard = JSONSchemaGuard(spec)
                state = JSONSchemaState()
                tokens = []
                try:
                    for _ in range(props_count * 5 + 5):
                        mask = planner.step(state).plan_mask & guard.mask(tokens, state)
                        mask.assert_nonempty()
                        token = rng.choice(sorted(mask.allowed))
                        tokens.append(token)
                        state = guard.update(state, token)
                        if token == JSONSchemaPlanner.EOS:
                            break
                    parsed = json.loads(render_json_tokens(tokens))
                    if set(parsed) != set(props):
                        failures += 1
                    emitted_counts.append(len(tokens))
                except Exception:
                    failures += 1
            return failures, statistics.fmean(emitted_counts)

        rows.append(timed_row("JSON properties", props_count, cases, work, "mean emitted tokens"))
    return rows


def make_literals(count: int) -> list[str]:
    return [f"enum/shared/prefix/{i:06d}/snowman-\u2603/value {i % 97}" for i in range(count)]


def tokenizer_rows() -> list[ScaleRow]:
    rows: list[ScaleRow] = []
    vocab = {
        "enum": 1,
        "/": 2,
        "shared": 3,
        "prefix": 4,
        "snowman": 5,
        "-": 6,
        "value": 7,
        " ": 8,
    }
    for literal_count in [32, 128, 512, 2048, 4096]:
        tokenizers = [ByteTokenizer(), WordPieceTokenizer(vocab)]
        cases = literal_count * len(tokenizers)

        def work(literal_count=literal_count, tokenizers=tokenizers):
            failures = 0
            nodes = []
            literals = make_literals(literal_count)
            for tokenizer in tokenizers:
                automaton = TokenPrefixAutomaton(tokenizer, literals)
                nodes.append(len(automaton.nodes))
                for literal in literals:
                    state = automaton.initial_state()
                    for token_id in tokenizer.encode(literal):
                        if token_id not in automaton.allowed_token_ids(state):
                            failures += 1
                            break
                        state = automaton.update(state, token_id)
                    if state.complete_value != literal:
                        failures += 1
            return failures, statistics.fmean(nodes)

        rows.append(timed_row("Tokenizer enums", literal_count, cases, work, "mean automaton nodes"))
    return rows


def workflow_rows() -> list[ScaleRow]:
    rows: list[ScaleRow] = []
    rng = random.Random(202)
    for node_count in [16, 64, 256, 1024, 2048]:
        traversals = 128 if node_count <= 256 else 64

        def work(node_count=node_count, traversals=traversals):
            graph: dict[str, list[str]] = {}
            nodes = [f"N{i}" for i in range(node_count)]
            for idx, node in enumerate(nodes):
                if idx == node_count - 1:
                    graph[node] = ["DONE"]
                    continue
                choices = {nodes[idx + 1]}
                for jump in [2, 5, 17]:
                    if idx + jump < node_count:
                        choices.add(nodes[idx + jump])
                if idx > node_count * 3 // 4:
                    choices.add("DONE")
                graph[node] = sorted(choices)
            planner = ToolWorkflowPlanner(graph, nodes[0])
            guard = ToolWorkflowGuard(graph)
            failures = 0
            lengths = []
            for _ in range(traversals):
                state = planner.initial_state()
                steps = 0
                try:
                    for _ in range(node_count + 2):
                        mask = planner.step(state).plan_mask & guard.mask([], state)
                        mask.assert_nonempty()
                        token = rng.choice(sorted(mask.allowed))
                        state = guard.update(state, token)
                        steps += 1
                        if token == "DONE":
                            break
                    else:
                        failures += 1
                    lengths.append(steps)
                except Exception:
                    failures += 1
            return failures, statistics.fmean(lengths)

        rows.append(timed_row("Workflow nodes", node_count, traversals, work, "mean route length"))
    return rows


def control_delta_rows() -> list[ScaleRow]:
    rows: list[ScaleRow] = []
    torch.manual_seed(303)
    configs = [
        (32, 4, 12, 8, 16),
        (128, 6, 16, 16, 32),
        (512, 8, 24, 24, 48),
        (1024, 8, 32, 32, 64),
    ]
    for time_steps, batch, d_in, d_mem, vocab in configs:
        cases = batch * time_steps

        def work(time_steps=time_steps, batch=batch, d_in=d_in, d_mem=d_mem, vocab=vocab):
            failures = 0
            block = ControlDeltaBlock(d_in, d_mem, vocab, channel_decay=True)
            base = torch.randn(batch, time_steps, d_in, requires_grad=True)
            out = block(base, chunk_size=64)
            tensors = [v for v in out.values() if torch.is_tensor(v)]
            if not all(torch.isfinite(v).all().item() for v in tensors):
                failures += 1
            loss = out["logit_bias"].mean() + out["winner_logits"].mean()
            loss.backward()
            if base.grad is None or not torch.isfinite(base.grad).all().item():
                failures += 1
            return failures, float(out["memory"].numel())

        rows.append(timed_row("ControlDelta tokens", time_steps, cases, work, "memory elements"))
    return rows


def write_rows(rows: list[ScaleRow]) -> None:
    OUT_CSV.parent.mkdir(exist_ok=True)
    fields = ["Track", "Size", "Cases", "Failures", "DurationMs", "Throughput", "PrimaryMetric", "Notes"]
    with OUT_CSV.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "Track": row.track,
                    "Size": row.size,
                    "Cases": row.cases,
                    "Failures": row.failures,
                    "DurationMs": f"{row.duration_ms:.3f}",
                    "Throughput": f"{row.throughput:.3f}",
                    "PrimaryMetric": f"{row.primary_metric:.3f}",
                    "Notes": row.notes,
                }
            )
    with OUT_MD.open("w", encoding="utf-8") as fh:
        fh.write("| Track | Size | Cases | Failures | DurationMs | Throughput | PrimaryMetric | Notes |\n")
        fh.write("| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |\n")
        for row in rows:
            fh.write(
                f"| {row.track} | {row.size} | {row.cases} | {row.failures} | "
                f"{row.duration_ms:.3f} | {row.throughput:.3f} | {row.primary_metric:.3f} | {row.notes} |\n"
            )


if __name__ == "__main__":
    all_rows = dyck_rows() + json_rows() + tokenizer_rows() + workflow_rows() + control_delta_rows()
    write_rows(all_rows)
    for row in all_rows:
        print(
            f"{row.track}: size={row.size} cases={row.cases} failures={row.failures} "
            f"duration_ms={row.duration_ms:.3f} throughput={row.throughput:.3f} "
            f"primary={row.primary_metric:.3f} notes={row.notes}"
        )
    print(f"Wrote {OUT_CSV} and {OUT_MD}")
    if any(row.failures for row in all_rows):
        raise SystemExit(1)
