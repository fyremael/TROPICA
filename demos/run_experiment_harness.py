from __future__ import annotations

import csv
import math
import os
import random
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cdsd.decoder import SupportDecoder
from cdsd.guards.dyck import DyckGuard, is_valid_dyck
from cdsd.masks import SupportMask, intersect_masks
from cdsd.planners.dyck import CLOSE, EOS, OPEN, DyckPlanner, DyckState


VOCAB = [OPEN, CLOSE, EOS]


class RandomGenerator:
    def __init__(self, seed: int):
        self.rng = random.Random(seed)

    def logits(self, prefix, state, control=None):
        bias = 0.25 if control else 0.0
        return {tok: self.rng.uniform(-1.0, 1.0) + bias for tok in VOCAB}


def entropy_allowed(mask: SupportMask) -> float:
    return math.log2(max(1, len(mask.allowed)))


def run_masked(mode: str, seed: int, horizon: int = 48):
    planner = DyckPlanner(horizon=horizon, finish_slack=8)
    guard = DyckGuard()
    gen = RandomGenerator(seed)
    prefix = []
    state = DyckState(horizon=horizon)
    rng = random.Random(seed)
    start = time.perf_counter()
    empty = 0
    support_sizes = []
    entropies = []
    illegal_pressure = []
    invalid = False
    for _ in range(horizon + 16):
        pout = planner.step(state)
        gmask = guard.mask(prefix, state)
        if mode in {"external_mask_only", "grammar_only"}:
            final = gmask
        elif mode in {"planner_guided", "control_delta_plus_external"}:
            final = intersect_masks(pout.plan_mask, gmask)
        else:
            final = SupportMask.from_iter(VOCAB)
        if not final.allowed:
            empty += 1
            break
        logits = gen.logits(prefix, state, control=pout.control_features if "control_delta" in mode else None)
        allowed = set(final.allowed)
        illegal_pressure.append(sum(max(0.0, logits[t]) for t in VOCAB if t not in allowed))
        support_sizes.append(len(final.allowed))
        entropies.append(entropy_allowed(final))
        tok = rng.choice(sorted(final.allowed))
        prefix.append(tok)
        if tok not in gmask.allowed:
            invalid = True
            break
        try:
            state = guard.update(state, tok)
        except AssertionError:
            invalid = True
            break
        if tok == EOS:
            break
    latency = time.perf_counter() - start
    return {
        "Mode": mode,
        "InvalidRate": float(invalid or not is_valid_dyck(prefix)),
        "DeltaCost": 1.0 if "control_delta" in mode else 0.0,
        "EmptySupportRate": float(empty > 0),
        "WinnerCardinality": statistics.fmean(support_sizes) if support_sizes else 0.0,
        "EntropyAllowed": statistics.fmean(entropies) if entropies else 0.0,
        "IllegalLogitPressure": statistics.fmean(illegal_pressure) if illegal_pressure else 0.0,
        "Latency": latency,
    }


def summarize(rows):
    modes = sorted({r["Mode"] for r in rows})
    metrics = [k for k in rows[0] if k != "Mode"]
    summary = []
    for mode in modes:
        subset = [r for r in rows if r["Mode"] == mode]
        item = {"Mode": mode}
        for metric in metrics:
            item[metric] = statistics.fmean(float(r[metric]) for r in subset)
        summary.append(item)
    return summary


if __name__ == "__main__":
    modes = [
        "raw_generator",
        "external_mask_only",
        "control_delta_only",
        "control_delta_plus_external",
        "grammar_only",
        "planner_guided",
    ]
    rows = [run_masked(mode, seed) for mode in modes for seed in range(100)]
    summary = summarize(rows)
    out_dir = Path(os.environ.get("CDSD_ARTIFACT_DIR", "artifacts"))
    out_dir.mkdir(exist_ok=True)
    csv_path = out_dir / "experiment_summary.csv"
    md_path = out_dir / "experiment_summary.md"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    with md_path.open("w", encoding="utf-8") as fh:
        headers = list(summary[0])
        fh.write("| " + " | ".join(headers) + " |\n")
        fh.write("| " + " | ".join("---" for _ in headers) + " |\n")
        for row in summary:
            fh.write("| " + " | ".join(f"{row[h]:.4f}" if h != "Mode" else str(row[h]) for h in headers) + " |\n")
    for row in summary:
        print(row)
    print(f"Wrote {csv_path} and {md_path}")
