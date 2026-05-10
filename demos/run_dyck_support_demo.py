from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


import random

from cdsd.decoder import SupportDecoder
from cdsd.planners.dyck import DyckPlanner, DyckState, OPEN, CLOSE, EOS
from cdsd.guards.dyck import DyckGuard, is_valid_dyck


class RandomGenerator:
    def __init__(self, vocab, seed=0):
        self.vocab = list(vocab)
        self.rng = random.Random(seed)

    def logits(self, prefix, state, control=None):
        # Deliberately dumb. The mask makes it safe.
        return {tok: self.rng.uniform(-1.0, 1.0) for tok in self.vocab}


def sample_trio(horizon=48, seed=0):
    planner = DyckPlanner(horizon=horizon, finish_slack=8)
    guard = DyckGuard()
    gen = RandomGenerator([OPEN, CLOSE, EOS], seed=seed)
    dec = SupportDecoder(planner, guard, gen, rng=random.Random(seed))
    prefix = []
    state = DyckState(balance=0, step=0, horizon=horizon)
    traces = []
    for _ in range(horizon + 16):
        tok, state, trace = dec.step(prefix, state)
        prefix.append(tok)
        traces.append(trace)
        if tok == EOS:
            break
    return prefix, traces


def sample_raw(horizon=48, seed=0):
    rng = random.Random(seed)
    seq = []
    for _ in range(horizon):
        seq.append(rng.choice([OPEN, CLOSE, EOS]))
        if seq[-1] == EOS:
            break
    return seq


if __name__ == "__main__":
    n = 500
    raw_bad = 0
    trio_bad = 0
    winner_sizes = []
    for seed in range(n):
        raw = sample_raw(seed=seed)
        trio, traces = sample_trio(seed=seed)
        raw_bad += not is_valid_dyck(raw)
        trio_bad += not is_valid_dyck(trio)
        winner_sizes.extend(len(t.final_support) for t in traces)
    print("Raw invalid rate:", raw_bad / n)
    print("Planner&Guard invalid rate:", trio_bad / n)
    print("Mean final support size:", sum(winner_sizes) / len(winner_sizes))
    print("Example trio:", "".join(tok for tok in sample_trio(seed=7)[0]))
