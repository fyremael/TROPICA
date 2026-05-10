from cdsd.planners.dyck import DyckPlanner, DyckState, OPEN, CLOSE, EOS
from cdsd.guards.dyck import DyckGuard, is_valid_dyck
from cdsd.decoder import SupportDecoder
import random


class RandomGenerator:
    def logits(self, prefix, state, control=None):
        return {OPEN: 0.0, CLOSE: 0.0, EOS: 0.0}


def test_trio_dyck_validity_many_seeds():
    for seed in range(100):
        planner = DyckPlanner(horizon=32, finish_slack=8)
        guard = DyckGuard()
        decoder = SupportDecoder(planner, guard, RandomGenerator(), rng=random.Random(seed))
        prefix = []
        state = DyckState(balance=0, step=0, horizon=32)
        for _ in range(48):
            tok, state, trace = decoder.step(prefix, state)
            assert tok in trace.final_support
            prefix.append(tok)
            if tok == EOS:
                break
        assert is_valid_dyck(prefix)
