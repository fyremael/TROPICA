from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence
import math
import random

from .contracts import EmptySupportViolation


@dataclass(frozen=True)
class SupportMask:
    """A set-valued token mask.

    This keeps demos tokenizer-agnostic. Production code should compile these sets
    to tensor masks over tokenizer IDs.
    """
    allowed: frozenset[str]

    @classmethod
    def from_iter(cls, items: Iterable[str]) -> "SupportMask":
        return cls(frozenset(items))

    def __and__(self, other: "SupportMask") -> "SupportMask":
        return SupportMask(self.allowed & other.allowed)

    def __len__(self) -> int:
        return len(self.allowed)

    def assert_nonempty(self) -> None:
        if not self.allowed:
            raise EmptySupportError("Final support is empty. Abstain, backtrack, or replan; never sample.")


class EmptySupportError(EmptySupportViolation):
    pass


def intersect_masks(*masks: SupportMask) -> SupportMask:
    if not masks:
        raise ValueError("intersect_masks requires at least one mask")
    out = masks[0]
    for m in masks[1:]:
        out = out & m
    return out


def masked_softmax_sample(logits: dict[str, float], mask: SupportMask, temperature: float = 1.0, rng: random.Random | None = None) -> str:
    """Sample from logits restricted to a support mask.

    Tokens outside the support are absent from the normalizer; in tensor code this
    corresponds to assigning -inf before softmax.
    """
    mask.assert_nonempty()
    rng = rng or random
    tau = max(float(temperature), 1e-9)
    vals = {tok: logits.get(tok, float("-inf")) / tau for tok in mask.allowed}
    finite = {tok: v for tok, v in vals.items() if math.isfinite(v)}
    if not finite:
        # Uniform fallback over valid support. This is defensive. Production code
        # should log it as a model failure.
        return rng.choice(sorted(mask.allowed))
    m = max(finite.values())
    weights = {tok: math.exp(v - m) for tok, v in finite.items()}
    total = sum(weights.values())
    r = rng.random() * total
    acc = 0.0
    for tok, w in weights.items():
        acc += w
        if acc >= r:
            return tok
    return next(iter(finite))
