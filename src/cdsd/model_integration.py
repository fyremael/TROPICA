from __future__ import annotations

import importlib.util
import json
import math
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Protocol

from .structured_output import StructuredOutputCompiler, StructuredOutputError
from .tokenizer_compiler import TokenPrefixState, Tokenizer


class StructuredOutputDecodeError(StructuredOutputError):
    pass


class LocalModelBridgeError(StructuredOutputDecodeError):
    pass


class LogitProvider(Protocol):
    def next_logits(self, emitted_token_ids: tuple[int, ...], allowed_token_ids: set[int]) -> Mapping[int, float]: ...


def _json_float(value: float | None) -> float | None:
    if value is None or not math.isfinite(value):
        return None
    return float(value)


def _decode_token(tokenizer: Tokenizer | None, token_id: int | None) -> str | None:
    if tokenizer is None or token_id is None:
        return None
    try:
        return tokenizer.decode([token_id])
    except Exception:
        return None


def _decode_tokens(tokenizer: Tokenizer | None, token_ids: tuple[int, ...]) -> str | None:
    if tokenizer is None:
        return None
    try:
        return tokenizer.decode(list(token_ids))
    except Exception:
        return None


@dataclass(frozen=True)
class DecodeEvent:
    step: int
    allowed_count: int
    allowed_token_ids: tuple[int, ...]
    selected_token_id: int
    selected_score: float
    top_illegal_token_id: int | None
    top_illegal_score: float | None
    accepted: bool
    complete_value: str | None

    def to_dict(self, tokenizer: Tokenizer | None = None) -> dict[str, object]:
        return {
            "step": self.step,
            "allowed_count": self.allowed_count,
            "allowed_token_ids": list(self.allowed_token_ids),
            "selected_token_id": self.selected_token_id,
            "selected_token_text": _decode_token(tokenizer, self.selected_token_id),
            "selected_score": _json_float(self.selected_score),
            "selected_was_allowed": self.selected_token_id in self.allowed_token_ids,
            "top_illegal_token_id": self.top_illegal_token_id,
            "top_illegal_token_text": _decode_token(tokenizer, self.top_illegal_token_id),
            "top_illegal_score": _json_float(self.top_illegal_score),
            "accepted": self.accepted,
            "complete_value": self.complete_value,
        }


@dataclass(frozen=True)
class MaskedDecodeResult:
    value: str | None
    parsed: dict[str, object] | None
    emitted_token_ids: tuple[int, ...]
    steps: int
    accepted: bool
    events: tuple[DecodeEvent, ...]

    def to_trace_dict(
        self,
        tokenizer: Tokenizer | None = None,
        scenario: str | Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        if scenario is None:
            scenario_value: object = {}
        elif isinstance(scenario, str):
            scenario_value = {"name": scenario}
        else:
            scenario_value = dict(scenario)
        trace = {
            "schema_version": 1,
            "scenario": scenario_value,
            "accepted": self.accepted,
            "value": self.value,
            "parsed": self.parsed,
            "emitted_token_ids": list(self.emitted_token_ids),
            "emitted_text": _decode_tokens(tokenizer, self.emitted_token_ids),
            "steps": self.steps,
            "events": [event.to_dict(tokenizer) for event in self.events],
        }
        json.dumps(trace, ensure_ascii=False, allow_nan=False)
        return trace


class StructuredOutputDecoder:
    def __init__(self, compiler: StructuredOutputCompiler):
        self.compiler = compiler

    def decode(self, provider: LogitProvider, *, max_steps: int = 512) -> MaskedDecodeResult:
        result, _ = self._decode(
            lambda state, allowed: provider.next_logits(state.emitted, allowed),
            max_steps=max_steps,
        )
        return result

    def decode_with_state_logits(
        self,
        logits_fn: Callable[[TokenPrefixState, set[int]], Mapping[int, float]],
        *,
        max_steps: int = 512,
        stop_on_accepting: bool = True,
    ) -> tuple[MaskedDecodeResult, TokenPrefixState]:
        return self._decode(logits_fn, max_steps=max_steps, stop_on_accepting=stop_on_accepting)

    def _decode(
        self,
        logits_fn: Callable[[TokenPrefixState, set[int]], Mapping[int, float]],
        *,
        max_steps: int,
        stop_on_accepting: bool = True,
    ) -> tuple[MaskedDecodeResult, TokenPrefixState]:
        state = self.compiler.initial_state()
        events: list[DecodeEvent] = []
        for step in range(max_steps):
            allowed = self.compiler.allowed_token_ids(state)
            if self.compiler.is_accepting(state) and (stop_on_accepting or not allowed):
                return self._result(state, events), state
            if not allowed:
                raise StructuredOutputDecodeError("Structured decoder reached empty support before accepting")
            logits = {int(tok): float(score) for tok, score in logits_fn(state, allowed).items()}
            selected = max(sorted(allowed), key=lambda tok: (logits.get(tok, float("-inf")), tok))
            selected_score = logits.get(selected, float("-inf"))
            illegal_scores = [(tok, score) for tok, score in logits.items() if tok not in allowed]
            top_illegal_token_id: int | None = None
            top_illegal_score: float | None = None
            if illegal_scores:
                top_illegal_token_id, top_illegal_score = max(illegal_scores, key=lambda item: (item[1], item[0]))
            state = self.compiler.update(state, selected)
            events.append(
                DecodeEvent(
                    step=step,
                    allowed_count=len(allowed),
                    allowed_token_ids=tuple(sorted(allowed)),
                    selected_token_id=selected,
                    selected_score=selected_score,
                    top_illegal_token_id=top_illegal_token_id,
                    top_illegal_score=top_illegal_score,
                    accepted=self.compiler.is_accepting(state),
                    complete_value=self.compiler.complete_value(state),
                )
            )
        raise StructuredOutputDecodeError(f"Structured decoder exceeded max_steps={max_steps}")

    def _result(self, state: TokenPrefixState, events: list[DecodeEvent]) -> MaskedDecodeResult:
        value = self.compiler.complete_value(state)
        parsed: dict[str, object] | None = None
        if value is not None:
            parsed_value = json.loads(value)
            if isinstance(parsed_value, dict):
                parsed = parsed_value
        return MaskedDecodeResult(
            value=value,
            parsed=parsed,
            emitted_token_ids=state.emitted,
            steps=len(events),
            accepted=self.compiler.is_accepting(state),
            events=tuple(events),
        )


class HostileLogitProvider:
    def __init__(self, illegal_token_ids: list[int] | tuple[int, ...] = (0, 1, 2), *, illegal_score: float = 1_000_000.0):
        self.illegal_token_ids = tuple(illegal_token_ids)
        self.illegal_score = illegal_score

    def next_logits(self, emitted_token_ids: tuple[int, ...], allowed_token_ids: set[int]) -> Mapping[int, float]:
        scores = {tok: float(tok % 997) for tok in allowed_token_ids}
        for tok in self.illegal_token_ids:
            scores[tok] = self.illegal_score
        return scores


class ScriptedLogitProvider:
    def __init__(
        self,
        target_token_ids: list[int] | tuple[int, ...],
        *,
        illegal_token_ids: list[int] | tuple[int, ...] = (),
        target_score: float = 10_000.0,
        illegal_score: float = 1_000_000.0,
    ):
        self.target_token_ids = tuple(target_token_ids)
        self.illegal_token_ids = tuple(illegal_token_ids)
        self.target_score = target_score
        self.illegal_score = illegal_score

    def next_logits(self, emitted_token_ids: tuple[int, ...], allowed_token_ids: set[int]) -> Mapping[int, float]:
        scores = {tok: 0.0 for tok in allowed_token_ids}
        idx = len(emitted_token_ids)
        if idx < len(self.target_token_ids):
            scores[self.target_token_ids[idx]] = self.target_score
        for tok in self.illegal_token_ids:
            scores[tok] = self.illegal_score
        return scores


class CallableLogitProvider:
    def __init__(self, fn: Callable[[tuple[int, ...], set[int]], Mapping[int, float]]):
        self.fn = fn

    def next_logits(self, emitted_token_ids: tuple[int, ...], allowed_token_ids: set[int]) -> Mapping[int, float]:
        return self.fn(emitted_token_ids, allowed_token_ids)


class HFLocalLogitProvider:
    """Experimental offline bridge for already-loaded Hugging Face objects."""

    def __init__(self, model: Any | None = None, tokenizer: Any | None = None, *, device: str = "cpu"):
        if model is None or tokenizer is None:
            raise LocalModelBridgeError("HFLocalLogitProvider requires already-loaded model and tokenizer objects")
        if importlib.util.find_spec("transformers") is None:
            raise LocalModelBridgeError("transformers is required for HFLocalLogitProvider; install .[local-models]")
        try:
            import torch
        except ImportError as exc:
            raise LocalModelBridgeError("torch is required for HFLocalLogitProvider") from exc
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self._torch = torch

    def next_logits(self, emitted_token_ids: tuple[int, ...], allowed_token_ids: set[int]) -> Mapping[int, float]:
        if not allowed_token_ids:
            return {}
        if not emitted_token_ids:
            bos_token_id = getattr(self.tokenizer, "bos_token_id", None)
            if bos_token_id is None:
                return {tok: 0.0 for tok in allowed_token_ids}
            emitted_token_ids = (int(bos_token_id),)
        input_ids = self._torch.tensor([list(emitted_token_ids)], device=self.device)
        try:
            with self._torch.no_grad():
                output = self.model(input_ids=input_ids)
            logits = output.logits[0, -1]
            return {tok: float(logits[tok].detach().cpu()) for tok in allowed_token_ids}
        except Exception as exc:
            raise LocalModelBridgeError(f"HFLocalLogitProvider failed to score logits: {exc}") from exc
