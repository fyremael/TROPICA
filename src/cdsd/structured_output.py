from __future__ import annotations

import itertools
import json
from dataclasses import dataclass
from typing import Any, Callable

from .tokenizer_compiler import TokenPrefixAutomaton, TokenPrefixState, Tokenizer, TokenizerPrefixError


class StructuredOutputError(ValueError):
    pass


class UnboundedSchemaError(StructuredOutputError):
    pass


@dataclass(frozen=True)
class ToolCallSpec:
    name: str
    arguments_schema: dict[str, Any]


class StructuredOutputCompiler:
    def __init__(self, tokenizer: Tokenizer, specs: list[ToolCallSpec] | tuple[ToolCallSpec, ...], *, max_outputs: int = 50_000):
        if not specs:
            raise StructuredOutputError("At least one ToolCallSpec is required")
        self.tokenizer = tokenizer
        self.specs = tuple(specs)
        self.max_outputs = max_outputs
        self.outputs: list[str] = []
        self.output_to_tool: dict[str, str] = {}
        for spec in self.specs:
            for args in enumerate_schema(spec.arguments_schema, max_outputs=max_outputs):
                value = canonical_tool_call(spec.name, args)
                self.outputs.append(value)
                self.output_to_tool[value] = spec.name
                if len(self.outputs) > max_outputs:
                    raise UnboundedSchemaError(f"Structured output enumeration exceeds cap {max_outputs}")
        self.automaton = TokenPrefixAutomaton(tokenizer, self.outputs)

    def initial_state(self) -> TokenPrefixState:
        return self.automaton.initial_state()

    def allowed_token_ids(self, state: TokenPrefixState) -> set[int]:
        return self.automaton.allowed_token_ids(state)

    def update(self, state: TokenPrefixState, token_id: int) -> TokenPrefixState:
        return self.automaton.update(state, token_id)

    def is_accepting(self, state: TokenPrefixState) -> bool:
        return self.automaton.is_accepting(state)

    def complete_value(self, state: TokenPrefixState) -> str | None:
        return state.complete_value

    def parse_complete(self, state: TokenPrefixState) -> dict[str, Any]:
        if state.complete_value is None:
            raise StructuredOutputError("Structured output state is not accepting")
        return json.loads(state.complete_value)

    def matches_declared_tool(self, state: TokenPrefixState) -> bool:
        if state.complete_value is None:
            return False
        try:
            parsed = json.loads(state.complete_value)
        except json.JSONDecodeError:
            return False
        return parsed.get("tool") == self.output_to_tool.get(state.complete_value)


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def canonical_tool_call(name: str, arguments: dict[str, Any]) -> str:
    return canonical_json({"tool": name, "arguments": arguments})


def enumerate_schema(schema: dict[str, Any], *, max_outputs: int = 50_000) -> list[Any]:
    values = _enumerate_schema(schema, path="$")
    if len(values) > max_outputs:
        raise UnboundedSchemaError(f"Schema enumeration exceeds cap {max_outputs}")
    return values


def _enumerate_schema(schema: dict[str, Any], *, path: str) -> list[Any]:
    stype = schema.get("type")
    if "enum" in schema:
        return list(schema["enum"])
    if stype == "object":
        return _enumerate_object(schema, path=path)
    if stype == "array":
        return _enumerate_array(schema, path=path)
    if stype in {"string", "number", "integer", "boolean", "null"}:
        raise UnboundedSchemaError(f"{path}: type {stype!r} requires an enum")
    raise UnboundedSchemaError(f"{path}: unsupported or unbounded schema")


def _enumerate_object(schema: dict[str, Any], *, path: str) -> list[dict[str, Any]]:
    if schema.get("additionalProperties", True) is not False:
        raise UnboundedSchemaError(f"{path}: additionalProperties must be false")
    properties = schema.get("properties")
    if not isinstance(properties, dict):
        raise UnboundedSchemaError(f"{path}: object properties must be specified")
    required = set(schema.get("required", []))
    unknown_required = required - set(properties)
    if unknown_required:
        raise StructuredOutputError(f"{path}: required properties missing schemas: {sorted(unknown_required)}")

    property_options: list[tuple[str, list[Any], bool]] = []
    for key, subschema in properties.items():
        values = _enumerate_schema(subschema, path=f"{path}.{key}")
        property_options.append((key, values, key in required))

    outputs: list[dict[str, Any]] = []

    def visit(idx: int, current: dict[str, Any]) -> None:
        if idx == len(property_options):
            outputs.append(dict(current))
            return
        key, values, is_required = property_options[idx]
        if not is_required:
            visit(idx + 1, current)
        for value in values:
            current[key] = value
            visit(idx + 1, current)
            current.pop(key, None)

    visit(0, {})
    return outputs


def _enumerate_array(schema: dict[str, Any], *, path: str) -> list[list[Any]]:
    min_items = int(schema.get("minItems", 0))
    if "maxItems" not in schema:
        raise UnboundedSchemaError(f"{path}: arrays require maxItems")
    max_items = int(schema["maxItems"])
    if min_items < 0 or max_items < min_items or max_items > 4:
        raise UnboundedSchemaError(f"{path}: arrays require 0 <= minItems <= maxItems <= 4")
    item_schema = schema.get("items")
    if not isinstance(item_schema, dict):
        raise UnboundedSchemaError(f"{path}: arrays require an item schema")
    item_values = _enumerate_schema(item_schema, path=f"{path}[]")
    outputs: list[list[Any]] = []
    for length in range(min_items, max_items + 1):
        for combo in itertools.product(item_values, repeat=length):
            outputs.append(list(combo))
    return outputs


def decode_with_logits(
    compiler: StructuredOutputCompiler,
    logits_fn: Callable[[TokenPrefixState, set[int]], dict[int, float]],
    *,
    max_steps: int = 512,
) -> TokenPrefixState:
    state = compiler.initial_state()
    for _ in range(max_steps):
        if compiler.is_accepting(state) and not compiler.allowed_token_ids(state):
            return state
        allowed = compiler.allowed_token_ids(state)
        if not allowed:
            raise StructuredOutputError("Structured decoder reached empty support before accepting")
        logits = logits_fn(state, allowed)
        token_id = max(allowed, key=lambda tok: logits.get(tok, float("-inf")))
        state = compiler.update(state, token_id)
    raise StructuredOutputError("Structured decoder exceeded max_steps")


class HostileStructuredLogitGenerator:
    def __init__(self, illegal_token_ids: list[int] | tuple[int, ...] = (0, 1, 2)):
        self.illegal_token_ids = tuple(illegal_token_ids)

    def logits(self, state: TokenPrefixState, allowed: set[int]) -> dict[int, float]:
        scores = {tok: float(tok % 997) for tok in allowed}
        for tok in self.illegal_token_ids:
            scores[tok] = 1_000_000.0
        return scores


class HFLogitGenerator:
    """Optional local-model bridge. CI uses HostileStructuredLogitGenerator."""

    def __init__(self, model, device: str = "cpu"):
        self.model = model
        self.device = device

    def logits(self, state: TokenPrefixState, allowed: set[int]) -> dict[int, float]:
        try:
            import torch
        except ImportError as exc:
            raise StructuredOutputError("torch is required for HFLogitGenerator") from exc
        if not state.emitted:
            return {tok: 0.0 for tok in allowed}
        input_ids = torch.tensor([list(state.emitted)], device=self.device)
        with torch.no_grad():
            out = self.model(input_ids=input_ids)
        next_logits = out.logits[0, -1]
        return {tok: float(next_logits[tok].detach().cpu()) for tok in allowed}
