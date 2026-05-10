from __future__ import annotations

from dataclasses import dataclass
from cdsd.decoder import PlannerOutput
from cdsd.masks import EmptySupportError, SupportMask


@dataclass(frozen=True)
class JSONSchemaSpec:
    required: tuple[str, ...]
    properties: dict[str, tuple[str, ...]]

    @classmethod
    def enum_object(cls, properties: dict[str, list[str] | tuple[str, ...]]) -> "JSONSchemaSpec":
        return cls(tuple(properties.keys()), {k: tuple(v) for k, v in properties.items()})


@dataclass(frozen=True)
class JSONSchemaState:
    emitted: tuple[str, ...] = ()
    current_key: str | None = None
    phase: str = "start"


class JSONSchemaPlanner:
    """Planner/guard pair for a deterministic JSON object subset.

    The token model is intentionally simple and tokenizer-ready: punctuation,
    property names, enum values, and EOS are atomic tokens that can be compiled to
    tokenizer IDs by cdsd.tokenizer.compiler.
    """

    EOS = "<eos>"

    def __init__(self, spec: JSONSchemaSpec):
        self.spec = spec

    def step(self, state: JSONSchemaState) -> PlannerOutput:
        winners = _allowed(self.spec, state)
        return PlannerOutput(
            plan_mask=SupportMask.from_iter(winners),
            winners=winners,
            margin=0.0 if len(winners) > 1 else float("inf"),
            control_features={
                "phase": state.phase,
                "emitted_count": len(state.emitted),
                "remaining": [k for k in self.spec.required if k not in state.emitted],
                "current_key": state.current_key,
            },
            trace={"phase": state.phase, "emitted": list(state.emitted), "current_key": state.current_key},
        )


class JSONSchemaGuard:
    def __init__(self, spec: JSONSchemaSpec):
        self.spec = spec

    def mask(self, prefix: list[str], state: JSONSchemaState) -> SupportMask:
        return SupportMask.from_iter(_allowed(self.spec, state))

    def update(self, state: JSONSchemaState, token: str) -> JSONSchemaState:
        allowed = _allowed(self.spec, state)
        if token not in allowed:
            raise EmptySupportError(f"Token {token!r} is illegal in JSON phase {state.phase!r}")
        if state.phase == "start":
            return JSONSchemaState(phase="key")
        if state.phase == "key":
            return JSONSchemaState(emitted=state.emitted, current_key=token, phase="colon")
        if state.phase == "colon":
            return JSONSchemaState(emitted=state.emitted, current_key=state.current_key, phase="value")
        if state.phase == "value":
            assert state.current_key is not None
            return JSONSchemaState(emitted=state.emitted + (state.current_key,), phase="comma_or_end")
        if state.phase == "comma_or_end":
            return JSONSchemaState(emitted=state.emitted, phase="key" if token == "," else "done")
        if state.phase == "done":
            return state
        raise ValueError(f"Unknown phase: {state.phase}")


def _allowed(spec: JSONSchemaSpec, state: JSONSchemaState) -> set[str]:
    if state.phase == "start":
        return {"{"}
    if state.phase == "key":
        return {k for k in spec.required if k not in state.emitted}
    if state.phase == "colon":
        return {":"}
    if state.phase == "value":
        if state.current_key is None:
            return set()
        return set(spec.properties.get(state.current_key, ()))
    if state.phase == "comma_or_end":
        remaining = [k for k in spec.required if k not in state.emitted]
        return {","} if remaining else {"}"}
    if state.phase == "done":
        return {JSONSchemaPlanner.EOS}
    return set()


def render_json_tokens(tokens: list[str]) -> str:
    body = [t for t in tokens if t != JSONSchemaPlanner.EOS]
    out = ""
    for tok in body:
        if tok in {"{", "}", ":", ","}:
            out += tok
        else:
            out += f'"{tok}"'
    return out
