import json
import importlib.util

import pytest

from cdsd.model_integration import CallableLogitProvider, HFLocalLogitProvider, HostileLogitProvider, LocalModelBridgeError, ScriptedLogitProvider, StructuredOutputDecodeError, StructuredOutputDecoder
from cdsd.structured_output import HostileStructuredLogitGenerator, StructuredOutputCompiler, ToolCallSpec, decode_with_logits
from cdsd.tokenizer_compiler import ByteTokenizer


def schema():
    return {
        "type": "object",
        "required": ["query", "limit"],
        "properties": {
            "query": {"type": "string", "enum": ["alpha", "beta"]},
            "limit": {"type": "integer", "enum": [1, 2]},
        },
        "additionalProperties": False,
    }


def compiler() -> StructuredOutputCompiler:
    return StructuredOutputCompiler(ByteTokenizer(), [ToolCallSpec("search", schema())])


def test_hostile_provider_never_selects_illegal_token():
    dec = StructuredOutputDecoder(compiler())
    result = dec.decode(HostileLogitProvider(illegal_token_ids=(ord("}"), 999_999)), max_steps=256)

    assert result.accepted
    assert result.parsed is not None
    assert result.parsed["tool"] == "search"
    assert any(event.top_illegal_score is not None and event.top_illegal_score > event.selected_score for event in result.events)
    assert all(event.selected_token_id != 999_999 for event in result.events)


def test_scripted_provider_produces_exact_tool_call():
    comp = compiler()
    target = comp.outputs[0]
    provider = ScriptedLogitProvider(ByteTokenizer().encode(target), illegal_token_ids=(255,))
    result = StructuredOutputDecoder(comp).decode(provider, max_steps=256)

    assert result.accepted
    assert result.value == target
    assert result.parsed == json.loads(target)
    assert tuple(ByteTokenizer().encode(target)) == result.emitted_token_ids


def test_callable_provider_receives_prefix_and_allowed_set():
    calls: list[tuple[tuple[int, ...], set[int]]] = []

    def choose_min(emitted: tuple[int, ...], allowed: set[int]):
        calls.append((emitted, set(allowed)))
        chosen = min(allowed)
        return {tok: 0.0 for tok in allowed} | {chosen: 1.0, 999_999: 1_000_000.0}

    result = StructuredOutputDecoder(compiler()).decode(CallableLogitProvider(choose_min), max_steps=256)

    assert result.accepted
    assert len(calls) == result.steps
    assert calls[0][0] == ()
    assert all(allowed for _, allowed in calls)


def test_trace_events_are_complete_and_ordered():
    result = StructuredOutputDecoder(compiler()).decode(HostileLogitProvider(), max_steps=256)

    assert [event.step for event in result.events] == list(range(result.steps))
    assert all(event.allowed_count > 0 for event in result.events)
    assert result.events[-1].accepted
    assert result.events[-1].complete_value == result.value


def test_trace_serialization_is_json_safe_and_ordered():
    comp = compiler()
    result = StructuredOutputDecoder(comp).decode(HostileLogitProvider(), max_steps=256)
    trace = result.to_trace_dict(ByteTokenizer(), scenario={"provider": "hostile", "suite": "unit"})

    json.dumps(trace, allow_nan=False)
    assert trace["schema_version"] == 1
    assert trace["accepted"] is True
    assert trace["value"] == result.value
    assert trace["parsed"] == result.parsed
    assert trace["emitted_token_ids"] == list(result.emitted_token_ids)
    assert [event["step"] for event in trace["events"]] == list(range(result.steps))
    assert all(event["selected_was_allowed"] for event in trace["events"])
    assert all(event["selected_token_id"] in event["allowed_token_ids"] for event in trace["events"])


def test_hostile_trace_records_illegal_token_pressure():
    result = StructuredOutputDecoder(compiler()).decode(HostileLogitProvider(illegal_token_ids=(999_999,)), max_steps=256)
    trace = result.to_trace_dict(ByteTokenizer(), scenario="hostile")

    assert any(
        event["top_illegal_token_id"] == 999_999
        and event["top_illegal_score"] is not None
        and event["selected_score"] is not None
        and event["top_illegal_score"] > event["selected_score"]
        for event in trace["events"]
    )


def test_max_step_exhaustion_raises_typed_error():
    with pytest.raises(StructuredOutputDecodeError):
        StructuredOutputDecoder(compiler()).decode(HostileLogitProvider(), max_steps=1)


def test_hf_local_provider_requires_loaded_objects():
    with pytest.raises(LocalModelBridgeError):
        HFLocalLogitProvider(model=None, tokenizer=object())


def test_hf_local_provider_requires_transformers(monkeypatch):
    real_find_spec = importlib.util.find_spec

    def fake_find_spec(name, *args, **kwargs):
        if name == "transformers":
            return None
        return real_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)
    with pytest.raises(LocalModelBridgeError):
        HFLocalLogitProvider(model=object(), tokenizer=object())


def test_decode_with_logits_compatibility_wrapper_returns_state():
    comp = compiler()
    state = decode_with_logits(comp, HostileStructuredLogitGenerator().logits, max_steps=256)

    assert comp.is_accepting(state)
    assert comp.matches_declared_tool(state)
