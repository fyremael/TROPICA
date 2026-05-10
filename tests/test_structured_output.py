import json
import importlib.util

import pytest

from cdsd.structured_output import (
    HostileStructuredLogitGenerator,
    StructuredOutputCompiler,
    ToolCallSpec,
    UnboundedSchemaError,
    canonical_tool_call,
    decode_with_logits,
    enumerate_schema,
)
from cdsd.tokenizer_compiler import ByteTokenizer, HFTokenizerAdapter, TiktokenAdapter, TokenizerPrefixError


def schema():
    return {
        "type": "object",
        "required": ["query", "limit"],
        "properties": {
            "query": {"type": "string", "enum": ["alpha", "quote \"x\"", "snow ☃"]},
            "limit": {"type": "integer", "enum": [1, 2]},
            "fresh": {"type": "boolean", "enum": [True, False]},
            "tags": {"type": "array", "minItems": 1, "maxItems": 2, "items": {"type": "string", "enum": ["a", "b"]}},
            "nothing": {"type": "null", "enum": [None]},
        },
        "additionalProperties": False,
    }


def test_canonical_json_enumeration_required_optional_and_arrays():
    values = enumerate_schema(schema())
    assert len(values) == 3 * 2 * 3 * 7 * 2
    first = values[0]
    assert list(first) == ["query", "limit"]
    rendered = canonical_tool_call("search", first)
    assert json.loads(rendered)["tool"] == "search"
    assert rendered.startswith('{"tool":"search","arguments":{"query"')


def test_structured_compiler_decodes_with_hostile_logits():
    compiler = StructuredOutputCompiler(ByteTokenizer(), [ToolCallSpec("search", schema())])
    state = decode_with_logits(compiler, HostileStructuredLogitGenerator().logits, max_steps=256)
    assert compiler.is_accepting(state)
    assert compiler.matches_declared_tool(state)
    parsed = compiler.parse_complete(state)
    assert parsed["tool"] == "search"
    assert set(parsed["arguments"]).issubset(set(schema()["properties"]))


def test_negative_controls_illegal_suffix_unknown_key_and_unbounded_schema():
    compiler = StructuredOutputCompiler(ByteTokenizer(), [ToolCallSpec("search", schema())])
    state = compiler.initial_state()
    with pytest.raises(TokenizerPrefixError):
        compiler.update(state, 255)
    truncated = compiler.update(state, ord("{"))
    assert not compiler.is_accepting(truncated)
    with pytest.raises(UnboundedSchemaError):
        enumerate_schema({"type": "object", "properties": {}, "additionalProperties": True})
    with pytest.raises(UnboundedSchemaError):
        enumerate_schema({"type": "object", "required": ["x"], "properties": {"x": {"type": "string"}}, "additionalProperties": False})
    with pytest.raises(UnboundedSchemaError):
        enumerate_schema({"type": "array", "minItems": 0, "maxItems": 5, "items": {"type": "string", "enum": ["x"]}})


def test_enumeration_cap_overflow_rejected():
    huge = {
        "type": "object",
        "required": ["x"],
        "properties": {"x": {"type": "array", "minItems": 4, "maxItems": 4, "items": {"type": "string", "enum": [str(i) for i in range(20)]}}},
        "additionalProperties": False,
    }
    with pytest.raises(UnboundedSchemaError):
        StructuredOutputCompiler(ByteTokenizer(), [ToolCallSpec("huge", huge)], max_outputs=50_000)


@pytest.mark.skipif(importlib.util.find_spec("tiktoken") is None, reason="tiktoken not installed")
def test_structured_compiler_tiktoken_exact_generation():
    compiler = StructuredOutputCompiler(TiktokenAdapter("cl100k_base"), [ToolCallSpec("search", schema())])
    state = decode_with_logits(compiler, HostileStructuredLogitGenerator().logits, max_steps=256)
    assert compiler.matches_declared_tool(state)


@pytest.mark.skipif(importlib.util.find_spec("tokenizers") is None, reason="tokenizers not installed")
def test_structured_compiler_hf_tokenizer_exact_generation():
    from demos.run_structured_output_harness import make_hf_structured_adapter

    specs = [ToolCallSpec("search", schema())]
    literals = [canonical_tool_call("search", args) for args in enumerate_schema(schema())]
    compiler = StructuredOutputCompiler(make_hf_structured_adapter(literals), specs)
    state = decode_with_logits(compiler, HostileStructuredLogitGenerator().logits, max_steps=512)
    assert compiler.matches_declared_tool(state)
