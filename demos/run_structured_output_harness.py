from __future__ import annotations

import csv
import importlib.util
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cdsd.structured_output import (
    HostileStructuredLogitGenerator,
    StructuredOutputCompiler,
    ToolCallSpec,
    UnboundedSchemaError,
    canonical_tool_call,
    decode_with_logits,
    enumerate_schema,
)
from cdsd.tokenizer_compiler import HFTokenizerAdapter, TiktokenAdapter, TokenizerPrefixError


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = Path(os.environ.get("CDSD_ARTIFACT_DIR", ROOT / "artifacts"))
OUT_CSV = ARTIFACT_DIR / "structured_output_summary.csv"
OUT_MD = ARTIFACT_DIR / "structured_output_summary.md"


@dataclass
class StructuredResult:
    adapter: str
    suite: str
    cases: int
    failures: int
    duration_ms: float
    outputs: int
    notes: str


def require_deps() -> None:
    missing = [name for name in ["tiktoken", "tokenizers"] if importlib.util.find_spec(name) is None]
    if missing:
        raise RuntimeError(f"Missing structured-output tokenizer dependencies: {', '.join(missing)}")


def tool_specs() -> list[ToolCallSpec]:
    locations = [f"node_{i:02d}" for i in range(12)]
    return [
        ToolCallSpec(
            "search",
            {
                "type": "object",
                "required": ["query", "limit"],
                "properties": {
                    "query": {"type": "string", "enum": [f"topic {i}" for i in range(15)] + ['quote "x"', "snow \u2603"]},
                    "limit": {"type": "integer", "enum": [1, 3, 5]},
                    "fresh": {"type": "boolean", "enum": [True, False]},
                },
                "additionalProperties": False,
            },
        ),
        ToolCallSpec(
            "summarize",
            {
                "type": "object",
                "required": ["doc", "style"],
                "properties": {
                    "doc": {"type": "string", "enum": [f"doc_{i:02d}" for i in range(20)]},
                    "style": {"type": "string", "enum": ["brief", "bullets", "technical"]},
                    "include_quotes": {"type": "boolean", "enum": [True, False]},
                },
                "additionalProperties": False,
            },
        ),
        ToolCallSpec(
            "route",
            {
                "type": "object",
                "required": ["origin", "dest", "mode"],
                "properties": {
                    "origin": {"type": "string", "enum": locations},
                    "dest": {"type": "string", "enum": locations},
                    "mode": {"type": "string", "enum": ["walk", "bike", "train"]},
                    "avoid": {"type": "array", "minItems": 0, "maxItems": 2, "items": {"type": "string", "enum": ["stairs", "tolls", "rain"]}},
                },
                "additionalProperties": False,
            },
        ),
        ToolCallSpec(
            "write_file",
            {
                "type": "object",
                "required": ["path", "mode", "content"],
                "properties": {
                    "path": {"type": "string", "enum": [f"reports/out_{i:02d}.txt" for i in range(24)]},
                    "mode": {"type": "string", "enum": ["create", "append"]},
                    "content": {"type": "string", "enum": ["alpha", "line\nbreak", "tab\tvalue", "unicode \u00e9", "json {}"]},
                    "dry_run": {"type": "boolean", "enum": [True, False]},
                },
                "additionalProperties": False,
            },
        ),
    ]


def all_literals(specs: list[ToolCallSpec]) -> list[str]:
    out: list[str] = []
    for spec in specs:
        for args in enumerate_schema(spec.arguments_schema):
            out.append(canonical_tool_call(spec.name, args))
    return out


def make_hf_structured_adapter(literals: list[str]) -> HFTokenizerAdapter:
    from tokenizers import Tokenizer
    from tokenizers import decoders
    from tokenizers.models import BPE
    from tokenizers.pre_tokenizers import WhitespaceSplit

    alphabet = sorted({ch for literal in literals for word in literal.split() for ch in word})
    vocab = {"[UNK]": 0}
    for ch in alphabet:
        vocab.setdefault(ch, len(vocab))
        vocab.setdefault(f"{ch}</w>", len(vocab))
    tokenizer = Tokenizer(BPE(vocab=vocab, merges=[], unk_token="[UNK]", end_of_word_suffix="</w>"))
    tokenizer.pre_tokenizer = WhitespaceSplit()
    tokenizer.decoder = decoders.BPEDecoder(suffix="</w>")
    return HFTokenizerAdapter(tokenizer, name="hf/structured-bpe")


def validate_value(value: str, tool_names: set[str]) -> bool:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return False
    return isinstance(parsed, dict) and parsed.get("tool") in tool_names and isinstance(parsed.get("arguments"), dict)


def run_positive(adapter, specs: list[ToolCallSpec], suite: str) -> StructuredResult:
    start = time.perf_counter()
    failures = 0
    try:
        compiler = StructuredOutputCompiler(adapter, specs)
        tool_names = {spec.name for spec in specs}
        for value in compiler.outputs:
            if not validate_value(value, tool_names):
                failures += 1
            state = compiler.initial_state()
            for token_id in adapter.encode(value):
                if token_id not in compiler.allowed_token_ids(state):
                    failures += 1
                    break
                state = compiler.update(state, token_id)
            if not compiler.is_accepting(state) or compiler.complete_value(state) != value or not compiler.matches_declared_tool(state):
                failures += 1
    except Exception:
        failures += len(all_literals(specs))
        compiler = None
    duration_ms = (time.perf_counter() - start) * 1000.0
    outputs = len(compiler.outputs) if compiler is not None else 0
    return StructuredResult(adapter.name, suite, outputs, failures, duration_ms, outputs, "exact structured tool-call literals")


def run_hostile_decode(adapter, specs: list[ToolCallSpec], suite: str) -> StructuredResult:
    start = time.perf_counter()
    failures = 0
    cases = 200
    compiler = StructuredOutputCompiler(adapter, specs)
    generator = HostileStructuredLogitGenerator(illegal_token_ids=(0, 1, 2, 3, 4, 5))
    for _ in range(cases):
        try:
            state = decode_with_logits(compiler, generator.logits, max_steps=1024)
            if not compiler.is_accepting(state) or not compiler.matches_declared_tool(state):
                failures += 1
        except Exception:
            failures += 1
    duration_ms = (time.perf_counter() - start) * 1000.0
    return StructuredResult(adapter.name, suite, cases, failures, duration_ms, len(compiler.outputs), "hostile logits choose only legal token IDs")


def run_negative_controls(adapter, specs: list[ToolCallSpec]) -> StructuredResult:
    start = time.perf_counter()
    failures = 0
    cases = 5
    compiler = StructuredOutputCompiler(adapter, specs)
    state = compiler.initial_state()
    try:
        compiler.update(state, 999_999_999)
        failures += 1
    except TokenizerPrefixError:
        pass
    first_value = compiler.outputs[0]
    ids = adapter.encode(first_value)
    truncated = state
    for token_id in ids[:-1]:
        truncated = compiler.update(truncated, token_id)
    if compiler.is_accepting(truncated):
        failures += 1
    for bad in [
        first_value + "_illegal_suffix",
        first_value[:-1] + ',"extra":true}',
        '{"tool":"unknown","arguments":{}}',
    ]:
        probe = compiler.initial_state()
        try:
            for token_id in adapter.encode(bad):
                probe = compiler.update(probe, token_id)
            failures += 1
        except TokenizerPrefixError:
            pass
    duration_ms = (time.perf_counter() - start) * 1000.0
    return StructuredResult(adapter.name, "negative controls", cases, failures, duration_ms, len(compiler.outputs), "invalid token, truncated JSON, suffix, extra key, unknown tool")


def run_unbounded_controls() -> StructuredResult:
    start = time.perf_counter()
    failures = 0
    checks = [
        lambda: enumerate_schema({"type": "string"}),
        lambda: enumerate_schema({"type": "number"}),
        lambda: enumerate_schema({"type": "object", "properties": {}, "additionalProperties": True}),
        lambda: enumerate_schema({"type": "array", "minItems": 0, "items": {"type": "string", "enum": ["x"]}}),
        lambda: enumerate_schema({"type": "array", "minItems": 0, "maxItems": 5, "items": {"type": "string", "enum": ["x"]}}),
    ]
    for check in checks:
        try:
            check()
            failures += 1
        except UnboundedSchemaError:
            pass
    duration_ms = (time.perf_counter() - start) * 1000.0
    return StructuredResult("schema-controls", "unbounded rejection", len(checks), failures, duration_ms, 0, "free strings, numbers, additional properties, unbounded arrays")


def write_results(results: list[StructuredResult]) -> None:
    OUT_CSV.parent.mkdir(exist_ok=True)
    fields = ["Adapter", "Suite", "Cases", "Failures", "DurationMs", "Outputs", "Notes"]
    with OUT_CSV.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "Adapter": result.adapter,
                    "Suite": result.suite,
                    "Cases": result.cases,
                    "Failures": result.failures,
                    "DurationMs": f"{result.duration_ms:.3f}",
                    "Outputs": result.outputs,
                    "Notes": result.notes,
                }
            )
    with OUT_MD.open("w", encoding="utf-8") as fh:
        fh.write("| Adapter | Suite | Cases | Failures | DurationMs | Outputs | Notes |\n")
        fh.write("| --- | --- | ---: | ---: | ---: | ---: | --- |\n")
        for result in results:
            fh.write(
                f"| {result.adapter} | {result.suite} | {result.cases} | {result.failures} | "
                f"{result.duration_ms:.3f} | {result.outputs} | {result.notes} |\n"
            )


if __name__ == "__main__":
    require_deps()
    specs = tool_specs()
    literals = all_literals(specs)
    adapters = [TiktokenAdapter("cl100k_base"), make_hf_structured_adapter(literals)]
    results: list[StructuredResult] = []
    for adapter in adapters:
        results.append(run_positive(adapter, specs, "exact structured outputs"))
        results.append(run_hostile_decode(adapter, specs, "hostile decode"))
        results.append(run_negative_controls(adapter, specs))
    results.append(run_unbounded_controls())
    write_results(results)
    for result in results:
        print(
            f"{result.adapter}: suite={result.suite} cases={result.cases} failures={result.failures} "
            f"duration_ms={result.duration_ms:.3f} outputs={result.outputs} notes={result.notes}"
        )
    print(f"Wrote {OUT_CSV} and {OUT_MD}")
    if any(result.failures for result in results):
        raise SystemExit(1)
