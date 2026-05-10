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

from cdsd.model_integration import CallableLogitProvider, HostileLogitProvider, ScriptedLogitProvider, StructuredOutputDecodeError, StructuredOutputDecoder
from cdsd.structured_output import StructuredOutputCompiler
from cdsd.tokenizer_compiler import TiktokenAdapter, TokenizerPrefixError
from demos.run_structured_output_harness import tool_specs


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = Path(os.environ.get("CDSD_ARTIFACT_DIR", ROOT / "artifacts"))
OUT_CSV = ARTIFACT_DIR / "model_integration_summary.csv"
OUT_MD = ARTIFACT_DIR / "model_integration_summary.md"


@dataclass
class ModelIntegrationResult:
    provider: str
    adapter: str
    suite: str
    cases: int
    failures: int
    duration_ms: float
    outputs: int
    trace_steps: int
    notes: str


def require_deps() -> None:
    if importlib.util.find_spec("tiktoken") is None:
        raise RuntimeError("Missing model-integration tokenizer dependency: tiktoken")


def illegal_token_ids() -> tuple[int, ...]:
    return (999_999_001, 999_999_002, 999_999_003)


def make_compiler() -> tuple[TiktokenAdapter, StructuredOutputCompiler]:
    adapter = TiktokenAdapter("cl100k_base")
    return adapter, StructuredOutputCompiler(adapter, tool_specs())


def valid_result(compiler: StructuredOutputCompiler, result) -> bool:
    if not result.accepted or result.value is None or result.parsed is None:
        return False
    if result.value not in compiler.output_to_tool:
        return False
    if result.parsed.get("tool") != compiler.output_to_tool[result.value]:
        return False
    try:
        json.loads(result.value)
    except json.JSONDecodeError:
        return False
    return True


def run_scripted_exact(adapter: TiktokenAdapter, compiler: StructuredOutputCompiler) -> ModelIntegrationResult:
    start = time.perf_counter()
    decoder = StructuredOutputDecoder(compiler)
    failures = 0
    trace_steps = 0
    for value in compiler.outputs:
        provider = ScriptedLogitProvider(adapter.encode(value), illegal_token_ids=illegal_token_ids())
        try:
            result = decoder.decode(provider, max_steps=1024)
            trace_steps += result.steps
            if not valid_result(compiler, result) or result.value != value:
                failures += 1
        except Exception:
            failures += 1
    duration_ms = (time.perf_counter() - start) * 1000.0
    return ModelIntegrationResult("scripted", adapter.name, "exact scripted tool calls", len(compiler.outputs), failures, duration_ms, len(compiler.outputs), trace_steps, "scripted provider targets every canonical output")


def run_hostile_decode(adapter: TiktokenAdapter, compiler: StructuredOutputCompiler) -> ModelIntegrationResult:
    start = time.perf_counter()
    decoder = StructuredOutputDecoder(compiler)
    provider = HostileLogitProvider(illegal_token_ids=illegal_token_ids())
    failures = 0
    trace_steps = 0
    cases = 500
    for _ in range(cases):
        try:
            result = decoder.decode(provider, max_steps=1024)
            trace_steps += result.steps
            if not valid_result(compiler, result):
                failures += 1
            if not any(event.top_illegal_score is not None and event.top_illegal_score > event.selected_score for event in result.events):
                failures += 1
        except Exception:
            failures += 1
    duration_ms = (time.perf_counter() - start) * 1000.0
    return ModelIntegrationResult("hostile", adapter.name, "illegal logits fail closed", cases, failures, duration_ms, len(compiler.outputs), trace_steps, "illegal token IDs outrank legal IDs but cannot be selected")


def run_callable_smoke(adapter: TiktokenAdapter, compiler: StructuredOutputCompiler) -> ModelIntegrationResult:
    start = time.perf_counter()
    decoder = StructuredOutputDecoder(compiler)
    failures = 0
    trace_steps = 0
    callback_calls = 0

    def callable_logits(emitted: tuple[int, ...], allowed: set[int]):
        nonlocal callback_calls
        callback_calls += 1
        scores = {tok: float((tok * 31 + len(emitted)) % 1009) for tok in allowed}
        for tok in illegal_token_ids():
            scores[tok] = 1_000_000.0
        return scores

    provider = CallableLogitProvider(callable_logits)
    cases = 250
    for _ in range(cases):
        before = callback_calls
        try:
            result = decoder.decode(provider, max_steps=1024)
            trace_steps += result.steps
            if not valid_result(compiler, result) or callback_calls <= before:
                failures += 1
        except Exception:
            failures += 1
    duration_ms = (time.perf_counter() - start) * 1000.0
    return ModelIntegrationResult("callable", adapter.name, "callable adapter smoke", cases, failures, duration_ms, len(compiler.outputs), trace_steps, "user-supplied callback receives prefix and allowed set")


def run_negative_controls(adapter: TiktokenAdapter, compiler: StructuredOutputCompiler) -> ModelIntegrationResult:
    start = time.perf_counter()
    decoder = StructuredOutputDecoder(compiler)
    failures = 0
    cases = 4
    try:
        decoder.decode(HostileLogitProvider(illegal_token_ids()), max_steps=1)
        failures += 1
    except StructuredOutputDecodeError:
        pass
    try:
        compiler.update(compiler.initial_state(), illegal_token_ids()[0])
        failures += 1
    except TokenizerPrefixError:
        pass
    state = compiler.initial_state()
    for token_id in adapter.encode(compiler.outputs[0])[:-1]:
        state = compiler.update(state, token_id)
    if compiler.is_accepting(state):
        failures += 1
    invalid_value = '{"tool":"unknown","arguments":{"query":"alpha"}}'
    state = compiler.initial_state()
    try:
        for token_id in adapter.encode(invalid_value):
            state = compiler.update(state, token_id)
        failures += 1
    except TokenizerPrefixError:
        pass
    duration_ms = (time.perf_counter() - start) * 1000.0
    return ModelIntegrationResult("negative-controls", adapter.name, "fail-closed controls", cases, failures, duration_ms, len(compiler.outputs), 0, "max steps, illegal transition, truncation, schema mismatch")


def write_results(results: list[ModelIntegrationResult]) -> None:
    OUT_CSV.parent.mkdir(exist_ok=True)
    fields = ["Provider", "Adapter", "Suite", "Cases", "Failures", "DurationMs", "Outputs", "TraceSteps", "Notes"]
    with OUT_CSV.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "Provider": result.provider,
                    "Adapter": result.adapter,
                    "Suite": result.suite,
                    "Cases": result.cases,
                    "Failures": result.failures,
                    "DurationMs": f"{result.duration_ms:.3f}",
                    "Outputs": result.outputs,
                    "TraceSteps": result.trace_steps,
                    "Notes": result.notes,
                }
            )
    with OUT_MD.open("w", encoding="utf-8") as fh:
        fh.write("| Provider | Adapter | Suite | Cases | Failures | DurationMs | Outputs | TraceSteps | Notes |\n")
        fh.write("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |\n")
        for result in results:
            fh.write(
                f"| {result.provider} | {result.adapter} | {result.suite} | {result.cases} | {result.failures} | "
                f"{result.duration_ms:.3f} | {result.outputs} | {result.trace_steps} | {result.notes} |\n"
            )


if __name__ == "__main__":
    require_deps()
    adapter, compiler = make_compiler()
    results = [
        run_scripted_exact(adapter, compiler),
        run_hostile_decode(adapter, compiler),
        run_callable_smoke(adapter, compiler),
        run_negative_controls(adapter, compiler),
    ]
    write_results(results)
    for result in results:
        print(
            f"{result.provider}: suite={result.suite} cases={result.cases} failures={result.failures} "
            f"duration_ms={result.duration_ms:.3f} outputs={result.outputs} trace_steps={result.trace_steps} notes={result.notes}"
        )
    print(f"Wrote {OUT_CSV} and {OUT_MD}")
    if any(result.failures for result in results):
        raise SystemExit(1)
