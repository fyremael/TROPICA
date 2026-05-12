from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


REQUIRED_ARTIFACTS = [
    "experiment_summary.csv",
    "experiment_summary.md",
    "experiment_visuals.svg",
    "stress_summary.csv",
    "stress_summary.md",
    "stress_visuals.svg",
    "scale_summary.csv",
    "scale_summary.md",
    "scale_visuals.svg",
    "tokenizer_correctness_summary.csv",
    "tokenizer_correctness_summary.md",
    "tokenizer_correctness_visuals.svg",
    "structured_output_summary.csv",
    "structured_output_summary.md",
    "structured_output_visuals.svg",
    "model_integration_summary.csv",
    "model_integration_summary.md",
    "model_integration_visuals.svg",
    "model_integration_traces.jsonl",
    "trace_explorer.html",
]


@dataclass(frozen=True)
class GateResult:
    name: str
    passed: bool
    detail: str

    def to_dict(self) -> dict[str, str | bool]:
        return asdict(self)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def validate_command_results(command_results: Iterable[dict[str, object]]) -> list[GateResult]:
    gates = []
    for result in command_results:
        name = str(result["name"])
        code = int(result["returncode"])
        gates.append(GateResult(f"command:{name}", code == 0, f"returncode={code}"))
    return gates


def validate_required_artifacts(artifact_dir: Path, required: Iterable[str] = REQUIRED_ARTIFACTS) -> list[GateResult]:
    gates = []
    for name in required:
        path = artifact_dir / name
        exists = path.exists() and path.is_file() and path.stat().st_size > 0
        size = path.stat().st_size if path.exists() else 0
        gates.append(GateResult(f"artifact:{name}", exists, f"size={size}"))
    return gates


def validate_experiment(rows: list[dict[str, str]]) -> list[GateResult]:
    by_mode = {row["Mode"]: row for row in rows}

    def metric(mode: str, name: str) -> float:
        return float(by_mode[mode][name])

    gates = [
        GateResult("experiment:planner_guided_invalid_zero", metric("planner_guided", "InvalidRate") == 0.0, f"InvalidRate={metric('planner_guided', 'InvalidRate')}"),
        GateResult(
            "experiment:control_delta_plus_external_invalid_zero",
            metric("control_delta_plus_external", "InvalidRate") == 0.0,
            f"InvalidRate={metric('control_delta_plus_external', 'InvalidRate')}",
        ),
        GateResult("experiment:raw_generator_invalid_high", metric("raw_generator", "InvalidRate") > 0.5, f"InvalidRate={metric('raw_generator', 'InvalidRate')}"),
        GateResult(
            "experiment:control_delta_only_invalid_high",
            metric("control_delta_only", "InvalidRate") > 0.5,
            f"InvalidRate={metric('control_delta_only', 'InvalidRate')}",
        ),
    ]
    empty_support_values = [float(row["EmptySupportRate"]) for row in rows]
    gates.append(
        GateResult(
            "experiment:empty_support_zero",
            all(value == 0.0 for value in empty_support_values),
            f"max={max(empty_support_values) if empty_support_values else 'missing'}",
        )
    )
    return gates


def validate_stress(rows: list[dict[str, str]]) -> list[GateResult]:
    failures = [int(float(row["Failures"])) for row in rows]
    total_cases = sum(int(float(row["Cases"])) for row in rows)
    return [
        GateResult("stress:failures_zero", all(value == 0 for value in failures), f"total_failures={sum(failures)}"),
        GateResult("stress:case_floor", total_cases >= 6000, f"cases={total_cases}"),
    ]


def validate_scale(rows: list[dict[str, str]]) -> list[GateResult]:
    failures = [int(float(row["Failures"])) for row in rows]

    def max_size(track: str) -> int:
        sizes = [int(float(row["Size"])) for row in rows if row["Track"] == track]
        return max(sizes) if sizes else 0

    return [
        GateResult("scale:failures_zero", all(value == 0 for value in failures), f"total_failures={sum(failures)}"),
        GateResult("scale:dyck_horizon_floor", max_size("Dyck horizon") >= 1024, f"max={max_size('Dyck horizon')}"),
        GateResult("scale:tokenizer_enum_floor", max_size("Tokenizer enums") >= 4096, f"max={max_size('Tokenizer enums')}"),
        GateResult("scale:workflow_node_floor", max_size("Workflow nodes") >= 2048, f"max={max_size('Workflow nodes')}"),
        GateResult("scale:control_delta_sequence_floor", max_size("ControlDelta tokens") >= 1024, f"max={max_size('ControlDelta tokens')}"),
    ]


def validate_tokenizer_correctness(rows: list[dict[str, str]]) -> list[GateResult]:
    failures = [int(float(row["Failures"])) for row in rows]
    total_cases = sum(int(float(row["Cases"])) for row in rows)
    adapters = {row["Adapter"] for row in rows}
    has_tiktoken = any(adapter.startswith("tiktoken/") for adapter in adapters)
    has_hf = any(adapter.startswith("hf/") for adapter in adapters)
    return [
        GateResult("tokenizer:failures_zero", all(value == 0 for value in failures), f"total_failures={sum(failures)}"),
        GateResult("tokenizer:adapter_tiktoken_present", has_tiktoken, f"adapters={sorted(adapters)}"),
        GateResult("tokenizer:adapter_hf_present", has_hf, f"adapters={sorted(adapters)}"),
        GateResult("tokenizer:case_floor", total_cases >= 5000, f"cases={total_cases}"),
    ]


def validate_structured_output(rows: list[dict[str, str]]) -> list[GateResult]:
    failures = [int(float(row["Failures"])) for row in rows]
    total_cases = sum(int(float(row["Cases"])) for row in rows)
    adapters = {row["Adapter"] for row in rows}
    has_tiktoken = any(adapter.startswith("tiktoken/") for adapter in adapters)
    has_hf = any(adapter.startswith("hf/") for adapter in adapters)
    has_schema_controls = "schema-controls" in adapters
    return [
        GateResult("structured:failures_zero", all(value == 0 for value in failures), f"total_failures={sum(failures)}"),
        GateResult("structured:case_floor", total_cases >= 5000, f"cases={total_cases}"),
        GateResult("structured:adapter_tiktoken_present", has_tiktoken, f"adapters={sorted(adapters)}"),
        GateResult("structured:adapter_hf_present", has_hf, f"adapters={sorted(adapters)}"),
        GateResult("structured:schema_controls_present", has_schema_controls, f"adapters={sorted(adapters)}"),
    ]


def validate_model_integration(rows: list[dict[str, str]]) -> list[GateResult]:
    failures = [int(float(row["Failures"])) for row in rows]
    total_cases = sum(int(float(row["Cases"])) for row in rows)
    providers = {row["Provider"] for row in rows}
    has_hostile = "hostile" in providers
    has_scripted_or_callable = bool({"scripted", "callable"} & providers)
    return [
        GateResult("model_integration:failures_zero", all(value == 0 for value in failures), f"total_failures={sum(failures)}"),
        GateResult("model_integration:case_floor", total_cases >= 5000, f"cases={total_cases}"),
        GateResult("model_integration:hostile_present", has_hostile, f"providers={sorted(providers)}"),
        GateResult("model_integration:scripted_or_callable_present", has_scripted_or_callable, f"providers={sorted(providers)}"),
    ]


def validate_all(command_results: list[dict[str, object]], artifact_dir: Path) -> list[GateResult]:
    gates = []
    gates.extend(validate_command_results(command_results))
    gates.extend(validate_required_artifacts(artifact_dir))

    try:
        gates.extend(validate_experiment(read_csv_rows(artifact_dir / "experiment_summary.csv")))
    except Exception as exc:
        gates.append(GateResult("experiment:readable", False, repr(exc)))
    try:
        gates.extend(validate_stress(read_csv_rows(artifact_dir / "stress_summary.csv")))
    except Exception as exc:
        gates.append(GateResult("stress:readable", False, repr(exc)))
    try:
        gates.extend(validate_scale(read_csv_rows(artifact_dir / "scale_summary.csv")))
    except Exception as exc:
        gates.append(GateResult("scale:readable", False, repr(exc)))
    try:
        gates.extend(validate_tokenizer_correctness(read_csv_rows(artifact_dir / "tokenizer_correctness_summary.csv")))
    except Exception as exc:
        gates.append(GateResult("tokenizer:readable", False, repr(exc)))
    try:
        gates.extend(validate_structured_output(read_csv_rows(artifact_dir / "structured_output_summary.csv")))
    except Exception as exc:
        gates.append(GateResult("structured:readable", False, repr(exc)))
    try:
        gates.extend(validate_model_integration(read_csv_rows(artifact_dir / "model_integration_summary.csv")))
    except Exception as exc:
        gates.append(GateResult("model_integration:readable", False, repr(exc)))
    return gates


def all_passed(gates: Iterable[GateResult]) -> bool:
    return all(gate.passed for gate in gates)
