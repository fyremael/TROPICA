from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from cdsd.reporting import all_passed, read_csv_rows, validate_all


REPORT_TRACKS = [
    (
        "experiment",
        [
            ("experiment_harness", "demos.run_experiment_harness"),
            ("experiment_visuals", "demos.render_experiment_visuals"),
        ],
    ),
    (
        "stress",
        [
            ("stress_harness", "demos.run_stress_harness"),
            ("stress_visuals", "demos.render_stress_visuals"),
        ],
    ),
    (
        "scale",
        [
            ("scale_harness", "demos.run_scale_harness"),
            ("scale_visuals", "demos.render_scale_visuals"),
        ],
    ),
    (
        "tokenizer_correctness",
        [
            ("tokenizer_correctness_harness", "demos.run_tokenizer_correctness_harness"),
            ("tokenizer_correctness_visuals", "demos.render_tokenizer_correctness_visuals"),
        ],
    ),
    (
        "structured_output",
        [
            ("structured_output_harness", "demos.run_structured_output_harness"),
            ("structured_output_visuals", "demos.render_structured_output_visuals"),
        ],
    ),
    (
        "model_integration",
        [
            ("model_integration_harness", "demos.run_model_integration_harness"),
            ("model_integration_visuals", "demos.render_model_integration_visuals"),
        ],
    ),
]


def tail(text: str, limit: int = 40) -> str:
    lines = text.splitlines()
    return "\n".join(lines[-limit:])


def build_commands(with_pytest: bool) -> list[tuple[str, list[str]]]:
    commands: list[tuple[str, list[str]]] = []
    if with_pytest:
        commands.append(("pytest", [sys.executable, "-m", "pytest", "tests"]))
    for _, track in REPORT_TRACKS:
        commands.extend((name, [sys.executable, "-m", module]) for name, module in track)
    return commands


def build_command_groups(with_pytest: bool) -> list[tuple[str, list[tuple[str, list[str]]]]]:
    groups: list[tuple[str, list[tuple[str, list[str]]]]] = []
    if with_pytest:
        groups.append(("pytest", [("pytest", [sys.executable, "-m", "pytest", "tests"])]))
    for track_name, track in REPORT_TRACKS:
        groups.append((track_name, [(name, [sys.executable, "-m", module]) for name, module in track]))
    return groups


def run_command(name: str, command: list[str], *, cwd: Path, artifact_dir: Path) -> dict[str, object]:
    start = time.perf_counter()
    env = os.environ.copy()
    env["CDSD_ARTIFACT_DIR"] = str(artifact_dir)
    print(f"[cdsd-report] Running {name}: {' '.join(command)}", flush=True)
    completed = subprocess.run(command, cwd=cwd, env=env, text=True, capture_output=True)
    duration_ms = (time.perf_counter() - start) * 1000.0
    if completed.stdout:
        print(tail(completed.stdout, 20))
    if completed.stderr:
        print(tail(completed.stderr, 20), file=sys.stderr)
    print(f"[cdsd-report] {name} returncode={completed.returncode} duration_ms={duration_ms:.3f}", flush=True)
    return {
        "name": name,
        "command": command,
        "returncode": completed.returncode,
        "duration_ms": round(duration_ms, 3),
        "stdout_tail": tail(completed.stdout),
        "stderr_tail": tail(completed.stderr),
    }


def run_command_group(
    group_name: str,
    commands: list[tuple[str, list[str]]],
    *,
    cwd: Path,
    artifact_dir: Path,
) -> list[dict[str, object]]:
    print(f"[cdsd-report] Starting track {group_name}", flush=True)
    results = [run_command(name, command, cwd=cwd, artifact_dir=artifact_dir) for name, command in commands]
    print(f"[cdsd-report] Finished track {group_name}", flush=True)
    return results


def run_command_groups(
    groups: list[tuple[str, list[tuple[str, list[str]]]]],
    *,
    cwd: Path,
    artifact_dir: Path,
    jobs: int,
) -> list[dict[str, object]]:
    if jobs <= 1 or len(groups) <= 1:
        results: list[dict[str, object]] = []
        for group_name, commands in groups:
            results.extend(run_command_group(group_name, commands, cwd=cwd, artifact_dir=artifact_dir))
        return results

    ordered_results: list[list[dict[str, object]] | None] = [None] * len(groups)
    workers = min(jobs, len(groups))
    print(f"[cdsd-report] Running {len(groups)} tracks with jobs={workers}", flush=True)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(run_command_group, group_name, commands, cwd=cwd, artifact_dir=artifact_dir): idx
            for idx, (group_name, commands) in enumerate(groups)
        }
        for future in as_completed(futures):
            idx = futures[future]
            ordered_results[idx] = future.result()

    results = []
    for group_results in ordered_results:
        if group_results is not None:
            results.extend(group_results)
    return results


def artifact_entry(artifact_dir: Path, name: str) -> dict[str, object]:
    path = artifact_dir / name
    return {
        "name": name,
        "path": str(path),
        "exists": path.exists(),
        "size": path.stat().st_size if path.exists() else 0,
    }


def experiment_interpretation(artifact_dir: Path) -> str:
    rows = {row["Mode"]: row for row in read_csv_rows(artifact_dir / "experiment_summary.csv")}
    planner = float(rows["planner_guided"]["InvalidRate"])
    cd_external = float(rows["control_delta_plus_external"]["InvalidRate"])
    raw = float(rows["raw_generator"]["InvalidRate"])
    cd_only = float(rows["control_delta_only"]["InvalidRate"])
    return (
        f"Planner-guided and ControlDelta+external both hold InvalidRate at {planner:.4f}/{cd_external:.4f}, "
        f"while raw and ControlDelta-only remain high at {raw:.4f}/{cd_only:.4f}. "
        "The evidence supports the core separation: internal awareness can shape logits, but external support is the authority."
    )


def stress_interpretation(artifact_dir: Path) -> str:
    rows = read_csv_rows(artifact_dir / "stress_summary.csv")
    cases = sum(int(float(row["Cases"])) for row in rows)
    failures = sum(int(float(row["Failures"])) for row in rows)
    domains = len(rows)
    return (
        f"The stress pass covers {cases:,} cases across {domains} domains with {failures} failures. "
        "This stresses adversarial decoding, empty-support behavior, tokenizer prefixes, workflow routing, grid planning, and ControlDelta numerics."
    )


def scale_interpretation(artifact_dir: Path) -> str:
    rows = read_csv_rows(artifact_dir / "scale_summary.csv")
    failures = sum(int(float(row["Failures"])) for row in rows)

    def max_size(track: str) -> int:
        return max(int(float(row["Size"])) for row in rows if row["Track"] == track)

    return (
        f"The scale sweep has {failures} failures at the largest checkpoints: "
        f"Dyck horizon {max_size('Dyck horizon')}, tokenizer enums {max_size('Tokenizer enums')}, "
        f"workflow nodes {max_size('Workflow nodes')}, and ControlDelta sequence size {max_size('ControlDelta tokens')}. "
        "Runtime rises with problem size, but no correctness cliff appears."
    )


def tokenizer_interpretation(artifact_dir: Path) -> str:
    rows = read_csv_rows(artifact_dir / "tokenizer_correctness_summary.csv")
    cases = sum(int(float(row["Cases"])) for row in rows)
    failures = sum(int(float(row["Failures"])) for row in rows)
    adapters = sorted({row["Adapter"] for row in rows})
    return (
        f"The tokenizer correctness pass covers {cases:,} exact-generation and negative-control cases with {failures} failures. "
        f"It exercises real adapters: {', '.join(adapters)}."
    )


def structured_output_interpretation(artifact_dir: Path) -> str:
    rows = read_csv_rows(artifact_dir / "structured_output_summary.csv")
    cases = sum(int(float(row["Cases"])) for row in rows)
    failures = sum(int(float(row["Failures"])) for row in rows)
    max_outputs = max(int(float(row["Outputs"])) for row in rows)
    adapters = sorted({row["Adapter"] for row in rows})
    return (
        f"The structured-output pass covers {cases:,} exact JSON/tool-call and negative-control cases with {failures} failures. "
        f"The largest compiled frontier has {max_outputs:,} canonical outputs across {', '.join(adapters)}."
    )


def model_integration_interpretation(artifact_dir: Path) -> str:
    rows = read_csv_rows(artifact_dir / "model_integration_summary.csv")
    cases = sum(int(float(row["Cases"])) for row in rows)
    failures = sum(int(float(row["Failures"])) for row in rows)
    trace_steps = sum(int(float(row["TraceSteps"])) for row in rows)
    providers = sorted({row["Provider"] for row in rows})
    return (
        f"The offline model-integration pass covers {cases:,} provider-driven decode cases with {failures} failures "
        f"and {trace_steps:,} trace events. Providers exercised: {', '.join(providers)}."
    )


def write_report_index(artifact_dir: Path, gates) -> Path:
    path = artifact_dir / "report_index.md"
    passed = all_passed(gates)
    failed = [gate for gate in gates if not gate.passed]
    lines = [
        "# Control-Delta Support Decoding Evidence Report",
        "",
        f"Overall gate: **{'PASS' if passed else 'FAIL'}**",
        "",
        "## Experiment Dashboard",
        "",
        "![Experiment visuals](experiment_visuals.svg)",
        "",
        experiment_interpretation(artifact_dir),
        "",
        "## Tokenizer Correctness Dashboard",
        "",
        "![Tokenizer correctness visuals](tokenizer_correctness_visuals.svg)",
        "",
        tokenizer_interpretation(artifact_dir),
        "",
        "## Structured Output Dashboard",
        "",
        "![Structured output visuals](structured_output_visuals.svg)",
        "",
        structured_output_interpretation(artifact_dir),
        "",
        "## Model Integration Dashboard",
        "",
        "![Model integration visuals](model_integration_visuals.svg)",
        "",
        model_integration_interpretation(artifact_dir),
        "",
        "## Stress Dashboard",
        "",
        "![Stress visuals](stress_visuals.svg)",
        "",
        stress_interpretation(artifact_dir),
        "",
        "## Scale Dashboard",
        "",
        "![Scale visuals](scale_visuals.svg)",
        "",
        scale_interpretation(artifact_dir),
        "",
        "## Gate Summary",
        "",
        "| Gate | Status | Detail |",
        "| --- | --- | --- |",
    ]
    for gate in gates:
        status = "PASS" if gate.passed else "FAIL"
        lines.append(f"| {gate.name} | {status} | {gate.detail} |")
    if failed:
        lines.extend(["", "## Failed Gates", ""])
        lines.extend(f"- `{gate.name}`: {gate.detail}" for gate in failed)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_manifest(artifact_dir: Path, command_results, gates) -> Path:
    path = artifact_dir / "report_manifest.json"
    manifest = {
        "passed": all_passed(gates),
        "commands": command_results,
        "artifacts": [
            artifact_entry(artifact_dir, "experiment_summary.csv"),
            artifact_entry(artifact_dir, "experiment_summary.md"),
            artifact_entry(artifact_dir, "experiment_visuals.svg"),
            artifact_entry(artifact_dir, "stress_summary.csv"),
            artifact_entry(artifact_dir, "stress_summary.md"),
            artifact_entry(artifact_dir, "stress_visuals.svg"),
            artifact_entry(artifact_dir, "scale_summary.csv"),
            artifact_entry(artifact_dir, "scale_summary.md"),
            artifact_entry(artifact_dir, "scale_visuals.svg"),
            artifact_entry(artifact_dir, "tokenizer_correctness_summary.csv"),
            artifact_entry(artifact_dir, "tokenizer_correctness_summary.md"),
            artifact_entry(artifact_dir, "tokenizer_correctness_visuals.svg"),
            artifact_entry(artifact_dir, "structured_output_summary.csv"),
            artifact_entry(artifact_dir, "structured_output_summary.md"),
            artifact_entry(artifact_dir, "structured_output_visuals.svg"),
            artifact_entry(artifact_dir, "model_integration_summary.csv"),
            artifact_entry(artifact_dir, "model_integration_summary.md"),
            artifact_entry(artifact_dir, "model_integration_visuals.svg"),
            artifact_entry(artifact_dir, "report_index.md"),
        ],
        "gates": [gate.to_dict() for gate in gates],
    }
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


def run_all(
    *,
    artifact_dir: Path | str = "artifacts",
    with_pytest: bool = False,
    cwd: Path | str | None = None,
    jobs: int = 1,
) -> int:
    artifact_path = Path(artifact_dir).resolve()
    artifact_path.mkdir(parents=True, exist_ok=True)
    workdir = Path(cwd).resolve() if cwd is not None else Path.cwd()
    if jobs < 1:
        raise ValueError("jobs must be >= 1")
    command_results = run_command_groups(build_command_groups(with_pytest), cwd=workdir, artifact_dir=artifact_path, jobs=jobs)
    gates = validate_all(command_results, artifact_path)
    index_path = write_report_index(artifact_path, gates)
    manifest_path = write_manifest(artifact_path, command_results, gates)
    failed = [gate for gate in gates if not gate.passed]
    if failed:
        print("[cdsd-report] FAILED gates:")
        for gate in failed:
            print(f"  - {gate.name}: {gate.detail}")
        return 1
    print(f"[cdsd-report] PASS. Wrote {manifest_path} and {index_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    def positive_int(text: str) -> int:
        value = int(text)
        if value < 1:
            raise argparse.ArgumentTypeError("must be >= 1")
        return value

    parser = argparse.ArgumentParser(description="Run CDSD evidence reports and CI-style gates.")
    parser.add_argument("--artifacts", default="artifacts", help="Directory for generated reports, CSV files, and SVG dashboards.")
    parser.add_argument("--with-pytest", action="store_true", help="Run pytest before evidence harnesses. Use from a source checkout.")
    parser.add_argument("--jobs", type=positive_int, default=1, help="Number of independent report tracks to run in parallel.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return run_all(artifact_dir=args.artifacts, with_pytest=args.with_pytest, jobs=args.jobs)
