from pathlib import Path

from cdsd.reporting import (
    REQUIRED_ARTIFACTS,
    all_passed,
    validate_experiment,
    validate_required_artifacts,
    validate_scale,
    validate_stress,
    validate_structured_output,
    validate_model_integration,
    validate_tokenizer_correctness,
)


def test_experiment_validator_accepts_expected_ablation_shape():
    rows = [
        {"Mode": "planner_guided", "InvalidRate": "0.0", "EmptySupportRate": "0.0"},
        {"Mode": "control_delta_plus_external", "InvalidRate": "0.0", "EmptySupportRate": "0.0"},
        {"Mode": "raw_generator", "InvalidRate": "0.94", "EmptySupportRate": "0.0"},
        {"Mode": "control_delta_only", "InvalidRate": "0.94", "EmptySupportRate": "0.0"},
    ]
    assert all_passed(validate_experiment(rows))


def test_stress_validator_rejects_nonzero_failure():
    rows = [
        {"Domain": "Tokenizer automata", "Cases": "6000", "Failures": "0"},
        {"Domain": "ControlDelta numerics", "Cases": "80", "Failures": "1"},
    ]
    gates = validate_stress(rows)
    assert not all_passed(gates)
    assert any(gate.name == "stress:failures_zero" and not gate.passed for gate in gates)


def test_scale_validator_accepts_required_size_floors():
    rows = [
        {"Track": "Dyck horizon", "Size": "1024", "Failures": "0"},
        {"Track": "Tokenizer enums", "Size": "4096", "Failures": "0"},
        {"Track": "Workflow nodes", "Size": "2048", "Failures": "0"},
        {"Track": "ControlDelta tokens", "Size": "1024", "Failures": "0"},
    ]
    assert all_passed(validate_scale(rows))


def test_required_artifacts_validator_checks_nonempty_files(tmp_path: Path):
    (tmp_path / "present.txt").write_text("ok", encoding="utf-8")
    gates = validate_required_artifacts(tmp_path, ["present.txt", "missing.txt"])
    assert gates[0].passed
    assert not gates[1].passed


def test_required_artifacts_include_model_trace_explorer():
    assert "model_integration_traces.jsonl" in REQUIRED_ARTIFACTS
    assert "trace_explorer.html" in REQUIRED_ARTIFACTS


def test_trace_artifacts_are_required_and_nonempty(tmp_path: Path):
    (tmp_path / "model_integration_traces.jsonl").write_text('{"schema_version": 1}\n', encoding="utf-8")
    (tmp_path / "trace_explorer.html").write_text("<!doctype html>\n", encoding="utf-8")

    gates = validate_required_artifacts(tmp_path, ["model_integration_traces.jsonl", "trace_explorer.html", "empty.html"])

    assert gates[0].passed
    assert gates[1].passed
    assert not gates[2].passed


def test_tokenizer_correctness_validator_requires_real_adapters_and_case_floor():
    rows = [
        {"Adapter": "tiktoken/cl100k_base", "Cases": "2600", "Failures": "0"},
        {"Adapter": "hf/wordpiece", "Cases": "1800", "Failures": "0"},
        {"Adapter": "hf/bpe", "Cases": "1800", "Failures": "0"},
    ]
    assert all_passed(validate_tokenizer_correctness(rows))


def test_tokenizer_correctness_validator_rejects_missing_hf_or_failures():
    rows = [
        {"Adapter": "tiktoken/cl100k_base", "Cases": "5000", "Failures": "1"},
    ]
    gates = validate_tokenizer_correctness(rows)
    assert not all_passed(gates)
    assert any(gate.name == "tokenizer:adapter_hf_present" and not gate.passed for gate in gates)
    assert any(gate.name == "tokenizer:failures_zero" and not gate.passed for gate in gates)


def test_structured_output_validator_requires_cases_adapters_and_schema_controls():
    rows = [
        {"Adapter": "tiktoken/cl100k_base", "Cases": "5200", "Failures": "0", "Outputs": "5200"},
        {"Adapter": "hf/structured-bpe", "Cases": "5200", "Failures": "0", "Outputs": "5200"},
        {"Adapter": "schema-controls", "Cases": "5", "Failures": "0", "Outputs": "0"},
    ]
    assert all_passed(validate_structured_output(rows))


def test_structured_output_validator_rejects_failures_or_missing_controls():
    rows = [
        {"Adapter": "tiktoken/cl100k_base", "Cases": "5000", "Failures": "1", "Outputs": "5000"},
    ]
    gates = validate_structured_output(rows)
    assert not all_passed(gates)
    assert any(gate.name == "structured:failures_zero" and not gate.passed for gate in gates)
    assert any(gate.name == "structured:adapter_hf_present" and not gate.passed for gate in gates)


def test_model_integration_validator_requires_cases_and_providers():
    rows = [
        {"Provider": "scripted", "Cases": "7101", "Failures": "0"},
        {"Provider": "hostile", "Cases": "500", "Failures": "0"},
        {"Provider": "callable", "Cases": "250", "Failures": "0"},
    ]
    assert all_passed(validate_model_integration(rows))


def test_model_integration_validator_rejects_failures_or_missing_provider():
    rows = [
        {"Provider": "scripted", "Cases": "5000", "Failures": "1"},
    ]
    gates = validate_model_integration(rows)
    assert not all_passed(gates)
    assert any(gate.name == "model_integration:failures_zero" and not gate.passed for gate in gates)
    assert any(gate.name == "model_integration:hostile_present" and not gate.passed for gate in gates)
