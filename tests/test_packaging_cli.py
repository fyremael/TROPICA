from __future__ import annotations

import json
from pathlib import Path

import cdsd
from cdsd.evidence import runner


def test_public_exports_import_cleanly() -> None:
    assert cdsd.StructuredOutputCompiler is not None
    assert cdsd.StructuredOutputDecoder is not None
    assert cdsd.HostileLogitProvider is not None
    assert cdsd.HFLocalLogitProvider is not None
    assert cdsd.LocalModelBridgeError is not None
    assert cdsd.ToolCallSpec is not None
    assert cdsd.TiktokenAdapter is not None
    assert cdsd.HFTokenizerAdapter is not None


def test_build_commands_respects_pytest_flag() -> None:
    without_pytest = [name for name, _ in runner.build_commands(with_pytest=False)]
    with_pytest = [name for name, _ in runner.build_commands(with_pytest=True)]

    assert "pytest" not in without_pytest
    assert with_pytest[0] == "pytest"
    assert "structured_output_visuals" in without_pytest
    assert "trace_explorer" in without_pytest


def test_cli_main_propagates_runner_exit(monkeypatch, tmp_path: Path) -> None:
    def fake_run_all(*, artifact_dir: str | Path, with_pytest: bool, jobs: int) -> int:
        assert Path(artifact_dir) == tmp_path
        assert with_pytest is True
        assert jobs == 3
        return 17

    monkeypatch.setattr(runner, "run_all", fake_run_all)

    assert runner.main(["--artifacts", str(tmp_path), "--with-pytest", "--jobs", "3"]) == 17


def test_evidence_runner_writes_custom_artifact_dir(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(runner, "build_command_groups", lambda with_pytest: [])
    monkeypatch.setattr(runner, "validate_all", lambda command_results, artifact_dir: [])

    def fake_index(artifact_dir: Path, gates: list[object]) -> Path:
        path = artifact_dir / "report_index.md"
        path.write_text("# ok\n", encoding="utf-8")
        return path

    def fake_manifest(artifact_dir: Path, command_results: list[object], gates: list[object]) -> Path:
        path = artifact_dir / "report_manifest.json"
        path.write_text(json.dumps({"passed": True}), encoding="utf-8")
        return path

    monkeypatch.setattr(runner, "write_report_index", fake_index)
    monkeypatch.setattr(runner, "write_manifest", fake_manifest)

    assert runner.run_all(artifact_dir=tmp_path, with_pytest=False, cwd=tmp_path) == 0
    assert (tmp_path / "report_index.md").read_text(encoding="utf-8") == "# ok\n"
    assert json.loads((tmp_path / "report_manifest.json").read_text(encoding="utf-8"))["passed"] is True
