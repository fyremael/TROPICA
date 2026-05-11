import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "notebooks"
NOTEBOOKS = [
    NOTEBOOK_DIR / "01_operator_onboarding.ipynb",
    NOTEBOOK_DIR / "02_benchmark_suite.ipynb",
    NOTEBOOK_DIR / "03_researcher_showcase.ipynb",
]


def load_notebook(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def notebook_text(path: Path) -> str:
    notebook = load_notebook(path)
    return "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])


def test_colab_pack_has_expected_notebooks():
    assert (NOTEBOOK_DIR / "README.md").is_file()
    for path in NOTEBOOKS:
        assert path.is_file()


def test_notebooks_are_clean_nbformat_json():
    for path in NOTEBOOKS:
        notebook = load_notebook(path)
        assert notebook["nbformat"] == 4
        assert notebook["nbformat_minor"] >= 5
        assert notebook["metadata"]["kernelspec"]["name"] == "python3"
        assert notebook["cells"]
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                assert cell["execution_count"] is None
                assert cell["outputs"] == []


def test_notebook_code_cells_compile():
    for path in NOTEBOOKS:
        notebook = load_notebook(path)
        for idx, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] == "code":
                compile("".join(cell["source"]), f"{path.name}:cell-{idx}", "exec")


def test_notebooks_link_to_real_colab_repo():
    for path in NOTEBOOKS:
        text = notebook_text(path)
        assert "colab.research.google.com/github/fyremael/TROPICA" in text
        assert "control-delta-support-decoding[real-tokenizers,dev] @ git+https://github.com/fyremael/TROPICA.git" in text


def test_operator_and_benchmark_run_report_command():
    operator = notebook_text(NOTEBOOK_DIR / "01_operator_onboarding.ipynb")
    benchmark = notebook_text(NOTEBOOK_DIR / "02_benchmark_suite.ipynb")
    assert "cdsd-report" in operator
    assert "report_manifest.json" in operator
    assert "model_integration_visuals.svg" in operator
    assert "cdsd-report" in benchmark
    assert "experiment_visuals.svg" in benchmark
    assert "model_integration_visuals.svg" in benchmark
    assert "scale_visuals.svg" in benchmark


def test_researcher_notebook_uses_model_integration_api():
    text = notebook_text(NOTEBOOK_DIR / "03_researcher_showcase.ipynb")
    for name in [
        "ToolCallSpec",
        "StructuredOutputCompiler",
        "StructuredOutputDecoder",
        "TiktokenAdapter",
        "HostileLogitProvider",
        "ScriptedLogitProvider",
        "CallableLogitProvider",
        "StructuredOutputDecodeError",
    ]:
        assert name in text
