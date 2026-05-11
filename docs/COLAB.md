# Colab Notebook Pack

TROPICA ships a small Colab pack for people who want to evaluate the project
from a browser runtime before investing in local setup.

## Notebooks

| Notebook | Link | Use |
| --- | --- | --- |
| Operator onboarding | [Open in Colab](https://colab.research.google.com/github/fyremael/TROPICA/blob/main/notebooks/01_operator_onboarding.ipynb) | Install TROPICA, run the evidence suite, and inspect the report index plus model-integration dashboard. |
| Benchmark suite | [Open in Colab](https://colab.research.google.com/github/fyremael/TROPICA/blob/main/notebooks/02_benchmark_suite.ipynb) | Run `cdsd-report`, summarize gate status, display all dashboards, and zip artifacts for review. |
| Researcher showcase | [Open in Colab](https://colab.research.google.com/github/fyremael/TROPICA/blob/main/notebooks/03_researcher_showcase.ipynb) | Compile bounded tool-call specs into real tokenizer masks, run offline logit providers, inspect traces, and exercise negative controls. |

## Runtime Contract

Each notebook installs from the GitHub repository when opened in a plain Colab
runtime. If the notebook is run from a cloned source checkout, it uses editable
install instead:

```bash
python -m pip install -e ".[real-tokenizers,dev]"
```

The benchmark and onboarding notebooks write generated outputs under
`colab_*_artifacts/`. These directories are ignored by git through the
`*_artifacts/` rule.

## Evidence Interpretation

The notebooks expose the same evidence surfaces as CI:

- `report_manifest.json` is the pass/fail source of truth.
- `report_index.md` is the reviewer-facing interpretation layer.
- SVG dashboards show experiment, tokenizer, structured-output,
  model-integration, stress, and scale behavior.

The researcher notebook is intentionally offline. It shows how a local model or
synthetic provider can drive `StructuredOutputDecoder` without network calls or
hosted inference.
