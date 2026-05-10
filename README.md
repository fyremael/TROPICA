# Control-Delta Support Decoding

**Planner-derived semantic support + multi-winner optimal-frontier exposure + streaming guard intersection at token time**, with a gated DeltaNet-style internal control lane.

This repository is a Codex-ready implementation scaffold for turning constrained decoding into a **planner-backed control plane for reliable agentic output**.

## Core doctrine

```text
Internalize awareness. Externalize authority.
```

The transformer or recurrent generator may see planner/guard state through a learned control lane, but legality is still enforced by the external pre-softmax support contract:

```text
final_mask_t = plan_mask_t ∧ guard_mask_t ∧ policy_mask_t
```

Tokens outside `final_mask_t` receive `-inf` before softmax. They are not discouraged. They are unrepresentable.

## Why a gated Delta control lane?

Planner/guard state is streaming control state: phase, stack, route flags, planner margin, remaining obligations, and winner sets. A gated delta-style recurrent memory is a natural internal place to store and exploit these features, while the external mask remains the non-negotiable authority.

## Install and verify

TROPICA is packaged as an installable Python project. From a fresh checkout:

```bash
python -m pip install -e ".[real-tokenizers,dev]"
cdsd-report --with-pytest --artifacts artifacts --jobs 4
```

`cdsd-report` runs the evidence suite, writes CSV summaries and SVG dashboards,
then validates hard pass/fail gates. Start with:

- `docs/INTRODUCTION.md` for the new-user overview: motivation, methodology, evidence, usage, and roadmap.
- `artifacts/report_index.md` for the human-readable report with visuals and short interpretations.
- `artifacts/report_manifest.json` for the machine-readable pass/fail manifest.
- `docs/INSTALL.md` for editable install, wheel build, clean wheel smoke, and CLI workflow.
- `docs/REPORTING.md` for evidence artifacts, gates, and faster parallel report runs.
- `docs/API_QUICKSTART.md` for a minimal `ToolCallSpec -> StructuredOutputCompiler` example.

Build a wheel:

```bash
python -m build
python -m pip install --force-reinstall "dist/control_delta_support_decoding-0.1.0-py3-none-any.whl[real-tokenizers]"
cdsd-report --artifacts smoke_artifacts --jobs 4
```

## Package contents

- `src/cdsd/control_delta_block.py` — Gated delta-style control lane.
- `src/cdsd/decoder.py` — support-contract decoder.
- `src/cdsd/masks.py` — mask utilities and safety checks.
- `src/cdsd/cli.py` — public `cdsd-report` command.
- `src/cdsd/evidence/` — importable report orchestration and CI gate runner.
- `src/cdsd/planners/dyck.py` — Dyck planner with multi-winner support.
- `src/cdsd/guards/dyck.py` — streaming Dyck guard.
- `src/cdsd/planners/grid_ltl.py` — product-state route planner for LTL-style constraints.
- `src/cdsd/planners/json_schema.py` — JSON schema subset planner/guard.
- `src/cdsd/planners/tool_workflow.py` — tool workflow graph planner/guard.
- `src/cdsd/tokenizer_compiler.py` — literal/enum token-prefix automata.
- `src/cdsd/structured_output.py` — bounded JSON/tool-call compiler over real tokenizer IDs.
- `demos/run_dyck_support_demo.py` — invalid-rate and winner-set demo.
- `demos/run_grid_ltl_demo.py` — route planner demo with audit.
- `demos/run_experiment_harness.py` — ablation metrics to CSV/Markdown.
- `demos/run_scale_harness.py` — deterministic scale sweeps for core surfaces.
- `demos/run_tokenizer_correctness_harness.py` — real-tokenizer exactness and negative controls.
- `demos/run_structured_output_harness.py` — bounded JSON/tool-call masks over real tokenizers.
- `tests/` — unit, validator, tokenizer, structured-output, CLI, and packaging tests.
- `prompts/` — Codex implementation prompts by module.
- `docs/` — architecture, metrics, and experiment plans.

## Compatibility commands

```bash
python demos/run_all_reports.py
```

The demo command remains valid and now delegates to the same package runner as
`cdsd-report`. Individual report stages can still be run directly:

```bash
python demos/run_dyck_support_demo.py
python demos/run_grid_ltl_demo.py
python demos/run_experiment_harness.py
python demos/render_experiment_visuals.py
python demos/run_stress_harness.py
python demos/render_stress_visuals.py
python demos/run_scale_harness.py
python demos/render_scale_visuals.py
python demos/run_tokenizer_correctness_harness.py
python demos/render_tokenizer_correctness_visuals.py
python demos/run_structured_output_harness.py
python demos/render_structured_output_visuals.py
python -m pytest tests
```

The demos are intentionally small. The point is the control law, not to make the world’s most dramatic parenthesis generator. Humanity has suffered enough.
