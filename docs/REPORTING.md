# Reporting And Evidence Guide

`cdsd-report` is the main trust surface for TROPICA. It runs the evidence
suite, renders dashboards, validates thresholds, and exits nonzero if the
system does not meet the contract.

## Full Source-Checkout Run

```bash
cdsd-report --with-pytest --artifacts artifacts --jobs 4
```

Use this command when working from a repository checkout. It runs unit tests,
all harnesses, all visual renderers, and all report gates.

## Wheel Smoke Run

```bash
cdsd-report --artifacts smoke_artifacts --jobs 4
```

Use this after installing a wheel. It skips pytest because a wheel install does
not normally include the source checkout's `tests/` directory.

## Parallel Tracks

The `--jobs` flag runs independent report tracks concurrently:

- experiment
- stress
- scale
- tokenizer correctness
- structured output
- pytest, when `--with-pytest` is set

Each track still preserves its internal order. For example, experiment visuals
render only after the experiment harness has written its CSV.

Use `--jobs 1` for fully serial logs. Use `--jobs 4` or higher for faster local
iteration and CI.

## Outputs

Every run writes the following artifact family:

- `report_index.md`: human-readable report with all dashboards and short
  interpretations
- `report_manifest.json`: machine-readable command status, artifact metadata,
  gate status, and overall pass/fail
- `*_summary.csv`: metric tables used by gates and visuals
- `*_summary.md`: readable metric summaries
- `*_visuals.svg`: dashboards for review and CI artifacts

## Gate Reading

The manifest top-level field is the first checkpoint:

```json
{
  "passed": true
}
```

If it is false, inspect `gates` for the failed names. The report index also
contains a `Failed Gates` section when any threshold fails.

## Result Interpretation

The dashboards are not decorative. Each one answers a specific trust question:

| Dashboard | Question |
| --- | --- |
| Experiment | Does external support eliminate invalid outputs where raw and internal-only paths fail? |
| Tokenizer correctness | Do real tokenizer masks survive exact literals and negative controls? |
| Structured output | Can bounded JSON/tool-call specs compile to real token-ID masks and decode under hostile logits? |
| Stress | Do the contracts survive randomized and adversarial cases? |
| Scale | Do larger frontiers preserve correctness without obvious cliffs? |

## CI Pattern

The repository workflow runs two tiers:

- source validation: editable install, pytest, full report with tests, artifact
  upload
- wheel smoke: build wheel, install it cleanly on Ubuntu and Windows, import
  `cdsd`, run `cdsd-report`, upload artifacts

That means a reviewer can trust both source behavior and installable package
behavior.
