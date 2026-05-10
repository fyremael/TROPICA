# Metrics

## Correctness

- `invalid_rate`: fraction of samples violating constraints.
- `empty_support_rate`: fraction of steps where final support is empty.
- `delta_cost`: realized plan cost minus planner optimum.
- `prefix_violation_count`: guard-level prefix violations.

## Internalization quality

- `illegal_logit_pressure`: max illegal logit minus max legal logit before masking.
- `winner_prediction_accuracy`: auxiliary head accuracy for winner set.
- `phase_prediction_accuracy`: auxiliary head accuracy for planner/guard phase.
- `margin_calibration`: predicted vs true planner margin bucket.

## Diversity and quality

- `entropy_allowed`: entropy inside final support.
- `winner_cardinality`: size of planner/guard support.
- `distinct_outputs`: diversity across legal samples.

## Operations

- `latency_planner_ms`
- `latency_guard_ms`
- `latency_model_ms`
- `latency_total_ms`
- `trace_completeness`

## Acceptance criteria for first milestone

- Dyck demo: `invalid_rate == 0.0` for Planner∧Guard.
- Grid LTL demo: `delta_cost == 0`, audit all true.
- Mask tests: sampled tokens always in final support.
- Empty support: decoder raises an explicit error, never samples.
