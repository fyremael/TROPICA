# Experiment Plan

## Ablations

1. Raw generator.
2. External mask only.
3. Internal ControlDelta lane only.
4. ControlDelta + external support contract.
5. Grammar-only constrained decoder.
6. Planner-guided support decoder.

## Hypotheses

H1: External support masking drives invalid emissions to zero under correct masks.

H2: ControlDelta conditioning reduces illegal logit pressure compared with external-only masking.

H3: Multi-winner support preserves entropy and diversity inside constraints compared with single-action planning.

H4: Product-state planning enforces semantic constraints that grammar-only systems do not express naturally.

## Benchmarks

- Dyck-k balanced structures.
- JSON schema subset with required fields, enum values, arrays, and ordering flexibility.
- Tool-call workflow graph.
- Grid/LTL route planning.
- Read-only SQL generation with table ACL masks.

## Required tables

| Method | InvalidRate | DeltaCost | EmptySupport | EntropyAllowed | Latency |
|---|---:|---:|---:|---:|---:|
| Raw generator | | | | | |
| External mask | | | | | |
| ControlDelta only | | | | | |
| ControlDelta + external mask | | | | | |
