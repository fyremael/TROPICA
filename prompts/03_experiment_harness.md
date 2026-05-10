# Codex Prompt: Experiment Harness Engineer

Build an experiment harness comparing:

1. Raw generator.
2. External mask only.
3. ControlDelta only.
4. ControlDelta + external support mask.
5. Grammar-only.
6. Planner-guided support decoding.

Metrics:

- InvalidRate
- DeltaCost
- EmptySupportRate
- WinnerCardinality
- EntropyAllowed
- IllegalLogitPressure
- Latency

Output CSV and markdown summary tables.
