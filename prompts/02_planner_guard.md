# Codex Prompt: Planner and Guard Engineer

Implement planners and guards for:

1. Dyck-k.
2. JSON schema subset.
3. Grid/LTL product-state routing.
4. Tool workflow graph.

Each planner must expose:

- `plan_mask`
- `winners`
- `margin`
- `control_features`
- `trace`

Each guard must expose:

- `mask(prefix, state)`
- `update(state, token)`
- property tests for prefix safety.

Acceptance:

- InvalidRate = 0 for supported tasks under correct specs.
- Empty support raises a typed error.
