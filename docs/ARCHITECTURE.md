# Architecture

## Runtime contract

At token step `t`:

```text
Planner.step(state) -> plan_mask, winners, margin
Guard.mask(prefix, state) -> guard_mask
Policy.mask(context) -> policy_mask

final_mask = plan_mask & guard_mask & policy_mask
logits = generator(prefix, control_state)
logits[~final_mask] = -inf
sample(logits)
```

`cdsd.contracts` makes this explicit in code. The decoder validates that final
support is the exact intersection of planner, guard, and policy masks, raises
typed violations for empty support or illegal selections, and can emit unified
trace events with planner support, guard support, policy support, final support,
selected token/action, state summary, accepting state, and failure reason.

## Internal control lane

The control lane consumes planner/guard features and maintains a small recurrent memory:

\[
M_t = \lambda_t M_{t-1} + \beta_t (v_t - M_{t-1}k_t) k_t^\top.
\]

This is a delta-rule style update. In this scaffold we provide a compact PyTorch module that emits:

- `control_state`: recurrent memory summary.
- `logit_bias`: bias to improve logits inside legal support.
- `gate`: optional hidden-state modulation.
- `aux`: phase/winner/margin prediction hooks.

The external support contract still enforces correctness.

## Separation of authority

| Component | Role | Authority? |
|---|---|---:|
| Planner | semantic feasibility / optimal frontier | yes |
| Guard | prefix legality / grammar / schema / dynamics | yes |
| Policy | organization-specific restrictions | yes |
| Generator | style and ranking inside legal support | no |
| Control Delta lane | learned awareness of planner/guard state | no |

The generator may learn the rules. It does not get to decide the rules.
