# Codex Prompt: ControlDeltaBlock Engineer

Implement and harden `src/cdsd/control_delta_block.py`.

Tasks:

1. Add batch-first recurrent stepping and chunked scan mode.
2. Add optional channel-wise decay gates.
3. Add auxiliary heads for phase, winner-set, and margin-bucket prediction.
4. Add unit tests for shapes, finite outputs, gradient flow, and memory reset behavior.
5. Do not make this module responsible for legality. It only improves logits.

Acceptance:

- Forward pass supports `[B, D]` and `[B, T, D]` control features.
- No NaNs for random inputs.
- Tests pass.
