from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class ControlDeltaBlock(nn.Module):
    """Small gated delta-style control lane.

    This module is intentionally compact. It is not a drop-in replacement for a
    production Gated DeltaNet kernel; it is a research scaffold for wiring planner
    and guard features into a generator.

    Inputs:
      control_features: [batch, d_in] or [batch, time, d_in]
      memory: [batch, d_mem, d_mem] or None

    Outputs:
      memory_new: recurrent control memory
      summary: [batch, d_mem]
      logit_bias: [batch, vocab_size]

    The update follows a delta-rule shape:
      read = M @ k
      delta = outer(v - read, k)
      M_new = lambda * M + beta * delta
    """

    def __init__(
        self,
        d_in: int,
        d_mem: int,
        vocab_size: int,
        *,
        phase_classes: int = 8,
        winner_classes: int | None = None,
        margin_buckets: int = 4,
        channel_decay: bool = True,
    ):
        super().__init__()
        self.d_mem = d_mem
        self.channel_decay = channel_decay
        self.k_proj = nn.Linear(d_in, d_mem)
        self.v_proj = nn.Linear(d_in, d_mem)
        self.beta_proj = nn.Linear(d_in, d_mem)
        self.lambda_proj = nn.Linear(d_in, d_mem)
        self.summary_proj = nn.Linear(d_mem, d_mem)
        self.logit_bias = nn.Linear(d_mem, vocab_size)
        self.phase_head = nn.Linear(d_mem, phase_classes)
        self.winner_head = nn.Linear(d_mem, winner_classes or vocab_size)
        self.margin_head = nn.Linear(d_mem, margin_buckets)

    def reset_memory(self, batch_size: int, *, device: torch.device | None = None, dtype: torch.dtype | None = None) -> torch.Tensor:
        return torch.zeros(batch_size, self.d_mem, self.d_mem, device=device, dtype=dtype)

    def _step(self, control_features: torch.Tensor, memory: torch.Tensor | None = None):
        bsz = control_features.shape[0]
        device = control_features.device
        if memory is None:
            memory = self.reset_memory(bsz, device=device, dtype=control_features.dtype)

        k = F.normalize(self.k_proj(control_features), dim=-1)
        v = self.v_proj(control_features)
        beta = torch.sigmoid(self.beta_proj(control_features))
        lam = torch.sigmoid(self.lambda_proj(control_features))
        if not self.channel_decay:
            lam = lam.mean(dim=-1, keepdim=True).expand_as(lam)

        read = torch.bmm(memory, k.unsqueeze(-1)).squeeze(-1)
        correction = (v - read) * beta
        delta = torch.bmm(correction.unsqueeze(-1), k.unsqueeze(1))
        memory_new = memory * lam.unsqueeze(-1) + delta

        summary_raw = torch.bmm(memory_new, k.unsqueeze(-1)).squeeze(-1)
        summary = torch.tanh(self.summary_proj(summary_raw))
        return {
            "memory": memory_new,
            "summary": summary,
            "logit_bias": self.logit_bias(summary),
            "phase_logits": self.phase_head(summary),
            "winner_logits": self.winner_head(summary),
            "margin_logits": self.margin_head(summary),
        }

    def forward(self, control_features: torch.Tensor, memory: torch.Tensor | None = None, *, chunk_size: int | None = None):
        if control_features.ndim == 2:
            return self._step(control_features, memory)
        if control_features.ndim != 3:
            raise ValueError("control_features must have shape [B, D] or [B, T, D]")

        bsz, time, _ = control_features.shape
        if memory is None:
            memory = self.reset_memory(bsz, device=control_features.device, dtype=control_features.dtype)
        step = max(1, int(chunk_size or time or 1))
        outputs: dict[str, list[torch.Tensor]] = {
            "summary": [],
            "logit_bias": [],
            "phase_logits": [],
            "winner_logits": [],
            "margin_logits": [],
        }
        current = memory
        for start in range(0, time, step):
            end = min(time, start + step)
            for t in range(start, end):
                out = self._step(control_features[:, t, :], current)
                current = out["memory"]
                for key in outputs:
                    outputs[key].append(out[key])
        return {
            "memory": current,
            **{key: torch.stack(vals, dim=1) for key, vals in outputs.items()},
        }
