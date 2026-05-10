import torch

from cdsd.control_delta_block import ControlDeltaBlock


def test_control_delta_step_shapes_finite_and_gradients():
    block = ControlDeltaBlock(5, 7, 3)
    x = torch.randn(4, 5, requires_grad=True)
    out = block(x)
    assert out["memory"].shape == (4, 7, 7)
    assert out["summary"].shape == (4, 7)
    assert out["logit_bias"].shape == (4, 3)
    assert out["winner_logits"].shape == (4, 3)
    assert all(torch.isfinite(v).all() for v in out.values())
    loss = out["logit_bias"].sum() + out["phase_logits"].sum() + out["margin_logits"].sum()
    loss.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_control_delta_sequence_chunked_scan_and_reset():
    torch.manual_seed(0)
    block = ControlDeltaBlock(5, 7, 3, channel_decay=False)
    x = torch.randn(2, 6, 5)
    full = block(x)
    chunked = block(x, chunk_size=2)
    reset = block.reset_memory(2, device=x.device, dtype=x.dtype)
    fresh = block(x, memory=reset)
    assert full["logit_bias"].shape == (2, 6, 3)
    assert full["phase_logits"].shape == (2, 6, 8)
    assert full["margin_logits"].shape == (2, 6, 4)
    assert torch.allclose(full["memory"], chunked["memory"])
    assert torch.allclose(full["memory"], fresh["memory"])
