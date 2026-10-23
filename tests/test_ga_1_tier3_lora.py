"""Lane T0 -- GA-1 tier 3 rung A: adapter canaries (no backbone weights needed)."""

from __future__ import annotations

import pathlib
import sys

import pytest
import torch
from torch import nn

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))

import ga_1_tier3_lora as L  # noqa: E402


class TinyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(8, 8)
        self.k_proj = nn.Linear(8, 8)
        self.v_proj = nn.Linear(8, 8)
        self.o_proj = nn.Linear(8, 8)
        self.mlp = nn.Linear(8, 8)


class TinyModel(nn.Module):
    def __init__(self, n=3):
        super().__init__()
        self.layers = nn.ModuleList([TinyBlock() for _ in range(n)])
        self.embed = nn.Embedding(10, 8)
        self.lm_head = nn.Linear(8, 10)

    def forward(self, x):
        h = self.embed(x)
        for blk in self.layers:
            h = blk.o_proj(blk.q_proj(h) + blk.v_proj(h) + blk.k_proj(h))
        return self.lm_head(h)


def test_adapter_is_an_exact_no_op_at_initialisation():
    """B is zero-init, so the wrapped model must reproduce the frozen one bit-for-bit."""
    torch.manual_seed(0)
    m = TinyModel()
    x = torch.randint(0, 10, (2, 5))
    before = m(x).clone()
    L.attach_plain_lora(m, seed=0)
    after = m(x)
    assert torch.equal(before, after), "LoRA at init must be an exact no-op"


def test_only_qv_are_adapted_and_only_lora_is_trainable():
    m = TinyModel(n=3)
    adapters = L.attach_plain_lora(m)
    assert len(adapters) == 6, "3 layers x {q,v} expected"
    assert all(n.endswith(("q_proj", "v_proj")) for n in adapters)
    assert not any("k_proj" in n or "o_proj" in n or "mlp" in n for n in adapters)
    trainable = L.trainable_parameters(m)
    assert len(trainable) == 12, "two tensors per adapted layer, nothing else"
    names = {n for n, p in m.named_parameters() if p.requires_grad}
    assert all("lora_" in n for n in names), f"non-LoRA params are trainable: {names}"


def test_every_adapter_receives_gradient_and_moves_under_a_step():
    """Canary family #6: forwards running is not evidence that a module learns."""
    torch.manual_seed(0)
    m = TinyModel()
    adapters = L.attach_plain_lora(m)
    opt = torch.optim.Adam(L.trainable_parameters(m), lr=1e-2)
    before = {n: a.lora_b.detach().clone() for n, a in adapters.items()}
    out = m(torch.randint(0, 10, (2, 5)))
    out.square().mean().backward()
    for n, a in adapters.items():
        assert a.lora_a.grad is not None and torch.isfinite(a.lora_a.grad).all(), n
        assert a.lora_b.grad is not None and a.lora_b.grad.abs().sum() > 0, n
    opt.step()
    for n, a in adapters.items():
        assert not torch.equal(before[n], a.lora_b), f"{n} did not move after a step"


def test_adapters_disabled_restores_the_base_function_and_the_scaling():
    torch.manual_seed(0)
    m = TinyModel()
    x = torch.randint(0, 10, (2, 5))
    adapters = L.attach_plain_lora(m)
    with torch.no_grad():
        for a in adapters.values():
            a.lora_b.add_(torch.randn_like(a.lora_b) * 0.1)
    adapted = m(x).clone()
    with L.adapters_disabled(adapters):
        base = m(x).clone()
    restored = m(x)
    assert not torch.allclose(adapted, base), "disabling the adapter changed nothing"
    assert torch.equal(adapted, restored), "scaling was not restored on exit"
    assert all(a.scaling == L.LORA_ALPHA / L.LORA_R for a in adapters.values())


def test_adapter_state_roundtrip():
    torch.manual_seed(0)
    m1, m2 = TinyModel(), TinyModel()
    a1 = L.attach_plain_lora(m1, seed=0)
    a2 = L.attach_plain_lora(m2, seed=7)
    with torch.no_grad():
        for a in a1.values():
            a.lora_b.add_(torch.randn_like(a.lora_b))
    L.load_adapter_state(a2, L.adapter_state_dict(a1))
    for n in a1:
        assert torch.equal(a1[n].lora_b.cpu(), a2[n].lora_b.cpu())


def test_load_rejects_an_incomplete_state():
    m = TinyModel()
    a = L.attach_plain_lora(m)
    state = L.adapter_state_dict(a)
    state.pop(next(iter(state)))
    with pytest.raises(KeyError):
        L.load_adapter_state(a, state)
