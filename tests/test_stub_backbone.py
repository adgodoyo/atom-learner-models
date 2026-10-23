"""Tests for src/atom/binding/stub_backbone.py -- the tiny, never-a-real-result backbone
the P5-0 kill test trains through (see its module docstring for why real gemma4-e4b is
deliberately not used here)."""

import torch

from atom.binding import attach_lora_basis
from atom.binding.stub_backbone import StubBackbone


def test_forward_shape():
    model = StubBackbone(dim=8, n_layers=2, seed=0)
    x = torch.randn(3, 5, 1)
    out = model(x)
    assert out.shape == (3, 5, 1)


def test_frozen_by_default():
    model = StubBackbone(dim=8, n_layers=2, seed=0)
    assert all(not p.requires_grad for p in model.parameters())


def test_deterministic_given_seed():
    a = StubBackbone(dim=8, n_layers=2, seed=7)
    b = StubBackbone(dim=8, n_layers=2, seed=7)
    for pa, pb in zip(a.parameters(), b.parameters(), strict=True):
        assert torch.equal(pa, pb)


def test_different_seeds_differ():
    a = StubBackbone(dim=8, n_layers=2, seed=1)
    b = StubBackbone(dim=8, n_layers=2, seed=2)
    assert any(
        not torch.equal(pa, pb) for pa, pb in zip(a.parameters(), b.parameters(), strict=True)
    )


def test_hookable_like_a_real_hf_model():
    # The whole point of naming these self_attn.q_proj/v_proj -- attach_lora_basis must
    # find them exactly as it would on a real backbone (already exercised for real on
    # gemma4-e4b in scripts/p5_00_hook_validate.py; this is the fast unit-level check).
    model = StubBackbone(dim=8, n_layers=3, seed=0)
    bases, context = attach_lora_basis(model, m=4, r=2)
    assert len(bases) == 3 * 2  # q_proj + v_proj per layer, k_proj/o_proj excluded
    assert all(name.endswith(("q_proj", "v_proj")) for name in bases)
