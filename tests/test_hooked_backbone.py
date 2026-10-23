"""Tests for src/atom/binding/hooked_backbone.py (P5/C6: attaching the shared LoRA basis
to a frozen backbone's q/v projections).

Uses a small synthetic model with HF-style naming (self_attn.q_proj/k_proj/v_proj across
several "layers", plus a vision_tower with its own q/v) rather than a real multi-GB
backbone -- this exercises the exact same named-module structure the real hook walks,
without a gated multi-GB download.
"""

import pytest
import torch
from torch import nn

from atom.binding import (GlobalMixingVector, Hypernetwork, LoRAHookedLinear, attach_lora_basis,
                          build_second_basis, set_active_basis)


class ToyAttn(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)  # must NOT be touched
        self.v_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        return self.q_proj(x) + self.v_proj(x)  # toy combine, just needs both on the path


class ToyLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.self_attn = ToyAttn(dim)

    def forward(self, x):
        return self.self_attn(x)


class ToyModel(nn.Module):
    def __init__(self, dim, n_layers):
        super().__init__()
        self.layers = nn.ModuleList([ToyLayer(dim) for _ in range(n_layers)])
        self.vision_tower = ToyAttn(dim)  # text-only filter must skip this entirely

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def test_attach_targets_only_text_qv_not_k_proj_or_vision():
    model = ToyModel(dim=6, n_layers=3)
    bases, _ = attach_lora_basis(model, m=4, r=2)
    assert len(bases) == 3 * 2  # q_proj + v_proj per layer, k_proj excluded
    assert all("vision" not in name for name in bases)
    assert all(name.endswith(("q_proj", "v_proj")) for name in bases)
    # k_proj must remain a bare nn.Linear, untouched
    for layer in model.layers:
        assert isinstance(layer.self_attn.k_proj, nn.Linear)
        assert not isinstance(layer.self_attn.k_proj, LoRAHookedLinear)
    assert isinstance(model.vision_tower.q_proj, nn.Linear)
    assert not isinstance(model.vision_tower.q_proj, LoRAHookedLinear)


def test_attach_raises_when_no_qv_modules_found():
    bare = nn.Sequential(nn.Linear(4, 4))
    try:
        attach_lora_basis(bare)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_frozen_weights_have_no_grad_after_attach():
    model = ToyModel(dim=6, n_layers=2)
    bases, _ = attach_lora_basis(model, m=4, r=2)
    for layer in model.layers:
        assert layer.self_attn.q_proj.frozen.weight.requires_grad is False
        assert layer.self_attn.v_proj.frozen.weight.requires_grad is False


def test_every_non_basis_param_in_the_whole_model_is_frozen():
    # Regression test for a real P5-1 bug: the original attach_lora_basis only froze the
    # specific wrapped q/v Linears (the assertion above) -- everything ELSE already on the
    # model (k_proj, o_proj/MLP-equivalent, norms, embeddings-equivalent) silently kept
    # HF's/PyTorch's default requires_grad=True. On the real gemma4-e2b that was 4.97B
    # untracked trainable params, discovered only when a real backward pass OOM'd on a
    # smoke test. This checks EVERY parameter in the whole model tree, not just the
    # wrapped ones -- basis params (inside the LoRAHookedLinear wrappers) are the only
    # ones allowed to still require grad.
    model = ToyModel(dim=6, n_layers=2)
    basis_param_ids = set()
    bases, _ = attach_lora_basis(model, m=4, r=2)
    for basis in bases.values():
        basis_param_ids |= {id(p) for p in basis.parameters()}

    for name, p in model.named_parameters():
        if id(p) in basis_param_ids:
            assert p.requires_grad is True, f"basis param {name} should be trainable"
        else:
            assert p.requires_grad is False, f"non-basis param {name} was left trainable"


def test_untrained_basis_leaves_output_unchanged():
    # B is zero-init (LoRABasis convention) -- attaching an untrained basis must be an
    # exact no-op on the model's output, same invariant as test_binding.py but end-to-end
    # through a hooked multi-layer forward.
    torch.manual_seed(0)
    model = ToyModel(dim=6, n_layers=3)
    x = torch.randn(5, 7, 6)  # (batch, seq, dim)
    with torch.no_grad():
        before = model(x)

    _, context = attach_lora_basis(model, m=4, r=2, seed=1)
    context.set(torch.randn(5, 4))  # arbitrary c -- must not matter while B is all-zero
    with torch.no_grad():
        after = model(x)
    assert torch.allclose(before, after)


def test_hooked_model_changes_output_once_basis_is_trained():
    torch.manual_seed(0)
    model = ToyModel(dim=6, n_layers=2)
    x = torch.randn(3, 5, 6)
    with torch.no_grad():
        before = model(x)

    bases, context = attach_lora_basis(model, m=4, r=2, seed=2)
    for basis in bases.values():
        with torch.no_grad():
            basis.B.normal_(generator=torch.Generator().manual_seed(3))
    context.set(torch.randn(3, 4))
    with torch.no_grad():
        after = model(x)
    assert not torch.allclose(before, after)


def test_context_broadcasts_c_per_example_across_sequence_positions():
    # Personalisation is per-example, not per-token: two different rows with the SAME c
    # must get the same delta contribution shape-for-shape; changing only one row's c must
    # leave the other row's output untouched.
    torch.manual_seed(0)
    model = ToyModel(dim=6, n_layers=1)
    bases, context = attach_lora_basis(model, m=4, r=2, seed=4)
    for basis in bases.values():
        with torch.no_grad():
            basis.B.normal_(generator=torch.Generator().manual_seed(5))

    x = torch.randn(2, 3, 6)
    c = torch.stack([torch.zeros(4), torch.randn(4)])  # row 0: c=0 -> delta=0 for row 0
    context.set(c)
    with torch.no_grad():
        out = model(x)

    context.set(torch.zeros(2, 4))
    with torch.no_grad():
        baseline = model(x)
    assert torch.allclose(out[0], baseline[0])       # row 0 (c=0) matches the c=0 baseline
    assert not torch.allclose(out[1], baseline[1])    # row 1 (c!=0) must differ


def test_missing_context_raises():
    model = ToyModel(dim=6, n_layers=1)
    attach_lora_basis(model, m=4, r=2)
    x = torch.randn(2, 3, 6)
    try:
        model(x)
        raise AssertionError("expected RuntimeError for unset context.c")
    except RuntimeError:
        pass


def test_mismatched_context_batch_raises():
    model = ToyModel(dim=6, n_layers=1)
    _, context = attach_lora_basis(model, m=4, r=2)
    context.set(torch.randn(3, 4))  # batch=3, but x below has batch=2
    x = torch.randn(2, 5, 6)
    try:
        model(x)
        raise AssertionError("expected ValueError for mismatched batch")
    except ValueError:
        pass


def test_hooked_linear_survives_bf16_frozen_weights():
    # The real backbone loads in bf16 (P2-1); the basis stays fp32 for the low-rank
    # branch's stability. A bf16 frozen linear + fp32 basis must not raise a dtype
    # mismatch, and the untrained-no-op invariant must still hold in mixed dtype.
    torch.manual_seed(0)
    model = ToyModel(dim=6, n_layers=2).to(torch.bfloat16)
    x = torch.randn(3, 5, 6, dtype=torch.bfloat16)
    with torch.no_grad():
        before = model(x)

    _, context = attach_lora_basis(model, m=4, r=2, seed=9)  # basis stays fp32 by default
    context.set(torch.randn(3, 4))  # fp32 c
    with torch.no_grad():
        after = model(x)
    assert after.dtype == torch.bfloat16
    assert torch.allclose(before, after)


def test_gradients_flow_from_hooked_output_to_basis_and_hypernetwork():
    # End-to-end check that the whole chain (u -> hypernetwork -> c -> hooked layers ->
    # output) is differentiable, the property the pilot training loop actually depends on.
    torch.manual_seed(0)
    model = ToyModel(dim=6, n_layers=2)
    bases, context = attach_lora_basis(model, m=4, r=2, seed=6)
    for basis in bases.values():
        with torch.no_grad():
            basis.B.normal_(generator=torch.Generator().manual_seed(7))

    hyper = Hypernetwork(u_dim=3, m=4, kind="mlp", hidden=8, seed=8)
    u = torch.randn(2, 3, requires_grad=True)
    c = hyper(u)
    context.set(c)

    x = torch.randn(2, 5, 6)
    out = model(x)
    out.sum().backward()

    assert u.grad is not None and u.grad.abs().sum() > 0
    for name, p in hyper.named_parameters():
        assert p.grad is not None and p.grad.abs().sum() > 0, name
    for name, basis in bases.items():
        assert basis.A.grad is not None and basis.A.grad.abs().sum() > 0, name
        assert basis.B.grad is not None and basis.B.grad.abs().sum() > 0, name


# -- FA-6c global-LoRA control arm: build_second_basis / set_active_basis --------------

def test_build_second_basis_raises_before_attach_lora_basis():
    model = ToyModel(dim=6, n_layers=2)
    with pytest.raises(ValueError):
        build_second_basis(model)


def test_build_second_basis_matches_shapes_and_is_independent():
    model = ToyModel(dim=6, n_layers=3)
    bases_a, _ = attach_lora_basis(model, m=4, r=2, seed=0)
    bases_b = build_second_basis(model, seed=1000)

    assert set(bases_a.keys()) == set(bases_b.keys())
    for name in bases_a:
        a, b = bases_a[name], bases_b[name]
        assert (b.A.shape, b.B.shape) == (a.A.shape, a.B.shape)
        assert b.in_dim == a.in_dim and b.out_dim == a.out_dim
        assert b.m == a.m and b.r == a.r and b.alpha == a.alpha
        # different objects, different (random) A-init -- not aliased params.
        assert b is not a
        assert not torch.allclose(b.A, a.A)


def test_set_active_basis_switches_which_weights_a_forward_pass_uses():
    torch.manual_seed(0)
    model = ToyModel(dim=6, n_layers=2)
    bases_a, context = attach_lora_basis(model, m=4, r=2, seed=0)
    bases_b = build_second_basis(model, seed=1000)
    # un-zero BOTH bases' B (LoRABasis zero-inits B, so an untrained basis is a no-op --
    # need a real, DIFFERENT delta from each to prove the switch actually changes output).
    for name in bases_a:
        with torch.no_grad():
            bases_a[name].B.normal_(generator=torch.Generator().manual_seed(1))
            bases_b[name].B.normal_(generator=torch.Generator().manual_seed(2))

    x = torch.randn(3, 5, 6)
    context.set(torch.randn(3, 4))
    with torch.no_grad():
        out_a = model(x)

    set_active_basis(model, bases_b)
    with torch.no_grad():
        out_b = model(x)

    assert not torch.allclose(out_a, out_b)

    set_active_basis(model, bases_a)  # switching back is exact, not lossy
    with torch.no_grad():
        out_a_again = model(x)
    assert torch.allclose(out_a, out_a_again)


def test_set_active_basis_raises_on_missing_layer():
    model = ToyModel(dim=6, n_layers=2)
    bases_a, _ = attach_lora_basis(model, m=4, r=2, seed=0)
    incomplete = dict(list(bases_a.items())[:-1])  # drop one layer
    with pytest.raises(KeyError):
        set_active_basis(model, incomplete)


def test_global_mixing_vector_broadcast_through_a_hooked_forward_ignores_which_row_is_which():
    """The global-control arm's whole point: `GlobalMixingVector` broadcasts ONE learned
    `c` to every row, so two DIFFERENT input rows under the control arm's basis get the
    SAME mixing vector (unlike the personalised arm, where `c=hyper(u)` varies per row)."""
    torch.manual_seed(0)
    model = ToyModel(dim=6, n_layers=2)
    _, context = attach_lora_basis(model, m=4, r=2, seed=0)
    gmv = GlobalMixingVector(m=4, seed=3)

    n = 3
    context.set(gmv(n))
    assert context.c.shape == (n, 4)
    assert torch.equal(context.c[0], context.c[1])
    assert torch.equal(context.c[1], context.c[2])
