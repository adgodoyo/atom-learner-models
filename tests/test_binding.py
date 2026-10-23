"""Tests for src/atom/binding/ (P5/C6: LoRA basis, hypernetwork, K-sample machinery)."""

import torch

from atom.binding import (
    GlobalMixingVector,
    Hypernetwork,
    LoRABasis,
    mean_prediction,
    person_epistemic_spread,
    reparameterize,
)


def test_lora_basis_untrained_is_exact_zero():
    # B is zero-initialised (standard LoRA convention) -- an untrained basis must be a
    # strict no-op regardless of what c is, so training starts from the frozen backbone
    # exactly, not from some random perturbation of it.
    basis = LoRABasis(in_dim=6, out_dim=4, m=8, r=2, seed=0)
    x = torch.randn(5, 6)
    c = torch.randn(5, 8)
    out = basis.apply(x, c)
    assert torch.allclose(out, torch.zeros(5, 4))


def test_lora_basis_apply_matches_naive_materialised_delta():
    # The whole point of `apply` is to never materialise (N, out, in) DeltaW -- verify its
    # efficient einsum path agrees with the naive "build DeltaW then matmul" computation.
    basis = LoRABasis(in_dim=10, out_dim=7, m=6, r=3, seed=1)
    with torch.no_grad():
        basis.B.normal_(generator=torch.Generator().manual_seed(2))  # un-zero B for this test
    x = torch.randn(4, 10)
    c = torch.randn(4, 6)

    fast = basis.apply(x, c)

    naive = torch.empty(4, 7)
    for b in range(4):
        delta_w = torch.zeros(7, 10)
        for i in range(6):
            delta_w += c[b, i] * (basis.B[i] @ basis.A[i])  # B_i (out,r) @ A_i (r,in)
        naive[b] = (basis.alpha / basis.r) * (delta_w @ x[b])

    assert torch.allclose(fast, naive, atol=1e-5)


def test_lora_basis_delta_weight_matches_apply():
    basis = LoRABasis(in_dim=5, out_dim=4, m=4, r=2, seed=3)
    with torch.no_grad():
        basis.B.normal_(generator=torch.Generator().manual_seed(4))
    x = torch.randn(3, 5)
    c = torch.randn(3, 4)
    via_apply = basis.apply(x, c)
    dw = basis.delta_weight(c)  # (3, out, in)
    via_delta_weight = torch.einsum("noi,ni->no", dw, x)
    assert torch.allclose(via_apply, via_delta_weight, atol=1e-5)


def test_lora_basis_mismatched_leading_dim_raises():
    basis = LoRABasis(in_dim=4, out_dim=4, m=4, r=2)
    x = torch.randn(3, 4)
    c = torch.randn(5, 4)
    try:
        basis.apply(x, c)
        raise AssertionError("expected ValueError for mismatched N")
    except ValueError:
        pass


def test_lora_basis_gradients_flow_to_A_and_B():
    basis = LoRABasis(in_dim=6, out_dim=5, m=4, r=2, seed=5)
    with torch.no_grad():
        basis.B.normal_(generator=torch.Generator().manual_seed(6))
    x = torch.randn(3, 6)
    c = torch.randn(3, 4, requires_grad=True)
    out = basis.apply(x, c)
    out.sum().backward()
    assert basis.A.grad is not None and basis.A.grad.abs().sum() > 0
    assert basis.B.grad is not None and basis.B.grad.abs().sum() > 0
    assert c.grad is not None and c.grad.abs().sum() > 0


def test_hypernetwork_identity_requires_matching_dims():
    Hypernetwork(u_dim=8, m=8, kind="identity")  # ok
    try:
        Hypernetwork(u_dim=8, m=4, kind="identity")
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_hypernetwork_identity_passes_through():
    hn = Hypernetwork(u_dim=6, m=6, kind="identity")
    u = torch.randn(4, 6)
    assert torch.equal(hn(u), u)


def test_hypernetwork_linear_and_mlp_shapes():
    for kind in ("linear", "mlp"):
        hn = Hypernetwork(u_dim=10, m=32, kind=kind, hidden=16)
        u = torch.randn(7, 10)
        c = hn(u)
        assert c.shape == (7, 32)


def test_hypernetwork_rejects_unknown_kind():
    try:
        Hypernetwork(u_dim=4, m=4, kind="bogus")
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_reparameterize_shape_and_determinism_given_generator():
    mu = torch.zeros(3, 5)
    sigma = torch.ones(3, 5)
    g1 = torch.Generator().manual_seed(0)
    g2 = torch.Generator().manual_seed(0)
    s1 = reparameterize(mu, sigma, k=8, generator=g1)
    s2 = reparameterize(mu, sigma, k=8, generator=g2)
    assert s1.shape == (3, 8, 5)
    assert torch.equal(s1, s2)


def test_reparameterize_zero_sigma_collapses_to_mu():
    mu = torch.randn(2, 4)
    sigma = torch.zeros(2, 4)
    s = reparameterize(mu, sigma, k=5)
    for k in range(5):
        assert torch.allclose(s[:, k, :], mu)


def test_reparameterize_gradients_flow_to_mu_and_sigma():
    mu = torch.zeros(2, 3, requires_grad=True)
    sigma = torch.ones(2, 3, requires_grad=True)
    s = reparameterize(mu, sigma, k=4, generator=torch.Generator().manual_seed(0))
    s.sum().backward()
    assert mu.grad is not None and sigma.grad is not None
    assert sigma.grad.abs().sum() > 0  # eps must be nonzero for at least one draw


def test_reparameterize_mismatched_shapes_raise():
    try:
        reparameterize(torch.zeros(2, 3), torch.zeros(2, 4), k=2)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_mean_and_spread_on_identical_samples_is_degenerate():
    preds = torch.full((5, 8), 0.7)  # K=8 identical predictions per example
    assert torch.allclose(mean_prediction(preds), torch.full((5,), 0.7))
    assert torch.allclose(person_epistemic_spread(preds), torch.zeros(5))


def test_person_epistemic_spread_reflects_real_disagreement():
    # Two examples: one where all K samples agree (spread ~0), one where they don't.
    preds = torch.stack([torch.full((8,), 0.5), torch.linspace(0.0, 1.0, 8)])
    spread = person_epistemic_spread(preds)
    assert spread[0] < 1e-6
    assert spread[1] > spread[0]


# -- FA-6 gate (iii)'s global-LoRA control --------------------------------------------

def test_global_mixing_vector_broadcasts_the_same_vector_to_every_row():
    gmv = GlobalMixingVector(m=6, seed=0)
    c = gmv(n=4)
    assert c.shape == (4, 6)
    for i in range(1, 4):
        assert torch.equal(c[0], c[i])  # every row is the SAME vector, not sampled per-row


def test_global_mixing_vector_ignores_n_when_computing_the_shared_vector():
    """Calling with different batch sizes must not change the underlying learned vector --
    it's one parameter, broadcast, never re-derived from n."""
    gmv = GlobalMixingVector(m=5, seed=1)
    c_small = gmv(n=2)
    c_large = gmv(n=7)
    assert torch.equal(c_small[0], c_large[0])


def test_global_mixing_vector_is_trainable():
    gmv = GlobalMixingVector(m=4, seed=2)
    c = gmv(n=3)
    c.sum().backward()
    assert gmv.c.grad is not None
    assert torch.isfinite(gmv.c.grad).all()
