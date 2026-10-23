"""Tests for src/atom/heads/s2_heads.py (P5-2a: the S2 correctness/engagement head)."""

import torch

from atom.heads.s2_heads import S2CorrectnessHead, product_probability


def test_forward_shapes_no_k_axis():
    head = S2CorrectnessHead(hidden_dim=16, head_hidden=8)
    pooled = torch.randn(5, 16)
    correctness_logit, engagement_logit = head(pooled)
    assert correctness_logit.shape == (5,)
    assert engagement_logit.shape == (5,)


def test_forward_shapes_with_k_axis():
    # Caller manages the K-sample axis (K belief draws -> K personalised forwards -> K
    # pooled hidden states); the head must consume only the trailing hidden_dim axis and
    # leave every leading axis untouched, matching reparameterize's (..., k, d) convention.
    head = S2CorrectnessHead(hidden_dim=16, head_hidden=8)
    pooled = torch.randn(4, 8, 16)  # (N, K, hidden_dim)
    correctness_logit, engagement_logit = head(pooled)
    assert correctness_logit.shape == (4, 8)
    assert engagement_logit.shape == (4, 8)


def test_two_channels_are_independent_outputs():
    # Correctness and engagement must be two genuinely different linear read-outs of the
    # shared trunk, not the same logit duplicated -- a regression here would silently make
    # the "two-channel from the start" design (P3x-2) a no-op.
    head = S2CorrectnessHead(hidden_dim=16, head_hidden=8)
    pooled = torch.randn(10, 16)
    correctness_logit, engagement_logit = head(pooled)
    assert not torch.allclose(correctness_logit, engagement_logit)


def test_gradient_flows_to_pooled_hidden_and_all_params():
    # This is the property P5-2a's whole design hinges on: gradients from the correctness
    # loss must be able to flow back through `pooled_hidden` into whatever produced it
    # (ultimately, in the real pipeline, the personalised backbone -> LoRA basis ->
    # hypernetwork). A head with a dead gradient path here would silently defeat option (a)
    # even if the surrounding pipeline were wired correctly.
    head = S2CorrectnessHead(hidden_dim=12, head_hidden=6)
    pooled = torch.randn(7, 12, requires_grad=True)
    correctness_logit, engagement_logit = head(pooled)
    (correctness_logit.sum() + engagement_logit.sum()).backward()
    assert pooled.grad is not None
    assert torch.isfinite(pooled.grad).all()
    assert (pooled.grad.abs().sum() > 0)
    for p in head.parameters():
        assert p.grad is not None
        assert torch.isfinite(p.grad).all()


def test_product_probability_matches_manual_formula():
    correctness_logit = torch.tensor([0.0, 2.0, -3.0])
    engagement_logit = torch.tensor([0.0, -1.0, 4.0])
    p = product_probability(correctness_logit, engagement_logit)
    expected = (1 - torch.sigmoid(engagement_logit)) * torch.sigmoid(correctness_logit)
    assert torch.allclose(p, expected)


def test_product_probability_in_unit_interval():
    torch.manual_seed(0)
    correctness_logit = torch.randn(50) * 5
    engagement_logit = torch.randn(50) * 5
    p = product_probability(correctness_logit, engagement_logit)
    assert bool((p >= 0).all())
    assert bool((p <= 1).all())


def test_product_probability_certain_giveup_zeroes_out():
    # P(give-up) -> 1 must drive the product to ~0 regardless of how confident the
    # correctness channel is -- this is the routing rule (P3x-2) made numerically concrete:
    # a give-up dominates the observed outcome no matter what "would have happened" if
    # attempted.
    correctness_logit = torch.tensor([10.0])   # p_solve ~= 1
    engagement_logit = torch.tensor([50.0])    # p_giveup ~= 1
    p = product_probability(correctness_logit, engagement_logit)
    assert p.item() < 1e-4
