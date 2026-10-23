"""Unit tests for the S1 filter: episode encoder, GRU belief filter, FiLM heads,
free-bits KL -- all on synthetic tensors, no real data dependency (fast, CPU).

The causality tests are the load-bearing ones: they turn "predict, then correct, and only
ever predict from the pre-observation belief" into a script-checked
invariant, the same discipline the leakage audits apply to the data splits.
"""

import numpy as np
import pytest
import torch
from torch import nn

from atom.filter.diagnostics import collapse_diagnostics, per_learner_mean_error
from atom.filter.episode import EpisodeEncoder, build_prev_feat
from atom.filter.gru import KnowledgeFilter, sample_u
from atom.filter.losses import (gaussian_kl_free_bits, gaussian_kl_free_bits_mu_only,
                                gaussian_kl_two_diagonal_free_bits,
                                kl_anneal_weight, masked_bce)
from atom.filter.s1 import S1Filter
from atom.heads.s1_heads import FiLMHeads


def test_episode_encoder_shape():
    enc = EpisodeEncoder(codec_dim=8, hidden=16)
    item = torch.randn(3, 5, 8)
    prev = torch.randn(3, 5, 4)
    out = enc(item, prev)
    assert out.shape == (3, 5, 16)


def test_build_prev_feat_shifts_by_one_with_start_flag():
    correct = torch.tensor([[1.0, 0.0, 1.0]])
    hint = torch.tensor([[0.0, 1.0, 0.0]])
    saw = torch.tensor([[0.0, 0.0, 1.0]])
    prev = build_prev_feat(correct, hint, saw)
    assert prev.shape == (1, 3, 4)
    # t=0: no history yet -> all zero, including the valid flag.
    assert torch.equal(prev[0, 0], torch.zeros(4))
    # t=1: carries t=0's (correct=1, hint=0, saw=0), valid=1.
    assert torch.equal(prev[0, 1], torch.tensor([1.0, 0.0, 0.0, 1.0]))
    # t=2: carries t=1's (correct=0, hint=1, saw=0), valid=1.
    assert torch.equal(prev[0, 2], torch.tensor([0.0, 1.0, 0.0, 1.0]))


def test_knowledge_filter_shapes_and_log_sigma_clamp():
    kf = KnowledgeFilter(input_dim=16, hidden=32, latent_dim=6,
                         log_sigma_min=-2.0, log_sigma_max=1.0)
    e_ep = torch.randn(4, 7, 16) * 100  # large input to try to push log_sigma out of range
    mu, log_sigma = kf(e_ep)
    assert mu.shape == (4, 7, 6)
    assert log_sigma.shape == (4, 7, 6)
    assert torch.all(log_sigma >= -2.0) and torch.all(log_sigma <= 1.0)


def test_precision_accumulating_sigma_is_prior_at_t0_and_only_narrows():
    torch.manual_seed(4)
    kf = KnowledgeFilter(
        input_dim=7, hidden=9, latent_dim=5,
        sigma_parameterization="precision_accumulating",
    )
    e_ep = torch.randn(3, 12, 7)
    _, log_sigma = kf(e_ep)

    assert torch.equal(log_sigma[:, 0], torch.zeros_like(log_sigma[:, 0]))
    assert torch.all(log_sigma[:, 1:] <= log_sigma[:, :-1])
    assert torch.all(log_sigma <= 0.0)
    assert torch.all(log_sigma > kf.log_sigma_min)


def test_precision_accumulating_sigma_matches_registered_formula_and_gets_gradient():
    torch.manual_seed(5)
    kf = KnowledgeFilter(
        input_dim=6, hidden=8, latent_dim=4,
        sigma_parameterization="precision_accumulating",
    )
    e_ep = torch.randn(2, 7, 6)
    _, log_sigma = kf(e_ep)

    increments = torch.nn.functional.softplus(kf.precision_head(e_ep))
    increments = torch.cat([torch.zeros_like(increments[:, :1]), increments[:, 1:]], dim=1)
    expected = -0.5 * torch.log(1.0 + torch.cumsum(increments, dim=1))
    assert torch.allclose(log_sigma, expected)

    log_sigma[:, -1].sum().backward()
    for name, p in kf.precision_head.named_parameters():
        assert p.grad is not None, f"precision_head.{name} received no gradient"
        assert torch.isfinite(p.grad).all()
        assert p.grad.abs().sum() > 0


def test_direct_sigma_parameterization_remains_the_exact_default():
    torch.manual_seed(6)
    implicit = KnowledgeFilter(input_dim=5, hidden=7, latent_dim=3)
    torch.manual_seed(6)
    explicit = KnowledgeFilter(
        input_dim=5, hidden=7, latent_dim=3, sigma_parameterization="direct"
    )
    assert implicit.state_dict().keys() == explicit.state_dict().keys()
    for key in implicit.state_dict():
        assert torch.equal(implicit.state_dict()[key], explicit.state_dict()[key])
    x = torch.randn(2, 4, 5)
    got = implicit(x)
    want = explicit(x)
    assert all(torch.equal(a, b) for a, b in zip(got, want))


def test_sample_u_shape_and_variability():
    mu = torch.zeros(2, 3, 5)
    log_sigma = torch.zeros(2, 3, 5)  # sigma = 1
    u = sample_u(mu, log_sigma, k=8)
    assert u.shape == (8, 2, 3, 5)
    # 8 independent samples of a non-degenerate Gaussian must not all coincide.
    assert u.std(dim=0).mean().item() > 0.1


def test_s1_filter_end_to_end_shape():
    model = S1Filter(codec_dim=10, episode_hidden=12, filter_hidden=14, latent_dim=6)
    item = torch.randn(3, 9, 10)
    prev = torch.randn(3, 9, 4)
    mu, log_sigma = model(item, prev)
    assert mu.shape == (3, 9, 6)
    assert log_sigma.shape == (3, 9, 6)


def test_film_heads_shapes_and_near_identity_at_init():
    heads = FiLMHeads(codec_dim=10, latent_dim=6, hidden=14)
    item = torch.randn(2, 4, 10)
    u = torch.randn(5, 2, 4, 6)
    correctness_logits, engagement_logits = heads(item, u)
    assert correctness_logits.shape == (5, 2, 4)
    assert engagement_logits.shape == (5, 2, 4, 2)
    # Near-identity init: predictions for different samples k of u should be close (not
    # identical, since gamma/beta weights start with small nonzero noise, not exact zero).
    spread = correctness_logits.std(dim=0).mean().item()
    assert spread < 0.5


def test_gaussian_kl_free_bits_zero_at_prior_but_floored():
    mu = torch.zeros(2, 3, 4)
    log_sigma = torch.zeros(2, 3, 4)  # matches the prior N(0, I) exactly -> raw KL = 0
    kl = gaussian_kl_free_bits(mu, log_sigma, free_bits=0.15)
    # Free bits floors each of the 4 dims at 0.15 before summing.
    assert torch.allclose(kl, torch.full((2, 3), 0.15 * 4))


def test_gaussian_kl_free_bits_matches_closed_form_away_from_prior():
    mu = torch.tensor([[[2.0, 0.0]]])       # [1,1,2]
    log_sigma = torch.tensor([[[0.0, 0.0]]])  # sigma = 1
    kl = gaussian_kl_free_bits(mu, log_sigma, free_bits=0.0)
    # dim0: 0.5*(mu^2) = 0.5*4 = 2.0 ; dim1: 0 -> floored at 0 (free_bits=0, no-op)
    assert torch.allclose(kl, torch.tensor([[2.0]]), atol=1e-5)


def test_gaussian_kl_free_bits_mu_only_floors_only_the_mu_term():
    """R2-b (ROUND 2): at the prior width (log_sigma=0, no width penalty) but mu != 0,
    the mu-term is floored by free_bits_mu -- same floor behaviour as the combined
    function, isolated to mu alone."""
    mu = torch.zeros(2, 3, 4)
    log_sigma = torch.zeros(2, 3, 4)
    kl = gaussian_kl_free_bits_mu_only(mu, log_sigma, free_bits_mu=0.15)
    assert torch.allclose(kl, torch.full((2, 3), 0.15 * 4))


def test_gaussian_kl_free_bits_mu_only_never_floors_the_width_term():
    """The defining difference from `gaussian_kl_free_bits`: with mu=0 (so the mu-term is
    exactly 0, below any positive free_bits_mu) and log_sigma far from the prior, the
    width term is NOT clamped -- it passes through at its own closed-form value, unlike
    the combined function which would floor the WHOLE per-dim KL (mu-term + width-term)
    together and could mask a large negative width contribution."""
    mu = torch.zeros(1, 1, 2)
    log_sigma = torch.tensor([[[2.0, 0.0]]])  # sigma = e^2 >> 1: width term is large and positive
    kl = gaussian_kl_free_bits_mu_only(mu, log_sigma, free_bits_mu=0.05)
    # dim0: mu-term 0 -> floored to 0.05; width-term 0.5*(e^4 - 1 - 4) = 0.5*(54.598-5)
    # dim1: mu-term 0 -> floored to 0.05; width-term 0 (sigma=1, at the prior)
    expected_dim0 = 0.05 + 0.5 * (torch.tensor(4.0).exp() - 1.0 - 4.0)
    expected_dim1 = 0.05 + 0.0
    assert torch.allclose(kl, (expected_dim0 + expected_dim1).reshape(1, 1), atol=1e-4)


def test_gaussian_kl_free_bits_mu_only_matches_closed_form_away_from_prior():
    mu = torch.tensor([[[2.0, 0.0]]])
    log_sigma = torch.tensor([[[0.0, 0.0]]])  # sigma = 1, width term = 0 in both dims
    kl = gaussian_kl_free_bits_mu_only(mu, log_sigma, free_bits_mu=0.0)
    assert torch.allclose(kl, torch.tensor([[2.0]]), atol=1e-5)


def test_gaussian_kl_two_diagonal_free_bits_floors_the_summed_total_at_identical_distributions():
    """Sum-then-clamp (Dreamer's own convention, NOT `gaussian_kl_free_bits`'s per-
    dimension form -- see the function's own docstring for why): q == p exactly -> raw
    per-dim KL is 0 in all 4 dims, summed total is 0, floored at the SCALAR free_bits
    value once, not 4x."""
    mu_q = mu_p = torch.zeros(2, 3, 4)
    log_sigma_q = log_sigma_p = torch.zeros(2, 3, 4)
    kl = gaussian_kl_two_diagonal_free_bits(mu_q, log_sigma_q, mu_p, log_sigma_p, free_bits=0.15)
    assert torch.allclose(kl, torch.full((2, 3), 0.15))


def test_gaussian_kl_two_diagonal_free_bits_matches_unclamped_away_from_floor():
    from atom.filter.losses import gaussian_kl_two_diagonal
    mu_q = torch.tensor([[[2.0, 0.0]]])
    log_sigma_q = torch.tensor([[[0.0, 1.0]]])
    mu_p = torch.tensor([[[0.0, 0.0]]])
    log_sigma_p = torch.tensor([[[0.0, 0.0]]])
    unclamped = gaussian_kl_two_diagonal(mu_q, log_sigma_q, mu_p, log_sigma_p)
    clamped = gaussian_kl_two_diagonal_free_bits(mu_q, log_sigma_q, mu_p, log_sigma_p, free_bits=0.0)
    # free_bits=0.0 is a no-op floor whenever every per-dim term is already >= 0, so the
    # per-dim-then-sum form must match the unclamped closed form exactly.
    assert torch.allclose(clamped, unclamped, atol=1e-5)


def test_gaussian_kl_two_diagonal_free_bits_is_asymmetric_in_q_and_p():
    """The Dreamer-style balanced-KL use relies on `KL(q||p) != KL(p||q)` in
    general -- L_dyn and L_rep swap which distribution is detached and which is q vs p, so
    this must not silently be a symmetric quantity."""
    mu_a = torch.tensor([[[1.5, -0.5]]])
    log_sigma_a = torch.tensor([[[0.3, -0.2]]])
    mu_b = torch.tensor([[[-0.5, 0.5]]])
    log_sigma_b = torch.tensor([[[-0.1, 0.4]]])
    kl_ab = gaussian_kl_two_diagonal_free_bits(mu_a, log_sigma_a, mu_b, log_sigma_b, free_bits=0.0)
    kl_ba = gaussian_kl_two_diagonal_free_bits(mu_b, log_sigma_b, mu_a, log_sigma_a, free_bits=0.0)
    assert not torch.allclose(kl_ab, kl_ba)


def test_kl_anneal_weight_ramps_linearly_then_holds():
    assert kl_anneal_weight(0, max_weight=0.4, anneal_epochs=10) == 0.0
    assert abs(kl_anneal_weight(5, max_weight=0.4, anneal_epochs=10) - 0.2) < 1e-9
    assert kl_anneal_weight(10, max_weight=0.4, anneal_epochs=10) == 0.4
    assert kl_anneal_weight(50, max_weight=0.4, anneal_epochs=10) == 0.4


# -- causality (no-peek) -------------------------------------------------------------------

def test_belief_at_t_is_unaffected_by_outcome_at_t():
    """'Predict, then correct, and only ever predict from the
    pre-observation belief'. Changing step t's OWN outcome must not change mu/log_sigma
    AT t (only at t+1 onward, once it becomes history)."""
    torch.manual_seed(0)
    model = S1Filter(codec_dim=6, episode_hidden=8, filter_hidden=10, latent_dim=4)
    model.eval()

    item = torch.randn(1, 5, 6)
    correct_a = torch.tensor([[1.0, 0.0, 1.0, 0.0, 1.0]])
    correct_b = correct_a.clone()
    correct_b[0, 2] = 1.0 - correct_b[0, 2]  # flip ONLY step t=2's own outcome
    hint = torch.zeros(1, 5)
    saw = torch.zeros(1, 5)

    with torch.no_grad():
        prev_a = build_prev_feat(correct_a, hint, saw)
        prev_b = build_prev_feat(correct_b, hint, saw)
        mu_a, ls_a = model(item, prev_a)
        mu_b, ls_b = model(item, prev_b)

    # Steps 0, 1, 2 (up to and including the flipped step) must be identical: none of them
    # have "seen" t=2's outcome yet -- t=2's own belief is formed before t=2 is observed.
    assert torch.allclose(mu_a[0, :3], mu_b[0, :3])
    assert torch.allclose(ls_a[0, :3], ls_b[0, :3])
    # Step 3 onward MUST differ: t=2's outcome is now history feeding prev_feat[3].
    assert not torch.allclose(mu_a[0, 3:], mu_b[0, 3:])


def test_belief_at_t_is_unaffected_by_future_items():
    """Truncating the sequence after t must not change mu/log_sigma at steps <= t (GRU
    causality: no leakage from future items/padding into past predictions)."""
    torch.manual_seed(1)
    model = S1Filter(codec_dim=6, episode_hidden=8, filter_hidden=10, latent_dim=4)
    model.eval()

    item_full = torch.randn(1, 6, 6)
    correct = torch.randint(0, 2, (1, 6)).float()
    hint = torch.zeros(1, 6)
    saw = torch.zeros(1, 6)

    with torch.no_grad():
        prev_full = build_prev_feat(correct, hint, saw)
        mu_full, _ = model(item_full, prev_full)

        item_trunc = item_full[:, :4]
        prev_trunc = build_prev_feat(correct[:, :4], hint[:, :4], saw[:, :4])
        mu_trunc, _ = model(item_trunc, prev_trunc)

    assert torch.allclose(mu_full[:, :4], mu_trunc)


# -- shared helpers (batch.py / diagnostics.py / losses.py), used by both filters

def test_masked_bce_ignores_masked_out_entries():
    loss_fn = nn.BCEWithLogitsLoss(reduction="none")
    logits = torch.tensor([[0.0, 100.0]])   # second logit would blow up BCE if scored
    target = torch.tensor([[0.0, 0.0]])
    mask = torch.tensor([[True, False]])
    loss = masked_bce(logits, target, mask, loss_fn)
    expected = loss_fn(torch.tensor([0.0]), torch.tensor([0.0]))
    assert torch.allclose(loss, expected)


def test_collapse_diagnostics_detects_a_collapsed_population():
    """Every learner has the IDENTICAL, constant-over-time mu -- the textbook collapse case
    (encoder ignores both who the learner is and what happened to them)."""
    mu = torch.zeros(5, 4, 3)  # [N=5, T=4, L=3], all identical
    valid_mask = torch.ones(5, 4, dtype=torch.bool)
    lengths = torch.full((5,), 4, dtype=torch.long)
    diag = collapse_diagnostics(mu, valid_mask, lengths)
    assert diag["cross_learner_mu_std_mean"] == 0.0
    assert diag["within_learner_mu_std_mean"] == 0.0


def test_collapse_diagnostics_detects_healthy_variation():
    torch.manual_seed(0)
    n, t, latent_dim = 20, 10, 4
    # Each learner gets a distinct random offset (cross-learner variation) plus a per-step
    # cumulative drift (within-learner variation) -- a clearly non-collapsed population.
    offsets = torch.randn(n, 1, latent_dim) * 2.0
    drift = torch.cumsum(torch.randn(n, t, latent_dim) * 0.5, dim=1)
    mu = offsets + drift
    valid_mask = torch.ones(n, t, dtype=torch.bool)
    lengths = torch.full((n,), t, dtype=torch.long)
    diag = collapse_diagnostics(mu, valid_mask, lengths)
    assert diag["cross_learner_mu_std_mean"] > 0.5
    assert diag["within_learner_mu_std_mean"] > 0.1
    per_learner = diag.pop("within_learner_per_learner")
    assert per_learner.shape == (n,)


def test_collapse_diagnostics_percentiles_bracket_the_mean_reading():
    torch.manual_seed(1)
    n, t, latent_dim = 50, 8, 3
    mu = torch.cumsum(torch.randn(n, t, latent_dim), dim=1)
    valid_mask = torch.ones(n, t, dtype=torch.bool)
    lengths = torch.full((n,), t, dtype=torch.long)
    diag = collapse_diagnostics(mu, valid_mask, lengths)
    assert diag["within_learner_mu_std_p5"] <= diag["within_learner_mu_std_median"]
    assert diag["within_learner_mu_std_median"] <= diag["within_learner_mu_std_p95"]
    assert diag["within_learner_mu_std_min_over_learners"] <= diag["within_learner_mu_std_p5"]


def test_per_learner_mean_error_basic():
    correct = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    probs = torch.tensor([[0.9, 0.9, 0.1], [0.5, 0.5, 0.5]])
    mask = torch.tensor([[True, True, True], [True, True, False]])
    err = per_learner_mean_error(mask, correct, probs)
    # learner 0: |1-.9| + |0-.9| + |1-.1| = .1+.9+.9 -> mean .6333
    assert err[0] == pytest.approx((0.1 + 0.9 + 0.9) / 3, abs=1e-5)
    # learner 1: only first 2 cols masked: |0-.5| + |1-.5| = .5+.5 -> mean .5
    assert err[1] == pytest.approx(0.5, abs=1e-5)


def test_per_learner_mean_error_nan_when_no_masked_cells():
    correct = torch.zeros(2, 3)
    probs = torch.zeros(2, 3)
    mask = torch.tensor([[True, False, False], [False, False, False]])
    err = per_learner_mean_error(mask, correct, probs)
    assert not np.isnan(err[0])
    assert np.isnan(err[1])
