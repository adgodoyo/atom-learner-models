"""The planted-world positive control must be VALID before its result means anything: the two zero-information
families must carry exactly zero information, the oracle belief must be the real posterior,
and the belief must be query-independent. The semantic-canary rule applies
to a positive control as much as to a model -- a control that runs but does not mean what
it claims turns a PASS into false assurance, which is worse than no control at all.
"""

from __future__ import annotations

import numpy as np
import torch

from atom.filter.episode import build_prev_feat
from atom.planner.replay import predict_on_targets
from atom.planner.synthetic import (
    OracleFilter,
    OracleHeads,
    assert_query_independent,
    build_grid,
    grid_exact_predictive,
    make_item_bank,
    true_prob,
)


def _bank(seed: int = 0, codec_dim: int = 16):
    rng = np.random.default_rng(seed)
    return make_item_bank(n_diagnostic=8, n_trap=6, n_easy=6, codec_dim=codec_dim,
                          diag_loading=2.0, easy_b=2.2, nuisance_std=0.5, rng=rng)


def test_trap_items_are_exactly_p_half_for_every_learner():
    """The aleatoric trap must be EXACTLY 0.5 for every theta -- maximum predictive
    entropy, zero information. If it drifts off 0.5 the control silently stops testing
    the thing it exists to test (PLANNER_VISION 2's "coin-flip everyone is equally
    unsure about")."""
    codec, family = _bank()
    rng = np.random.default_rng(1)
    for _ in range(50):
        theta = rng.normal(size=2)
        p = true_prob(codec, theta)
        assert np.allclose(p[family == 1], 0.5, atol=0.0), "trap items must be p == 0.5 exactly"


def test_zero_information_families_do_not_vary_with_theta_at_all():
    """Both trap and easy items must be theta-INDEPENDENT: across many learners their
    probability has exactly zero spread. This is what makes 'greedy must avoid them' a
    known-correct answer rather than a preference."""
    codec, family = _bank()
    rng = np.random.default_rng(2)
    probs = np.stack([true_prob(codec, rng.normal(size=2)) for _ in range(200)])
    spread = probs.std(axis=0)
    # Trap items are p == 0.5 EXACTLY (sigmoid(0) is exact in floating point), so their
    # spread is exactly zero. Easy items are sigmoid(b) for a float32 b, constant only to
    # double-precision round-off (~1e-15) -- asserted tightly rather than at atol=0 so the
    # test states the real property instead of an unachievable one.
    assert np.allclose(spread[family == 1], 0.0, atol=0.0)
    assert (spread[family == 2] < 1e-12).all()
    assert (spread[family == 0] > 0.05).all(), "diagnostic items must actually vary with theta"


def test_oracle_belief_is_query_independent_bit_for_bit():
    """The query-independent belief `b_t = f(H_<t)`, checked on the control's own belief exactly as
    it is checked on the shipped filter: perturbing the PROBE item's codec must leave
    the belief at that position bit-identical (max|d mu| == 0.0, not approximately)."""
    codec, _ = _bank()
    grid, log_prior = build_grid(4.5, 41)
    oracle = OracleFilter(grid, log_prior)

    torch.manual_seed(0)
    t = 7
    item_codec = torch.tensor(codec[:t], dtype=torch.float32).unsqueeze(0)
    correct = torch.randint(0, 2, (1, t)).float()
    zeros = torch.zeros_like(correct)
    prev_feat = build_prev_feat(correct, zeros, zeros)

    replacement = torch.tensor(codec[9], dtype=torch.float32).unsqueeze(0)
    delta = assert_query_independent(oracle, item_codec, prev_feat, t - 1, replacement)
    assert delta == 0.0, f"oracle belief leaked the probe item: max|delta| = {delta}"


def test_oracle_posterior_concentrates_on_the_planted_theta():
    """The control's belief must actually track the planted latent -- otherwise a greedy
    win would be measuring something other than 'it learned the person'."""
    rng = np.random.default_rng(3)
    codec, family = make_item_bank(n_diagnostic=60, n_trap=0, n_easy=0, codec_dim=16,
                                   diag_loading=2.0, easy_b=2.2, nuisance_std=0.5, rng=rng)
    grid, log_prior = build_grid(4.5, 61)
    oracle = OracleFilter(grid, log_prior)

    theta = np.array([1.2, -0.8])
    p = true_prob(codec, theta)
    y = (rng.random(codec.shape[0]) < p).astype(np.float32)

    item_codec = torch.tensor(codec, dtype=torch.float32).unsqueeze(0)
    correct = torch.tensor(y).unsqueeze(0)
    zeros = torch.zeros_like(correct)
    with torch.no_grad():
        mu, log_sigma = oracle(item_codec, build_prev_feat(correct, zeros, zeros))

    err_start = np.linalg.norm(mu[0, 0].numpy() - theta)
    err_end = np.linalg.norm(mu[0, -1].numpy() - theta)
    assert err_end < err_start, "posterior mean must move toward the planted theta"
    assert err_end < 0.6, f"posterior mean should be close after 60 diagnostic items, got {err_end}"
    assert log_sigma[0, -1].mean() < log_sigma[0, 0].mean(), "posterior must contract"


def test_oracle_heads_reproduce_the_true_likelihood():
    """Heads and world must decode the SAME (a, b) by value -- if they can disagree, the
    control is scoring one world and simulating another."""
    codec, _ = _bank()
    heads = OracleHeads()
    theta = torch.tensor([[0.7, -1.1]], dtype=torch.float32)
    item_codec = torch.tensor(codec, dtype=torch.float32).unsqueeze(0)  # [1, M, D]
    u = theta.view(1, 1, 1, 2).expand(1, 1, codec.shape[0], 2)
    with torch.no_grad():
        logits, _ = heads(item_codec, u)
    got = torch.sigmoid(logits)[0, 0].numpy()
    want = true_prob(codec, theta[0].numpy())
    assert np.allclose(got, want, atol=1e-5)


def test_gaussian_sampled_predictive_tracks_the_exact_grid_predictive():
    """The ONE approximation in this arm is the diagonal-Gaussian moment match that
    `sample_u` requires. Price it rather than assume it: the K-sample predictive must
    track the exact grid predictive closely."""
    rng = np.random.default_rng(4)
    codec, _ = make_item_bank(n_diagnostic=30, n_trap=4, n_easy=4, codec_dim=16,
                              diag_loading=2.0, easy_b=2.2, nuisance_std=0.5, rng=rng)
    grid, log_prior = build_grid(4.5, 61)
    oracle, heads = OracleFilter(grid, log_prior), OracleHeads()

    theta = np.array([0.9, 0.4])
    p = true_prob(codec, theta)
    y = (rng.random(codec.shape[0]) < p).astype(np.float32)

    revealed = torch.tensor(codec[:14], dtype=torch.float32)
    revealed_y = torch.tensor(y[:14])
    probe = torch.tensor(codec[14:26], dtype=torch.float32)
    zeros = torch.zeros(14)

    torch.manual_seed(0)
    out = predict_on_targets(oracle, heads, revealed, revealed_y, zeros, zeros,
                             probe, k=512, device="cpu")
    exact = grid_exact_predictive(oracle, revealed, revealed_y, probe)
    dev = (out["mean_prob"] - exact.float()).abs().max().item()
    assert dev < 0.02, f"moment-matched predictive deviates from exact posterior by {dev}"


def test_zero_information_candidates_score_near_zero_epig_through_the_shipped_scorer():
    """The end-to-end statement of the control's answer key, through the SHIPPED
    `candidate_evsi_scores`: trap and easy candidates must score ~0, diagnostic ones
    strictly above them. A trap item has MAXIMUM entropy, so a raw-uncertainty score
    ranks it top -- this test is what says EPIG does not."""
    from atom.planner.evsi import _value, candidate_evsi_scores

    rng = np.random.default_rng(5)
    codec, family = make_item_bank(n_diagnostic=6, n_trap=6, n_easy=6, codec_dim=16,
                                   diag_loading=2.0, easy_b=2.2, nuisance_std=0.5, rng=rng)
    diag_bank, _ = make_item_bank(n_diagnostic=10, n_trap=0, n_easy=0, codec_dim=16,
                                  diag_loading=2.0, easy_b=2.2, nuisance_std=0.5,
                                  rng=np.random.default_rng(6))
    grid, log_prior = build_grid(4.5, 41)
    oracle, heads = OracleFilter(grid, log_prior), OracleHeads()

    theta = np.array([0.5, -0.5])
    y = (rng.random(codec.shape[0]) < true_prob(codec, theta)).astype(np.float32)

    prefix = {"item_codec": torch.tensor(diag_bank[:4], dtype=torch.float32),
              "correct": torch.zeros(4), "hint_bin": torch.zeros(4), "saw_answer": torch.zeros(4)}
    cand = {"item_codec": torch.tensor(codec, dtype=torch.float32),
            "correct": torch.tensor(y), "hint_bin": torch.zeros(len(y)),
            "saw_answer": torch.zeros(len(y))}
    targets = torch.tensor(diag_bank[4:], dtype=torch.float32)

    torch.manual_seed(0)
    noise = torch.randn(256, 2)
    out0 = predict_on_targets(oracle, heads, prefix["item_codec"], prefix["correct"],
                              prefix["hint_bin"], prefix["saw_answer"], targets, 256, "cpu", noise=noise)
    scores = candidate_evsi_scores(oracle, heads, prefix, cand, [], list(range(len(y))),
                                   targets, 256, "cpu", _value(out0, "jensen"), noise=noise)
    s = np.array([scores[i] for i in range(len(y))])

    assert s[family == 0].min() > s[family == 1].max(), "every diagnostic must outscore every trap"
    assert s[family == 0].min() > s[family == 2].max(), "every diagnostic must outscore every easy item"
    assert np.abs(s[family != 0]).max() < 1e-3, "zero-information items must score ~0 EPIG"
