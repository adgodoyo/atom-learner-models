"""Unit tests for the replay harness (src/atom/planner/replay.py) -- synthetic
tensors, no real checkpoint/data dependency (fast, CPU).

The causality tests are load-bearing, same discipline as tests/test_s1_filter.py:
predict_on_targets must condition each target's prediction on the shared revealed
history ONLY, never on another target's own (unknown-until-scored) label; and the
windowed split must never let a target index precede a candidate index.
"""

import numpy as np
import torch

from atom.filter.s1 import S1Filter
from atom.heads.s1_heads import FiLMHeads
from atom.planner.replay import (
    predict_on_targets,
    sanity_check_reconstruction,
    split_prefix_candidate_target,
)


def test_split_windows_are_contiguous_and_ordered():
    prefix_idx, candidate_idx, target_idx = split_prefix_candidate_target(
        t_max=20, prefix_k=5, candidate_w=6, n_target=4)
    assert list(prefix_idx) == list(range(5))
    assert list(candidate_idx) == list(range(5, 11))
    assert list(target_idx) == list(range(11, 15))


def test_split_every_target_strictly_after_every_candidate():
    prefix_idx, candidate_idx, target_idx = split_prefix_candidate_target(
        t_max=30, prefix_k=3, candidate_w=8, n_target=10)
    assert candidate_idx.max() < target_idx.min()
    assert prefix_idx.max() < candidate_idx.min()


def test_split_truncates_when_t_max_too_short():
    prefix_idx, candidate_idx, target_idx = split_prefix_candidate_target(
        t_max=7, prefix_k=5, candidate_w=6, n_target=4)
    assert list(prefix_idx) == list(range(5))
    assert list(candidate_idx) == [5, 6]
    assert len(target_idx) == 0


def _tiny_model(seed=0):
    torch.manual_seed(seed)
    codec_dim, latent_dim, hidden = 6, 4, 8
    filter_model = S1Filter(codec_dim, hidden, hidden, latent_dim)
    heads = FiLMHeads(codec_dim, latent_dim, hidden)
    filter_model.eval()
    heads.eval()
    return filter_model, heads, codec_dim


def test_predict_on_targets_shapes_and_fields():
    filter_model, heads, codec_dim = _tiny_model()
    m, n_targets = 3, 5
    revealed_item_codec = torch.randn(m, codec_dim)
    revealed_correct = torch.randint(0, 2, (m,)).float()
    revealed_hint = torch.zeros(m)
    revealed_saw = torch.zeros(m)
    target_item_codec = torch.randn(n_targets, codec_dim)

    out = predict_on_targets(filter_model, heads, revealed_item_codec, revealed_correct,
                             revealed_hint, revealed_saw, target_item_codec, k=8, device="cpu")
    for key in ("mean_prob", "total_entropy", "jensen_gap", "epistemic_std"):
        assert key in out
        assert out[key].shape == (n_targets,)
    assert torch.all((out["mean_prob"] >= 0) & (out["mean_prob"] <= 1))
    assert torch.all(out["total_entropy"] >= 0)
    assert torch.all(out["jensen_gap"] >= 0)  # Jensen's inequality
    assert torch.all(out["epistemic_std"] >= 0)
    # Jensen gap is the epistemic component of total entropy -- must not exceed it.
    assert torch.all(out["jensen_gap"] <= out["total_entropy"] + 1e-5)


def test_predict_on_targets_handles_empty_revealed_set():
    filter_model, heads, codec_dim = _tiny_model()
    empty = torch.zeros(0, codec_dim)
    empty_1d = torch.zeros(0)
    target_item_codec = torch.randn(3, codec_dim)
    out = predict_on_targets(filter_model, heads, empty, empty_1d, empty_1d, empty_1d,
                             target_item_codec, k=8, device="cpu")
    assert out["mean_prob"].shape == (3,)
    assert torch.isfinite(out["mean_prob"]).all()


def test_predict_on_targets_common_random_numbers_are_exactly_reused():
    """The variance-control mechanism (common random numbers): passing the SAME `noise` tensor to two calls
    with identical inputs must give EXACTLY identical outputs (no fresh sampling happens),
    and passing a DIFFERENT noise tensor must generally change the result -- confirms
    `noise` is actually wired into sample_u, not silently ignored."""
    filter_model, heads, codec_dim = _tiny_model(seed=2)
    m = 3
    revealed_item_codec = torch.randn(m, codec_dim)
    revealed_correct = torch.randint(0, 2, (m,)).float()
    revealed_hint = torch.zeros(m)
    revealed_saw = torch.zeros(m)
    target_item_codec = torch.randn(4, codec_dim)
    latent_dim = filter_model.knowledge_filter.mu_head.out_features
    k = 8

    noise_a = torch.randn(k, latent_dim)
    out1 = predict_on_targets(filter_model, heads, revealed_item_codec, revealed_correct,
                              revealed_hint, revealed_saw, target_item_codec, k, "cpu", noise=noise_a)
    out2 = predict_on_targets(filter_model, heads, revealed_item_codec, revealed_correct,
                              revealed_hint, revealed_saw, target_item_codec, k, "cpu", noise=noise_a)
    assert torch.equal(out1["mean_prob"], out2["mean_prob"])

    noise_b = torch.randn(k, latent_dim)
    out3 = predict_on_targets(filter_model, heads, revealed_item_codec, revealed_correct,
                              revealed_hint, revealed_saw, target_item_codec, k, "cpu", noise=noise_b)
    assert not torch.equal(out1["mean_prob"], out3["mean_prob"])


def test_predict_on_targets_deterministic_given_fixed_noise():
    """With k sampling seeded identically, permuting targets must permute predictions
    identically (up to benign batched-float32 noise) -- the cross-target-independence
    invariant: no target's prediction depends on which OTHER targets share its batch."""
    filter_model, heads, codec_dim = _tiny_model()
    m = 2
    revealed_item_codec = torch.randn(m, codec_dim)
    revealed_correct = torch.tensor([1.0, 0.0])
    revealed_hint = torch.zeros(m)
    revealed_saw = torch.zeros(m)
    target_item_codec = torch.randn(4, codec_dim)

    torch.manual_seed(123)
    out_orig = predict_on_targets(filter_model, heads, revealed_item_codec, revealed_correct,
                                  revealed_hint, revealed_saw, target_item_codec, k=1, device="cpu")

    perm = torch.tensor([3, 1, 0, 2])
    torch.manual_seed(123)
    out_perm = predict_on_targets(filter_model, heads, revealed_item_codec, revealed_correct,
                                  revealed_hint, revealed_saw, target_item_codec[perm], k=1, device="cpu")

    # atol is loose (not 1e-6): batched CPU GRU matmuls are not exactly permutation-
    # invariant at the float32 level (non-associative summation order across different
    # batch layouts) -- benign noise, not a leakage bug. A real leak (e.g. target j's
    # label entering target i's prev_feat) would show up as a large, systematic shift,
    # not O(1e-3) jitter; this bound is tight enough to still catch that.
    assert torch.allclose(out_orig["mean_prob"][perm], out_perm["mean_prob"], atol=1e-2)


def test_predict_on_targets_sensitive_to_revealed_history():
    """Sanity: the belief actually uses the revealed evidence -- flipping it changes
    predictions (not a no-op / dead code path)."""
    filter_model, heads, codec_dim = _tiny_model()
    m = 4
    revealed_item_codec = torch.randn(m, codec_dim)
    target_item_codec = torch.randn(2, codec_dim)

    torch.manual_seed(7)
    out_a = predict_on_targets(filter_model, heads, revealed_item_codec,
                               torch.tensor([1.0, 1.0, 1.0, 1.0]), torch.zeros(m),
                               torch.zeros(m), target_item_codec, k=1, device="cpu")

    torch.manual_seed(7)
    out_b = predict_on_targets(filter_model, heads, revealed_item_codec,
                               torch.tensor([0.0, 0.0, 0.0, 0.0]), torch.zeros(m),
                               torch.zeros(m), target_item_codec, k=1, device="cpu")

    assert not torch.allclose(out_a["mean_prob"], out_b["mean_prob"], atol=1e-8)


def test_sanity_check_reconstruction_matches_natural_sequence():
    """The load-bearing correctness check. It compares (mu, log_sigma) directly -- no
    K-sampling, no heads -- so the floor here is TRUE float32 noise, not the ~0.010
    K-sample sampling-noise floor a probability-based comparison would measure instead
    of a real bug. With every
    candidate revealed (no skipping at all), reconstruction must reproduce the natural,
    uninterrupted sequence's own belief -- any divergence beyond float32 noise is a bug."""
    filter_model, heads, codec_dim = _tiny_model(seed=1)
    del heads  # not used by the deterministic check (filter belief only)
    t_total = 12
    item_codec_full = torch.randn(t_total, codec_dim)
    correct_full = torch.randint(0, 2, (t_total,)).float()
    hint_full = torch.zeros(t_total)
    saw_full = torch.zeros(t_total)
    prefix_idx = np.arange(0, 4)
    candidate_idx = np.arange(4, 9)

    diff = sanity_check_reconstruction(filter_model, item_codec_full, correct_full,
                                       hint_full, saw_full, prefix_idx, candidate_idx,
                                       device="cpu")
    assert diff < 1e-4
