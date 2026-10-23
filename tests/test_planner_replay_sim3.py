"""Tests for the planner's FA-SIM3 replay path. FA-SIM3 is the planner's substrate on the
mathematics corpus, so its wiring carries every FA planner result. These are canaries -- each one can FAIL,
and each pins a property that would otherwise be true-looking and wrong: a belief that is
query-independent, a fast path that is genuinely the same belief, a five-channel prev_feat
that carries score_missing, no double shift, and an answer space that is asymmetric ON
PURPOSE (categorical targets, binary candidates) rather than by accident.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from atom.filter.episode_v2 import build_prev_feat_v2
from atom.filter.item_encoder_mlp import S1FilterV2ItemMLP
from atom.heads.fa_hierarchical_heads import OutcomeHead
from atom.planner.replay_sim3 import (
    UNAIDED_CORRECT,
    _belief_with_probe,
    _categorical_entropy,
    assert_query_independent_sim3,
    categorical_score_step,
    filter_belief_sim3,
    predict_on_targets_sim3,
    sigma_trace,
)

CODEC_DIM, LATENT, HID, N_CLS = 64, 32, 48, 5


def _stack(seed: int = 0, query_independent: bool = True):
    torch.manual_seed(seed)
    fm = S1FilterV2ItemMLP(CODEC_DIM, HID, HID, LATENT, query_independent=query_independent).eval()
    hd = OutcomeHead(CODEC_DIM, LATENT, HID, n_classes=N_CLS).eval()
    return fm, hd


def _hist(m: int = 9, seed: int = 1):
    torch.manual_seed(seed)
    return (torch.randn(m, CODEC_DIM), torch.randint(0, 2, (m,)).float(),
            torch.zeros(m), torch.zeros(m), torch.zeros(m))


# --- belief properties -------------------------------------------------------------------

def test_belief_is_query_independent_bit_for_bit():
    fm, _ = _stack()
    torch.manual_seed(7)
    delta = assert_query_independent_sim3(fm, *_hist(), torch.randn(1, CODEC_DIM),
                                          torch.randn(1, CODEC_DIM), "cpu")
    assert delta == 0.0, f"the probe item leaked into the belief: max|delta| = {delta}"


def test_peek_canary_can_fail_on_the_leaky_variant():
    """A canary that cannot fail is not a canary."""
    fm, _ = _stack(query_independent=False)
    torch.manual_seed(7)
    delta = assert_query_independent_sim3(fm, *_hist(), torch.randn(1, CODEC_DIM),
                                          torch.randn(1, CODEC_DIM), "cpu")
    assert delta > 1e-4, "the leaky variant must be detected by this canary, but was not"


def test_shared_belief_fast_path_matches_per_probe_to_float32_tolerance():
    """`filter_belief_sim3` runs the GRU ONCE and expands, which is only valid because the
    belief is query-independent. That shortcut is worth ~15x on the planner's inner loop, so it is
    pinned against the honest per-probe construction rather than trusted. Different batch
    sizes may select different float32 kernels, so the comparison uses a float32-scale bound."""
    fm, _ = _stack()
    hist = _hist()
    torch.manual_seed(3)
    probes = torch.randn(6, CODEC_DIM)
    fast_mu, fast_ls = filter_belief_sim3(fm, *hist, 6, "cpu")
    slow_mu, slow_ls = _belief_with_probe(fm, *hist, probes, "cpu")
    torch.testing.assert_close(fast_mu, slow_mu, atol=1e-7, rtol=0)
    torch.testing.assert_close(fast_ls, slow_ls, atol=1e-7, rtol=0)


def test_score_missing_channel_actually_reaches_the_belief():
    """The manifest's flag-don't-impute discipline, one layer down: a placeholder score in the
    history is a FACT the filter must see, not a silent zero."""
    fm, _ = _stack()
    codec, correct, hint, saw, missing = _hist()
    correct = torch.ones_like(correct)
    mu_a, _ = filter_belief_sim3(fm, codec, correct, hint, saw, missing, 1, "cpu")
    missing_b = missing.clone()
    missing_b[3] = 1.0
    mu_b, _ = filter_belief_sim3(fm, codec, correct, hint, saw, missing_b, 1, "cpu")
    assert (mu_a - mu_b).abs().max().item() > 1e-6, "score_missing never reached the belief"


def test_no_double_shift_against_a_hand_built_forward():
    """`S1FilterV2ItemMLP` shifts internally. If this module also shifted, every belief would
    be off by one event and nothing else would look wrong."""
    fm, _ = _stack()
    codec, correct, hint, saw, missing = _hist(m=5)
    got_mu, got_ls = filter_belief_sim3(fm, codec, correct, hint, saw, missing, 1, "cpu")
    full_codec = torch.cat([codec, torch.zeros(1, CODEC_DIM)], dim=0).unsqueeze(0)
    z = torch.zeros(1, 6)
    pf = build_prev_feat_v2(torch.cat([correct, torch.zeros(1)]).unsqueeze(0), z, z, z)
    with torch.no_grad():
        want_mu, want_ls = fm(full_codec, pf)
    assert torch.equal(got_mu[0], want_mu[0, -1])
    assert torch.equal(got_ls[0], want_ls[0, -1])


def test_revealing_more_evidence_changes_the_belief():
    fm, _ = _stack()
    codec, correct, hint, saw, missing = _hist(m=9)
    short = filter_belief_sim3(fm, codec[:2], correct[:2], hint[:2], saw[:2], missing[:2], 1, "cpu")
    long = filter_belief_sim3(fm, codec, correct, hint, saw, missing, 1, "cpu")
    assert (short[0] - long[0]).abs().max().item() > 1e-6


# --- the answer space --------------------------------------------------------------------

def _out(answer_space: str, k: int = 16, n_t: int = 4, seed: int = 5):
    fm, hd = _stack()
    torch.manual_seed(seed)
    targets = torch.randn(n_t, CODEC_DIM)
    return predict_on_targets_sim3(fm, hd, *_hist(), targets, k, "cpu",
                                   answer_space=answer_space), targets


def test_categorical_arm_exposes_target_side_keys_and_binary_arm_does_not():
    """The asymmetry is the whole design (reports/pl_3c_answer_space_prereg.md). If arm B
    ever grew `value_*` keys it would silently start pricing targets categorically while
    claiming to be the binary control."""
    cat, _ = _out("categorical")
    binr, _ = _out("binary")
    for key in ("mean_probs", "value_total_entropy", "value_jensen_gap"):
        assert key in cat
        assert key not in binr, f"arm B must not carry {key}"
    for key in ("mean_prob", "total_entropy", "jensen_gap", "epistemic_std", "mu", "log_sigma"):
        assert key in cat and key in binr


def test_candidate_side_quantities_are_identical_across_arms():
    """`p_e` and both max-uncertainty baselines read the candidate-side keys. Those must NOT
    move when the target answer space changes, or arm A vs arm B would confound the answer
    space with the selection rule."""
    cat, _ = _out("categorical")
    binr, _ = _out("binary")
    for key in ("mean_prob", "total_entropy", "jensen_gap", "epistemic_std"):
        assert torch.equal(cat[key], binr[key]), f"{key} changed with the answer space"


def test_binary_marginal_is_the_unaided_correct_class():
    """The binary channel must be P(UNAIDED_CORRECT) -- class 0 -- and not some other
    pooling. This is what makes "revealing a candidate conveys correctness only" true."""
    cat, _ = _out("categorical")
    assert torch.allclose(cat["mean_prob"], cat["mean_probs"][:, UNAIDED_CORRECT], atol=1e-6)


def test_categorical_predictive_is_a_distribution_and_its_gap_is_non_negative():
    cat, _ = _out("categorical")
    assert torch.allclose(cat["mean_probs"].sum(dim=-1), torch.ones(4), atol=1e-5)
    assert (cat["value_jensen_gap"] >= 0).all()
    assert (cat["value_total_entropy"] >= cat["total_entropy"] - 1e-6).all(), (
        "a 5-class entropy cannot be below the entropy of a binary coarsening of it")


def test_categorical_entropy_matches_a_hand_computation():
    p = torch.tensor([[0.5, 0.25, 0.125, 0.0625, 0.0625]])
    want = float(-(p * p.log()).sum())
    assert abs(float(_categorical_entropy(p)) - want) < 1e-9


def test_uniform_five_class_gives_log5_and_a_certain_one_gives_zero():
    assert abs(float(_categorical_entropy(torch.full((1, 5), 0.2))) - np.log(5)) < 1e-7
    assert float(_categorical_entropy(torch.tensor([[1.0, 0, 0, 0, 0]]))) < 1e-9


def test_value_reads_categorical_under_arm_a_and_binary_under_arm_b():
    """The `value_*` preference in `evsi._value` is what wires the asymmetry into V(b)."""
    from atom.planner.evsi import _value
    cat, _ = _out("categorical")
    binr, _ = _out("binary")
    assert _value(cat, "jensen") == -float(cat["value_jensen_gap"].mean())
    assert _value(cat, "total") == -float(cat["value_total_entropy"].mean())
    assert _value(binr, "jensen") == -float(binr["jensen_gap"].mean())
    assert _value(binr, "total") == -float(binr["total_entropy"].mean())


def test_value_is_unchanged_for_a_predictor_without_value_keys():
    """The additive hook must be a strict no-op for every caller whose predictor emits no value keys."""
    from atom.planner.evsi import _value
    out = {"total_entropy": torch.tensor([0.5, 0.7]), "jensen_gap": torch.tensor([0.1, 0.2])}
    assert _value(out, "total") == pytest.approx(-0.6, abs=1e-7)
    assert abs(_value(out, "jensen") - (-0.15000000596046448)) < 1e-6


def test_categorical_score_step_matches_a_hand_computed_logloss():
    out = {"mean_probs": torch.tensor([[0.6, 0.1, 0.1, 0.1, 0.1],
                                       [0.1, 0.2, 0.3, 0.2, 0.2]]),
           "value_jensen_gap": torch.tensor([0.0, 0.0])}
    y = torch.tensor([0, 2])
    got = categorical_score_step(out, y, n_asked=3)
    want = float(-(torch.tensor([0.6, 0.3]).log().mean()))
    assert abs(got["logloss"] - want) < 1e-6
    assert got["accuracy"] == 1.0          # argmax is 0 and 2
    assert got["n_asked"] == 3


def test_common_random_numbers_make_repeat_calls_bit_identical():
    fm, hd = _stack()
    torch.manual_seed(11)
    targets, noise = torch.randn(3, CODEC_DIM), torch.randn(16, LATENT)
    a = predict_on_targets_sim3(fm, hd, *_hist(), targets, 16, "cpu", noise=noise)
    b = predict_on_targets_sim3(fm, hd, *_hist(), targets, 16, "cpu", noise=noise)
    assert torch.equal(a["mean_probs"], b["mean_probs"])


def test_sigma_trace_is_positive():
    _, _ = _stack()
    fm, _ = _stack()
    _, ls = filter_belief_sim3(fm, *_hist(), 1, "cpu")
    assert sigma_trace(ls) > 0


def test_unknown_answer_space_raises():
    fm, hd = _stack()
    try:
        predict_on_targets_sim3(fm, hd, *_hist(), torch.randn(2, CODEC_DIM), 8, "cpu",
                                answer_space="five_class")
    except ValueError as e:
        assert "answer_space" in str(e)
    else:
        raise AssertionError("an unknown answer space must raise, not default silently")


# --- end to end through the SHIPPED policy code -------------------------------------------

def _windows(n_cand: int = 6, n_prefix: int = 7, n_target: int = 3, seed: int = 2):
    torch.manual_seed(seed)
    keys = ("hint_bin", "saw_answer", "score_missing")
    prefix = {"item_codec": torch.randn(n_prefix, CODEC_DIM),
              "correct": torch.randint(0, 2, (n_prefix,)).float(),
              **{k: torch.zeros(n_prefix) for k in keys}}
    cand = {"item_codec": torch.randn(n_cand, CODEC_DIM),
            "correct": torch.randint(0, 2, (n_cand,)).float(),
            **{k: torch.zeros(n_cand) for k in keys}}
    return prefix, cand, torch.randn(n_target, CODEC_DIM)


def _predictor(answer_space: str):
    def f(filter_model, head, window, target_codec, k, device, noise):
        return predict_on_targets_sim3(filter_model, head, window["item_codec"], window["correct"],
                                       window["hint_bin"], window["saw_answer"],
                                       window["score_missing"], target_codec, k, device,
                                       noise=noise, answer_space=answer_space)
    return f


def test_both_arms_drive_the_shipped_policies_end_to_end():
    from atom.planner.evsi import run_episode
    fm, hd = _stack()
    prefix, cand, tgt_codec = _windows()
    tgt_class = torch.tensor([0, 2, 4])
    noise = torch.randn(8, LATENT)
    for policy in ("greedy_evsi", "max_entropy", "max_epistemic"):
        trace, first = run_episode(fm, hd, prefix, cand, tgt_codec, tgt_class, 8, "cpu", policy, 3,
                                   rng=np.random.default_rng(0), noise=noise,
                                   predictor=_predictor("categorical"),
                                   scorer=categorical_score_step)
        assert len(trace) == 4
        assert all(np.isfinite(s["logloss"]) and np.isfinite(s["V"]) for s in trace)
        if policy == "greedy_evsi":
            assert first is not None


def test_the_help_channels_of_a_candidate_can_never_reach_any_score():
    """The replay leak canary, re-fired on this substrate. Three of the five classes ARE the
    help-seeking channel, so if a candidate's logged hint/saw_answer could move a score, the
    5-class arm would re-open the help-channel leak in a new form -- the exact failure the
    asymmetric answer space (binary candidates) exists to prevent."""
    from atom.planner.evsi import run_episode
    fm, hd = _stack()
    prefix, cand, tgt_codec = _windows()
    tgt_class = torch.tensor([0, 2, 4])
    noise = torch.randn(8, LATENT)
    kw = dict(rng=np.random.default_rng(0), noise=noise, predictor=_predictor("categorical"),
              scorer=categorical_score_step)
    base, _ = run_episode(fm, hd, prefix, cand, tgt_codec, tgt_class, 8, "cpu", "greedy_evsi", 3, **kw)
    poisoned = {**cand, "hint_bin": torch.ones_like(cand["hint_bin"]),
                "saw_answer": torch.ones_like(cand["saw_answer"])}
    after, _ = run_episode(fm, hd, prefix, poisoned, tgt_codec, tgt_class, 8, "cpu", "greedy_evsi", 3,
                           **{**kw, "rng": np.random.default_rng(0)})
    assert base == after, "a candidate's logged help channels moved a score -- help-channel leak reopened"
