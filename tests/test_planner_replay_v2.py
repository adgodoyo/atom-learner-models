"""The S1v2 planner path must be verified as MEANING what it claims, not merely as running:
a query-independent belief that is actually query-independent, a five-channel prev_feat that
actually carries score_missing, and no double-shift. A test
that shapes match is not a test that the belief is the belief.
"""

from __future__ import annotations

import torch

from atom.filter.episode_v2 import build_prev_feat_v2
from atom.filter.s1v2 import S1FilterV2
from atom.heads.s1_heads import FiLMHeads
from atom.planner.replay_v2 import (
    assert_query_independent_v2,
    filter_belief_v2,
    predict_on_targets_v2,
    sigma_trace,
)

CODEC_DIM, LATENT = 45, 32


def _model(seed: int = 0, query_independent: bool = True):
    torch.manual_seed(seed)
    fm = S1FilterV2(CODEC_DIM, 64, 64, LATENT, query_independent=query_independent).eval()
    hd = FiLMHeads(CODEC_DIM, LATENT, 64).eval()
    return fm, hd


def _window(m: int = 9, seed: int = 1):
    torch.manual_seed(seed)
    return (torch.randn(m, CODEC_DIM), torch.randint(0, 2, (m,)).float(),
            torch.zeros(m), torch.zeros(m), torch.zeros(m))


def test_belief_is_query_independent_bit_for_bit():
    """The query-independent belief `b_t = f(H_<t)`, checked through the PLANNER's wiring rather than
    inherited from the filter's own tests: same history, two different candidate questions must give
    an EXACTLY identical belief."""
    fm, _ = _model()
    rev = _window()
    torch.manual_seed(7)
    delta = assert_query_independent_v2(fm, *rev, torch.randn(1, CODEC_DIM),
                                        torch.randn(1, CODEC_DIM), "cpu")
    assert delta == 0.0, f"probe item leaked into the belief: max|delta| = {delta}"


def test_canary_can_fail_on_the_leaky_variant():
    """A canary that cannot fail is not a canary. The SAME assertion against
    `query_independent=False` — v1's defect reproduced inside the v2 class — must fire."""
    fm, _ = _model(query_independent=False)
    rev = _window()
    torch.manual_seed(7)
    delta = assert_query_independent_v2(fm, *rev, torch.randn(1, CODEC_DIM),
                                        torch.randn(1, CODEC_DIM), "cpu")
    assert delta > 1e-4, "the leaky variant must be detected by this canary, but was not"


def test_all_probe_rows_share_one_belief():
    """A consequence of query-independence worth pinning separately: under a query-
    independent filter every probe row's belief is the same object, so Σ-trace and μ are
    properties of the HISTORY, not of which question is being priced."""
    fm, _ = _model()
    rev = _window()
    torch.manual_seed(3)
    mu, ls = filter_belief_v2(fm, *rev, torch.randn(6, CODEC_DIM), "cpu")
    assert (mu - mu[0]).abs().max().item() == 0.0
    assert (ls - ls[0]).abs().max().item() == 0.0


def test_score_missing_channel_actually_reaches_the_belief():
    """The fifth channel must be load-bearing. Flipping a revealed event's `score_missing`
    must move the belief — otherwise the planner is silently feeding the manifest's placeholder
    zeros to the filter as though they were observed failures, which is the exact defect
    the manifest's score_missing flag exists to prevent, relocated one layer down."""
    fm, _ = _model()
    codec, correct, hint, saw, missing = _window()
    correct = torch.ones_like(correct)          # so zeroing prev_correct is visible
    probe = torch.randn(1, CODEC_DIM)
    mu_a, _ = filter_belief_v2(fm, codec, correct, hint, saw, missing, probe, "cpu")
    missing_b = missing.clone()
    missing_b[3] = 1.0
    mu_b, _ = filter_belief_v2(fm, codec, correct, hint, saw, missing_b, probe, "cpu")
    assert (mu_a - mu_b).abs().max().item() > 1e-6, "score_missing never reached the belief"


def test_no_double_shift_against_a_hand_built_forward():
    """`S1FilterV2` shifts internally. If this module ALSO shifted, every belief would be
    off by one event and nothing else would look wrong. Pin it against a hand-built forward
    that passes the unshifted current-item codec, as the class's own contract requires."""
    fm, _ = _model()
    codec, correct, hint, saw, missing = _window(m=5)
    probe = torch.randn(1, CODEC_DIM)
    got_mu, got_ls = filter_belief_v2(fm, codec, correct, hint, saw, missing, probe, "cpu")

    full_codec = torch.cat([codec, probe], dim=0).unsqueeze(0)
    z = torch.zeros(1, 6)
    pf = build_prev_feat_v2(torch.cat([correct, torch.zeros(1)]).unsqueeze(0), z, z, z)
    with torch.no_grad():
        want_mu, want_ls = fm(full_codec, pf)
    assert torch.equal(got_mu[0], want_mu[0, -1])
    assert torch.equal(got_ls[0], want_ls[0, -1])


def test_predict_on_targets_v2_contract_and_sigma_trace():
    """Return contract matches v1's (so the FA planner run can swap paths without re-plumbing), plus the
    μ/log_sigma passthrough the headroom curve needs, and a non-negative Jensen gap."""
    fm, hd = _model()
    rev = _window()
    torch.manual_seed(5)
    targets = torch.randn(4, CODEC_DIM)
    out = predict_on_targets_v2(fm, hd, *rev, targets, k=16, device="cpu")
    for key in ("mean_prob", "total_entropy", "jensen_gap", "epistemic_std"):
        assert out[key].shape == (4,)
    assert (out["jensen_gap"] >= 0).all()
    assert out["mu"].shape == (4, LATENT)
    assert sigma_trace(out["log_sigma"]) > 0


def test_common_random_numbers_make_repeat_calls_bit_identical():
    """CRN is what makes the curve's successive budget steps a paired comparison rather
    than a noisy one; if it stopped working the curve would show sampling noise as belief
    movement."""
    fm, hd = _model()
    rev = _window()
    torch.manual_seed(11)
    targets, noise = torch.randn(3, CODEC_DIM), torch.randn(16, LATENT)
    a = predict_on_targets_v2(fm, hd, *rev, targets, 16, "cpu", noise=noise)
    b = predict_on_targets_v2(fm, hd, *rev, targets, 16, "cpu", noise=noise)
    assert torch.equal(a["mean_prob"], b["mean_prob"])


def test_revealing_more_evidence_changes_the_belief():
    """A positive control for the curve itself: if consuming evidence did not move the
    belief, a flat headroom curve would be an artifact rather than a finding."""
    fm, _ = _model()
    codec, correct, hint, saw, missing = _window(m=9)
    probe = torch.randn(1, CODEC_DIM)
    short = filter_belief_v2(fm, codec[:2], correct[:2], hint[:2], saw[:2], missing[:2], probe, "cpu")
    long = filter_belief_v2(fm, codec, correct, hint, saw, missing, probe, "cpu")
    assert (short[0] - long[0]).abs().max().item() > 1e-6


# --- the predictor injection ----------------------------------------------------------
# `evsi.py` has a `predictor` hook so the SAME policy code -- the code the planted-world
# positive control (`pl_2_positive_control.py`) certified -- can score a v2 belief. These pin that the hook is a true
# no-op on the default path, and that the v2 path reaches the policies intact.

def _windows(n_cand: int = 6, n_prefix: int = 7, n_target: int = 3, seed: int = 2):
    torch.manual_seed(seed)
    prefix = {"item_codec": torch.randn(n_prefix, CODEC_DIM),
              "correct": torch.randint(0, 2, (n_prefix,)).float(),
              "hint_bin": torch.zeros(n_prefix), "saw_answer": torch.zeros(n_prefix),
              "score_missing": torch.zeros(n_prefix)}
    cand = {"item_codec": torch.randn(n_cand, CODEC_DIM),
            "correct": torch.randint(0, 2, (n_cand,)).float(),
            "hint_bin": torch.zeros(n_cand), "saw_answer": torch.zeros(n_cand),
            "score_missing": torch.zeros(n_cand)}
    return prefix, cand, torch.randn(n_target, CODEC_DIM), torch.randint(0, 2, (n_target,)).float()


def test_predictor_injection_is_a_no_op_on_the_default_path():
    """Passing an explicit predictor that just calls v1's `predict_on_targets` must give a
    BIT-IDENTICAL trace to `predictor=None`. If this drifts, the earlier replay numbers and the
    positive control's certification stop applying to the code the FA planner run uses."""
    import numpy as np

    from atom.filter.s1 import S1Filter
    from atom.planner.evsi import run_episode
    from atom.planner.replay import predict_on_targets

    torch.manual_seed(0)
    fm = S1Filter(CODEC_DIM, 32, 32, LATENT).eval()
    hd = FiLMHeads(CODEC_DIM, LATENT, 32).eval()
    prefix, cand, tgt_codec, tgt_y = _windows()
    noise = torch.randn(8, LATENT)

    def v1(filter_model, heads, window, target_codec, k, device, noise):
        return predict_on_targets(filter_model, heads, window["item_codec"], window["correct"],
                                  window["hint_bin"], window["saw_answer"], target_codec, k,
                                  device, noise=noise)

    for policy in ("greedy_evsi", "max_entropy", "max_epistemic"):
        a, _ = run_episode(fm, hd, prefix, cand, tgt_codec, tgt_y, 8, "cpu", policy, 3,
                           rng=np.random.default_rng(0), noise=noise)
        b, _ = run_episode(fm, hd, prefix, cand, tgt_codec, tgt_y, 8, "cpu", policy, 3,
                           rng=np.random.default_rng(0), noise=noise, predictor=v1)
        assert a == b, f"predictor injection changed the {policy} trace"


def test_materialize_carries_score_missing_and_neutralises_it_for_revealed_candidates():
    """The prefix's real score_missing must survive (a placeholder score there is a FACT
    about the history the filter must see), while revealed candidates carry 0 -- the FA planner run
    restricts the pool to label-valid rows, so that zero is true rather than imputed."""
    from atom.planner.evsi import _materialize

    prefix, cand, _, _ = _windows()
    prefix["score_missing"] = torch.tensor([0., 1., 0., 0., 1., 0., 0.])
    w = _materialize(prefix, cand, [2, 0])
    assert "score_missing" in w
    assert torch.equal(w["score_missing"][:7], prefix["score_missing"])
    assert torch.equal(w["score_missing"][7:], torch.zeros(2))


def test_materialize_omits_score_missing_when_the_caller_does_not_supply_it():
    """v1 callers must get exactly the four-key window they always got."""
    from atom.planner.evsi import _materialize

    prefix, cand, _, _ = _windows()
    del prefix["score_missing"], cand["score_missing"]
    assert "score_missing" not in _materialize(prefix, cand, [1])


def test_v2_predictor_drives_the_policies_end_to_end():
    """The real FA planner path: an S1FilterV2 belief scored by the shipped policy code."""
    import numpy as np

    from atom.planner.evsi import run_episode

    fm, hd = _model()
    prefix, cand, tgt_codec, tgt_y = _windows()
    noise = torch.randn(8, LATENT)

    def v2(filter_model, heads, window, target_codec, k, device, noise):
        return predict_on_targets_v2(filter_model, heads, window["item_codec"], window["correct"],
                                     window["hint_bin"], window["saw_answer"],
                                     window["score_missing"], target_codec, k, device, noise=noise)

    trace, first = run_episode(fm, hd, prefix, cand, tgt_codec, tgt_y, 8, "cpu", "greedy_evsi", 3,
                               rng=np.random.default_rng(0), noise=noise, predictor=v2)
    assert len(trace) == 4 and first is not None
    assert all(np.isfinite(s["logloss"]) for s in trace)
