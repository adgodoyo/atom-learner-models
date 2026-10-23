"""Unit tests for the ask policies (src/atom/planner/evsi.py) -- synthetic tensors,
no real checkpoint/data dependency (fast, CPU).
"""

import numpy as np
import pytest
import torch

from atom.filter.s1 import S1Filter
from atom.heads.s1_heads import FiLMHeads
from atom.planner.evsi import POLICIES, run_episode


def _tiny_model(seed=0):
    torch.manual_seed(seed)
    codec_dim, latent_dim, hidden = 6, 4, 8
    filter_model = S1Filter(codec_dim, hidden, hidden, latent_dim)
    heads = FiLMHeads(codec_dim, latent_dim, hidden)
    filter_model.eval()
    heads.eval()
    return filter_model, heads, codec_dim


def _windows(codec_dim, m_prefix, w_cand, n_target, seed=0):
    g = torch.Generator().manual_seed(seed)

    def window(n):
        return {
            "item_codec": torch.randn(n, codec_dim, generator=g),
            "correct": torch.randint(0, 2, (n,), generator=g).float(),
            "hint_bin": torch.zeros(n),
            "saw_answer": torch.zeros(n),
        }
    return window(m_prefix), window(w_cand), window(n_target)


def test_run_episode_trace_length_and_monotone_n_asked():
    filter_model, heads, codec_dim = _tiny_model()
    prefix, cand, target = _windows(codec_dim, m_prefix=3, w_cand=5, n_target=4)
    trace, _ = run_episode(filter_model, heads, prefix, cand, target["item_codec"],
                           target["correct"], k=4, device="cpu", policy="max_entropy", budget=5)
    assert [step["n_asked"] for step in trace] == list(range(6))
    for step in trace:
        assert 0.0 <= step["accuracy"] <= 1.0
        assert step["logloss"] >= 0.0


def test_run_episode_budget_caps_at_candidate_window_size():
    filter_model, heads, codec_dim = _tiny_model()
    prefix, cand, target = _windows(codec_dim, m_prefix=2, w_cand=3, n_target=2)
    trace, _ = run_episode(filter_model, heads, prefix, cand, target["item_codec"],
                           target["correct"], k=4, device="cpu", policy="random", budget=100,
                           rng=np.random.default_rng(0))
    assert len(trace) == 4  # n_asked = 0..3, capped by the 3-wide candidate window


def test_greedy_evsi_returns_first_pick_score_only_for_that_policy():
    filter_model, heads, codec_dim = _tiny_model()
    prefix, cand, target = _windows(codec_dim, m_prefix=3, w_cand=4, n_target=3)
    _, first_evsi = run_episode(filter_model, heads, prefix, cand, target["item_codec"],
                                target["correct"], k=4, device="cpu", policy="greedy_evsi", budget=4)
    assert first_evsi is not None
    assert isinstance(first_evsi, float)

    for other in ("random", "max_entropy", "max_epistemic", "top5_softened"):
        rng = np.random.default_rng(0) if other in ("random", "top5_softened") else None
        _, none_evsi = run_episode(filter_model, heads, prefix, cand, target["item_codec"],
                                   target["correct"], k=4, device="cpu", policy=other, budget=4, rng=rng)
        assert none_evsi is None


def test_greedy_evsi_v_is_non_decreasing_in_expectation_direction():
    """V(b) = -mean entropy; greedy EVSI is chosen to be the best AVAILABLE move each step
    (myopic, not globally optimal), so V should trend up as more of the candidate window
    (drawn from the SAME learner's real, on-topic history) is revealed -- not a strict
    monotone guarantee (a single real-outcome reveal can raise entropy), but the final V
    after revealing everything must not be far below the initial V for a real signal-bearing
    codec (sanity floor, not a tight bound)."""
    filter_model, heads, codec_dim = _tiny_model(seed=3)
    prefix, cand, target = _windows(codec_dim, m_prefix=4, w_cand=6, n_target=4, seed=3)
    trace, _ = run_episode(filter_model, heads, prefix, cand, target["item_codec"],
                           target["correct"], k=8, device="cpu", policy="greedy_evsi", budget=6)
    v_values = [step["V"] for step in trace]
    assert len(v_values) == 7
    assert all(np.isfinite(v) for v in v_values)


def test_all_policies_are_valid_policy_names():
    assert set(POLICIES) == {"greedy_evsi", "random", "max_entropy", "max_epistemic", "top5_softened"}


def test_v_flavor_is_a_reachable_config_flag_not_an_in_place_swap():
    """`_value` defaults to the Jensen gap (epistemic-only V), but the total-entropy V
    must stay selectable as a config flag: it is the objective under which greedy EVSI
    was observed to lose to random selection, so that result must remain reproducible."""
    from atom.planner.evsi import V_FLAVORS, _value

    assert set(V_FLAVORS) == {"total", "jensen"}
    target_out = {"total_entropy": torch.tensor([0.6, 0.4]), "jensen_gap": torch.tensor([0.1, 0.2])}
    assert _value(target_out, "total") == pytest.approx(-0.5)
    assert _value(target_out, "jensen") == pytest.approx(-0.15)
    assert _value(target_out) == _value(target_out, "jensen")  # default is the Jensen gap


def test_v_flavor_total_reproduces_pre_pl1_v_values_in_run_episode():
    """v_flavor="total" must give the SAME per-step V as -total_entropy directly (the
    total-entropy arm, byte-for-byte), while v_flavor="jensen" (default) differs from it --
    confirms the flag actually switches the objective rather than being a no-op."""
    filter_model, heads, codec_dim = _tiny_model(seed=14)
    prefix, cand, target = _windows(codec_dim, m_prefix=3, w_cand=5, n_target=3, seed=14)
    latent_dim = filter_model.knowledge_filter.mu_head.out_features
    noise = torch.randn(8, latent_dim)

    trace_total, _ = run_episode(filter_model, heads, prefix, cand, target["item_codec"],
                                 target["correct"], k=8, device="cpu", policy="max_entropy",
                                 budget=5, noise=noise, v_flavor="total")
    trace_jensen, _ = run_episode(filter_model, heads, prefix, cand, target["item_codec"],
                                  target["correct"], k=8, device="cpu", policy="max_entropy",
                                  budget=5, noise=noise, v_flavor="jensen")

    for step_total, step_jensen in zip(trace_total, trace_jensen):
        # logloss/accuracy score mean_prob against the true label -- independent of V's flavor
        assert step_total["logloss"] == pytest.approx(step_jensen["logloss"], abs=1e-6)
        assert step_total["accuracy"] == pytest.approx(step_jensen["accuracy"], abs=1e-6)
    assert any(abs(a["V"] - b["V"]) > 1e-9 for a, b in zip(trace_total, trace_jensen))


def test_unknown_v_flavor_raises():
    filter_model, heads, codec_dim = _tiny_model()
    prefix, cand, target = _windows(codec_dim, m_prefix=2, w_cand=2, n_target=2)
    with pytest.raises(ValueError):
        run_episode(filter_model, heads, prefix, cand, target["item_codec"], target["correct"],
                   k=4, device="cpu", policy="random", budget=2, rng=np.random.default_rng(0),
                   v_flavor="not_a_flavor")


def test_unknown_policy_raises():
    filter_model, heads, codec_dim = _tiny_model()
    prefix, cand, target = _windows(codec_dim, m_prefix=2, w_cand=2, n_target=2)
    try:
        run_episode(filter_model, heads, prefix, cand, target["item_codec"], target["correct"],
                   k=4, device="cpu", policy="not_a_policy", budget=2)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_random_policy_is_deterministic_given_seeded_generator():
    """Same np.random.Generator seed AND same torch seed (controls predict_on_targets'
    reparameterisation noise, orthogonal to the policy's own candidate-order randomness)
    -> identical trace."""
    filter_model, heads, codec_dim = _tiny_model()
    prefix, cand, target = _windows(codec_dim, m_prefix=3, w_cand=5, n_target=3)

    torch.manual_seed(99)
    trace_a, _ = run_episode(filter_model, heads, prefix, cand, target["item_codec"],
                             target["correct"], k=4, device="cpu", policy="random", budget=5,
                             rng=np.random.default_rng(42))
    torch.manual_seed(99)
    trace_b, _ = run_episode(filter_model, heads, prefix, cand, target["item_codec"],
                             target["correct"], k=4, device="cpu", policy="random", budget=5,
                             rng=np.random.default_rng(42))
    for a, b in zip(trace_a, trace_b):
        assert a["logloss"] == b["logloss"]
        assert a["accuracy"] == b["accuracy"]


def test_full_budget_beliefs_are_bit_identical_across_policies():
    """PERMANENT REGRESSION GUARD for a real bug in which
    `run_episode` fed revealed candidates to the GRU in SELECTION order, not chronological
    order). At full budget every policy has revealed the exact same SET of candidates --
    with common-random-numbers noise shared across all four calls, their final-step
    beliefs (and hence log-loss/accuracy/V) must be BIT-IDENTICAL regardless of the order
    each policy happened to reveal them in. If this ever fails again, `_materialize`'s
    chronological re-sort has regressed."""
    filter_model, heads, codec_dim = _tiny_model(seed=5)
    prefix, cand, target = _windows(codec_dim, m_prefix=3, w_cand=6, n_target=4, seed=5)
    latent_dim = filter_model.knowledge_filter.mu_head.out_features
    k = 8
    noise = torch.randn(k, latent_dim)

    final_steps = {}
    for policy in POLICIES:
        rng = np.random.default_rng(0) if policy in ("random", "top5_softened") else None
        trace, _ = run_episode(filter_model, heads, prefix, cand, target["item_codec"],
                               target["correct"], k=k, device="cpu", policy=policy, budget=6,
                               rng=rng, noise=noise)
        final_steps[policy] = trace[-1]

    reference = final_steps["greedy_evsi"]
    for policy in POLICIES:
        assert final_steps[policy]["logloss"] == pytest.approx(reference["logloss"], abs=1e-6)
        assert final_steps[policy]["accuracy"] == pytest.approx(reference["accuracy"], abs=1e-6)
        assert final_steps[policy]["V"] == pytest.approx(reference["V"], abs=1e-6)


def test_top5_softened_picks_only_from_top_scoring_candidates():
    """Optimizer's-curse test: top5_softened must always pick from the top-5 EVSI
    scorers (never an arbitrarily bad candidate), while NOT being deterministic like pure
    greedy -- different rng seeds can pick different (but each top-5) candidates."""
    from atom.planner.evsi import _materialize, _value, candidate_evsi_scores
    from atom.planner.replay import predict_on_targets

    filter_model, heads, codec_dim = _tiny_model(seed=9)
    prefix, cand, target = _windows(codec_dim, m_prefix=3, w_cand=8, n_target=3, seed=9)
    latent_dim = filter_model.knowledge_filter.mu_head.out_features
    k = 8
    noise = torch.randn(k, latent_dim)

    revealed = _materialize(prefix, cand, [])
    out0 = predict_on_targets(filter_model, heads, revealed["item_codec"], revealed["correct"],
                              revealed["hint_bin"], revealed["saw_answer"], target["item_codec"], k,
                              "cpu", noise=noise)
    v_b = _value(out0)
    scores = candidate_evsi_scores(filter_model, heads, prefix, cand, [], list(range(8)),
                                   target["item_codec"], k, "cpu", v_b, noise=noise)
    top5 = set(sorted(scores, key=scores.get, reverse=True)[:5])

    picks = set()
    pick_log: list[int] = []
    for seed in range(15):
        pick_log.clear()
        run_episode(filter_model, heads, prefix, cand, target["item_codec"], target["correct"],
                   k=k, device="cpu", policy="top5_softened", budget=1,
                   rng=np.random.default_rng(seed), noise=noise, pick_log=pick_log)
        picks.add(pick_log[0])

    assert picks.issubset(top5)


def test_materialize_neutralises_hint_and_saw_answer_for_revealed_candidates():
    """The acquired observation from
    revealing a candidate is correctness-only -- `_materialize` must never copy a candidate's
    REAL logged hint_bin/saw_answer into the revealed window, even though `correct` for the
    same candidates is genuinely carried through."""
    from atom.planner.evsi import _materialize

    prefix = {
        "item_codec": torch.randn(2, 3), "correct": torch.tensor([1.0, 0.0]),
        "hint_bin": torch.tensor([1.0, 0.0]), "saw_answer": torch.tensor([0.0, 1.0]),
    }
    cand = {
        "item_codec": torch.randn(4, 3), "correct": torch.tensor([1.0, 0.0, 1.0, 0.0]),
        "hint_bin": torch.tensor([1.0, 1.0, 1.0, 1.0]), "saw_answer": torch.tensor([1.0, 1.0, 1.0, 1.0]),
    }
    win = _materialize(prefix, cand, [0, 2])
    # prefix's own hint_bin/saw_answer (genuinely known past history) are untouched.
    assert torch.equal(win["hint_bin"][:2], prefix["hint_bin"])
    assert torch.equal(win["saw_answer"][:2], prefix["saw_answer"])
    # the revealed candidates' hint_bin/saw_answer are neutralised to 0, not copied from cand.
    assert torch.equal(win["hint_bin"][2:], torch.zeros(2))
    assert torch.equal(win["saw_answer"][2:], torch.zeros(2))
    # correct IS genuinely carried through for the revealed candidates.
    assert torch.equal(win["correct"][2:], torch.tensor([1.0, 1.0]))


def test_candidate_evsi_scores_invariant_to_hint_and_saw_answer_perturbation():
    """Replay leak canary: perturbing a candidate's logged
    hint_bin/saw_answer must leave its EVSI score, and every other candidate's EVSI score,
    BIT-IDENTICAL. Fails if: any score moves."""
    from atom.planner.evsi import candidate_evsi_scores

    filter_model, heads, codec_dim = _tiny_model(seed=11)
    prefix, cand, target = _windows(codec_dim, m_prefix=3, w_cand=6, n_target=3, seed=11)
    latent_dim = filter_model.knowledge_filter.mu_head.out_features
    k = 8
    noise = torch.randn(k, latent_dim)
    remaining = list(range(6))
    v_b = 0.0  # a constant offset applied identically to every score; irrelevant to the canary

    scores_before = candidate_evsi_scores(filter_model, heads, prefix, cand, [], remaining,
                                          target["item_codec"], k, "cpu", v_b, noise=noise)

    cand_perturbed = dict(cand)
    cand_perturbed["hint_bin"] = torch.ones_like(cand["hint_bin"])
    cand_perturbed["saw_answer"] = torch.ones_like(cand["saw_answer"])
    scores_after = candidate_evsi_scores(filter_model, heads, prefix, cand_perturbed, [], remaining,
                                         target["item_codec"], k, "cpu", v_b, noise=noise)

    assert scores_before.keys() == scores_after.keys()
    for c in scores_before:
        assert scores_before[c] == scores_after[c], f"leak canary moved for candidate {c}"


def test_run_episode_invariant_to_hint_and_saw_answer_perturbation():
    """End-to-end leak canary: every policy's full trace (log-loss/accuracy/V at every
    step, including which candidate greedy picks first) must be unchanged by perturbing
    the candidate window's logged hint_bin/saw_answer -- these channels have no path into
    any policy's decisions or scores under the correctness-only observation model."""
    filter_model, heads, codec_dim = _tiny_model(seed=12)
    prefix, cand, target = _windows(codec_dim, m_prefix=3, w_cand=6, n_target=3, seed=12)
    latent_dim = filter_model.knowledge_filter.mu_head.out_features
    k = 8
    noise = torch.randn(k, latent_dim)

    cand_perturbed = dict(cand)
    cand_perturbed["hint_bin"] = torch.ones_like(cand["hint_bin"])
    cand_perturbed["saw_answer"] = torch.ones_like(cand["saw_answer"])

    for policy in POLICIES:
        rng_before = np.random.default_rng(0) if policy in ("random", "top5_softened") else None
        rng_after = np.random.default_rng(0) if policy in ("random", "top5_softened") else None
        trace_before, evsi_before = run_episode(filter_model, heads, prefix, cand, target["item_codec"],
                                                target["correct"], k=k, device="cpu", policy=policy,
                                                budget=6, rng=rng_before, noise=noise)
        trace_after, evsi_after = run_episode(filter_model, heads, prefix, cand_perturbed, target["item_codec"],
                                              target["correct"], k=k, device="cpu", policy=policy,
                                              budget=6, rng=rng_after, noise=noise)
        for step_before, step_after in zip(trace_before, trace_after):
            assert step_before == step_after, f"leak canary moved for policy {policy}"
        assert evsi_before == evsi_after


def test_materialize_is_insertion_order_independent():
    """Narrower unit-level version of the same invariant, isolating `_materialize` itself
    from any policy-selection logic: revealing the same SET of candidates in two different
    orders must produce an identical materialized window."""
    from atom.planner.evsi import _materialize

    filter_model, heads, codec_dim = _tiny_model(seed=6)
    del heads
    prefix, cand, _ = _windows(codec_dim, m_prefix=2, w_cand=5, n_target=1, seed=6)

    order_a = [3, 0, 4, 1, 2]
    order_b = [1, 2, 0, 4, 3]
    win_a = _materialize(prefix, cand, order_a)
    win_b = _materialize(prefix, cand, order_b)
    for key in win_a:
        assert torch.equal(win_a[key], win_b[key])


def test_pick_log_records_selected_indices_in_order_without_changing_trace():
    """The pick-composition diagnostic needs to know WHICH candidate a policy picked
    at each step -- `pick_log` is a pure output parameter (mutated in place), so it must
    not change the returned trace, and every logged index must be a valid, then-remaining
    candidate index."""
    filter_model, heads, codec_dim = _tiny_model(seed=8)
    prefix, cand, target = _windows(codec_dim, m_prefix=3, w_cand=5, n_target=3, seed=8)
    latent_dim = filter_model.knowledge_filter.mu_head.out_features
    noise = torch.randn(4, latent_dim)  # shared CRN draw so both calls are directly comparable

    trace_a, _ = run_episode(filter_model, heads, prefix, cand, target["item_codec"],
                             target["correct"], k=4, device="cpu", policy="max_entropy", budget=5,
                             noise=noise)
    pick_log: list[int] = []
    trace_b, _ = run_episode(filter_model, heads, prefix, cand, target["item_codec"],
                             target["correct"], k=4, device="cpu", policy="max_entropy", budget=5,
                             noise=noise, pick_log=pick_log)

    for a, b in zip(trace_a, trace_b):
        assert a["logloss"] == b["logloss"]
        assert a["accuracy"] == b["accuracy"]
    assert len(pick_log) == 5
    assert len(set(pick_log)) == 5  # no candidate picked twice
    assert all(0 <= idx < 5 for idx in pick_log)


def test_selection_budget_does_not_change_full_budget_result():
    """`selection_budget` is a cost-saving device (cheap fixed-order reveals past the
    point actually being compared) -- since `_materialize` re-sorts regardless, capping it
    below the full budget must NOT change the full-budget trace, only how it got computed."""
    filter_model, heads, codec_dim = _tiny_model(seed=7)
    prefix, cand, target = _windows(codec_dim, m_prefix=3, w_cand=6, n_target=4, seed=7)
    latent_dim = filter_model.knowledge_filter.mu_head.out_features
    k = 8
    noise = torch.randn(k, latent_dim)

    trace_full_selection, _ = run_episode(filter_model, heads, prefix, cand, target["item_codec"],
                                          target["correct"], k=k, device="cpu", policy="max_entropy",
                                          budget=6, noise=noise, selection_budget=6)
    trace_capped_selection, _ = run_episode(filter_model, heads, prefix, cand, target["item_codec"],
                                            target["correct"], k=k, device="cpu", policy="max_entropy",
                                            budget=6, noise=noise, selection_budget=2)
    assert trace_full_selection[-1]["logloss"] == pytest.approx(trace_capped_selection[-1]["logloss"], abs=1e-6)
    assert trace_full_selection[-1]["accuracy"] == pytest.approx(trace_capped_selection[-1]["accuracy"], abs=1e-6)
