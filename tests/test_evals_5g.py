"""Unit tests for `atom.eval.evals_5g` -- pure functions, synthetic data, no model/GPU needed."""

from __future__ import annotations

import numpy as np
import pytest

from atom.eval.evals_5g import (belief_swap_directed_accuracy, history_curve,
                                loo_oracle_ceiling, per_type_split)


def test_per_type_split_separates_types_and_computes_ci_per_cell():
    n = 40
    rng = np.random.default_rng(0)
    types = np.array((["fill_in"] * 20) + (["mc1"] * 20))
    learners = np.arange(n) % 10
    personalised_nll = rng.normal(1.0, 0.1, n)
    shuffled_nll = personalised_nll + 0.05  # personalised is uniformly BETTER (lower)
    personalised_em = rng.integers(0, 2, n).astype(float)
    shuffled_em = rng.integers(0, 2, n).astype(float)

    out = per_type_split(types, learners, personalised_nll, shuffled_nll,
                         personalised_em, shuffled_em, n_boot=200, seed=0)
    assert set(out.keys()) == {"fill_in", "mc1"}
    assert out["fill_in"]["n"] == 20
    ci = out["fill_in"]["nll_delta_personalised_minus_shuffled_u"]
    assert ci["excludes_zero"] is True
    assert ci["point_delta"] < 0  # personalised NLL is lower -> negative delta


def test_per_type_split_reports_unmapped_and_skips_tiny_cells():
    types = np.array([None, None, "a"])
    learners = np.array([0, 1, 2])
    x = np.array([1.0, 1.0, 1.0])
    out = per_type_split(types, learners, x, x, x, x, n_boot=10, seed=0)
    assert "__unmapped__" in out
    assert out["__unmapped__"]["n"] == 2
    assert out["a"]["n"] == 1
    assert "note" in out["a"]  # too few rows for a CI


def test_history_curve_bins_and_zero_gain_at_t_zero():
    n = 60
    rng = np.random.default_rng(1)
    t = np.concatenate([np.zeros(20), rng.integers(10, 24, 20), rng.integers(100, 200, 20)])
    learners = np.arange(n) % 6
    personalised_nll = np.full(n, 1.0)
    shuffled_nll = np.full(n, 1.0)  # no gain anywhere in this synthetic set
    out = history_curve(t, learners, personalised_nll, shuffled_nll,
                        bins=(0, 10, 25, 10_000), n_boot=200, seed=0)
    assert out["[0,10)"]["n"] == 20
    assert out["[0,10)"]["gain_ci"]["point_delta"] == pytest.approx(0.0)
    assert out["[10,25)"]["n"] == 20
    assert out["[25,inf)"]["n"] == 20


def test_loo_oracle_ceiling_recovers_a_known_modal_answer():
    # cell (problem=1, class=0) has 3 rows: two answer "4", one answers "5" -> LOO modal
    # for the "5" row is "4" (wrong); LOO modal for each "4" row is "4" (right, since the
    # OTHER "4" row + the "5" row -> mode among {"4","5"} is "4", a tie broken toward the
    # sorted-first value only if counts tie; here "4" appears once among the others too,
    # "5" once -- genuine tie. Use a 4th row so one "4"-row's LOO set is unambiguous.
    problem_ids = [1, 1, 1, 1]
    true_class = [0, 0, 0, 0]
    answers = ["4", "4", "4", "5"]
    personalised_em = [1.0, 1.0, 1.0, 0.0]
    out = loo_oracle_ceiling(problem_ids, true_class, answers, personalised_em)
    assert out["n_covered"] == 4
    assert out["n_cells"] == 1
    # LOO for each "4" row: others = {"4","4","5"} -> modal "4" -> correct.
    # LOO for the "5" row: others = {"4","4","4"} -> modal "4" -> wrong (answer is "5").
    assert out["loo_oracle_exact_match_mean"] == pytest.approx(0.75)
    assert out["atom_personalised_exact_match_mean_same_subset"] == pytest.approx(0.75)


def test_loo_oracle_ceiling_excludes_singleton_cells():
    problem_ids = [1, 2]  # each its own cell, size 1 -> not covered
    true_class = [0, 0]
    answers = ["a", "b"]
    personalised_em = [1.0, 0.0]
    out = loo_oracle_ceiling(problem_ids, true_class, answers, personalised_em)
    assert out["n_covered"] == 0


def test_loo_oracle_ceiling_reports_exposure_quartile_gradient():
    rng = np.random.default_rng(2)
    n_problems = 20
    problem_ids = np.repeat(np.arange(n_problems), 5)  # 5 val rows per problem, same class
    true_class = np.zeros(len(problem_ids))
    answers = np.array(["ans"] * len(problem_ids))  # everyone agrees -> oracle always right
    personalised_em = rng.integers(0, 2, len(problem_ids)).astype(float)
    exposure = {pid: int(rng.integers(10, 2000)) for pid in range(n_problems)}
    out = loo_oracle_ceiling(problem_ids, true_class, answers, personalised_em,
                             train_exposure_by_problem=exposure, n_exposure_quantiles=4)
    assert out["loo_oracle_exact_match_mean"] == pytest.approx(1.0)
    assert "by_train_exposure_quartile" in out
    quartiles = out["by_train_exposure_quartile"]
    assert len(quartiles) <= 4 and len(quartiles) >= 1
    total_n = sum(q["n"] for q in quartiles.values())
    assert total_n == out["n_covered"]


def test_belief_swap_directed_accuracy_matches_and_scoring_when_all_agree():
    # every pair: both i and j favour their own belief -> AND accuracy 1.0, directed 1.0
    n_pairs = 10
    nll_i_own = np.full(n_pairs, 1.0)
    nll_i_other = np.full(n_pairs, 2.0)
    nll_j_own = np.full(n_pairs, 1.0)
    nll_j_other = np.full(n_pairs, 2.0)
    out = belief_swap_directed_accuracy(nll_i_own, nll_i_other, nll_j_own, nll_j_other)
    assert out["and_both_directions"]["null"] == 0.25
    assert out["and_both_directions"]["accuracy"] == pytest.approx(1.0)
    assert out["directed"]["null"] == 0.5
    assert out["directed"]["accuracy"] == pytest.approx(1.0)
    assert out["directed"]["n_directed_comparisons"] == 2 * n_pairs


def test_belief_swap_directed_accuracy_half_agree_gives_and_below_directed():
    # i always favours own; j never does -> AND accuracy 0.0, directed accuracy 0.5
    n_pairs = 20
    nll_i_own = np.full(n_pairs, 1.0)
    nll_i_other = np.full(n_pairs, 2.0)  # i favours own
    nll_j_own = np.full(n_pairs, 2.0)
    nll_j_other = np.full(n_pairs, 1.0)  # j favours OTHER
    out = belief_swap_directed_accuracy(nll_i_own, nll_i_other, nll_j_own, nll_j_other)
    assert out["and_both_directions"]["accuracy"] == pytest.approx(0.0)
    assert out["directed"]["accuracy"] == pytest.approx(0.5)


def test_belief_swap_directed_accuracy_empty_input_is_nan_not_a_crash():
    out = belief_swap_directed_accuracy([], [], [], [])
    assert out["n_pairs"] == 0
    assert np.isnan(out["and_both_directions"]["accuracy"])
    assert np.isnan(out["directed"]["accuracy"])
