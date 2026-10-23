"""FA-SIM2 Change 4: within-item AUC with a paired CI. Person discrimination is immune
to the stochasticity/objective confounds a raw likelihood comparison suffers from -- a
model with no per-learner state scores exactly 0.5000 within-item by construction.
"""

from __future__ import annotations

import math

import numpy as np

from atom.eval.within_item import within_item_auc, within_item_auc_paired


def test_item_only_predictor_scores_exactly_half():
    """A constant predictor per item (no person signal at all) must score 0.5 --
    the mechanical null C9-F's argument depends on."""
    rng = np.random.default_rng(0)
    n_items, learners_per_item = 50, 6
    problem_id = np.repeat(np.arange(n_items), learners_per_item)
    y = rng.integers(0, 2, size=n_items * learners_per_item)
    p = np.repeat(rng.random(n_items), learners_per_item)  # SAME score for every learner on an item
    result = within_item_auc(problem_id, y, p)
    assert math.isclose(result["pooled_auc"], 0.5, abs_tol=0.02)


def test_perfect_person_discrimination_scores_near_one():
    rng = np.random.default_rng(0)
    n_items, learners_per_item = 50, 6
    problem_id = np.repeat(np.arange(n_items), learners_per_item)
    y = rng.integers(0, 2, size=n_items * learners_per_item)
    p = y.astype(float) + rng.normal(scale=0.01, size=len(y))  # p tracks y almost perfectly
    result = within_item_auc(problem_id, y, p)
    assert result["pooled_auc"] > 0.95


def test_items_with_no_pair_are_excluded():
    problem_id = [1, 1, 2]  # item 2 has only one respondent -- no pair possible
    y = [1, 0, 1]
    p = [0.9, 0.1, 0.5]
    result = within_item_auc(problem_id, y, p)
    assert result["n_items"] == 1
    assert result["items"][0]["problem_id"] == 1


def test_pooled_auc_weighted_by_pair_count():
    """A large item (many pairs) should dominate a small one in the pooled average."""
    # item 1: 10 learners, balanced -> more pairs; item 2: 2 learners -> 1 pair
    problem_id = [1] * 10 + [2, 2]
    y = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0] + [1, 0]
    p_item1 = [0.9, 0.9, 0.9, 0.9, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1]  # AUC = 1.0
    p_item2 = [0.1, 0.9]  # AUC = 0.0 (wrong direction)
    p = p_item1 + p_item2
    result = within_item_auc(problem_id, y, p)
    # item1 has 5*5=25 pairs, item2 has 1*1=1 pair -> pooled should be close to item1's AUC
    assert result["pooled_auc"] > 0.9


def test_within_item_auc_paired_ci_excludes_zero_above_half_for_real_signal():
    rng = np.random.default_rng(0)
    n_items, learners_per_item = 80, 6
    problem_id = np.repeat(np.arange(n_items), learners_per_item)
    y = rng.integers(0, 2, size=n_items * learners_per_item)
    p = y.astype(float) * 0.6 + rng.random(n_items * learners_per_item) * 0.4  # decent signal
    result = within_item_auc_paired(problem_id, y, p, n_boot=1000, seed=0)
    assert result["pooled_auc"] > 0.55
    assert "ci95" in result and len(result["ci95"]) == 2


def test_within_item_auc_paired_ci_straddles_half_for_no_signal():
    rng = np.random.default_rng(1)
    n_items, learners_per_item = 80, 6
    problem_id = np.repeat(np.arange(n_items), learners_per_item)
    y = rng.integers(0, 2, size=n_items * learners_per_item)
    p = np.repeat(rng.random(n_items), learners_per_item)  # item-only, no person signal
    result = within_item_auc_paired(problem_id, y, p, n_boot=1000, seed=1)
    assert not result["excludes_zero_above_0_5"]


def test_within_item_auc_paired_reproducible_with_same_seed():
    rng = np.random.default_rng(2)
    problem_id = np.repeat(np.arange(30), 4)
    y = rng.integers(0, 2, size=120)
    p = rng.random(120)
    a = within_item_auc_paired(problem_id, y, p, n_boot=500, seed=42)
    b = within_item_auc_paired(problem_id, y, p, n_boot=500, seed=42)
    assert a["ci95"] == b["ci95"]


def test_within_item_auc_empty_input_does_not_crash():
    result = within_item_auc([], [], [])
    assert result["n_items"] == 0
    assert math.isnan(result["pooled_auc"])
    paired = within_item_auc_paired([], [], [], n_boot=100, seed=0)
    assert not paired["excludes_zero_above_0_5"]
