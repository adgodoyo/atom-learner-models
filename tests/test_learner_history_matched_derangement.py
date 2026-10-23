"""FA-SIM3 requirement 10 -- `atom.binding.derangement.learner_history_matched_derangement`
and its permanent canary `assert_learner_matched_derangement`. Per the SEMANTIC CANARIES
rule, checks the STRUCTURAL invariant directly: every deranged row receives a DIFFERENT
learner's belief (not merely a different row index, which a plain row-level derangement
already guarantees but which is insufficient once one learner owns several rows), and
history-bin membership is respected (no cross-bin assignment at all).
"""

from __future__ import annotations

import numpy as np
import pytest

from atom.binding.derangement import assert_learner_matched_derangement, learner_history_matched_derangement


def test_every_deranged_row_maps_to_a_different_learner():
    rng = np.random.default_rng(0)
    n_learners, rows_per_learner = 50, 4
    learner_ids = np.repeat(np.arange(n_learners), rows_per_learner)
    bin_ids = np.zeros(len(learner_ids), dtype=int)  # single bin
    perm, deranged_mask = learner_history_matched_derangement(learner_ids, bin_ids, rng=rng)
    assert_learner_matched_derangement(perm, learner_ids, deranged_mask)
    assert deranged_mask.all()  # 50 distinct learners, ample room to derange fully


def test_never_crosses_a_history_bin():
    rng = np.random.default_rng(1)
    n = 300
    learner_ids = rng.integers(0, 100, size=n)
    bin_ids = rng.integers(0, 3, size=n)  # 3 history bins
    perm, deranged_mask = learner_history_matched_derangement(learner_ids, bin_ids, rng=rng)
    assert_learner_matched_derangement(perm, learner_ids, deranged_mask)
    assert np.array_equal(bin_ids[perm], bin_ids)  # every row stays within its own bin


def test_single_learner_bin_is_flagged_not_silently_self_mapped():
    """A bin with only one distinct learner cannot be deranged -- must be flagged
    `deranged_mask=False`, not silently left mapping to itself and passed off as a
    control (the exact failure this requirement exists to prevent)."""
    learner_ids = np.array([7, 7, 7, 3, 3, 9, 9])
    # bin 0: only learner 7 (infeasible: 1 distinct learner); bin 1: learners 3,9 with a
    # balanced 2/2 split (feasible: largest group 2 <= floor(4/2)=2).
    bin_ids = np.array([0, 0, 0, 1, 1, 1, 1])
    perm, deranged_mask = learner_history_matched_derangement(
        learner_ids, bin_ids, rng=np.random.default_rng(2))
    assert not deranged_mask[:3].any()  # bin 0 entirely un-deranged
    assert np.array_equal(perm[:3], [0, 1, 2])  # self-mapped
    assert deranged_mask[3:].all()  # bin 1: balanced 2-learner split -- fully deranged
    assert_learner_matched_derangement(perm, learner_ids, deranged_mask)


def test_deranged_mask_reflects_learner_diversity_not_row_count():
    """Two learners, many rows each -- deranging IS possible (row-level derangement
    within a 2-learner bin just swaps every row of learner A onto a row of learner B and
    vice versa), unlike the single-learner case above."""
    learner_ids = np.array([1, 1, 1, 1, 2, 2, 2, 2])
    bin_ids = np.zeros(8, dtype=int)
    perm, deranged_mask = learner_history_matched_derangement(
        learner_ids, bin_ids, rng=np.random.default_rng(3))
    assert deranged_mask.all()
    assert_learner_matched_derangement(perm, learner_ids, deranged_mask)
    assert (learner_ids[perm] != learner_ids).all()


def test_would_have_caught_a_plain_row_level_derangement_on_multi_row_learners():
    """Would-have-caught regression check (SEMANTIC CANARIES convention): a plain
    row-level derangement (`perm[i] != i`, FA-SIM2's `derangement_permutation`) is
    insufficient once a learner owns multiple rows in the same bin -- it can (and, given
    enough rows per learner, systematically will over many draws) map one of a learner's
    rows onto ANOTHER of that same learner's own rows, which is not a personalisation
    control at all. Construct exactly that permutation by hand and confirm the new
    canary correctly REJECTS it."""
    learner_ids = np.array([1, 1, 1, 1, 2, 2])
    deranged_mask = np.ones(6, dtype=bool)
    # a valid ROW-level derangement (no perm[i]==i) that nonetheless maps row 0 -> row 1,
    # both belonging to learner 1 -- exactly the case a row-only check cannot see.
    bad_perm = np.array([1, 0, 3, 2, 5, 4])
    with pytest.raises(AssertionError):
        assert_learner_matched_derangement(bad_perm, learner_ids, deranged_mask)


def test_raises_fast_when_one_learner_dominates_a_bin_pigeonhole_infeasible():
    """>=2 distinct learners is NECESSARY but not SUFFICIENT: [1, 1, 2] has 2 distinct
    learners in a 3-row bin, yet is provably infeasible by pigeonhole -- learner 1's two
    rows would each need a non-learner-1 target, but only one such row (learner 2's)
    exists. Confirms the feasibility check raises promptly (not silently returning a
    self-mapped row, and not burning `max_attempts` rejection-sampling draws on an
    input that could never succeed)."""
    rng = np.random.default_rng(4)
    learner_ids = np.array([1, 1, 2])
    bin_ids = np.zeros(3, dtype=int)
    with pytest.raises(RuntimeError, match="pigeonhole"):
        learner_history_matched_derangement(learner_ids, bin_ids, rng=rng, max_attempts=50)


def test_two_learners_balanced_split_is_feasible_and_succeeds():
    """The feasible counterpart to the pigeonhole test above: 2 distinct learners with a
    balanced split (largest group <= floor(n/2)) must succeed, confirming the
    feasibility check is not overly conservative."""
    rng = np.random.default_rng(5)
    learner_ids = np.array([1, 1, 2, 2])
    bin_ids = np.zeros(4, dtype=int)
    perm, deranged_mask = learner_history_matched_derangement(learner_ids, bin_ids, rng=rng)
    assert deranged_mask.all()
    assert_learner_matched_derangement(perm, learner_ids, deranged_mask)
