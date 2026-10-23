"""FA-SIM2's paired belief-swap eval -- the model-free half (pairing, edit distance,
the graded report). The likelihood-scoring half needs a real backbone and is tested via
the fa_sim2 smoke test instead.
"""

from __future__ import annotations

import math

from atom.eval.belief_swap import edit_distance, find_discriminating_pairs, graded_accuracy_by_edit_distance


def test_find_discriminating_pairs_basic_case():
    problem_id = [1, 1, 1, 2, 2, 3]
    learner = ["A", "B", "C", "A", "B", "A"]
    answer_text = ["x", "y", "x", "same", "same", "solo"]
    pairs = find_discriminating_pairs(problem_id, learner, answer_text)
    # item 1: A/B differ (pair), A/C same answer (excluded), B/C differ (pair)
    # item 2: A/B same answer -> excluded
    # item 3: only one respondent -> no pairs
    assert set(pairs) == {(0, 1), (1, 2)}


def test_find_discriminating_pairs_excludes_same_learner_repeats():
    problem_id = [1, 1]
    learner = ["A", "A"]  # same learner answered twice (e.g. retry) -- not a discriminating pair
    answer_text = ["x", "y"]
    assert find_discriminating_pairs(problem_id, learner, answer_text) == []


def test_find_discriminating_pairs_excludes_single_respondent_items():
    problem_id = [1, 2, 3]
    learner = ["A", "B", "C"]
    answer_text = ["x", "y", "z"]
    assert find_discriminating_pairs(problem_id, learner, answer_text) == []


def test_find_discriminating_pairs_matches_measured_fa_val_scale():
    """Not a re-derivation of the exact FA numbers (needs real data), but a scale sanity
    check: the measured stratum (883 items with >=2 learners, 582 with >=2 distinct
    answers, 1,892 discriminating pairs) implies an average of ~3.25 discriminating pairs
    per multi-answer item, well within what a synthetic population of a similar shape
    should also produce -- catches a wildly wrong pairing (e.g. O(n^2) over the whole
    population instead of within-item) without needing the real corpus."""
    import numpy as np

    rng = np.random.default_rng(0)
    n_items, learners_per_item = 100, 4
    problem_id = np.repeat(np.arange(n_items), learners_per_item)
    learner = np.tile(np.arange(learners_per_item), n_items)
    # answers drawn from a small alphabet so some collisions (same-answer pairs) occur naturally
    answer_text = rng.integers(0, 3, size=n_items * learners_per_item).astype(str)
    pairs = find_discriminating_pairs(problem_id, learner, answer_text)
    max_possible = n_items * math.comb(learners_per_item, 2)
    assert 0 < len(pairs) < max_possible  # some excluded by same-answer, not all/none


def test_edit_distance_known_values():
    assert edit_distance("", "") == 0
    assert edit_distance("abc", "abc") == 0
    assert edit_distance("", "abc") == 3
    assert edit_distance("abc", "") == 3
    assert edit_distance("kitten", "sitting") == 3  # classic textbook example
    assert edit_distance("42", "24") == 2


def test_graded_accuracy_by_edit_distance_overall_matches_mean():
    correct = [1, 1, 0, 1, 0]
    dist = [0, 1, 2, 3, 4]
    report = graded_accuracy_by_edit_distance(correct, dist)
    assert report["overall"]["n"] == 5
    assert math.isclose(report["overall"]["accuracy"], 3 / 5)


def test_graded_accuracy_by_edit_distance_rising_curve_detected():
    """A model that is genuinely personalised should do BETTER at distinguishing
    strongly-contrasting answer pairs than near-identical ones -- construct exactly
    that shape and confirm the binned report surfaces it. Checked via the FIRST-vs-LAST
    bin contrast (not strict per-bin monotonicity, which is too strict a bar under finite
    binomial noise per bin -- a real rising signal doesn't have to be monotone in every
    intermediate bin to be a real rising signal)."""
    import numpy as np

    rng = np.random.default_rng(0)
    n = 2000
    dist = rng.integers(0, 10, size=n)
    # accuracy rises with edit distance: p(correct) = 0.5 + 0.04*dist, capped at 0.95
    p_correct = np.clip(0.5 + 0.04 * dist, 0, 0.95)
    correct = (rng.random(n) < p_correct).astype(float)
    report = graded_accuracy_by_edit_distance(correct, dist)
    bins = [b for b in report["bins"] if b["n"] > 0]
    assert bins[0]["accuracy"] < bins[-1]["accuracy"] - 0.1  # clearly rises, not flat/noisy


def test_graded_accuracy_by_edit_distance_flat_curve_is_honest_null():
    import numpy as np

    rng = np.random.default_rng(1)
    n = 400
    dist = rng.integers(0, 10, size=n)
    correct = (rng.random(n) < 0.5).astype(float)  # no relationship to edit distance
    report = graded_accuracy_by_edit_distance(correct, dist)
    accuracies = [b["accuracy"] for b in report["bins"] if b["n"] > 0]
    assert min(accuracies) > 0.3 and max(accuracies) < 0.7  # roughly flat around 0.5


def test_graded_accuracy_by_edit_distance_empty_input_does_not_crash():
    report = graded_accuracy_by_edit_distance([], [])
    assert report["overall"]["n"] == 0
    assert math.isnan(report["overall"]["accuracy"])
