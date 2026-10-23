"""Paired belief-swap evaluation of personalisation in the mathematics simulator.

Generation-based discrimination is uninformative on FoundationalASSIST answers, which are
very short (median 3 characters, 53.3% <=3): there is no room for stylistic signal in "42".
This instrument SCORES rather than generates: for each pair of learners A, B who answered
the SAME item DIFFERENTLY, it asks whether
`p(A's answer | A's belief) > p(A's answer | B's belief)`. This is length-independent
(teacher-forced likelihood, not generation), has a mechanical null of exactly 0.5 (absent
personalisation, which belief you condition on cannot matter), and is the shuffled-`u`
control at the pair level, so it is immune to the stochasticity and KL confounds that
affect generation-based comparisons.

This module holds the model-free logic (finding pairs, edit distance, the graded report)
so it is testable without a backbone or GPU. The likelihood scoring needs real forward
passes with a swapped belief and lives in the training script.
"""

from __future__ import annotations

import itertools
from typing import Any, Sequence

import numpy as np


def find_discriminating_pairs(problem_id: Any, learner: Any,
                              answer_text: Any) -> list[tuple[int, int]]:
    """Row-index pairs `(i, j)` such that rows `i` and `j` share a `problem_id`, belong
    to DIFFERENT learners, and gave a DIFFERENT `answer_text`. Pure indexing logic -- no
    model, no belief -- so the expensive (GPU-bound) scoring step can be reviewed and
    tested against a frozen, cheap-to-recompute pairing.

    `O(sum over items of pairs within that item)`, not `O(n^2)` over the whole
    population -- items with a single respondent contribute nothing and are skipped
    before any pairwise comparison.
    """
    problem_id = np.asarray(problem_id)
    learner = np.asarray(learner)
    answer_text = np.asarray(answer_text, dtype=object)
    assert problem_id.shape == learner.shape == answer_text.shape

    pairs: list[tuple[int, int]] = []
    order = np.argsort(problem_id, kind="mergesort")
    sorted_pid = problem_id[order]
    boundaries = np.flatnonzero(np.diff(sorted_pid)) + 1
    groups = np.split(order, boundaries)
    for idx in groups:
        if len(idx) < 2:
            continue
        for i, j in itertools.combinations(idx.tolist(), 2):
            if learner[i] == learner[j]:
                continue
            if answer_text[i] == answer_text[j]:
                continue
            pairs.append((i, j))
    return pairs


def edit_distance(a: str, b: str) -> int:
    """Plain Levenshtein distance, O(len(a)*len(b)) DP. FA answers are short (median 3
    characters), so this is never a performance concern and needs no library
    dependency."""
    a, b = str(a), str(b)
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        cur = [i] + [0] * lb
        ai = a[i - 1]
        for j in range(1, lb + 1):
            cost = 0 if ai == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[lb]


def graded_accuracy_by_edit_distance(correct: Any, edit_distances: Any,
                                     bin_edges: Sequence[float] | None = None) -> dict:
    """`correct[k]` = 1 if `p(A's answer | A's belief) > p(A's answer | B's belief)` for
    discriminating pair `k`, else 0 (a tie counts as NOT correct -- conservative, since a
    genuinely personalised model should break ties in its own learner's favour, not by
    chance). `edit_distances[k]` = `edit_distance` between the two learners' answers for
    pair `k`.

    Reports accuracy overall and within edit-distance bins: a curve rising with edit
    distance is stronger evidence than a point estimate, and a flat one is a clear null
    rather than an ambiguous result. Default bins widen geometrically (0-1, 1-2,
    2-4, 4-8, 8+) since edit distance on short answers is right-skewed.
    """
    correct = np.asarray(correct, dtype=float)
    edit_distances = np.asarray(edit_distances, dtype=float)
    assert correct.shape == edit_distances.shape
    n = len(correct)

    if bin_edges is None:
        top = float(edit_distances.max()) + 1 if n else 1.0
        default_edges = [0, 1, 2, 4, 8, top]
        bin_edges = sorted({e for e in default_edges if e <= top} | {top})
    bin_edges = sorted(set(bin_edges))

    overall = {"n": int(n), "accuracy": float(correct.mean()) if n else float("nan")}
    bins = []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (edit_distances >= lo) & (edit_distances < hi)
        bin_n = int(mask.sum())
        bins.append({
            "edit_distance_range": [lo, hi], "n": bin_n,
            "accuracy": float(correct[mask].mean()) if bin_n else float("nan"),
        })
    return {"overall": overall, "bins": bins}
