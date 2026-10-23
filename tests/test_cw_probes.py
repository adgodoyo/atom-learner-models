"""Tests for src/atom/codec/cw_probes.py (P2-5b probe-battery pair construction)."""

import pandas as pd

from atom.codec.cw_probes import (
    build_behaviour_gross_pairs,
    build_behaviour_nearmiss_pairs,
    build_person_pairs,
    build_submission_pool,
    build_task_pairs,
    edit_similarity,
    split_problems,
)


def test_split_problems_deterministic_and_covers_all():
    calib, confirm = split_problems(list(range(1, 18)), confirm_frac=0.3, seed=0)
    calib2, confirm2 = split_problems(list(range(1, 18)), confirm_frac=0.3, seed=0)
    assert calib == calib2 and confirm == confirm2
    assert set(calib) | set(confirm) == set(range(1, 18))
    assert set(calib) & set(confirm) == set()
    assert len(confirm) == round(17 * 0.3)


def test_split_problems_different_seed_can_differ():
    calib_a, confirm_a = split_problems(list(range(1, 18)), confirm_frac=0.3, seed=0)
    calib_b, confirm_b = split_problems(list(range(1, 18)), confirm_frac=0.3, seed=1)
    assert (calib_a, confirm_a) != (calib_b, confirm_b)


def _toy_df():
    rows = []
    # Problem 1: 3 students, each with 2 attempts (some same, some near-miss, some gross).
    rows += [
        {"ProblemID": 1, "SubjectID": "s1", "CodeStateID": "a1", "Code": "return a+b;",
         "binary_correctness": [1, 1, 0, 1]},
        {"ProblemID": 1, "SubjectID": "s1", "CodeStateID": "a2", "Code": "int s=a+b; return s;",
         "binary_correctness": [1, 1, 0, 1]},  # same vector as a1, same student -> same-author + same-behaviour
        {"ProblemID": 1, "SubjectID": "s2", "CodeStateID": "b1", "Code": "return a-b;",
         "binary_correctness": [1, 1, 0, 0]},  # differs from a1 by 1 (near-miss vs a1)
        {"ProblemID": 1, "SubjectID": "s3", "CodeStateID": "c1", "Code": "return 0;",
         "binary_correctness": [0, 0, 0, 0]},  # differs from a1 by 3 (gross vs a1)
    ]
    # Problem 2: gives the task probe a second problem to separate from.
    rows += [
        {"ProblemID": 2, "SubjectID": "s1", "CodeStateID": "d1", "Code": "return x*y;",
         "binary_correctness": [1, 0]},
        {"ProblemID": 2, "SubjectID": "s4", "CodeStateID": "d2", "Code": "return y*x;",
         "binary_correctness": [1, 0]},
    ]
    return pd.DataFrame(rows)


def test_build_submission_pool_dedups_and_samples():
    df = _toy_df()
    dup = pd.DataFrame([{"ProblemID": 1, "SubjectID": "s5", "CodeStateID": "a1dup",
                        "Code": "return a+b;", "binary_correctness": [1, 1, 0, 1]}])
    df = pd.concat([df, dup], ignore_index=True)
    pool = build_submission_pool(df, [1, 2], pool_per_problem=100, seed=0)
    # "return a+b;" appears twice (a1, a1dup) -- drop_duplicates(subset=Code) keeps one.
    assert pool["Code"].tolist().count("return a+b;") == 1


def test_build_task_pairs_labels_same_vs_different_problem():
    pool = build_submission_pool(_toy_df(), [1, 2], pool_per_problem=100, seed=0)
    pairs = build_task_pairs(pool, pairs_total=20, seed=0)
    assert set(pairs["label"]) == {True, False}
    for r in pairs.itertuples(index=False):
        pid_a = pool.loc[r.id_a, "ProblemID"]
        pid_b = pool.loc[r.id_b, "ProblemID"]
        assert r.label == (pid_a == pid_b)


def _hamming(v1, v2):
    return sum(1 for x, y in zip(v1, v2) if x != y)


def test_behaviour_gross_and_nearmiss_select_the_right_distance_regime():
    pool = build_submission_pool(_toy_df(), [1, 2], pool_per_problem=100, seed=0)
    gross = build_behaviour_gross_pairs(pool, pairs_per_problem=10, seed=0)
    nearmiss = build_behaviour_nearmiss_pairs(pool, pairs_per_problem=10, seed=0)
    assert len(gross[~gross["label"]]) > 0 and len(nearmiss[~nearmiss["label"]]) > 0
    for r in gross[~gross["label"]].itertuples(index=False):
        va = tuple(pool.loc[r.id_a, "binary_correctness"])
        vb = tuple(pool.loc[r.id_b, "binary_correctness"])
        assert _hamming(va, vb) >= max(1, len(va) // 2)
    for r in nearmiss[~nearmiss["label"]].itertuples(index=False):
        va = tuple(pool.loc[r.id_a, "binary_correctness"])
        vb = tuple(pool.loc[r.id_b, "binary_correctness"])
        assert _hamming(va, vb) == 1


def test_build_person_pairs_problem_identity_fixed():
    pool = build_submission_pool(_toy_df(), [1, 2], pool_per_problem=100, seed=0)
    pairs = build_person_pairs(pool, pairs_per_problem=10, seed=0)
    for r in pairs.itertuples(index=False):
        pid_a = pool.loc[r.id_a, "ProblemID"]
        pid_b = pool.loc[r.id_b, "ProblemID"]
        assert pid_a == pid_b  # problem identity never varies within a person-probe pair
        sid_a = pool.loc[r.id_a, "SubjectID"]
        sid_b = pool.loc[r.id_b, "SubjectID"]
        assert r.label == (sid_a == sid_b)


def test_edit_similarity_identical_is_one_and_different_is_lower():
    assert edit_similarity("return a+b;", "return a+b;") == 1.0
    assert edit_similarity("return a+b;", "return 0;") < 1.0
    assert edit_similarity("return a+b;", "return 0;") >= 0.0
