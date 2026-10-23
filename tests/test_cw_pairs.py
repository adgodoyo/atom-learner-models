"""Tests for src/atom/codec/cw_pairs.py (P2-5 behaviour-pair construction)."""

import pandas as pd

from atom.codec.cw_pairs import build_behaviour_pairs


def _toy_df():
    # Problem 1: two submissions with IDENTICAL vectors (different code text) -> a
    # same-behaviour pair is available; plus one with a DIFFERENT vector -> a
    # different-behaviour pair is available too.
    # Problem 2: only one submission -> no pairs possible (must be skipped, not crash).
    rows = [
        {"ProblemID": 1, "CodeStateID": "c1", "Code": "return a+b;", "prompt": "p1",
         "binary_correctness": [1, 1, 0]},
        {"ProblemID": 1, "CodeStateID": "c2", "Code": "int s=a+b; return s;", "prompt": "p1",
         "binary_correctness": [1, 1, 0]},  # same behaviour as c1, different style
        {"ProblemID": 1, "CodeStateID": "c3", "Code": "return 0;", "prompt": "p1",
         "binary_correctness": [0, 0, 0]},  # different behaviour
        {"ProblemID": 2, "CodeStateID": "c4", "Code": "return a;", "prompt": "p2",
         "binary_correctness": [1]},
    ]
    return pd.DataFrame(rows)


def test_build_behaviour_pairs_basic():
    pairs = build_behaviour_pairs(_toy_df(), pairs_per_problem=10, seed=0)
    # Only problem 1 can form pairs (problem 2 has a single submission).
    assert set(pairs["problem_id"]) == {1}
    assert set(pairs["same_behaviour"]) == {True, False}
    same = pairs[pairs["same_behaviour"]]
    diff = pairs[~pairs["same_behaviour"]]
    assert len(same) == 1  # only (c1, c2) share a vector
    assert len(diff) == 2  # (c1,c3) and (c2,c3)
    assert set(same.iloc[0][["id_a", "id_b"]]) == {"c1", "c2"}


def test_build_behaviour_pairs_deterministic():
    p1 = build_behaviour_pairs(_toy_df(), pairs_per_problem=10, seed=0)
    p2 = build_behaviour_pairs(_toy_df(), pairs_per_problem=10, seed=0)
    pd.testing.assert_frame_equal(p1, p2)


def test_build_behaviour_pairs_requires_different_code_text():
    # A duplicate-text submission (same Code as c1) must not create a trivial pair with c1.
    df = _toy_df()
    dup = pd.DataFrame([{"ProblemID": 1, "CodeStateID": "c5", "Code": "return a+b;",
                        "prompt": "p1", "binary_correctness": [1, 1, 0]}])
    df = pd.concat([df, dup], ignore_index=True)
    pairs = build_behaviour_pairs(df, pairs_per_problem=10, seed=0)
    ids_used = set(pairs["id_a"]) | set(pairs["id_b"])
    assert "c5" not in ids_used  # deduped away as identical text to c1


def test_build_behaviour_pairs_respects_pairs_per_problem_cap():
    rows = []
    for i in range(20):
        rows.append({"ProblemID": 1, "CodeStateID": f"same_{i}", "Code": f"int x={i};",
                    "prompt": "p", "binary_correctness": [1, 1]})
    for i in range(20):
        rows.append({"ProblemID": 1, "CodeStateID": f"diff_{i}", "Code": f"int y={i};",
                    "prompt": "p", "binary_correctness": [0, 0]})
    df = pd.DataFrame(rows)
    pairs = build_behaviour_pairs(df, pairs_per_problem=3, seed=0)
    assert int(pairs["same_behaviour"].sum()) == 3
    assert int((~pairs["same_behaviour"]).sum()) == 3
