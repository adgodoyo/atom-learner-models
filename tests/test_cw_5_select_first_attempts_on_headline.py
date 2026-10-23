"""CW-SIM launch 2 row selection (`reports/cw_sim_launch2_pregistration.md`, 1 Sep):
`select_first_attempts_on_headline` must return exactly one row per (val learner,
headline problem) pair -- the EARLIEST attempt by `t`, never an arbitrary one, and
never a row for a learner/problem pair with no attempts."""
from __future__ import annotations

import pathlib
import sys

import pandas as pd

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "infra"))

from cw_sim_core import HEADLINE_PROBLEM_IDS, select_first_attempts_on_headline  # noqa: E402


def _attempts(rows):
    return pd.DataFrame(rows)


def test_picks_earliest_t_not_an_arbitrary_row():
    # Learner l1 attempts headline problem 1 three times (t=0,2,5); must keep t=0.
    attempts = _attempts([
        {"learner": "l1", "problem_id": 1, "t": 2, "code_state_id": "b"},
        {"learner": "l1", "problem_id": 1, "t": 0, "code_state_id": "a"},  # earliest
        {"learner": "l1", "problem_id": 1, "t": 5, "code_state_id": "c"},
    ])
    out = select_first_attempts_on_headline(attempts, val_learners=["l1"])
    assert len(out) == 1
    assert out.iloc[0]["code_state_id"] == "a"
    assert out.iloc[0]["t"] == 0


def test_one_row_per_learner_per_headline_problem_pair():
    attempts = _attempts([
        {"learner": "l1", "problem_id": 1, "t": 0, "code_state_id": "a"},
        {"learner": "l1", "problem_id": 3, "t": 1, "code_state_id": "b"},
        {"learner": "l1", "problem_id": 3, "t": 4, "code_state_id": "c"},  # dupe, later
        {"learner": "l2", "problem_id": 1, "t": 0, "code_state_id": "d"},
    ])
    out = select_first_attempts_on_headline(attempts, val_learners=["l1", "l2"])
    assert len(out) == 3  # (l1,1) (l1,3) (l2,1) -- never 4
    pairs = set(zip(out["learner"], out["problem_id"]))
    assert pairs == {("l1", 1), ("l1", 3), ("l2", 1)}


def test_non_headline_problems_excluded():
    attempts = _attempts([
        {"learner": "l1", "problem_id": 1, "t": 0, "code_state_id": "a"},
        {"learner": "l1", "problem_id": 999, "t": 1, "code_state_id": "b"},  # not headline
    ])
    out = select_first_attempts_on_headline(attempts, val_learners=["l1"])
    assert len(out) == 1
    assert out.iloc[0]["problem_id"] == 1


def test_non_val_learners_excluded():
    attempts = _attempts([
        {"learner": "l1", "problem_id": 1, "t": 0, "code_state_id": "a"},
        {"learner": "l_not_val", "problem_id": 1, "t": 0, "code_state_id": "b"},
    ])
    out = select_first_attempts_on_headline(attempts, val_learners=["l1"])
    assert len(out) == 1
    assert (out["learner"] == "l1").all()


def test_learner_with_zero_headline_attempts_contributes_zero_rows_no_error():
    attempts = _attempts([
        {"learner": "l1", "problem_id": 999, "t": 0, "code_state_id": "a"},
    ])
    out = select_first_attempts_on_headline(attempts, val_learners=["l1"])
    assert len(out) == 0


def test_empty_input_returns_empty_no_error():
    attempts = _attempts([])
    # Empty frame has no problem_id/learner columns at all -- guard against KeyError.
    attempts = pd.DataFrame(columns=["learner", "problem_id", "t", "code_state_id"])
    out = select_first_attempts_on_headline(attempts, val_learners=["l1"])
    assert len(out) == 0


def test_default_headline_ids_match_the_13_join_verified_problems():
    assert set(HEADLINE_PROBLEM_IDS) == {1, 3, 5, 12, 13, 17, 20, 21, 22, 24, 25, 46, 71}
    assert 108 not in HEADLINE_PROBLEM_IDS  # the dropped secondary row, never re-added
