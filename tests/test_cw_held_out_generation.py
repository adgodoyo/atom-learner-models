"""`atom.data.cw_held_out_generation`: `select_first_attempts` (earliest-attempt selection + outcome labelling) and
`join_nearest_prior_belief` (learner-scoped nearest-backward belief join) are both
pure/synthetic-testable; `load_held_out_problem_attempts` itself needs real CodeWorkout raw
tables and is exercised by the actual launch, not here."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from atom.data.cw_held_out_generation import join_nearest_prior_belief, select_first_attempts


def _attempts(rows):
    return pd.DataFrame(rows)


def test_select_first_attempts_picks_earliest_t_per_learner_problem_pair():
    attempts = _attempts([
        {"learner": "l1", "problem_id": 51, "episode_id": "e1", "t": 2, "code": "b",
         "compile_result": "Success", "score_frac": 1.0},
        {"learner": "l1", "problem_id": 51, "episode_id": "e0", "t": 0, "code": "a",
         "compile_result": "Success", "score_frac": 0.0},
        {"learner": "l2", "problem_id": 56, "episode_id": "e2", "t": 0, "code": "c",
         "compile_result": "Error", "score_frac": np.nan},
    ])
    out = select_first_attempts(attempts)
    assert len(out) == 2
    l1_row = out[out["learner"] == "l1"].iloc[0]
    assert l1_row["t"] == 0 and l1_row["code"] == "a"
    assert l1_row["outcome_code"] == 1  # Success, score 0 -> CLEAN_ZERO
    l2_row = out[out["learner"] == "l2"].iloc[0]
    assert l2_row["outcome_code"] == 0  # Error -> COMPILE_ERROR


def test_select_first_attempts_prior_code_is_always_none():
    # A learner's FIRST attempt on a problem starts a fresh episode
    # (attempt_within_idx == 0), so prior_code must be None regardless of the row order
    # fed in, even when the same episode_id happens to have a later duplicate.
    attempts = _attempts([
        {"learner": "l1", "problem_id": 51, "episode_id": "e0", "t": 0, "code": "a",
         "compile_result": "Success", "score_frac": 0.5},
        {"learner": "l1", "problem_id": 51, "episode_id": "e0", "t": 1, "code": "a2",
         "compile_result": "Success", "score_frac": 0.5},
    ])
    out = select_first_attempts(attempts)
    assert len(out) == 1
    assert pd.isna(out.iloc[0]["prior_code"]) or out.iloc[0]["prior_code"] is None


def test_select_first_attempts_drops_outcome_invalid_rows():
    attempts = _attempts([
        {"learner": "l1", "problem_id": 51, "episode_id": "e0", "t": 0, "code": "a",
         "compile_result": "Success", "score_frac": np.nan},  # clean but no score -> invalid
    ])
    out = select_first_attempts(attempts)
    assert len(out) == 0


def test_select_first_attempts_empty_input_returns_empty_no_error():
    out = select_first_attempts(pd.DataFrame(columns=[
        "learner", "problem_id", "episode_id", "t", "code", "compile_result", "score_frac"]))
    assert len(out) == 0


def _dense_beliefs(rows):
    return {
        "learner": np.array([r[0] for r in rows]),
        "episode_id": np.array([r[1] for r in rows]),
        "mu": np.array([r[2] for r in rows], dtype=np.float32),
        "log_sigma": np.array([r[3] for r in rows], dtype=np.float32),
    }


def test_join_picks_the_learners_most_recent_prior_episode():
    held_out = pd.DataFrame({
        "learner": ["l1"], "problem_id": [51], "timestamp": [100.0],
    })
    dense = _dense_beliefs([
        ("l1", "e0", [1.0, 1.0], [0.0, 0.0]),   # start_timestamp 10 -- too far back
        ("l1", "e1", [2.0, 2.0], [0.0, 0.0]),   # start_timestamp 50 -- the answer
        ("l1", "e2", [3.0, 3.0], [0.0, 0.0]),   # start_timestamp 150 -- AFTER the attempt
    ])
    episodes = pd.DataFrame({"episode_id": ["e0", "e1", "e2"],
                             "start_timestamp": [10.0, 50.0, 150.0]})
    out = join_nearest_prior_belief(held_out, dense, episodes)
    assert len(out) == 1
    np.testing.assert_allclose(out.iloc[0]["mu"], [2.0, 2.0])
    assert out.attrs["n_dropped_no_prior_belief"] == 0
    assert out.attrs["n_input_rows"] == 1


def test_join_never_borrows_another_learners_belief():
    held_out = pd.DataFrame({"learner": ["l2"], "problem_id": [51], "timestamp": [100.0]})
    dense = _dense_beliefs([("l1", "e0", [9.0, 9.0], [0.0, 0.0])])  # only l1 has any belief
    episodes = pd.DataFrame({"episode_id": ["e0"], "start_timestamp": [10.0]})
    out = join_nearest_prior_belief(held_out, dense, episodes)
    assert len(out) == 0  # l2 has no belief at all -- dropped, not borrowed from l1
    assert out.attrs["n_dropped_no_prior_belief"] == 1


def test_join_drops_rows_with_no_qualifying_prior_episode():
    held_out = pd.DataFrame({"learner": ["l1"], "problem_id": [51], "timestamp": [5.0]})
    dense = _dense_beliefs([("l1", "e0", [1.0, 1.0], [0.0, 0.0])])
    episodes = pd.DataFrame({"episode_id": ["e0"], "start_timestamp": [10.0]})  # AFTER the attempt
    out = join_nearest_prior_belief(held_out, dense, episodes)
    assert len(out) == 0
    assert out.attrs["n_dropped_no_prior_belief"] == 1
    assert out.attrs["n_input_rows"] == 1


def test_join_empty_input_returns_empty_no_error():
    dense = _dense_beliefs([("l1", "e0", [1.0], [0.0])])
    episodes = pd.DataFrame({"episode_id": ["e0"], "start_timestamp": [10.0]})
    out = join_nearest_prior_belief(pd.DataFrame(columns=["learner", "problem_id", "timestamp"]),
                                    dense, episodes)
    assert len(out) == 0
    assert out.attrs["n_input_rows"] == 0


def test_join_handles_int_episode_ids_in_episodes_table_vs_string_in_dense_beliefs():
    # `build_episode_level_table`'s own output keys episode_id as int64; the dense
    # belief npz export persists it as a string array (canary-#7's value-join
    # convention) -- a real dtype mismatch this join must coerce through, not refuse.
    held_out = pd.DataFrame({"learner": ["l1"], "problem_id": [51], "timestamp": [100.0]})
    dense = _dense_beliefs([("l1", "7", [5.0], [0.0])])  # string episode_id, as in the npz
    episodes = pd.DataFrame({"episode_id": [7], "start_timestamp": [10.0]})  # int64
    out = join_nearest_prior_belief(held_out, dense, episodes)
    assert len(out) == 1
    np.testing.assert_allclose(out.iloc[0]["mu"], [5.0])


def test_join_raises_if_a_belief_episode_id_is_missing_from_episodes_table():
    held_out = pd.DataFrame({"learner": ["l1"], "problem_id": [51], "timestamp": [100.0]})
    dense = _dense_beliefs([("l1", "e_missing", [1.0], [0.0])])
    episodes = pd.DataFrame({"episode_id": ["e_other"], "start_timestamp": [10.0]})
    with pytest.raises(AssertionError):
        join_nearest_prior_belief(held_out, dense, episodes)
