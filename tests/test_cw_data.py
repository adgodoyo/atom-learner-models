"""Tests for src/atom/data/cw.py + cw_tuples.py (P1-3 CodeWorkout ingest).

Synthetic tests exercise the event/episode/audit logic directly (no dependency on the real
TIKTOC files). A small set of real-data tests (skipped if the dataset is absent) sanity-check
the loader against the actual pickle -- the full gate (learner/problem counts vs the TIKTOC
paper, structural audit, sampled trajectories) is scripts/p1_03_cw_tuples.py's job.
"""

import pandas as pd
import pytest

from atom.data.cw import cw_stats, default_cw_dir, load_granular
from atom.data.cw_tuples import (
    audit_episode_structure,
    build_cw_events,
    build_episode_table,
    sample_trajectories,
)


def _toy_granular():
    # Learner "s1": episode on problem 10 (2 attempts), then problem 20 (1 attempt).
    # Learner "s2": single attempt on problem 10, solved first try.
    rows = [
        {"SubjectID": "s1", "AssignmentID": 1, "ProblemID": 10, "CodeStateID": "c1",
         "Score": 0.5, "binary_correctness": [1, 0], "ServerTimestamp": "2026-01-01T10:00:00",
         "timestep": 0},
        {"SubjectID": "s1", "AssignmentID": 1, "ProblemID": 10, "CodeStateID": "c2",
         "Score": 1.0, "binary_correctness": [1, 1], "ServerTimestamp": "2026-01-01T10:05:00",
         "timestep": 1},
        {"SubjectID": "s1", "AssignmentID": 1, "ProblemID": 20, "CodeStateID": "c3",
         "Score": 0.75, "binary_correctness": [1, 1, 1, 0], "ServerTimestamp": "2026-01-01T11:00:00",
         "timestep": 2},
        {"SubjectID": "s2", "AssignmentID": 1, "ProblemID": 10, "CodeStateID": "c4",
         "Score": 1.0, "binary_correctness": [1], "ServerTimestamp": "2026-01-01T09:00:00",
         "timestep": 0},
    ]
    df = pd.DataFrame(rows)
    df["AssignmentID"] = df["AssignmentID"].astype("int64")
    df["ServerTimestamp"] = pd.to_datetime(df["ServerTimestamp"])
    return df


def test_cw_stats_on_toy_data():
    stats = cw_stats(_toy_granular())
    assert stats["submissions"] == 4
    assert stats["students"] == 2
    assert stats["problems"] == 2
    assert stats["student_problem_pairs"] == 3  # (s1,10) (s1,20) (s2,10)
    assert stats["exact_solve_rate"] == 0.5  # c2 and c4 are exactly 1.0, out of 4


def test_build_cw_events_episode_and_attempt_idx():
    events = build_cw_events(_toy_granular())
    assert len(events) == 4

    # Recover per-learner rows by re-hashing the original SubjectID (avoids depending on
    # internal row ordering).
    from atom.data.tuples import hash_learner
    h1, h2 = hash_learner("s1"), hash_learner("s2")

    s1_events = events[events["learner"] == h1].sort_values("t")
    assert s1_events["t"].tolist() == [0, 1, 2]
    # First two attempts share an episode (same problem 10), third is a new episode (problem 20).
    ep_ids = s1_events["episode_id"].tolist()
    assert ep_ids[0] == ep_ids[1]
    assert ep_ids[2] != ep_ids[1]
    assert s1_events["attempt_idx"].tolist() == [0, 1, 0]
    assert s1_events["correct"].tolist() == [0, 1, 0]  # c1=0.5, c2=1.0, c3=0.75 -> not exact

    s2_events = events[events["learner"] == h2]
    assert len(s2_events) == 1
    assert s2_events["attempt_idx"].iloc[0] == 0


def test_build_episode_table_final_score_and_solved():
    events = build_cw_events(_toy_granular())
    episodes = build_episode_table(events)
    assert len(episodes) == 3  # (s1,10) (s1,20) (s2,10)

    multi = episodes[episodes["n_attempts"] > 1]
    assert len(multi) == 1
    row = multi.iloc[0]
    assert row["n_attempts"] == 2
    assert row["final_score"] == 1.0  # last attempt (c2) was fully correct
    assert bool(row["solved"])


def test_audit_episode_structure_clean_on_valid_events():
    events = build_cw_events(_toy_granular())
    audit = audit_episode_structure(events)
    assert audit == {
        "episodes_with_multiple_problem_ids": 0,
        "backwards_timestamp_steps": 0,
        "attempt_idx_mismatches": 0,
    }


def test_audit_episode_structure_catches_a_broken_episode():
    """The audit must actually fire on bad input, not just pass on good input -- inject a
    deliberately corrupted attempt_idx and confirm it's caught."""
    events = build_cw_events(_toy_granular())
    broken = events.copy()
    broken.loc[broken.index[1], "attempt_idx"] = 99
    audit = audit_episode_structure(broken)
    assert audit["attempt_idx_mismatches"] == 1


def test_sample_trajectories_is_deterministic_and_prefers_multi_attempt():
    events = build_cw_events(_toy_granular())
    episodes = build_episode_table(events)
    t1 = sample_trajectories(events, episodes, n=1, seed=0)
    t2 = sample_trajectories(events, episodes, n=1, seed=0)
    assert t1 == t2
    assert t1[0]["n_attempts"] == 2  # the only multi-attempt episode in this toy set


pytestmark_real = pytest.mark.skipif(
    not (default_cw_dir() / "dataset_granular_all.zip").exists()
    and not (default_cw_dir() / "dataset_granular_all.pkl").exists(),
    reason="TIKTOC granular dataset not present locally",
)


@pytestmark_real
def test_load_granular_real_data_schema_and_ballpark_counts():
    df = load_granular()
    for col in ["SubjectID", "ProblemID", "CodeStateID", "Score", "binary_correctness",
                "ServerTimestamp"]:
        assert col in df.columns
    assert "Code" not in df.columns  # raw text must be dropped, not just unused
    stats = cw_stats(df)
    assert stats["students"] == 246
    assert stats["problems"] == 17


pytestmark_real_1st = pytest.mark.skipif(
    not (default_cw_dir() / "dataset_granular_1st.zip").exists()
    and not (default_cw_dir() / "dataset_granular_1st.pkl").exists(),
    reason="TIKTOC granular dataset (1st variant) not present locally",
)


@pytestmark_real_1st
def test_load_granular_variant_1st_renames_binary_score_and_loads(tmp_path):
    """Regression test for a real bug found 30 Aug 2026 while scoping the CW-SIM
    TIKTOC-transfer eval: `dataset_granular_1st.pkl`'s per-test-case pass/fail list is
    named `binary_score`, not `binary_correctness` -- `_LIGHT_COLUMNS` hardcoded the
    latter unconditionally, so `load_granular(variant="1st")` raised KeyError before this
    fix, entirely untriggered until now because every existing caller uses the default
    variant="all"."""
    df = load_granular(variant="1st")
    for col in ["SubjectID", "ProblemID", "CodeStateID", "Score", "binary_correctness",
                "ServerTimestamp"]:
        assert col in df.columns
    assert "binary_score" not in df.columns  # renamed, not duplicated
    assert "Code" not in df.columns
    stats = cw_stats(df)
    assert stats["students"] == 246
    assert stats["problems"] == 17
    assert stats["submissions"] == 3714
