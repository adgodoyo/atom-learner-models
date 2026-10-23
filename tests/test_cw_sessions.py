"""Tests for src/atom/data/cw_sessions.py (CW-2 sessionisation, episode targets, splits).

Synthetic tests build a toy `attempts`-shaped DataFrame directly (the schema
`cw_events.build_attempts` produces) and exercise sessionisation, episode targets,
censoring, the compile<->score reconciliation, and the item split. A small real-data
smoke test (skipped if the raw CodeWorkout zips are absent) sanity-checks against S19/F19 --
the full gate is scripts/cw_2_sessions.py's job.
"""

import pandas as pd
import pytest

from atom.data.cw_events import build_attempts, default_cw_raw_dir, load_raw_tables
from atom.data.cw_sessions import (
    audit_episode_partition,
    build_episode_table,
    make_item_split,
    reconcile_compile_score,
    same_problem_gaps,
    sessionize,
    tiktoc_stratum_ids,
)


def _toy_attempts() -> pd.DataFrame:
    """Learner 'a': problem 10 twice 5 min apart (same session under tau=60), then problem
    10 again 200 min later (a RETURN -- new episode under tau=60, still same episode under
    tau=300); problem 20 once, right after (a genuine problem switch -- always a new
    episode). Learner 'b': one attempt, the release's only event for 'b' (so 'b's single
    episode must be right-censored)."""
    rows = [
        # learner a, episode 1 candidate: problem 10, two attempts 5 min apart
        {"learner": "a", "id": "e1", "problem_id": 10, "timestamp": "2026-01-01T10:00:00",
         "score_frac": 0.0, "correct": 0, "n_compile_errors": 1, "release": "TOY"},
        {"learner": "a", "id": "e2", "problem_id": 10, "timestamp": "2026-01-01T10:05:00",
         "score_frac": 1.0, "correct": 1, "n_compile_errors": 0, "release": "TOY"},
        # learner a, 200 min later, SAME problem 10 (a return within tau=300, a new episode under tau=60)
        {"learner": "a", "id": "e3", "problem_id": 10, "timestamp": "2026-01-01T13:25:00",
         "score_frac": 0.5, "correct": 0, "n_compile_errors": 0, "release": "TOY"},
        # learner a, immediately after, problem 20 (problem switch -- ALWAYS a new episode)
        {"learner": "a", "id": "e4", "problem_id": 20, "timestamp": "2026-01-01T13:26:00",
         "score_frac": 1.0, "correct": 1, "n_compile_errors": 0, "release": "TOY"},
        # learner b, one attempt, the release's last event overall for b
        {"learner": "b", "id": "e5", "problem_id": 10, "timestamp": "2026-01-01T09:00:00",
         "score_frac": 1.0, "correct": 1, "n_compile_errors": 0, "release": "TOY"},
    ]
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def test_sessionize_boundaries_gap_problem_and_learner():
    attempts = _toy_attempts()
    sess = sessionize(attempts, tau_session_minutes=60.0)
    by_id = sess.set_index("id")["episode_id"]
    # e1, e2 same episode (5 min apart, same problem)
    assert by_id["e1"] == by_id["e2"]
    # e3 is 200 min after e2 on the SAME problem -> new episode under tau=60
    assert by_id["e3"] != by_id["e2"]
    # e4 is a problem switch -> always a new episode, regardless of gap
    assert by_id["e4"] != by_id["e3"]
    # e5 (learner b) is its own episode
    assert by_id["e5"] not in {by_id["e1"], by_id["e3"], by_id["e4"]}


def test_sessionize_larger_tau_merges_the_return_episode():
    attempts = _toy_attempts()
    sess = sessionize(attempts, tau_session_minutes=300.0)
    by_id = sess.set_index("id")["episode_id"]
    # under tau=300, e3's 200-minute gap no longer crosses the boundary
    assert by_id["e1"] == by_id["e2"] == by_id["e3"]
    # a problem switch is still always a boundary, independent of tau
    assert by_id["e4"] != by_id["e3"]


def test_sessionize_is_deterministic():
    attempts = _toy_attempts()
    s1 = sessionize(attempts, 60.0)["episode_id"]
    s2 = sessionize(attempts, 60.0)["episode_id"]
    assert s1.equals(s2)


def test_audit_episode_partition_clean_on_valid_sessionisation():
    attempts = _toy_attempts()
    sess = sessionize(attempts, 60.0)
    audit = audit_episode_partition(sess)
    assert audit == {
        "episodes_with_multiple_problem_ids": 0,
        "episodes_with_multiple_learners": 0,
        "backwards_timestamp_steps": 0,
        "attempts_not_covered_exactly_once": 0,
    }


def test_audit_episode_partition_catches_a_cross_problem_episode():
    attempts = _toy_attempts()
    sess = sessionize(attempts, 60.0)
    broken = sess.copy()
    # force e4 into e3's episode, creating a two-problem episode
    e3_episode = broken.loc[broken["id"] == "e3", "episode_id"].iloc[0]
    broken.loc[broken["id"] == "e4", "episode_id"] = e3_episode
    audit = audit_episode_partition(broken)
    assert audit["episodes_with_multiple_problem_ids"] == 1


def test_build_episode_table_targets_and_effort():
    attempts = _toy_attempts()
    sess = sessionize(attempts, 60.0)
    episodes = build_episode_table(sess)
    assert len(episodes) == 4  # (a,10,ep1) (a,10,ep2) (a,20,ep3) (b,10,ep4)

    ep1 = episodes[(episodes["learner"] == "a") & (episodes["problem_id"] == 10)
                   & (episodes["n_attempts"] == 2)].iloc[0]
    assert not ep1["y_first"]   # e1: score 0.0
    assert ep1["y_ever"]        # e2: score 1.0
    assert ep1["y_terminal"]    # last attempt (e2) is correct
    assert ep1["first_score"] == 0.0
    assert ep1["best_score"] == 1.0
    assert ep1["terminal_score"] == 1.0
    assert ep1["n_compile_errors_total"] == 1
    assert ep1["time_to_first_success"] == 1  # e2 is attempt index 1 within the episode
    assert ep1["duration_seconds"] == 300.0   # 5 minutes

    ep_return = episodes[(episodes["learner"] == "a") & (episodes["problem_id"] == 10)
                          & (episodes["n_attempts"] == 1)].iloc[0]
    assert not ep_return["y_ever"]  # e3: score 0.5, never solved
    assert pd.isna(ep_return["time_to_first_success"])


def test_build_episode_table_right_censoring():
    attempts = _toy_attempts()
    sess = sessionize(attempts, 60.0)
    episodes = build_episode_table(sess)

    # learner b's only episode ends at b's only (hence last) event -> censored
    ep_b = episodes[episodes["learner"] == "b"].iloc[0]
    assert ep_b["is_right_censored"]

    # learner a's problem-10 first episode (e1,e2) is NOT a's last event -> not censored
    ep_a_first = episodes[(episodes["learner"] == "a") & (episodes["n_attempts"] == 2)].iloc[0]
    assert not ep_a_first["is_right_censored"]

    # learner a's LAST episode (problem 20, e4) IS a's last event -> censored
    ep_a_last = episodes[(episodes["learner"] == "a") & (episodes["problem_id"] == 20)].iloc[0]
    assert ep_a_last["is_right_censored"]


def test_same_problem_gaps_excludes_problem_switches_and_cross_learner_pairs():
    attempts = _toy_attempts()
    gaps = same_problem_gaps(attempts)
    # only e1->e2 (5 min) and e2->e3 (200 min) qualify -- e3->e4 is a problem switch,
    # e4->e5/e5-> anything crosses release-sort but also differs in learner or problem.
    assert sorted(round(g, 1) for g in gaps) == [5.0, 200.0]


def test_reconcile_compile_score_arithmetic():
    attempts = _toy_attempts()
    r = reconcile_compile_score(attempts)
    assert r["n_attempts"] == 5
    assert r["n_compile_error_attempts"] == 1  # e1
    assert r["compile_error_implies_zero_score_rate"] == 1.0  # e1 has score 0.0
    # zero scores: only e1 (score 0.0); e1 has a compile error -> 100% explained
    assert r["n_zero_score"] == 1
    assert r["zero_score_explained_by_compile_error_frac"] == 1.0
    # clean-compile attempts: e2 (1.0), e3 (0.5), e4 (1.0), e5 (1.0) -> n_clean=4
    assert r["n_clean_compile"] == 4
    assert r["clean_compile_zero_n"] == 0
    assert r["clean_compile_full_n"] == 3   # e2, e4, e5
    assert r["clean_compile_partial_n"] == 1  # e3


def test_make_item_split_is_deterministic_and_covers_every_id():
    ids = list(range(1, 51))
    s1 = make_item_split(ids, heldout_frac=0.15, seed=0)
    s2 = make_item_split(ids, heldout_frac=0.15, seed=0)
    assert s1.equals(s2)
    assert set(s1["problem_id"]) == set(ids)
    assert int((s1["item_split"] == "heldout").sum()) == round(50 * 0.15)


def test_make_item_split_different_seed_gives_a_different_partition():
    ids = list(range(1, 51))
    s1 = make_item_split(ids, heldout_frac=0.15, seed=0)
    s2 = make_item_split(ids, heldout_frac=0.15, seed=1)
    assert not s1["item_split"].equals(s2["item_split"])


pytestmark_real = pytest.mark.skipif(
    not (default_cw_raw_dir() / "S19_All_Release_2_10_22.zip").exists(),
    reason="Raw CodeWorkout S19/F19 releases not present locally",
)


@pytestmark_real
def test_real_sessionisation_partition_is_clean_on_both_releases():
    frames = []
    for release in ["S19", "F19"]:
        main, _ = load_raw_tables(release)
        frames.append(build_attempts(main))
    attempts = pd.concat(frames, ignore_index=True)

    sess = sessionize(attempts, tau_session_minutes=60.0)
    audit = audit_episode_partition(sess)
    assert audit == {
        "episodes_with_multiple_problem_ids": 0,
        "episodes_with_multiple_learners": 0,
        "backwards_timestamp_steps": 0,
        "attempts_not_covered_exactly_once": 0,
    }


@pytestmark_real
def test_real_item_bank_is_identical_across_releases():
    ids_by_release = {}
    for release in ["S19", "F19"]:
        main, _ = load_raw_tables(release)
        attempts = build_attempts(main)
        ids_by_release[release] = set(attempts["problem_id"].unique())
    assert ids_by_release["S19"] == ids_by_release["F19"]

    tiktoc = tiktoc_stratum_ids()
    assert set(tiktoc) <= ids_by_release["S19"]
    assert len(tiktoc) == 17
