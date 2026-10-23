"""Tests for src/atom/data/cw_events.py (CW-1 full-release event-graph ingest).

Synthetic tests exercise the join/assertion/episode logic directly, including the exact
F19 float-suffix ParentEventID defect the design doc warns about. A small set of real-data
tests (skipped if the raw CodeWorkout zips are absent) sanity-check the loader against the actual
S19/F19 releases -- the full gate is scripts/cw_1_event_graph.py's job.
"""

import pandas as pd
import pytest

from atom.data.cw_events import (
    assert_event_graph,
    build_attempts,
    build_problem_episodes,
    canonicalize_parent_id,
    default_cw_raw_dir,
    load_raw_tables,
    release_stats,
    sample_stratified_episodes,
)


def _toy_main_raw(release: str, float_suffix: bool) -> pd.DataFrame:
    """One learner, two attempts on problem 10 (one compile error then success), one
    attempt on problem 20 (clean first try). `float_suffix=True` reproduces F19's
    ParentEventID formatting defect ('10.0' instead of '10') on every parent link, left
    UNCANONICALISED (as it would arrive straight off a `pd.read_csv`) -- callers that want
    the canonicalised, joinable form should use `_toy_main` below.
    """
    def pid(x):
        return f"{x}.0" if float_suffix else str(x)

    rows = [
        # attempt 1: run -> compile -> one compile error, Score 0.0
        {"SubjectID": "s1", "AssignmentID": "1", "ProblemID": 10, "CodeStateID": "c1",
         "EventType": "Run.Program", "Score": "0.0", "Compile.Result": None,
         "CompileMessageType": None, "EventID": "10", "ParentEventID": None,
         "ServerTimestamp": "2026-01-01T10:00:00"},
        {"SubjectID": "s1", "AssignmentID": "1", "ProblemID": 10, "CodeStateID": "c1",
         "EventType": "Compile", "Score": None, "Compile.Result": "Error",
         "CompileMessageType": None, "EventID": "11", "ParentEventID": pid(10),
         "ServerTimestamp": "2026-01-01T10:00:00"},
        {"SubjectID": "s1", "AssignmentID": "1", "ProblemID": 10, "CodeStateID": "c1",
         "EventType": "Compile.Error", "Score": None, "Compile.Result": None,
         "CompileMessageType": "SyntaxError", "EventID": "12", "ParentEventID": pid(11),
         "ServerTimestamp": "2026-01-01T10:00:00"},
        # attempt 2: run -> compile, clean, Score 1.0
        {"SubjectID": "s1", "AssignmentID": "1", "ProblemID": 10, "CodeStateID": "c2",
         "EventType": "Run.Program", "Score": "1.0", "Compile.Result": None,
         "CompileMessageType": None, "EventID": "20", "ParentEventID": None,
         "ServerTimestamp": "2026-01-01T10:05:00"},
        {"SubjectID": "s1", "AssignmentID": "1", "ProblemID": 10, "CodeStateID": "c2",
         "EventType": "Compile", "Score": None, "Compile.Result": "Success",
         "CompileMessageType": None, "EventID": "21", "ParentEventID": pid(20),
         "ServerTimestamp": "2026-01-01T10:05:00"},
        # attempt 3: new problem, clean first try, Score 0.75 (partial credit)
        {"SubjectID": "s1", "AssignmentID": "1", "ProblemID": 20, "CodeStateID": "c3",
         "EventType": "Run.Program", "Score": "0.75", "Compile.Result": None,
         "CompileMessageType": None, "EventID": "30", "ParentEventID": None,
         "ServerTimestamp": "2026-01-01T11:00:00"},
        {"SubjectID": "s1", "AssignmentID": "1", "ProblemID": 20, "CodeStateID": "c3",
         "EventType": "Compile", "Score": None, "Compile.Result": "Success",
         "CompileMessageType": None, "EventID": "31", "ParentEventID": pid(30),
         "ServerTimestamp": "2026-01-01T11:00:00"},
    ]
    df = pd.DataFrame(rows).astype(str).replace("None", pd.NA)
    df["ServerTimestamp"] = pd.to_datetime(df["ServerTimestamp"])
    df["release"] = release
    return df


def _toy_main(release: str, float_suffix: bool) -> pd.DataFrame:
    """The canonicalised, joinable form of `_toy_main_raw` -- what `load_raw_tables` would
    hand back after its own `canonicalize_parent_id` pass. Regardless of `float_suffix`,
    the joins in this frame are clean."""
    df = _toy_main_raw(release, float_suffix)
    df["ParentEventID"] = canonicalize_parent_id(df["ParentEventID"])
    return df


def test_canonicalize_parent_id_strips_float_suffix_only():
    s = pd.Series(["11161.0", "13-54894", pd.NA])
    out = canonicalize_parent_id(s)
    assert out.tolist()[:2] == ["11161", "13-54894"]
    assert pd.isna(out.iloc[2])


def test_assert_event_graph_clean_on_valid_synthetic_data():
    main = _toy_main("TOY", float_suffix=False)
    audit = assert_event_graph(main)
    assert audit["compiles_with_bad_parent"] == 0
    assert audit["runs_with_not_exactly_one_compile"] == 0
    assert audit["errors_with_bad_parent"] == 0
    assert audit["n_runs"] == 3 and audit["n_compiles"] == 3 and audit["n_errors"] == 1


def test_assert_event_graph_fires_on_the_f19_float_suffix_defect_if_uncanonicalized():
    """Reproduces the design doc's exact warning: without canonicalisation, F19-style
    ParentEventID formatting ('10.0') breaks every join."""
    main = _toy_main_raw("TOY", float_suffix=True)
    audit = assert_event_graph(main)
    assert audit["compiles_with_bad_parent"] == 3
    assert audit["errors_with_bad_parent"] == 1

    # The canonicalised form of the same data joins cleanly.
    canon = _toy_main("TOY", float_suffix=True)
    audit_canon = assert_event_graph(canon)
    assert audit_canon["compiles_with_bad_parent"] == 0
    assert audit_canon["errors_with_bad_parent"] == 0


def test_assert_event_graph_catches_a_run_with_two_compiles():
    main = _toy_main("TOY", float_suffix=False)
    extra_compile = main[main["EventID"] == "11"].copy()
    extra_compile["EventID"] = "13"
    broken = pd.concat([main, extra_compile], ignore_index=True)
    audit = assert_event_graph(broken)
    assert audit["runs_with_not_exactly_one_compile"] == 1


def test_build_attempts_join_and_score_derivation():
    main = _toy_main("TOY", float_suffix=False)
    attempts = build_attempts(main)
    assert len(attempts) == 3
    assert set(attempts["problem_id"]) == {10, 20}
    # attempt on c1: one compile error, not correct (Score 0.0)
    a1 = attempts[attempts["code_state_id"] == "c1"].iloc[0]
    assert a1["n_compile_errors"] == 1
    assert a1["correct"] == 0
    assert a1["compile_result"] == "Error"
    # attempt on c2: clean compile, correct (Score 1.0)
    a2 = attempts[attempts["code_state_id"] == "c2"].iloc[0]
    assert a2["n_compile_errors"] == 0
    assert a2["correct"] == 1
    assert a2["compile_result"] == "Success"
    # chronological ordering within learner
    assert attempts.sort_values("t")["t"].tolist() == [0, 1, 2]


def test_build_attempts_code_join_survives_the_chronological_sort():
    """Regression test: code text must be joined via the POST-sort code_state_id column,
    not the pre-sort `runs` row order -- otherwise sorting misaligns code with attempts."""
    main = _toy_main("TOY", float_suffix=False)
    code_states = pd.DataFrame({
        "CodeStateID": ["c1", "c2", "c3"],
        "Code": ["CODE_C1", "CODE_C2", "CODE_C3"],
    })
    attempts = build_attempts(main, code_states=code_states)
    for _, row in attempts.iterrows():
        assert row["code"] == f"CODE_{row['code_state_id'].upper()}"


def test_release_stats_matches_expected_toy_counts():
    main = _toy_main("TOY", float_suffix=False)
    attempts = build_attempts(main)
    stats = release_stats(main, attempts)
    assert stats == {
        "raw_rows": 7, "attempts": 3, "compiles": 3, "compile_errors": 1,
        "learners": 1, "problems": 2,
    }


def test_build_problem_episodes_boundaries_and_strata_flags():
    main = _toy_main("TOY", float_suffix=False)
    attempts = build_attempts(main)
    attempts_ep, episodes = build_problem_episodes(attempts)
    assert len(episodes) == 2  # (s1,10) two attempts, (s1,20) one attempt
    ep10 = episodes[episodes["problem_id"] == 10].iloc[0]
    assert ep10["n_attempts"] == 2
    assert bool(ep10["solved"])  # final score 1.0
    assert not bool(ep10["has_regression"])  # scores 0.0 -> 1.0, monotonic
    ep20 = episodes[episodes["problem_id"] == 20].iloc[0]
    assert ep20["n_attempts"] == 1
    assert not bool(ep20["solved"])  # final score 0.75


def test_sample_stratified_episodes_is_deterministic():
    main = _toy_main("TOY", float_suffix=False)
    attempts = build_attempts(main)
    attempts_ep, episodes = build_problem_episodes(attempts)
    s1 = sample_stratified_episodes(attempts_ep, episodes, n=10, seed=0)
    s2 = sample_stratified_episodes(attempts_ep, episodes, n=10, seed=0)
    assert s1 == s2
    assert len(s1) == 2  # only 2 episodes exist in this toy set


pytestmark_real = pytest.mark.skipif(
    not (default_cw_raw_dir() / "S19_All_Release_2_10_22.zip").exists(),
    reason="Raw CodeWorkout S19/F19 releases not present locally",
)


@pytestmark_real
@pytest.mark.parametrize("release,expected", [
    ("S19", {"raw_rows": 201570, "attempts": 69627, "compiles": 69627,
             "compile_errors": 62316, "learners": 413, "problems": 50}),
    ("F19", {"raw_rows": 360176, "attempts": 125578, "compiles": 125578,
             "compile_errors": 109020, "learners": 506, "problems": 50}),
])
def test_real_release_join_is_clean_and_counts_match_design_doc(release, expected):
    main, _ = load_raw_tables(release)
    audit = assert_event_graph(main)
    assert audit["compiles_with_bad_parent"] == 0
    assert audit["runs_with_not_exactly_one_compile"] == 0
    assert audit["errors_with_bad_parent"] == 0

    attempts = build_attempts(main)
    assert release_stats(main, attempts) == expected
