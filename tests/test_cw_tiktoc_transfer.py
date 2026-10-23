"""CW-SIM TIKTOC-transfer eval (item 5), the Mac/$0 ingest adapter -- toy-fixture
correctness tests plus real-data schema/canary checks (skipped if the licensed file
isn't present locally, same pattern as `test_cw_data.py`)."""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
if str(REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts"))

from atom.data.cw import default_cw_dir
from atom.data.cw_tiktoc_transfer import (
    build_tiktoc_attempts, build_tiktoc_attempts_all, build_tiktoc_manifest,
    build_tiktoc_manifest_all, build_tiktoc_test_vector_labels,
)


def _write_toy_pkl(tmp_path):
    df = pd.DataFrame([
        {"SubjectID": "s1", "AssignmentID": 10, "ProblemID": 1, "CodeStateID": "cs1",
         "Code": "int a=1;", "prompt": "p1", "Score": 1.0, "binary_score": [1, 1, 0],
         "ServerTimestamp": "2026-01-01T10:00:00", "timestep": 0, "SubjectID_appendix": 1},
        {"SubjectID": "s1", "AssignmentID": 11, "ProblemID": 2, "CodeStateID": "cs2",
         "Code": "int b=2;", "prompt": "p2", "Score": 0.5, "binary_score": [1, 0],
         "ServerTimestamp": "2026-01-01T11:00:00", "timestep": 1, "SubjectID_appendix": 1},
        {"SubjectID": "s2", "AssignmentID": 10, "ProblemID": 1, "CodeStateID": "cs3",
         "Code": "int c=3;", "prompt": "p1", "Score": 0.0, "binary_score": [0, 0, 0],
         "ServerTimestamp": "2026-01-01T09:00:00", "timestep": 0, "SubjectID_appendix": 1},
    ])
    df.to_pickle(tmp_path / "dataset_granular_1st.pkl")
    return tmp_path


def _write_toy_pkl_all(tmp_path):
    """The "_1st" toy fixture's own three targets, PLUS one extra earlier attempt for
    s1/ProblemID=1 (real prior_code/diff should appear at cs1) and a fourth
    (subject,problem) pair absent from "_1st" entirely (must never surface as a target).
    `binary_correctness` is "_all"'s own native raw column name (no rename needed)."""
    df = pd.DataFrame([
        {"SubjectID": "s1", "AssignmentID": 10, "ProblemID": 1, "CodeStateID": "cs1-pre",
         "Code": "int a=0;", "prompt": "p1", "Score": 0.5, "binary_correctness": [1, 0, 0],
         "ServerTimestamp": "2026-01-01T09:50:00", "timestep": 0, "SubjectID_appendix": 1},
        {"SubjectID": "s1", "AssignmentID": 10, "ProblemID": 1, "CodeStateID": "cs1",
         "Code": "int a=1;", "prompt": "p1", "Score": 1.0, "binary_correctness": [1, 1, 0],
         "ServerTimestamp": "2026-01-01T10:00:00", "timestep": 1, "SubjectID_appendix": 1},
        {"SubjectID": "s1", "AssignmentID": 11, "ProblemID": 2, "CodeStateID": "cs2",
         "Code": "int b=2;", "prompt": "p2", "Score": 0.5, "binary_correctness": [1, 0],
         "ServerTimestamp": "2026-01-01T11:00:00", "timestep": 2, "SubjectID_appendix": 1},
        {"SubjectID": "s2", "AssignmentID": 10, "ProblemID": 1, "CodeStateID": "cs3",
         "Code": "int c=3;", "prompt": "p1", "Score": 0.0, "binary_correctness": [0, 0, 0],
         "ServerTimestamp": "2026-01-01T09:00:00", "timestep": 0, "SubjectID_appendix": 1},
        {"SubjectID": "s3", "AssignmentID": 12, "ProblemID": 5, "CodeStateID": "cs4",
         "Code": "int d=4;", "prompt": "p5", "Score": 1.0, "binary_correctness": [1],
         "ServerTimestamp": "2026-01-01T08:00:00", "timestep": 0, "SubjectID_appendix": 1},
    ])
    df.to_pickle(tmp_path / "dataset_granular_all.pkl")
    return tmp_path


def test_build_tiktoc_attempts_matches_cw_events_schema_and_sorts_chronologically(tmp_path):
    d = _write_toy_pkl(tmp_path)
    attempts = build_tiktoc_attempts(base_dir=d)

    expected_columns = {
        "release", "learner", "id", "problem_id", "assignment_id", "code_state_id",
        "timestamp", "score_frac", "correct", "compile_result", "n_compile_errors",
        "t", "code",
    }
    assert expected_columns.issubset(set(attempts.columns))
    assert (attempts["release"] == "TIKTOC").all()
    assert set(attempts["learner"]) == {"tiktoc:s1", "tiktoc:s2"}  # namespaced, not hashed
    # Every row a disclosed proxy: no compile-event data exists in this schema at all.
    assert (attempts["compile_result"] == "Success").all()
    assert (attempts["n_compile_errors"] == 0).all()
    # Chronological within learner, t = 0-based per-learner position.
    s1 = attempts[attempts["learner"] == "tiktoc:s1"].sort_values("t")
    assert list(s1["t"]) == [0, 1]
    assert list(s1["problem_id"]) == [1, 2]  # timestamp order, not input row order
    assert list(s1["code"]) == ["int a=1;", "int b=2;"]  # value-joined by code_state_id
    assert attempts["correct"].tolist() == [1, 0, 0]  # score_frac == 1.0 exactly


def test_build_tiktoc_manifest_shapes_a_cw5_manifest_compatible_dict(tmp_path):
    d = _write_toy_pkl(tmp_path)
    manifest = build_tiktoc_manifest(base_dir=d)

    assert set(manifest) >= {"attempts", "episodes", "val_learners",
                             "n_attempts_before_outcome_filter", "n_dropped_invalid_outcome"}
    attempts = manifest["attempts"]
    # Every row is its own episode (module docstring's disclosed difference 2) -- one
    # row per (learner, problem_id) in this schema, so no two consecutive same-learner
    # rows ever share a problem_id and `sessionize` degenerates correctly.
    assert attempts["episode_id"].nunique() == len(attempts)
    assert "outcome_code" in attempts.columns
    assert (attempts["outcome_code"] >= 0).all()  # -1 would mean an invalid row leaked through
    # `attach_prior_attempt_code` output (2 Sep pilot regression: `build_tiktoc_attempts`
    # does not derive these itself, and their absence is an AttributeError deep inside
    # `build_class_conditioned_examples`, not a NaN-value issue). Every TIKTOC episode is
    # length 1, so this must degenerate to "every row is a first attempt".
    assert "attempt_within_idx" in attempts.columns
    assert (attempts["attempt_within_idx"] == 0).all()
    assert "prior_code" in attempts.columns
    assert attempts["prior_code"].isna().all()
    # No train/val split notion for an external transfer-eval set -- every learner is "val".
    assert manifest["val_learners"] == set(attempts["learner"])
    episodes = manifest["episodes"]
    assert len(episodes) == len(attempts)  # one episode per attempt, confirmed above
    assert set(episodes["episode_id"]) == set(attempts["episode_id"])


def test_build_tiktoc_test_vector_labels_pads_and_masks_correctly(tmp_path):
    d = _write_toy_pkl(tmp_path)
    labels = build_tiktoc_test_vector_labels(base_dir=d, max_test_cases=5)

    assert labels["test_vector"].shape == (3, 5)
    assert labels["valid_mask"].shape == (3, 5)
    assert list(labels["n_cases"]) == [3, 2, 3]
    # Row 0 (cs1): binary_score [1,1,0] -> first 3 slots valid & matching, rest invalid.
    row0 = dict(zip(labels["code_state_id"], range(len(labels["code_state_id"]))))
    i = row0["cs1"]
    assert labels["valid_mask"][i].tolist() == [True, True, True, False, False]
    assert labels["test_vector"][i, :3].tolist() == [True, True, False]
    assert not labels["test_vector"][i, 3:].any()  # padding is False, never garbage
    assert labels["compiled"].all()  # every TIKTOC row ran (it has a real test result)
    assert set(labels["learner"]) == {"tiktoc:s1", "tiktoc:s2"}


def test_build_tiktoc_test_vector_labels_raises_rather_than_truncates_on_overflow(tmp_path):
    d = _write_toy_pkl(tmp_path)
    with pytest.raises(ValueError, match="max_test_cases"):
        build_tiktoc_test_vector_labels(base_dir=d, max_test_cases=2)  # cs1 has 3 cases


def test_build_tiktoc_attempts_all_has_real_multi_attempt_episodes(tmp_path):
    d = _write_toy_pkl_all(tmp_path)
    attempts = build_tiktoc_attempts_all(base_dir=d)

    assert len(attempts) == 5
    s1 = attempts[attempts["learner"] == "tiktoc:s1"].sort_values("t")
    assert list(s1["code_state_id"]) == ["cs1-pre", "cs1", "cs2"]  # chronological, 2 rows on P1
    assert list(s1["problem_id"]) == [1, 1, 2]


def test_build_tiktoc_manifest_all_targets_match_1st_and_restore_real_prior_code(tmp_path):
    d = _write_toy_pkl_all(tmp_path)
    _write_toy_pkl(tmp_path)  # build_tiktoc_manifest_all also reads "_1st" for targets
    manifest = build_tiktoc_manifest_all(base_dir=d)

    # 3 targets in the "_1st" fixture (cs1, cs2, cs3); all 3 are present in "_all" too.
    assert manifest["n_targets_1st"] == 3
    assert manifest["n_targets_found_in_all"] == 3
    assert manifest["n_targets_missing_from_all"] == 0

    attempts = manifest["attempts"]
    target_rows = attempts[attempts["is_target"]]
    assert set(target_rows["code_state_id"]) == {"cs1", "cs2", "cs3"}
    # cs2/cs3 are each their own episode's first attempt (nothing precedes them on that
    # problem in "_all" either) -- cs1 is NOT (see below): the empirical regularity the
    # module docstring describes is not a logical guarantee, and this fixture is built
    # specifically to demonstrate the case where it doesn't hold.
    assert attempts.set_index("code_state_id").loc["cs2", "attempt_within_idx"] == 0
    assert attempts.set_index("code_state_id").loc["cs3", "attempt_within_idx"] == 0

    # cs1's real prior attempt (cs1-pre, same problem, 10 min earlier) is now visible --
    # the whole point of "_all" over "_1st", which forces prior_code=NaN universally.
    # cs1 is consequently attempt_within_idx==1 in its own episode, not 0.
    cs1_row = attempts[attempts["code_state_id"] == "cs1"].iloc[0]
    assert cs1_row["attempt_within_idx"] == 1
    assert cs1_row["prior_code"] == "int a=0;"
    # cs4 (subject s3, problem 5) has no counterpart in the "_1st" fixture at all --
    # must never be marked a target even though it IS that subject's first-ever row.
    cs4_row = attempts[attempts["code_state_id"] == "cs4"].iloc[0]
    assert not cs4_row["is_target"]


def test_build_tiktoc_manifest_all_discloses_1st_targets_absent_from_all(tmp_path):
    d = _write_toy_pkl_all(tmp_path)
    # Drop cs3 from the "_1st" fixture read by build_tiktoc_attempts inside
    # build_tiktoc_manifest_all, simulating the real 21/3714 gap (a "_1st" target with
    # no counterpart anywhere in "_all") -- write a smaller "_1st" pkl at the same path.
    df = pd.DataFrame([
        {"SubjectID": "s1", "AssignmentID": 10, "ProblemID": 1, "CodeStateID": "cs1",
         "Code": "int a=1;", "prompt": "p1", "Score": 1.0, "binary_score": [1, 1, 0],
         "ServerTimestamp": "2026-01-01T10:00:00", "timestep": 0, "SubjectID_appendix": 1},
        {"SubjectID": "s9", "AssignmentID": 99, "ProblemID": 99, "CodeStateID": "cs-nowhere",
         "Code": "int z=9;", "prompt": "p99", "Score": 1.0, "binary_score": [1],
         "ServerTimestamp": "2026-01-01T12:00:00", "timestep": 0, "SubjectID_appendix": 1},
    ])
    df.to_pickle(d / "dataset_granular_1st.pkl")

    manifest = build_tiktoc_manifest_all(base_dir=d)
    assert manifest["n_targets_1st"] == 2
    assert manifest["n_targets_found_in_all"] == 1  # only cs1 exists in "_all"
    assert manifest["n_targets_missing_from_all"] == 1  # cs-nowhere does not


pytestmark_real = pytest.mark.skipif(
    not (default_cw_dir() / "dataset_granular_1st.zip").exists()
    and not (default_cw_dir() / "dataset_granular_1st.pkl").exists(),
    reason="TIKTOC granular dataset (1st variant) not present locally",
)


@pytestmark_real
def test_build_tiktoc_attempts_real_data_schema_and_ballpark_counts():
    attempts = build_tiktoc_attempts()
    assert len(attempts) == 3714
    assert attempts["learner"].nunique() == 246
    assert attempts["problem_id"].nunique() == 17
    # Every episode is length 1 in TIKTOC's first-submission file (module docstring's
    # disclosed difference 2) -- confirmed directly, not assumed.
    assert (attempts.groupby(["learner", "problem_id"]).size() == 1).all()


@pytestmark_real
def test_build_tiktoc_manifest_real_data_all_episodes_length_one_and_all_learners_val():
    manifest = build_tiktoc_manifest()
    attempts = manifest["attempts"]
    assert len(attempts) <= 3714  # <= the raw ingest count; outcome_valid may drop some
    assert (attempts.groupby("episode_id").size() == 1).all()
    assert manifest["val_learners"] == set(attempts["learner"])
    assert manifest["n_dropped_invalid_outcome"] == manifest["n_attempts_before_outcome_filter"] - len(attempts)


@pytestmark_real
def test_build_tiktoc_test_vector_labels_real_data_matches_our_own_headline_test_case_counts():
    """The slot-order canary (research-report risk #4): for every one of the 13 headline
    problems, TIKTOC's own per-row test-case count must match our own
    `cw_5_functional_equivalence.load_test_cases`'s count exactly and be constant across
    every row on that problem -- verified directly 30 Aug 2026, both true for all 13."""
    from cw_5_functional_equivalence import HEADLINE_PROBLEM_IDS, load_test_cases

    labels = build_tiktoc_test_vector_labels()
    problem_id = labels["problem_id"]
    n_cases = labels["n_cases"]
    for pid in HEADLINE_PROBLEM_IDS:
        rows = n_cases[problem_id == pid]
        assert len(rows) > 0
        assert len(set(rows.tolist())) == 1, f"pid={pid} has inconsistent n_cases across rows"
        assert int(rows[0]) == len(load_test_cases(pid)), f"pid={pid} slot-count mismatch"


pytestmark_real_all = pytest.mark.skipif(
    not (default_cw_dir() / "dataset_granular_all.zip").exists()
    and not (default_cw_dir() / "dataset_granular_all.pkl").exists(),
    reason="TIKTOC granular dataset (all variant) not present locally",
)


@pytestmark_real_all
def test_build_tiktoc_attempts_all_real_data_schema_and_ballpark_counts():
    attempts = build_tiktoc_attempts_all()
    assert len(attempts) == 13083
    assert attempts["learner"].nunique() == 246
    assert attempts["problem_id"].nunique() == 17
    # Unlike "_1st" (always exactly 1), real multi-attempt (subject,problem) groups exist.
    assert attempts.groupby(["learner", "problem_id"]).size().mean() > 1


@pytestmark_real
@pytestmark_real_all
def test_build_tiktoc_manifest_all_real_data_target_gap_matches_measured_3sep_value():
    """Verified directly 3 Sep 2026 (see module docstring for `build_tiktoc_manifest_all`):
    21/3,714 "_1st" targets (2 pairs entirely absent from "_all", 19 with a
    "_1st"-designated CodeStateID that appears nowhere in "_all") are unscoreable under
    the granular_all history. A regression here means either file changed underfoot --
    investigate, do not just update the expected numbers."""
    manifest = build_tiktoc_manifest_all()
    assert manifest["n_targets_1st"] == 3714
    assert manifest["n_targets_found_in_all"] == 3693
    assert manifest["n_targets_missing_from_all"] == 21
    attempts = manifest["attempts"]
    assert (attempts.loc[attempts["is_target"], "attempt_within_idx"] == 0).all()
    # The real multi-attempt gain over "_1st": some targets now have genuine prior_code
    # from an earlier same-problem attempt in their own episode (episode boundaries can
    # still put the target itself first even with real history elsewhere in the manifest).
    assert attempts["prior_code"].notna().any()
