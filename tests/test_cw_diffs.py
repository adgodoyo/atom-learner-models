"""Tests for src/atom/data/cw_diffs.py (CW-3 episode-local code-diff construction)."""

import numpy as np
import pandas as pd
import pytest

from atom.data.cw_diffs import build_code_diffs, unified_diff_text


def _toy_attempts_with_episode() -> pd.DataFrame:
    """episode 'e1' (learner a, problem 10): three attempts, incremental edits.
    episode 'e2' (learner a, problem 10, a RETURN episode after a big gap): one
    attempt -- must get NO previous-attempt row even though 'e1' has code on the
    same problem, because diffs are episode-local. episode 'e3' (learner b): two
    attempts, the second identical to the first (a genuine no-edit resubmit)."""
    rows = [
        {"episode_id": "e1", "code_state_id": "c1", "timestamp": "2026-01-01T10:00:00",
         "code": "int x = 0;"},
        {"episode_id": "e1", "code_state_id": "c2", "timestamp": "2026-01-01T10:05:00",
         "code": "int x = 0;\nint y = 1;"},
        {"episode_id": "e1", "code_state_id": "c3", "timestamp": "2026-01-01T10:10:00",
         "code": "int x = 0;\nint y = 2;"},
        {"episode_id": "e2", "code_state_id": "c4", "timestamp": "2026-01-02T09:00:00",
         "code": "int x = 0;\nint y = 2;\nreturn y;"},
        {"episode_id": "e3", "code_state_id": "c5", "timestamp": "2026-01-01T08:00:00",
         "code": "print('hi')"},
        {"episode_id": "e3", "code_state_id": "c6", "timestamp": "2026-01-01T08:01:00",
         "code": "print('hi')"},
    ]
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def test_unified_diff_text_reflects_the_edit():
    d = unified_diff_text("int x = 0;", "int x = 0;\nint y = 1;")
    assert "+int y = 1;" in d
    removed = [ln for ln in d.splitlines() if ln.startswith("-") and not ln.startswith("---")]
    assert removed == []  # no line removed, only added
    assert not d.startswith("---")  # header pair stripped


def test_unified_diff_text_empty_for_identical_code():
    assert unified_diff_text("print('hi')", "print('hi')") == ""


def test_build_code_diffs_excludes_episode_first_attempts():
    attempts = _toy_attempts_with_episode()
    diffs = build_code_diffs(attempts)
    # c1 (first of e1), c4 (first, and ONLY, attempt of e2), c5 (first of e3) excluded
    assert set(diffs["code_state_id"]) == {"c2", "c3", "c6"}


def test_build_code_diffs_return_episode_gets_no_previous_row_despite_same_problem():
    """e2's single attempt (c4) has real prior code on the same problem (from e1's c3),
    but diffs are episode-local -- c4 must NOT appear as a diffed row at all."""
    attempts = _toy_attempts_with_episode()
    diffs = build_code_diffs(attempts)
    assert "c4" not in set(diffs["code_state_id"])


def test_build_code_diffs_prev_id_is_correct_within_episode():
    attempts = _toy_attempts_with_episode()
    diffs = build_code_diffs(attempts)
    by_id = diffs.set_index("code_state_id")
    assert by_id.loc["c2", "prev_code_state_id"] == "c1"
    assert by_id.loc["c3", "prev_code_state_id"] == "c2"
    assert by_id.loc["c6", "prev_code_state_id"] == "c5"


def test_build_code_diffs_diff_text_matches_direct_call():
    attempts = _toy_attempts_with_episode()
    diffs = build_code_diffs(attempts)
    c3_diff = diffs.set_index("code_state_id").loc["c3", "diff_text"]
    assert c3_diff == unified_diff_text("int x = 0;\nint y = 1;", "int x = 0;\nint y = 2;")


def test_build_code_diffs_flags_empty_diff_on_identical_resubmit():
    attempts = _toy_attempts_with_episode()
    diffs = build_code_diffs(attempts)
    c6 = diffs.set_index("code_state_id").loc["c6"]
    assert c6["is_empty_diff"]
    assert c6["diff_text"] == ""


def test_build_code_diffs_requires_episode_id_and_code_columns():
    bad = pd.DataFrame({"timestamp": [1], "code_state_id": ["c1"]})
    with pytest.raises(ValueError, match="missing required columns"):
        build_code_diffs(bad)


def test_build_code_diffs_skips_rows_with_missing_or_empty_code():
    """Regression test for the 11 Aug real-data finding (10/195,205 attempts in the full
    corpus have NaN or empty code, most likely an unjoined CodeStateID): such a row must
    never appear as either the current or the previous member of a diff pair. c2 here has
    NaN code -- c3 (which follows it) must diff against c1 (skipping over c2 as if
    absent), not crash and not fabricate a diff against NaN."""
    rows = [
        {"episode_id": "e1", "code_state_id": "c1", "timestamp": "2026-01-01T10:00:00",
         "code": "int x = 0;"},
        {"episode_id": "e1", "code_state_id": "c2", "timestamp": "2026-01-01T10:05:00",
         "code": np.nan},
        {"episode_id": "e1", "code_state_id": "c3", "timestamp": "2026-01-01T10:10:00",
         "code": "int x = 0;\nint y = 2;"},
        {"episode_id": "e1", "code_state_id": "c4", "timestamp": "2026-01-01T10:15:00",
         "code": "   "},  # whitespace-only counts as empty
    ]
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    diffs = build_code_diffs(df)
    assert set(diffs["code_state_id"]) == {"c3"}  # only c3 has both real code and a real predecessor
    assert diffs.set_index("code_state_id").loc["c3", "prev_code_state_id"] == "c1"


def test_build_code_diffs_deterministic():
    attempts = _toy_attempts_with_episode()
    d1 = build_code_diffs(attempts)
    d2 = build_code_diffs(attempts)
    pd.testing.assert_frame_equal(d1, d2)
