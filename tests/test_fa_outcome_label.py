"""FA-6c -- tests for the 5-class hierarchical outcome label."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from atom.data.fa_outcome_label import (
    N_OUTCOME, OUTCOME_CLASSES, outcome_label, outcome_label_report,
)


def _series(support_code, score, score_missing) -> tuple[pd.Series, pd.Series, pd.Series]:
    return pd.Series(support_code), pd.Series(score, dtype="float64"), pd.Series(score_missing)


def test_unaided_correct_and_wrong_split_by_score():
    sc, s, sm = _series([0, 0], [1, 0], [False, False])
    out = outcome_label(sc, s, sm)
    assert out["outcome_code"].tolist() == [0, 1]  # UNAIDED_CORRECT, UNAIDED_WRONG
    assert out["outcome_valid"].tolist() == [True, True]


def test_help_classes_absorb_regardless_of_score():
    """support_code 1/2/3 map to outcome 2/3/4 REGARDLESS of score -- the '<2%
    help-but-credited' rows absorb into their help class, never split further."""
    sc, s, sm = _series([1, 1, 2, 2, 3, 3], [0, 1, 0, 1, 0, 1], [False] * 6)
    out = outcome_label(sc, s, sm)
    assert out["outcome_code"].tolist() == [2, 2, 3, 3, 4, 4]
    assert out["outcome_valid"].tolist() == [True] * 6


def test_help_classes_valid_even_when_score_missing():
    """Score is irrelevant to a help-class label, so a missing score there must NOT
    invalidate the row -- only unaided rows depend on knowing the score."""
    sc, s, sm = _series([1, 2, 3], [np.nan, np.nan, np.nan], [True, True, True])
    out = outcome_label(sc, s, sm)
    assert out["outcome_code"].tolist() == [2, 3, 4]
    assert out["outcome_valid"].tolist() == [True, True, True]


def test_unaided_with_missing_score_is_invalid_not_defaulted():
    sc, s, sm = _series([0, 0], [np.nan, 1], [True, False])
    out = outcome_label(sc, s, sm)
    assert out["outcome_valid"].tolist() == [False, True]
    assert out["outcome_code"].iloc[0] == -1  # never silently defaulted to a real class
    assert out["outcome_code"].iloc[1] == 0


def test_outcome_classes_has_five_entries_in_the_documented_order():
    assert OUTCOME_CLASSES == (
        "UNAIDED_CORRECT", "UNAIDED_WRONG", "HINT_ONLY", "REVEAL_ONLY", "HINT_AND_REVEAL",
    )
    assert N_OUTCOME == 5


def test_outcome_label_report_discloses_help_credited_mass():
    sc, s, sm = _series([0, 0, 1, 1, 2], [1, 0, 0, 1, 0], [False] * 5)
    report = outcome_label_report(sc, s, sm)
    assert report["n_total"] == 5
    assert report["n_valid"] == 5
    assert report["n_help_total"] == 3  # two support_code=1, one support_code=2
    assert report["n_help_credited"] == 1  # the support_code=1, score=1 row
    assert report["help_credited_frac"] == pytest.approx(1 / 3)
    assert report["class_counts"]["UNAIDED_CORRECT"] == 1
    assert report["class_counts"]["UNAIDED_WRONG"] == 1
    assert report["class_counts"]["HINT_ONLY"] == 2
    assert report["class_counts"]["REVEAL_ONLY"] == 1
    assert report["class_counts"]["HINT_AND_REVEAL"] == 0


def test_outcome_label_report_counts_invalid_rows():
    sc, s, sm = _series([0, 0, 1], [np.nan, 1, np.nan], [True, False, True])
    report = outcome_label_report(sc, s, sm)
    assert report["n_invalid_unaided_score_missing"] == 1
    assert report["n_valid"] == 2
