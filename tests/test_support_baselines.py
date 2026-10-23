"""Support-label baselines: learner-prefix (shrinkage) and lag-1+gap. Synthetic manifests
only -- these tests exist to prove the causal/shrinkage/fallback arithmetic is right, not
to reproduce FA's real numbers (that is scripts/fa_06_support_baselines.py's job, run
directly against the FA manifest, no model needed)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from atom.eval.metrics import multiclass_brier, multiclass_nll
from atom.eval.support_baselines import (FIRST_EVENT_BUCKET, N_SUPPORT,
                                         fit_global_support_freq, fit_lag1_gap_table,
                                         gap_bucket, lag1_gap_probs, learner_prefix_probs)


def _toy_ev(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    return df.sort_values(["learner", "t"], kind="mergesort").reset_index(drop=True)


def test_fit_global_support_freq_sums_to_one_and_matches_counts():
    ev = _toy_ev([{"learner": "a", "t": 0, "support_code": 0},
                 {"learner": "a", "t": 1, "support_code": 0},
                 {"learner": "b", "t": 0, "support_code": 3}])
    freq = fit_global_support_freq(ev)
    assert freq.shape == (N_SUPPORT,)
    assert math.isclose(freq.sum(), 1.0)
    assert math.isclose(freq[0], 2 / 3)
    assert math.isclose(freq[3], 1 / 3)


def test_fit_global_support_freq_raises_on_empty_input():
    with pytest.raises(ValueError):
        fit_global_support_freq(pd.DataFrame({"support_code": []}))


def test_learner_prefix_probs_first_event_falls_back_to_global_freq_exactly():
    ev = _toy_ev([{"learner": "a", "t": 0, "support_code": 2},
                 {"learner": "b", "t": 0, "support_code": 1}])
    global_freq = np.array([0.1, 0.2, 0.3, 0.4])
    probs = learner_prefix_probs(ev, global_freq, shrinkage_k=5.0)
    # both rows are each learner's own first event -> n_prior=0 -> exactly global_freq.
    np.testing.assert_allclose(probs[0], global_freq)
    np.testing.assert_allclose(probs[1], global_freq)


def test_learner_prefix_probs_never_uses_the_rows_own_label():
    """A learner who has said support_code=0 every single time so far must not have
    their NEXT prediction informed by that next row's own (different) label."""
    ev = _toy_ev([{"learner": "a", "t": 0, "support_code": 0},
                 {"learner": "a", "t": 1, "support_code": 0},
                 {"learner": "a", "t": 2, "support_code": 3}])  # label differs, must not leak
    global_freq = np.full(N_SUPPORT, 1 / N_SUPPORT)
    probs = learner_prefix_probs(ev, global_freq, shrinkage_k=0.001)
    # row 2's prediction is built ONLY from rows 0-1 (both class 0) -> heavily class-0,
    # regardless of row 2's own class-3 label.
    assert probs[2][0] > 0.9


def test_learner_prefix_probs_rows_are_valid_simplices():
    ev = _toy_ev([{"learner": "a", "t": t, "support_code": t % N_SUPPORT} for t in range(6)])
    global_freq = fit_global_support_freq(ev)
    probs = learner_prefix_probs(ev, global_freq, shrinkage_k=3.0)
    np.testing.assert_allclose(probs.sum(axis=1), np.ones(len(ev)), rtol=1e-9)
    assert (probs >= 0).all()


def test_learner_prefix_probs_rejects_nonpositive_shrinkage():
    ev = _toy_ev([{"learner": "a", "t": 0, "support_code": 0}])
    with pytest.raises(ValueError):
        learner_prefix_probs(ev, np.full(N_SUPPORT, 0.25), shrinkage_k=0.0)


def test_gap_bucket_first_event_is_sentinel_regardless_of_nan_gap():
    first = np.array([True, False, False])
    gap_log1p = np.array([np.nan, np.log1p(2.0), np.log1p(2000.0)])
    buckets = gap_bucket(gap_log1p, first)
    assert buckets[0] == FIRST_EVENT_BUCKET
    assert buckets[1] < buckets[2]  # a short gap buckets below a long one


def test_fit_lag1_gap_table_unseen_combo_falls_back_to_global_freq():
    train_ev = _toy_ev([
        {"learner": "a", "t": 0, "support_code": 0, "gap_log1p_min": np.nan, "first_event": True},
        {"learner": "a", "t": 1, "support_code": 1, "gap_log1p_min": np.log1p(1.0), "first_event": False},
    ])
    global_freq = fit_global_support_freq(train_ev)
    table = fit_lag1_gap_table(train_ev, global_freq, shrinkage_k=5.0)
    # a (prev_code, bucket) combination that never occurred in train.
    unseen_key = (2, 3)
    assert unseen_key not in table
    probs = lag1_gap_probs(
        _toy_ev([{"learner": "z", "t": 5, "support_code": 0,
                  "gap_log1p_min": np.log1p(9999.0), "first_event": False}]),
        {}, global_freq)  # empty table -> every row must fall back
    np.testing.assert_allclose(probs[0], global_freq)


def test_lag1_gap_probs_recovers_a_strong_deterministic_pattern():
    """If, in TRAIN, support_code==1 is ALWAYS immediately followed by support_code==2
    (same learner, short gap), the fitted table must predict that with high confidence
    -- otherwise the baseline has no chance of being a meaningful comparator."""
    rows = []
    for learner_i in range(30):
        rows.append({"learner": f"l{learner_i}", "t": 0, "support_code": 1,
                    "gap_log1p_min": np.nan, "first_event": True})
        rows.append({"learner": f"l{learner_i}", "t": 1, "support_code": 2,
                    "gap_log1p_min": np.log1p(1.0), "first_event": False})
    train_ev = _toy_ev(rows)
    global_freq = fit_global_support_freq(train_ev)
    table = fit_lag1_gap_table(train_ev, global_freq, shrinkage_k=1.0)

    # the val learner needs its OWN t=0 row so _prev_support_code has something to shift
    # from -- a bare t=1 row with no history looks like a first event (prev=-1), which
    # was this test's original bug: it asserted on a row that could never hit the fitted
    # (prev_code=1, bucket=0) cell in the first place.
    val_ev = _toy_ev([
        {"learner": "new", "t": 0, "support_code": 1, "gap_log1p_min": np.nan, "first_event": True},
        {"learner": "new", "t": 1, "support_code": 2, "gap_log1p_min": np.log1p(1.0), "first_event": False},
    ])
    probs = lag1_gap_probs(val_ev, table, global_freq)
    assert probs[1][2] > 0.8


def test_baselines_are_scoreable_with_multiclass_metrics():
    """End-to-end: baseline probs feed directly into the same multiclass_nll/brier the
    trained support head is scored with (gate (i) needs a like-for-like comparison)."""
    ev = _toy_ev([{"learner": "a", "t": t, "support_code": t % N_SUPPORT,
                  "gap_log1p_min": np.log1p(1.0) if t else np.nan, "first_event": t == 0}
                 for t in range(8)])
    global_freq = fit_global_support_freq(ev)
    probs = learner_prefix_probs(ev, global_freq, shrinkage_k=3.0)
    target = ev["support_code"].to_numpy()
    nll = multiclass_nll(probs, target)
    br = multiclass_brier(probs, target)
    assert math.isfinite(nll) and nll >= 0.0
    assert math.isfinite(br) and 0.0 <= br <= 2.0


# -- the class_col/n_classes generalisation -- an arbitrary 5-class ----------------------
# column must behave identically to the hardcoded 4-class support_code path above, using
# the SAME functions with a different class_col rather than a forked copy.

def test_generalised_functions_work_on_a_5_class_column_not_just_support_code():
    N_5 = 5
    ev = _toy_ev([{"learner": "a", "t": t, "outcome_code": t % N_5,
                  "gap_log1p_min": np.log1p(1.0) if t else np.nan, "first_event": t == 0}
                 for t in range(10)])
    global_freq = fit_global_support_freq(ev, class_col="outcome_code", n_classes=N_5)
    assert global_freq.shape == (N_5,)
    np.testing.assert_allclose(global_freq.sum(), 1.0)

    lp_probs = learner_prefix_probs(ev, global_freq, shrinkage_k=3.0, class_col="outcome_code")
    assert lp_probs.shape == (10, N_5)
    np.testing.assert_allclose(lp_probs.sum(axis=1), np.ones(10), rtol=1e-9)

    table = fit_lag1_gap_table(ev, global_freq, shrinkage_k=3.0, class_col="outcome_code")
    lg_probs = lag1_gap_probs(ev, table, global_freq, class_col="outcome_code")
    assert lg_probs.shape == (10, N_5)
    np.testing.assert_allclose(lg_probs.sum(axis=1), np.ones(10), rtol=1e-9)

    target = ev["outcome_code"].to_numpy()
    nll = multiclass_nll(lp_probs, target)
    assert math.isfinite(nll) and nll >= 0.0


def test_generalised_functions_default_to_support_code_unchanged():
    """The default `class_col="support_code"` must reproduce the ORIGINAL (pre-13-Aug)
    behaviour exactly -- every existing caller (fa_06_support_baselines.py) omits this
    argument entirely."""
    ev = _toy_ev([{"learner": "a", "t": 0, "support_code": 2},
                 {"learner": "b", "t": 0, "support_code": 1}])
    freq_default = fit_global_support_freq(ev)
    freq_explicit = fit_global_support_freq(ev, class_col="support_code", n_classes=N_SUPPORT)
    np.testing.assert_array_equal(freq_default, freq_explicit)
