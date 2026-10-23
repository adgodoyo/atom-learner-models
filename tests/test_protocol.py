"""Tests for the frozen calib/eval split protocol shared by every P3+ model comparison."""

import pandas as pd

from atom.eval.protocol import calib_eval_split


def _toy():
    rows = []
    for learner, split, n in (("a", "train", 10), ("b", "val", 10), ("c", "test", 10), ("d", "val", 4)):
        for t in range(n):
            rows.append({"learner": learner, "split": split, "t": t, "correct": t % 2})
    return pd.DataFrame(rows)


def test_train_learners_excluded():
    calib, ev = calib_eval_split(_toy(), calib_frac=0.5, min_interactions=6)
    assert "a" not in set(calib["learner"]) | set(ev["learner"])


def test_calib_eval_partition_by_threshold():
    calib, ev = calib_eval_split(_toy(), calib_frac=0.5, min_interactions=6)
    b_calib = calib[calib["learner"] == "b"]["t"]
    b_eval = ev[ev["learner"] == "b"]["t"]
    assert set(b_calib) == {0, 1, 2, 3, 4}
    assert set(b_eval) == {5, 6, 7, 8, 9}


def test_no_time_overlap_any_learner():
    calib, ev = calib_eval_split(_toy(), calib_frac=0.5, min_interactions=6)
    for learner in set(calib["learner"]) | set(ev["learner"]):
        c = calib[calib["learner"] == learner]["t"]
        e = ev[ev["learner"] == learner]["t"]
        if len(c) and len(e):
            assert c.max() < e.min()


def test_short_learner_dropped_by_min_interactions():
    calib, ev = calib_eval_split(_toy(), calib_frac=0.5, min_interactions=6)
    assert "d" not in set(calib["learner"]) | set(ev["learner"])  # only 4 events, below min 6


def test_calib_frac_shifts_the_threshold():
    calib_lo, ev_lo = calib_eval_split(_toy(), calib_frac=0.25, min_interactions=6)
    calib_hi, ev_hi = calib_eval_split(_toy(), calib_frac=0.75, min_interactions=6)
    n_c_lo = len(calib_lo[calib_lo["learner"] == "b"])
    n_c_hi = len(calib_hi[calib_hi["learner"] == "b"])
    assert n_c_lo < n_c_hi  # a smaller calib_frac means a shorter calibration burn-in
