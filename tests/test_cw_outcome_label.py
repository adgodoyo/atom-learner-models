import pandas as pd

from atom.data.cw_outcome_label import N_OUTCOME, OUTCOME_CLASSES, outcome_label, outcome_label_report


def _rows(pairs):
    compile_result, score_frac = zip(*pairs)
    return pd.Series(compile_result), pd.Series(score_frac, dtype="float64")


def test_compile_error_is_class_0_regardless_of_score():
    cr, sf = _rows([("Error", 0.0), ("Error", float("nan"))])
    out = outcome_label(cr, sf)
    assert (out["outcome_code"] == 0).all()
    assert out["outcome_valid"].all()


def test_clean_zero_partial_full_split():
    cr, sf = _rows([("Success", 0.0), ("Success", 0.5), ("Success", 1.0)])
    out = outcome_label(cr, sf)
    assert out["outcome_code"].tolist() == [1, 2, 3]
    assert out["outcome_valid"].all()


def test_clean_with_missing_score_is_invalid_not_silently_zero():
    cr, sf = _rows([("Success", float("nan"))])
    out = outcome_label(cr, sf)
    assert out["outcome_code"].iloc[0] == -1
    assert not out["outcome_valid"].iloc[0]


def test_unknown_compile_result_is_invalid():
    cr, sf = _rows([("Timeout", 0.0)])
    out = outcome_label(cr, sf)
    assert out["outcome_code"].iloc[0] == -1
    assert not out["outcome_valid"].iloc[0]


def test_n_outcome_matches_class_tuple():
    assert N_OUTCOME == len(OUTCOME_CLASSES) == 4


def test_outcome_label_report_counts_and_fractions():
    cr, sf = _rows([
        ("Error", 0.0), ("Error", 0.0),
        ("Success", 0.0),
        ("Success", 0.5), ("Success", 0.5), ("Success", 0.5),
        ("Success", 1.0),
    ])
    report = outcome_label_report(cr, sf)
    assert report["n_total"] == 7
    assert report["n_valid"] == 7
    assert report["n_invalid"] == 0
    assert report["class_counts"] == {
        "COMPILE_ERROR": 2, "CLEAN_ZERO": 1, "CLEAN_PARTIAL": 3, "CLEAN_FULL": 1,
    }
    # 5 clean compiles (1 zero + 3 partial + 1 full); clean_full/clean = 1/5
    assert abs(report["clean_full_frac_of_clean"] - 0.2) < 1e-9


def test_outcome_label_report_on_all_invalid_batch_has_zero_counts_not_crash():
    cr, sf = _rows([("Success", float("nan"))])
    report = outcome_label_report(cr, sf)
    assert report["n_valid"] == 0
    assert report["n_invalid"] == 1
    assert sum(report["class_counts"].values()) == 0
