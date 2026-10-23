"""CW-SIM 30-Aug audit items 1/4/5 -- the value-keyed eval aggregation, the paired
learner-clustered personalisation-gap CI, the test-vector AUC panel, and the per-block
checkpoint/resume round trip. All pure-array/IO logic, testable without a real model or
manifest (mirroring `tests/test_cw_5_score_arm.py`'s own established split)."""
from __future__ import annotations

import importlib.util
import pathlib

import numpy as np
import pandas as pd

REPO = pathlib.Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("cw_sim_full_run", REPO / "infra" / "cw_sim_full_run.py")
sim = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sim)


def _fake_scores(learners, problem_ids, code_state_ids, nll, top1, exact) -> dict:
    n = len(learners)
    return {
        "kept_row_idx": np.arange(n), "true_class_nll": np.asarray(nll, dtype=float),
        "true_class_top1": np.asarray(top1, dtype=float), "true_class_exact_match": np.asarray(exact, dtype=float),
    }


def _fake_attempts(learners, problem_ids, code_state_ids) -> pd.DataFrame:
    return pd.DataFrame({"learner": learners, "problem_id": problem_ids, "code_state_id": code_state_ids})


def test_value_keyed_rows_round_trip_identity():
    learners = ["a", "a", "b"]
    pids = [1, 2, 1]
    csids = ["c0", "c1", "c2"]
    scores = _fake_scores(learners, pids, csids, nll=[0.1, 0.2, 0.3], top1=[1, 0, 1], exact=[1, 0, 0])
    attempts = _fake_attempts(learners, pids, csids)
    rows = sim.value_keyed_rows_from_scores(scores, attempts)
    np.testing.assert_array_equal(rows["learner"], learners)
    np.testing.assert_array_equal(rows["problem_id"], pids)
    np.testing.assert_array_equal(rows["code_state_id"], csids)
    np.testing.assert_allclose(rows["true_class_nll"], [0.1, 0.2, 0.3])


def test_concat_value_keyed_rows_across_blocks():
    a = sim.value_keyed_rows_from_scores(
        _fake_scores(["a"], [1], ["c0"], [0.5], [1], [1]), _fake_attempts(["a"], [1], ["c0"]))
    b = sim.value_keyed_rows_from_scores(
        _fake_scores(["b"], [2], ["c1"], [0.7], [0], [0]), _fake_attempts(["b"], [2], ["c1"]))
    merged = sim.concat_value_keyed_rows([a, b])
    np.testing.assert_array_equal(merged["learner"], ["a", "b"])
    np.testing.assert_allclose(merged["true_class_nll"], [0.5, 0.7])


def test_concat_value_keyed_rows_empty_list():
    merged = sim.concat_value_keyed_rows([])
    assert len(merged["learner"]) == 0
    assert len(merged["true_class_nll"]) == 0


def test_value_keyed_stratum_summary_matches_manual_mean():
    rows = {"learner": np.array(["a", "b", "c"]), "true_class_nll": np.array([1.0, 2.0, 3.0]),
           "true_class_top1": np.array([1.0, 0.0, 1.0]), "true_class_exact_match": np.array([1.0, 0.0, 0.0])}
    out = sim.value_keyed_stratum_summary(rows, None)
    assert out["n"] == 3
    assert out["token_nll_mean"] == pytest_approx(2.0)
    assert out["top1_rate"] == pytest_approx(2.0 / 3.0)
    assert out["exact_match_rate"] == pytest_approx(1.0 / 3.0)


def test_value_keyed_stratum_summary_with_mask_and_empty():
    rows = {"learner": np.array(["a", "b"]), "true_class_nll": np.array([1.0, 5.0]),
           "true_class_top1": np.array([1.0, 0.0]), "true_class_exact_match": np.array([0.0, 0.0])}
    masked = sim.value_keyed_stratum_summary(rows, np.array([True, False]))
    assert masked["n"] == 1
    assert masked["token_nll_mean"] == pytest_approx(1.0)
    empty = sim.value_keyed_stratum_summary(rows, np.array([False, False]))
    assert empty == {"n": 0}


def test_value_keyed_bottom_tercile_mask_joins_by_learner_value():
    # 6 val learners, CLEAN_FULL(=3) rate strictly increasing a..f -- bottom third = {a, b}.
    manifest_attempts = pd.DataFrame({
        "learner": ["a"] * 4 + ["b"] * 4 + ["c"] * 4 + ["d"] * 4 + ["e"] * 4 + ["f"] * 4,
        "outcome_code": ([3, 0, 0, 0] + [3, 3, 0, 0] + [3, 3, 3, 0] * 1 + [3, 3, 3, 0]
                         + [3, 3, 3, 3] + [3, 3, 3, 3]),
    })
    manifest = {"attempts": manifest_attempts, "val_learners": {"a", "b", "c", "d", "e", "f"}}
    # rows in a DIFFERENT order than manifest_attempts, exercising the join-by-value.
    rows = {"learner": np.array(["f", "a", "c", "b"])}
    mask = sim.value_keyed_bottom_tercile_mask(rows, manifest)
    np.testing.assert_array_equal(mask, [False, True, False, True])


def test_paired_personalisation_gap_ci_only_uses_the_intersection():
    personalised = {"learner": np.array(["a", "a", "b"]), "problem_id": np.array([1, 2, 1]),
                    "code_state_id": np.array(["c0", "c1", "c2"]), "true_class_nll": np.array([1.0, 2.0, 3.0])}
    # shuffled_u drops (a, 2, c1) entirely (e.g. no derangement partner) and adds an
    # extra row personalised never scored (learner c) -- only (a,1,c0) and (b,1,c2)
    # are common; the CI must be computed on exactly those two rows.
    shuffled = {"learner": np.array(["a", "b", "c"]), "problem_id": np.array([1, 1, 5]),
               "code_state_id": np.array(["c0", "c2", "c9"]), "true_class_nll": np.array([1.5, 2.5, 9.0])}
    ci = sim.paired_personalisation_gap_ci(personalised, shuffled, n_boot=200, seed=0)
    assert ci is not None
    assert ci["n_rows_intersection"] == 2
    # point delta = mean([1.0, 3.0]) - mean([1.5, 2.5]) = 2.0 - 2.0 = 0.0
    assert ci["point_delta"] == pytest_approx(0.0)


def test_paired_personalisation_gap_ci_empty_intersection_returns_none():
    personalised = {"learner": np.array(["a"]), "problem_id": np.array([1]),
                    "code_state_id": np.array(["c0"]), "true_class_nll": np.array([1.0])}
    shuffled = {"learner": np.array(["a"]), "problem_id": np.array([2]),
               "code_state_id": np.array(["c1"]), "true_class_nll": np.array([1.0])}
    assert sim.paired_personalisation_gap_ci(personalised, shuffled) is None


def test_test_vector_auc_panel_pooled_and_per_problem():
    # a perfectly-discriminating predictor for problem 1, a coin-flip for problem 2.
    preds = [{
        "learner": np.array(["a", "b"]), "problem_id": np.array([1, 2]),
        "code_state_id": np.array(["c0", "c1"]),
        "pred_prob": np.array([[0.9, 0.1], [0.5, 0.5]], dtype=np.float32),
        "target": np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32),
        "valid_mask": np.array([[True, True], [True, True]]),
    }]
    panel = sim.test_vector_auc_panel(preds)
    assert panel["n_pairs"] == 4
    assert panel["per_problem"]["1"]["auc"] == pytest_approx(1.0)
    assert panel["per_problem"]["1"]["n_pairs"] == 2
    assert panel["per_problem"]["2"]["auc"] is None or 0.0 <= panel["per_problem"]["2"]["auc"] <= 1.0


def test_test_vector_auc_panel_empty_input():
    panel = sim.test_vector_auc_panel([])
    assert panel == {"pooled_auc": None, "n_pairs": 0, "n_rows": 0, "per_problem": {}}


def test_test_vector_auc_panel_single_class_group_reports_none_not_a_default():
    preds = [{
        "learner": np.array(["a"]), "problem_id": np.array([1]), "code_state_id": np.array(["c0"]),
        "pred_prob": np.array([[0.9, 0.2]], dtype=np.float32),
        "target": np.array([[1.0, 1.0]], dtype=np.float32),  # only ONE class present
        "valid_mask": np.array([[True, True]]),
    }]
    panel = sim.test_vector_auc_panel(preds)
    assert panel["pooled_auc"] is None  # NaN from sklearn's single-class case, surfaced as None
    assert panel["per_problem"]["1"]["auc"] is None


def test_block_scores_save_and_resume_round_trip(tmp_path):
    scores = {
        "mixture_nll": np.array([0.1, 0.2]), "mixture_nll_raw": np.array([1.0, 2.0]),
        "true_class_nll": np.array([0.5, 0.6]), "true_class_top1": np.array([1.0, 0.0]),
        "true_class_exact_match": np.array([1.0, 0.0]), "p_class_meanK": np.zeros((2, 4)),
        "true_class_nll_per_k": np.zeros((3, 2)), "kept_row_idx": np.array([0, 1]),
        "mu": np.zeros((2, 32), dtype=np.float32), "log_sigma": np.zeros((2, 32), dtype=np.float32),
        "n_dropped_budget": 0, "elapsed_s": 1.0, "n_forward_calls": 4, "n_rows_scored": 2,
        "n_dropped_derangement": 0,
    }
    sim.save_block_scores(scores, tmp_path, block=3, arm="personalised", bucket=None, out_prefix="x")
    resumed = sim.try_resume_block_scores(tmp_path, block=3, arm="personalised", bucket=None, out_prefix="x")
    assert resumed is not None
    np.testing.assert_allclose(resumed["true_class_nll"], scores["true_class_nll"])
    assert resumed["n_rows_scored"] == 2


def test_block_scores_resume_missing_block_returns_none(tmp_path):
    assert sim.try_resume_block_scores(tmp_path, block=0, arm="dw0", bucket=None, out_prefix="x") is None


def test_block_tv_preds_save_and_resume_round_trip_including_none(tmp_path):
    # `learner`/`code_state_id` as OBJECT-dtype arrays (via a pandas object-dtype
    # column's `.to_numpy()`, the real upstream construct) -- NOT `np.array(["a"])`,
    # whose short uniform-length strings numpy infers as fixed-width unicode and which
    # round-trips fine under `allow_pickle=False`, masking a real resume-path bug that
    # only fired against actual pandas-sourced identity columns (30-Aug audit rerun).
    tv = {"learner": pd.array(["a"], dtype=object).to_numpy(),
         "problem_id": np.array([1]),
         "code_state_id": pd.array(["c0"], dtype=object).to_numpy(),
         "pred_prob": np.array([[0.5]], dtype=np.float32), "target": np.array([[1.0]], dtype=np.float32),
         "valid_mask": np.array([[True]])}
    assert tv["learner"].dtype == object  # canary: the fixture must exercise the real dtype
    sim.save_block_tv_preds(tv, tmp_path, block=0, bucket=None, out_prefix="x")
    resumed = sim.try_resume_block_tv_preds(tmp_path, block=0, bucket=None, out_prefix="x")
    assert resumed is not None
    np.testing.assert_array_equal(resumed["learner"], tv["learner"])

    sim.save_block_tv_preds(None, tmp_path, block=1, bucket=None, out_prefix="x")
    resumed_none = sim.try_resume_block_tv_preds(tmp_path, block=1, bucket=None, out_prefix="x")
    assert resumed_none is None

    not_found = sim.try_resume_block_tv_preds(tmp_path, block=99, bucket=None, out_prefix="x")
    assert not_found is sim._TV_RESUME_NOT_FOUND


def pytest_approx(x, rel=1e-6):
    import pytest
    return pytest.approx(x, rel=rel)
