"""CW-SIM TIKTOC-transfer eval (item 5) -- toy-fixture tests for the model-independent
scoring half (`atom.eval.tiktoc_metrics`), built ahead of the real predictions so only
the embedding+belief pass (needs `final.pt`, still training) is left once it exists."""
from __future__ import annotations

import numpy as np

from atom.eval.metrics import f1_score
from atom.eval.tiktoc_metrics import TIKTOC_TABLE3_REFERENCE, tiktoc_comparison_panel


def _pred(learner, pid, csid, pred_prob, target, valid_mask):
    return {
        "learner": np.asarray([learner]), "problem_id": np.asarray([pid]),
        "code_state_id": np.asarray([csid]),
        "pred_prob": np.asarray([pred_prob], dtype=np.float32),
        "target": np.asarray([target], dtype=np.float32),
        "valid_mask": np.asarray([valid_mask], dtype=bool),
    }


def test_f1_score_matches_hand_computed_value():
    y_true = [1, 1, 0, 0, 1]
    y_prob = [0.9, 0.4, 0.2, 0.6, 0.8]  # thresholded @0.5: [1,0,0,1,1] -> TP=2,FP=1,FN=1
    # precision=2/3, recall=2/3 -> F1=2/3
    assert abs(f1_score(y_true, y_prob) - (2 / 3)) < 1e-9


def test_f1_score_all_negative_prediction_and_target_is_zero_not_nan():
    assert f1_score([0, 0, 0], [0.1, 0.2, 0.3]) == 0.0


def test_tiktoc_comparison_panel_empty_preds_returns_all_none():
    panel = tiktoc_comparison_panel([], headline_problem_ids=[1, 2, 3])
    assert panel["pooled"]["auc"] is None
    assert panel["pooled"]["n_pairs"] == 0
    assert panel["reference"] is TIKTOC_TABLE3_REFERENCE


def test_tiktoc_comparison_panel_splits_headline_vs_unseen_correctly():
    headline_ids = [1, 3]
    preds = [
        # Headline problem 1: perfectly separable (auc=1.0)
        _pred("l1", 1, "cs1", pred_prob=[0.9, 0.9], target=[1, 1], valid_mask=[True, True]),
        _pred("l2", 1, "cs2", pred_prob=[0.1, 0.1], target=[0, 0], valid_mask=[True, True]),
        # Unseen problem 34: perfectly WRONG (auc=0.0)
        _pred("l1", 34, "cs3", pred_prob=[0.1, 0.9], target=[1, 0], valid_mask=[True, True]),
        _pred("l2", 34, "cs4", pred_prob=[0.9, 0.1], target=[0, 1], valid_mask=[True, True]),
    ]
    panel = tiktoc_comparison_panel(preds, headline_problem_ids=headline_ids)

    assert panel["headline"]["n_pairs"] == 4
    assert panel["headline"]["auc"] == 1.0
    assert panel["unseen"]["n_pairs"] == 4
    assert panel["unseen"]["auc"] == 0.0
    assert panel["pooled"]["n_pairs"] == 8  # headline + unseen, never silently pooled-only
    assert set(panel["per_problem"].keys()) == {"1", "34"}
    assert panel["per_problem"]["1"]["auc"] == 1.0
    assert panel["per_problem"]["34"]["auc"] == 0.0


def test_tiktoc_comparison_panel_respects_valid_mask_padding():
    # 3 padded slots, only 2 real test cases -- padding must never enter the metric.
    preds = [_pred("l1", 1, "cs1", pred_prob=[0.9, 0.1, 0.5], target=[1, 0, 1],
                    valid_mask=[True, True, False])]
    panel = tiktoc_comparison_panel(preds, headline_problem_ids=[1])
    assert panel["pooled"]["n_pairs"] == 2  # not 3 -- the padded slot is excluded
    assert panel["pooled"]["auc"] == 1.0  # would be 0.5 if the padded slot leaked in


def test_tiktoc_comparison_panel_discloses_threshold_assumption():
    panel = tiktoc_comparison_panel([], headline_problem_ids=[1])
    assert panel["threshold"] == 0.5
    assert panel["threshold_assumption_disclosed"] is True


def test_tiktoc_comparison_panel_includes_calibrated_f1_beside_fixed_threshold():
    # Positives rank strictly above negatives (real separability) but the whole
    # distribution sits below 0.5, so the fixed threshold predicts no positives at all
    # (F1=0.0) -- the calibrated field must surface the real separability, labelled
    # separately from the fixed-threshold `f1` (never silently replacing it).
    preds = [
        _pred("l1", 1, "cs1", pred_prob=[0.1, 0.35], target=[0, 1], valid_mask=[True, True]),
        _pred("l2", 1, "cs2", pred_prob=[0.15, 0.4], target=[0, 1], valid_mask=[True, True]),
    ]
    panel = tiktoc_comparison_panel(preds, headline_problem_ids=[1])
    assert panel["pooled"]["f1"] == 0.0
    assert panel["pooled"]["f1_calibrated"] == 1.0
    assert panel["pooled"]["f1_calibrated_threshold"] is not None
    assert panel["pooled"]["f1_calibrated_threshold"] < 0.5


def test_tiktoc_comparison_panel_empty_preds_calibrated_fields_are_none():
    panel = tiktoc_comparison_panel([], headline_problem_ids=[1, 2, 3])
    assert panel["pooled"]["f1_calibrated"] is None
    assert panel["pooled"]["f1_calibrated_threshold"] is None
