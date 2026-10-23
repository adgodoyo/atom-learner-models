"""Comparison of the CodeWorkout generative simulator's test-vector predictions with
TIKTOC's published Table 3.

Model- and GPU-independent by design: consumes prediction records shaped like
`infra.cw_sim_full_run.test_vector_eval_predictions`'s output (`learner`, `problem_id`,
`code_state_id`, `pred_prob`, `target`, `valid_mask`), regardless of what produced them
(the real forward pass, which needs the trained filter and codec-embedded TIKTOC code, or
a toy fixture in this module's tests).

TIKTOC's reference numbers are taken from their paper's Table 3 and not re-derived here
(their vendored CodeBLEU implementation and generation panel are handled elsewhere, see
`atom.eval.codebleu`): TIKTOC 0.764/0.794/0.723 (AUC/F1/accuracy), Code-DKT-TC AUC 0.661,
CodeBLEU 0.554 / OKT-TC CodeBLEU 0.522.

DISCLOSED ASSUMPTION: TIKTOC's F1/accuracy classification threshold is not confirmed from
their paper. This module uses the standard 0.5 and states it in every panel it returns,
so a reader can judge comparability. The comparison is also asymmetric (their 5-fold CV
vs our zero-shot transfer; pooled vs per-problem): a disclosed comparability caveat.
"""
from __future__ import annotations

import numpy as np

from atom.eval.metrics import accuracy, auc, best_threshold_f1, f1_score

TIKTOC_TABLE3_REFERENCE = {
    "TIKTOC": {"auc": 0.764, "f1": 0.794, "accuracy": 0.723},
    "Code-DKT-TC": {"auc": 0.661},
    "CodeBLEU": {"TIKTOC": 0.554, "OKT-TC": 0.522},
}


def _pooled_metrics(targets: np.ndarray, probs: np.ndarray, threshold: float) -> dict:
    n = int(len(targets))
    if n == 0:
        return {"auc": None, "f1": None, "accuracy": None, "n_pairs": 0,
               "f1_calibrated": None, "f1_calibrated_threshold": None}
    auc_val = auc(targets, probs)
    cal_t, cal_f1 = best_threshold_f1(targets, probs)
    return {
        "auc": None if np.isnan(auc_val) else auc_val,
        "f1": f1_score(targets, probs, threshold=threshold),
        "accuracy": accuracy(targets, probs, threshold=threshold),
        "n_pairs": n,
        # IN-SAMPLE optimal threshold on THIS SAME population -- a diagnostic note
        # on threshold sensitivity, never a held-out/generalising calibration claim.
        # See `atom.eval.metrics.best_threshold_f1`'s own docstring for the disclosure.
        "f1_calibrated": cal_f1, "f1_calibrated_threshold": cal_t,
    }


def tiktoc_comparison_panel(preds: list[dict], headline_problem_ids, threshold: float = 0.5) -> dict:
    """Pools prediction records (same shape as
    `infra.cw_sim_full_run.test_vector_eval_predictions`'s output) into pooled,
    per-problem, and headline-vs-unseen AUC/F1/accuracy, alongside TIKTOC's own Table 3
    reference numbers for direct reading. `headline_problem_ids` -- the 13 join-verified
    problems -- must be passed explicitly by the caller (not imported from the
    JDK-dependent `cw_5_functional_equivalence` module, keeping this module import-light
    and independently testable); the unseen cell (34/37/39/40) is whatever's left. An
    empty `preds` list returns an all-None panel rather than raising, matching
    `test_vector_auc_panel`'s own convention for "nothing matched"."""
    headline_set = set(int(p) for p in headline_problem_ids)
    empty = {"auc": None, "f1": None, "accuracy": None, "n_pairs": 0,
             "f1_calibrated": None, "f1_calibrated_threshold": None}
    if not preds:
        return {
            "threshold": threshold, "threshold_assumption_disclosed": True,
            "pooled": empty, "headline": dict(empty), "unseen": dict(empty),
            "per_problem": {}, "reference": TIKTOC_TABLE3_REFERENCE,
        }

    probs = np.concatenate([p["pred_prob"] for p in preds], axis=0)
    targets = np.concatenate([p["target"] for p in preds], axis=0)
    masks = np.concatenate([p["valid_mask"] for p in preds], axis=0).astype(bool)
    problem_ids = np.concatenate([p["problem_id"] for p in preds], axis=0)

    flat_mask = masks
    pooled = _pooled_metrics(targets[flat_mask], probs[flat_mask], threshold)

    headline_row = np.isin(problem_ids, list(headline_set))
    headline_mask = masks & headline_row[:, None]
    headline = _pooled_metrics(targets[headline_mask], probs[headline_mask], threshold)

    unseen_row = ~headline_row
    unseen_mask = masks & unseen_row[:, None]
    unseen = _pooled_metrics(targets[unseen_mask], probs[unseen_mask], threshold)

    per_problem = {}
    for pid in sorted(set(int(p) for p in problem_ids)):
        row_sel = problem_ids == pid
        per_problem[str(pid)] = _pooled_metrics(
            targets[row_sel][masks[row_sel]], probs[row_sel][masks[row_sel]], threshold,
        )

    return {
        "threshold": threshold, "threshold_assumption_disclosed": True,
        "pooled": pooled, "headline": headline, "unseen": unseen,
        "per_problem": per_problem, "reference": TIKTOC_TABLE3_REFERENCE,
    }
