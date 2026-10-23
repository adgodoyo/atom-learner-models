"""Prediction metrics: correctness AUC, calibration (ECE), Brier, accuracy.

Also multiclass NLL/Brier/ECE, F1, and bootstrap confidence intervals (i.i.d., paired,
and learner-clustered). These back the gate checks (e.g. "IRT test AUC in [0.68, 0.76]",
"ECE <= 0.05"). Inputs are array-likes of floats; torch/numpy are both accepted.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _to_numpy(x: Any) -> np.ndarray:
    try:
        import torch

        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except ImportError:
        pass
    return np.asarray(x)


def auc(y_true: Any, y_score: Any) -> float:
    """Area under the ROC curve. Returns NaN if only one class is present."""
    yt = _to_numpy(y_true).ravel()
    ys = _to_numpy(y_score).ravel()
    if len(np.unique(yt)) < 2:
        return float("nan")
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(yt, ys))


def accuracy(y_true: Any, y_prob: Any, threshold: float = 0.5) -> float:
    yt = _to_numpy(y_true).ravel()
    yp = (_to_numpy(y_prob).ravel() >= threshold).astype(int)
    return float((yt == yp).mean())


def f1_score(y_true: Any, y_prob: Any, threshold: float = 0.5) -> float:
    """Binary F1 at a fixed probability threshold. Returns 0.0 (sklearn's own
    `zero_division=0` convention) if the thresholded predictions contain no positives
    and there are no true positives to find -- never NaN, since an all-negative
    prediction on an all-negative slice is a defined (if uninformative) F1 of 0."""
    yt = _to_numpy(y_true).ravel()
    yp = (_to_numpy(y_prob).ravel() >= threshold).astype(int)
    from sklearn.metrics import f1_score as _sk_f1

    return float(_sk_f1(yt, yp, zero_division=0))


def best_threshold_f1(y_true: Any, y_prob: Any, n_candidates: int = 200) -> tuple[float, float]:
    """Sweeps candidate thresholds (the unique predicted probabilities themselves,
    subsampled to `n_candidates` quantiles when there are more than that) and returns
    `(best_threshold, best_f1)` -- the F1-MAXIMISING threshold found and evaluated on
    THIS SAME population. This is an IN-SAMPLE OPTIMUM, never a calibrated or held-out
    number: it answers "how threshold-sensitive is this F1", not "what F1 would this
    model achieve on new data at a properly chosen threshold". Callers must label it as
    such wherever it is reported (`atom.eval.tiktoc_metrics._pooled_metrics`'s own
    `f1_calibrated`/`f1_calibrated_threshold` keys carry this disclosure by name).
    Returns `(None, None)` if fewer than 2 classes are present (mirrors `auc`'s own
    degenerate-input convention, rather than a misleading 0.5/0.0 default)."""
    yt = _to_numpy(y_true).ravel()
    yp = _to_numpy(y_prob).ravel()
    if len(np.unique(yt)) < 2:
        return None, None
    thresholds = np.unique(np.clip(yp, 0.0, 1.0))
    if len(thresholds) > n_candidates:
        thresholds = np.unique(np.quantile(thresholds, np.linspace(0.0, 1.0, n_candidates)))
    best_t, best_f1 = 0.5, -1.0
    for t in thresholds:
        val = f1_score(yt, yp, threshold=float(t))
        if val > best_f1:
            best_f1, best_t = val, float(t)
    return best_t, best_f1


def brier(y_true: Any, y_prob: Any) -> float:
    yt = _to_numpy(y_true).ravel().astype(float)
    yp = _to_numpy(y_prob).ravel().astype(float)
    return float(np.mean((yp - yt) ** 2))


def multiclass_nll(probs: Any, target: Any, eps: float = 1e-12) -> float:
    """Mean negative log-likelihood of the observed class under a predicted C-way
    simplex. `probs`: (N, C) rows (need not be exactly normalised -- callers that build
    probabilities from counts/shrinkage already sum to 1 by construction, but `eps`
    clipping guards against a stray 0 either way); `target`: (N,) int class indices."""
    p = _to_numpy(probs)
    t = _to_numpy(target).ravel().astype(int)
    selected = p[np.arange(len(t)), t]
    return float(-np.log(np.clip(selected, eps, 1.0)).mean())


def multiclass_brier(probs: Any, target: Any, n_classes: int | None = None) -> float:
    """Mean squared error between the predicted C-way simplex and the one-hot target
    (multiclass Brier score, sums to at most 2 per row: this is the un-normalised
    "original" definition, not the /2 variant -- matches how support Brier is already
    computed inline in infra/fa_06_stage2_train.py's val_evaluate, so the two stay
    directly comparable)."""
    p = _to_numpy(probs)
    t = _to_numpy(target).ravel().astype(int)
    c = p.shape[1] if n_classes is None else n_classes
    onehot = np.eye(c)[t]
    return float(((p - onehot) ** 2).sum(axis=1).mean())


def with_population(metrics: dict[str, Any], n: int) -> dict[str, Any]:
    """Stamp the evaluation-population count onto a metrics row. Every metric the
    harness emits states how many examples it was computed over, so a population
    difference between two numbers being compared (e.g. AKT's 6,980-event stream
    exclusion vs the filter's and simulator's loss-only masking of the same rows) is
    visible at the point the number is read, not discoverable only by re-reading each
    script's own report. A plain dict merge, not a class, so existing
    `gate_report(metrics=...)` call sites need no restructuring to adopt it."""
    return {**metrics, "n_eval_population": int(n)}


def bootstrap_mean_ci(x: Any, n_boot: int, alpha: float, seed: int) -> tuple[float, float, float]:
    """Percentile bootstrap CI for the mean of `x`. Returns (mean, ci_lo, ci_hi) at
    confidence 1 - alpha (e.g. alpha=0.05 -> 95% CI).
    """
    x = _to_numpy(x)
    rng = np.random.default_rng(seed)
    n = len(x)
    boot_means = np.empty(n_boot)
    for i in range(n_boot):
        sample = x[rng.integers(0, n, size=n)]
        boot_means[i] = sample.mean()
    lo, hi = np.percentile(boot_means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(x.mean()), float(lo), float(hi)


def learner_clustered_auc_ci(y_true: Any, y_score: Any, learner_ids: Any, n_boot: int,
                             seed: int, alpha: float = 0.05) -> dict:
    """Percentile bootstrap CI on a POOLED AUC, resampling LEARNERS with replacement
    (not rows) -- AUC is a ranking statistic over the whole population, not a per-row
    mean, so `bootstrap_mean_ci`/`learner_clustered_bootstrap_ci` (both mean-based)
    cannot cover it; every resample recomputes AUC from scratch on that resample's
    pooled rows. Rows sharing a learner move together, matching
    `learner_clustered_bootstrap_ci`'s own rationale under repeated measures. A
    resample that happens to be single-class (can occur with few learners or a rare
    positive class) contributes no bootstrap draw rather than a spurious NaN/0.5;
    if every resample degenerates this way, returns a CI of `(nan, nan)` around the
    point estimate rather than raising."""
    yt = _to_numpy(y_true).ravel()
    ys = _to_numpy(y_score).ravel()
    lids = _to_numpy(learner_ids).ravel()
    assert len(yt) == len(ys) == len(lids)
    point = auc(yt, ys)

    unique_learners = np.unique(lids)
    by_learner = {lr: np.where(lids == lr)[0] for lr in unique_learners}
    rng = np.random.default_rng(seed)
    n_learners = len(unique_learners)
    boot_aucs = []
    for _ in range(n_boot):
        sampled = unique_learners[rng.integers(0, n_learners, size=n_learners)]
        idx = np.concatenate([by_learner[lr] for lr in sampled])
        val = auc(yt[idx], ys[idx])
        if not np.isnan(val):
            boot_aucs.append(val)
    if not boot_aucs:
        return {"auc": point, "ci95": [float("nan"), float("nan")], "n_boot_used": 0, "n_learners": n_learners}
    lo, hi = np.percentile(boot_aucs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"auc": None if np.isnan(point) else point, "ci95": [float(lo), float(hi)],
           "n_boot_used": len(boot_aucs), "n_learners": n_learners}


def paired_bootstrap_ci(a: Any, b: Any, n_boot: int, seed: int) -> dict:
    """Same row order, same resample index applied to both arrays -- a properly PAIRED
    CI on the delta mean(a) - mean(b) (matching `fa_01o_paired_ci_clean_heldout.py`'s
    and `infra/fa_sim1_full_run.py`'s own convention; kept here as the one canonical
    implementation). i.i.d. row resampling -- for repeated-measures data (multiple
    rows per learner) prefer `learner_clustered_bootstrap_ci` instead.
    """
    a, b = _to_numpy(a), _to_numpy(b)
    assert a.shape == b.shape
    rng = np.random.default_rng(seed)
    n = len(a)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        diffs[i] = a[idx].mean() - b[idx].mean()
    point = float(a.mean() - b.mean())
    lo, hi = float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))
    return {"point_delta": point, "ci95": [lo, hi], "excludes_zero": bool(lo > 0 or hi < 0),
            "n_boot": n_boot}


def learner_clustered_bootstrap_ci(a: Any, b: Any, learner_ids: Any, n_boot: int,
                                   seed: int) -> dict:
    """Paired CI on mean(a) - mean(b), resampling LEARNERS with replacement rather than
    rows -- the correct estimator under repeated measures (FA has multiple rows per
    learner, whose errors are correlated within-learner). This is the default estimator
    for simulator comparisons. On the first simulator's evaluation (3,398 rows from 499
    learners) the maximum design effect is 2.6x, whereas its +0.091 result would need a
    design effect of 6x to flip, so row-level CIs there lead to the same conclusion; the
    clustered estimator is used because it is the right default under repeated measures.

    Standard cluster/block bootstrap: draw `n_learners` learner ids WITH replacement; the
    resampled dataset is the UNION of every row belonging to each drawn id (a learner
    drawn twice contributes their rows twice). `a`, `b`, `learner_ids` must be the same
    length and row-aligned.
    """
    a, b, learner_ids = _to_numpy(a), _to_numpy(b), _to_numpy(learner_ids)
    assert a.shape == b.shape == learner_ids.shape
    unique_learners = np.unique(learner_ids)
    n_learners = len(unique_learners)
    rows_by_learner = {lid: np.flatnonzero(learner_ids == lid) for lid in unique_learners}
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        drawn = unique_learners[rng.integers(0, n_learners, size=n_learners)]
        idx = np.concatenate([rows_by_learner[lid] for lid in drawn])
        diffs[i] = a[idx].mean() - b[idx].mean()
    point = float(a.mean() - b.mean())
    lo, hi = float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))
    return {"point_delta": point, "ci95": [lo, hi], "excludes_zero": bool(lo > 0 or hi < 0),
            "n_boot": n_boot, "n_learners": n_learners, "n_rows": int(len(a))}


def multiclass_ece(probs: Any, target: Any, n_bins: int = 10) -> float:
    """Top-1 (argmax-confidence) Expected Calibration Error for a C-way classifier --
    the standard multiclass generalisation of `ece`: bin by the predicted class's OWN
    confidence, compare the bin's mean confidence against the bin's top-1 ACCURACY
    (whether the argmax class was the true class), not against a single positive-class
    rate. `probs`: (N, C); `target`: (N,) int class indices."""
    p = _to_numpy(probs)
    t = _to_numpy(target).ravel().astype(int)
    if len(t) == 0:
        return float("nan")
    pred = p.argmax(axis=1)
    conf = p[np.arange(len(t)), pred]
    correct = (pred == t).astype(float)
    return ece(correct, conf, n_bins=n_bins)


def ece(y_true: Any, y_prob: Any, n_bins: int = 10) -> float:
    """Expected Calibration Error with equal-width probability bins."""
    yt = _to_numpy(y_true).ravel().astype(float)
    yp = _to_numpy(y_prob).ravel().astype(float)
    if yt.size == 0:
        return float("nan")
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    # Bin index in [0, n_bins-1]; clip so p == 1.0 lands in the last bin.
    idx = np.clip(np.digitize(yp, bins) - 1, 0, n_bins - 1)
    total = 0.0
    n = yt.size
    for b in range(n_bins):
        mask = idx == b
        if not mask.any():
            continue
        conf = yp[mask].mean()
        acc = yt[mask].mean()
        total += (mask.sum() / n) * abs(acc - conf)
    return float(total)
