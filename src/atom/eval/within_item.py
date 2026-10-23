"""Within-item AUC with a paired CI. AUC across learners WITHIN each item, pooled over items
weighted by usable pair count (n_pos * n_neg). An item-only (constant) predictor scores
exactly 0.5 by construction, so person discrimination measured this way is immune to the
stochasticity and KL confounds that make a raw likelihood comparison unfair to a
stochastic, multi-objective model.

The construction is the same as the `within_item_auc` in
`scripts/fa_01j_person_item_decomposition.py` and `scripts/fa_01k_b4_close.py` (which keep
their own copies so their runs stay reproducible); this version adds a bootstrap CI,
which matters because a point estimate from roughly 1,400 pairs is thin.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from atom.eval.metrics import _to_numpy, auc

MIN_PAIRS_PER_ITEM = 1  # an item needs >=1 positive AND >=1 negative to contribute an AUC


def within_item_auc(problem_ids: Any, y: Any, p: Any,
                    min_pairs_per_item: int = MIN_PAIRS_PER_ITEM) -> dict:
    """Per-item AUC (learners within that item only), pooled by pair-count weight
    (n_pos * n_neg per item). Returns `pooled_auc` (the headline number), plus per-item
    detail (`items`: list of `{problem_id, auc, pairs, n}`) so a bootstrap CI can
    resample items directly without recomputing per-item AUCs each draw."""
    problem_ids = _to_numpy(problem_ids)
    y = _to_numpy(y).astype(int)
    p = _to_numpy(p)
    df = pd.DataFrame({"problem_id": problem_ids, "y": y, "p": p})
    items = []
    for pid, g in df.groupby("problem_id", sort=False):
        yy = g["y"].to_numpy().astype(int)
        n_pos, n_neg = int(yy.sum()), int((1 - yy).sum())
        if n_pos < min_pairs_per_item or n_neg < min_pairs_per_item:
            continue
        a = auc(yy, g["p"].to_numpy())
        if not np.isfinite(a):
            continue
        items.append({"problem_id": pid, "auc": a, "pairs": n_pos * n_neg, "n": len(g)})
    if not items:
        return {"pooled_auc": float("nan"), "n_items": 0, "n_pairs": 0, "items": []}
    aucs = np.array([it["auc"] for it in items])
    weights = np.array([it["pairs"] for it in items], dtype=float)
    return {
        "pooled_auc": float(np.average(aucs, weights=weights)),
        "unweighted_mean": float(aucs.mean()),
        "n_items": len(items), "n_pairs": int(weights.sum()),
        "n_rows_covered": int(sum(it["n"] for it in items)),
        "items": items,
    }


def within_item_auc_paired(problem_ids: Any, y: Any, p: Any, n_boot: int, seed: int,
                           min_pairs_per_item: int = MIN_PAIRS_PER_ITEM) -> dict:
    """`within_item_auc` plus a percentile bootstrap CI, resampling ITEMS with
    replacement (not rows) -- each bootstrap draw is a fresh weighted-average over a
    resampled set of items' own (fixed) per-item AUCs, which is standard for a
    pair-weighted pooled statistic and is far cheaper than re-deriving per-item AUCs
    from resampled rows on every draw.

    Reports `excludes_zero_above_0_5` since the mechanical null here is 0.5, not 0:
    any model without per-learner state scores exactly 0.5000 within-item by
    construction.
    """
    base = within_item_auc(problem_ids, y, p, min_pairs_per_item=min_pairs_per_item)
    items = base["items"]
    if not items:
        return {**base, "ci95": [float("nan"), float("nan")], "excludes_zero_above_0_5": False, "n_boot": n_boot}

    aucs = np.array([it["auc"] for it in items])
    weights = np.array([it["pairs"] for it in items], dtype=float)
    n_items = len(items)
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot)
    for i in range(n_boot):
        draw = rng.integers(0, n_items, size=n_items)
        boot[i] = np.average(aucs[draw], weights=weights[draw])
    lo, hi = float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))
    return {**base, "ci95": [lo, hi], "excludes_zero_above_0_5": bool(lo > 0.5),
           "n_boot": n_boot}
