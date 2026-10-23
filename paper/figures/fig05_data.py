#!/usr/bin/env python
"""Data for thesis Figure 5: FA correctness AUC by history length, own belief vs a stranger's.

Inputs: FA-SIM3's evaluation outputs, written locally by `infra/fa_sim3_full_run.py`
(results/fa_sim3_full_run/predictions.csv.gz and belief_diagnostics.npz; they need the data).
Output: paper/figures/data/fig05_fa_auc_by_history.csv (one row per history bucket; aggregate only),
with learner-clustered 95% bootstrap intervals.
"""
from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

REPO = pathlib.Path(__file__).resolve().parents[2]
RUN = REPO / "results" / "fa_sim3_full_run"
OUT = pathlib.Path(__file__).resolve().parent / "data" / "fig05_fa_auc_by_history.csv"
BUCKETS = [(0, 10), (10, 25), (25, 50), (50, 100), (100, 200), (200, 10 ** 9)]
NBOOT, SEED = 2000, 20260904


def boot_auc(y, p, learners, rng):
    uniq = np.unique(learners)
    idx = {u: np.flatnonzero(learners == u) for u in uniq}
    out = []
    for _ in range(NBOOT):
        rows = np.concatenate([idx[u] for u in rng.choice(uniq, len(uniq), replace=True)])
        if len(np.unique(y[rows])) == 2:
            out.append(roc_auc_score(y[rows], p[rows]))
    return np.quantile(out, [0.025, 0.975])


def main() -> None:
    pred = pd.read_csv(RUN / "predictions.csv.gz")
    diag = np.load(RUN / "belief_diagnostics.npz")
    y = (pred["true_class"] == 0).astype(int).to_numpy()
    own = diag["outcome_prob_meanK_full_personalised"][:, 0]
    stranger = diag["outcome_prob_meanK_full_shuffled_u"][:, 0]
    learners = pred["learner"].to_numpy()
    rng = np.random.default_rng(SEED)
    rows = []
    for lo, hi in BUCKETS:
        m = ((pred["t"] >= lo) & (pred["t"] < hi)).to_numpy()
        a_lo, a_hi = boot_auc(y[m], own[m], learners[m], rng)
        s_lo, s_hi = boot_auc(y[m], stranger[m], learners[m], rng)
        rows.append({"bucket": f"[{lo},{hi if hi < 10 ** 9 else 'inf'})", "n": int(m.sum()),
                     "n_learners": int(len(np.unique(learners[m]))),
                     "auc_own": roc_auc_score(y[m], own[m]), "auc_own_lo": a_lo, "auc_own_hi": a_hi,
                     "auc_shuffled": roc_auc_score(y[m], stranger[m]),
                     "auc_shuffled_lo": s_lo, "auc_shuffled_hi": s_hi})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT, index=False, float_format="%.6f")
    print(pd.DataFrame(rows).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
