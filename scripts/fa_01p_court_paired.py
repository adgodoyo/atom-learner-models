#!/usr/bin/env python
"""Paired learner-clustered bootstrap contrasts between predictors on the shared FA court.

The marginal CIs from `fa_01p_court_baselines.py` overlap, but every arm is scored on the
same rows, so the correlated part of the noise cancels in a paired test. This script applies
the same clustered bootstrap (learners resampled with replacement, seed 20260824, 1000 draws)
to the difference in AUC and in accuracy for each pair of arms.

Input: `results/fa_court_baselines/court_predictions.csv.gz` (per-row predictions, no refit).
Output: `results/fa_court_baselines/paired.json`. Run: `python scripts/fa_01p_court_paired.py`.
Serves Table 8 (FA locating).
"""
from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

REPO = pathlib.Path(__file__).resolve().parents[1]
OUT = REPO / "results" / "fa_court_baselines"
NBOOT, SEED, THR = 1000, 20260824, 0.5


def paired(y, pa, pb, learners, stat):
    rng = np.random.default_rng(SEED)
    uniq = np.unique(learners)
    idx = {l: np.flatnonzero(learners == l) for l in uniq}
    pack = np.stack([y, pa, pb], axis=1)
    point = stat(pack)
    draws = np.empty(NBOOT)
    for b in range(NBOOT):
        rows = np.concatenate([idx[l] for l in rng.choice(uniq, size=len(uniq), replace=True)])
        draws[b] = stat(pack[rows])
    lo, hi = np.nanpercentile(draws, [2.5, 97.5])
    return round(float(point), 4), round(float(lo), 4), round(float(hi), 4)


def d_auc(a):
    y = a[:, 0]
    if y.min() == y.max():
        return np.nan
    return roc_auc_score(y, a[:, 1]) - roc_auc_score(y, a[:, 2])


def d_acc(a):
    y = a[:, 0]
    return float(((a[:, 1] >= THR) == y).mean() - ((a[:, 2] >= THR) == y).mean())


def main() -> int:
    df = pd.read_csv(OUT / "court_predictions.csv.gz")
    sub = df[df["p_akt"].notna()].copy()
    y, lr = sub["y"].to_numpy().astype(float), sub["learner"].to_numpy()
    print(f"common court: {len(sub)} rows, {len(np.unique(lr))} learners\n")

    pairs = [
        ("ATOM  -  AKT",            "p_atom_personalised", "p_akt"),
        ("ATOM  -  IRT (causal)",   "p_atom_personalised", "p_irt_causal"),
        ("AKT   -  IRT (causal)",   "p_akt",               "p_irt_causal"),
        ("ATOM  -  ATOM shuffled-u", "p_atom_personalised", "p_atom_shuffled_u"),
    ]
    out = {}
    print(f"{'contrast':30s}{'dAUC':>9s}{'95% CI':>22s}{'  ':2s}{'dAcc':>8s}{'95% CI':>22s}  verdict")
    for name, ca, cb in pairs:
        a, b = sub[ca].to_numpy(), sub[cb].to_numpy()
        pa, la, ha = paired(y, a, b, lr, d_auc)
        pc, lc, hc = paired(y, a, b, lr, d_acc)
        clear = "CI-clear" if (la > 0 or ha < 0) else "not CI-clear"
        out[name] = {"d_auc": pa, "d_auc_ci95": [la, ha], "d_acc": pc,
                     "d_acc_ci95": [lc, hc], "auc_verdict": clear}
        print(f"{name:30s}{pa:>+9.4f}  [{la:+.4f}, {ha:+.4f}]  {pc:>+8.4f}  "
              f"[{lc:+.4f}, {hc:+.4f}]  {clear}")

    (OUT / "paired.json").write_text(json.dumps(
        {"n": int(len(sub)), "nboot": NBOOT, "seed": SEED, "contrasts": out}, indent=1))
    print(f"\nwrote {OUT / 'paired.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
