"""Model-free person-signal comparison between CodeWorkout and FoundationalASSIST.

Computes, with the same arithmetic on the raw logs of both corpora (no model), two quantities:
(1) a variance decomposition: after removing item difficulty, the share of outcome variance
that belongs to the person versus the item, plus split-half reliability of the person
residual; (2) the value of k observations: the correlation between a person estimate from k
random observations and the same person's remaining items. Both corpora use first exposures
only (CodeWorkout: first submission per learner and problem; FA: `first_exposure`), binary
outcomes and the same seeds and resample counts. FA is reported with and without give-ups,
because its `correct` label mixes not giving up with solving, and the gap between the two rows
measures the help-seeking component.

Inputs: `artifacts/cw_first_submissions.parquet` (or the CodeWorkout manifest) and
`artifacts/fa_manifest_v2.parquet`. Output: `results/corpus_person_signal/panel.json`.
Run: `python scripts/corpus_person_signal.py` (deterministic, CPU, about two minutes).
Serves Tables 13 and 14 (Chapter 8).
"""
from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd

REPO = pathlib.Path(__file__).resolve().parents[1]
OUT = REPO / "results" / "corpus_person_signal"
SEED_SPLIT, SEED_K = 0, 1
REPS_SPLIT, REPS_K = 25, 30
KS = (1, 2, 3, 5, 10, 20)
MIN_ITEMS_SPLIT = 20     # for the variance/reliability panel
MIN_ITEMS_K = 45         # for the value-of-k curve: every learner must have >= 45 to spare


def _prep(df: pd.DataFrame, learner: str, item: str, y: str, min_items: int) -> pd.DataFrame:
    """One row per (learner, item) observation, with item difficulty removed.

    `r` is the residual y - mean(y | item): the part of the outcome NOT explained by
    which item it was. Person variance is then measured on `r`, so it cannot be inflated
    by a corpus simply having more variable item difficulty.
    """
    d = df[[learner, item, y]].dropna().copy()
    d.columns = ["lr", "it", "y"]
    d["r"] = d["y"] - d.groupby("it")["y"].transform("mean")
    counts = d.groupby("lr").size()
    return d[d.lr.isin(counts[counts >= min_items].index)].copy()


def variance_panel(df, learner, item, y, name, min_items=MIN_ITEMS_SPLIT):
    """Split-half reliability of the person residual + the person/item variance split."""
    d = _prep(df, learner, item, y, min_items)
    rng = np.random.default_rng(SEED_SPLIT)
    cors = []
    for _ in range(REPS_SPLIT):
        d["h"] = rng.integers(0, 2, len(d))
        g = d.groupby(["lr", "h"])["r"].mean().unstack().dropna()
        if len(g) > 10:
            cors.append(float(np.corrcoef(g[0], g[1])[0, 1]))
    r = float(np.mean(cors))
    total = float(d["y"].var())
    v_item = float(d.groupby("it")["y"].mean().var())
    v_person = float(d.groupby("lr")["r"].mean().var())   # person variance AFTER difficulty
    return {
        "corpus": name,
        "n_events": int(len(d)),
        "n_learners": int(d.lr.nunique()),
        "n_items": int(d.it.nunique()),
        "median_items_per_learner": float(d.groupby("lr").size().median()),
        "base_rate": round(float(d["y"].mean()), 4),
        "split_half_r": round(r, 4),
        "reliability_spearman_brown": round(2 * r / (1 + r), 4),
        "var_total": round(total, 4),
        "pct_variance_item": round(100 * v_item / total, 2),
        "pct_variance_person": round(100 * v_person / total, 2),
        "person_to_item_ratio": round((v_person / total) / (v_item / total), 3),
    }


def value_of_k(df, learner, item, y, name, ks=KS, min_items=MIN_ITEMS_K):
    """Correlation between a person-estimate from k observations and their REMAINING items."""
    d = _prep(df, learner, item, y, min_items)
    by_learner = {k: v["r"].to_numpy() for k, v in d.groupby("lr")}
    learners = [k for k, v in by_learner.items() if len(v) >= min_items]
    rng = np.random.default_rng(SEED_K)
    curve = {}
    for k in ks:
        cs = []
        for _ in range(REPS_K):
            a, b = [], []
            for lr in learners:
                v = by_learner[lr]
                p = rng.permutation(len(v))
                a.append(v[p[:k]].mean())
                b.append(v[p[k:]].mean())
            cs.append(float(np.corrcoef(a, b)[0, 1]))
        curve[str(k)] = round(float(np.mean(cs)), 4)
    return {"corpus": name, "n_learners": len(learners), "curve": curve}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    cw = pd.read_parquet(REPO / "artifacts" / "cw_first_submissions.parquet") \
        if (REPO / "artifacts" / "cw_first_submissions.parquet").exists() else None
    if cw is None:
        from atom.data.cw_manifest import load_attempts_with_episodes
        a = load_attempts_with_episodes().sort_values(["learner", "t"])
        cw = a.groupby(["learner", "problem_id"], as_index=False).first()
        cw["correct"] = (cw["score_frac"] >= 1.0).astype(int)

    fa = pd.read_parquet(
        REPO / "artifacts" / "fa_manifest_v2.parquet",
        columns=["learner", "problem_id", "score", "score_missing", "saw_answer", "first_exposure"],
    )
    fa = fa[(~fa.score_missing) & (fa.first_exposure)]

    corpora = [
        (cw, "learner", "problem_id", "correct", "CodeWorkout - first submissions"),
        (fa, "learner", "problem_id", "score", "FoundationalASSIST - first exposures (all)"),
        (fa[fa.saw_answer == 0], "learner", "problem_id", "score",
         "FoundationalASSIST - first exposures, give-ups removed"),
    ]

    variance = [variance_panel(*c) for c in corpora]
    curves = [value_of_k(*c) for c in corpora]

    print("\n=== 1. Variance decomposition (item difficulty removed first) ===")
    print(pd.DataFrame(variance).to_string(index=False))
    print("\n=== 2. Value of k observations ===")
    print("corr(person estimate from k observations, that person's remaining items)\n")
    hdr = f"{'corpus':56s}{'n':>7s}" + "".join(f"{'k=' + str(k):>9s}" for k in KS)
    print(hdr)
    for c in curves:
        print(f"{c['corpus']:56s}{c['n_learners']:>7d}"
              + "".join(f"{c['curve'][str(k)]:>9.4f}" for k in KS))

    payload = {
        "seeds": {"split_half": SEED_SPLIT, "value_of_k": SEED_K},
        "reps": {"split_half": REPS_SPLIT, "value_of_k": REPS_K},
        "min_items": {"variance_panel": MIN_ITEMS_SPLIT, "value_of_k": MIN_ITEMS_K},
        "variance_decomposition": variance,
        "value_of_k": curves,
    }
    (OUT / "panel.json").write_text(json.dumps(payload, indent=1))
    print(f"\nwrote {OUT / 'panel.json'}")


if __name__ == "__main__":
    main()
