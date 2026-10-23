"""FA-SIM3 knowledge-tracing and cognitive-accuracy panel, reported in the shape used by the
FoundationalASSIST paper; reads persisted artifacts only (no training, no GPU).

Inputs: `results/fa_sim3_full_run/` (predictions.csv.gz, belief_diagnostics.npz),
`artifacts/fa_manifest_v2.parquet`, `artifacts/fa_problem_type_map.csv`, and optionally
`results/fa_sim3_eval_5e5f/sim_scores.csv.gz`.
Computes: (1) binary correctness AUC/accuracy from the 5-class outcome head, per arm, on four
evaluation sets (marginal; unaided rows only; t>=50, the paper's warm-up regime; unaided and t>=50),
plus learner-blind item-rate baselines; (2) accuracy split by ground truth (optimism bias);
(3) exact-match cognitive accuracy by ground truth x problem type; (4) soft generation metrics by
stratum; (5) personalisation gap against history length; (6) an indicative item-difficulty
pairwise comparison. All CIs are learner-clustered bootstraps (1000 draws) over the val learners.
Output: `results/fa_sim3_kt_panel/panel.json`.  Run: `python scripts/fa_sim3_kt_panel.py`.
Serves Table 8 (FA locating).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[1]
RUN = REPO / "results" / "fa_sim3_full_run"
OUT = REPO / "results" / "fa_sim3_kt_panel"
OUT.mkdir(parents=True, exist_ok=True)
RNG = np.random.default_rng(20260824)
NBOOT = 1000
ARMS = ("personalised", "dw0", "shuffled_u", "u_mean")
CLASSES = ("UNAIDED_CORRECT", "UNAIDED_WRONG", "HINT_ONLY", "REVEAL_ONLY", "HINT_AND_REVEAL")


def cluster_boot(values_by_row: np.ndarray, learners: np.ndarray, stat, nboot=NBOOT):
    """Learner-clustered bootstrap of `stat(values_of_resampled_rows)`."""
    uniq = np.unique(learners)
    idx_by_learner = {l: np.flatnonzero(learners == l) for l in uniq}
    point = stat(values_by_row)
    draws = np.empty(nboot)
    for b in range(nboot):
        pick = RNG.choice(uniq, size=len(uniq), replace=True)
        rows = np.concatenate([idx_by_learner[l] for l in pick])
        draws[b] = stat(values_by_row[rows])
    lo, hi = np.nanpercentile(draws, [2.5, 97.5])
    return float(point), float(lo), float(hi)


def auc_boot(y: np.ndarray, p: np.ndarray, learners: np.ndarray):
    pack = np.stack([y, p], axis=1)

    def stat(a):
        yy, pp = a[:, 0], a[:, 1]
        if yy.min() == yy.max():
            return np.nan
        return roc_auc_score(yy, pp)

    return cluster_boot(pack, learners, stat)


def paired_auc_diff(y, p_a, p_b, learners):
    pack = np.stack([y, p_a, p_b], axis=1)

    def stat(a):
        yy = a[:, 0]
        if yy.min() == yy.max():
            return np.nan
        return roc_auc_score(yy, a[:, 1]) - roc_auc_score(yy, a[:, 2])

    return cluster_boot(pack, learners, stat)


def mean_boot(v, learners):
    return cluster_boot(v.astype(float), learners, lambda a: float(np.mean(a)))


def ece(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    out = 0.0
    for i in range(bins):
        m = (p >= edges[i]) & (p < edges[i + 1] if i < bins - 1 else p <= 1.0)
        if m.sum() == 0:
            continue
        out += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(out)


# ---------------------------------------------------------------- load
pred = pd.read_csv(RUN / "predictions.csv.gz")
diag = np.load(RUN / "belief_diagnostics.npz")
ptype = (pd.read_csv(REPO / "artifacts" / "fa_problem_type_map.csv")
         .drop_duplicates("problem_id").set_index("problem_id")["problem_type"])
pred["problem_type"] = pred["problem_id"].map(ptype).fillna("UNKNOWN")
for arm in ARMS:
    post = diag[f"outcome_prob_meanK_full_{arm}"]
    assert post.shape[0] == len(pred)
    pred[f"p_correct_{arm}"] = post[:, 0]
    pred[f"p_unaided_{arm}"] = post[:, 0] + post[:, 1]
    with np.errstate(divide="ignore", invalid="ignore"):
        pred[f"p_solve_given_attempt_{arm}"] = post[:, 0] / np.clip(post[:, 0] + post[:, 1], 1e-9, None)

pred["y_correct"] = (pred["true_class"] == 0).astype(int)
pred["unaided"] = pred["true_class"].isin([0, 1])
learners_all = pred["learner"].to_numpy()

report: dict = {"n_rows": int(len(pred)), "n_learners": int(pred["learner"].nunique())}

# ---------------------------------------------------------------- (1)+(2) FKT courts
courts = {
    "marginal_all": np.ones(len(pred), bool),
    "marginal_t_ge_50": (pred["t"] >= 50).to_numpy(),
    "unaided_stratum": pred["unaided"].to_numpy(),
    "unaided_and_t_ge_50": (pred["unaided"] & (pred["t"] >= 50)).to_numpy(),
}
fkt = {}
for cname, mask in courts.items():
    sub = pred[mask]
    lz = sub["learner"].to_numpy()
    y = sub["y_correct"].to_numpy()
    entry = {"n": int(mask.sum()), "n_learners": int(sub["learner"].nunique()),
             "base_rate_correct": float(y.mean()),
             "majority_always_correct_acc": float(max(y.mean(), 1 - y.mean()))}
    unaided_court = cname.startswith("unaided")
    for arm in ARMS:
        p = sub[f"p_solve_given_attempt_{arm}" if unaided_court else f"p_correct_{arm}"].to_numpy()
        a, alo, ahi = auc_boot(y, p, lz)
        pr = (p >= 0.5).astype(int)
        entry[arm] = {
            "auc": a, "auc_ci": [alo, ahi],
            "acc": float((pr == y).mean()),
            "acc_when_student_correct": float((pr[y == 1] == 1).mean()) if (y == 1).any() else None,
            "acc_when_student_incorrect": float((pr[y == 0] == 0).mean()) if (y == 0).any() else None,
            "balanced_acc": float(0.5 * ((pr[y == 1] == 1).mean() + (pr[y == 0] == 0).mean())),
            "brier": float(np.mean((p - y) ** 2)), "ece": ece(y, p),
            "mean_p": float(p.mean()),
        }
    # personalisation contrast on the correctness channel
    d, dlo, dhi = paired_auc_diff(y, sub[("p_solve_given_attempt_personalised" if unaided_court
                                          else "p_correct_personalised")].to_numpy(),
                                  sub[("p_solve_given_attempt_shuffled_u" if unaided_court
                                       else "p_correct_shuffled_u")].to_numpy(), lz)
    entry["auc_personalised_minus_shuffled_u"] = {"delta": d, "ci": [dlo, dhi]}
    d, dlo, dhi = paired_auc_diff(y, sub[("p_solve_given_attempt_personalised" if unaided_court
                                          else "p_correct_personalised")].to_numpy(),
                                  sub[("p_solve_given_attempt_dw0" if unaided_court
                                       else "p_correct_dw0")].to_numpy(), lz)
    entry["auc_personalised_minus_dw0"] = {"delta": d, "ci": [dlo, dhi]}
    fkt[cname] = entry
report["fkt"] = fkt

# learner-blind item baselines (zero person information)
man = pd.read_parquet(REPO / "artifacts" / "fa_manifest_v2.parquet",
                      columns=["learner", "problem_id", "score", "score_missing",
                               "support_code", "split_learner"])
man = man[~man["score_missing"]]
man["y"] = ((man["support_code"] == 0) & (man["score"] == 1)).astype(int)
train = man[man["split_learner"] == "train"]
item_rate = train.groupby("problem_id")["y"].agg(["mean", "count"])
item_rate_unaided = (train[train["support_code"] == 0].groupby("problem_id")["score"]
                     .agg(["mean", "count"]))
base_all = float(train["y"].mean())

item_baselines = {}
for cname, mask in courts.items():
    sub = pred[mask].copy()
    lz = sub["learner"].to_numpy()
    y = sub["y_correct"].to_numpy()
    if cname.startswith("unaided"):
        src = item_rate_unaided
    else:
        src = item_rate
    p = sub["problem_id"].map(src["mean"]).fillna(base_all).to_numpy()
    a, alo, ahi = auc_boot(y, p, lz)
    item_baselines[cname] = {"item_only_train_rate_auc": a, "ci": [alo, ahi],
                             "coverage": float(sub["problem_id"].isin(src.index).mean())}
report["item_only_baseline"] = item_baselines

# ---------------------------------------------------------------- (3) Cog Acc grid
cog = {}
gt_defs = {
    "correct(UNAIDED_CORRECT)": pred["true_class"] == 0,
    "incorrect_unaided(UNAIDED_WRONG)": pred["true_class"] == 1,
    "incorrect_any(all non-correct classes)": pred["true_class"] != 0,
}
for gname, gmask in gt_defs.items():
    cell = {}
    for tname in ["ALL"] + sorted(pred["problem_type"].unique().tolist()):
        m = gmask & (pred["problem_type"] == tname if tname != "ALL" else True)
        if m.sum() < 5:
            continue
        sub = pred[m]
        row = {"n": int(m.sum())}
        for arm in ARMS:
            v = sub[f"exact_match_{arm}"].to_numpy()
            pt, lo, hi = mean_boot(v, sub["learner"].to_numpy())
            row[arm] = {"exact_match": pt, "ci": [lo, hi]}
        cell[tname] = row
    cog[gname] = cell
report["cognitive_accuracy"] = cog

# ---------------------------------------------------------------- (4) soft metrics
sim_path = REPO / "results" / "fa_sim3_eval_5e5f" / "sim_scores.csv.gz"
soft = {}
if sim_path.exists():
    sim = pd.read_csv(sim_path)
    sim["stratum"] = np.where(sim["true_class"] == "UNAIDED_CORRECT", "easy",
                              np.where(sim["true_class"] == "UNAIDED_WRONG", "hard", "help_used"))
    for stratum, g in sim.groupby("stratum"):
        cell = {"n_rows_per_arm": int(len(g) / g["arm"].nunique())}
        for arm, ga in g.groupby("arm"):
            lz = ga["learner"].to_numpy()
            e = mean_boot(ga["exact_match_free"].to_numpy(), lz)
            ned = mean_boot(ga["normalised_edit_distance_free"].to_numpy(), lz)
            sm = mean_boot(ga["sim_vs_real_free"].to_numpy(), lz)
            cell[arm] = {"exact": e, "ned": ned, "codec_sim": sm}
        soft[stratum] = cell
report["soft_generation_by_stratum"] = soft

# ---------------------------------------------------------------- (5) history curve
buckets = [(0, 10), (10, 25), (25, 50), (50, 100), (100, 200), (200, 10 ** 9)]
curve = []
for lo_t, hi_t in buckets:
    m = (pred["t"] >= lo_t) & (pred["t"] < hi_t)
    if m.sum() < 20:
        continue
    sub = pred[m]
    lz = sub["learner"].to_numpy()
    gap_dw0 = (sub["mixture_nll_dw0"] - sub["mixture_nll_personalised"]).to_numpy()
    gap_shuf = (sub["mixture_nll_shuffled_u"] - sub["mixture_nll_personalised"]).to_numpy()
    y = sub["y_correct"].to_numpy()
    auc_here = auc_boot(y, sub["p_correct_personalised"].to_numpy(), lz) if 0 < y.mean() < 1 else (np.nan,) * 3
    curve.append({
        "bucket": f"[{lo_t},{hi_t if hi_t < 10 ** 9 else 'inf'})", "n": int(m.sum()),
        "n_learners": int(sub["learner"].nunique()),
        "nll_personalised": float(sub["mixture_nll_personalised"].mean()),
        "adaptation_gap_vs_dw0": mean_boot(gap_dw0, lz),
        "identity_gap_vs_shuffled_u": mean_boot(gap_shuf, lz),
        "exact_match": mean_boot(sub["exact_match_personalised"].to_numpy(), lz),
        "correctness_auc": list(auc_here),
    })
report["history_curve"] = curve

# ---------------------------------------------------------------- (6) difficulty pairwise
val_item = pred.groupby("problem_id").agg(
    pred_p=("p_correct_personalised", "mean"), n_val=("p_correct_personalised", "size"))
gt = item_rate[item_rate["count"] >= 50]
j = val_item.join(gt, how="inner")
j = j[j["n_val"] >= 2]
diffs, correct_calls = [], []
ids = j.index.to_numpy()
n_pairs = 20000
if len(ids) >= 2:
    ia = RNG.integers(0, len(ids), n_pairs)
    ib = RNG.integers(0, len(ids), n_pairs)
    keep = ia != ib
    ia, ib = ia[keep], ib[keep]
    ga, gb = j["mean"].to_numpy()[ia], j["mean"].to_numpy()[ib]
    pa, pb = j["pred_p"].to_numpy()[ia], j["pred_p"].to_numpy()[ib]
    # difficulty = 1 - empirical success; ground truth "which is harder"
    gtruth = (ga < gb).astype(int)      # 1 => B harder
    call = (pa < pb).astype(int)
    d = np.abs(np.log(np.clip(ga, 1e-3, 1 - 1e-3) / (1 - np.clip(ga, 1e-3, 1 - 1e-3)))
               - np.log(np.clip(gb, 1e-3, 1 - 1e-3) / (1 - np.clip(gb, 1e-3, 1 - 1e-3))))
    ok = d >= 0.1
    strata = {"all(|dlogit|>=0.1)": ok,
              "small(0.1-0.5)": ok & (d < 0.5),
              "medium(0.5-1.0)": (d >= 0.5) & (d < 1.0),
              "large(>1.0)": d >= 1.0}
    report["difficulty_pairwise"] = {
        "n_items": int(len(j)), "note": "ground truth = TRAIN empirical unaided-correct rate "
                                        "(>=50 train responses); prediction = ATOM mean p(correct) "
                                        "over that item's val rows; items were SEEN in training.",
        **{k: {"n_pairs": int(v.sum()), "accuracy": float((call[v] == gtruth[v]).mean())}
           for k, v in strata.items()},
    }

(OUT / "panel.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2)[:400])
print(f"\n[fa-sim3-kt-panel] wrote {OUT / 'panel.json'}")
