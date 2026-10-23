"""Probe whether item difficulty and discrimination are recoverable from the frozen problem-text codec.

Asks the FoundationalASSIST paper's Tasks 3 and 4 ("which of two problems is harder?" and
"which is more discriminating?") of ATOM's item pathway. Ground truth is a 2PL IRT fit on the
training learners only (items with at least 50 responses); ridge and MLP probes on the frozen
codec vectors are trained on training-split items and scored on held-out items by R^2,
Spearman and pairwise accuracy, in the paper's difference strata (0.1/0.5/1.0) and in our own
terciles, since IRT scales differ between fits. The probes are supervised on other items'
response data, whereas the paper's LLMs are zero-shot, so only the recoverability from text
is comparable, not the amount of supervision. Inputs: `artifacts/fa_manifest_v2.parquet`,
`artifacts/problem_codec_mean_L15.npz`. Output: `results/fa_pg_probe/probe.json`.
Run: `python scripts/fa_pg_probe.py`. Serves Table 7.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from atom.models.irt import fit_2pl  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results" / "fa_pg_probe"
OUT.mkdir(parents=True, exist_ok=True)
RNG = np.random.default_rng(20260824)
MIN_RESP = 50          # the paper's own item filter
N_PAIRS = 20000

# ------------------------------------------------------------------ ground truth (IRT)
man = pd.read_parquet(REPO / "artifacts" / "fa_manifest_v2.parquet",
                      columns=["learner", "problem_id", "score", "score_missing",
                               "support_code", "split_learner", "split_item"])
man = man[~man["score_missing"]].copy()
train = man[man["split_learner"] == "train"]
print(f"[fa-pg] train events {len(train):,}  learners {train['learner'].nunique():,} "
      f"items {train['problem_id'].nunique():,}")

learners = {l: i for i, l in enumerate(train["learner"].unique())}
items = {p: i for i, p in enumerate(sorted(train["problem_id"].unique()))}
pi = torch.tensor(train["learner"].map(learners).to_numpy(), dtype=torch.long)
ii = torch.tensor(train["problem_id"].map(items).to_numpy(), dtype=torch.long)
yy = torch.tensor(train["score"].to_numpy(), dtype=torch.float32)
params = fit_2pl(pi, ii, yy, n_persons=len(learners), n_items=len(items), epochs=300, lr=0.05)
b = params.b.numpy()
a = params.a.numpy()
counts = train.groupby("problem_id").size()

irt = pd.DataFrame({"problem_id": list(items.keys()), "b": b, "a": a})
irt["n_resp"] = irt["problem_id"].map(counts).fillna(0).astype(int)
irt = irt[irt["n_resp"] >= MIN_RESP]
item_split = man.drop_duplicates("problem_id").set_index("problem_id")["split_item"]
irt["split_item"] = irt["problem_id"].map(item_split)
print(f"[fa-pg] IRT items with >={MIN_RESP} responses: {len(irt)}  "
      f"b range [{irt['b'].min():.2f},{irt['b'].max():.2f}] sd {irt['b'].std():.2f} | "
      f"a range [{irt['a'].min():.2f},{irt['a'].max():.2f}] sd {irt['a'].std():.2f}")
print(irt["split_item"].value_counts().to_dict())

# ------------------------------------------------------------------ features (frozen codec)
z = np.load(REPO / "artifacts" / "problem_codec_mean_L15.npz", allow_pickle=True)
codec = pd.DataFrame(z["codec"])
codec["problem_id"] = z["problem_id"]
d = irt.merge(codec, on="problem_id", how="inner")
feat_cols = [c for c in d.columns if isinstance(c, int)]
print(f"[fa-pg] items with codec vectors: {len(d)} ({len(feat_cols)} dims)")

tr = d[d["split_item"] == "train"]
te = d[d["split_item"] != "train"]
print(f"[fa-pg] probe train items {len(tr)} | held-out items {len(te)} "
      f"(splits held out: {sorted(te['split_item'].unique())})")

scaler = StandardScaler().fit(tr[feat_cols].to_numpy())
Xtr, Xte = scaler.transform(tr[feat_cols].to_numpy()), scaler.transform(te[feat_cols].to_numpy())


def probes(target_tr, target_te):
    out = {}
    ridge = RidgeCV(alphas=np.logspace(-2, 4, 25)).fit(Xtr, target_tr)
    out["ridge"] = ridge.predict(Xte)
    mlp = MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=2000, random_state=0,
                       early_stopping=True, n_iter_no_change=25).fit(Xtr, target_tr)
    out["mlp"] = mlp.predict(Xte)
    return out


def pairwise(pred, truth, label):
    n = len(truth)
    ia, ib = RNG.integers(0, n, N_PAIRS), RNG.integers(0, n, N_PAIRS)
    keep = ia != ib
    ia, ib = ia[keep], ib[keep]
    dt = np.abs(truth[ia] - truth[ib])
    right = ((truth[ia] > truth[ib]) == (pred[ia] > pred[ib]))
    q1, q2 = np.quantile(dt, [1 / 3, 2 / 3])
    strata = {
        f"all(|d{label}|>=0.1)": dt >= 0.1,
        "paper_small(0.1-0.5)": (dt >= 0.1) & (dt < 0.5),
        "paper_medium(0.5-1.0)": (dt >= 0.5) & (dt < 1.0),
        "paper_large(>1.0)": dt >= 1.0,
        f"ours_tercile_1(<{q1:.2f})": dt < q1,
        f"ours_tercile_2({q1:.2f}-{q2:.2f})": (dt >= q1) & (dt < q2),
        f"ours_tercile_3(>{q2:.2f})": dt >= q2,
    }
    return {k: {"n_pairs": int(v.sum()),
                "accuracy": float(right[v].mean()) if v.sum() else None}
            for k, v in strata.items()}


report = {"n_items_irt": int(len(irt)), "n_items_with_codec": int(len(d)),
          "n_probe_train_items": int(len(tr)), "n_heldout_items": int(len(te)),
          "min_responses_per_item": MIN_RESP,
          "irt_scale": {"b_sd": float(irt["b"].std()), "b_min": float(irt["b"].min()),
                        "b_max": float(irt["b"].max()), "a_sd": float(irt["a"].std()),
                        "a_min": float(irt["a"].min()), "a_max": float(irt["a"].max())},
          "paper_scale_note": "paper reports b in [-1.35, 0.91], a in [0.01, 0.91]"}

for tname, col in (("difficulty_b", "b"), ("discrimination_a", "a")):
    ytr, yte = tr[col].to_numpy(), te[col].to_numpy()
    preds = probes(ytr, yte)
    cell = {}
    for pname, p in preds.items():
        ss_res = float(np.sum((yte - p) ** 2))
        ss_tot = float(np.sum((yte - yte.mean()) ** 2))
        cell[pname] = {
            "r2_heldout_items": 1 - ss_res / ss_tot,
            "spearman": float(spearmanr(p, yte).statistic),
            "pairwise": pairwise(p, yte, col),
        }
    # a learner-blind, text-free control: predict the mean (chance on pairwise)
    cell["chance"] = {"pairwise_accuracy": 0.5}
    report[tname] = cell

(OUT / "probe.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
