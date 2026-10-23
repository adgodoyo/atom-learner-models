"""Tests whether the soft-match generation metrics (edit distance, codec similarity) for FA-SIM3
are a by-product of exact match, by re-analysing already-measured per-row scores.

Input: `results/fa_sim3_eval_5e5f/sim_scores.csv.gz` (no generation, no model). For each comparator
arm (dw0, shuffled_u, u_mean) against the personalised arm, four conditioning sets:
  ALL        - marginal, as reported (sanity check: must reproduce that run's summary.json)
  P_MISS     - rows where the personalised arm missed exact match (selection-biased: the
               comparator's rows are chosen by the personalised arm's own outcome)
  BOTH_MISS  - rows where both arms of the pair missed (arm-symmetric; the defensible conditional)
  NONE_MATCH - rows where all four arms missed (one common population for the whole panel)
plus an arithmetic decomposition of the marginal gap. Learner-clustered bootstrap CIs.
Output: `results/fa_sim3_softmatch_conditional/summary.json`.
Run: `python scripts/fa_sim3_softmatch_conditional.py`.  Serves Tables 10 and 12.
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from atom.eval.metrics import learner_clustered_bootstrap_ci  # noqa: E402

d = pd.read_csv(REPO / "results/fa_sim3_eval_5e5f/sim_scores.csv.gz")
ARMS = ("personalised", "dw0", "shuffled_u", "u_mean")
piv = {a: d[d.arm == a].sort_values("row_idx").reset_index(drop=True) for a in ARMS}
n = len(piv["personalised"])
for a in ARMS:
    assert len(piv[a]) == n
    assert (piv[a].row_idx.to_numpy() == piv["personalised"].row_idx.to_numpy()).all(), a
learners = piv["personalised"].learner.to_numpy()

em = {a: piv[a].exact_match_free.to_numpy() for a in ARMS}
print(f"n rows/arm = {n}; exact-match rate: " +
      ", ".join(f"{a}={em[a].mean():.4f}" for a in ARMS))
none_match = np.ones(n, bool)
for a in ARMS:
    none_match &= (em[a] == 0)
print(f"rows where ALL FOUR arms missed exact match: {none_match.sum()}/{n} "
      f"({none_match.mean():.1%})\n")

METRICS = [("normalised_edit_distance_free", "edit distance (lower better)", True),
           ("sim_vs_real_free", "codec similarity (higher better)", False)]
out = {}
for col, label, lower_better in METRICS:
    print(f"=== {label} ===")
    v = {a: piv[a][col].to_numpy() for a in ARMS}
    print(f"  arm means (ALL rows): " + ", ".join(f"{a}={v[a].mean():.4f}" for a in ARMS))
    for comp in ("dw0", "shuffled_u", "u_mean"):
        both_miss = (em["personalised"] == 0) & (em[comp] == 0)
        sets = {"ALL": np.ones(n, bool), "P_MISS": em["personalised"] == 0,
                "BOTH_MISS": both_miss, "NONE_MATCH": none_match}
        for sname, mask in sets.items():
            if mask.sum() < 5:
                print(f"  personalised - {comp:11s} [{sname:10s}] n={mask.sum()} too few"); continue
            r = learner_clustered_bootstrap_ci(v["personalised"][mask], v[comp][mask],
                                               learners[mask], n_boot=2000, seed=0)
            star = "*" if r["excludes_zero"] else " "
            print(f"  personalised - {comp:11s} [{sname:10s}] n={mask.sum():3d} "
                  f"({r['n_learners']:3d} learners)  d={r['point_delta']:+.4f} "
                  f"[{r['ci95'][0]:+.4f},{r['ci95'][1]:+.4f}] {star}")
            out[f"{col}|{comp}|{sname}"] = r
    print()

# how much of the marginal gap is arithmetically carried by the exact-match rows?
print("=== decomposition of the ALL-rows gap (personalised - comparator) ===")
for col, label, _ in METRICS:
    v = {a: piv[a][col].to_numpy() for a in ARMS}
    for comp in ("dw0", "shuffled_u", "u_mean"):
        total = v["personalised"].mean() - v[comp].mean()
        both_miss = (em["personalised"] == 0) & (em[comp] == 0)
        # contribution of the rows where the two arms DISAGREE on exact match
        disagree = (em["personalised"] != em[comp])
        c_dis = (v["personalised"][disagree] - v[comp][disagree]).sum() / n
        c_bm = (v["personalised"][both_miss] - v[comp][both_miss]).sum() / n
        c_bh = total - c_dis - c_bm
        print(f"  {label[:14]:14s} vs {comp:11s} total={total:+.4f} | "
              f"exact-match-disagreement rows={c_dis:+.4f} ({abs(c_dis)/max(abs(total),1e-9):5.1%}) | "
              f"both-miss rows={c_bm:+.4f} | both-hit rows={c_bh:+.4f}")
json.dump(out, open(REPO / "results/fa_sim3_softmatch_conditional/summary.json", "w"), indent=2, default=str)
