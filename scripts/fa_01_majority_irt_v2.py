#!/usr/bin/env python
"""Majority and IRT-2PL baselines on the rebuilt FA manifest (v2).

Item parameters are fitted on TRAIN; each VAL learner's theta is calibrated on the first
`calib_frac` of their own chronological history (`eval.protocol.calib_eval_split`) and the
rest is scored. This is the protocol of `p3_01_baselines.py`, run on the v2 manifest and
learner split instead of the earlier (v1) ones. Two differences from v1 drive the
reported v1 -> v2 delta:

  1. Exact-ID deduplication and the new learner split (train 3,500 instead of 4,000; the
     val learners are the same).
  2. Missing platform scores are excluded from every fit and evaluation, never imputed
     (v1's `fa.py::load_interactions` mapped all 8,952 missing scores to `correct=0`).
     The majority constant changes too: v1 averaged all train rows (missing counted as
     0); v2 averages only score-labelled train rows.

Two scores are reported: MARGINAL (all labelled VAL eval-half events) and the
`support=NONE` stratum (attempts without support, i.e. solve-given-attempt under the
two-process reading of correctness). The chronological calibration split point is computed
over every event, including support-missing rows (they are still real practice history),
but only score-labelled rows are fitted or scored.

Inputs: the v2 manifest and splits (`fa_00_manifest_rebuild.py`), `configs/irt.yaml`.
Output: `reports/fa_01_majority_irt_v2.md`. Run: `python scripts/fa_01_majority_irt_v2.py`.
Serves the Majority and IRT baselines reported in Chapter 6.
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

from atom.data.fa_manifest import load_v2_events  # noqa: E402
from atom.eval.metrics import accuracy, auc, brier, ece  # noqa: E402
from atom.eval.protocol import calib_eval_split  # noqa: E402
from atom.eval.report import gate_report  # noqa: E402
from atom.models.irt import fit_2pl, predict_2pl  # noqa: E402
from atom.utils.logging import RunLogger  # noqa: E402
from atom.utils.seeds import seed_everything  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[1]

# v1 reference from p3_01_baselines.py (calib_frac=0.5, VAL; the development_exposed split
# is not evaluated here). Hard-coded rather than re-derived: that script is deterministic,
# so re-running it would reproduce the same numbers.
V1_VAL_AUC = {"irt": 0.770, "majority": 0.500}


def _metrics(y, p) -> dict:
    return {"auc": auc(y, p), "accuracy": accuracy(y, p), "brier": brier(y, p), "ece": ece(y, p)}


def _leakage_violations(calib_rows, eval_rows) -> int:
    bad = 0
    for learner in set(calib_rows["learner"]) & set(eval_rows["learner"]):
        c_max = calib_rows.loc[calib_rows["learner"] == learner, "t"].max()
        e_min = eval_rows.loc[eval_rows["learner"] == learner, "t"].min()
        if c_max >= e_min:
            bad += 1
    return bad


def main() -> int:
    irt_cfg = yaml.safe_load((REPO / "configs" / "irt.yaml").read_text())
    v2cfg = yaml.safe_load((REPO / "configs" / "data.yaml").read_text())["fa"]["manifest_v2"]
    seed_everything(irt_cfg["seed"])
    logger = RunLogger(config={"task": "fa_01_majority_irt_v2", **irt_cfg},
                       data_version="FA-v2-2026-08-09")

    logger.heartbeat(step=0, force=True, stage="loading FA-0 v2 manifest + splits")
    ev = load_v2_events(REPO / v2cfg["manifest_out"], REPO / v2cfg["learner_split_out"],
                        REPO / v2cfg["item_split_out"])
    ev = ev[ev["split"].isin(["train", "val"])].copy()
    all_pids = sorted(ev["problem_id"].unique())
    item_of = {p: i for i, p in enumerate(all_pids)}
    ev["item"] = ev["problem_id"].map(item_of)

    train_ev = ev[ev["split"] == "train"]
    train_labeled = train_ev[~train_ev["score_missing"]]
    n_train_missing = int(train_ev["score_missing"].sum())
    logger.heartbeat(step=0, force=True, stage="events loaded", n_train=len(train_ev),
                     n_train_labeled=len(train_labeled), n_train_score_missing=n_train_missing)

    # -- Stage 1: item params, fit on TRAIN's LABELLED rows only -----------------------------
    train_learners = sorted(train_labeled["learner"].unique())
    tperson_of = {u: i for i, u in enumerate(train_learners)}
    tp = torch.tensor(train_labeled["learner"].map(tperson_of).values, dtype=torch.long)
    ti = torch.tensor(train_labeled["item"].values, dtype=torch.long)
    ty = torch.tensor(train_labeled["score"].astype("int64").values, dtype=torch.long)
    item_params = fit_2pl(tp, ti, ty, n_persons=len(train_learners), n_items=len(all_pids),
                          epochs=irt_cfg["train_epochs"], lr=irt_cfg["train_lr"],
                          l2_theta=irt_cfg["l2_theta"], l2_log_a=irt_cfg["l2_log_a"],
                          l2_b=irt_cfg["l2_b"], seed=irt_cfg["seed"])
    train_auc = auc(ty.numpy(), predict_2pl(item_params, tp, ti).numpy())
    logger.heartbeat(step=1, force=True, stage="item params fit", train_insample_auc=round(train_auc, 4))

    # -- Stage 2: VAL calib/eval split (canonical calib_frac from irt.yaml) ------------------
    val_ev = ev[ev["split"] == "val"]
    calib_rows, eval_rows = calib_eval_split(val_ev, irt_cfg["calib_frac"], irt_cfg["min_interactions"])
    leakage = _leakage_violations(calib_rows, eval_rows)

    calib_labeled = calib_rows[~calib_rows["score_missing"]]
    eval_labeled = eval_rows[~eval_rows["score_missing"]]
    n_eval_missing = int(eval_rows["score_missing"].sum())

    calib_learners = sorted(set(calib_labeled["learner"]) | set(eval_labeled["learner"]))
    cperson_of = {u: i for i, u in enumerate(calib_learners)}
    cp = torch.tensor(calib_labeled["learner"].map(cperson_of).values, dtype=torch.long)
    ci = torch.tensor(calib_labeled["item"].values, dtype=torch.long)
    cy = torch.tensor(calib_labeled["score"].astype("int64").values, dtype=torch.long)
    calib_params = fit_2pl(cp, ci, cy, n_persons=len(calib_learners), n_items=len(all_pids),
                           epochs=irt_cfg["calib_epochs"], lr=irt_cfg["calib_lr"],
                           l2_theta=irt_cfg["l2_theta"], fixed_log_a=item_params.log_a,
                           fixed_b=item_params.b, seed=irt_cfg["seed"])

    ep = torch.tensor(eval_labeled["learner"].map(cperson_of).values, dtype=torch.long)
    ei = torch.tensor(eval_labeled["item"].values, dtype=torch.long)
    ey = eval_labeled["score"].astype("int64").values
    irt_pred = predict_2pl(calib_params, ep, ei).numpy()

    train_correct_rate = float(train_labeled["score"].astype("float32").mean())
    maj_pred = np.full_like(irt_pred, train_correct_rate, dtype=np.float64)

    is_none = (eval_labeled["support_code"].values == 0)
    n_none, n_marginal = int(is_none.sum()), len(ey)

    marginal = {"irt": _metrics(ey, irt_pred), "majority": _metrics(ey, maj_pred)}
    none_stratum = {
        "irt": _metrics(ey[is_none], irt_pred[is_none]),
        "majority": _metrics(ey[is_none], maj_pred[is_none]),
    }

    delta = {m: marginal[m]["auc"] - V1_VAL_AUC[m] for m in ("irt", "majority")}

    diagnostics_clean = leakage == 0 and abs(marginal["majority"]["auc"] - 0.5) < 0.05
    lo, hi = irt_cfg["gate_auc_lo"], irt_cfg["gate_auc_hi"]
    irt_marginal_auc = marginal["irt"]["auc"]
    band_status = ("below_band" if irt_marginal_auc < lo
                   else "above_band" if irt_marginal_auc > hi else "within_band")
    passed = diagnostics_clean and band_status != "below_band"

    logger.log_metrics(
        passed=passed, train_insample_auc=train_auc, leakage_violations=leakage,
        n_train_missing=n_train_missing, n_eval_missing=n_eval_missing,
        train_correct_rate=train_correct_rate,
        irt_marginal_auc=marginal["irt"]["auc"], irt_none_auc=none_stratum["irt"]["auc"],
        majority_marginal_auc=marginal["majority"]["auc"],
        majority_none_auc=none_stratum["majority"]["auc"],
        irt_delta_v1_to_v2=delta["irt"], majority_delta_v1_to_v2=delta["majority"],
        band_status=band_status,
    )
    logger.summary(status="ok" if passed else "gate-fail", band_status=band_status)

    def row(d):
        return f"AUC {d['auc']:.3f} · acc {d['accuracy']:.3f} · ECE {d['ece']:.3f} · Brier {d['brier']:.3f}"

    gate_report(
        path="reports/fa_01_majority_irt_v2.md",
        title="Majority and IRT-2PL baselines on the rebuilt FA manifest (v2)",
        meta={"run_id": logger.run_id, "git_sha": logger.git_sha, "config_hash": logger.config_hash,
              "train_learners (v2, labelled)": len(train_learners),
              "calib_learners (v2 val)": len(calib_learners), "band_status": band_status},
        metrics={
            "IRT marginal (VAL, n={:,})".format(n_marginal): row(marginal["irt"]),
            "IRT support=NONE stratum (n={:,})".format(n_none): row(none_stratum["irt"]),
            "Majority marginal (VAL)": row(marginal["majority"]),
            "Majority support=NONE stratum": row(none_stratum["majority"]),
            "v1 -> v2 delta, IRT (VAL AUC)": f"{V1_VAL_AUC['irt']:.3f} -> {marginal['irt']['auc']:.3f}  (Δ{delta['irt']:+.3f})",
            "v1 -> v2 delta, Majority (VAL AUC)": f"{V1_VAL_AUC['majority']:.3f} -> {marginal['majority']['auc']:.3f}  (Δ{delta['majority']:+.3f})",
            "train correct rate (v2, labelled only)": round(train_correct_rate, 4),
            "IRT train in-sample AUC (diagnostic)": round(train_auc, 4),
            "n_train_missing / n_eval_missing (excluded, not imputed)": f"{n_train_missing:,} / {n_eval_missing:,}",
            "leakage violations": leakage,
        },
        passed=passed,
        gate_desc="Sanity band [{:.2f}, {:.2f}] is diagnostic; PASS requires zero calib/eval "
                  "leakage, majority AUC within 0.05 of 0.5 (v2's true constant "
                  "may differ slightly from v1's exact 0.500 since it now excludes missing "
                  "scores rather than averaging them in as zeros), and AUC not below the "
                  "band.".format(lo, hi),
        notes=[
            "v1 reference (p3_01_baselines.py on the earlier manifest, calib_frac=0.5, VAL, "
            "hard-coded, not re-run): "
            f"IRT {V1_VAL_AUC['irt']:.3f}, Majority {V1_VAL_AUC['majority']:.3f}.",
            "The v1->v2 delta combines every manifest change at once (exact-ID dedup, the new "
            "learner split 4000->3500 train, and missing scores excluded rather than "
            "imputed to 0): this script does not separate the effects, it reports only "
            "their combined result.",
            "support=NONE stratum has no v1 analogue (v1 never computed this cut); it is "
            "reported alongside the marginal score under the two-process reading of "
            "correctness, not as a delta.",
            f"Majority's constant is the v2 TRAIN correct rate over LABELLED rows only "
            f"({train_correct_rate:.4f}) -- v1's 0.613 mixed missing-as-0 into the same "
            "average; the two are not measuring the same quantity.",
            "IRT/Majority never use item_codec or the filter, so the query-independent "
            "belief does not apply here; the filter's v2 baseline is in "
            "scripts/fa_01_s1v2.py.",
        ],
    )
    print(f"\nFA-1 (majority+IRT v2): {'PASS' if passed else 'FAIL'} · band {band_status} · "
          f"report: reports/fa_01_majority_irt_v2.md")
    print(f"  IRT marginal {marginal['irt']['auc']:.3f} (v1 {V1_VAL_AUC['irt']:.3f}, "
          f"Δ{delta['irt']:+.3f}) · support=NONE {none_stratum['irt']['auc']:.3f}")
    print(f"  Majority marginal {marginal['majority']['auc']:.3f} (v1 {V1_VAL_AUC['majority']:.3f}, "
          f"Δ{delta['majority']:+.3f}) · support=NONE {none_stratum['majority']['auc']:.3f}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
