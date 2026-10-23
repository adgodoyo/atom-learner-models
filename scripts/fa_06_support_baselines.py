#!/usr/bin/env python
"""Score two closed-form support baselines (learner-prefix and lag-1+gap) on the FA validation split.

Both are causal within each learner's own history, with every global statistic fit on the
training learners only (see `atom.eval.support_baselines`). They are scored by multiclass NLL
and Brier on two populations, kept separate: the fixed validation bundle that the trained
support head is evaluated on (matched by exact event id) and the full validation split
(lower-variance context). The self-check is that both baselines beat a global-frequency floor
on the bundle. Inputs: the v2 manifest and splits, `staging/bundles/fa_06_val_eval.tar.gz`.
Output: `reports/fa_06_support_baselines.md`. Run: `python scripts/fa_06_support_baselines.py`.
Serves Appendix A.
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import tarfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from atom.data.fa_manifest import load_v2_events  # noqa: E402
from atom.eval.metrics import multiclass_brier, multiclass_nll, with_population  # noqa: E402
from atom.eval.report import gate_report  # noqa: E402
from atom.eval.support_baselines import (fit_global_support_freq, fit_lag1_gap_table,  # noqa: E402
                                         lag1_gap_probs, learner_prefix_probs)
from atom.utils.logging import git_sha  # noqa: E402


def _score(probs: np.ndarray, target: np.ndarray, mask: np.ndarray | None = None) -> dict:
    p = probs[mask] if mask is not None else probs
    t = target[mask] if mask is not None else target
    return {"nll": multiclass_nll(p, t), "brier": multiclass_brier(p, t), "n": int(len(t))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest-path", default="artifacts/fa_manifest_v2.parquet")
    ap.add_argument("--learner-split-path", default="splits/fa_learner_split_v2.csv")
    ap.add_argument("--item-split-path", default="splits/fa_item_split_v2.csv")
    ap.add_argument("--val-bundle-path", default="staging/bundles/fa_06_val_eval.tar.gz",
                    help="the fixed bundle that infra/fa_06_stage2_train.py::val_evaluate "
                        "scores a trained head on; bundle-restricted numbers are the matched "
                        "comparison, full-val numbers are lower-variance context, not a "
                        "substitute.")
    ap.add_argument("--shrinkage-k", type=float, default=5.0)
    ap.add_argument("--report-path", default="reports/fa_06_support_baselines.md")
    args = ap.parse_args()

    ev = load_v2_events(args.manifest_path, args.learner_split_path, args.item_split_path)
    train_ev = ev[ev["split"] == "train"].sort_values(["learner", "t"], kind="mergesort")
    val_ev = ev[ev["split"] == "val"].sort_values(["learner", "t"], kind="mergesort")

    extract_dir = pathlib.Path("/tmp/fa_06_support_baselines_val_bundle")
    extract_dir.mkdir(exist_ok=True)
    with tarfile.open(args.val_bundle_path, "r:gz") as tf:
        tf.extractall(extract_dir, filter="data")
    bundle_ids = set(pd.read_json(extract_dir / "events.jsonl", lines=True)["id"])
    bundle_mask = val_ev["id"].isin(bundle_ids).to_numpy()
    n_bundle_found = int(bundle_mask.sum())
    if n_bundle_found != len(bundle_ids):
        raise AssertionError(f"val bundle has {len(bundle_ids)} rows but only "
                             f"{n_bundle_found} were found in the val split's has_valid_time "
                             f"events -- population mismatch, do not trust the bundle numbers "
                             f"below until this is understood")

    global_freq = fit_global_support_freq(train_ev)
    target = val_ev["support_code"].to_numpy().astype(int)
    n_val = len(val_ev)

    # floor: the same global_freq row repeated for every val example, ignoring history
    # entirely -- what "no causal signal used at all" scores, so a baseline that claims to
    # use history has something concrete to be checked against.
    floor_probs = np.tile(global_freq, (n_val, 1))
    lp_probs = learner_prefix_probs(val_ev, global_freq, shrinkage_k=args.shrinkage_k)
    table = fit_lag1_gap_table(train_ev, global_freq, shrinkage_k=args.shrinkage_k)
    lg_probs = lag1_gap_probs(val_ev, table, global_freq)

    full = {
        "floor": _score(floor_probs, target), "learner_prefix": _score(lp_probs, target),
        "lag1_gap": _score(lg_probs, target),
    }
    bundle = {
        "floor": _score(floor_probs, target, bundle_mask),
        "learner_prefix": _score(lp_probs, target, bundle_mask),
        "lag1_gap": _score(lg_probs, target, bundle_mask),
    }

    lp_beats_floor = bundle["learner_prefix"]["nll"] < bundle["floor"]["nll"]
    lg_beats_floor = bundle["lag1_gap"]["nll"] < bundle["floor"]["nll"]
    passed = bool(lp_beats_floor and lg_beats_floor)

    metrics = {
        "BUNDLE global_freq_floor_nll": bundle["floor"]["nll"],
        "BUNDLE learner_prefix_nll": bundle["learner_prefix"]["nll"],
        "BUNDLE learner_prefix_brier": bundle["learner_prefix"]["brier"],
        "BUNDLE learner_prefix_beats_floor": lp_beats_floor,
        "BUNDLE lag1_gap_nll": bundle["lag1_gap"]["nll"],
        "BUNDLE lag1_gap_brier": bundle["lag1_gap"]["brier"],
        "BUNDLE lag1_gap_beats_floor": lg_beats_floor,
        "BUNDLE n_eval_population": bundle["learner_prefix"]["n"],
        "full-val global_freq_floor_nll": full["floor"]["nll"],
        "full-val learner_prefix_nll": full["learner_prefix"]["nll"],
        "full-val learner_prefix_brier": full["learner_prefix"]["brier"],
        "full-val lag1_gap_nll": full["lag1_gap"]["nll"],
        "full-val lag1_gap_brier": full["lag1_gap"]["brier"],
        "full-val n_eval_population": full["learner_prefix"]["n"],
    }
    meta = {
        "git_sha": git_sha(), "shrinkage_k": args.shrinkage_k,
        "n_train_events": len(train_ev), "n_val_events_full": n_val,
        "n_val_events_bundle": n_bundle_found,
        "n_train_learners": train_ev["learner"].nunique(),
        "n_val_learners_full": val_ev["learner"].nunique(),
        "global_freq": [round(float(f), 4) for f in global_freq],
    }
    notes = [
        "Not a verdict on a trained support head: these are the comparator numbers. "
        "infra/fa_06_stage2_train.py's val_evaluate scores a trained head on this exact "
        "3,398-row/499-learner bundle, not the full val split (checked here by exact id "
        "match), so the matched comparison is "
        "`val_support_nll < BUNDLE_learner_prefix_nll and val_support_nll < BUNDLE_lag1_gap_nll`. "
        "Full-val numbers (~170K events/500 learners) are lower-variance context using the "
        "identical causal-history machinery, not a substitute for the matched comparison.",
        "Both baselines are causal within one learner's own chronological history and any "
        "global statistic (population frequency, the lag/gap lookup table) is fit on TRAIN "
        "only -- see atom.eval.support_baselines's module docstring.",
        "Multiclass NLL/Brier use the same un-normalised formula as "
        "infra/fa_06_stage2_train.py's val_evaluate (atom.eval.metrics.multiclass_nll/"
        "multiclass_brier), so these numbers are directly comparable, not just similarly "
        "named.",
    ]
    gate_report(
        args.report_path,
        "Support baselines for the FA support head (learner-prefix, lag-1+gap)",
        meta, metrics, passed,
        "Both baselines must beat the naive global-frequency-only floor on the BUNDLE "
        "population's val NLL (a sanity check on the causal-history plumbing, not a "
        "verdict on the support head -- see Notes).",
        notes,
    )
    print(f"[fa-06-support-baselines] BUNDLE(n={bundle['learner_prefix']['n']}): "
         f"floor_nll={bundle['floor']['nll']:.4f} "
         f"learner_prefix_nll={bundle['learner_prefix']['nll']:.4f} "
         f"lag1_gap_nll={bundle['lag1_gap']['nll']:.4f} -- full-val(n={full['learner_prefix']['n']}): "
         f"floor_nll={full['floor']['nll']:.4f} learner_prefix_nll={full['learner_prefix']['nll']:.4f} "
         f"lag1_gap_nll={full['lag1_gap']['nll']:.4f} ({'PASS' if passed else 'FAIL'})")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
