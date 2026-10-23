#!/usr/bin/env python
"""Derive and freeze the NEAR / FAR / REACH-BACK strata used by the filter-body comparison.

The strata depend on the data alone (prefix distances in the frozen production codec over the
`calib_eval_split` eval half of the 500 validation learners), so no model is loaded and they
are fixed before either filter body is scored. Definitions come from `atom.eval.strata`.
Inputs: the v2 manifest and splits from fa_00_manifest_rebuild.py, and
`artifacts/problem_codec_mean_L15.npz`. Outputs: `artifacts/fa_02_strata_v2.npz` (every eval
cell, with a `label_valid` column so the correctness head can be restricted to scored rows
while the support head uses all of them) and `reports/fa_02a_strata_v2.md`. The gate is power
only: REACH-BACK must hold enough labelled cells for the comparison in fa_02_bakeoff.py.
Run: `python scripts/fa_02a_strata_v2.py`. Serves Appendix A (filter body).
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import yaml  # noqa: E402

from atom.data.fa_manifest import load_v2_events  # noqa: E402
from atom.eval.protocol import calib_eval_split  # noqa: E402
from atom.eval.report import gate_report  # noqa: E402
from atom.eval.strata import learner_prefix_distances, tercile_strata  # noqa: E402
from atom.filter.batch import build_padded_batch_v2, check_coverage, load_production_codec  # noqa: E402
from atom.utils.logging import RunLogger  # noqa: E402
from atom.utils.seeds import seed_everything  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[1]

# Tercile boundaries on the earlier (pre-deduplication) substrate, for a drift check only:
# reported, never used to adjust anything.
V1_BOUNDARIES = {"d_rec_lo": 0.9275, "d_rec_hi": 1.0354, "d_min_lo": 0.4214}
V1_REACH_BACK_N = 5256


def main() -> int:
    p36 = yaml.safe_load((REPO / "configs" / "p3_06_far_from_prefix.yaml").read_text())
    s1_cfg = yaml.safe_load((REPO / "configs" / "s1_filter.yaml").read_text())
    irt_cfg = yaml.safe_load((REPO / "configs" / "irt.yaml").read_text())
    v2cfg = yaml.safe_load((REPO / "configs" / "data.yaml").read_text())["fa"]["manifest_v2"]
    seed_everything(p36["seed"])
    logger = RunLogger(config={"task": "fa_02a_strata_v2", "recency_window": p36["recency_window"],
                               "reach_back_power_floor": p36["reach_back_power_floor"],
                               "calib_frac": irt_cfg["calib_frac"]},
                       data_version="FA-v2-2026-08-09", seed=p36["seed"], heartbeat_every_s=10.0)

    logger.heartbeat(step=0, force=True, stage="loading FA-0 v2 manifest + codec")
    ev = load_v2_events(REPO / v2cfg["manifest_out"], REPO / v2cfg["learner_split_out"],
                        REPO / v2cfg["item_split_out"])
    val_ev = ev[ev["split"] == "val"].copy()

    codec_matrix, pid_to_idx = load_production_codec(
        REPO / "artifacts" / "problem_codec_mean_L15.npz", s1_cfg["codec_dim"])
    check_coverage(val_ev["problem_id"].unique(), pid_to_idx)

    val_batch = build_padded_batch_v2(val_ev, codec_matrix, pid_to_idx, "cpu")
    n_val, t_max = val_batch["valid_mask"].shape
    learners = val_batch["learners"]

    # The scored cell set: v2's eval half, the same cells the FA baseline headline uses.
    _, eval_rows = calib_eval_split(val_ev, irt_cfg["calib_frac"], irt_cfg["min_interactions"])
    learner_of = {u: i for i, u in enumerate(learners)}
    rows_n = eval_rows["learner"].map(learner_of).to_numpy()
    rows_t = eval_rows["t"].to_numpy().astype(np.int64)

    logger.heartbeat(step=0, force=True, stage="computing prefix geometry", n_cells=len(rows_n))
    item_codec_np = val_batch["item_codec"].numpy()
    lengths_np = val_batch["lengths"].numpy()
    d_min_full = np.full((n_val, t_max), np.nan)
    d_rec_full = np.full((n_val, t_max), np.nan)
    for n in range(n_val):
        length = int(lengths_np[n])
        d_min_full[n, :length], d_rec_full[n, :length] = learner_prefix_distances(
            item_codec_np[n], length, p36["recency_window"])
        if (n + 1) % 100 == 0:
            logger.heartbeat(step=n + 1, total=n_val, stage="prefix geometry")

    d_min = d_min_full[rows_n, rows_t]
    d_rec = d_rec_full[rows_n, rows_t]
    if np.isnan(d_min).any() or np.isnan(d_rec).any():
        raise RuntimeError("NaN prefix distance on a scored cell -- the eval half is expected "
                            "to start well after t=0 for every learner")

    strata = tercile_strata(d_min, d_rec)
    n_cells = len(d_min)
    label_valid_np = val_batch["label_valid"].numpy()[rows_n, rows_t]
    counts = {name: int(strata[f"{name.lower().replace('-', '_')}_mask"].sum())
              for name in ("NEAR", "FAR", "REACH-BACK")}
    labelled_counts = {
        name: int((strata[f"{name.lower().replace('-', '_')}_mask"] & label_valid_np).sum())
        for name in ("NEAR", "FAR", "REACH-BACK")}

    n_reach_back = labelled_counts["REACH-BACK"]
    passed = n_reach_back >= p36["reach_back_power_floor"]

    out_path = REPO / "artifacts" / "fa_02_strata_v2.npz"
    np.savez(
        out_path,
        learner=np.asarray(learners)[rows_n], learner_idx=rows_n.astype(np.int64), t=rows_t,
        d_min=d_min, d_rec=d_rec, label_valid=label_valid_np,
        near_mask=strata["near_mask"], far_mask=strata["far_mask"],
        reach_back_mask=strata["reach_back_mask"],
        d_rec_tercile_lo=strata["d_rec_tercile_lo"], d_rec_tercile_hi=strata["d_rec_tercile_hi"],
        d_min_tercile_lo=strata["d_min_tercile_lo"],
        recency_window=p36["recency_window"], calib_frac=irt_cfg["calib_frac"],
        min_interactions=irt_cfg["min_interactions"], n_cells=n_cells,
        data_version="FA-v2-2026-08-09",
    )

    logger.log_metrics(passed=passed, n_cells=n_cells, n_reach_back_labelled=n_reach_back,
                        **{f"n_{k}": v for k, v in counts.items()},
                        **{k: strata[k] for k in
                           ("d_rec_tercile_lo", "d_rec_tercile_hi", "d_min_tercile_lo")})
    logger.summary(status="ok" if passed else "underpowered", n_cells=n_cells,
                    n_reach_back=n_reach_back)

    def _drift(new: float, old: float) -> str:
        return f"{new:.4f}  (v1 {old:.4f}, Δ{new - old:+.4f})"

    gate_report(
        path="reports/fa_02a_strata_v2.md",
        title="NEAR/FAR/REACH-BACK strata on the deduplicated v2 substrate "
              "(fixed before any filter body is scored)",
        meta={"run_id": logger.run_id, "git_sha": logger.git_sha,
              "config_hash": logger.config_hash, "n_val_learners": n_val,
              "recency_window": p36["recency_window"], "calib_frac": irt_cfg["calib_frac"],
              "artifact": "artifacts/fa_02_strata_v2.npz"},
        metrics={
            "scored cells (v2 eval half, all)": n_cells,
            "scored cells with a valid platform score (correctness court)":
                f"{int(label_valid_np.sum())} "
                f"({100 * label_valid_np.mean():.2f}% -- the rest are score-missing, masked "
                f"out of correctness but KEPT for the support head)",
            "d_rec tercile boundaries (lo / hi)":
                f"{_drift(strata['d_rec_tercile_lo'], V1_BOUNDARIES['d_rec_lo'])}  /  "
                f"{_drift(strata['d_rec_tercile_hi'], V1_BOUNDARIES['d_rec_hi'])}",
            "d_min tercile boundary (lo)":
                _drift(strata["d_min_tercile_lo"], V1_BOUNDARIES["d_min_lo"]),
            "[NEAR] n cells (all / labelled)": f"{counts['NEAR']} / {labelled_counts['NEAR']}",
            "[FAR] n cells (all / labelled)": f"{counts['FAR']} / {labelled_counts['FAR']}",
            "[REACH-BACK] n cells (all / labelled)":
                f"{counts['REACH-BACK']} / {labelled_counts['REACH-BACK']}  "
                f"(v1/P3-6: {V1_REACH_BACK_N})",
            "REACH-BACK power (labelled n vs floor)":
                f"{n_reach_back} vs {p36['reach_back_power_floor']} -- "
                f"{'ADEQUATE' if passed else 'UNDERPOWERED'}",
        },
        passed=passed,
        gate_desc="Not a verdict on any model: no model is loaded by this script. The gated "
                  "claim is power, stated before the comparison: REACH-BACK must hold at least "
                  f"{p36['reach_back_power_floor']} scored, labelled cells for the filter-body "
                  "comparison on that cell to be informative. Fails if it does not.",
        notes=[
            "Model-independent by construction: this script loads no checkpoint and builds no "
            "network. The strata are prefix geometry in the frozen production codec plus the "
            "calib_eval_split cell set, so they are properties of the data and are fixed "
            "before any filter body is scored.",
            "Definitions are imported from atom.eval.strata rather than copied or re-derived, "
            "so the same functions define the strata on every substrate.",
            "The strata are recomputed on the v2 manifest because deduplication removed "
            "31,441 rows and the splits were re-cut, so (learner, t) cells from the earlier "
            "substrate do not exist here.",
            "The boundary drift shown above is a sanity check on the substrate change, "
            "reported and then left alone: boundaries are whatever the deduplicated data "
            "gives, never adjusted towards the earlier values.",
            "Distance: Euclidean on L2-normalised codec vectors (monotonic with cosine). "
            "Audit-only: it never enters a prediction or a loss, and the transformer body's "
            "attention bias computes its own distances independently from the same geometry.",
            "Cells are fixed for both heads at once. The geometry is label-independent, so "
            "the artifact carries every eval cell plus a `label_valid` column; the "
            "correctness comparison intersects with it, the support comparison does not need "
            "to (support is never missing).",
        ],
    )
    print(f"\nstrata: {'ADEQUATE POWER' if passed else 'UNDERPOWERED'} · n_cells={n_cells} · "
          f"NEAR {counts['NEAR']} · FAR {counts['FAR']} · REACH-BACK {counts['REACH-BACK']} "
          f"({n_reach_back} labelled) · artifact: artifacts/fa_02_strata_v2.npz")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
