#!/usr/bin/env python
"""Selects the token-loss weight alpha for the FA simulator (Appendix A) by applying the
pre-registered selection rule to the persisted sweep results; no training and no new evaluation.

Inputs: `results/sim_alpha_sweep/alpha_{a}/summary.json` and `raw_scores.npz` for each arm of
`infra/sim_alpha_sweep`. Grid: {0.3, 0.75, 1.0, 1.5, 2.0}, with 0.3 the control arm (alpha=1.0
was added to the pre-registered grid {0.3, 0.75, 1.5, 2.0} during the sweep; the output says
so).

Rule: alpha* is the largest alpha in the grid satisfying (i) val 5-class log-loss within +0.02 of
the alpha=0.3 control arm and (ii) bottom-tercile wrong-detection AUC not CI-clear worse than the
control arm (both AUC definitions); among qualifiers, maximise the personalised-shuffled_u gap.
If no alpha > 0.3 qualifies, alpha* = 0.3 -- the output states which branch applied.

On the two ranking clauses: the primary criterion is the literal "largest alpha satisfying (i)
and (ii)". The script separately reports which qualifier maximises the personalisation gap
(shuffled_u NLL minus personalised NLL; positive means personalisation helps). The two are
expected to coincide if the gap grows monotonically with alpha, since a larger alpha puts more
pressure on the belief to carry generation-relevant information; if they diverge, both are
reported and the largest qualifying alpha governs.
Output: `results/sim_alpha_sweep/selection_decision.json`.
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from wd_1_wrong_detection_panel import _learner_clustered_bootstrap_auc_diff  # noqa: E402

GRID = [0.3, 0.75, 1.0, 1.5, 2.0]
CONTROL_ALPHA = 0.3
LOG_LOSS_TOLERANCE = 0.02
N_BOOT = 1000
SEED = 20260829


def load_arm(alpha: float) -> tuple[dict, dict]:
    out_dir = REPO / "results" / "sim_alpha_sweep" / f"alpha_{alpha}"
    summary = json.loads((out_dir / "summary.json").read_text())
    raw = dict(np.load(out_dir / "raw_scores.npz", allow_pickle=True))
    return summary, raw


def main() -> int:
    arms = {a: load_arm(a) for a in GRID}
    control_summary, control_raw = arms[CONTROL_ALPHA]
    control_log_loss = control_summary["classification_and_wd_panel"]["log_loss_5class_overall"]
    control_bottom = control_raw["row_tercile"] == "bottom"

    print(f"[sim-alpha-selection] control (alpha=0.3) log_loss={control_log_loss:.4f}", flush=True)

    qualifiers = [CONTROL_ALPHA]
    per_arm_report = {}
    for alpha in GRID:
        summary, raw = arms[alpha]
        log_loss = summary["classification_and_wd_panel"]["log_loss_5class_overall"]
        pers_panel = summary["personalisation_panel"]
        personalisation_advantage = (pers_panel["shuffled_u_mixture_nll_mean"]
                                     - pers_panel["personalised_mixture_nll_mean"])
        entry = {"alpha": alpha, "log_loss_5class_overall": log_loss,
                 "log_loss_delta_vs_control": log_loss - control_log_loss,
                 "personalisation_advantage_nll": personalisation_advantage,
                 "personalised_mixture_nll_mean": pers_panel["personalised_mixture_nll_mean"],
                 "shuffled_u_mixture_nll_mean": pers_panel["shuffled_u_mixture_nll_mean"]}

        if alpha == CONTROL_ALPHA:
            entry["gate_i_log_loss_ok"] = True
            entry["gate_ii_wd_auc_ok"] = True
            entry["qualifies"] = True
        else:
            bottom = raw["row_tercile"] == "bottom"
            gate_i = (log_loss - control_log_loss) <= LOG_LOSS_TOLERANCE

            point_uw, lo_uw, hi_uw = _learner_clustered_bootstrap_auc_diff(
                raw["y_uw"][bottom], raw["s_uw"][bottom],
                control_raw["y_uw"][control_bottom], control_raw["s_uw"][control_bottom],
                raw["learner_np"][bottom], n_boot=N_BOOT, seed=SEED)
            point_nc, lo_nc, hi_nc = _learner_clustered_bootstrap_auc_diff(
                raw["y_notcorrect"][bottom], raw["s_notcorrect"][bottom],
                control_raw["y_notcorrect"][control_bottom], control_raw["s_notcorrect"][control_bottom],
                raw["learner_np"][bottom], n_boot=N_BOOT, seed=SEED + 1)
            gate_ii = bool(hi_uw >= 0.0 and hi_nc >= 0.0)

            entry["gate_i_log_loss_ok"] = bool(gate_i)
            entry["gate_ii_wd_auc_ok"] = gate_ii
            entry["wd_auc_diff_vs_control_bottom_tercile"] = {
                "auc_unaided_wrong": {"point": point_uw, "ci95": [lo_uw, hi_uw]},
                "auc_not_unaided_correct": {"point": point_nc, "ci95": [lo_nc, hi_nc]},
            }
            entry["qualifies"] = bool(gate_i and gate_ii)
            if entry["qualifies"]:
                qualifiers.append(alpha)

        per_arm_report[str(alpha)] = entry
        print(f"[sim-alpha-selection] alpha={alpha}: log_loss={log_loss:.4f} "
             f"(delta={entry['log_loss_delta_vs_control']:+.4f}) "
             f"pers_advantage={personalisation_advantage:+.4f} qualifies={entry['qualifies']}", flush=True)

    alpha_star_largest = max(qualifiers)
    qualifier_gaps = {a: per_arm_report[str(a)]["personalisation_advantage_nll"] for a in qualifiers}
    alpha_star_max_gap = max(qualifier_gaps, key=qualifier_gaps.get)
    coincide = alpha_star_largest == alpha_star_max_gap

    alpha_star = alpha_star_largest  # literal primary clause governs
    branch = "no alpha > 0.3 qualified -- alpha*=0.3, so the full training run replicates " \
            "the FA-SIM3 setting" if alpha_star == CONTROL_ALPHA else \
            f"alpha={alpha_star} qualified and is the largest qualifying value -- the full " \
            "training run uses this alpha"

    decision = {
        "grid": GRID, "grid_amendment": "alpha=1.0 added to the grid during the sweep "
                                        "(the pre-registered grid was {0.3, 0.75, 1.5, 2.0})",
        "control_alpha": CONTROL_ALPHA, "log_loss_tolerance": LOG_LOSS_TOLERANCE,
        "qualifiers": qualifiers, "alpha_star_largest_qualifying": alpha_star_largest,
        "alpha_star_max_personalisation_gap": alpha_star_max_gap,
        "two_criteria_coincide": coincide, "alpha_star": alpha_star, "branch": branch,
        "per_arm": per_arm_report,
    }
    print(f"[sim-alpha-selection] QUALIFIERS: {qualifiers}", flush=True)
    print(f"[sim-alpha-selection] alpha* (largest qualifying) = {alpha_star_largest}", flush=True)
    print(f"[sim-alpha-selection] alpha* (max personalisation gap among qualifiers) = {alpha_star_max_gap}",
         flush=True)
    print(f"[sim-alpha-selection] criteria coincide: {coincide}", flush=True)
    print(f"[sim-alpha-selection] DECISION: alpha*={alpha_star} -- {branch}", flush=True)

    out_dir = REPO / "results" / "sim_alpha_sweep"
    (out_dir / "selection_decision.json").write_text(json.dumps(decision, indent=2, default=str))
    print(f"[sim-alpha-selection] wrote {out_dir}/selection_decision.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
