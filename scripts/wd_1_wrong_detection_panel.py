#!/usr/bin/env python
"""Wrong-detection panel for the FA simulator variants (Appendix A): evaluation only, on existing
checkpoints (FA-SIM3, the width-parameterisation variant `a3`, and the `br4_r1/r2/r3` variants),
with no further training.

Two definitions this panel applies:
  (1) The committee-variance check. An earlier version of the G1/G2 bands demanded the committee
      carry the full per-item population variance ([0.10,0.21] / >=0.135). A learner-level
      latent can carry only the trait-explainable share of that variance (the rest is
      state/attempt-level aleatoric noise no persistent per-learner z could predict), so G1
      (cold start) and G2 (late floor) are graded against TRAIT_CEILING_MID = cluster-explainable
      fraction (0.1246) x item population variance (0.1364) = 0.016992..., CI [0.016, 0.018],
      both from the FA exploratory variance-decomposition bootstrap.
  (2) The primary metric is wrong-detection: how well the model detects that a student is wrong.
      A change that hurts wrong-detection is not adopted.

Four parts, in order: (1) the G1/G2 grade, (2) the wrong-detection panel (primary), (3) bias
decomposition plus three "artifact vs miscentring" separating tests (width-zeroed bias,
bias-vs-t curve, mean-vs-median), (4) the pre-registered adoption decision between `br4_r3`
and FA-SIM3. Output: `results/wd_1/summary.json`.
Run: `python scripts/wd_1_wrong_detection_panel.py`.
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import pandas as pd
import torch

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "infra"))

from atom.binding import reparameterize_from_log_sigma  # noqa: E402
from atom.data.fa_manifest import load_v2_events  # noqa: E402
from atom.data.fa_outcome_label import N_OUTCOME, OUTCOME_CLASSES, outcome_label  # noqa: E402
from atom.filter.batch import build_padded_batch_v2, load_production_codec  # noqa: E402
from atom.filter.episode_v2 import build_prev_feat_v2  # noqa: E402
from atom.filter.item_encoder_mlp import S1FilterV2ItemMLP  # noqa: E402
from atom.heads.fa_hierarchical_heads import OutcomeHead  # noqa: E402
from atom.simulator.sampler_semantics import CURRENT_SAMPLER_SEMANTICS  # noqa: E402
from fa_filter_common import B4_CONFIG, CODEC_PATH, chunked_filter_forward  # noqa: E402

K = 32
EVAL_CHUNK = 4000
UNAIDED_CORRECT, UNAIDED_WRONG = 0, 1
TIMES_EXACT = (0, 10, 25, 50, 100)  # part (3c) bias-vs-t curve, exact positions
LATE_T_LO = 160  # matches T_BUCKETS' own "(160, 10000]" bucket, as in the original G1/G2

# Trait ceiling: cluster-explainable fraction x item population variance, both from the
# FA exploratory variance-decomposition bootstrap.
CLUSTER_EXPLAINABLE_FRACTION = 0.1246
ITEM_POPULATION_VARIANCE = 0.1364
TRAIT_CEILING_MID = CLUSTER_EXPLAINABLE_FRACTION * ITEM_POPULATION_VARIANCE
TRAIT_CEILING_CI = (0.016, 0.018)  # CI of the product, from the same bootstrap

CHECKPOINTS = {
    "fa_sim3": (REPO / "artifacts/checkpoints/fa_sim3_final.pt", "direct"),
    "a3": (REPO / "results/fa_sigma_a3/final.pt", "precision_accumulating"),
    "br4_r1": (REPO / "results/fa_br4_full/final.pt", "direct"),
    "br4_r2": (REPO / "results/fa_br4_r2_full/final.pt", "direct"),
    "br4_r3": (REPO / "results/fa_br4_r3_full/final.pt", "direct"),
}


def load_checkpoint(path: pathlib.Path, sigma_param: str, device: str):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    assert ck["sampler_semantics"] == CURRENT_SAMPLER_SEMANTICS, (
        f"{path.name}: sampler_semantics={ck.get('sampler_semantics')!r} -- refusing a "
        "checkpoint trained with a different sampler")
    filt = S1FilterV2ItemMLP(**B4_CONFIG, sigma_parameterization=sigma_param).to(device)
    filt.load_state_dict(ck["filter"], strict=True)
    hidden = int(ck["outcome_head"]["trunk.weight"].shape[0])
    head = OutcomeHead(codec_dim=B4_CONFIG["codec_dim"], latent_dim=B4_CONFIG["latent_dim"],
                       hidden=hidden).to(device)
    head.load_state_dict(ck["outcome_head"], strict=True)
    filt.eval()
    head.eval()
    return filt, head, ck.get("git_sha")


def _ece_per_class(probs: np.ndarray, target: np.ndarray, n_bins: int = 10) -> dict[int, float]:
    out = {}
    for c in range(N_OUTCOME):
        p, y = probs[:, c], (target == c).astype(float)
        total = 0.0
        for lo in range(n_bins):
            m = (p >= lo / n_bins) & (p < (lo + 1) / n_bins if lo < n_bins - 1 else p <= 1.0)
            if m.any():
                total += m.sum() * abs(y[m].mean() - p[m].mean())
        out[c] = float(total / max(len(p), 1))
    return out


def _reliability_bins(p_wrong: np.ndarray, is_wrong: np.ndarray, n_bins: int = 10) -> list[dict]:
    bins = []
    for lo in range(n_bins):
        m = (p_wrong >= lo / n_bins) & (p_wrong < (lo + 1) / n_bins if lo < n_bins - 1 else p_wrong <= 1.0)
        if not m.any():
            continue
        bins.append({"bin_lo": lo / n_bins, "n": int(m.sum()),
                     "mean_predicted": float(p_wrong[m].mean()),
                     "empirical_wrong_rate": float(is_wrong[m].mean())})
    return bins


def _auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y_true, y_score))


def _learner_clustered_bootstrap_auc_diff(y_a: np.ndarray, s_a: np.ndarray, y_b: np.ndarray,
                                          s_b: np.ndarray, learners: np.ndarray, n_boot: int,
                                          seed: int) -> tuple[float, float, float]:
    """Paired (same rows, two scorers) learner-clustered bootstrap of AUC(a) - AUC(b)."""
    uniq = np.unique(learners)
    rows_by_learner = {lid: np.flatnonzero(learners == lid) for lid in uniq}
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        drawn = uniq[rng.integers(0, len(uniq), size=len(uniq))]
        idx = np.concatenate([rows_by_learner[lid] for lid in drawn])
        da = _auc(y_a[idx], s_a[idx])
        db = _auc(y_b[idx], s_b[idx])
        if np.isfinite(da) and np.isfinite(db):
            diffs.append(da - db)
    diffs = np.asarray(diffs)
    point = _auc(y_a, s_a) - _auc(y_b, s_b)
    return point, float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def main() -> int:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[wd-1] device={device}", flush=True)
    ev = load_v2_events(REPO / "artifacts/fa_manifest_v2.parquet",
                        REPO / "splits/fa_learner_split_v2.csv",
                        REPO / "splits/fa_item_split_v2.csv")
    ev_val = ev[ev["split"] == "val"].reset_index(drop=True)
    codec_matrix, pid_to_idx = load_production_codec(CODEC_PATH, B4_CONFIG["codec_dim"])

    def build_fb(ev_split):
        fb = build_padded_batch_v2(ev_split, codec_matrix, pid_to_idx, device="cpu", include_time_min=True)
        fb["prev_feat"] = build_prev_feat_v2(fb["correct"], fb["hint_bin"], fb["saw_answer"], fb["score_missing"])
        for k in ("item_codec", "prev_feat", "time_min", "valid_mask"):
            fb[k] = fb[k].to(device)
        return fb

    val_fb = build_fb(ev_val)
    labels = outcome_label(ev_val["support_code"], ev_val["score"], ev_val["score_missing"])
    valid = labels["outcome_valid"].to_numpy()
    rows = ev_val.iloc[np.flatnonzero(valid)].reset_index(drop=True)
    learner_to_idx = {u: i for i, u in enumerate(val_fb["learners"])}
    l_idx = torch.tensor([learner_to_idx[u] for u in rows["learner"]], dtype=torch.long)
    t_idx = torch.tensor(rows["t"].astype("int64").to_numpy(), dtype=torch.long)
    outcome_target = labels["outcome_code"].to_numpy()[valid]
    item_codec_row_cpu = val_fb["item_codec"].cpu()[l_idx, t_idx]
    t_np = t_idx.numpy()
    learner_np = rows["learner"].astype(str).to_numpy()

    # Ability tercile: CHECKPOINT-INDEPENDENT (true labels only), computed once, shared.
    ability = (pd.Series(outcome_target == UNAIDED_CORRECT, index=learner_np)
              .groupby(level=0).mean())
    tercile_map = pd.qcut(ability.rank(method="first"), 3, labels=["bottom", "middle", "top"])
    row_tercile = pd.Series(learner_np).map(tercile_map).to_numpy()
    print(f"[wd-1] {len(rows)} val rows, {len(ability)} learners, "
         f"tercile counts: {pd.Series(row_tercile).value_counts().to_dict()}", flush=True)

    results: dict[str, dict] = {}
    for name, (path, sigma_param) in CHECKPOINTS.items():
        print(f"[wd-1] === {name} ===", flush=True)
        filt, head, git_sha = load_checkpoint(path, sigma_param, device)
        filt_forward = lambda ic, pf, tm, vm: filt(ic, pf)  # noqa: E731
        with torch.no_grad():
            mu_all, log_sigma_all = chunked_filter_forward(filt_forward, val_fb)
        mu_sel = mu_all[l_idx.to(device), t_idx.to(device)]
        log_sigma_sel = log_sigma_all[l_idx.to(device), t_idx.to(device)]
        item_codec_row = item_codec_row_cpu.to(device)

        n_rows = mu_sel.shape[0]
        gen = torch.Generator(device=device).manual_seed(20260826)
        probs_mean_parts, probs_median_parts, probs_mu_parts, p0_var_parts = [], [], [], []
        with torch.no_grad():
            for lo in range(0, n_rows, EVAL_CHUNK):
                hi = min(lo + EVAL_CHUNK, n_rows)
                mu_c, ls_c, ic_c = mu_sel[lo:hi], log_sigma_sel[lo:hi], item_codec_row[lo:hi]
                u_k = reparameterize_from_log_sigma(mu_c, ls_c, k=K, generator=gen)  # [n,K,L]
                n_c = hi - lo
                logits_k = torch.stack([head(ic_c, u_k[:, j, :]) for j in range(K)], dim=1)  # [n,K,C]
                probs_k = torch.softmax(logits_k, dim=-1)
                probs_mean_parts.append(probs_k.mean(dim=1).cpu())
                probs_median_parts.append(probs_k.median(dim=1).values.cpu())
                p0_var_parts.append(probs_k[..., UNAIDED_CORRECT].var(dim=1, unbiased=True).cpu())
                logits_mu = head(ic_c, mu_c)
                probs_mu_parts.append(torch.softmax(logits_mu, dim=-1).cpu())
        probs_mean = torch.cat(probs_mean_parts, dim=0).numpy()
        probs_median = torch.cat(probs_median_parts, dim=0).numpy()
        probs_mu = torch.cat(probs_mu_parts, dim=0).numpy()
        p0_var = torch.cat(p0_var_parts, dim=0).numpy()

        # --- (1) corrected G1/G2 re-grade ---
        g1_mask = t_np == 0
        g2_mask = t_np > LATE_T_LO
        g1_val = float(p0_var[g1_mask].mean()) if g1_mask.any() else float("nan")
        g2_val = float(p0_var[g2_mask].mean()) if g2_mask.any() else float("nan")

        # --- (2) wrong-detection panel ---
        y_uw = (outcome_target == UNAIDED_WRONG).astype(int)
        s_uw = probs_mean[:, UNAIDED_WRONG]
        y_notcorrect = (outcome_target != UNAIDED_CORRECT).astype(int)
        s_notcorrect = 1.0 - probs_mean[:, UNAIDED_CORRECT]
        wrong_detection: dict[str, dict] = {"overall": {
            "auc_unaided_wrong": _auc(y_uw, s_uw), "auc_not_unaided_correct": _auc(y_notcorrect, s_notcorrect),
            "n": int(len(y_uw)),
        }}
        for terc in ("bottom", "middle", "top"):
            m = row_tercile == terc
            wrong_detection[terc] = {
                "auc_unaided_wrong": _auc(y_uw[m], s_uw[m]),
                "auc_not_unaided_correct": _auc(y_notcorrect[m], s_notcorrect[m]),
                "n": int(m.sum()),
                "ece_by_class": _ece_per_class(probs_mean[m], outcome_target[m]),
                "wrong_reliability_bins": _reliability_bins(s_notcorrect[m], y_notcorrect[m]),
            }
        ece_overall = _ece_per_class(probs_mean, outcome_target)

        # --- (3) bias decomposition, bottom tercile ---
        bottom = row_tercile == "bottom"
        pred_alloc = probs_mean[bottom].mean(axis=0)
        emp_alloc = np.array([(outcome_target[bottom] == c).mean() for c in range(N_OUTCOME)])
        bias_decomposition = {OUTCOME_CLASSES[c]: {
            "predicted_mean_prob": float(pred_alloc[c]), "empirical_frequency": float(emp_alloc[c]),
            "predicted_minus_empirical": float(pred_alloc[c] - emp_alloc[c]),
        } for c in range(N_OUTCOME)}

        # (3b) width-zeroed bias, natural committee bias, by tercile
        y0 = (outcome_target == UNAIDED_CORRECT).astype(float)
        signed_bias_by_tercile, width_zeroed_bias_by_tercile = {}, {}
        for terc in ("bottom", "middle", "top"):
            m = row_tercile == terc
            signed_bias_by_tercile[terc] = float((probs_mean[m, UNAIDED_CORRECT] - y0[m]).mean())
            width_zeroed_bias_by_tercile[terc] = float((probs_mu[m, UNAIDED_CORRECT] - y0[m]).mean())

        # (3c) bias vs t curve, bottom tercile, exact t, natural committee
        bias_vs_t = {}
        for t_exact in TIMES_EXACT:
            m = bottom & (t_np == t_exact)
            bias_vs_t[str(t_exact)] = {
                "n": int(m.sum()),
                "signed_bias": float((probs_mean[m, UNAIDED_CORRECT] - y0[m]).mean()) if m.any() else None,
            }

        # mean-vs-median bias, bottom tercile
        mean_vs_median_bias = {
            "mean_committee_bias": float((probs_mean[bottom, UNAIDED_CORRECT] - y0[bottom]).mean()),
            "median_committee_bias": float((probs_median[bottom, UNAIDED_CORRECT] - y0[bottom]).mean()),
        }

        results[name] = {
            "checkpoint_path": str(path), "git_sha": git_sha, "sigma_parameterization": sigma_param,
            "g1_corrected": g1_val, "g2_corrected": g2_val,
            "trait_ceiling_mid": TRAIT_CEILING_MID, "trait_ceiling_ci": list(TRAIT_CEILING_CI),
            "wrong_detection": wrong_detection, "ece_overall_by_class": ece_overall,
            "bias_decomposition_bottom_tercile": bias_decomposition,
            "signed_bias_by_tercile_natural": signed_bias_by_tercile,
            "signed_bias_by_tercile_width_zeroed": width_zeroed_bias_by_tercile,
            "bias_vs_t_bottom_tercile": bias_vs_t,
            "mean_vs_median_bias_bottom_tercile": mean_vs_median_bias,
        }
        print(f"[wd-1] {name}: G1={g1_val:.5f} G2={g2_val:.5f} "
             f"AUC(UW,bottom)={wrong_detection['bottom']['auc_unaided_wrong']:.4f} "
             f"width_zeroed_bias(bottom)={width_zeroed_bias_by_tercile['bottom']:.4f}", flush=True)

        if name in ("fa_sim3", "br4_r3"):
            results.setdefault("_raw_for_ship_decision", {})[name] = {
                "y_uw": y_uw, "s_uw": s_uw, "y_notcorrect": y_notcorrect,
                "s_notcorrect": s_notcorrect, "row_tercile": row_tercile, "learner_np": learner_np,
            }
        del filt, head
        if device == "cuda":
            torch.cuda.empty_cache()

    # --- (4) ship recommendation, pre-registered rule ---
    raw = results.pop("_raw_for_ship_decision")
    bottom_mask_fa = raw["fa_sim3"]["row_tercile"] == "bottom"
    bottom_mask_r3 = raw["br4_r3"]["row_tercile"] == "bottom"
    # Rows are the SAME val cohort for every checkpoint (identical l_idx/t_idx selection),
    # so the bottom-tercile row masks are identical across checkpoints by construction --
    # asserted, not assumed.
    assert np.array_equal(bottom_mask_fa, bottom_mask_r3), "tercile assignment drifted between checkpoints"
    bm = bottom_mask_fa
    point_uw, lo_uw, hi_uw = _learner_clustered_bootstrap_auc_diff(
        raw["br4_r3"]["y_uw"][bm], raw["br4_r3"]["s_uw"][bm],
        raw["fa_sim3"]["y_uw"][bm], raw["fa_sim3"]["s_uw"][bm],
        raw["fa_sim3"]["learner_np"][bm], n_boot=1000, seed=20260826)
    point_nc, lo_nc, hi_nc = _learner_clustered_bootstrap_auc_diff(
        raw["br4_r3"]["y_notcorrect"][bm], raw["br4_r3"]["s_notcorrect"][bm],
        raw["fa_sim3"]["y_notcorrect"][bm], raw["fa_sim3"]["s_notcorrect"][bm],
        raw["fa_sim3"]["learner_np"][bm], n_boot=1000, seed=20260827)
    # Pre-registered rule: ship R3 iff its bottom-tercile wrong-detection is >= FA-SIM3's
    # WITHIN CI -- i.e. the paired difference's CI does not show R3 significantly WORSE
    # (upper bound of "R3 minus FA-SIM3" must not sit entirely below zero).
    ship_r3 = bool(hi_uw >= 0.0 and hi_nc >= 0.0)
    ship_decision = {
        "rule": "ship R3 iff bottom-tercile wrong-detection AUC (R3 - FA-SIM3, learner-clustered "
                "bootstrap, n_boot=1000) CI does not show R3 significantly worse (CI upper bound < 0)",
        "auc_unaided_wrong_diff_r3_minus_fasim3": {"point": point_uw, "ci95": [lo_uw, hi_uw]},
        "auc_not_unaided_correct_diff_r3_minus_fasim3": {"point": point_nc, "ci95": [lo_nc, hi_nc]},
        "ship_r3": ship_r3,
        "verdict": ("R3 is adopted as the FA belief -- wrong-detection is not significantly worse "
                   "than FA-SIM3's" if ship_r3 else
                   "FA-SIM3 remains the simulator; R3 is reported as the "
                   "uncertainty-mechanism result, not adopted"),
    }
    print(f"[wd-1] SHIP DECISION: {ship_decision['verdict']}", flush=True)

    summary = {
        "task": "WD-1", "k": K, "trait_ceiling_mid": TRAIT_CEILING_MID,
        "trait_ceiling_ci": list(TRAIT_CEILING_CI), "checkpoints": results,
        "ece_overall_note": "ece_overall_by_class is over ALL terciles combined; "
                            "wrong_detection[tercile].ece_by_class is the stratified version.",
        "ship_decision": ship_decision,
    }
    out_dir = REPO / "results/wd_1"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"[wd-1] wrote {out_dir}/summary.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
