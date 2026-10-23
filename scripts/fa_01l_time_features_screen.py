#!/usr/bin/env python
"""Screen time, session and repeat-exposure features on the 4-class support head and on correctness.

`build_prev_feat_v2` carries five non-temporal channels. Give-up is session-autocorrelated, so
time features are expected to matter more for support than for correctness. Two arms, same
GRU body, same seed, minibatch order and (wherever shapes match) head initialisation:
    baseline   S1FilterV2(prev_feat_dim=5) + FiLMHeadsSupport (the GRU arm of fa_02_bakeoff.py)
    time       S1FilterV2(prev_feat_dim=8) + FiLMHeadsSupportTime
The recurrent input gains three channels shifted to describe event t-1 (gap since the previous
event, session-boundary flag, position in session), because the next engagement moment is not
known at belief time; the emission heads get repeat-exposure features for the current item,
which they may read like `item_codec[t]` (see `atom.filter.fa_time_features`).

Checks: a peek canary (identical history, two different current items) must give
max|Δμ| = 0 exactly, also covered by `tests/test_fa_time_features.py`; the baseline arm must
reproduce the fa_02_bakeoff.py GRU-arm numbers within 0.005 (skipped for pilot runs); both arms
must train. Outcome, from the paired support-NLL CI (time minus baseline): entirely below zero
= positive, entirely above = negative, straddling zero = ambiguous.

Inputs: the v2 FA manifest and splits, `artifacts/problem_codec_mean_L15.npz`,
`configs/{fa_02_bakeoff,s1_filter,irt,data}.yaml`. Outputs: `artifacts/checkpoints/fa_01l_{baseline,time}.pt`,
`reports/fa_01l_predictions.csv.gz`, `reports/fa_01l_time_features_screen{_raw.json,.md}`.
Run: `python scripts/fa_01l_time_features_screen.py [--epochs N] [--pilot-n-learners N]`.
Serves Appendix A (time features).
"""

from __future__ import annotations

import argparse
import math
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import yaml  # noqa: E402
from torch import nn  # noqa: E402

from atom.data.fa_manifest import load_v2_events  # noqa: E402
from atom.eval.metrics import accuracy, auc, brier, ece  # noqa: E402
from atom.eval.protocol import calib_eval_split  # noqa: E402
from atom.eval.report import gate_report  # noqa: E402
from atom.filter.batch import build_padded_batch_v2, check_coverage, load_production_codec  # noqa: E402
from atom.filter.episode_v2 import build_prev_feat_v2  # noqa: E402
from atom.filter.fa_time_features import (  # noqa: E402
    DEFAULT_SESSION_GAP_MIN,
    PREV_FEAT_DIM_V2_TIME,
    build_current_repeat_features,
    build_prev_feat_v2_time,
    compute_session_and_exposure,
    scatter_field,
)
from atom.filter.gru import sample_u  # noqa: E402
from atom.filter.losses import gaussian_kl_free_bits, kl_anneal_weight  # noqa: E402
from atom.filter.s1v2 import S1FilterV2  # noqa: E402
from atom.heads.fa2_heads import N_SUPPORT, FiLMHeadsSupport  # noqa: E402
from atom.heads.fa_time_heads import FiLMHeadsSupportTime  # noqa: E402
from atom.utils.logging import RunLogger, default_device  # noqa: E402
from atom.utils.seeds import seed_everything  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[1]
ARMS = ("baseline", "time")
SESSION_GAP_LOG1P_THRESHOLD = math.log1p(DEFAULT_SESSION_GAP_MIN)

# GRU-arm numbers of fa_02_bakeoff.py -- the baseline arm here uses the identical
# seed/config/data/order, so it should reproduce these closely.
FA2_GRU_REF = {"correct_auc": 0.786, "correct_nll": 0.5293, "support_nll": 0.4588}
FA2_GRU_REPRO_TOL = 0.005


def run_peek_canary(codec_dim: int) -> float:
    """Inline peek check: identical history, two different CURRENT
    questions, 8-channel time-augmented recurrent input -- b_t must be bit-identical at
    every position. Returns max|Δμ|; the caller asserts it is exactly 0.0."""
    torch.manual_seed(1)
    n, tlen = 2, 6
    history = torch.randn(n, tlen, codec_dim)
    correct = torch.randint(0, 2, (n, tlen)).float()
    hint = torch.zeros(n, tlen)
    saw = torch.zeros(n, tlen)
    score_missing = torch.zeros(n, tlen)
    gap = torch.rand(n, tlen) * 3.0
    new_session = torch.randint(0, 2, (n, tlen)).float()
    pos_in_session = torch.arange(tlen).float().unsqueeze(0).expand(n, tlen).clone()
    prev_feat = build_prev_feat_v2_time(correct, hint, saw, score_missing, gap, new_session,
                                        pos_in_session)
    item_q1, item_q2 = history.clone(), history.clone()
    item_q2[:, -1] = torch.randn(n, codec_dim)

    torch.manual_seed(0)
    model = S1FilterV2(codec_dim=codec_dim, episode_hidden=10, filter_hidden=12, latent_dim=4,
                       query_independent=True, prev_feat_dim=PREV_FEAT_DIM_V2_TIME)
    model.eval()
    with torch.no_grad():
        mu1, _ = model(item_q1, prev_feat)
        mu2, _ = model(item_q2, prev_feat)
    return (mu1 - mu2).abs().max().item()


# -- models --------------------------------------------------------------------------------

def build_arm(arm: str, s1: dict, device: str):
    torch.manual_seed(s1["seed"])
    prev_feat_dim = PREV_FEAT_DIM_V2_TIME if arm == "time" else None
    model = S1FilterV2(s1["codec_dim"], s1["episode_hidden"], s1["filter_hidden"],
                       s1["latent_dim"], prev_feat_dim=prev_feat_dim).to(device)
    torch.manual_seed(s1["seed"])  # heads identical across arms wherever shapes match
    if arm == "time":
        heads = FiLMHeadsSupportTime(s1["codec_dim"], s1["latent_dim"], s1["filter_hidden"]).to(device)
    else:
        heads = FiLMHeadsSupport(s1["codec_dim"], s1["latent_dim"], s1["filter_hidden"]).to(device)
    return model, heads


def slice_batch(batch: dict, idx: torch.Tensor, prev_feat: torch.Tensor, support: torch.Tensor,
                repeat_feat: torch.Tensor) -> dict:
    t_max = int(batch["lengths"][idx].max().item())
    s = lambda x: x[idx, :t_max]  # noqa: E731
    return {"item_codec": s(batch["item_codec"]), "correct": s(batch["correct"]),
            "label_valid": s(batch["label_valid"]), "valid_mask": s(batch["valid_mask"]),
            "prev_feat": s(prev_feat), "support": s(support), "repeat_feat": s(repeat_feat)}


def losses_for(arm: str, mb: dict, mu, log_sigma, heads, s1, cfg, bce, k: int):
    u = sample_u(mu, log_sigma, k=k)
    if arm == "time":
        correctness_logits, support_logits = heads(mb["item_codec"], u, mb["repeat_feat"])
    else:
        correctness_logits, support_logits = heads(mb["item_codec"], u)

    label_mask_k = mb["label_valid"].unsqueeze(0).expand(k, -1, -1)
    correct_k = mb["correct"].unsqueeze(0).expand(k, -1, -1)
    per_c = bce(correctness_logits, correct_k)
    correctness_loss = (per_c * label_mask_k).sum() / label_mask_k.sum().clamp(min=1.0)

    support_k = mb["support"].unsqueeze(0).expand(k, -1, -1)
    per_cell = F.cross_entropy(support_logits.reshape(-1, N_SUPPORT), support_k.reshape(-1),
                               reduction="none").view(support_k.shape)
    vm_k = mb["valid_mask"].unsqueeze(0).expand(k, -1, -1).float()
    support_loss = (per_cell * vm_k).sum() / vm_k.sum().clamp(min=1.0)

    kl = gaussian_kl_free_bits(mu, log_sigma, s1["free_bits"])
    vm = mb["valid_mask"].float()
    kl_loss = (kl * vm).sum() / vm.sum().clamp(min=1.0)
    return correctness_loss, support_loss, kl_loss


def train_arm(arm, cfg, s1, device, logger, batch, prev_feat, support, repeat_feat, n_learners):
    model, heads = build_arm(arm, s1, device)
    params = list(model.parameters()) + list(heads.parameters())
    opt = torch.optim.Adam(params, lr=cfg["lr"])
    bce = nn.BCEWithLogitsLoss(reduction="none")

    steps_per_epoch = int(np.ceil(n_learners / cfg["batch_learners"]))
    total_steps = steps_per_epoch * cfg["epochs"]
    anneal_steps = max(1, int(0.075 * total_steps))  # matches fa_02_bakeoff's KL_ANNEAL_FRAC
    order_rng = np.random.default_rng(cfg["seed"])  # SAME minibatch order for both arms

    t0 = time.time()
    step = 0
    first_loss = last_loss = None
    for epoch in range(cfg["epochs"]):
        model.train()
        heads.train()
        perm = torch.as_tensor(order_rng.permutation(n_learners), device=device)
        for b0 in range(0, n_learners, cfg["batch_learners"]):
            idx = perm[b0:b0 + cfg["batch_learners"]]
            mb = slice_batch(batch, idx, prev_feat, support, repeat_feat)
            mu, log_sigma = model(mb["item_codec"], mb["prev_feat"])
            c_loss, s_loss, kl_loss = losses_for(arm, mb, mu, log_sigma, heads, s1, cfg, bce,
                                                 s1["k_train"])
            loss = (c_loss + cfg["support_weight"] * s_loss
                    + kl_anneal_weight(step, s1["kl_weight_max"], anneal_steps) * kl_loss)
            logger.tripwire_finite(f"{arm}_loss", loss)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, cfg["grad_clip"])
            opt.step()

            last_loss = c_loss.item()
            first_loss = last_loss if first_loss is None else first_loss
            step += 1
            logger.heartbeat(step=step, total=total_steps, stage=arm, epoch=epoch + 1,
                             correctness=round(last_loss, 4), support=round(s_loss.item(), 4))

    wall_s = time.time() - t0
    return model, heads, {"wall_s": wall_s, "first_loss": first_loss, "last_loss": last_loss}


@torch.no_grad()
def predict_val(model, heads, arm, batch, prev_feat, repeat_feat, device, k_eval, latent_dim,
                chunk: int, seed: int):
    g = torch.Generator(device="cpu").manual_seed(seed)
    noise = torch.randn(k_eval, latent_dim, generator=g).to(device)
    n = batch["valid_mask"].shape[0]
    p_correct, p_support = [], []
    model.eval()
    heads.eval()
    for i in range(0, n, chunk):
        idx = torch.arange(i, min(i + chunk, n), device=device)
        mb = slice_batch(batch, idx, prev_feat, batch["support"], repeat_feat)
        mu, log_sigma = model(mb["item_codec"], mb["prev_feat"])
        u = sample_u(mu, log_sigma, k=k_eval, noise=noise)
        if arm == "time":
            c_logits, s_logits = heads(mb["item_codec"], u, mb["repeat_feat"])
        else:
            c_logits, s_logits = heads(mb["item_codec"], u)
        pc = torch.sigmoid(c_logits).mean(dim=0)
        ps = torch.softmax(s_logits, dim=-1).mean(dim=0)
        pad = batch["valid_mask"].shape[1] - pc.shape[1]
        if pad:
            pc = F.pad(pc, (0, pad))
            ps = F.pad(ps, (0, 0, 0, pad))
        p_correct.append(pc.cpu())
        p_support.append(ps.cpu())
    return torch.cat(p_correct).numpy(), torch.cat(p_support).numpy()


def macro_auc(y: np.ndarray, p: np.ndarray) -> float:
    aucs = [auc((y == c).astype(float), p[:, c]) for c in range(N_SUPPORT)
            if 0 < (y == c).sum() < len(y)]
    return float(np.mean(aucs)) if aucs else float("nan")


def paired_ci(diff: np.ndarray, n_boot: int, alpha: float, seed: int):
    rng = np.random.default_rng(seed)
    means = np.array([diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(n_boot)])
    return (float(diff.mean()), float(np.percentile(means, 100 * alpha / 2)),
            float(np.percentile(means, 100 * (1 - alpha / 2))))


def paired_auc_ci(y, p_arm, p_base, n_boot, alpha, seed):
    rng = np.random.default_rng(seed)
    d = np.empty(n_boot)
    for b in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        yi = y[i]
        d[b] = (auc(yi, p_arm[i]) - auc(yi, p_base[i])) if 0 < yi.sum() < len(yi) else np.nan
    d = d[~np.isnan(d)]
    return (float(auc(y, p_arm) - auc(y, p_base)),
            float(np.percentile(d, 100 * alpha / 2)), float(np.percentile(d, 100 * (1 - alpha / 2))))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=None,
                    help="override for smoke-testing only; the real run/reproduction check "
                         "needs the configured epochs (12)")
    ap.add_argument("--pilot-n-learners", type=int, default=None,
                    help="subsample train learners for smoke-testing only")
    args = ap.parse_args()

    cfg_full = yaml.safe_load((REPO / "configs" / "fa_02_bakeoff.yaml").read_text())
    cfg = {k: cfg_full[k] for k in ("seed", "batch_learners", "epochs", "lr", "grad_clip",
                                    "support_weight", "bootstrap_n", "ci_alpha")}
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    s1 = yaml.safe_load((REPO / "configs" / "s1_filter.yaml").read_text())
    irt_cfg = yaml.safe_load((REPO / "configs" / "irt.yaml").read_text())
    v2cfg = yaml.safe_load((REPO / "configs" / "data.yaml").read_text())["fa"]["manifest_v2"]

    assert s1["latent_dim"] == 32, "latent_dim must stay 32 (as in the baseline filter)"
    seed_everything(cfg["seed"])
    device = default_device()
    logger = RunLogger(config={"task": "fa_01l_time_features_screen", **cfg,
                               "session_gap_min": DEFAULT_SESSION_GAP_MIN},
                       data_version="FA-v2-2026-08-09", seed=cfg["seed"], heartbeat_every_s=20.0)

    # -- the peek canary, first, before anything trains --------------------------------------
    max_abs_diff = run_peek_canary(s1["codec_dim"])
    logger.heartbeat(step=0, force=True, stage="peek canary", max_abs_diff_mu=max_abs_diff)
    assert max_abs_diff == 0.0, f"PEEK CANARY FAILED: max|Δμ|={max_abs_diff:.2e}, expected 0.00e+00"

    # -- data ----------------------------------------------------------------------------------
    logger.heartbeat(step=0, force=True, stage="loading FA-0 v2 manifest + codec", device=device)
    ev = load_v2_events(REPO / v2cfg["manifest_out"], REPO / v2cfg["learner_split_out"],
                        REPO / v2cfg["item_split_out"])
    train_ev_raw = ev[ev["split"] == "train"]
    val_ev_raw = ev[ev["split"] == "val"].copy()
    if args.pilot_n_learners:
        keep = sorted(train_ev_raw["learner"].unique())[:args.pilot_n_learners]
        train_ev_raw = train_ev_raw[train_ev_raw["learner"].isin(keep)]

    codec_matrix, pid_to_idx = load_production_codec(
        REPO / "artifacts" / "problem_codec_mean_L15.npz", s1["codec_dim"])
    check_coverage(ev["problem_id"].unique(), pid_to_idx)

    def prepare(sub_ev_raw):
        sub_ev = compute_session_and_exposure(sub_ev_raw, SESSION_GAP_LOG1P_THRESHOLD)
        b = build_padded_batch_v2(sub_ev, codec_matrix, pid_to_idx, device)
        b["support"] = (b["hint_bin"] + 2 * b["saw_answer"]).long()
        learners, t_max = b["learners"], b["valid_mask"].shape[1]

        gap = torch.tensor(scatter_field(sub_ev, learners, t_max, "gap_log1p_min"),
                           device=device)
        new_session = torch.tensor(scatter_field(sub_ev, learners, t_max, "is_new_session"),
                                   device=device)
        pos_in_session = torch.tensor(
            scatter_field(sub_ev, learners, t_max, "pos_in_session"), device=device)
        first_exposure = torch.tensor(
            scatter_field(sub_ev, learners, t_max, "first_exposure", fill=1.0, dtype=np.float32),
            device=device).bool()
        exposure_index = torch.tensor(
            scatter_field(sub_ev, learners, t_max, "exposure_index"), device=device)

        prev_feat_baseline = build_prev_feat_v2(b["correct"], b["hint_bin"], b["saw_answer"],
                                                b["score_missing"])
        prev_feat_time = build_prev_feat_v2_time(b["correct"], b["hint_bin"], b["saw_answer"],
                                                 b["score_missing"], gap, new_session,
                                                 pos_in_session)
        repeat_feat = build_current_repeat_features(first_exposure, exposure_index)
        return b, {"baseline": prev_feat_baseline, "time": prev_feat_time}, repeat_feat

    train_batch, prev_feat_tr, repeat_feat_tr = prepare(train_ev_raw)
    val_batch, prev_feat_val, repeat_feat_val = prepare(val_ev_raw)
    n_train = len(train_batch["learners"])
    logger.heartbeat(step=0, force=True, stage="batches built", n_train=n_train,
                     n_val=len(val_batch["learners"]),
                     t_max=int(train_batch["valid_mask"].shape[1]))

    # -- eval population: calib_eval_split's held-out half, same court as fa_02_bakeoff -------
    _, eval_rows = calib_eval_split(val_ev_raw, irt_cfg["calib_frac"], irt_cfg["min_interactions"])
    learner_of = {u: i for i, u in enumerate(val_batch["learners"])}
    rows_n = eval_rows["learner"].map(learner_of).to_numpy()
    rows_t = eval_rows["t"].to_numpy().astype(np.int64)
    correct_valid = (~eval_rows["score_missing"]).to_numpy()
    y_correct = val_batch["correct"].cpu().numpy()[rows_n, rows_t]
    y_support = val_batch["support"].cpu().numpy()[rows_n, rows_t]

    # -- train both arms -----------------------------------------------------------------------
    (REPO / "artifacts" / "checkpoints").mkdir(parents=True, exist_ok=True)
    results, per_row, timings = {}, {}, {}
    t_start = time.time()
    for arm in ARMS:
        logger.heartbeat(step=0, force=True, stage=f"training {arm}")
        model, heads, tinfo = train_arm(arm, cfg, s1, device, logger, train_batch,
                                        prev_feat_tr[arm], train_batch["support"],
                                        repeat_feat_tr, n_train)
        timings[arm] = tinfo
        # Checkpoint persisted, not only aggregate metrics, so a later analysis (a new
        # stratification, a calibration curve) needs no retrain.
        ckpt_path = REPO / "artifacts" / "checkpoints" / f"fa_01l_{arm}.pt"
        torch.save({"model_state_dict": model.state_dict(), "heads_state_dict": heads.state_dict(),
                   "arm": arm, "prev_feat_dim": PREV_FEAT_DIM_V2_TIME if arm == "time" else 5,
                   "s1_config": s1, "epochs": cfg["epochs"], "seed": cfg["seed"]}, ckpt_path)
        p_correct_full, p_support_full = predict_val(
            model, heads, arm, val_batch, prev_feat_val[arm], repeat_feat_val, device,
            s1["k_eval"], s1["latent_dim"], cfg["batch_learners"], cfg["seed"])
        pc = p_correct_full[rows_n, rows_t]
        ps = p_support_full[rows_n, rows_t]
        per_row[arm] = {"pc": pc, "ps": ps}

        lab = correct_valid
        results[arm] = {
            "correct_auc": auc(y_correct[lab], pc[lab]),
            "correct_nll": float(-np.mean(y_correct[lab] * np.log(np.clip(pc[lab], 1e-9, 1))
                                          + (1 - y_correct[lab]) * np.log(np.clip(1 - pc[lab], 1e-9, 1)))),
            "correct_ece": ece(y_correct[lab], pc[lab]),
            "correct_acc": accuracy(y_correct[lab], pc[lab]),
            "correct_brier": brier(y_correct[lab], pc[lab]),
            "support_nll": float(-np.mean(np.log(np.clip(ps[np.arange(len(y_support)), y_support], 1e-9, 1)))),
            "support_macro_auc": macro_auc(y_support, ps),
            "support_acc": float((ps.argmax(1) == y_support).mean()),
            "n_correct": int(lab.sum()), "n_support": int(len(y_support)),
        }
        logger.heartbeat(step=0, force=True, stage=f"{arm} scored",
                         correct_auc=round(results[arm]["correct_auc"], 4),
                         support_nll=round(results[arm]["support_nll"], 4),
                         wall_s=round(tinfo["wall_s"]))
    total_wall = time.time() - t_start

    # -- paired comparison: time vs baseline --------------------------------------------------
    lab = correct_valid
    d_c_nll = (-np.log(np.clip(per_row["time"]["pc"][lab], 1e-9, 1)) * y_correct[lab]
              - np.log(np.clip(1 - per_row["time"]["pc"][lab], 1e-9, 1)) * (1 - y_correct[lab])
              ) - (-np.log(np.clip(per_row["baseline"]["pc"][lab], 1e-9, 1)) * y_correct[lab]
                  - np.log(np.clip(1 - per_row["baseline"]["pc"][lab], 1e-9, 1)) * (1 - y_correct[lab]))
    idx_sup = np.arange(len(y_support))
    d_s_nll = (-np.log(np.clip(per_row["time"]["ps"][idx_sup, y_support], 1e-9, 1))
              + np.log(np.clip(per_row["baseline"]["ps"][idx_sup, y_support], 1e-9, 1)))
    paired = {
        "correct_nll": paired_ci(d_c_nll, cfg["bootstrap_n"], cfg["ci_alpha"], cfg["seed"]),
        "support_nll": paired_ci(d_s_nll, cfg["bootstrap_n"], cfg["ci_alpha"], cfg["seed"]),
        "correct_auc": paired_auc_ci(y_correct[lab], per_row["time"]["pc"][lab],
                                     per_row["baseline"]["pc"][lab], cfg["bootstrap_n"],
                                     cfg["ci_alpha"], cfg["seed"]),
    }

    # -- per-row val predictions, not just aggregates -----------------------------------------
    import pandas as pd
    pred_df = eval_rows[["learner", "t", "problem_id", "score_missing"]].copy()
    if "split_item" in eval_rows.columns:
        pred_df["split_item"] = eval_rows["split_item"].to_numpy()
    pred_df["y_correct"] = y_correct
    pred_df["y_support"] = y_support
    for arm in ARMS:
        pred_df[f"{arm}_p_correct"] = per_row[arm]["pc"]
        for c in range(N_SUPPORT):
            pred_df[f"{arm}_p_support_{c}"] = per_row[arm]["ps"][:, c]
    pred_path = REPO / "reports" / "fa_01l_predictions.csv.gz"
    pred_df.to_csv(pred_path, index=False, compression="gzip")

    # -- reproduction check: baseline arm should reproduce fa_02_bakeoff's GRU arm ------------
    # Only meaningful at full scale/epochs -- a pilot (fewer epochs/learners) is expected to
    # miss the converged numbers, as fa_02_bakeoff.py's gru_ref_ok also skips this under
    # --pilot.
    is_pilot = args.epochs is not None or args.pilot_n_learners is not None
    repro = {k: results["baseline"][k] - FA2_GRU_REF[k] for k in FA2_GRU_REF}
    repro_ok = is_pilot or all(abs(v) <= FA2_GRU_REPRO_TOL for v in repro.values())

    # -- outcome rule, from the paired support-NLL CI -------------------------------------------
    s_lo, s_hi = paired["support_nll"][1], paired["support_nll"][2]
    if s_hi < 0:
        outcome = "CLEARLY POSITIVE -- time features improve support NLL"
    elif s_lo > 0:
        outcome = "CLEARLY NEGATIVE -- time features worsen support NLL"
    else:
        outcome = "AMBIGUOUS -- the paired support-NLL CI straddles zero"

    passed = bool(repro_ok and all(t["last_loss"] < t["first_loss"] for t in timings.values())
                 and max_abs_diff == 0.0)

    summary = {
        "task": "fa_01l_time_features_screen", "peek_canary_max_abs_diff": max_abs_diff,
        "results": results, "paired_time_minus_baseline": paired,
        "reproduction_check": {"delta_vs_fa2_gru": repro, "ok": repro_ok, "tol": FA2_GRU_REPRO_TOL},
        "outcome": outcome, "n_train": n_train, "n_val": len(val_batch["learners"]),
        "total_wall_s": total_wall, "session_gap_min": DEFAULT_SESSION_GAP_MIN,
        "device": device, "epochs": cfg["epochs"],
        "predictions_path": str(pred_path.relative_to(REPO)), "n_prediction_rows": len(pred_df),
        "checkpoints": {arm: f"artifacts/checkpoints/fa_01l_{arm}.pt" for arm in ARMS},
    }
    import json
    (REPO / "reports" / "fa_01l_time_features_screen_raw.json").write_text(
        json.dumps(summary, indent=2, default=float))

    def _pair(t):
        d, lo, hi = t
        star = "" if lo <= 0 <= hi else "  **CI clear of zero**"
        return f"{d:+.4f} [{lo:+.4f}, {hi:+.4f}]{star}"

    metrics = {
        "peek canary max|Δμ|": f"{max_abs_diff:.2e}  (must be exactly 0.00e+00)",
        "reproduction check vs FA-2 GRU arm (correct_auc/correct_nll/support_nll)":
            f"Δ={repro['correct_auc']:+.4f}/{repro['correct_nll']:+.4f}/{repro['support_nll']:+.4f} "
            f"(tol ±{FA2_GRU_REPRO_TOL})",
        "session-gap threshold": f"{DEFAULT_SESSION_GAP_MIN:.0f} min (screen-level choice, not swept)",
        "total wall (s), both arms": f"{total_wall:.0f}",
        "--- baseline (FA-2's GRU arm, 5-channel prev_feat) ---": "",
        "baseline · correctness": f"AUC {results['baseline']['correct_auc']:.4f} · "
                                  f"NLL {results['baseline']['correct_nll']:.4f} · "
                                  f"ECE {results['baseline']['correct_ece']:.4f}",
        "baseline · support": f"NLL {results['baseline']['support_nll']:.4f} · "
                              f"macroAUC {results['baseline']['support_macro_auc']:.4f} · "
                              f"acc {results['baseline']['support_acc']:.4f}",
        "--- time (8-channel prev_feat + repeat-exposure emission features) ---": "",
        "time · correctness": f"AUC {results['time']['correct_auc']:.4f} · "
                              f"NLL {results['time']['correct_nll']:.4f} · "
                              f"ECE {results['time']['correct_ece']:.4f}",
        "time · support": f"NLL {results['time']['support_nll']:.4f} · "
                          f"macroAUC {results['time']['support_macro_auc']:.4f} · "
                          f"acc {results['time']['support_acc']:.4f}",
        "--- paired Δ (time - baseline) ---": "",
        "correctness NLL": _pair(paired["correct_nll"]),
        "correctness AUC": _pair(paired["correct_auc"]),
        "support NLL": _pair(paired["support_nll"]),
        "OUTCOME (pre-registered rule)": outcome,
        "per-row val predictions (retention policy, 15 Aug)":
            f"reports/fa_01l_predictions.csv.gz ({len(pred_df):,} rows)",
        "checkpoints (retention policy, 15 Aug)":
            f"artifacts/checkpoints/fa_01l_{{{','.join(ARMS)}}}.pt",
    }

    gate_report(
        path="reports/fa_01l_time_features_screen.md",
        title="Time/session/repeat-exposure feature screen on the 4-class support head",
        meta={"run_id": logger.run_id, "git_sha": logger.git_sha,
              "config_hash": logger.config_hash, "device": device,
              "epochs": cfg["epochs"], "batch_learners": cfg["batch_learners"], "lr": cfg["lr"],
              "train_learners": n_train, "val_learners": len(val_batch["learners"]),
              "session_gap_min": DEFAULT_SESSION_GAP_MIN},
        metrics=metrics,
        passed=passed,
        gate_desc="This is a screen, not a pass/fail on the effect itself -- the outcome "
                  "rule (positive/negative/ambiguous) reports the effect, not this gate. "
                  "What IS gated: (a) "
                  "the peek canary reads exactly 0.00e+00 (a nonzero value would mean the "
                  "time-augmented recurrent input leaks the current event, invalidating "
                  "b_t = f(H_<t)); (b) the baseline arm reproduces the fa_02_bakeoff GRU-arm numbers "
                  f"within ±{FA2_GRU_REPRO_TOL} (else the harness, not the objective, drifted); "
                  "(c) both arms actually trained (correctness loss decreased). FAILS IF: the "
                  "canary is nonzero, the reproduction check misses tolerance, or either arm "
                  "did not train.",
        notes=[
            "TWO KINDS OF TIME FEATURE, legal in different places -- see "
            "`atom.filter.fa_time_features` module docstring. RECURRENT input (shifted, "
            "describes event t-1): gap-since-previous-event (log1p minutes), session-boundary "
            "flag, position-in-session. EMISSION path (current item, legal exactly like "
            "item_codec[t] itself): is-repeat-exposure, log1p(exposure index).",
            "WHY THE RECURRENT CHANNELS ARE SHIFTED, not read at t directly: at belief time "
            "(forming mu_t before item t is answered) the learner's NEXT engagement moment is "
            "not yet known -- the same reason event_transformer.py shifts node_time_min "
            "before using it in the transformer body's attention bias.",
            "SESSION-GAP THRESHOLD is a screen-level choice (30 min), not swept -- the "
            "candidate set is {30, 60, 120} min; this picks the shortest, most "
            "conservative value without tuning.",
            "SAME GRU BODY THROUGHOUT -- this is not a comparison of filter bodies, "
            "only of the recurrent-input and emission features.",
            "PAIRING: identical seed, identical minibatch order (one RNG, seeded the same for "
            "both arms), heads seeded identically wherever shapes match (only the trunk widens "
            "for the time arm) -- a between-arm difference is not sampling noise on any of "
            "these axes.",
            "Support macro-AUC is one-vs-rest, averaged over classes present in the eval "
            "population (identical population for both arms).",
        ],
    )
    logger.summary(status="ok" if passed else "gate-fail", total_wall_s=round(total_wall),
                   outcome=outcome)
    print(f"\ntime-feature screen: {'PASS' if passed else 'GATE-FAIL'} · {outcome}")
    print(f"  peek canary max|Δμ| = {max_abs_diff:.2e}")
    print(f"  baseline vs fa_02_bakeoff GRU repro: {repro}")
    print(f"  support NLL: baseline {results['baseline']['support_nll']:.4f} -> "
          f"time {results['time']['support_nll']:.4f}  (Δ {_pair(paired['support_nll'])})")
    print(f"  correctness AUC: baseline {results['baseline']['correct_auc']:.4f} -> "
          f"time {results['time']['correct_auc']:.4f}  (Δ {_pair(paired['correct_auc'])})")
    print("  report: reports/fa_01l_time_features_screen.md")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
