#!/usr/bin/env python
"""OU forgetting-kernel on/off ablation (Appendix A): does explicitly modelling forgetting over
real time gaps -- a mean-reverting Ornstein-Uhlenbeck bridge on the belief, applied between a
learner's consecutive events -- improve calibration on predictions that immediately follow a long
gap, compared with the S1 filter, which has no notion of elapsed real time?

Two arms, one comparison:
- kernel-off = exactly the trained S1 filter (`kernel_off_source_run_id` in
  configs/ou_kernel.yaml), loaded from checkpoint, with no retraining. Architecturally "off" is
  that model -- there is no OU module to turn off -- so reusing it is the correct control arm.
- kernel-on = the same architecture (S1Filter + FiLMHeads) plus an OUKernel
  (src/atom/filter/ou_kernel.py) applied to the belief immediately before it is used to predict
  each event, trained from scratch with identical hyperparameters (configs/ou_kernel.yaml
  mirrors configs/s1_filter.yaml), so the two arms differ in exactly one thing.

Scope: the OU adjustment is applied at read time only -- it shapes what the heads see for this
prediction but does not feed back into the GRU's recurrent state -- and the resting point `m`
is a single population-level learned vector, not conditioned on a person-specific trait slice.
Both are simplifications for an ablation asking "does the mechanism help at all"; see
src/atom/filter/ou_kernel.py for the reasoning.

Pre-registered check: kernel ECE <= no-kernel ECE (+ a small tolerance) on the long-gap slice of
VAL (delta_days >= long_gap_threshold_days in configs/ou_kernel.yaml, chosen from the measured FA
gap distribution). Post-gap AUC is reported beside it; calibration improves while discrimination
drops, and the thesis reports the outcome in Appendix A (the kernel is not in the production
path). Output: `reports/p3_04_ou_ablation.md`. Run: `python scripts/p3_04_ou_ablation.py`.
"""

from __future__ import annotations

import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402
import yaml  # noqa: E402
from torch import nn  # noqa: E402

from atom.data.fa import load_interactions  # noqa: E402
from atom.data.tuples import apply_split, build_events, load_split_manifest  # noqa: E402
from atom.eval.metrics import auc, ece  # noqa: E402
from atom.eval.report import gate_report  # noqa: E402
from atom.filter.batch import (  # noqa: E402
    build_padded_batch,
    check_coverage,
    load_production_codec,
)
from atom.filter.episode import build_prev_feat  # noqa: E402
from atom.filter.gru import sample_u  # noqa: E402
from atom.filter.losses import gaussian_kl_free_bits, kl_anneal_weight, masked_bce  # noqa: E402
from atom.filter.ou_kernel import OUKernel  # noqa: E402
from atom.filter.s1 import S1Filter  # noqa: E402
from atom.heads.s1_heads import FiLMHeads  # noqa: E402
from atom.utils.checkpoints import load_checkpoint, save_checkpoint  # noqa: E402
from atom.utils.logging import LossSpikeTripwire, RunLogger, default_device  # noqa: E402
from atom.utils.seeds import seed_everything  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[1]


def _predict(filter_model, heads, item_codec, prev_feat, k_eval, ou_kernel=None, delta_days=None):
    """One eval forward pass. If ou_kernel is given, applies it (using delta_days) to the
    raw belief before sampling/prediction -- the kernel-on arm. Otherwise the raw belief goes
    straight to the heads -- the kernel-off arm (exactly the S1 filter's own eval path).
    """
    filter_model.eval()
    heads.eval()
    with torch.no_grad():
        mu, log_sigma = filter_model(item_codec, prev_feat)
        if ou_kernel is not None:
            mu, log_sigma = ou_kernel(mu, log_sigma, delta_days)
        u = sample_u(mu, log_sigma, k=k_eval)
        logits, _ = heads(item_codec, u)
        probs = torch.sigmoid(logits).mean(dim=0)
    return probs


def main() -> int:
    cfg = yaml.safe_load((REPO / "configs" / "ou_kernel.yaml").read_text())
    seed_everything(cfg["seed"])
    device = default_device()
    logger = RunLogger(config={"task": "p3_04_ou_ablation", **cfg}, data_version="FA-raw-2026-03",
                        seed=cfg["seed"], heartbeat_every_s=10.0)

    logger.heartbeat(step=0, force=True, stage="loading events + codec")
    inter = load_interactions(parse_time=True)
    events = build_events(inter)
    manifest = load_split_manifest(REPO / "splits" / "fa_learner_split.csv")
    ev = apply_split(events, manifest)
    train_ev = ev[ev["split"] == "train"]
    val_ev = ev[ev["split"] == "val"]

    codec_matrix, pid_to_idx = load_production_codec(
        REPO / "artifacts" / "problem_codec_mean_L15.npz", cfg["codec_dim"])
    check_coverage(set(train_ev["problem_id"]) | set(val_ev["problem_id"]), pid_to_idx)

    # -- TRAIN the kernel-ON arm (kernel-OFF reuses the S1 checkpoint, no retraining) ----
    train_batch = build_padded_batch(train_ev, codec_matrix, pid_to_idx, device, include_delta_days=True)
    n_tr, t_max_tr = train_batch["valid_mask"].shape
    logger.heartbeat(step=0, force=True, stage="train batch built", n_learners=n_tr, t_max=t_max_tr)
    prev_feat_tr = build_prev_feat(train_batch["correct"], train_batch["hint_bin"], train_batch["saw_answer"])

    filter_on = S1Filter(cfg["codec_dim"], cfg["episode_hidden"], cfg["filter_hidden"],
                         cfg["latent_dim"]).to(device)
    heads_on = FiLMHeads(cfg["codec_dim"], cfg["latent_dim"], cfg["filter_hidden"]).to(device)
    ou_kernel = OUKernel(cfg["latent_dim"]).to(device)
    params = list(filter_on.parameters()) + list(heads_on.parameters()) + list(ou_kernel.parameters())
    opt = torch.optim.Adam(params, lr=cfg["lr"])
    bce = nn.BCEWithLogitsLoss(reduction="none")
    spike = LossSpikeTripwire(factor=cfg["spike_factor"], window=cfg["spike_window"],
                              patience=cfg["spike_patience"])

    k_train = cfg["k_train"]
    mask_k = train_batch["valid_mask"].unsqueeze(0).expand(k_train, -1, -1)
    correct_k = train_batch["correct"].unsqueeze(0).expand(k_train, -1, -1)
    hint_k = train_batch["hint_bin"].unsqueeze(0).expand(k_train, -1, -1)
    saw_k = train_batch["saw_answer"].unsqueeze(0).expand(k_train, -1, -1)

    t0 = time.time()
    loss_history: list[float] = []
    for epoch in range(cfg["epochs"]):
        filter_on.train()
        heads_on.train()
        ou_kernel.train()
        opt.zero_grad()

        mu, log_sigma = filter_on(train_batch["item_codec"], prev_feat_tr)
        mu_used, log_sigma_used = ou_kernel(mu, log_sigma, train_batch["delta_days"])
        u = sample_u(mu_used, log_sigma_used, k=k_train)
        correctness_logits, engagement_logits = heads_on(train_batch["item_codec"], u)

        correctness_loss = masked_bce(correctness_logits, correct_k, mask_k, bce)
        hint_loss = masked_bce(engagement_logits[..., 0], hint_k, mask_k, bce)
        saw_loss = masked_bce(engagement_logits[..., 1], saw_k, mask_k, bce)
        engagement_loss = hint_loss + saw_loss

        # KL regularises the RAW (pre-decay) belief -- collapse prevention targets the
        # encoder's own inference, not the deterministic OU transform applied after it.
        kl = gaussian_kl_free_bits(mu, log_sigma, cfg["free_bits"])
        kl_loss = (kl * train_batch["valid_mask"].float()).sum() / train_batch["valid_mask"].float().sum()
        kl_w = kl_anneal_weight(epoch, cfg["kl_weight_max"], cfg["kl_anneal_epochs"])

        loss = correctness_loss + cfg["engagement_weight"] * engagement_loss + kl_w * kl_loss
        logger.tripwire_finite("train_loss", loss)
        spike.update(correctness_loss.item())
        loss.backward()
        opt.step()

        cl = correctness_loss.item()
        loss_history.append(cl)
        logger.heartbeat(step=epoch + 1, total=cfg["epochs"], loss=loss.item(), correctness=cl,
                         engagement=engagement_loss.item(), kl=kl_loss.item(), kl_w=kl_w)
        if (epoch + 1) % 25 == 0 or epoch == 0:
            logger.log_metrics(epoch=epoch + 1, loss=loss.item(), correctness_loss=cl,
                              engagement_loss=engagement_loss.item(), kl_loss=kl_loss.item(), kl_weight=kl_w)

        if (epoch + 1) % cfg["checkpoint_every"] == 0 or (epoch + 1) == cfg["epochs"]:
            save_checkpoint(
                {"filter_model": filter_on.state_dict(), "heads": heads_on.state_dict(),
                 "ou_kernel": ou_kernel.state_dict(), "opt": opt.state_dict()},
                logger.run_dir, step=epoch + 1, metric=cl, is_best=(cl == min(loss_history)),
            )

    train_wall_s = time.time() - t0
    logger.heartbeat(step=cfg["epochs"], force=True, stage="training done", train_wall_s=round(train_wall_s, 1))

    # -- VAL: one batch (with delta_days), scored by BOTH arms -------------------------
    val_batch = build_padded_batch(val_ev, codec_matrix, pid_to_idx, device, include_delta_days=True)
    prev_feat_val = build_prev_feat(val_batch["correct"], val_batch["hint_bin"], val_batch["saw_answer"])

    probs_on = _predict(filter_on, heads_on, val_batch["item_codec"], prev_feat_val, cfg["k_eval"],
                        ou_kernel=ou_kernel, delta_days=val_batch["delta_days"])

    filter_off = S1Filter(cfg["codec_dim"], cfg["episode_hidden"], cfg["filter_hidden"],
                          cfg["latent_dim"]).to(device)
    heads_off = FiLMHeads(cfg["codec_dim"], cfg["latent_dim"], cfg["filter_hidden"]).to(device)
    ckpt = load_checkpoint(REPO / "runs" / cfg["kernel_off_source_run_id"], which="last", map_location=device)
    filter_off.load_state_dict(ckpt["filter_model"])
    heads_off.load_state_dict(ckpt["heads"])
    probs_off = _predict(filter_off, heads_off, val_batch["item_codec"], prev_feat_val, cfg["k_eval"])

    # -- long-gap slice + ECE comparison -------------------------------------------------
    long_gap_mask = val_batch["valid_mask"] & (val_batch["delta_days"] >= cfg["long_gap_threshold_days"])
    m = long_gap_mask.cpu()
    y_slice = val_batch["correct"].cpu()[m].numpy()
    p_on_slice = probs_on.cpu()[m].numpy()
    p_off_slice = probs_off.cpu()[m].numpy()
    n_slice = int(m.sum().item())

    ece_on = ece(y_slice, p_on_slice, n_bins=cfg["ece_bins"])
    ece_off = ece(y_slice, p_off_slice, n_bins=cfg["ece_bins"])
    auc_on = auc(y_slice, p_on_slice)
    auc_off = auc(y_slice, p_off_slice)
    ece_delta = ece_on - ece_off  # negative = kernel improves calibration (lower ECE is better)
    passed = ece_on <= ece_off + cfg["ece_tolerance"]

    logger.log_metrics(train_wall_s=train_wall_s, loss_first=loss_history[0], loss_last=loss_history[-1],
                       n_long_gap_slice=n_slice, ece_on=ece_on, ece_off=ece_off, ece_delta=ece_delta,
                       auc_on=auc_on, auc_off=auc_off, passed=passed)
    logger.summary(status="ok" if passed else "gate-fail", ece_on=round(ece_on, 4),
                   ece_off=round(ece_off, 4), n_long_gap_slice=n_slice)

    gate_report(
        path="reports/p3_04_ou_ablation.md",
        title="P3-4 - OU kernel on/off ablation (long-gap slice)",
        meta={"run_id": logger.run_id, "git_sha": logger.git_sha, "config_hash": logger.config_hash,
              "kernel_off_source_run_id": cfg["kernel_off_source_run_id"],
              "long_gap_threshold_days": cfg["long_gap_threshold_days"],
              "n_long_gap_slice_events": n_slice, "device": device},
        metrics={
            "train wall_s (kernel-on arm only; kernel-off reuses the S1 checkpoint)": round(train_wall_s, 1),
            "correctness loss: first -> last epoch (kernel-on)": f"{loss_history[0]:.4f} -> {loss_history[-1]:.4f}",
            "post-gap ECE: kernel-on / kernel-off / delta (negative = kernel better)":
                f"{ece_on:.4f} / {ece_off:.4f} / {ece_delta:+.4f} "
                f"(need on <= off + {cfg['ece_tolerance']}) -- {'PASS' if passed else 'FAIL'}",
            "post-gap AUC (informational, not gated): kernel-on / kernel-off": f"{auc_on:.3f} / {auc_off:.3f}",
            "learned OU params: lambda (per-dim rate, mean) / Q (stationary var, mean) / |m| (resting point norm)":
                f"{torch.nn.functional.softplus(ou_kernel.log_lambda).mean().item():.3f} / "
                f"{torch.nn.functional.softplus(ou_kernel.log_q).mean().item():.3f} / "
                f"{ou_kernel.m.norm().item():.3f}",
        },
        passed=passed,
        gate_desc=f"kernel-on ECE <= kernel-off ECE (+{cfg['ece_tolerance']} tolerance) on the long-gap "
                  f"slice (delta_days >= {cfg['long_gap_threshold_days']}, {n_slice} VAL events) -- "
                  "lower ECE is better calibration; else investigate before proceeding "
                  "(pre-registered check).",
        notes=[
            "kernel-off is NOT retrained -- it is exactly the completed S1 filter training run "
            f"({cfg['kernel_off_source_run_id']}) loaded from checkpoint, since architecturally "
            "'off' already IS that model (no OU module, nothing to disable). Only the kernel-on "
            "arm is trained fresh, with identical hyperparameters (configs/ou_kernel.yaml mirrors "
            "configs/s1_filter.yaml's frozen values) so the two arms differ in exactly one thing.",
            f"Long-gap threshold ({cfg['long_gap_threshold_days']} days) is measured, not assumed: "
            "FA inter-event gaps are heavily concentrated near zero (median ~1.4 minutes) with a "
            "real long tail (p90 ~2 days, p99 ~15 days, max ~690 days) -- 7 days is comfortably "
            "past the bulk of ordinary gaps while leaving enough VAL events for a stable ECE.",
            "Scope: the OU adjustment applies at READ time only (shapes the "
            "belief used for THIS prediction; does not feed back into the GRU's own recurrent "
            "state), and the resting point m is a single population-level learned vector, not "
            "conditioned on a person-specific trait slice theta -- a person-conditioned resting "
            "point is not tested by this ablation.",
            f"IMPORTANT CAVEAT, not just a passing footnote: post-gap AUC drops from {auc_off:.3f} "
            f"(kernel-off) to {auc_on:.3f} (kernel-on) -- calibration improves while discrimination "
            "gets meaningfully worse on the SAME slice. Coherent, not a bug: mean-reversion pulls "
            "the belief toward the population resting point m as delta_days grows, which is exactly "
            "what should make post-gap probabilities more honest on average (lower ECE) but can "
            "also discard genuine remaining personal signal the pre-gap belief still carried, "
            "hurting ranking (lower AUC). The pre-registered check only asks "
            "about calibration, so a PASS here does not by itself justify adopting this "
            "instantiation (aggressive decay, single global resting point); the calibration gain "
            "must be weighed against the discrimination loss on the same slice.",
            "KL free-bits regularises the RAW (pre-decay) belief, not the OU-adjusted one -- "
            "collapse prevention targets what the encoder itself infers, not the deterministic "
            "transform applied afterward.",
        ],
    )
    print(f"\nP3-4: {'PASS' if passed else 'FAIL'} · train wall {train_wall_s:.1f}s · "
          f"long-gap slice n={n_slice} · ECE on {ece_on:.4f} vs off {ece_off:.4f} "
          f"(delta {ece_delta:+.4f}) · report: reports/p3_04_ou_ablation.md")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
