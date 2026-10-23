#!/usr/bin/env python
"""Screen the filter body: GRU against a causal event-transformer and three attention-bias ablations.

Five arms (gru; c_distance_time, attention biased by semantic distance and elapsed time;
c_distance; c_time; c_none, plain causal attention) train on the FA v2 training learners with
identical data, seed, minibatch order, heads and loss (correctness BCE + 4-class support CE +
KL), under a matched body-parameter budget, and are scored on the fixed NEAR/FAR/REACH-BACK
strata from fa_02a_strata_v2.py with common random numbers and paired bootstrap CIs against the
GRU. This is a screen under a cheap objective, not the body selection for the simulator (whose
filter trains end to end through the backbone). Output: `reports/fa_02_body_c_screen<tag>.md`.
Run: `python scripts/fa_02_bakeoff.py [--pilot] [--arms gru,c_distance_time]`. Serves Appendix A.
"""

from __future__ import annotations

import argparse
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
from atom.filter.event_transformer import S1FilterV2C, count_body_params  # noqa: E402
from atom.filter.gru import sample_u  # noqa: E402
from atom.filter.losses import gaussian_kl_free_bits, kl_anneal_weight, masked_bce  # noqa: E402
from atom.filter.s1v2 import S1FilterV2  # noqa: E402
from atom.heads.fa2_heads import N_SUPPORT, FiLMHeadsSupport  # noqa: E402
from atom.utils.checkpoints import save_checkpoint  # noqa: E402
from atom.utils.logging import LossSpikeTripwire, RunLogger, default_device  # noqa: E402
from atom.utils.seeds import seed_everything  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[1]

ARM_BIAS_MODE = {"c_distance_time": "distance_time", "c_distance": "distance",
                 "c_time": "time", "c_none": "none"}
STRATA = ("ALL", "NEAR", "FAR", "REACH-BACK")
KL_ANNEAL_FRAC = 0.075   # the full-batch GRU baseline's 30/400 epochs, in optimiser steps


# -- models ------------------------------------------------------------------------------

def build_arm(arm: str, cfg: dict, s1: dict, device: str):
    """Both bodies behind one calling convention. The GRU arm is `S1FilterV2` unchanged (the
    baseline filter, not a re-implementation), wrapped only so this script can call either
    body with the same argument list."""
    torch.manual_seed(cfg["seed"])
    if arm == "gru":
        model = S1FilterV2(s1["codec_dim"], s1["episode_hidden"], s1["filter_hidden"],
                           s1["latent_dim"]).to(device)
        forward = lambda b, pf, tm, vm: model(b, pf)  # noqa: E731  (GRU ignores the geometry)
    else:
        model = S1FilterV2C(s1["codec_dim"], s1["episode_hidden"], s1["filter_hidden"],
                            s1["latent_dim"], n_heads=cfg["n_heads"], d_ff=cfg["d_ff"],
                            n_rbf=cfg["n_rbf"], bias_mode=ARM_BIAS_MODE[arm]).to(device)
        forward = lambda b, pf, tm, vm: model(b, pf, tm, vm)  # noqa: E731
    torch.manual_seed(cfg["seed"])   # heads identical across arms, not just same-shaped
    heads = FiLMHeadsSupport(s1["codec_dim"], s1["latent_dim"], s1["filter_hidden"]).to(device)
    return model, heads, forward


def slice_batch(batch: dict, idx: torch.Tensor, prev_feat: torch.Tensor, support: torch.Tensor):
    """Take a learner minibatch and trim it to that minibatch's own longest sequence -- the
    trimmed positions are pure right-padding, so this is exact, and it removes ~20% of the
    O(T^2) attention work at FA's length distribution."""
    t_max = int(batch["lengths"][idx].max().item())
    s = lambda x: x[idx, :t_max]  # noqa: E731
    return {"item_codec": s(batch["item_codec"]), "correct": s(batch["correct"]),
            "label_valid": s(batch["label_valid"]), "valid_mask": s(batch["valid_mask"]),
            "time_min": s(batch["time_min"]), "prev_feat": s(prev_feat),
            "support": s(support)}


def losses_for(mb: dict, mu, log_sigma, heads, cfg, s1, bce, k: int):
    u = sample_u(mu, log_sigma, k=k)
    correctness_logits, support_logits = heads(mb["item_codec"], u)

    label_mask_k = mb["label_valid"].unsqueeze(0).expand(k, -1, -1)
    correct_k = mb["correct"].unsqueeze(0).expand(k, -1, -1)
    correctness_loss = masked_bce(correctness_logits, correct_k, label_mask_k, bce)

    # Joint 4-class support CE instead of two independent engagement BCEs (see
    # heads/fa2_heads.py for why the joint is the output family the simulator uses).
    support_k = mb["support"].unsqueeze(0).expand(k, -1, -1)
    per_cell = F.cross_entropy(support_logits.reshape(-1, N_SUPPORT), support_k.reshape(-1),
                               reduction="none").view(support_k.shape)
    vm_k = mb["valid_mask"].unsqueeze(0).expand(k, -1, -1).float()
    support_loss = (per_cell * vm_k).sum() / vm_k.sum().clamp(min=1.0)

    kl = gaussian_kl_free_bits(mu, log_sigma, s1["free_bits"])
    vm = mb["valid_mask"].float()
    kl_loss = (kl * vm).sum() / vm.sum().clamp(min=1.0)
    return correctness_loss, support_loss, kl_loss


def train_arm(arm, cfg, s1, device, logger, batch, prev_feat, support, n_learners):
    model, heads, forward = build_arm(arm, cfg, s1, device)
    params = list(model.parameters()) + list(heads.parameters())
    opt = torch.optim.Adam(params, lr=cfg["lr"])
    bce = nn.BCEWithLogitsLoss(reduction="none")
    spike = LossSpikeTripwire(factor=s1["spike_factor"], window=s1["spike_window"],
                              patience=s1["spike_patience"])

    steps_per_epoch = int(np.ceil(n_learners / cfg["batch_learners"]))
    total_steps = steps_per_epoch * cfg["epochs"]
    anneal_steps = max(1, int(KL_ANNEAL_FRAC * total_steps))
    warmup_steps = int(cfg.get("warmup_frac", 0.0) * total_steps)
    # One RNG per arm, seeded identically -> every arm sees the SAME learner minibatches in
    # the same order (paired training, not merely matched hyperparameters).
    order_rng = np.random.default_rng(cfg["seed"])

    t0 = time.time()
    step = 0
    first_loss = last_loss = None
    for epoch in range(cfg["epochs"]):
        model.train()
        heads.train()
        perm = torch.as_tensor(order_rng.permutation(n_learners), device=device)
        for b0 in range(0, n_learners, cfg["batch_learners"]):
            mb = slice_batch(batch, perm[b0:b0 + cfg["batch_learners"]], prev_feat, support)
            mu, log_sigma = forward(mb["item_codec"], mb["prev_feat"], mb["time_min"],
                                    mb["valid_mask"])
            c_loss, s_loss, kl_loss = losses_for(mb, mu, log_sigma, heads, cfg, s1, bce,
                                                 s1["k_train"])
            loss = (c_loss + cfg["support_weight"] * s_loss
                    + kl_anneal_weight(step, s1["kl_weight_max"], anneal_steps) * kl_loss)
            logger.tripwire_finite(f"{arm}_loss", loss)
            spike.update(c_loss.item())

            if warmup_steps and step < warmup_steps:
                for group in opt.param_groups:
                    group["lr"] = cfg["lr"] * (step + 1) / warmup_steps

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, cfg["grad_clip"])
            opt.step()

            last_loss = c_loss.item()
            first_loss = last_loss if first_loss is None else first_loss
            step += 1
            logger.heartbeat(step=step, total=total_steps, stage=arm, epoch=epoch + 1,
                             correctness=round(last_loss, 4), support=round(s_loss.item(), 4),
                             kl=round(kl_loss.item(), 4))

    wall_s = time.time() - t0
    save_checkpoint({"filter_model": model.state_dict(), "heads": heads.state_dict()},
                    logger.run_dir / arm, step=total_steps, metric=last_loss, is_best=True)
    return model, heads, forward, {"wall_s": wall_s, "first_loss": first_loss,
                                    "last_loss": last_loss, "steps": total_steps}


# -- evaluation ---------------------------------------------------------------------------

@torch.no_grad()
def predict_val(model, heads, forward, batch, prev_feat, device, k_eval, latent_dim,
                chunk: int, seed: int):
    """Val predictions in learner chunks. Common random numbers: ONE frozen [K, latent_dim]
    noise draw, shared by every arm and every chunk, so an arm never wins or loses on the
    luck of its K samples."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    noise = torch.randn(k_eval, latent_dim, generator=g).to(device)

    n = batch["valid_mask"].shape[0]
    p_correct, p_support = [], []
    model.eval()
    heads.eval()
    for i in range(0, n, chunk):
        idx = torch.arange(i, min(i + chunk, n), device=device)
        mb = slice_batch(batch, idx, prev_feat, batch["support"])
        mu, log_sigma = forward(mb["item_codec"], mb["prev_feat"], mb["time_min"],
                                mb["valid_mask"])
        u = sample_u(mu, log_sigma, k=k_eval, noise=noise)
        c_logits, s_logits = heads(mb["item_codec"], u)
        pc = torch.sigmoid(c_logits).mean(dim=0)
        ps = torch.softmax(s_logits, dim=-1).mean(dim=0)
        pad = batch["valid_mask"].shape[1] - pc.shape[1]
        if pad:   # restore the full T so cell indices stay comparable across chunks
            pc = F.pad(pc, (0, pad))
            ps = F.pad(ps, (0, 0, 0, pad))
        p_correct.append(pc.cpu())
        p_support.append(ps.cpu())
    return torch.cat(p_correct).numpy(), torch.cat(p_support).numpy()


def macro_auc(y: np.ndarray, p: np.ndarray) -> float:
    """One-vs-rest macro AUC over the support classes present in this cell set. Classes
    absent from a stratum are skipped (AUC undefined), and how many were skipped is
    reported alongside so a macro average is never quietly taken over a different set of
    classes for different arms -- the class SET depends only on the cells, which are shared."""
    aucs = [auc((y == c).astype(float), p[:, c]) for c in range(N_SUPPORT)
            if 0 < (y == c).sum() < len(y)]
    return float(np.mean(aucs)) if aucs else float("nan")


def paired_ci(diff: np.ndarray, n_boot: int, alpha: float, seed: int) -> tuple[float, float, float]:
    """Percentile bootstrap over CELLS for a paired per-cell difference (arm - GRU). Paired
    at the cell level, which is stronger than a shared-resample pairing: both arms scored the
    same cell, so the difference is computed before any resampling."""
    rng = np.random.default_rng(seed)
    means = np.array([diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(n_boot)])
    return (float(diff.mean()), float(np.percentile(means, 100 * alpha / 2)),
            float(np.percentile(means, 100 * (1 - alpha / 2))))


def paired_auc_ci(y, p_arm, p_gru, n_boot, alpha, seed):
    """Shared-resample bootstrap for the AUC difference (AUC is not a per-cell statistic, so
    it cannot be paired the way NLL is)."""
    rng = np.random.default_rng(seed)
    d = np.empty(n_boot)
    for b in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        yi = y[i]
        d[b] = (auc(yi, p_arm[i]) - auc(yi, p_gru[i])) if 0 < yi.sum() < len(yi) else np.nan
    d = d[~np.isnan(d)]
    return (float(auc(y, p_arm) - auc(y, p_gru)),
            float(np.percentile(d, 100 * alpha / 2)), float(np.percentile(d, 100 * (1 - alpha / 2))))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true",
                    help="subsample learners/epochs and extrapolate the full-run wall-clock time")
    ap.add_argument("--arms", type=str, default=None, help="comma-separated subset of arms")
    ap.add_argument("--warmup", type=float, default=None,
                    help="diagnostic: linear LR warmup fraction, applied to every arm")
    ap.add_argument("--tag", type=str, default="",
                    help="report filename suffix, so a diagnostic run does not overwrite the "
                         "main report")
    args = ap.parse_args()

    cfg = yaml.safe_load((REPO / "configs" / "fa_02_bakeoff.yaml").read_text())
    s1 = yaml.safe_load((REPO / "configs" / "s1_filter.yaml").read_text())
    irt_cfg = yaml.safe_load((REPO / "configs" / "irt.yaml").read_text())
    v2cfg = yaml.safe_load((REPO / "configs" / "data.yaml").read_text())["fa"]["manifest_v2"]
    arms = args.arms.split(",") if args.arms else cfg["arms"]
    if args.pilot:
        cfg = {**cfg, "epochs": cfg["pilot_epochs"]}
    if args.warmup is not None:
        cfg = {**cfg, "warmup_frac": args.warmup}

    assert s1["latent_dim"] == 32, "the comparison holds latent_dim at 32"
    seed_everything(cfg["seed"])
    device = default_device()
    logger = RunLogger(config={"task": "fa_02_bakeoff", "pilot": args.pilot, "arms": arms, **cfg},
                       data_version="FA-v2-2026-08-09", seed=cfg["seed"], heartbeat_every_s=20.0)

    # -- matched body-parameter budget, asserted before anything trains --------------------
    from atom.filter.gru import KnowledgeFilter
    n_gru = count_body_params(KnowledgeFilter(s1["episode_hidden"], s1["filter_hidden"],
                                              s1["latent_dim"]))
    body_params = {"gru": n_gru}
    for arm in arms:
        if arm != "gru":
            m = S1FilterV2C(s1["codec_dim"], s1["episode_hidden"], s1["filter_hidden"],
                            s1["latent_dim"], n_heads=cfg["n_heads"], d_ff=cfg["d_ff"],
                            n_rbf=cfg["n_rbf"], bias_mode=ARM_BIAS_MODE[arm])
            body_params[arm] = count_body_params(m.knowledge_filter)
    budget_ok = all(abs(v - n_gru) / n_gru <= cfg["param_budget_tol"] for v in body_params.values())
    if not budget_ok:
        raise AssertionError(f"matched parameter budget violated: {body_params}")

    # -- data --------------------------------------------------------------------------------
    logger.heartbeat(step=0, force=True, stage="loading FA-0 v2 manifest + codec", device=device)
    ev = load_v2_events(REPO / v2cfg["manifest_out"], REPO / v2cfg["learner_split_out"],
                        REPO / v2cfg["item_split_out"])
    train_ev, val_ev = ev[ev["split"] == "train"], ev[ev["split"] == "val"].copy()
    if args.pilot:
        keep = sorted(train_ev["learner"].unique())[:cfg["pilot_n_learners"]]
        train_ev = train_ev[train_ev["learner"].isin(keep)]

    codec_matrix, pid_to_idx = load_production_codec(
        REPO / "artifacts" / "problem_codec_mean_L15.npz", s1["codec_dim"])
    check_coverage(ev["problem_id"].unique(), pid_to_idx)

    def prepare(sub_ev):
        b = build_padded_batch_v2(sub_ev, codec_matrix, pid_to_idx, device, include_time_min=True)
        b["support"] = (b["hint_bin"] + 2 * b["saw_answer"]).long()   # == manifest support_code
        pf = build_prev_feat_v2(b["correct"], b["hint_bin"], b["saw_answer"], b["score_missing"])
        return b, pf

    train_batch, prev_feat_tr = prepare(train_ev)
    val_batch, prev_feat_val = prepare(val_ev)
    n_train = len(train_batch["learners"])
    logger.heartbeat(step=0, force=True, stage="batches built", n_train=n_train,
                     n_val=len(val_batch["learners"]),
                     t_max=int(train_batch["valid_mask"].shape[1]))

    # -- the fixed strata (never recomputed here) ---------------------------------------------
    strata_npz = np.load(REPO / cfg["strata_artifact"], allow_pickle=True)
    if int(strata_npz["recency_window"]) != yaml.safe_load(
            (REPO / "configs" / "p3_06_far_from_prefix.yaml").read_text())["recency_window"]:
        raise AssertionError("strata artifact was built with a different recency_window")
    rows_n = strata_npz["learner_idx"]
    rows_t = strata_npz["t"]
    label_valid_cells = strata_npz["label_valid"].astype(bool)
    # Re-derive the cell set independently and assert it matches the frozen artifact: the
    # artifact's (learner_idx, t) are only meaningful against the SAME learner ordering, and
    # a silent reordering would misalign every stratum without changing any count.
    _, eval_rows = calib_eval_split(val_ev, irt_cfg["calib_frac"], irt_cfg["min_interactions"])
    learner_of = {u: i for i, u in enumerate(val_batch["learners"])}
    if not (np.array_equal(eval_rows["learner"].map(learner_of).to_numpy(), rows_n)
            and np.array_equal(eval_rows["t"].to_numpy().astype(np.int64), rows_t)):
        raise AssertionError("frozen strata cells do not match this run's eval cells")

    stratum_masks = {"ALL": np.ones(len(rows_n), dtype=bool),
                     "NEAR": strata_npz["near_mask"], "FAR": strata_npz["far_mask"],
                     "REACH-BACK": strata_npz["reach_back_mask"]}

    y_correct = val_batch["correct"].cpu().numpy()[rows_n, rows_t]
    y_support = val_batch["support"].cpu().numpy()[rows_n, rows_t]

    # -- run the arms ---------------------------------------------------------------------------
    results, per_cell_nll, timings = {}, {}, {}
    t_start = time.time()
    for arm in arms:
        logger.heartbeat(step=0, force=True, stage=f"training {arm}")
        model, heads, forward, tinfo = train_arm(arm, cfg, s1, device, logger, train_batch,
                                                 prev_feat_tr, train_batch["support"], n_train)
        timings[arm] = tinfo
        p_correct_full, p_support_full = predict_val(
            model, heads, forward, val_batch, prev_feat_val, device, s1["k_eval"],
            s1["latent_dim"], cfg["batch_learners"], cfg["seed"])
        pc = p_correct_full[rows_n, rows_t]
        ps = p_support_full[rows_n, rows_t]
        per_cell_nll[arm] = {
            "correct": -(y_correct * np.log(np.clip(pc, 1e-9, 1))
                         + (1 - y_correct) * np.log(np.clip(1 - pc, 1e-9, 1))),
            "support": -np.log(np.clip(ps[np.arange(len(y_support)), y_support], 1e-9, 1)),
            "p_correct": pc, "p_support": ps,
        }
        results[arm] = {}
        for name, m in stratum_masks.items():
            lab = m & label_valid_cells
            results[arm][name] = {
                "n_correct": int(lab.sum()), "n_support": int(m.sum()),
                "correct_auc": auc(y_correct[lab], pc[lab]),
                "correct_nll": float(per_cell_nll[arm]["correct"][lab].mean()),
                "correct_ece": ece(y_correct[lab], pc[lab]),
                "correct_acc": accuracy(y_correct[lab], pc[lab]),
                "correct_brier": brier(y_correct[lab], pc[lab]),
                "support_nll": float(per_cell_nll[arm]["support"][m].mean()),
                "support_macro_auc": macro_auc(y_support[m], ps[m]),
                "support_acc": float((ps[m].argmax(1) == y_support[m]).mean()),
            }
        logger.heartbeat(step=0, force=True, stage=f"{arm} scored",
                         all_auc=round(results[arm]["ALL"]["correct_auc"], 4),
                         reach_back_auc=round(results[arm]["REACH-BACK"]["correct_auc"], 4),
                         support_nll=round(results[arm]["ALL"]["support_nll"], 4),
                         wall_s=round(tinfo["wall_s"]))
    total_wall = time.time() - t_start

    # -- paired comparisons against the GRU -------------------------------------------------------
    paired: dict[str, dict] = {}
    if "gru" in arms and not args.pilot:   # a pilot measures cost and mechanism, not effects
        for arm in [a for a in arms if a != "gru"]:
            paired[arm] = {}
            for name, m in stratum_masks.items():
                lab = m & label_valid_cells
                d_c = per_cell_nll[arm]["correct"][lab] - per_cell_nll["gru"]["correct"][lab]
                d_s = per_cell_nll[arm]["support"][m] - per_cell_nll["gru"]["support"][m]
                paired[arm][name] = {
                    "correct_nll": paired_ci(d_c, cfg["bootstrap_n"], cfg["ci_alpha"], cfg["seed"]),
                    "support_nll": paired_ci(d_s, cfg["bootstrap_n"], cfg["ci_alpha"], cfg["seed"]),
                    "correct_auc": paired_auc_ci(
                        y_correct[lab], per_cell_nll[arm]["p_correct"][lab],
                        per_cell_nll["gru"]["p_correct"][lab], cfg["bootstrap_n"],
                        cfg["ci_alpha"], cfg["seed"]),
                }
            logger.heartbeat(step=0, force=True, stage=f"paired CIs done: {arm}")

    # -- gates ------------------------------------------------------------------------------------
    gru_auc = results.get("gru", {}).get("ALL", {}).get("correct_auc")
    # The GRU-reference gate checks that the harness reproduces the full-batch GRU baseline,
    # which only means anything at full scale: a pilot trains on 200 learners for 3 epochs and
    # is expected to be far from 0.786, so the gate is skipped there rather than failing by
    # construction.
    gru_ref_ok = (args.pilot or gru_auc is None
                  or abs(gru_auc - cfg["gru_reference_auc"]) <= cfg["gru_reference_tol"])
    trained_ok = all(t["last_loss"] < t["first_loss"] for t in timings.values())
    within_wall = total_wall <= cfg["max_wall_s"]
    passed = bool(budget_ok and gru_ref_ok and trained_ok and within_wall)

    logger.log_metrics(passed=passed, pilot=args.pilot, total_wall_s=total_wall,
                       body_params=body_params, gru_reference_auc=gru_auc,
                       **{f"{a}_{s}_{k}": v for a, r in results.items()
                          for s, d in r.items() for k, v in d.items()})
    logger.summary(status="ok" if passed else "gate-fail", total_wall_s=round(total_wall),
                   arms=len(arms))

    if args.pilot:
        per_arm_epoch = np.mean([t["wall_s"] / cfg["epochs"] for t in timings.values()])
        full_cfg = yaml.safe_load((REPO / "configs" / "fa_02_bakeoff.yaml").read_text())
        scale = (len(full_cfg["arms"]) * full_cfg["epochs"]
                 * (3500 / cfg["pilot_n_learners"]) * per_arm_epoch)
        print(f"\nfilter-body PILOT · wall {total_wall:.0f}s · per-arm-epoch {per_arm_epoch:.1f}s "
              f"({cfg['pilot_n_learners']} learners) · projected FULL run "
              f"{scale / 60:.0f} min for {len(full_cfg['arms'])} arms x "
              f"{full_cfg['epochs']} epochs x 3500 learners (local)")
        return 0 if passed else 1

    # -- report ---------------------------------------------------------------------------------
    def _cell(r):
        return (f"AUC {r['correct_auc']:.3f} · NLL {r['correct_nll']:.4f} · "
                f"ECE {r['correct_ece']:.3f}")

    def _sup(r):
        return (f"NLL {r['support_nll']:.4f} · macroAUC {r['support_macro_auc']:.3f} · "
                f"acc {r['support_acc']:.3f}")

    def _pair(t):
        d, lo, hi = t
        star = "" if lo <= 0 <= hi else "  **CI clear of zero**"
        return f"{d:+.4f} [{lo:+.4f}, {hi:+.4f}]{star}"

    metrics: dict[str, object] = {
        "body parameter budget (matched-budget contract)":
            " · ".join(f"{a} {n:,}" for a, n in body_params.items())
            + f"  (tol ±{100 * cfg['param_budget_tol']:.0f}%)",
        "GRU arm vs full-batch GRU reference (0.786; different objective and regime, see notes)":
            f"{gru_auc:.3f} (Δ{gru_auc - cfg['gru_reference_auc']:+.3f})" if gru_auc else "n/a",
        "total wall (s), all arms": f"{total_wall:.0f}",
    }
    for name in STRATA:
        n_c = results[arms[0]][name]["n_correct"]
        n_s = results[arms[0]][name]["n_support"]
        metrics[f"--- [{name}]  n={n_c} labelled / {n_s} cells ---"] = ""
        for arm in arms:
            metrics[f"[{name}] {arm} · correctness"] = _cell(results[arm][name])
            metrics[f"[{name}] {arm} · support"] = _sup(results[arm][name])
        for arm in [a for a in arms if a != "gru"]:
            p = paired[arm][name]
            metrics[f"[{name}] {arm} − gru · paired Δ"] = (
                f"correctness NLL {_pair(p['correct_nll'])} · AUC {_pair(p['correct_auc'])} · "
                f"support NLL {_pair(p['support_nll'])}")

    gate_report(
        path=f"reports/fa_02_body_c_screen{args.tag}.md",
        title="Filter-body screen: GRU vs causal event-transformer "
              "+ three attention-bias ablations",
        meta={"run_id": logger.run_id, "git_sha": logger.git_sha,
              "config_hash": logger.config_hash, "device": device,
              "arms": ", ".join(arms), "epochs": cfg["epochs"],
              "batch_learners": cfg["batch_learners"], "lr": cfg["lr"],
              "warmup_frac": cfg["warmup_frac"],
              "train_learners": n_train, "val_learners": len(val_batch["learners"]),
              "strata artifact (fixed before any arm trained)": cfg["strata_artifact"]},
        metrics=metrics,
        passed=passed,
        gate_desc="This is a screen, so no arm fails for losing. What is gated is that the "
                  "screen is interpretable: (a) the matched parameter-budget contract holds "
                  f"(±{100 * cfg['param_budget_tol']:.0f}%); (b) the GRU arm lands within "
                  f"{cfg['gru_reference_tol']} of the full-batch GRU's 0.786, so a drift is attributable to "
                  "the disclosed objective/regime change rather than to a broken harness; "
                  "(c) every arm actually trained (correctness loss decreased); (d) the whole "
                  "bake-off fits its wall-clock budget. FAILS IF: the budget contract is "
                  "violated, the GRU arm drifts beyond tolerance, any arm's loss did not "
                  "decrease, or the run exceeded its time budget.",
        notes=[
            "This screen does not select the body for the simulator. The filter is not frozen "
            "upstream of the simulator: gradients flow from the output losses through the LoRA "
            "edit and hypernetwork into the belief and the filter itself, so a body choice belongs "
            "under that full objective, where the pre-registered rule is: ship the transformer "
            "only if it wins production NLL with a paired CI clear of zero and does not lose "
            "support/success; a tie keeps the GRU.",
            "The comparison scores support as well as correctness, because correctness alone is "
            "biased against the transformer, whose claimed advantage (attention over semantic "
            "distance and elapsed time) plausibly pays more in production than in a correctness "
            "bit. The 4-class support head needs no backbone, so scoring it here is cheap.",
            "Support is the joint of the two engagement channels, not new information: "
            "support_code = (hint_count>0) + 2*saw_answer. It replaces two independent "
            "engagement BCEs because the FA simulator's output family is the joint categorical, "
            "which models a dependency the marginals miss (they imply 1.1% HINT_AND_REVEAL "
            "against a measured 4.0%). Consequence: the GRU arm here is trained under a "
            "different objective from the full-batch GRU baseline, so 0.786 is a reference, not "
            "an identity check.",
            "Regime change: the full-batch GRU baseline ran 400 full-batch steps (400 gradient "
            "updates in total). The transformer body is O(T^2) in memory and must minibatch, "
            "and a minibatch regime is not comparable to a full-batch one, so every arm trains "
            "under the minibatch regime here.",
            "Pairing is at three levels, so a between-arm difference is not sampling noise: "
            "identical seeds and identical minibatch order (one RNG per arm, seeded the same, "
            "so every arm sees the same learners in the same sequence); one frozen K-sample "
            "noise draw shared across arms at eval (common random numbers); and per-cell "
            "paired differences for the NLL comparisons, where both arms scored the same cell "
            "so the difference is taken before any resampling.",
            "The strata were fixed first, by a separate script "
            "(scripts/fa_02a_strata_v2.py -> artifacts/fa_02_strata_v2.npz, "
            "reports/fa_02a_strata_v2.md). This script never computes a tercile boundary; it "
            "loads them, and it re-derives the eval cell set independently and asserts it "
            "matches the artifact cell-for-cell, because the frozen (learner_idx, t) pairs are "
            "only meaningful against the same learner ordering.",
            "AUC differences use a shared-resample bootstrap because AUC is not a per-cell "
            "statistic; NLL differences use the stronger per-cell pairing. Both are "
            "event-level, not learner-clustered.",
            "Support macro-AUC is one-vs-rest, averaged over the classes actually present in "
            "each stratum. The class set depends only on the cells, which are shared across "
            "arms, so the average is over the same classes for every arm.",
        ],
    )
    print(f"\nfilter-body screen: {'PASS' if passed else 'GATE-FAIL'} · report: "
          f"reports/fa_02_body_c_screen{args.tag}.md")
    for arm in arms:
        r, rb = results[arm]["ALL"], results[arm]["REACH-BACK"]
        print(f"  {arm:17s} ALL AUC {r['correct_auc']:.4f} NLL {r['correct_nll']:.4f} "
              f"supNLL {r['support_nll']:.4f} | REACH-BACK AUC {rb['correct_auc']:.4f} "
              f"supNLL {rb['support_nll']:.4f}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
