#!/usr/bin/env python
"""Train the outcome-only filter used as FA-SIM3's warm start, and measure it.

Retrains the best item-encoder cell of `fa_01i_item_encoder_ablation.py` (`S1FilterV2ItemMLP`
with `FiLMHeadsItemMLP`, latent=256, filter_hidden=256, episode_hidden=256, experimental
codec_dim=64 from `artifacts/fa_item_codec_dim64.npz`) and produces three things:
  (i)   the checkpoint `artifacts/checkpoints/fa_outcome_filter.pt`, from the regime trained on
        all items for the training learners; FA-SIM3 warm-starts from it;
  (ii)  the within-item AUC of that model (AUC across learners within each item, pooled over
        items by pair count; an item-only predictor scores exactly 0.5), read against the
        person-only benchmark of 0.5942;
  (iii) an item-memorisation difference-in-differences, which the thesis no longer reports:
        a second regime trains only on train-split items, and with AUCs on train items (A, C)
        and held-out items (B, D) for the all-items and train-items-only regimes,
        memorisation_did = (B - D) - (A - C), with a bootstrap 95% CI.
The SHA-256 of both split files is recorded in the checkpoint and the summary, so later runs
can confirm they used the same split. Training uses the exact gradient-accumulation
micro-batching of `fa_01g_s1v2_width_sweep.py` by default (350 learners per chunk), because
this width runs out of memory on a laptop GPU without it.

Inputs: the v2 FA manifest and splits, the codec above, `configs/{s1_filter,irt,data}.yaml`.
Outputs: the checkpoint above and `reports/fa_01k_b4_close_raw.json`.
Run: `python scripts/fa_01k_b4_close.py [--epochs N] [--micro-batch-learners N]`.
Serves Chapter 6 (the warm-start filter for FA-SIM3, Tables 10 and 12).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from torch import nn  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[1]

from atom.data.fa_manifest import load_v2_events  # noqa: E402
from atom.eval.metrics import accuracy, auc, brier, ece  # noqa: E402
from atom.eval.protocol import calib_eval_split  # noqa: E402
from atom.filter.batch import build_padded_batch_v2, check_coverage  # noqa: E402
from atom.filter.episode_v2 import build_prev_feat_v2  # noqa: E402
from atom.filter.gru import sample_u  # noqa: E402
from atom.filter.item_encoder_mlp import FiLMHeadsItemMLP, S1FilterV2ItemMLP  # noqa: E402
from atom.filter.losses import gaussian_kl_free_bits, kl_anneal_weight, masked_bce  # noqa: E402
from atom.utils.logging import RunLogger, default_device  # noqa: E402
from atom.utils.seeds import seed_everything  # noqa: E402

B4_CELL = {"latent_dim": 256, "filter_hidden": 256, "episode_hidden": 256}
CODEC_DIM = 64
CODEC_PATH = REPO / "artifacts" / "fa_item_codec_dim64.npz"
CHECKPOINT_DIR = REPO / "artifacts" / "checkpoints"

# Prior measurements, context only -- not re-derived here.
D1_AKT_MEMORISATION_DID = 0.0991
D2_S1V2_MEMORISATION_DID = 0.0477      # S1v2 with the linear item_proj, codec 45; different architecture
FA_01I_B4_BEST_NONE_AUC = 0.6913       # fa_01i best cell (summary only); reproduction check
PERSON_ONLY_WITHIN_ITEM_AUC = 0.5942   # fa_01j constant person baseline
ITEM_ONLY_POOLED_AUC = 0.6425          # fa_01j constant item baseline
N_BOOT = 1000
MIN_PAIRS_PER_ITEM = 1


def _sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _yp(mask, correct, probs) -> tuple[np.ndarray, np.ndarray]:
    m = mask.cpu()
    return correct.cpu()[m].numpy().astype(np.int64), probs.cpu()[m].numpy().astype(np.float64)


def _score_yp(y, p) -> dict:
    return {"auc": auc(y, p), "accuracy": accuracy(y, p), "brier": brier(y, p),
            "ece": ece(y, p), "n": int(len(y)), "base_rate": float(y.mean())}


def _mask_from_rows(rows, learners, t_max, device):
    learner_of = {u: i for i, u in enumerate(learners)}
    mask = np.zeros((len(learners), t_max), dtype=bool)
    idx = rows["learner"].map(learner_of).to_numpy()
    t = rows["t"].to_numpy().astype(int)
    mask[idx, t] = True
    return torch.tensor(mask, dtype=torch.bool, device=device)


def _rederive_t(ev):
    """Identical to the helper in fa_01d_s1v2_item_holdout.py. Compacts each learner's `t`
    after heldout-item events are dropped from the clean regime; a no-op on the all_items
    regime (asserted by the caller)."""
    ev = ev.sort_values(["learner", "t"], kind="mergesort").copy()
    ev["t"] = ev.groupby("learner", sort=False).cumcount()
    return ev


def within_item_auc(problem_ids: np.ndarray, y: np.ndarray, p: np.ndarray) -> dict:
    """The same construction as `within_item_auc` in fa_01j: AUC across learners
    WITHIN each item, pooled over items weighted by usable pair count (n_pos * n_neg). An
    item-only (constant) predictor scores exactly 0.5 here by construction; a trained model's
    score above 0.5 is direct evidence of person-conditional discrimination.
    """
    df = pd.DataFrame({"problem_id": problem_ids, "y": y, "p": p})
    per_item = []
    for pid, g in df.groupby("problem_id", sort=False):
        yy = g["y"].to_numpy().astype(int)
        n_pos, n_neg = int(yy.sum()), int((1 - yy).sum())
        if n_pos < MIN_PAIRS_PER_ITEM or n_neg < MIN_PAIRS_PER_ITEM:
            continue
        a = auc(yy, g["p"].to_numpy())
        if not np.isfinite(a):
            continue
        per_item.append({"auc": a, "pairs": n_pos * n_neg, "n": len(g)})
    if not per_item:
        return {"pooled_auc": float("nan"), "n_items": 0, "n_pairs": 0}
    pi = pd.DataFrame(per_item)
    w = pi["pairs"].to_numpy().astype(float)
    return {"pooled_auc": float(np.average(pi["auc"].to_numpy(), weights=w)),
            "unweighted_mean": float(pi["auc"].mean()), "n_items": int(len(pi)),
            "n_pairs": int(pi["pairs"].sum()), "n_rows_covered": int(pi["n"].sum())}


def train_one_regime(regime: str, train_ev_all, codec_matrix, pid_to_idx, cfg: dict,
                     device: str, logger: RunLogger, micro_batch_learners: int | None):
    """Train S1FilterV2ItemMLP/FiLMHeadsItemMLP under one regime: `all_items` (every
    train-learner event) or `clean` (train-split items only). Gradient-accumulation
    micro-batching as in fa_01g_s1v2_width_sweep.py, mathematically exact.
    """
    train_ev = train_ev_all if regime == "all_items" else \
        train_ev_all[train_ev_all["split_item"] == "train"]
    n_before_rederive = len(train_ev)
    t_max_before = int(train_ev.groupby("learner")["t"].max().max()) + 1
    train_ev = _rederive_t(train_ev)
    if regime == "all_items":
        assert t_max_before == int(train_ev.groupby("learner")["t"].max().max()) + 1, \
            "re-deriving t changed the all_items arm; it must not"
    assert len(train_ev) == n_before_rederive, "re-deriving t must not drop events"

    train_batch = build_padded_batch_v2(train_ev, codec_matrix, pid_to_idx, device)
    prev_feat_tr = build_prev_feat_v2(train_batch["correct"], train_batch["hint_bin"],
                                      train_batch["saw_answer"], train_batch["score_missing"])
    logger.heartbeat(step=0, force=True, stage=f"{regime}: batch built",
                     n_train_events=int(len(train_ev)),
                     n_events_in_batch=int(train_batch["valid_mask"].sum().item()))

    torch.manual_seed(cfg["seed"])
    model = S1FilterV2ItemMLP(CODEC_DIM, B4_CELL["episode_hidden"], B4_CELL["filter_hidden"],
                              B4_CELL["latent_dim"], query_independent=True).to(device)
    heads = FiLMHeadsItemMLP(CODEC_DIM, B4_CELL["latent_dim"], B4_CELL["filter_hidden"]).to(device)
    opt = torch.optim.Adam(list(model.parameters()) + list(heads.parameters()), lr=cfg["lr"])
    bce = nn.BCEWithLogitsLoss(reduction="none")

    k = cfg["k_train"]
    label_k = train_batch["label_valid"].unsqueeze(0).expand(k, -1, -1)
    valid_k = train_batch["valid_mask"].unsqueeze(0).expand(k, -1, -1)
    correct_k = train_batch["correct"].unsqueeze(0).expand(k, -1, -1)
    hint_k = train_batch["hint_bin"].unsqueeze(0).expand(k, -1, -1)
    saw_k = train_batch["saw_answer"].unsqueeze(0).expand(k, -1, -1)
    vmf = train_batch["valid_mask"].float()
    n_learners = train_batch["item_codec"].shape[0]

    label_den = label_k.sum().clamp(min=1.0)
    valid_den = valid_k.sum().clamp(min=1.0)
    kl_den = vmf.sum().clamp(min=1.0)

    t0 = time.time()
    losses: list[float] = []
    mb = micro_batch_learners or n_learners
    for epoch in range(cfg["epochs"]):
        model.train()
        heads.train()
        opt.zero_grad()
        epoch_c_num = 0.0
        kl_w = kl_anneal_weight(epoch, cfg["kl_weight_max"], cfg["kl_anneal_epochs"])
        for lo in range(0, n_learners, mb):
            hi = min(lo + mb, n_learners)
            mu, log_sigma = model(train_batch["item_codec"][lo:hi], prev_feat_tr[lo:hi])
            u = sample_u(mu, log_sigma, k=k)
            c_logits, e_logits = heads(train_batch["item_codec"][lo:hi], u)
            c_per = bce(c_logits, correct_k[:, lo:hi])
            c_num = (c_per * label_k[:, lo:hi].float()).sum()
            hint_num = (bce(e_logits[..., 0], hint_k[:, lo:hi]) * valid_k[:, lo:hi].float()).sum()
            saw_num = (bce(e_logits[..., 1], saw_k[:, lo:hi]) * valid_k[:, lo:hi].float()).sum()
            kl = gaussian_kl_free_bits(mu, log_sigma, cfg["free_bits"])
            kl_num = (kl * vmf[lo:hi]).sum()
            mb_loss = (c_num / label_den
                      + cfg["engagement_weight"] * (hint_num / valid_den + saw_num / valid_den)
                      + kl_w * (kl_num / kl_den))
            logger.tripwire_finite(f"{regime}_train_loss", mb_loss)
            mb_loss.backward()
            epoch_c_num += c_num.item()
        opt.step()
        c_loss_epoch = epoch_c_num / label_den.item()
        losses.append(c_loss_epoch)
        logger.heartbeat(step=epoch + 1, total=cfg["epochs"], stage=f"{regime}: train",
                         correctness=round(c_loss_epoch, 4))
    return model, heads, losses, time.time() - t0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--micro-batch-learners", type=int, default=350)
    args = ap.parse_args()

    cfg_full = yaml.safe_load((REPO / "configs" / "s1_filter.yaml").read_text())
    irt_cfg = yaml.safe_load((REPO / "configs" / "irt.yaml").read_text())
    v2cfg = yaml.safe_load((REPO / "configs" / "data.yaml").read_text())["fa"]["manifest_v2"]
    cfg = {**cfg_full, "epochs": args.epochs or cfg_full["full_epochs"]}
    seed_everything(cfg["seed"])
    device = default_device()
    logger = RunLogger(config={"task": "fa_01k_b4_close", **cfg, **B4_CELL, "codec_dim": CODEC_DIM},
                       data_version="FA-v2-2026-08-09", seed=cfg["seed"], heartbeat_every_s=20.0)

    # -- split hashes, recorded so downstream runs can confirm the same split ----------------
    item_split_hash = _sha256(REPO / v2cfg["item_split_out"])
    learner_split_hash = _sha256(REPO / v2cfg["learner_split_out"])
    logger.heartbeat(step=0, force=True, stage="split hash recorded (gate)",
                     item_split_sha256=item_split_hash[:16], learner_split_sha256=learner_split_hash[:16])

    logger.heartbeat(step=0, force=True, stage="loading manifest + splits + codec", device=device)
    ev = load_v2_events(REPO / v2cfg["manifest_out"], REPO / v2cfg["learner_split_out"],
                        REPO / v2cfg["item_split_out"])
    ev = ev[ev["split"].isin(["train", "val"])].copy()
    train_ev_all, val_ev = ev[ev["split"] == "train"], ev[ev["split"] == "val"]

    npz = np.load(CODEC_PATH)
    pids_codec = npz["problem_id"].tolist()
    codec_matrix = npz["codec"].astype(np.float32)
    pid_to_idx = {int(p): i for i, p in enumerate(pids_codec)}
    check_coverage(ev["problem_id"].unique(), pid_to_idx)

    val_batch = build_padded_batch_v2(val_ev, codec_matrix, pid_to_idx, device)
    _, t_max_val = val_batch["valid_mask"].shape
    prev_feat_val = build_prev_feat_v2(val_batch["correct"], val_batch["hint_bin"],
                                       val_batch["saw_answer"], val_batch["score_missing"])
    _, eval_rows = calib_eval_split(val_ev, irt_cfg["calib_frac"], irt_cfg["min_interactions"])
    eval_labeled = eval_rows[~eval_rows["score_missing"]]
    none_rows = eval_labeled[eval_labeled["support_code"] == 0]
    masks = {
        "pooled": _mask_from_rows(none_rows, val_batch["learners"], t_max_val, device),
        "train_items": _mask_from_rows(none_rows[none_rows["split_item"] == "train"],
                                       val_batch["learners"], t_max_val, device),
        "heldout_items": _mask_from_rows(none_rows[none_rows["split_item"] == "heldout"],
                                         val_batch["learners"], t_max_val, device),
    }
    logger.heartbeat(step=0, force=True, stage="val strata built",
                     n_pooled=int(masks["pooled"].sum()),
                     n_train_items=int(masks["train_items"].sum()),
                     n_heldout_items=int(masks["heldout_items"].sum()))

    results = {}
    saved_model_state = None
    saved_heads_state = None
    all_items_predictions = None  # (problem_id[], y[], p[]) on the pooled support=NONE court
    for regime in ("all_items", "clean"):
        model, heads, losses, wall = train_one_regime(
            regime, train_ev_all, codec_matrix, pid_to_idx, cfg, device, logger,
            micro_batch_learners=args.micro_batch_learners)
        model.eval()
        heads.eval()
        with torch.no_grad():
            mu, log_sigma = model(val_batch["item_codec"], prev_feat_val)
            u_eval = sample_u(mu, log_sigma, k=cfg["k_eval"])
            c_logits, _ = heads(val_batch["item_codec"], u_eval)
            probs = torch.sigmoid(c_logits).mean(dim=0)
        r = {"regime": regime, "train_wall_s": round(wall, 1),
            "loss_first_last": [round(losses[0], 4), round(losses[-1], 4)]}
        for name, m in masks.items():
            y, p = _yp(m, val_batch["correct"], probs)
            r[name] = {"yp": (y, p), **_score_yp(y, p)}
        results[regime] = r

        if regime == "all_items":
            saved_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            saved_heads_state = {k: v.clone() for k, v in heads.state_dict().items()}
            m = masks["pooled"].cpu()
            pooled_learner_idx, pooled_t = torch.where(m)
            pids_row = none_rows.set_index(["learner", "t"])
            learners_arr = np.array(val_batch["learners"])
            row_learners = learners_arr[pooled_learner_idx.numpy()]
            row_t = pooled_t.numpy()
            row_pid = pids_row.loc[list(zip(row_learners, row_t))]["problem_id"].to_numpy()
            all_items_predictions = {
                "problem_id": row_pid, "y": results["all_items"]["pooled"]["yp"][0],
                "p": results["all_items"]["pooled"]["yp"][1],
            }

    A = results["all_items"]["train_items"]["auc"]
    B = results["all_items"]["heldout_items"]["auc"]
    C = results["clean"]["train_items"]["auc"]
    D = results["clean"]["heldout_items"]["auc"]
    did = (B - D) - (A - C)

    rs = np.random.default_rng(cfg["seed"])
    yp = {(r, s): results[r][s]["yp"] for r in results for s in ("train_items", "heldout_items")}
    n_tr = len(yp[("all_items", "train_items")][0])
    n_ho = len(yp[("all_items", "heldout_items")][0])
    boot = []
    for _ in range(N_BOOT):
        i_tr, i_ho = rs.integers(0, n_tr, n_tr), rs.integers(0, n_ho, n_ho)
        a = auc(yp[("all_items", "train_items")][0][i_tr], yp[("all_items", "train_items")][1][i_tr])
        b = auc(yp[("all_items", "heldout_items")][0][i_ho], yp[("all_items", "heldout_items")][1][i_ho])
        c = auc(yp[("clean", "train_items")][0][i_tr], yp[("clean", "train_items")][1][i_tr])
        d = auc(yp[("clean", "heldout_items")][0][i_ho], yp[("clean", "heldout_items")][1][i_ho])
        boot.append((b - d) - (a - c))
    lo, hi = float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))

    # -- (ii) within-item AUC, on the ALL_ITEMS arm (the checkpointed model) -----------------
    wi = within_item_auc(all_items_predictions["problem_id"], all_items_predictions["y"],
                         all_items_predictions["p"])

    # -- (i) checkpoint, from the ALL_ITEMS arm only ----------------------------------------
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = CHECKPOINT_DIR / "fa_outcome_filter.pt"
    torch.save({
        "model_state_dict": saved_model_state, "heads_state_dict": saved_heads_state,
        "architecture": {"class": "S1FilterV2ItemMLP/FiLMHeadsItemMLP", **B4_CELL,
                         "codec_dim": CODEC_DIM, "query_independent": True},
        "codec_path": str(CODEC_PATH.relative_to(REPO)),
        "train_regime": "all items (all train-learner events; no item holdout applied)",
        "item_split_sha256": item_split_hash, "learner_split_sha256": learner_split_hash,
        "support_none_pooled_auc": results["all_items"]["pooled"]["auc"],
        "within_item_auc": wi["pooled_auc"], "epochs": cfg["epochs"], "seed": cfg["seed"],
    }, ckpt_path)

    for r in results.values():
        for s in ("pooled", "train_items", "heldout_items"):
            r[s].pop("yp")

    summary = {
        "task": "fa_01k_b4_close",
        "gate": {"item_split_sha256": item_split_hash, "learner_split_sha256": learner_split_hash,
                 "checkpoint_saved": str(ckpt_path.relative_to(REPO)),
                 "checkpoint_exists": ckpt_path.exists()},
        "cells": {"A_all_items_train_items": A, "B_all_items_heldout_items": B,
                 "C_clean_train_items": C, "D_clean_heldout_items": D},
        "memorisation_did": did, "memorisation_did_ci95": [lo, hi],
        "did_ci_excludes_zero": bool(lo > 0 or hi < 0),
        "all_items_pooled_none_auc": results["all_items"]["pooled"]["auc"],
        "within_item_auc_b4": wi,
        "person_only_within_item_auc_1a": PERSON_ONLY_WITHIN_ITEM_AUC,
        "item_only_pooled_auc_1a": ITEM_ONLY_POOLED_AUC,
        "d2_s1v2_shipped_memorisation_did": D2_S1V2_MEMORISATION_DID,
        "d1_akt_memorisation_did": D1_AKT_MEMORISATION_DID,
        "fa_01i_b4_best_reference_auc": FA_01I_B4_BEST_NONE_AUC,
        "regimes": results, "device": device, "epochs": cfg["epochs"],
    }
    out = REPO / "reports" / "fa_01k_b4_close_raw.json"
    out.write_text(json.dumps(summary, indent=2, default=float))

    print("\n" + "=" * 80)
    print("outcome filter -- checkpoint, within-item AUC, memorisation DiD")
    print("=" * 80)
    print(f"reproduction check: all_items pooled support=NONE {results['all_items']['pooled']['auc']:.4f} "
          f"vs fa_01i best cell {FA_01I_B4_BEST_NONE_AUC:.4f}")
    print(f"\n{'':<22}{'train items':>16}{'heldout items':>16}")
    print(f"{'all_items':<22}{A:>16.4f}{B:>16.4f}")
    print(f"{'clean':<22}{C:>16.4f}{D:>16.4f}")
    print(f"\nmemorisation DiD      = {did:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  "
          f"excludes zero: {summary['did_ci_excludes_zero']}")
    print(f"linear-S1v2 DiD       = {D2_S1V2_MEMORISATION_DID:+.4f}  (reference, linear item pathway, codec 45)")
    print(f"AKT DiD               = {D1_AKT_MEMORISATION_DID:+.4f}  (reference, not re-run here)")
    print(f"\nwithin-item AUC       = {wi['pooled_auc']:.4f}  (n_items={wi['n_items']}, n_pairs={wi['n_pairs']:,})")
    print(f"person_only           = {PERSON_ONLY_WITHIN_ITEM_AUC:.4f}  <- benchmark")
    print(f"item_only within-item = 0.5000  (null)")
    print(f"\ncheckpoint -> {ckpt_path}")
    print(f"item_split sha256[:16]    = {item_split_hash[:16]}  (must match the split of downstream runs)")
    print(f"learner_split sha256[:16] = {learner_split_hash[:16]}")
    print(f"\nraw -> {out}")
    logger.summary(status="ok", memorisation_did=did, within_item_auc=wi["pooled_auc"],
                   checkpoint=str(ckpt_path.relative_to(REPO)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
