#!/usr/bin/env python
"""Item-encoder ablation: does a nonlinear item pathway raise filter AUC, and does the
latent:hidden ratio matter once it does?

A probe on the codec showed that a nonlinear readout predicts item difficulty much better than
a linear one, with dim=64 the best nonlinear cell; this script trains the real filters and
measures AUC. Grid (2x2, all at an experimental codec_dim=64):
  item_encoder in {linear, mlp}: `S1FilterV2`/`FiLMHeads` versus `S1FilterV2ItemMLP`/
    `FiLMHeadsItemMLP` (src/atom/filter/item_encoder_mlp.py; same filter core, only the item
    pathway swapped);
  latent_dim/hidden in {128/256, 256/256}: 256/256 decouples the 1:2 ratio that the width
    sweep (`fa_01g_s1v2_width_sweep.py`) held fixed.
Caveat, not controlled for: `free_bits` is per dimension, so latent=256 carries double the
total KL floor of latent=128; the 256/256 cells also raise the KL budget.

Codec: re-derived from the cached raw states at rank 69 (k_deflate=5, output_dim=64),
whiten=False as in production, or loaded from `artifacts/fa_item_codec_dim64.npz` when present.
This is an experimental configuration, not the frozen 45-dim production codec
(`artifacts/problem_codec_mean_L15.npz` is untouched). Training reuses the exact
gradient-accumulation micro-batching of `fa_01g_s1v2_width_sweep.py`. Scored on the
`calib_eval_split` court (support=NONE); the nearest width-sweep cell (codec_dim=45, AUC
0.6783) is printed as a reference only.

Inputs: the v2 FA manifest and splits, `configs/{s1_filter,irt,data}.yaml`, and the codec
source above. Output: `reports/fa_01i_item_encoder_ablation[_<cell>]_raw.json`, optionally
uploaded to the S3 bucket given with `--s3-bucket`.
Run: `python scripts/fa_01i_item_encoder_ablation.py [--only CELL] [--micro-batch-learners N]`.
Serves Section 6.2 and Appendix A (item encoder).
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from torch import nn  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[1]

from atom.codec.extract import RawStateCache  # noqa: E402
from atom.codec.transform import CodecTransform  # noqa: E402
from atom.data.fa_manifest import load_v2_events  # noqa: E402
from atom.eval.metrics import accuracy, auc, brier, ece  # noqa: E402
from atom.eval.protocol import calib_eval_split  # noqa: E402
from atom.filter.batch import build_padded_batch_v2, check_coverage  # noqa: E402
from atom.filter.episode_v2 import build_prev_feat_v2  # noqa: E402
from atom.filter.gru import sample_u  # noqa: E402
from atom.filter.item_encoder_mlp import (  # noqa: E402
    FiLMHeadsItemMLP,
    S1FilterV2ItemMLP,
    canary_item_mlp_differs_from_linear,
)
from atom.filter.losses import gaussian_kl_free_bits, kl_anneal_weight, masked_bce  # noqa: E402
from atom.filter.s1v2 import S1FilterV2  # noqa: E402
from atom.heads.s1_heads import FiLMHeads  # noqa: E402
from atom.utils.logging import RunLogger, default_device  # noqa: E402
from atom.utils.seeds import seed_everything  # noqa: E402

CODEC_OUTPUT_DIM = 64
K_DEFLATE = 5
POOLING, LAYER = "mean", 15
CACHE_DIR = REPO / "cache" / "fa_problems"

B1_WIDE_128_256_256_AUC_AT_DIM45 = 0.6783  # width sweep (fa_01g); reference only, different codec_dim
FA_01B_AKT_NO_TAGS_NONE_AUC = 0.6922

GRID = [
    {"name": "linear_128_256", "item_encoder": "linear", "latent_dim": 128, "filter_hidden": 256, "episode_hidden": 256},
    {"name": "mlp_128_256", "item_encoder": "mlp", "latent_dim": 128, "filter_hidden": 256, "episode_hidden": 256},
    {"name": "linear_256_256", "item_encoder": "linear", "latent_dim": 256, "filter_hidden": 256, "episode_hidden": 256},
    {"name": "mlp_256_256", "item_encoder": "mlp", "latent_dim": 256, "filter_hidden": 256, "episode_hidden": 256},
]


PRECOMPUTED_CODEC_PATH = REPO / "artifacts" / "fa_item_codec_dim64.npz"


def build_experimental_codec(codec_dim: int) -> tuple[np.ndarray, dict[int, int]]:
    """Codec at rank = K_DEFLATE + codec_dim. Returns the SAME (codec_matrix, pid_to_idx)
    shape `load_production_codec` returns, so it is a drop-in for `build_padded_batch_v2`,
    but it is an experimental codec, not the frozen production artifact.

    Uses the precomputed file (`PRECOMPUTED_CODEC_PATH`, about 900KB) when present, instead of
    rebuilding from the raw-state cache (about 700MB). Falls back to the full rebuild (needs
    CACHE_DIR) when no precomputed file exists, e.g. when regenerating it.
    """
    if codec_dim == CODEC_OUTPUT_DIM and PRECOMPUTED_CODEC_PATH.exists():
        npz = np.load(PRECOMPUTED_CODEC_PATH)
        pids = npz["problem_id"].tolist()
        return npz["codec"].astype(np.float32), {int(p): i for i, p in enumerate(pids)}

    meta = json.loads((CACHE_DIR / "meta.json").read_text())
    cache = RawStateCache(CACHE_DIR, meta["n_layers"], meta["hidden_dim"],
                          capacity=meta["completed"], model=meta.get("model", ""),
                          template=meta.get("template", ""))
    X = np.asarray(cache.states(POOLING)[:, LAYER, :], dtype=np.float64)
    ct = CodecTransform.fit(X, k_max=K_DEFLATE + codec_dim, layer=LAYER, pooling=POOLING,
                            source="fa_01i_item_encoder_ablation")
    Z = ct.apply(X, k=K_DEFLATE, whiten=False).astype(np.float32)
    pid_row = {pid: cache.row_of_hash[h] for h, pids in cache.ids_of_hash.items()
              for pid in map(int, pids)}
    pids = sorted(pid_row)
    codec_matrix = np.stack([Z[pid_row[p]] for p in pids]).astype(np.float32)
    pid_to_idx = {p: i for i, p in enumerate(pids)}
    return codec_matrix, pid_to_idx


def _score(mask, correct, probs) -> dict:
    m = mask.cpu()
    y = correct.cpu()[m].numpy()
    p = probs.cpu()[m].numpy()
    return {"auc": auc(y, p), "accuracy": accuracy(y, p), "brier": brier(y, p),
            "ece": ece(y, p), "n": int(m.sum().item())}


def _mask_from_rows(rows, learners, t_max, device):
    learner_of = {u: i for i, u in enumerate(learners)}
    mask = np.zeros((len(learners), t_max), dtype=bool)
    idx = rows["learner"].map(learner_of).to_numpy()
    t = rows["t"].to_numpy().astype(int)
    mask[idx, t] = True
    return torch.tensor(mask, dtype=torch.bool, device=device)


def build_model_heads(cell: dict, codec_dim: int, device: str):
    """The one thing that varies across the grid: which class pair gets built. Everything
    downstream (`train_one_cell`) is identical regardless of which pair this returns."""
    if cell["item_encoder"] == "linear":
        model = S1FilterV2(codec_dim, cell["episode_hidden"], cell["filter_hidden"],
                           cell["latent_dim"], query_independent=True).to(device)
        heads = FiLMHeads(codec_dim, cell["latent_dim"], cell["filter_hidden"]).to(device)
    elif cell["item_encoder"] == "mlp":
        model = S1FilterV2ItemMLP(codec_dim, cell["episode_hidden"], cell["filter_hidden"],
                                  cell["latent_dim"], query_independent=True).to(device)
        heads = FiLMHeadsItemMLP(codec_dim, cell["latent_dim"], cell["filter_hidden"]).to(device)
    else:
        raise ValueError(cell["item_encoder"])
    return model, heads


def train_one_cell(cell: dict, train_batch, cfg: dict, device: str, logger: RunLogger,
                   micro_batch_learners: int | None = None):
    """Gradient-accumulation micro-batching, copied from `train_one_width` in
    `fa_01g_s1v2_width_sweep.py` (exact; see that function's docstring for the
    argument). Only change here: model/heads construction
    is delegated to `build_model_heads` so the same loop trains either item-pathway variant.
    """
    prev_feat_tr = build_prev_feat_v2(train_batch["correct"], train_batch["hint_bin"],
                                      train_batch["saw_answer"], train_batch["score_missing"])
    torch.manual_seed(cfg["seed"])
    model, heads = build_model_heads(cell, cfg["codec_dim"], device)
    n_params = sum(p.numel() for p in model.parameters()) + sum(p.numel() for p in heads.parameters())
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
            hint_per = bce(e_logits[..., 0], hint_k[:, lo:hi])
            hint_num = (hint_per * valid_k[:, lo:hi].float()).sum()
            saw_per = bce(e_logits[..., 1], saw_k[:, lo:hi])
            saw_num = (saw_per * valid_k[:, lo:hi].float()).sum()
            kl = gaussian_kl_free_bits(mu, log_sigma, cfg["free_bits"])
            kl_num = (kl * vmf[lo:hi]).sum()

            mb_loss = (c_num / label_den
                      + cfg["engagement_weight"] * (hint_num / valid_den + saw_num / valid_den)
                      + kl_w * (kl_num / kl_den))
            logger.tripwire_finite(f"{cell['name']}_train_loss", mb_loss)
            mb_loss.backward()
            epoch_c_num += c_num.item()

        opt.step()
        c_loss_epoch = epoch_c_num / label_den.item()
        losses.append(c_loss_epoch)
        logger.heartbeat(step=epoch + 1, total=cfg["epochs"], stage=cell["name"],
                         correctness=round(c_loss_epoch, 4),
                         micro_batches=(n_learners + mb - 1) // mb)
    return model, heads, losses, time.time() - t0, n_params


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="run a single cell by name; default = full grid")
    ap.add_argument("--s3-bucket", default=None)
    ap.add_argument("--micro-batch-learners", type=int, default=None)
    args = ap.parse_args()
    grid = GRID if args.only is None else [c for c in GRID if c["name"] == args.only]
    if not grid:
        print(f"[fa-01i] unknown --only cell: {args.only}", file=sys.stderr)
        return 1

    canary_item_mlp_differs_from_linear()  # build-time check: the MLP path is genuinely nonlinear

    cfg_full = yaml.safe_load((REPO / "configs" / "s1_filter.yaml").read_text())
    irt_cfg = yaml.safe_load((REPO / "configs" / "irt.yaml").read_text())
    v2cfg = yaml.safe_load((REPO / "configs" / "data.yaml").read_text())["fa"]["manifest_v2"]
    cfg = {**cfg_full, "epochs": cfg_full["full_epochs"], "codec_dim": CODEC_OUTPUT_DIM}
    seed_everything(cfg["seed"])
    device = default_device()
    logger = RunLogger(config={"task": "fa_01i_item_encoder_ablation", **cfg},
                       data_version="FA-v2-2026-08-09", seed=cfg["seed"], heartbeat_every_s=20.0)

    logger.heartbeat(step=0, force=True, stage="loading manifest + splits", device=device)
    ev = load_v2_events(REPO / v2cfg["manifest_out"], REPO / v2cfg["learner_split_out"],
                        REPO / v2cfg["item_split_out"])
    ev = ev[ev["split"].isin(["train", "val"])].copy()
    train_ev, val_ev = ev[ev["split"] == "train"], ev[ev["split"] == "val"]

    logger.heartbeat(step=0, force=True, stage=f"re-deriving codec_dim={CODEC_OUTPUT_DIM} "
                     "(experimental, not the frozen production artifact) from the raw-state cache")
    codec_matrix, pid_to_idx = build_experimental_codec(CODEC_OUTPUT_DIM)
    check_coverage(ev["problem_id"].unique(), pid_to_idx)

    train_batch = build_padded_batch_v2(train_ev, codec_matrix, pid_to_idx, device)
    val_batch = build_padded_batch_v2(val_ev, codec_matrix, pid_to_idx, device)
    _, t_max_val = val_batch["valid_mask"].shape
    prev_feat_val = build_prev_feat_v2(val_batch["correct"], val_batch["hint_bin"],
                                       val_batch["saw_answer"], val_batch["score_missing"])

    _, eval_rows = calib_eval_split(val_ev, irt_cfg["calib_frac"], irt_cfg["min_interactions"])
    eval_labeled = eval_rows[~eval_rows["score_missing"]]
    none_rows = eval_labeled[eval_labeled["support_code"] == 0]
    eval_mask = _mask_from_rows(eval_labeled, val_batch["learners"], t_max_val, device)
    none_mask = _mask_from_rows(none_rows, val_batch["learners"], t_max_val, device)
    logger.heartbeat(step=0, force=True, stage="val court built",
                     n_marginal=int(eval_mask.sum()), n_none=int(none_mask.sum()))

    results = {}
    for cell in grid:
        model, heads, losses, wall, n_params = train_one_cell(
            cell, train_batch, cfg, device, logger,
            micro_batch_learners=args.micro_batch_learners)
        model.eval()
        heads.eval()
        with torch.no_grad():
            mu, log_sigma = model(val_batch["item_codec"], prev_feat_val)
            u_eval = sample_u(mu, log_sigma, k=cfg["k_eval"])
            c_logits, _ = heads(val_batch["item_codec"], u_eval)
            probs = torch.sigmoid(c_logits).mean(dim=0)
        marginal = _score(eval_mask, val_batch["correct"], probs)
        none_stratum = _score(none_mask, val_batch["correct"], probs)
        results[cell["name"]] = {**cell, "n_params": n_params, "train_wall_s": round(wall, 1),
                                 "loss_first_last": [round(losses[0], 4), round(losses[-1], 4)],
                                 "marginal": marginal, "support_none": none_stratum}
        logger.heartbeat(step=0, force=True, stage=f"{cell['name']} done",
                         none_auc=round(none_stratum["auc"], 4), n_params=n_params,
                         wall_s=round(wall, 1))

    linear_ref = (results["linear_128_256"]["support_none"]["auc"]
                 if "linear_128_256" in results else None)
    for name, r in results.items():
        r["delta_vs_linear_128_256"] = (
            r["support_none"]["auc"] - linear_ref if linear_ref is not None else None)

    summary = {"task": "fa_01i_item_encoder_ablation", "grid": results,
              "codec_dim": CODEC_OUTPUT_DIM,
              "codec_note": "experimental rank, not the frozen production codec artifact (45 dims)",
              "b1_wide_128_256_256_reference_dim45": B1_WIDE_128_256_256_AUC_AT_DIM45,
              "fa_01b_akt_no_tags_reference": FA_01B_AKT_NO_TAGS_NONE_AUC,
              "device": device, "epochs": cfg["epochs"]}
    out_name = ("fa_01i_item_encoder_ablation_raw.json" if args.only is None
               else f"fa_01i_item_encoder_ablation_{args.only}_raw.json")
    out = REPO / "reports" / out_name
    out.write_text(json.dumps(summary, indent=2, default=float))
    if args.s3_bucket:
        import boto3
        boto3.client("s3").upload_file(str(out), args.s3_bucket,
                                       f"fa-01i-item-encoder/results/{out_name}")
        print(f"[fa-01i] uploaded to bucket {args.s3_bucket}, key fa-01i-item-encoder/results/{out_name}")

    print("\n" + "=" * 86)
    print("item-encoder x latent:hidden ratio ablation, support=NONE AUC, codec_dim=64")
    print("=" * 86)
    print(f"{'cell':<18}{'encoder':>9}{'latent':>8}{'hidden':>8}{'n_params':>12}{'AUC':>10}"
          f"{'delta':>10}{'wall_s':>10}")
    for name, r in results.items():
        d = r["delta_vs_linear_128_256"]
        d_str = f"{d:+.4f}" if d is not None else "--"
        print(f"{name:<18}{r['item_encoder']:>9}{r['latent_dim']:>8}{r['filter_hidden']:>8}"
              f"{r['n_params']:>12,}{r['support_none']['auc']:>10.4f}{d_str:>10}"
              f"{r['train_wall_s']:>10.1f}")
    print(f"\nreference (different codec_dim=45): width-sweep wide_128_256_256 = "
          f"{B1_WIDE_128_256_256_AUC_AT_DIM45:.4f}")
    print(f"AKT-no-tags reference (different court, context only): {FA_01B_AKT_NO_TAGS_NONE_AUC:.4f}")
    print(f"\nraw -> {out}")
    widest_name = max(results, key=lambda n: results[n]["n_params"])
    logger.summary(status="ok",
                   best_auc=max(r["support_none"]["auc"] for r in results.values()),
                   widest_auc=results[widest_name]["support_none"]["auc"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
