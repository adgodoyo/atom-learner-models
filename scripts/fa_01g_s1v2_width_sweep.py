#!/usr/bin/env python
"""Belief-width sweep: does widening the S1v2 filter (`S1FilterV2`) close the gap to AKT?

A direct test of capacity, changing nothing but `latent_dim`, `filter_hidden` and
`episode_hidden` (default 32/64/64 in `configs/s1_filter.yaml`, against AKT's `d_model=256`).
Grid: 32/64/64, 64/128/128, 128/256/256 and 256/512/512, with the same query-independent
architecture, seed, data and protocol as `fa_01_s1v2.py`; the 32/64/64 cell doubles as a
reproduction check. Each cell is scored on the `calib_eval_split` court (marginal and
support=NONE stratum). Claims are about the widths relative to each other; the AKT-no-tags
reference AUC was measured on a different court and is printed for context only.

Inputs: the v2 FA manifest and splits, `artifacts/problem_codec_mean_L15.npz`, and
`configs/{s1_filter,irt,data}.yaml`. Output: `reports/fa_01g_s1v2_width_sweep[_<cell>]_raw.json`,
optionally uploaded to the S3 bucket given with `--s3-bucket`.
Run: `python scripts/fa_01g_s1v2_width_sweep.py [--only CELL] [--micro-batch-learners N]`.
Serves Appendix A (belief width).
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

from atom.data.fa_manifest import load_v2_events  # noqa: E402
from atom.eval.metrics import accuracy, auc, brier, ece  # noqa: E402
from atom.eval.protocol import calib_eval_split  # noqa: E402
from atom.filter.batch import build_padded_batch_v2, check_coverage, load_production_codec  # noqa: E402
from atom.filter.episode_v2 import build_prev_feat_v2  # noqa: E402
from atom.filter.gru import sample_u  # noqa: E402
from atom.filter.losses import gaussian_kl_free_bits, kl_anneal_weight, masked_bce  # noqa: E402
from atom.filter.s1v2 import S1FilterV2  # noqa: E402
from atom.heads.s1_heads import FiLMHeads  # noqa: E402
from atom.utils.logging import RunLogger, default_device  # noqa: E402
from atom.utils.seeds import seed_everything  # noqa: E402

FA_01_S1V2_NONE_AUC = 0.6520          # fa_01_s1v2 reference, support=NONE
D2_ALL_ITEMS_POOLED_NONE_AUC = 0.6528  # all-items arm of the item-holdout run; reproduction check
FA_01B_AKT_NO_TAGS_NONE_AUC = 0.6922
GRID = [
    {"name": "baseline_32_64_64", "latent_dim": 32, "filter_hidden": 64, "episode_hidden": 64},
    {"name": "wide_64_128_128", "latent_dim": 64, "filter_hidden": 128, "episode_hidden": 128},
    {"name": "wide_128_256_256", "latent_dim": 128, "filter_hidden": 256, "episode_hidden": 256},
    # Fourth point, beyond AKT's own d_model=256: checks whether the curve flattens toward a
    # plateau below AKT-no-tags' 0.6922 or still has room to climb.
    {"name": "wide_256_512_512", "latent_dim": 256, "filter_hidden": 512, "episode_hidden": 512},
]


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


def train_one_width(cell: dict, train_batch, cfg: dict, device: str, logger: RunLogger,
                    micro_batch_learners: int | None = None):
    """Train one width cell; return (model, heads, per-epoch losses, wall seconds, n_params).

    `micro_batch_learners=None` (default): a single full-batch forward/backward per epoch.

    When set (needed for the widest cells, which run out of memory on a 44GB GPU at
    latent=256/hidden=512 because S1FilterV2 keeps every timestep's activation live for
    backprop-through-time across the full learner batch): splits the LEARNER axis (dim 0 of item_codec/prev_feat, dim 1 of the k-expanded
    label/valid/correct/hint/saw tensors) into chunks, accumulating gradients via multiple
    backward() calls before ONE optimizer.step() per epoch.

    MATHEMATICALLY EXACT, not an approximation: every loss term here (`masked_bce`, the KL
    term) is (sum over masked elements) / (total valid count) -- a MEAN. Precomputing the
    FULL-batch denominators ONCE (unaffected by trainable params, so they are true epoch
    constants) and dividing each minibatch's UNNORMALISED sum by that same fixed denominator
    before calling backward() reproduces the identical total loss and identical total
    gradient as one full-batch backward() would -- verified in
    `scripts/fa_01g_verify_microbatch_equivalence.py` (loss and every parameter's .grad match
    the full-batch path to float32 precision on a real, smaller cell). No BatchNorm or other
    cross-example-coupled layer exists anywhere in S1FilterV2/FiLMHeads (checked directly) --
    the one thing that would break this equivalence is absent by construction. The realised
    stochastic samples (`sample_u`'s reparameterisation noise) will differ numerically from a
    full-batch run at the same seed, since the RNG is consumed in a different call sequence --
    expected and immaterial.
    """
    prev_feat_tr = build_prev_feat_v2(train_batch["correct"], train_batch["hint_bin"],
                                      train_batch["saw_answer"], train_batch["score_missing"])
    torch.manual_seed(cfg["seed"])
    model = S1FilterV2(cfg["codec_dim"], cell["episode_hidden"], cell["filter_hidden"],
                       cell["latent_dim"], query_independent=True).to(device)
    heads = FiLMHeads(cfg["codec_dim"], cell["latent_dim"], cell["filter_hidden"]).to(device)
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

    # Fixed FULL-batch denominators -- precomputed once, never recomputed per minibatch,
    # which is exactly what makes the accumulation below mathematically exact (see docstring).
    label_den = label_k.sum().clamp(min=1.0)
    valid_den = valid_k.sum().clamp(min=1.0)
    kl_den = vmf.sum().clamp(min=1.0)

    t0 = time.time()
    losses: list[float] = []
    mb = micro_batch_learners or n_learners  # None/unset -> one chunk = the original path
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
    ap.add_argument("--only", default=None,
                    help="run a single cell by name (e.g. wide_128_256_256); default = full grid")
    ap.add_argument("--s3-bucket", default=None,
                    help="if set, upload the raw JSON to this S3 bucket under fa-01g-width-sweep/results/")
    ap.add_argument("--micro-batch-learners", type=int, default=None,
                    help="split the learner axis into chunks of this size (gradient "
                         "accumulation, mathematically exact -- see train_one_width's "
                         "docstring); default = original single full-batch pass")
    args = ap.parse_args()
    grid = GRID if args.only is None else [c for c in GRID if c["name"] == args.only]
    if not grid:
        print(f"[fa-01g] unknown --only cell: {args.only}", file=sys.stderr)
        return 1

    cfg_full = yaml.safe_load((REPO / "configs" / "s1_filter.yaml").read_text())
    irt_cfg = yaml.safe_load((REPO / "configs" / "irt.yaml").read_text())
    v2cfg = yaml.safe_load((REPO / "configs" / "data.yaml").read_text())["fa"]["manifest_v2"]
    cfg = {**cfg_full, "epochs": cfg_full["full_epochs"]}
    seed_everything(cfg["seed"])
    device = default_device()
    logger = RunLogger(config={"task": "fa_01g_s1v2_width_sweep", **cfg},
                       data_version="FA-v2-2026-08-09", seed=cfg["seed"], heartbeat_every_s=20.0)

    logger.heartbeat(step=0, force=True, stage="loading manifest + splits + codec", device=device)
    ev = load_v2_events(REPO / v2cfg["manifest_out"], REPO / v2cfg["learner_split_out"],
                        REPO / v2cfg["item_split_out"])
    ev = ev[ev["split"].isin(["train", "val"])].copy()
    train_ev, val_ev = ev[ev["split"] == "train"], ev[ev["split"] == "val"]

    codec_matrix, pid_to_idx = load_production_codec(
        REPO / "artifacts" / "problem_codec_mean_L15.npz", cfg["codec_dim"])
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
        model, heads, losses, wall, n_params = train_one_width(
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

    # When --only skips the baseline cell, its full-grid AUC (0.6524) is the reference, so a
    # single-cell run still reports a delta without retraining the baseline.
    baseline_auc = (results["baseline_32_64_64"]["support_none"]["auc"]
                    if "baseline_32_64_64" in results else 0.6524)
    for name, r in results.items():
        r["delta_vs_baseline"] = r["support_none"]["auc"] - baseline_auc

    summary = {"task": "fa_01g_s1v2_width_sweep", "grid": results,
              "reproduction_check": {"baseline_none_auc": baseline_auc,
                                     "d2_all_items_pooled_none_auc": D2_ALL_ITEMS_POOLED_NONE_AUC,
                                     "delta": baseline_auc - D2_ALL_ITEMS_POOLED_NONE_AUC},
              "fa_01_s1v2_reference": FA_01_S1V2_NONE_AUC,
              "fa_01b_akt_no_tags_reference": FA_01B_AKT_NO_TAGS_NONE_AUC,
              "device": device, "epochs": cfg["epochs"]}
    out_name = ("fa_01g_s1v2_width_sweep_raw.json" if args.only is None
               else f"fa_01g_s1v2_width_sweep_{args.only}_raw.json")
    out = REPO / "reports" / out_name
    out.write_text(json.dumps(summary, indent=2, default=float))
    if args.s3_bucket:
        import boto3
        boto3.client("s3").upload_file(str(out), args.s3_bucket,
                                       f"fa-01g-width-sweep/results/{out_name}")
        print(f"[fa-01g] uploaded to bucket {args.s3_bucket}, key fa-01g-width-sweep/results/{out_name}")

    print("\n" + "=" * 78)
    print("S1v2 width sweep, support=NONE AUC")
    print("=" * 78)
    print(f"{'cell':<22}{'latent':>8}{'hidden':>8}{'n_params':>12}{'AUC':>10}{'delta':>10}{'wall_s':>10}")
    for name, r in results.items():
        print(f"{name:<22}{r['latent_dim']:>8}{r['filter_hidden']:>8}{r['n_params']:>12,}"
              f"{r['support_none']['auc']:>10.4f}{r['delta_vs_baseline']:>+10.4f}{r['train_wall_s']:>10.1f}")
    print(f"\nreproduction check: baseline {baseline_auc:.4f} vs the item-holdout all-items arm "
          f"{D2_ALL_ITEMS_POOLED_NONE_AUC:.4f} (delta {baseline_auc - D2_ALL_ITEMS_POOLED_NONE_AUC:+.4f})")
    print(f"AKT-no-tags reference (different court, context only): {FA_01B_AKT_NO_TAGS_NONE_AUC:.4f}")
    print(f"\nraw -> {out}")
    widest_name = max(results, key=lambda n: results[n]["n_params"])
    logger.summary(status="ok", baseline_auc=baseline_auc,
                   widest_auc=results[widest_name]["support_none"]["auc"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
