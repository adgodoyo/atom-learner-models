#!/usr/bin/env python
"""Trains one arm of the token-loss-weight sweep for FA-SIM3, the mathematics simulator,
and evaluates it on three panels. The training loss is
`micro_loss = l_outcome + alpha * l_a + beta_kl * kl`; arms differ only in alpha
(0.3 reproduces FA-SIM3's own weight); data, splits, codec, seed and recipe are otherwise
identical (conditional imitation on the true class, K=1 training, ~2.5k steps). Staging,
setup and `k_marginalised_eval` are imported from fa_sim3_full_run.py.

Panels: (a) personalisation -- K=8 mixture NLL for the personalised, dw0, shuffled_u and
u_mean arms on a fixed validation subsample (the same rows for every alpha); (b)
classification -- 5-class validation log-loss and wrong-detection AUC (both definitions,
overall and by ability tercile) on the full validation split, using the instruments of
wd_1_wrong_detection_panel.py; (c) health -- log sigma at the first and last step,
class-token truncation rate and loss-component trajectories. Short runs are a
direction-finder: rankings may not persist at full training length.

Inputs: staged FA-SIM3 bundles, the FA manifest and splits, the frozen codec and the
outcome-only filter used as FA-SIM3's warm start (`fa_outcome_filter.pt`). Outputs:
results/sim_alpha_sweep/alpha_<alpha>/{summary.json, raw_scores.npz} and a filter plus
outcome-head checkpoint (the LoRA basis and hypernetwork are not kept), optionally
uploaded to the S3 bucket given with `--bucket`. Run once per alpha:
`python infra/sim_alpha_sweep.py --alpha 0.75 --bucket <bucket>`; then
sim_alpha_selection.py compares arms. Serves the token-loss weight ablation (Appendix A).
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "infra"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[1]

from atom.binding import (Hypernetwork, attach_lora_basis, reparameterize_from_log_sigma)  # noqa: E402
from atom.data.fa_manifest import load_v2_events  # noqa: E402
from atom.data.fa_outcome_label import N_OUTCOME, outcome_label  # noqa: E402
from atom.eval.evals_5g import DEFAULT_HISTORY_BINS  # noqa: E402
from atom.eval.metrics import learner_clustered_bootstrap_ci, multiclass_nll  # noqa: E402
from atom.filter.batch import build_padded_batch_v2, load_production_codec  # noqa: E402
from atom.filter.episode_v2 import build_prev_feat_v2  # noqa: E402
from atom.filter.item_encoder_mlp import S1FilterV2ItemMLP  # noqa: E402
from atom.filter.losses import gaussian_kl_free_bits, kl_anneal_weight  # noqa: E402
from atom.heads.fa_hierarchical_heads import OutcomeHead  # noqa: E402
from atom.binding.derangement import (assert_learner_matched_derangement,  # noqa: E402
                                      learner_history_matched_derangement)
from atom.codec.backbone import load_backbone  # noqa: E402
from atom.simulator.fa_alt2 import collate, encode_example  # noqa: E402
from atom.simulator.fa_train_step import (chunked_filter_forward, gather_belief_for_batch,  # noqa: E402
                                          gather_item_codec_for_batch,
                                          losses_for_hierarchical_batch)
from atom.simulator.sampler_semantics import CURRENT_SAMPLER_SEMANTICS  # noqa: E402
from atom.utils.epoch_sampler import ShuffledEpochSampler  # noqa: E402
from atom.utils.logging import default_device  # noqa: E402
from atom.utils.seeds import seed_everything  # noqa: E402

from fa_sim3_full_run import (ARMS, B4_CONFIG, CODEC_PATH, EXPECTED_CODEC_SHA,  # noqa: E402
                              _codec_sha256, _git_sha, build_examples, k_marginalised_eval,
                              load_bundle_and_manifest, log_sigma_report, sync_progress,
                              warm_start_filter)
from wd_1_wrong_detection_panel import _auc, _learner_clustered_bootstrap_auc_diff  # noqa: E402

UNAIDED_CORRECT, UNAIDED_WRONG = 0, 1
K_WD_PANEL = 32  # committee size of wd_1_wrong_detection_panel.py -- filter-only, cheap, full val split


def classification_and_wd_panel(filt, outcome_head, manifest_path, learner_split_path,
                                 item_split_path, codec_path, device, seed) -> dict:
    """Wrong-detection panel using wd_1_wrong_detection_panel.py's instruments (`_auc`,
    tercile-by-ability), unchanged, on the FULL val split -- filter + outcome head only, no
    backbone forward, so this is cheap regardless of the personalisation panel's row cap.
    Returns the summary dict AND the raw per-row arrays (for the selection script's
    cross-arm paired bootstrap)."""
    ev = load_v2_events(manifest_path, learner_split_path, item_split_path)
    ev_val = ev[ev["split"] == "val"].reset_index(drop=True)
    codec_matrix, pid_to_idx = load_production_codec(codec_path, B4_CONFIG["codec_dim"])
    val_fb = build_padded_batch_v2(ev_val, codec_matrix, pid_to_idx, device="cpu", include_time_min=True)
    val_fb["prev_feat"] = build_prev_feat_v2(val_fb["correct"], val_fb["hint_bin"],
                                             val_fb["saw_answer"], val_fb["score_missing"])
    for k in ("item_codec", "prev_feat", "time_min", "valid_mask"):
        val_fb[k] = val_fb[k].to(device)

    labels = outcome_label(ev_val["support_code"], ev_val["score"], ev_val["score_missing"])
    valid = labels["outcome_valid"].to_numpy()
    rows = ev_val.iloc[np.flatnonzero(valid)].reset_index(drop=True)
    learner_to_idx = {u: i for i, u in enumerate(val_fb["learners"])}
    l_idx = torch.tensor([learner_to_idx[u] for u in rows["learner"]], dtype=torch.long)
    t_idx = torch.tensor(rows["t"].astype("int64").to_numpy(), dtype=torch.long)
    outcome_target = labels["outcome_code"].to_numpy()[valid]
    item_codec_row_cpu = val_fb["item_codec"].cpu()[l_idx, t_idx]
    learner_np = rows["learner"].astype(str).to_numpy()

    import pandas as pd
    ability = (pd.Series(outcome_target == UNAIDED_CORRECT, index=learner_np)
              .groupby(level=0).mean())
    tercile_map = pd.qcut(ability.rank(method="first"), 3, labels=["bottom", "middle", "top"])
    row_tercile = pd.Series(learner_np).map(tercile_map).to_numpy()

    filt_forward = lambda ic, pf, tm, vm: filt(ic, pf)  # noqa: E731
    with torch.no_grad():
        mu_all, log_sigma_all = chunked_filter_forward(filt_forward, val_fb)
    mu_sel = mu_all[l_idx.to(device), t_idx.to(device)]
    log_sigma_sel = log_sigma_all[l_idx.to(device), t_idx.to(device)]
    item_codec_row = item_codec_row_cpu.to(device)

    n_rows = mu_sel.shape[0]
    gen = torch.Generator(device=device).manual_seed(seed)
    probs_mean_parts = []
    with torch.no_grad():
        for lo in range(0, n_rows, 4000):
            hi = min(lo + 4000, n_rows)
            mu_c, ls_c, ic_c = mu_sel[lo:hi], log_sigma_sel[lo:hi], item_codec_row[lo:hi]
            u_k = reparameterize_from_log_sigma(mu_c, ls_c, k=K_WD_PANEL, generator=gen)
            logits_k = torch.stack([outcome_head(ic_c, u_k[:, j, :]) for j in range(K_WD_PANEL)], dim=1)
            probs_mean_parts.append(torch.softmax(logits_k, dim=-1).mean(dim=1).cpu())
    probs_mean = torch.cat(probs_mean_parts, dim=0).numpy()

    y_uw = (outcome_target == UNAIDED_WRONG).astype(int)
    s_uw = probs_mean[:, UNAIDED_WRONG]
    y_notcorrect = (outcome_target != UNAIDED_CORRECT).astype(int)
    s_notcorrect = 1.0 - probs_mean[:, UNAIDED_CORRECT]
    log_loss_5class = multiclass_nll(probs_mean, outcome_target)

    wrong_detection = {"overall": {
        "auc_unaided_wrong": _auc(y_uw, s_uw), "auc_not_unaided_correct": _auc(y_notcorrect, s_notcorrect),
        "n": int(len(y_uw)),
    }}
    for terc in ("bottom", "middle", "top"):
        m = row_tercile == terc
        wrong_detection[terc] = {
            "auc_unaided_wrong": _auc(y_uw[m], s_uw[m]),
            "auc_not_unaided_correct": _auc(y_notcorrect[m], s_notcorrect[m]),
            "n": int(m.sum()),
            "log_loss_5class": multiclass_nll(probs_mean[m], outcome_target[m]),
        }

    summary = {"k": K_WD_PANEL, "n_val_rows": int(n_rows), "log_loss_5class_overall": log_loss_5class,
              "wrong_detection": wrong_detection}
    raw = {"y_uw": y_uw, "s_uw": s_uw, "y_notcorrect": y_notcorrect, "s_notcorrect": s_notcorrect,
          "row_tercile": row_tercile, "learner_np": learner_np, "outcome_target": outcome_target,
          "probs_mean": probs_mean}
    return summary, raw


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, required=True)
    ap.add_argument("--bucket", required=True)
    ap.add_argument("--fuse-minutes", type=float, default=120.0)
    ap.add_argument("--bundle-path", default="staging/bundles/fa_sim3_train_corrected.tar.gz")
    ap.add_argument("--val-bundle-path", default="staging/bundles/fa_sim3_val_coldstart.tar.gz")
    ap.add_argument("--manifest-path", default="artifacts/fa_manifest_v2.parquet")
    ap.add_argument("--learner-split-path", default="splits/fa_learner_split_v2.csv")
    ap.add_argument("--item-split-path", default="splits/fa_item_split_v2.csv")
    ap.add_argument("--codec-path", default=str(CODEC_PATH))
    ap.add_argument("--warm-start-checkpoint",
                    default="artifacts/checkpoints/fa_outcome_filter.pt")
    ap.add_argument("--backbone", default="gemma4-e4b")
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--micro-batch-size", type=int, default=2)
    ap.add_argument("--grad-accum-steps", type=int, default=8)
    ap.add_argument("--eval-batch-size", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=2500)
    ap.add_argument("--k-eval", type=int, default=8)
    ap.add_argument("--n-personalisation-eval-rows", type=int, default=300,
                    help="FIXED (not resampled per alpha) val subsample size for the "
                         "backbone-dependent personalisation panel -- kept modest because "
                         "each row needs backbone forwards; the classification/WD panel "
                         "runs on the FULL val split regardless (filter-only, cheap).")
    ap.add_argument("--kl-max-weight", type=float, default=0.01)
    ap.add_argument("--kl-anneal-steps", type=int, default=1000)
    ap.add_argument("--free-bits", type=float, default=0.05)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--out-prefix", default="sim-alpha-sweep")
    ap.add_argument("--seed", type=int, default=0,
                    help="Same seed across all 4 arms -- recipe, config and seed are "
                         "identical; only alpha differs.")
    ap.add_argument("--git-sha", default=None)
    args = ap.parse_args()

    codec_path = pathlib.Path(args.codec_path)
    actual_sha = _codec_sha256(codec_path)
    assert actual_sha == EXPECTED_CODEC_SHA, f"codec SHA mismatch: {actual_sha} != {EXPECTED_CODEC_SHA}"
    git_sha = args.git_sha or _git_sha()
    print(f"[sim-alpha] alpha={args.alpha} git_sha={git_sha} codec verified", flush=True)

    seed_everything(args.seed)
    device = default_device()
    print(f"[sim-alpha] device={device} cuda={torch.cuda.is_available()}", flush=True)

    subprocess.run(["sudo", "shutdown", "-c"], check=False)
    subprocess.run(["sudo", "shutdown", "-h", f"+{int(args.fuse_minutes)}"], check=False)
    print(f"[sim-alpha] boot fuse armed: {args.fuse_minutes:.0f} min", flush=True)

    import boto3
    s3 = boto3.client("s3")
    out_prefix = f"{args.out_prefix}/alpha_{args.alpha}"
    t_start = time.time()

    bundle, filter_batch = load_bundle_and_manifest(
        args.bundle_path, args.manifest_path, args.learner_split_path, args.item_split_path,
        args.codec_path, split="train")
    val_bundle, val_filter_batch = load_bundle_and_manifest(
        args.val_bundle_path, args.manifest_path, args.learner_split_path, args.item_split_path,
        args.codec_path, split="val")
    for fb in (filter_batch, val_filter_batch):
        for k in ("item_codec", "prev_feat", "time_min", "valid_mask"):
            fb[k] = fb[k].to(device)
    print(f"[sim-alpha] {len(bundle)} train events, {len(val_bundle)} val events", flush=True)

    backbone = load_backbone(args.backbone, allow_download=True)
    model, tok = backbone["model"], backbone["tokenizer"]
    bases, context = attach_lora_basis(model, m=32, r=8, seed=args.seed)
    model.enable_input_require_grads()
    model.config.use_cache = False

    filt = S1FilterV2ItemMLP(**B4_CONFIG).to(device)
    warm_start_result = warm_start_filter(filt, REPO / args.warm_start_checkpoint)
    filt.to(device)
    print(f"[sim-alpha] warm start: {warm_start_result}", flush=True)

    filt_forward = lambda ic, pf, tm, vm: filt(ic, pf)  # noqa: E731
    hyper = Hypernetwork(u_dim=B4_CONFIG["latent_dim"], m=32, kind="mlp", seed=args.seed + 1).to(device)
    outcome_head = OutcomeHead(codec_dim=B4_CONFIG["codec_dim"], latent_dim=B4_CONFIG["latent_dim"],
                               hidden=backbone["hidden_dim"]).to(device)

    basis_params = [p for b in bases.values() for p in b.parameters()]
    trainable = list(filt.parameters()) + list(hyper.parameters()) + list(outcome_head.parameters()) + basis_params
    optimizer = torch.optim.Adam(trainable, lr=args.lr)

    pad_id = tok.pad_token_id
    examples_df, examples = build_examples(bundle, tok, args.max_len)
    class_token_truncated_rate_train = float(np.mean([e["class_token_truncated"] for e in examples]))
    print(f"[sim-alpha] {len(examples_df)} train examples, class_token_truncated_rate="
         f"{class_token_truncated_rate_train:.4f}", flush=True)

    learner_to_idx = {u: i for i, u in enumerate(filter_batch["learners"])}
    learner_idx_all = torch.tensor([learner_to_idx[u] for u in examples_df["learner"]], device=device)
    t_idx_all = torch.tensor(examples_df["t"].astype("int64").to_numpy(), device=device)
    outcome_target_all = torch.tensor(examples_df["outcome_code"].to_numpy(), dtype=torch.int64, device=device)
    outcome_valid_all = torch.ones(len(examples_df), dtype=torch.bool, device=device)

    n = len(examples_df)
    epoch_sampler = ShuffledEpochSampler(n, np.random.default_rng(args.seed))
    steps = args.max_steps
    loss_history = {"total": [], "outcome": [], "production": [], "kl": []}
    log_sigma_first = None

    print(f"[sim-alpha] training: {steps} steps, alpha={args.alpha}, "
         f"micro_batch={args.micro_batch_size} x accumulate={args.grad_accum_steps}", flush=True)
    for step in range(steps):
        optimizer.zero_grad()
        accum_loss = accum_outcome = accum_prod = accum_kl = 0.0
        for micro in range(args.grad_accum_steps):
            idx = epoch_sampler.next_batch(args.micro_batch_size)
            l_idx, t_idx = learner_idx_all[idx], t_idx_all[idx]
            tok_examples = [examples[i] for i in idx]
            tok_batch = collate(tok_examples, pad_id=pad_id, device=device)
            targets = {"outcome_target": outcome_target_all[idx], "outcome_valid": outcome_valid_all[idx]}

            beta_kl = kl_anneal_weight(step, max_weight=args.kl_max_weight, anneal_epochs=args.kl_anneal_steps)
            l_outcome, l_a, _kl_raw, out = losses_for_hierarchical_batch(
                filt_forward, hyper, model, context, outcome_head, filter_batch, l_idx, t_idx,
                tok_batch, targets, beta_kl, k=1)
            mu_sel, log_sigma_sel = gather_belief_for_batch(filt_forward, filter_batch, l_idx, t_idx)
            kl = gaussian_kl_free_bits(mu_sel, log_sigma_sel, free_bits=args.free_bits).mean()
            micro_loss = l_outcome + args.alpha * l_a + beta_kl * kl
            if not torch.isfinite(micro_loss):
                print(f"[sim-alpha] TRIPWIRE non-finite loss at step={step} micro={micro}: "
                     f"{micro_loss.item()}", flush=True)
                return 1
            (micro_loss / args.grad_accum_steps).backward()
            accum_loss += micro_loss.item() / args.grad_accum_steps
            accum_outcome += l_outcome.item() / args.grad_accum_steps
            accum_prod += l_a.item() / args.grad_accum_steps
            accum_kl += kl.item() / args.grad_accum_steps
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()

        loss_history["total"].append(accum_loss)
        loss_history["outcome"].append(accum_outcome)
        loss_history["production"].append(accum_prod)
        loss_history["kl"].append(accum_kl)

        if step == 0:
            with torch.no_grad():
                _mu_all, log_sigma_all = chunked_filter_forward(filt_forward, filter_batch)
                log_sigma_first = log_sigma_report(log_sigma_all, filter_batch["valid_mask"])
        if step % 250 == 0 or step == steps - 1:
            print(f"[hb] step={step}/{steps} loss={accum_loss:.4f} outcome={accum_outcome:.4f} "
                 f"prod={accum_prod:.4f} kl={accum_kl:.4f} elapsed={time.time()-t_start:.0f}s", flush=True)
            sync_progress(s3, args.bucket, out_prefix, {
                "stage": "training", "step": step, "steps": steps, "alpha": args.alpha,
                "loss": loss_history["total"][-5:], "elapsed_s": time.time() - t_start})

    model.eval()
    filt.eval()
    with torch.no_grad():
        _mu_all, log_sigma_all = chunked_filter_forward(filt_forward, filter_batch)
        log_sigma_last = log_sigma_report(log_sigma_all, filter_batch["valid_mask"])
    print(f"[sim-alpha] log_sigma FIRST: {log_sigma_first}", flush=True)
    print(f"[sim-alpha] log_sigma LAST:  {log_sigma_last}", flush=True)

    # -- (a) personalisation panel: FIXED val subsample, 4 arms --------------------------------
    n_eval = min(args.n_personalisation_eval_rows, len(val_bundle))
    val_learner_to_idx = {u: i for i, u in enumerate(val_filter_batch["learners"])}
    eval_bundle = val_bundle.iloc[:n_eval].reset_index(drop=True)
    labels = outcome_label(eval_bundle["support_code"], eval_bundle["score"], eval_bundle["score_missing"])
    valid = labels["outcome_valid"].to_numpy()
    eval_bundle = eval_bundle.iloc[np.flatnonzero(valid)].reset_index(drop=True)
    true_class = torch.tensor(labels["outcome_code"].to_numpy()[valid], dtype=torch.int64)
    print(f"[sim-alpha] personalisation eval population: {len(eval_bundle)} rows "
         f"(of {n_eval} requested)", flush=True)

    eval_examples_by_class = []
    class_token_truncated_flags = None
    for c in range(N_OUTCOME):
        exs = [encode_example(r["question_text"], r["answer_text"], tok, args.max_len, outcome_code=c)
              for _, r in eval_bundle.iterrows()]
        eval_examples_by_class.append(exs)
        if c == 0:
            class_token_truncated_flags = np.array([e["class_token_truncated"] for e in exs])
    eval_l_idx = torch.tensor([val_learner_to_idx[u] for u in eval_bundle["learner"]], device=device)
    eval_t_idx = torch.tensor(eval_bundle["t"].astype("int64").to_numpy(), device=device)
    tok_batch_by_class = {c: eval_examples_by_class[c] for c in range(N_OUTCOME)}
    tok_batch_by_class["pad_id"] = pad_id
    tok_batch_by_class["true_class"] = true_class
    tok_batch_by_class["outcome_head"] = outcome_head

    def _content_token_count(ex: dict) -> int:
        return sum(1 for t in ex["labels"] if t != -100)

    token_counts_by_class = np.array([
        [_content_token_count(tok_batch_by_class[c][i]) for c in range(N_OUTCOME)]
        for i in range(len(eval_bundle))
    ])
    budget_consistent = (token_counts_by_class == token_counts_by_class[:, :1]).all(axis=1)
    n_budget_mismatch = int((~budget_consistent).sum())
    if n_budget_mismatch:
        keep = budget_consistent
        kept_positions = np.flatnonzero(keep).tolist()
        eval_bundle = eval_bundle.iloc[kept_positions].reset_index(drop=True)
        true_class = true_class[torch.from_numpy(keep)]
        class_token_truncated_flags = class_token_truncated_flags[keep]
        eval_l_idx = eval_l_idx[torch.from_numpy(keep)]
        eval_t_idx = eval_t_idx[torch.from_numpy(keep)]
        for c in range(N_OUTCOME):
            tok_batch_by_class[c] = [tok_batch_by_class[c][i] for i in kept_positions]
        tok_batch_by_class["true_class"] = true_class

    t_values_eval = eval_bundle["t"].to_numpy()
    history_bin_ids = np.digitize(t_values_eval, DEFAULT_HISTORY_BINS[1:-1])
    derangement_rng = np.random.default_rng(args.seed + 100)
    learner_ids_np = eval_bundle["learner"].to_numpy()
    derangement_np, deranged_mask = learner_history_matched_derangement(
        learner_ids_np, history_bin_ids, rng=derangement_rng)
    n_underanged = int((~deranged_mask).sum())
    if n_underanged:
        keep = deranged_mask
        kept_positions = np.flatnonzero(keep).tolist()
        eval_bundle = eval_bundle.iloc[kept_positions].reset_index(drop=True)
        true_class = true_class[torch.from_numpy(keep)]
        eval_l_idx = eval_l_idx[torch.from_numpy(keep)]
        eval_t_idx = eval_t_idx[torch.from_numpy(keep)]
        for c in range(N_OUTCOME):
            tok_batch_by_class[c] = [tok_batch_by_class[c][i] for i in kept_positions]
        tok_batch_by_class["true_class"] = true_class
        deranged_mask = np.ones(len(eval_bundle), dtype=bool)
        derangement_np, _ = learner_history_matched_derangement(
            eval_bundle["learner"].to_numpy(), history_bin_ids[keep], rng=derangement_rng)
    assert_learner_matched_derangement(derangement_np, eval_bundle["learner"].to_numpy(), deranged_mask)
    derangement = torch.from_numpy(derangement_np)

    n_eval_rows = len(eval_bundle)
    crn_generator = torch.Generator(device="cpu").manual_seed(args.seed + 200)
    shared_eps = torch.randn(n_eval_rows, args.k_eval, B4_CONFIG["latent_dim"], generator=crn_generator)
    arm_results = {}
    for arm in ARMS:
        arm_results[arm] = k_marginalised_eval(
            filt_forward, hyper, model, context, val_filter_batch, eval_l_idx, eval_t_idx,
            tok_batch_by_class, k=args.k_eval, batch_size=args.eval_batch_size, arm=arm,
            generator=None, derangement=derangement, eps=shared_eps)
        print(f"[sim-alpha] {arm}: mixture_nll_mean={arm_results[arm]['mixture_nll'].mean():.4f} "
             f"elapsed={time.time()-t_start:.0f}s", flush=True)
        sync_progress(s3, args.bucket, out_prefix, {"stage": f"eval_{arm}_done", "elapsed_s": time.time() - t_start})

    learner_ids_eval = eval_bundle["learner"].to_numpy()
    personalisation_panel = {
        "n_rows": n_eval_rows, "n_budget_mismatch_dropped": n_budget_mismatch,
        "n_underanged_dropped": n_underanged,
        **{f"{arm}_mixture_nll_mean": float(arm_results[arm]["mixture_nll"].mean()) for arm in ARMS},
        "paired_ci_personalised_minus_shuffled_u": learner_clustered_bootstrap_ci(
            arm_results["personalised"]["mixture_nll"], arm_results["shuffled_u"]["mixture_nll"],
            learner_ids_eval, n_boot=args.n_boot, seed=args.seed),
        "paired_ci_personalised_minus_dw0": learner_clustered_bootstrap_ci(
            arm_results["personalised"]["mixture_nll"], arm_results["dw0"]["mixture_nll"],
            learner_ids_eval, n_boot=args.n_boot, seed=args.seed),
    }
    print(f"[sim-alpha] personalisation panel: {personalisation_panel}", flush=True)

    # -- (b) classification + WD panel: full val split, filter-only ----------------------------
    print("[sim-alpha] classification/WD panel on the FULL val split...", flush=True)
    wd_summary, wd_raw = classification_and_wd_panel(
        filt, outcome_head, REPO / args.manifest_path, REPO / args.learner_split_path,
        REPO / args.item_split_path, args.codec_path, device, args.seed)
    print(f"[sim-alpha] classification/WD panel: log_loss={wd_summary['log_loss_5class_overall']:.4f} "
         f"AUC(UW,bottom)={wd_summary['wrong_detection']['bottom']['auc_unaided_wrong']:.4f}", flush=True)

    health = {
        "log_sigma_first": log_sigma_first, "log_sigma_last": log_sigma_last,
        "class_token_truncated_rate_train": class_token_truncated_rate_train,
        "class_token_truncated_rate_eval": float(class_token_truncated_flags.mean()),
        "loss_trajectory_every_250_steps": {
            str(s): {"total": loss_history["total"][s], "outcome": loss_history["outcome"][s],
                    "production": loss_history["production"][s], "kl": loss_history["kl"][s]}
            for s in range(0, steps, 250)
        },
        "loss_first": {k: v[0] for k, v in loss_history.items()},
        "loss_last": {k: v[-1] for k, v in loss_history.items()},
    }

    summary = {
        "task": "SIM-alpha round 1 (sweep arm)", "alpha": args.alpha, "steps": steps,
        "seed": args.seed, "git_sha": git_sha, "warm_start": warm_start_result,
        "personalisation_panel": personalisation_panel, "classification_and_wd_panel": wd_summary,
        "health": health, "elapsed_s": time.time() - t_start,
    }
    out_dir = REPO / "results" / "sim_alpha_sweep" / f"alpha_{args.alpha}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    np.savez(out_dir / "raw_scores.npz", **wd_raw,
            personalised_mixture_nll=arm_results["personalised"]["mixture_nll"],
            dw0_mixture_nll=arm_results["dw0"]["mixture_nll"],
            shuffled_u_mixture_nll=arm_results["shuffled_u"]["mixture_nll"],
            u_mean_mixture_nll=arm_results["u_mean"]["mixture_nll"],
            personalisation_learner_ids=learner_ids_eval)
    print(f"[sim-alpha] wrote {out_dir}/summary.json + raw_scores.npz", flush=True)

    ckpt = {
        "filter": filt.state_dict(), "outcome_head": outcome_head.state_dict(),
        "architecture": dict(B4_CONFIG), "codec_path": str(args.codec_path),
        "codec_sha256": EXPECTED_CODEC_SHA, "git_sha": git_sha, "seed": args.seed,
        "sampler_semantics": CURRENT_SAMPLER_SEMANTICS, "alpha": args.alpha, "steps": steps,
        "warm_start": warm_start_result,
    }
    ckpt_path = pathlib.Path(f"/tmp/sim_alpha_{args.alpha}_final.pt")
    torch.save(ckpt, ckpt_path)
    s3.upload_file(str(out_dir / "summary.json"), args.bucket, f"{out_prefix}/results/summary.json")
    s3.upload_file(str(out_dir / "raw_scores.npz"), args.bucket, f"{out_prefix}/results/raw_scores.npz")
    s3.upload_file(str(ckpt_path), args.bucket, f"{out_prefix}/checkpoints/final.pt")
    print(f"[sim-alpha] alpha={args.alpha} DONE, elapsed={time.time()-t_start:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
