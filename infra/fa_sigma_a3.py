#!/usr/bin/env python
"""Retrains the filter and outcome head of FA-SIM3, the mathematics simulator, under an
alternative belief-width parameterisation and tests whether the belief width tracks
evidence. No backbone or raw text is used.

Recipe: `--sigma-param precision` gives sigma^-2 = 1 + cumsum(softplus(...)) (narrowing
is monotone by construction; `direct`, the default, is the original head); free bits clamp
only the KL's mu-term (`--free-bits-mu`), never the width term, so the KL cannot shrink mu
towards zero; the outcome likelihood is the K-sample mixture. Warm start is FA-SIM3's
final.pt; with the precision head the load is `strict=False` and asserts that the only
key differences are log_sigma_head.* versus precision_head.*. An unmodified frozen copy
of the original filter and head is the comparison baseline.

Gates (all must pass): 1a tr(Sigma) and 1b the K-sample Jensen gap both fall from t=10
to t=100 on a fixed learner cohort, paired learner-clustered CI clear of zero (magnitudes
reported); 2 validation outcome log-loss no worse than the original beyond
`--log-loss-tolerance`; 3 mu's between-learner variance share within 2*sigma_seed of
`--identity-reference-share` if `--identity-sigma-seed` is given (seed floor from
infra/fa_sigma_a3_seed_floor.sh), else not reduced versus the original (recorded as the
fallback rule); 5 log sigma finishes off both clamps.

Inputs: the FA manifest, splits, frozen codec and FA-SIM3 checkpoint. Outputs:
results/fa_sigma_a3/{final.pt, summary.json} and validation prefix beliefs
(artifacts/fa_v2_val_prefix_beliefs_a3.npz), optionally uploaded to the S3 bucket given
with `--bucket` (which also arms a shutdown fuse, re-armed every heartbeat). Run:
`python infra/fa_sigma_a3.py --sigma-param precision [--identity-sigma-seed S]`.
Serves the width-parameterisation ablation (Appendix A).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import subprocess
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[1]

from atom.binding import reparameterize_from_log_sigma  # noqa: E402
from atom.data.fa_manifest import load_v2_events  # noqa: E402
from atom.data.fa_outcome_label import outcome_label  # noqa: E402
from atom.eval.metrics import learner_clustered_bootstrap_ci  # noqa: E402
from atom.filter.batch import build_padded_batch_v2, load_production_codec  # noqa: E402
from atom.filter.episode_v2 import build_prev_feat_v2  # noqa: E402
from atom.filter.item_encoder_mlp import S1FilterV2ItemMLP  # noqa: E402
from atom.filter.losses import gaussian_kl_free_bits_mu_only, kl_anneal_weight  # noqa: E402
from atom.heads.fa_hierarchical_heads import OutcomeHead  # noqa: E402
from atom.simulator.fa_alt2 import outcome_kmixture_nll_per_row, outcome_loss_likelihood  # noqa: E402
from atom.simulator.fa_train_step import (chunked_filter_forward, gather_belief_for_batch,  # noqa: E402
                                          gather_item_codec_for_batch)
from atom.simulator.sampler_semantics import CURRENT_SAMPLER_SEMANTICS  # noqa: E402
from atom.utils.epoch_sampler import ShuffledEpochSampler  # noqa: E402
from atom.utils.logging import default_device, rearm_shutdown_fuse  # noqa: E402
from atom.utils.seeds import seed_everything  # noqa: E402

B4_CONFIG = {"codec_dim": 64, "episode_hidden": 256, "filter_hidden": 256, "latent_dim": 256}
CODEC_PATH = REPO / "artifacts" / "fa_item_codec_dim64.npz"
EXPECTED_CODEC_SHA = "3dc74d29072c967416351c305bae8d02952049e69052507845f98e0b444f7fe0"
FIXED_COHORT_T = (10, 100)  # the pre-registered t=10 -> t=100 gate pair


def _codec_sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def build_filter_batch_and_targets(ev_split: pd.DataFrame, codec_matrix, pid_to_idx, device: str):
    """Pads `ev_split` (already restricted to one split) into a filter batch, and returns
    the OUTCOME-VALID row set as flat (learner_idx, t_idx, outcome_target) tensors --
    everything a training/eval loop needs, matching `fa_sim3_full_run.py`'s own
    filter_batch + flat-row-index convention."""
    fb = build_padded_batch_v2(ev_split, codec_matrix, pid_to_idx, device="cpu", include_time_min=True)
    fb["prev_feat"] = build_prev_feat_v2(fb["correct"], fb["hint_bin"], fb["saw_answer"], fb["score_missing"])
    for k in ("item_codec", "prev_feat", "time_min", "valid_mask"):
        fb[k] = fb[k].to(device)
    learner_to_idx = {u: i for i, u in enumerate(fb["learners"])}

    labels = outcome_label(ev_split["support_code"], ev_split["score"], ev_split["score_missing"])
    valid = labels["outcome_valid"].to_numpy()
    rows = ev_split.iloc[np.flatnonzero(valid)].reset_index(drop=True)
    learner_idx = torch.tensor([learner_to_idx[u] for u in rows["learner"]], device=device)
    t_idx = torch.tensor(rows["t"].astype("int64").to_numpy(), device=device)
    outcome_target = torch.tensor(labels["outcome_code"].to_numpy()[valid], dtype=torch.int64, device=device)
    return fb, learner_idx, t_idx, outcome_target, rows


def persist_val_prefix_beliefs(ev_val: pd.DataFrame, val_fb: dict, mu_all: torch.Tensor,
                               log_sigma_all: torch.Tensor) -> dict:
    """Same field layout as `scripts/pl_2b_belief_curves.py::persist_beliefs`, applied to
    this filter: EVERY row with a defined `t` (not filtered to outcome-valid -- the belief
    is persisted at every prefix position, since it does not depend on whether THIS row's
    own outcome is scoreable)."""
    learners = val_fb["learners"]
    lidx = {u: i for i, u in enumerate(learners)}
    rows = ev_val.sort_values(["learner", "t"], kind="mergesort")
    n_idx = rows["learner"].map(lidx).to_numpy()
    t_idx = rows["t"].to_numpy().astype(np.int64)
    return {
        "learner": rows["learner"].to_numpy().astype(str),
        "id": rows["id"].to_numpy().astype(np.int64),
        "problem_id": rows["problem_id"].to_numpy().astype(np.int64),
        "t": t_idx,
        "score": rows["score"].to_numpy(),
        "score_missing": rows["score_missing"].to_numpy().astype(bool),
        "split_learner": rows["split_learner"].to_numpy().astype(str),
        "split_item": rows["split_item"].to_numpy().astype(str),
        "mu": mu_all[n_idx, t_idx].detach().cpu().numpy().astype(np.float32),
        "log_sigma": log_sigma_all[n_idx, t_idx].detach().cpu().numpy().astype(np.float32),
    }


def value_join_canary(beliefs: dict, manifest_path: pathlib.Path, rng: np.random.Generator,
                      n: int = 2000) -> dict:
    """By-value join canary, the same pattern as
    `scripts/pl_2b_belief_curves.py::value_join_canary` -- re-read the source
    manifest INDEPENDENTLY and assert the persisted keys agree BY VALUE, not merely that
    the join matched (counts are not identity)."""
    n = min(n, len(beliefs["id"]))
    pick = rng.choice(len(beliefs["id"]), size=n, replace=False)
    sample = pd.DataFrame({
        "learner": beliefs["learner"][pick], "id": beliefs["id"][pick],
        "t_persisted": beliefs["t"][pick], "pid_persisted": beliefs["problem_id"][pick],
        "split_persisted": beliefs["split_learner"][pick],
    })
    src = pd.read_parquet(manifest_path, columns=["learner", "id", "problem_id", "t", "split_learner"])
    src["learner"] = src["learner"].astype(str)
    j = sample.merge(src, on=["learner", "id"], how="left")
    matched = j["t"].notna()
    t_ok = (j.loc[matched, "t_persisted"].to_numpy() == j.loc[matched, "t"].to_numpy())
    pid_ok = (j.loc[matched, "pid_persisted"].to_numpy() == j.loc[matched, "problem_id"].to_numpy())
    split_ok = (j.loc[matched, "split_persisted"].to_numpy() == j.loc[matched, "split_learner"].to_numpy())
    return {"n_checked": int(n), "membership": float(matched.mean()),
           "t_by_value": float(t_ok.mean()) if matched.any() else 0.0,
           "problem_id_by_value": float(pid_ok.mean()) if matched.any() else 0.0,
           "split_by_value": float(split_ok.mean()) if matched.any() else 0.0}


def jensen_gap_categorical(outcome_head, item_codec: torch.Tensor, mu: torch.Tensor,
                           log_sigma: torch.Tensor, k: int, generator: torch.Generator | None
                           ) -> torch.Tensor:
    """`atom.planner.replay_v2.jensen_gap`'s own definition (`H(mean_k p_k) - mean_k
    H(p_k)`), generalised from `_binary_entropy` to the 5-way categorical outcome
    distribution -- the SAME epistemic-uncertainty decomposition, one entropy formula
    wider. Returns per-row `[N]`, clamped at 0 (Jensen's inequality guarantees
    non-negative in exact arithmetic; the clamp only guards floating-point noise, matching
    `replay_v2`'s own convention)."""
    with torch.no_grad():
        u_k = reparameterize_from_log_sigma(mu, log_sigma, k=k, generator=generator)
        probs_k = torch.stack(
            [torch.softmax(outcome_head(item_codec, u_k[:, j, :]), dim=-1) for j in range(k)], dim=0)
        mean_probs = probs_k.mean(dim=0)
        total_entropy = -(mean_probs.clamp(min=1e-8).log() * mean_probs).sum(dim=-1)
        per_sample_entropy = -(probs_k.clamp(min=1e-8).log() * probs_k).sum(dim=-1).mean(dim=0)
        return (total_entropy - per_sample_entropy).clamp(min=0.0)


def between_learner_mu_share(mu: np.ndarray, learners: np.ndarray) -> float:
    """Identity gate statistic -- variance-decomposition share of mu's total variance that is BETWEEN learners (their
    own mean mu differs) rather than within (their mu varies across their own rows),
    averaged over latent dims. 0 = mu carries no identity signal at all; 1 = mu is
    constant within every learner (pure identity, no within-learner evolution)."""
    df = pd.DataFrame(mu)
    df["learner"] = learners
    total_var = df.drop(columns="learner").var(ddof=0).to_numpy()
    between_var = df.groupby("learner").mean().var(ddof=0).to_numpy()
    return float(np.mean(between_var / np.clip(total_var, 1e-12, None)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint-path", default="results/fa_sim3_full_run/final.pt")
    ap.add_argument("--manifest-path", default="artifacts/fa_manifest_v2.parquet")
    ap.add_argument("--learner-split-path", default="splits/fa_learner_split_v2.csv")
    ap.add_argument("--item-split-path", default="splits/fa_item_split_v2.csv")
    ap.add_argument("--codec-path", default=str(CODEC_PATH))
    ap.add_argument("--steps", type=int, default=36000,
                    help="~1h at ~10 steps/s on an A10G")
    ap.add_argument("--micro-batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--k-train", type=int, default=8,
                    help="K_train for the kmixture likelihood; spec requires >=8")
    ap.add_argument("--k-eval", type=int, default=8)
    ap.add_argument("--kl-max-weight", type=float, default=0.01)
    ap.add_argument("--kl-anneal-steps", type=int, default=500)
    ap.add_argument("--sigma-param", choices=("direct", "precision"), default="direct",
                    help="'precision' uses the precision-accumulating "
                         "sigma^-2 = 1 + cumsum(softplus(...)) parameterisation of "
                         "atom.filter.gru.KnowledgeFilter (monotone narrowing by "
                         "construction). 'direct' (default) is the original head.")
    ap.add_argument("--free-bits-mu", type=float, default=0.05,
                    help="Floors ONLY the KL's mu-term (gaussian_kl_free_bits_mu_only) "
                         "-- the width term is never floored, whatever this value is. "
                         "0.05 is the usual free_bits default, applied where it protects "
                         "mu's identity signal rather than to the width term.")
    ap.add_argument("--log-loss-tolerance", type=float, default=0.02,
                    help="Disclosed 'seed floor' proxy for gate (2) -- no independent "
                         "seed-variance measurement exists for this stack; stated, not derived.")
    ap.add_argument("--identity-sigma-seed", type=float, default=None,
                    help="The identity gate becomes 'within 2*sigma_seed of "
                         "--identity-reference-share'. Omit to fall back to the 'not "
                         "reduced' rule (recorded as the fallback in the summary).")
    ap.add_argument("--identity-reference-share", type=float, default=0.6429,
                    help="Between-learner mu share of the reference filter (log-sigma-only retrain, free_bits=0, mu frozen).")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="results/fa_sigma_a3")
    ap.add_argument("--beliefs-out", default="artifacts/fa_v2_val_prefix_beliefs_a3.npz")
    ap.add_argument("--bucket", default="")
    ap.add_argument("--out-prefix", default="fa-sigma-a3")
    ap.add_argument("--fuse-minutes", type=float, default=60.0)
    ap.add_argument("--self-terminate", action="store_true")
    ap.add_argument("--git-sha", default=None)
    args = ap.parse_args()

    sigma_param_map = {"direct": "direct", "precision": "precision_accumulating"}
    sigma_parameterization = sigma_param_map[args.sigma_param]

    codec_path = pathlib.Path(args.codec_path)
    actual_sha = _codec_sha256(codec_path)
    assert actual_sha == EXPECTED_CODEC_SHA, f"codec SHA mismatch: {actual_sha} != {EXPECTED_CODEC_SHA}"
    git_sha = args.git_sha or _git_sha()
    print(f"[fa-sigma-a3] codec SHA verified, git_sha={git_sha}", flush=True)

    seed_everything(args.seed)
    device = default_device()
    print(f"[fa-sigma-a3] device={device} cuda_available={torch.cuda.is_available()}", flush=True)

    s3 = None
    if args.bucket:
        subprocess.run(["sudo", "shutdown", "-c"], check=False)
        subprocess.run(["sudo", "shutdown", "-h", f"+{int(args.fuse_minutes)}"], check=False)
        print(f"[fa-sigma-a3] boot fuse armed: {args.fuse_minutes:.0f} min", flush=True)
        import boto3
        s3 = boto3.client("s3")

    t_start = time.time()
    out_dir = REPO / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    def sync_to_s3() -> None:
        if s3 is None:
            return
        for p in out_dir.glob("*"):
            if p.is_file():
                s3.upload_file(str(p), args.bucket, f"{args.out_prefix}/{p.name}")

    # -- load FA-SIM3's final.pt, filter + outcome_head ONLY (never model/hyper/basis) --
    ckpt = torch.load(REPO / args.checkpoint_path, map_location="cpu", weights_only=False)
    assert ckpt["sampler_semantics"] == CURRENT_SAMPLER_SEMANTICS, (
        f"checkpoint sampler_semantics={ckpt['sampler_semantics']!r} != current "
        f"{CURRENT_SAMPLER_SEMANTICS!r} -- refusing to warm-start from a checkpoint trained under different sampler semantics")
    print(f"[fa-sigma-a3] checkpoint loaded: git_sha={ckpt.get('git_sha')} "
         f"sampler_semantics={ckpt['sampler_semantics']}", flush=True)
    outcome_hidden = ckpt["outcome_head"]["trunk.weight"].shape[0]

    filt = S1FilterV2ItemMLP(**B4_CONFIG, sigma_parameterization=sigma_parameterization).to(device)
    if sigma_parameterization == "direct":
        filt.load_state_dict(ckpt["filter"], strict=True)
    else:
        # strict=False: the checkpoint has knowledge_filter.log_sigma_head.*, the fresh
        # precision-parameterised model has knowledge_filter.precision_head.* instead --
        # CANARY, not a bare strict=False: assert the mismatch is EXACTLY that swap and
        # nothing else, so a real shape bug (e.g. a stale codec_dim) can never hide here.
        result = filt.load_state_dict(ckpt["filter"], strict=False)
        expected_missing = {"knowledge_filter.precision_head.weight", "knowledge_filter.precision_head.bias"}
        expected_unexpected = {"knowledge_filter.log_sigma_head.weight", "knowledge_filter.log_sigma_head.bias"}
        assert set(result.missing_keys) == expected_missing, (
            f"unexpected missing_keys on precision warm-start: {result.missing_keys}")
        assert set(result.unexpected_keys) == expected_unexpected, (
            f"unexpected unexpected_keys on precision warm-start: {result.unexpected_keys}")
        print(f"[fa-sigma-a3] precision warm-start canary PASS: missing={sorted(expected_missing)} "
             f"unexpected={sorted(expected_unexpected)}, nothing else", flush=True)
    outcome_head = OutcomeHead(codec_dim=B4_CONFIG["codec_dim"], latent_dim=B4_CONFIG["latent_dim"],
                               hidden=outcome_hidden).to(device)
    outcome_head.load_state_dict(ckpt["outcome_head"], strict=True)

    # -- an UNMODIFIED second copy, frozen throughout, for the canary + the "no worse than
    # the original" / "identity not reduced" baselines --
    filt_original = S1FilterV2ItemMLP(**B4_CONFIG).to(device)
    filt_original.load_state_dict(ckpt["filter"], strict=True)
    filt_original.eval()
    for p in filt_original.parameters():
        p.requires_grad = False
    outcome_head_original = OutcomeHead(codec_dim=B4_CONFIG["codec_dim"], latent_dim=B4_CONFIG["latent_dim"],
                                        hidden=outcome_hidden).to(device)
    outcome_head_original.load_state_dict(ckpt["outcome_head"], strict=True)
    outcome_head_original.eval()
    for p in outcome_head_original.parameters():
        p.requires_grad = False

    # -- FULL retrain, nothing frozen --
    n_trainable = sum(p.numel() for p in filt.parameters())
    print(f"[fa-sigma-a3] filter params: {n_trainable} trainable (full retrain, nothing frozen, "
         f"sigma_parameterization={sigma_parameterization})", flush=True)

    filt_forward = lambda ic, pf, tm, vm: filt(ic, pf)  # noqa: E731
    filt_forward_original = lambda ic, pf, tm, vm: filt_original(ic, pf)  # noqa: E731

    # -- data: no staged bundle needed, this stage never touches raw text --
    ev = load_v2_events(REPO / args.manifest_path, REPO / args.learner_split_path, REPO / args.item_split_path)
    ev_train = ev[ev["split"] == "train"].reset_index(drop=True)
    ev_val = ev[ev["split"] == "val"].reset_index(drop=True)
    codec_matrix, pid_to_idx = load_production_codec(codec_path, B4_CONFIG["codec_dim"])

    train_fb, train_l_idx, train_t_idx, train_target, _train_rows = build_filter_batch_and_targets(
        ev_train, codec_matrix, pid_to_idx, device)
    val_fb, val_l_idx, val_t_idx, val_target, val_rows = build_filter_batch_and_targets(
        ev_val, codec_matrix, pid_to_idx, device)
    print(f"[fa-sigma-a3] train: {len(train_l_idx)} outcome-valid rows / "
         f"{len(train_fb['learners'])} learners; val: {len(val_l_idx)} outcome-valid rows / "
         f"{len(val_fb['learners'])} learners", flush=True)

    trainable = [p for p in filt.parameters() if p.requires_grad] + list(outcome_head.parameters())
    optimizer = torch.optim.Adam(trainable, lr=args.lr)

    n_train = len(train_l_idx)
    epoch_sampler = ShuffledEpochSampler(n_train, np.random.default_rng(args.seed))
    loss_history = []
    print(f"[fa-sigma-a3] training: {args.steps} steps, micro_batch={args.micro_batch_size}, "
         f"likelihood=kmixture K_train={args.k_train}, {n_train} rows (true shuffled epochs)", flush=True)
    for step in range(args.steps):
        idx = epoch_sampler.next_batch(args.micro_batch_size)
        l_idx, t_idx = train_l_idx[idx], train_t_idx[idx]
        mu_sel, log_sigma_sel = gather_belief_for_batch(filt_forward, train_fb, l_idx, t_idx)
        item_codec_sel = gather_item_codec_for_batch(train_fb, l_idx, t_idx)
        target_sel = train_target[idx]
        valid_sel = torch.ones_like(target_sel, dtype=torch.bool)

        beta_kl = kl_anneal_weight(step, max_weight=args.kl_max_weight, anneal_epochs=args.kl_anneal_steps)
        l_outcome, _u = outcome_loss_likelihood(
            outcome_head, item_codec_sel, mu_sel, log_sigma_sel, target_sel, valid_sel,
            likelihood="kmixture", k=args.k_train)
        kl = gaussian_kl_free_bits_mu_only(mu_sel, log_sigma_sel, free_bits_mu=args.free_bits_mu).mean()
        loss = l_outcome + beta_kl * kl
        if not torch.isfinite(loss):
            print(f"[fa-sigma-a3] TRIPWIRE non-finite loss at step={step}: {loss.item()}", flush=True)
            return 1
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        loss_history.append(float(loss.item()))

        if step % 200 == 0 or step == args.steps - 1:
            elapsed = time.time() - t_start
            fuse_min = 0
            if s3 is not None:
                fuse_min = rearm_shutdown_fuse(elapsed, step, args.steps)
            print(f"[hb] step={step}/{args.steps} loss={loss.item():.4f} outcome={l_outcome.item():.4f} "
                 f"kl={kl.item():.4f} beta_kl={beta_kl:.5f} elapsed={elapsed:.0f}s fuse=+{fuse_min}min",
                 flush=True)
            if s3 is not None:
                (out_dir / "progress.json").write_text(json.dumps({
                    "step": step, "steps": args.steps, "loss": loss_history[-5:], "elapsed_s": elapsed,
                    "fuse_rearmed_minutes": fuse_min}))
                sync_to_s3()

    filt.eval()
    outcome_head.eval()
    loss_decreased = loss_history[-1] < loss_history[0]
    print(f"[fa-sigma-a3] training done. loss {loss_history[0]:.4f} -> {loss_history[-1]:.4f} "
         f"(decreased={loss_decreased})", flush=True)
    # No mu-unchanged canary here: in a full retrain mu is EXPECTED to change. Gate 3
    # below (mu's between-learner variance share) is the identity check.
    if device == "cuda":
        torch.cuda.empty_cache()

    # -- persist val prefix beliefs (pl_2b_belief_curves.py's layout) --
    with torch.no_grad():
        mu_val_all, log_sigma_val_all = chunked_filter_forward(filt_forward, val_fb)
    beliefs = persist_val_prefix_beliefs(ev_val, val_fb, mu_val_all, log_sigma_val_all)
    beliefs_path = REPO / args.beliefs_out
    beliefs_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(beliefs_path, **beliefs)
    print(f"[fa-sigma-a3] persisted {len(beliefs['id'])} val prefix beliefs -> {beliefs_path}", flush=True)
    # uploaded IMMEDIATELY, not deferred to end-of-run sync_to_s3() (which only globs
    # `out_dir`, never `beliefs_path`'s own `artifacts/` directory) -- an unattended run's
    # shutdown fuse can otherwise fire before this file is retrieved.
    if s3 is not None:
        s3.upload_file(str(beliefs_path), args.bucket, f"{args.out_prefix}/{beliefs_path.name}")
        print(f"[fa-sigma-a3] beliefs uploaded to bucket {args.bucket}: {args.out_prefix}/{beliefs_path.name}",
             flush=True)

    # -- by-value join canary --
    canary7 = value_join_canary(beliefs, REPO / args.manifest_path, np.random.default_rng(args.seed + 900))
    canary7_pass = (canary7["membership"] == 1.0 and canary7["t_by_value"] == 1.0
                    and canary7["problem_id_by_value"] == 1.0 and canary7["split_by_value"] == 1.0)
    print(f"[fa-sigma-a3] by-value join canary: {canary7}, PASS={canary7_pass}", flush=True)
    if not canary7_pass:
        print("[fa-sigma-a3] BY-VALUE JOIN CANARY FAILED -- halting before any gate is read", flush=True)
        return 1

    # -- GATE 1: tr(Sigma) + K-sample Jensen gap FALL from t=10 to t=100, fixed cohort --
    beliefs_df = pd.DataFrame({
        "learner": beliefs["learner"], "t": beliefs["t"],
        "row": np.arange(len(beliefs["learner"])),
    })
    t10_rows = beliefs_df[beliefs_df["t"] == FIXED_COHORT_T[0]].set_index("learner")
    t100_rows = beliefs_df[beliefs_df["t"] == FIXED_COHORT_T[1]].set_index("learner")
    cohort = t10_rows.index.intersection(t100_rows.index)
    print(f"[fa-sigma-a3] fixed cohort (learners with rows at t=10 AND t=100): {len(cohort)}", flush=True)

    idx10 = t10_rows.loc[cohort, "row"].to_numpy()
    idx100 = t100_rows.loc[cohort, "row"].to_numpy()
    mu10 = torch.tensor(beliefs["mu"][idx10], device=device)
    ls10 = torch.tensor(beliefs["log_sigma"][idx10], device=device)
    mu100 = torch.tensor(beliefs["mu"][idx100], device=device)
    ls100 = torch.tensor(beliefs["log_sigma"][idx100], device=device)

    sigma_trace_10 = (2.0 * ls10).exp().sum(dim=-1).detach().cpu().numpy()
    sigma_trace_100 = (2.0 * ls100).exp().sum(dim=-1).detach().cpu().numpy()

    item_codec10 = torch.tensor(
        np.stack([codec_matrix[pid_to_idx[int(p)]] for p in beliefs["problem_id"][idx10]]), device=device)
    item_codec100 = torch.tensor(
        np.stack([codec_matrix[pid_to_idx[int(p)]] for p in beliefs["problem_id"][idx100]]), device=device)
    gen = torch.Generator(device=device).manual_seed(args.seed + 901)
    jensen10 = jensen_gap_categorical(outcome_head, item_codec10, mu10, ls10, args.k_eval, gen).cpu().numpy()
    gen2 = torch.Generator(device=device).manual_seed(args.seed + 901)  # same noise draw shape/seed, paired
    jensen100 = jensen_gap_categorical(outcome_head, item_codec100, mu100, ls100, args.k_eval, gen2).cpu().numpy()

    cohort_ids = cohort.to_numpy()
    sigma_ci = learner_clustered_bootstrap_ci(sigma_trace_100, sigma_trace_10, cohort_ids,
                                              n_boot=args.n_boot, seed=args.seed)
    jensen_ci = learner_clustered_bootstrap_ci(jensen100, jensen10, cohort_ids,
                                               n_boot=args.n_boot, seed=args.seed)
    gate1_sigma_falls = bool(sigma_ci["ci95"][1] < 0)
    gate1_jensen_falls = bool(jensen_ci["ci95"][1] < 0)
    print(f"[fa-sigma-a3] GATE 1a tr(Sigma) t100-t10 paired CI: {sigma_ci}, FALLS={gate1_sigma_falls}", flush=True)
    print(f"[fa-sigma-a3] GATE 1b Jensen-gap t100-t10 paired CI: {jensen_ci}, FALLS={gate1_jensen_falls}", flush=True)

    # -- GATE 2: outcome log-loss no worse than the ORIGINAL head beyond the disclosed tolerance --
    item_codec_val = gather_item_codec_for_batch(val_fb, val_l_idx, val_t_idx)
    mu_val_sel, ls_val_sel = gather_belief_for_batch(filt_forward, val_fb, val_l_idx, val_t_idx)
    mu_val_sel_orig, ls_val_sel_orig = gather_belief_for_batch(filt_forward_original, val_fb, val_l_idx, val_t_idx)
    gen3 = torch.Generator(device=device).manual_seed(args.seed + 902)
    u_k_new = reparameterize_from_log_sigma(mu_val_sel, ls_val_sel, k=args.k_eval, generator=gen3)
    gen4 = torch.Generator(device=device).manual_seed(args.seed + 902)
    u_k_orig = reparameterize_from_log_sigma(mu_val_sel_orig, ls_val_sel_orig, k=args.k_eval, generator=gen4)
    with torch.no_grad():
        nll_new = outcome_kmixture_nll_per_row(outcome_head, item_codec_val, u_k_new, val_target)
        nll_orig = outcome_kmixture_nll_per_row(outcome_head_original, item_codec_val, u_k_orig, val_target)
    log_loss_new = float(nll_new.mean())
    log_loss_orig = float(nll_orig.mean())
    gate2_no_worse = log_loss_new <= log_loss_orig + args.log_loss_tolerance
    print(f"[fa-sigma-a3] GATE 2 outcome log-loss: A3={log_loss_new:.4f} original={log_loss_orig:.4f} "
         f"tolerance={args.log_loss_tolerance} NO_WORSE={gate2_no_worse}", flush=True)

    # -- GATE 3: identity (mu between-learner variance share), seed-floor-aware --
    mu_val_all_np = mu_val_all.detach().cpu().numpy()
    mu_val_all_orig, _ls_val_all_orig = chunked_filter_forward(filt_forward_original, val_fb)
    mu_val_all_orig_np = mu_val_all_orig.detach().cpu().numpy()
    vm_val = val_fb["valid_mask"].cpu().numpy()
    n_val, t_val = vm_val.shape
    learners_arr = np.array(val_fb["learners"])[:, None].repeat(t_val, axis=1)
    share_new = between_learner_mu_share(mu_val_all_np[vm_val], learners_arr[vm_val])
    share_orig = between_learner_mu_share(mu_val_all_orig_np[vm_val], learners_arr[vm_val])
    if args.identity_sigma_seed is not None:
        gate3_rule = "seed_floor"
        identity_delta = abs(share_new - args.identity_reference_share)
        identity_bound = 2.0 * args.identity_sigma_seed
        gate3_pass = identity_delta <= identity_bound
        print(f"[fa-sigma-a3] GATE 3 mu between-learner share: A3={share_new:.4f} "
             f"reference={args.identity_reference_share:.4f} sigma_seed={args.identity_sigma_seed:.4f} "
             f"|delta|={identity_delta:.4f} bound(2*sigma_seed)={identity_bound:.4f} "
             f"WITHIN_SEED_FLOOR={gate3_pass}", flush=True)
    else:
        gate3_rule = "not_reduced_fallback"
        gate3_pass = share_new >= share_orig - 1e-6
        print(f"[fa-sigma-a3] GATE 3 mu between-learner share: A3={share_new:.4f} "
             f"original={share_orig:.4f} NOT_REDUCED={gate3_pass} "
             f"[FALLBACK RULE -- no --identity-sigma-seed supplied, using 'not reduced']",
             flush=True)

    # -- GATE 5: log sigma finishes OFF both clamps (a smaller sigma pinned at a clamp is not
    # a pass; applies to the precision head's floor/ceiling too, not just the direct head) --
    ls_val_all_np = log_sigma_val_all.detach().cpu().numpy()
    clamp_min = filt.knowledge_filter.log_sigma_min
    clamp_max = filt.knowledge_filter.log_sigma_max
    frac_at_min = float(np.mean(np.isclose(ls_val_all_np[vm_val], clamp_min, atol=1e-4)))
    frac_at_max = float(np.mean(np.isclose(ls_val_all_np[vm_val], clamp_max, atol=1e-4)))
    gate5_off_clamps = frac_at_min < 0.01 and frac_at_max < 0.01
    print(f"[fa-sigma-a3] GATE 5 log-sigma off clamps: frac_at_min({clamp_min})={frac_at_min:.4f} "
         f"frac_at_max({clamp_max})={frac_at_max:.4f} OFF_CLAMPS={gate5_off_clamps}", flush=True)

    all_gates_pass = (gate1_sigma_falls and gate1_jensen_falls and gate2_no_worse
                      and gate3_pass and gate5_off_clamps)
    print(f"[fa-sigma-a3] ALL GATES PASS: {all_gates_pass}", flush=True)

    # -- persistence: standalone-reusable checkpoint (filter + outcome head) --
    ckpt_out = {
        "filter": filt.state_dict(), "outcome_head": outcome_head.state_dict(),
        "architecture": {"class": "S1FilterV2ItemMLP/OutcomeHead",
                         "sigma_parameterization": sigma_parameterization, **B4_CONFIG},
        "codec_path": str(args.codec_path), "codec_sha256": actual_sha,
        "sampler_semantics": CURRENT_SAMPLER_SEMANTICS, "git_sha": git_sha, "seed": args.seed,
        "warm_start_from": str(args.checkpoint_path), "warm_start_git_sha": ckpt.get("git_sha"),
        "likelihood": "kmixture", "k_train": args.k_train, "steps": args.steps,
        "free_bits_mu": args.free_bits_mu, "stage": "round2_full_retrain_fa_sim4_warm_start",
    }
    torch.save(ckpt_out, out_dir / "final.pt")

    summary = {
        "task": "fa_sigma_a3", "checkpoint_git_sha": git_sha, "warm_start_git_sha": ckpt.get("git_sha"),
        "sigma_parameterization": sigma_parameterization, "seed": args.seed, "steps": args.steps,
        "k_train": args.k_train, "k_eval": args.k_eval, "free_bits_mu": args.free_bits_mu,
        "loss_decreased": loss_decreased, "loss_first": loss_history[0], "loss_last": loss_history[-1],
        "canary7": canary7, "canary7_pass": canary7_pass,
        "fixed_cohort_n": int(len(cohort)),
        "gate1_sigma_trace_t100_minus_t10": sigma_ci, "gate1_sigma_falls": gate1_sigma_falls,
        "gate1_jensen_gap_t100_minus_t10": jensen_ci, "gate1_jensen_falls": gate1_jensen_falls,
        "gate2_log_loss_a3": log_loss_new, "gate2_log_loss_original": log_loss_orig,
        "gate2_tolerance": args.log_loss_tolerance, "gate2_no_worse": gate2_no_worse,
        "gate3_mu_between_learner_share_a3": share_new, "gate3_mu_between_learner_share_original": share_orig,
        "gate3_rule": gate3_rule, "gate3_identity_reference_share": args.identity_reference_share,
        "gate3_identity_sigma_seed": args.identity_sigma_seed, "gate3_pass": gate3_pass,
        "gate5_log_sigma_frac_at_min": frac_at_min, "gate5_log_sigma_frac_at_max": frac_at_max,
        "gate5_off_clamps": gate5_off_clamps,
        "all_gates_pass": all_gates_pass, "elapsed_s": time.time() - t_start,
        "beliefs_path": str(args.beliefs_out),
        "note_pl2b": "pl_2b_belief_curves.py is wired to a different architecture (S1FilterV2/"
                    "FiLMHeads, binary outcome, codec_dim=45) and has no CLI to accept an "
                    "external beliefs npz. Gate 1's "
                    "sigma_trace/jensen_gap formulas are reused verbatim from "
                    "atom.planner.replay_v2 (generalised to the 5-way categorical outcome); "
                    "the SCRIPT itself was not invoked.",
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[fa-sigma-a3] wrote {out_dir}/summary.json", flush=True)
    sync_to_s3()

    if args.self_terminate and s3 is not None:
        print("[fa-sigma-a3] self-terminating", flush=True)
        subprocess.run(["sudo", "shutdown", "-h", "now"], check=False)

    print(f"[fa-sigma-a3] DONE. elapsed={time.time() - t_start:.0f}s", flush=True)
    return 0 if all_gates_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
