#!/usr/bin/env python3
"""Retrain the outcome head of the CodeWorkout generative simulator on its frozen belief.

The simulator's jointly trained head gets 2 epochs at outcome-loss weight 0.3, against
the classifier's 60 epochs at weight 1.0, and its wrong-detection is worse on every panel
cell.  To test whether that is a head-budget effect rather than a belief defect, this
script freezes `attempt_encoder` and `filt` and trains a fresh `outcome_head` with the
classifier's recipe from `cw_sigma.py` (K-mixture, k=8, free bits 0.05, 60 epochs, lr 1e-3,
Adam, batch 32; KL dropped, as it has no gradient path into a frozen filter or the head).
Inputs: `--checkpoint` (the simulator's `final.pt`), raw CodeWorkout data and codec caches.
Outputs in `--out-dir`: `final.pt` (`ratchet-cwsim/refreshed-head`, bare-key schema, encoder
and filter verified unchanged with `torch.equal`) and `manifest.json`.  Serves Table 11.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
import time

REPO = pathlib.Path(__file__).resolve().parents[1]
for p in (REPO / "src", REPO / "infra", REPO / "scripts", REPO):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from cw_manifest_io import load_manifest  # noqa: E402
from cw_4g_codec_rank_item_mlp import rebuild_problem_codec_at_rank  # noqa: E402
from cw_classifier_common import CODEC_K, RANK, TRANSFORM_PATH  # noqa: E402

from atom.codec.transform import CodecTransform  # noqa: E402
from atom.data.cw_manifest import build_codec_vector_lookup  # noqa: E402
from atom.data.cw_outcome_label import N_OUTCOME  # noqa: E402
from atom.filter.cw_attempt_encoder import AttemptEncoder  # noqa: E402
from atom.filter.cw_filter_batch import build_attempt_batch, build_episode_batch  # noqa: E402
from atom.simulator.cw_4_train_loop import _distinct_episode_positions  # noqa: E402
from atom.simulator.cw_item_mlp import OutcomeHeadItemMLP, S1FilterV2ItemMLP  # noqa: E402
from atom.simulator.cw_train_step import build_cw_targets, losses_for_cw_hierarchical_batch  # noqa: E402
from atom.utils.seeds import seed_everything  # noqa: E402


def _sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _git_sha() -> str:
    import subprocess
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO).decode().strip()
    except Exception:
        return "nogit"


class _Args:
    pass


def freeze(module: torch.nn.Module) -> dict:
    """Return a CPU-detached snapshot of `module`'s state dict for the before/after
    freeze check, and set `requires_grad_(False)` + `.eval()` so the module
    contributes activations to the graph but never accumulates a gradient."""
    snapshot = {k: v.detach().clone().cpu() for k, v in module.state_dict().items()}
    for p in module.parameters():
        p.requires_grad_(False)
    module.eval()
    return snapshot


def assert_frozen(module: torch.nn.Module, snapshot: dict, name: str) -> None:
    current = module.state_dict()
    if set(current.keys()) != set(snapshot.keys()):
        raise AssertionError(f"freeze check: {name} state dict keys changed during training")
    for k, v0 in snapshot.items():
        if not torch.equal(v0, current[k].detach().cpu()):
            raise AssertionError(f"freeze check FAILED: {name}.{k} changed during head-only training")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="results/cw_sim_base_run/final.pt",
                    help="the generative simulator's final.pt (bare-key schema)")
    ap.add_argument("--raw-dir", default=None)
    for name in ("code_map", "diff_map", "codealone_meta", "codealone_states",
                "diff_meta", "diff_states"):
        ap.add_argument("--" + name.replace("_", "-"), required=True)
    ap.add_argument("--learner-batch-size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--k-train", type=int, default=8)
    ap.add_argument("--free-bits", type=float, default=0.05)
    ap.add_argument("--likelihood", choices=("single", "kmixture"), default="kmixture")
    ap.add_argument("--out-dir", default="results/cw_sim_head_refresh")
    ns = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = REPO / ns.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    manifest_args = _Args()
    manifest_args.raw_dir = ns.raw_dir
    manifest_args.code_map, manifest_args.diff_map = ns.code_map, ns.diff_map
    manifest_args.codealone_meta, manifest_args.codealone_states = ns.codealone_meta, ns.codealone_states
    manifest_args.diff_meta, manifest_args.diff_states = ns.diff_meta, ns.diff_states

    print("[head-refresh] loading manifest (S19+F19 train-pool+val, lockbox excluded)...", flush=True)
    base_manifest = load_manifest(manifest_args)
    # `load_manifest` builds its own code_of_hash/diff_of_hash/problem_codec at rank=50
    # (codec_dim=45), which does not fit this checkpoint. They are rebuilt at RANK=69
    # (codec_dim=64), as in `cw_sigma.py`'s main(), so the codec matches the
    # checkpoint's architecture dict and the frozen filter's input width.
    transform = CodecTransform.load(TRANSFORM_PATH)
    code_hashes = set(base_manifest["attempts"]["code_hash"].dropna())
    diff_hashes = set(base_manifest["attempts"].loc[
        base_manifest["attempts"]["diff_hash"].notna(), "diff_hash"
    ])
    code_of_hash = build_codec_vector_lookup(
        code_hashes, ns.codealone_meta, ns.codealone_states, transform, k=CODEC_K, rank=RANK)
    diff_of_hash = build_codec_vector_lookup(
        diff_hashes, ns.diff_meta, ns.diff_states, transform, k=CODEC_K, rank=RANK)
    problem_codec, pid_to_idx = rebuild_problem_codec_at_rank(RANK, transform=transform)
    if len(problem_codec) != 50 or len(pid_to_idx) != 50:
        raise AssertionError(f"fixed diagnostic cohort requires all 50 problems, got {len(problem_codec)}")
    manifest = {**base_manifest, "code_of_hash": code_of_hash, "diff_of_hash": diff_of_hash,
               "problem_codec": problem_codec, "pid_to_idx": pid_to_idx}
    print(f"[head-refresh] n_attempts={len(manifest['attempts'])} "
         f"n_train_learners={len(manifest['train_learners'])} "
         f"n_val_learners={len(manifest['val_learners'])}", flush=True)

    ckpt_path = REPO / ns.checkpoint
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    d = ckpt["architecture"]
    ratchet_cwsim_sha256 = _sha256(ckpt_path)

    attempt_encoder = AttemptEncoder(d["codec_dim"], d["codec_dim"], d["attempt_proj_dim"],
                                     d["attempt_hidden"], d["summary_dim"]).to(device)
    filt = S1FilterV2ItemMLP(codec_dim=d["codec_dim"], episode_hidden=d["episode_hidden"],
                             filter_hidden=d["filter_hidden"], latent_dim=d["latent_dim"],
                             prev_feat_dim=d["summary_dim"],
                             sigma_parameterization="precision_accumulating").to(device)
    attempt_encoder.load_state_dict(ckpt["attempt_encoder"])
    filt.load_state_dict(ckpt["filt"])
    attempt_encoder_snapshot = freeze(attempt_encoder)
    filt_snapshot = freeze(filt)
    filt_forward = lambda ic, pf, tm, vm: filt(ic, pf)  # noqa: E731 -- matches build_filter's own mlp convention

    seed_everything(ns.seed)
    outcome_head = OutcomeHeadItemMLP(codec_dim=d["codec_dim"], latent_dim=d["latent_dim"],
                                      hidden=d["attempt_hidden"], n_classes=N_OUTCOME,
                                      prefix_dim=d["attempt_hidden"]).to(device)
    optimizer = torch.optim.Adam(outcome_head.parameters(), lr=ns.lr)

    attempts = manifest["attempts"]
    episodes = manifest["episodes"]
    train_learners = sorted(manifest["train_learners"])
    rng = np.random.default_rng(ns.seed)
    n_batches_per_epoch = max(1, (len(train_learners) + ns.learner_batch_size - 1) // ns.learner_batch_size)
    total_steps = n_batches_per_epoch * ns.epochs
    print(f"[head-refresh] training: {ns.epochs} epochs x {n_batches_per_epoch} "
         f"batches/epoch = {total_steps} steps, likelihood={ns.likelihood} "
         f"k={ns.k_train} weight=1.0 (undemoted)", flush=True)

    step = 0
    loss_history: list[float] = []
    for epoch in range(ns.epochs):
        order = rng.permutation(len(train_learners))
        for bstart in range(0, len(train_learners), ns.learner_batch_size):
            batch_learners = {train_learners[i] for i in order[bstart:bstart + ns.learner_batch_size]}
            attempts_batch = attempts[attempts["learner"].isin(batch_learners)]
            episodes_batch = episodes[episodes["learner"].isin(batch_learners)]

            optimizer.zero_grad()
            attempt_batch = build_attempt_batch(attempts_batch, manifest["code_of_hash"],
                                                manifest["diff_of_hash"], d["codec_dim"],
                                                d["codec_dim"], device)
            prefix_states, summary = attempt_encoder(
                attempt_batch["code_vec"], attempt_batch["diff_vec"], attempt_batch["scalar_feats"],
                attempt_batch["lengths"])
            episode_ids_of_summary = attempt_batch["episode_ids"]
            ep_batch = build_episode_batch(episodes_batch, problem_codec, pid_to_idx, summary,
                                           episode_ids_of_summary, device)
            ep_learner_idx, ep_t_idx = _distinct_episode_positions(
                episodes_batch, ep_batch, episode_ids_of_summary, device)
            targets = build_cw_targets(attempts_batch, device)
            l_outcome, _kl, _logits = losses_for_cw_hierarchical_batch(
                filt_forward, outcome_head, ep_batch, ep_learner_idx, ep_t_idx, prefix_states,
                attempt_batch["attempt_episode_row_idx"], attempt_batch["attempt_within_idx"],
                targets, beta_kl=0.0, k=ns.k_train, free_bits=ns.free_bits, likelihood=ns.likelihood)
            total = l_outcome  # outcome-loss weight 1.0
            if not torch.isfinite(total):
                raise RuntimeError(f"[head-refresh] TRIPWIRE non-finite loss at step={step}: {total.item()}")
            total.backward()
            torch.nn.utils.clip_grad_norm_(outcome_head.parameters(), max_norm=1.0)
            optimizer.step()
            loss_history.append(l_outcome.item())
            step += 1
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"[head-refresh] epoch={epoch + 1}/{ns.epochs} L_outcome={loss_history[-1]:.4f}", flush=True)

    d0 = max(1, n_batches_per_epoch)
    loss_first = float(np.mean(loss_history[:d0]))
    loss_last = float(np.mean(loss_history[-d0:]))
    print(f"[head-refresh] done in {time.time() - t0:.1f}s: "
         f"L_outcome first-epoch-mean={loss_first:.4f} last-epoch-mean={loss_last:.4f}", flush=True)

    assert_frozen(attempt_encoder, attempt_encoder_snapshot, "attempt_encoder")
    assert_frozen(filt, filt_snapshot, "filt")
    print("[head-refresh] freeze check PASS: attempt_encoder + filt byte-identical before/after", flush=True)

    out_ckpt = {
        "architecture": d,
        "attempt_encoder": {k: v.detach().cpu() for k, v in attempt_encoder.state_dict().items()},
        "filt": {k: v.detach().cpu() for k, v in filt.state_dict().items()},
        "outcome_head": {k: v.detach().cpu() for k, v in outcome_head.state_dict().items()},
        "sigma_parameterization": "precision_accumulating",
        "source": "ratchet-cwsim/refreshed-head",
        "ratchet_cwsim_checkpoint": ns.checkpoint,
        "ratchet_cwsim_checkpoint_sha256": ratchet_cwsim_sha256,
        "head_refresh_recipe": {
            "epochs": ns.epochs, "lr": ns.lr, "seed": ns.seed, "learner_batch_size": ns.learner_batch_size,
            "likelihood": ns.likelihood, "k_train": ns.k_train, "free_bits": ns.free_bits,
            "outcome_loss_weight": 1.0, "optimizer": "Adam", "filter_frozen": True,
        },
        "loss_first_epoch_mean": loss_first, "loss_last_epoch_mean": loss_last,
        "git_sha": _git_sha(),
    }
    out_path = out_dir / "final.pt"
    torch.save(out_ckpt, out_path)
    sha = _sha256(out_path)
    manifest_out = {
        "checkpoint": str(out_path.relative_to(REPO)), "sha256": sha,
        "ratchet_cwsim_checkpoint_sha256": ratchet_cwsim_sha256,
        "recipe": out_ckpt["head_refresh_recipe"],
        "loss_first_epoch_mean": loss_first, "loss_last_epoch_mean": loss_last,
        "canary_6": "PASS -- attempt_encoder + filt byte-identical before/after",
        "elapsed_s": time.time() - t0,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest_out, indent=2))
    print(json.dumps(manifest_out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
