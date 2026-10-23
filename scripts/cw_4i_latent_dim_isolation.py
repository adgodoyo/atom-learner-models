#!/usr/bin/env python
"""Belief-dimension ablation for the CodeWorkout classifier: `latent_dim` 32 vs 64, all else pinned.

The joint width sweep (cw_4d_width_summary_sweep.py) moves five dims together, so its
degradation (-0.063 NLL at the first step, params 79,988 -> 295,124) cannot be attributed
to `latent_dim` alone. Here only `latent_dim` varies; `episode_hidden`=64, `filter_hidden`=64,
`attempt_proj_dim`=32, `attempt_hidden`=64 and `summary_dim`=16 stay at `DEFAULT_DIMS`
(they are absent from the `dims` override), with `codec_dim`=64, `item_encoder`='linear',
GRU body. Each model is trained on the WITHOUT-H arm (7 held-out items stripped from train
learners) and scored on held-out items and on full val; 5 paired seeds, since the measured
seed-to-seed spread (sigma ~0.0025 full-val AUC, ~0.0020 held-out) is too large for one.
Reported: all-attempts NLL, P(clean-full) AUC, the held-out contrast, n_params and the
final-epoch train/val gap. The mathematics simulator's own `latent_dim=256` is not used as
an argument here (about 3,500 vs 686 train learners).
Pre-registered rule: `latent_dim=64` ships ONLY IF it improves full-val AUC with CI clear of
zero AND its held-out CI is not clearly negative; otherwise 32 stays (parsimony breaks ties).
Inputs: code/diff hash maps and codec caches (flags below), the CodeWorkout splits.
Output: `reports/cw_4i_latent_dim_isolation.md`. Serves Appendix A (belief dimension).

    python scripts/cw_4i_latent_dim_isolation.py \\
        --code-map cache/cw_03_vectors/cw_code_state_hash_map_full.npz \\
        --diff-map cache/cw_03_vectors/cw_diff_hash_map_full.npz \\
        --codealone-meta cache/cw_03_vectors/codealone/meta.json \\
        --codealone-states cache/cw_03_vectors/codealone/states_mean.npy \\
        --diff-meta cache/cw_03_vectors/diff/meta.json \\
        --diff-states cache/cw_03_vectors/diff/states_mean.npy
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time

SCRIPTS_DIR = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR.parents[0] / "src"))
sys.path.insert(0, str(SCRIPTS_DIR))

import numpy as np  # noqa: E402

from cw_manifest_io import load_manifest  # noqa: E402
from cw_4g_codec_rank_item_mlp import rebuild_problem_codec_at_rank  # noqa: E402

from atom.codec.transform import CodecTransform  # noqa: E402
from atom.data.cw_challenge_split import HELD_OUT_ITEMS  # noqa: E402
from atom.data.cw_manifest import build_codec_vector_lookup, build_episode_level_table, strip_held_out_items  # noqa: E402
from atom.data.cw_outcome_label import OUTCOME_CLASSES  # noqa: E402
from atom.eval.metrics import auc, bootstrap_mean_ci  # noqa: E402
from atom.eval.report import gate_report  # noqa: E402
from atom.simulator.cw_4_train_loop import CODEC_K, run_one_seed_body  # noqa: E402
from atom.utils.logging import RunLogger  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[1]
BODY = "gru"
K_FIXED = CODEC_K  # 5
RANK = 69  # codec_dim = 64 -- FIXED at the final codec width
CODEC_DIM = RANK - K_FIXED
assert CODEC_DIM == 64
ITEM_ENCODER = "linear"  # the MLP did not resolve a gain, so parsimony keeps linear
LATENT_DIMS = [32, 64]  # the axis under test -- everything else pinned at baseline below
SEEDS = list(range(5))
HELD = set(HELD_OUT_ITEMS)
TRANSFORM_PATH = REPO / "transforms" / "problem_mean_L15.npz"
CLEAN_FULL_IDX = OUTCOME_CLASSES.index("CLEAN_FULL")


def cell_auc_on_mask(probs: np.ndarray, target: np.ndarray, mask: np.ndarray) -> tuple[float, int]:
    n = int(mask.sum())
    if n == 0:
        return float("nan"), 0
    y = (target[mask] == CLEAN_FULL_IDX).astype(int)
    p = probs[mask][:, CLEAN_FULL_IDX]
    if len(set(y.tolist())) != 2:
        return float("nan"), n
    return auc(y, p), n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default=None)
    ap.add_argument("--code-map", required=True)
    ap.add_argument("--diff-map", required=True)
    ap.add_argument("--codealone-meta", required=True)
    ap.add_argument("--codealone-states", required=True)
    ap.add_argument("--diff-meta", required=True)
    ap.add_argument("--diff-states", required=True)
    ap.add_argument("--learner-batch-size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    args = ap.parse_args()

    device = "cpu"
    logger = RunLogger(config={"task": "cw_4i_latent_dim_isolation", "body": BODY, "codec_dim": CODEC_DIM,
                               "item_encoder": ITEM_ENCODER, "latent_dims": LATENT_DIMS, "seeds": args.seeds,
                               "held_out_items": sorted(HELD)},
                       data_version="CSEDM-S19+F19-challenge-train-pool+val")

    logger.heartbeat(step=0, force=True, stage="loading base manifest + building the WITHOUT-H arm")
    base_manifest = load_manifest(args)
    without_h_attempts = strip_held_out_items(base_manifest["attempts"], HELD, base_manifest["train_learners"])
    without_h_episodes = build_episode_level_table(without_h_attempts)
    code_hashes_needed = set(without_h_attempts["code_hash"].dropna())
    diff_hashes_needed = set(without_h_attempts.loc[without_h_attempts["diff_hash"].notna(), "diff_hash"])

    transform = CodecTransform.load(TRANSFORM_PATH)
    assert transform.k_max >= RANK
    code_of_hash = build_codec_vector_lookup(code_hashes_needed, args.codealone_meta,
                                             args.codealone_states, transform, k=K_FIXED, rank=RANK)
    diff_of_hash = build_codec_vector_lookup(diff_hashes_needed, args.diff_meta,
                                             args.diff_states, transform, k=K_FIXED, rank=RANK)
    problem_codec, pid_to_idx = rebuild_problem_codec_at_rank(RANK, transform=transform)
    cell_manifest = {**base_manifest, "attempts": without_h_attempts, "episodes": without_h_episodes,
                     "code_of_hash": code_of_hash, "diff_of_hash": diff_of_hash,
                     "problem_codec": problem_codec, "pid_to_idx": pid_to_idx}
    logger.heartbeat(step=1, force=True, stage="WITHOUT-H arm + codec_dim=64 ready", n_seeds=len(args.seeds))

    per_seed: dict[int, list[dict]] = {L: [] for L in LATENT_DIMS}
    for L in LATENT_DIMS:
        # ONLY latent_dim and codec_dim are overridden -- episode_hidden/filter_hidden/
        # attempt_proj_dim/attempt_hidden/summary_dim stay at DEFAULT_DIMS (letting them
        # move together with latent_dim is the confound this ablation removes).
        dims = {"codec_dim": CODEC_DIM, "latent_dim": L}
        for seed in args.seeds:
            t0 = time.time()
            r = run_one_seed_body(BODY, seed, cell_manifest, args, device, dims=dims, item_encoder=ITEM_ENCODER)
            wall = time.time() - t0
            probs, target, pid = r["val"]["probs_real"], r["val"]["target"], r["val"]["problem_id"]
            held_mask = np.isin(pid, list(HELD))
            held_auc, held_n = cell_auc_on_mask(probs, target, held_mask)
            full_auc, full_n = cell_auc_on_mask(probs, target, np.ones_like(held_mask, dtype=bool))
            n_params = (sum(p.numel() for p in r["model"]["attempt_encoder"].parameters())
                       + sum(p.numel() for p in r["model"]["filt"].parameters())
                       + sum(p.numel() for p in r["model"]["outcome_head"].parameters())
                       + sum(p.numel() for p in r["model"]["y_ever_head"].parameters()))
            train_val_gap = r["loss_last"] - r["all_attempts"]["nll"]
            per_seed[L].append({
                "seed": seed, "nll_all_attempts": r["all_attempts"]["nll"],
                "held_auc": held_auc, "held_n": held_n, "full_auc": full_auc, "full_n": full_n,
                "n_params": n_params, "train_loss_last": r["loss_last"], "train_val_gap": train_val_gap,
                "wall_s": round(wall, 1),
            })
            logger.heartbeat(step=0, force=True, stage=f"latent_dim={L} seed={seed} done",
                             nll=round(r["all_attempts"]["nll"], 4),
                             held_auc=round(held_auc, 4) if held_auc == held_auc else held_auc,
                             full_auc=round(full_auc, 4) if full_auc == full_auc else full_auc,
                             gap=round(train_val_gap, 4), n_params=n_params, wall_s=round(wall, 1))

    # -- paired within-model contrast, 64 - 32, same seed order both cells --
    full_pairs = [(c64["full_auc"], c32["full_auc"]) for c64, c32 in zip(per_seed[64], per_seed[32])]
    held_pairs = [(c64["held_auc"], c32["held_auc"]) for c64, c32 in zip(per_seed[64], per_seed[32])]
    nll_pairs = [(c64["nll_all_attempts"], c32["nll_all_attempts"]) for c64, c32 in zip(per_seed[64], per_seed[32])]

    full_gap_all = [a - b for a, b in full_pairs]
    held_gap_all = [a - b for a, b in held_pairs if a == a and b == b]
    n_held_nan_dropped = len(held_pairs) - len(held_gap_all)
    nll_gap_all = [a - b for a, b in nll_pairs]  # positive = 64 WORSE (higher NLL)

    full_mean, full_lo, full_hi = bootstrap_mean_ci(np.array(full_gap_all), n_boot=5000, alpha=0.05, seed=args.seeds[0])
    held_mean, held_lo, held_hi = (bootstrap_mean_ci(np.array(held_gap_all), n_boot=5000, alpha=0.05, seed=args.seeds[0])
                                   if len(held_gap_all) > 1 else (float("nan"),) * 3)
    nll_mean, nll_lo, nll_hi = bootstrap_mean_ci(np.array(nll_gap_all), n_boot=5000, alpha=0.05, seed=args.seeds[0])

    full_ci_clear_positive = bool(full_lo == full_lo and full_lo > 0.0)
    held_ci_clear_negative = bool(held_hi == held_hi and held_hi < 0.0)  # 64 clearly LOSES on held-out

    if full_ci_clear_positive and not held_ci_clear_negative:
        verdict = "SHIP_64"
        verdict_text = ("Full-val AUC gain for latent_dim=64 is CI-clear of zero and positive, and it does "
                        "not clearly lose on held-out items -- ships per the pre-registered rule.")
    elif full_ci_clear_positive and held_ci_clear_negative:
        verdict = "KEEP_32_HELD_LOSS"
        verdict_text = ("Full-val AUC gain for latent_dim=64 is CI-clear positive, BUT the held-out-items "
                        "contrast is CI-clear NEGATIVE (64 measurably worse on items never trained on) -- "
                        "the same capacity-without-transfer signature the codec-rank item-holdout check "
                        "looks for. 32 stays per the pre-registered rule.")
    else:
        verdict = "KEEP_32_NO_FULL_GAIN"
        verdict_text = ("Full-val AUC gain for latent_dim=64 does not clear zero (tie or UNDER_RESOLUTION) "
                        "-- 32 stays per the pre-registered rule; parsimony breaks the tie.")

    passed = (len(per_seed[32]) == len(args.seeds) and len(per_seed[64]) == len(args.seeds)
             and all(c["held_n"] > 0 for c in per_seed[32] + per_seed[64]))
    logger.summary(status="ok" if passed else "gate-fail", verdict=verdict,
                  full_gap_mean=full_mean, held_gap_mean=held_mean)

    meta = {"run_id": logger.run_id, "git_sha": logger.git_sha, "body": BODY, "codec_dim": CODEC_DIM,
           "item_encoder": ITEM_ENCODER, "latent_dims": LATENT_DIMS, "seeds": args.seeds,
           "held_out_items": sorted(HELD), "n_train_learners": len(base_manifest["train_learners"]),
           "n_val_learners": len(base_manifest["val_learners"]),
           "pinned_dims": {"episode_hidden": 64, "filter_hidden": 64, "attempt_proj_dim": 32,
                          "attempt_hidden": 64, "summary_dim": 16}}
    metrics = {
        "per-seed cells (latent_dim -> [seed, nll, held_auc, n_held, full_auc, n_params, train_val_gap, wall_s])": per_seed,
        "paired contrast, latent64 - latent32, full-val AUC: mean/CI": {"mean": full_mean, "ci_lo": full_lo, "ci_hi": full_hi,
                                                                        "ci_clear_positive": full_ci_clear_positive},
        "paired contrast, latent64 - latent32, held-out-items AUC: mean/CI/n_pairs_used": {
            "mean": held_mean, "ci_lo": held_lo, "ci_hi": held_hi, "n_pairs_used": len(held_gap_all),
            "n_pairs_dropped_nan": n_held_nan_dropped, "ci_clear_negative": held_ci_clear_negative},
        "paired contrast, latent64 - latent32, all-attempts NLL (positive = 64 worse): mean/CI": {
            "mean": nll_mean, "ci_lo": nll_lo, "ci_hi": nll_hi},
        "PRE-REGISTERED VERDICT": verdict,
    }
    notes = [
        "**Isolating `latent_dim`** -- the joint width sweep moved 5 dims "
        "together, so `latent_dim` alone was not isolated there; this pins `episode_hidden`=64, "
        "`filter_hidden`=64, `attempt_proj_dim`=32, `attempt_hidden`=64, `summary_dim`=16 at the "
        "locked baseline (verified: these keys are simply ABSENT from the `dims` override at every "
        "cell, so `DEFAULT_DIMS` supplies them unchanged -- the confound this ablation removes).",
        f"**Verdict: {verdict}.** {verdict_text}",
        f"Full-val AUC contrast (64-32): {full_mean:+.4f} [{full_lo:+.4f}, {full_hi:+.4f}]. Held-out-items "
        f"AUC contrast (64-32): {held_mean:+.4f} [{held_lo:+.4f}, {held_hi:+.4f}]"
        + (f" ({n_held_nan_dropped} pair(s) dropped, degenerate class balance)" if n_held_nan_dropped else "")
        + f". All-attempts NLL contrast (64-32, positive=64 worse): {nll_mean:+.4f} [{nll_lo:+.4f}, {nll_hi:+.4f}].",
        "**Cross-corpus coherence NOT invoked on this knob** -- the mathematics simulator uses "
        "`latent_dim=256`, deliberately different from CodeWorkout's 32/64 on measured grounds (686 vs "
        "~3,500 train learners inverts this lever). Hidden widths (`episode_hidden`/`filter_hidden`) "
        "stay fixed, given the joint width sweep's degradation.",
        "**5 paired seeds** -- the measured seed-to-seed floor at this scale (sigma ~0.0025 full-val AUC, "
        "~0.0020 held-out) is why a single seed would not resolve an effect this small.",
    ]

    out_path = REPO / "reports" / "cw_4i_latent_dim_isolation.md"
    gate_report(out_path, "CodeWorkout belief dimension: latent_dim 32 vs 64, all other dims pinned",
               meta, metrics, passed,
               f"All {len(args.seeds)}x2 cells complete, non-empty held-out population every cell, "
               "no dim other than latent_dim/codec_dim overridden. This is a diagnostic (not a "
               "threshold) gate -- the finding is the PRE-REGISTERED VERDICT above.",
               notes)

    print("\n" + "=" * 100)
    print("latent_dim isolation, codec_dim=64/linear fixed, WITHOUT-H arm")
    print("=" * 100)
    for L in LATENT_DIMS:
        print(f"-- latent_dim={L} --")
        for c in per_seed[L]:
            print(f"  seed={c['seed']} nll={c['nll_all_attempts']:.4f} held_auc={c['held_auc']:.4f} "
                 f"(n={c['held_n']}) full_auc={c['full_auc']:.4f} n_params={c['n_params']:,} "
                 f"gap={c['train_val_gap']:+.4f} wall_s={c['wall_s']}")
    print(f"\nfull-val contrast (64-32): {full_mean:+.4f} [{full_lo:+.4f}, {full_hi:+.4f}]")
    print(f"held-out contrast (64-32): {held_mean:+.4f} [{held_lo:+.4f}, {held_hi:+.4f}]")
    print(f"\nVERDICT: {verdict}")
    print(f"cw_4i_latent_dim_isolation: {'PASS' if passed else 'FAIL'}")
    print(f"report: reports/cw_4i_latent_dim_isolation.md")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
