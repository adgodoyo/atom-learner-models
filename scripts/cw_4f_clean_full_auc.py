#!/usr/bin/env python
"""Train the CodeWorkout classifier at its final configuration and report its `P(clean-full)` AUC.

The 4-class outcome head is scored with NLL/Brier/ECE; this script adds a binary AUC so the
CodeWorkout and mathematics corpora can be reported in the same kind of number. It trains
one seed-0 GRU model (widths 32/64/64, `codec_dim`=64, `item_encoder`='linear',
`latent_dim`=32) on all 50 items, binarises `P(clean-full)` against the `clean-full` label,
and computes the AUC on all attempts and on the attempt-1 stratum, together with the same
binary AUC for two baselines fitted on train only (per-problem base rate;
learner-prefix with shrinkage). The model's AUC is never reported without them.
Comparability: the mathematics corpus is binary correctness at ~80% unaided-correct, this
is attempt-level full credit at ~24%; the numbers are comparable in kind, never in value.
Inputs: code/diff hash maps and codec caches (flags below), the CodeWorkout splits.
Outputs: `artifacts/cw_item_codec_dim64.npz` (50 items x 64 dims, rank=69, k=5; committed
so the checkpoint's `codec_path` is recoverable), `artifacts/checkpoints/cw_4f_shipped.pt`,
`reports/cw_4f_predictions.csv.gz` (per-row val predictions), `reports/cw_4f_clean_full_auc.md`.
Serves Table 9 and Figure 6 (the CodeWorkout classifier).

    python scripts/cw_4f_clean_full_auc.py \\
        --code-map cache/cw_03_vectors/cw_code_state_hash_map_full.npz \\
        --diff-map cache/cw_03_vectors/cw_diff_hash_map_full.npz \\
        --codealone-meta cache/cw_03_vectors/codealone/meta.json \\
        --codealone-states cache/cw_03_vectors/codealone/states_mean.npy \\
        --diff-meta cache/cw_03_vectors/diff/meta.json \\
        --diff-states cache/cw_03_vectors/diff/states_mean.npy
"""

from __future__ import annotations

import argparse
import hashlib
import pathlib
import sys

SCRIPTS_DIR = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR.parents[0] / "src"))
sys.path.insert(0, str(SCRIPTS_DIR))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from cw_manifest_io import load_manifest  # noqa: E402
from cw_4g_codec_rank_item_mlp import rebuild_problem_codec_at_rank  # noqa: E402

from atom.codec.transform import CodecTransform  # noqa: E402
from atom.data.cw_manifest import (  # noqa: E402
    build_codec_vector_lookup,
    fit_global_class_freq,
    fit_per_problem_base_rate,
    per_problem_base_rate_probs,
)
from atom.data.cw_outcome_label import N_OUTCOME, OUTCOME_CLASSES  # noqa: E402
from atom.eval.metrics import auc  # noqa: E402
from atom.eval.report import gate_report  # noqa: E402
from atom.eval.support_baselines import learner_prefix_probs  # noqa: E402
from atom.simulator.cw_4_train_loop import CODEC_K, run_one_seed_body  # noqa: E402
from atom.utils.logging import RunLogger  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[1]
BODY = "gru"
SEED = 0
K_FIXED = CODEC_K  # 5
RANK = 69  # codec_dim = 64, the final codec width
CODEC_DIM = RANK - K_FIXED
LATENT_DIM = 32  # final belief dimension (also DEFAULT_DIMS -- stated explicitly here)
ITEM_ENCODER = "linear"  # final item encoder
SHRINKAGE_K = 5.0
TRANSFORM_PATH = REPO / "transforms" / "problem_mean_L15.npz"
CODEC_ARTIFACT_PATH = REPO / "artifacts" / "cw_item_codec_dim64.npz"
CHECKPOINT_DIR = REPO / "artifacts" / "checkpoints"
CLEAN_FULL_IDX = OUTCOME_CLASSES.index("CLEAN_FULL")


def _sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def binary_auc(probs: np.ndarray, target: np.ndarray, mask: np.ndarray | None = None) -> tuple[float, int]:
    if mask is not None:
        probs, target = probs[mask], target[mask]
    n = len(target)
    if n == 0:
        return float("nan"), 0
    y = (target == CLEAN_FULL_IDX).astype(int)
    p = probs[:, CLEAN_FULL_IDX]
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
    args = ap.parse_args()

    device = "cpu"
    logger = RunLogger(config={"task": "cw_4f_clean_full_auc", "body": BODY, "seed": SEED,
                               "codec_dim": CODEC_DIM, "item_encoder": ITEM_ENCODER, "latent_dim": LATENT_DIM},
                       data_version="CSEDM-S19+F19-challenge-train-pool+val")

    logger.heartbeat(step=0, force=True, stage="loading WITH-H manifest (all 50 items)")
    manifest = load_manifest(args)

    transform = CodecTransform.load(TRANSFORM_PATH)
    assert transform.k_max >= RANK
    code_hashes_needed = set(manifest["attempts"]["code_hash"].dropna())
    diff_hashes_needed = set(manifest["attempts"].loc[manifest["attempts"]["diff_hash"].notna(), "diff_hash"])
    code_of_hash = build_codec_vector_lookup(code_hashes_needed, args.codealone_meta,
                                             args.codealone_states, transform, k=K_FIXED, rank=RANK)
    diff_of_hash = build_codec_vector_lookup(diff_hashes_needed, args.diff_meta,
                                             args.diff_states, transform, k=K_FIXED, rank=RANK)
    problem_codec, pid_to_idx = rebuild_problem_codec_at_rank(RANK, transform=transform)

    # -- new committed artifact, mirrors fa_item_codec_dim64.npz exactly --
    CODEC_ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    pids_sorted = sorted(pid_to_idx)
    codec_rows = np.stack([problem_codec[pid_to_idx[p]] for p in pids_sorted])
    np.savez(CODEC_ARTIFACT_PATH, problem_id=np.array(pids_sorted, dtype=np.int64), codec=codec_rows)
    logger.heartbeat(step=1, force=True, stage="committed codec_dim=64 item-codec artifact written",
                     path=str(CODEC_ARTIFACT_PATH.relative_to(REPO)), shape=list(codec_rows.shape))

    train_manifest = {**manifest, "code_of_hash": code_of_hash, "diff_of_hash": diff_of_hash,
                      "problem_codec": problem_codec, "pid_to_idx": pid_to_idx}

    logger.heartbeat(step=2, force=True, stage="training the shipped-config model (seed=0, WITH-H, all 50 items)")
    r = run_one_seed_body(BODY, SEED, train_manifest, args, device,
                          dims={"codec_dim": CODEC_DIM, "latent_dim": LATENT_DIM}, item_encoder=ITEM_ENCODER)
    val = r["val"]
    logger.heartbeat(step=3, force=True, stage="model trained + evaluated", nll_all=r["all_attempts"]["nll"])

    model_auc_all, n_all = binary_auc(val["probs_real"], val["target"])
    model_auc_a1, n_a1 = binary_auc(val["probs_real"], val["target"], val["attempt1_mask"])

    # -- both baselines, fit on TRAIN only, scored on the identical binary task --
    attempts = manifest["attempts"]
    n_invalid = int((~attempts["outcome_valid"]).sum())
    attempts_valid = attempts[attempts["outcome_valid"]]
    train_attempts = attempts_valid[attempts_valid["learner"].isin(manifest["train_learners"])]
    val_attempts_all = attempts_valid[attempts_valid["learner"].isin(manifest["val_learners"])]

    global_freq = fit_global_class_freq(train_attempts, "outcome_code", N_OUTCOME)
    per_problem_table = fit_per_problem_base_rate(train_attempts, "outcome_code", "problem_id",
                                                   N_OUTCOME, global_freq, SHRINKAGE_K)
    base_rate_probs_val = per_problem_base_rate_probs(val_attempts_all, per_problem_table, global_freq, "problem_id")
    base_rate_target_val = val_attempts_all["outcome_code"].to_numpy()
    base_rate_auc_all, n_br_all = binary_auc(base_rate_probs_val, base_rate_target_val)

    ev_sorted = attempts_valid.sort_values(["learner", "t"], kind="mergesort")
    prefix_probs_all = learner_prefix_probs(ev_sorted, global_freq, shrinkage_k=SHRINKAGE_K, class_col="outcome_code")
    val_row_mask = ev_sorted["learner"].isin(manifest["val_learners"]).to_numpy()
    prefix_auc_all, n_pf_all = binary_auc(prefix_probs_all[val_row_mask],
                                          ev_sorted.loc[val_row_mask, "outcome_code"].to_numpy())

    # attempt-1 stratum for both baselines: PER-EPISODE causal position 0 (first attempt,
    # by timestamp, WITHIN each episode) -- matches the model's own `attempt1_mask`
    # (built from `attempt_within_idx==0`, `cw_filter_batch.build_attempt_batch`'s own
    # per-episode causal index). A learner's GLOBAL first-ever event (`t`, which never
    # resets per episode) is a DIFFERENT, much smaller slice -- one row per learner rather
    # than one per episode -- and using it would degenerate the prefix baseline to exactly
    # 0.5000 (every learner contributing a single row at `n_prior=0`, `learner_prefix_probs`'s
    # own documented global-freq fallback), a real mechanical result for THAT slice but not
    # the attempt-1 stratum meant here.
    def episode_attempt1_mask(df: pd.DataFrame) -> np.ndarray:
        sorted_df = df.sort_values(["episode_id", "timestamp"], kind="mergesort")
        pos = sorted_df.groupby("episode_id", sort=False).cumcount()
        is_first = pd.Series((pos == 0).to_numpy(), index=sorted_df.index).reindex(df.index)
        return is_first.to_numpy()

    a1_mask_val_all = episode_attempt1_mask(val_attempts_all)
    base_rate_auc_a1, n_br_a1 = binary_auc(base_rate_probs_val, base_rate_target_val, a1_mask_val_all)

    a1_mask_prefix = episode_attempt1_mask(ev_sorted) & val_row_mask
    prefix_auc_a1, n_pf_a1 = binary_auc(prefix_probs_all[a1_mask_prefix],
                                        ev_sorted.loc[a1_mask_prefix, "outcome_code"].to_numpy())

    logger.heartbeat(step=4, force=True, stage="baselines scored",
                     model_auc_all=round(model_auc_all, 4), base_rate_auc_all=round(base_rate_auc_all, 4),
                     prefix_auc_all=round(prefix_auc_all, 4))

    # -- checkpoint + per-row predictions (both kept so the reported number can be
    # reproduced and re-analysed; codec_path points at the committed artifact above,
    # never the gitignored raw cache) --
    item_split_hash = _sha256(REPO / "splits" / "cw_s19_item_split.csv")
    learner_split_hash = _sha256(REPO / "splits" / "cw_learner_split.csv")
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = CHECKPOINT_DIR / "cw_4f_shipped.pt"
    torch.save({
        "model_state_dict": {
            "attempt_encoder": r["model"]["attempt_encoder"].state_dict(),
            "filt": r["model"]["filt"].state_dict(),
        },
        "heads_state_dict": {
            "outcome_head": r["model"]["outcome_head"].state_dict(),
            "y_ever_head": r["model"]["y_ever_head"].state_dict(),
        },
        "architecture": {"body": BODY, "item_encoder": ITEM_ENCODER, "codec_dim": CODEC_DIM,
                         "latent_dim": LATENT_DIM, "episode_hidden": 64, "filter_hidden": 64,
                         "attempt_proj_dim": 32, "attempt_hidden": 64, "summary_dim": 16,
                         "n_classes": N_OUTCOME},
        "codec_path": str(CODEC_ARTIFACT_PATH.relative_to(REPO)),
        "item_split_sha256": item_split_hash, "learner_split_sha256": learner_split_hash,
        "train_regime": "WITH-H (all 50 items, official-split pooled train-pool)",
        "seed": SEED, "epochs": args.epochs,
        "headline_nll_all_attempts": r["all_attempts"]["nll"], "headline_auc_all_attempts": model_auc_all,
        "git_sha": logger.git_sha,
    }, ckpt_path)

    pred_df = pd.DataFrame({
        "learner_row": np.arange(len(val["target"])), "problem_id": val["problem_id"],
        "attempt_within_idx": val["attempt_within_idx"], "target_outcome_code": val["target"],
        "p_compile_error": val["probs_real"][:, OUTCOME_CLASSES.index("COMPILE_ERROR")],
        "p_clean_full": val["probs_real"][:, CLEAN_FULL_IDX],
    })
    pred_path = REPO / "reports" / "cw_4f_predictions.csv.gz"
    pred_df.to_csv(pred_path, index=False, compression="gzip")
    logger.heartbeat(step=5, force=True, stage="checkpoint + predictions saved",
                     checkpoint=str(ckpt_path.relative_to(REPO)), predictions=str(pred_path.relative_to(REPO)),
                     n_rows=len(pred_df))

    passed = (model_auc_all == model_auc_all and base_rate_auc_all == base_rate_auc_all
             and prefix_auc_all == prefix_auc_all and n_all > 0)
    logger.summary(status="ok" if passed else "gate-fail", model_auc_all=model_auc_all)

    meta = {"run_id": logger.run_id, "git_sha": logger.git_sha, "body": BODY, "seed": SEED,
           "codec_dim": CODEC_DIM, "item_encoder": ITEM_ENCODER, "latent_dim": LATENT_DIM,
           "checkpoint": str(ckpt_path.relative_to(REPO)), "predictions": str(pred_path.relative_to(REPO)),
           "codec_artifact": str(CODEC_ARTIFACT_PATH.relative_to(REPO))}
    metrics = {
        "P(clean-full) AUC, all-attempts": {"model": model_auc_all, "n": n_all,
                                            "per_problem_base_rate": base_rate_auc_all, "n_baseline": n_br_all,
                                            "learner_prefix_shrinkage": prefix_auc_all, "n_baseline_pf": n_pf_all},
        "P(clean-full) AUC, attempt-1 stratum": {"model": model_auc_a1, "n": n_a1,
                                                 "per_problem_base_rate": base_rate_auc_a1, "n_baseline": n_br_a1,
                                                 "learner_prefix_shrinkage": prefix_auc_a1, "n_baseline_pf": n_pf_a1},
        "reference: Y_ever AUC (episode-level, different quantity)": 0.831,
        "reference: FA S1 headline AUC (different corpus/task, comparable in kind only)": 0.786,
        "reference: FA success base rate (~80% unaided-correct) vs CW clean-full rate (~24%)": None,
        "n_outcome_invalid (should be 0 on this corpus)": n_invalid,
    }
    notes = [
        f"**Model: {model_auc_all:.4f} (all-attempts, n={n_all}) / {model_auc_a1:.4f} (attempt-1, n={n_a1})** "
        f"vs baselines -- per-problem base rate {base_rate_auc_all:.4f}/{base_rate_auc_a1:.4f}, "
        f"learner-prefix-with-shrinkage {prefix_auc_all:.4f}/{prefix_auc_a1:.4f}. Both baselines fit on "
        "TRAIN only, scored on the identical binary `clean-full` task; all three are reported "
        "unconditionally, every stratum.",
        "**DISCLOSURE, required on any FA<->CW AUC sentence:** different corpora and different tasks -- "
        "FA is binary correctness on maths items at ~80% unaided-correct, CW is attempt-level full-"
        "credit at ~24%. AUC's base-rate insensitivity makes the pairing LEGIBLE (both metrics answer "
        "'does the model rank outcomes correctly', regardless of the very different base rates); it "
        "does NOT make the numbers interchangeable -- comparable in kind, never in value. CW's own "
        "`Y_ever` AUC (0.831, GRU) is a THIRD, still different quantity (episode-level ever-solves, "
        "not attempt-level full-credit) and must not be conflated with either number above.",
        f"**Training:** ONE seed=0 model at the final config (width 32/64/64, "
        "`codec_dim`=64, `item_encoder`='linear', `latent_dim`=32, selected by the codec-width, "
        f"item-encoder and belief-dimension ablations) on the WITH-H arm. Saved `{ckpt_path.relative_to(REPO)}` "
        f"(standalone schema, incl. `git_sha`) and `{pred_path.relative_to(REPO)}` ({len(pred_df)} rows); "
        "later CodeWorkout evaluations can load this checkpoint "
        "instead of retraining.",
        f"**New committed artifact `{CODEC_ARTIFACT_PATH.relative_to(REPO)}`** (50 items x 64-dim, "
        "rank=69/k=5 through the shared, committed transform) -- mirrors the mathematics "
        "`fa_item_codec_dim64.npz` exactly. `rebuild_problem_codec_at_rank`'s raw-state cache is "
        "gitignored, so without this small committed artifact, `codec_path` in the checkpoint above "
        "would point at nothing recoverable; an untracked or regenerated codec silently breaks reuse.",
        "**Attempt-1 stratum definition**: the model's own `attempt1_mask` (within-episode "
        "`attempt_within_idx==0`); both baselines use the MATCHING per-episode definition -- causal "
        "position 0 by timestamp WITHIN each episode, never a learner's global first-ever event "
        "(`t`, which does not reset per episode). The global-`t` definition would degenerate the "
        "prefix baseline's attempt-1 AUC to exactly 0.5000 (one row per learner, `n_prior=0` for all "
        "of them -> `learner_prefix_probs`' own documented global-freq fallback) -- a real mechanical "
        "result for that slice, but the wrong slice here. Hence the grouping on `episode_id` before scoring.",
    ]

    out_path = REPO / "reports" / "cw_4f_clean_full_auc.md"
    gate_report(out_path, "CodeWorkout classifier: P(clean-full) AUC against both baselines",
               meta, metrics, passed,
               "Model AUC reported on all-attempts AND attempt-1 stratum, AND both baselines' AUC "
               "reported on the identical binary task at both strata (never the model alone).",
               notes)

    print("\n" + "=" * 90)
    print("CodeWorkout classifier -- P(clean-full) AUC")
    print("=" * 90)
    print(f"{'':<28}{'all-attempts':>14}{'attempt-1':>14}")
    print(f"{'model':<28}{model_auc_all:>14.4f}{model_auc_a1:>14.4f}")
    print(f"{'per-problem base rate':<28}{base_rate_auc_all:>14.4f}{base_rate_auc_a1:>14.4f}")
    print(f"{'learner-prefix-shrinkage':<28}{prefix_auc_all:>14.4f}{prefix_auc_a1:>14.4f}")
    print(f"\ncw_4f_clean_full_auc: {'PASS' if passed else 'FAIL'}")
    print(f"report: reports/cw_4f_clean_full_auc.md")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
