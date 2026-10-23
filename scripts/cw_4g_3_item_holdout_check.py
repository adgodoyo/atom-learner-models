#!/usr/bin/env python
"""Item-holdout check for the CodeWorkout codec-width ablation: does a wider codec help on items never trained on?

The codec-width grid in cw_4g_codec_rank_item_mlp.py is measured on the learner holdout,
with every cell trained on all 50 items. With only ~50 distinct items, a wide codec
vector can fingerprint items individually without resolving any transferable meaning.
This script strips the 7 held-out items (`atom.data.cw_challenge_split.HELD_OUT_ITEMS`)
from train learners only (`atom.data.cw_manifest.strip_held_out_items`, the WITHOUT-H arm
of cw_4c_2_did_and_prefix_curve.py), trains the same codec_dim grid {45, 64, 128, 256}
(linear item encoder only, seed 0), and scores the P(clean-full) AUC on val attempts on
those 7 items, which no training learner's history or target included. A rank gain that
survives there is at least partly transferable; one that vanishes points to small-pool
fingerprinting. The non-held-out stratum and full val are reported for context (the
WITHOUT-H arm's full-val AUC is expected to sit below the all-items grid's).
Inputs: code/diff hash maps and codec caches (flags below), the CodeWorkout splits.
Output: `reports/cw_4g_3_item_holdout_check.md`. Serves Appendix A (codec width on CodeWorkout).

    python scripts/cw_4g_3_item_holdout_check.py \\
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
from atom.eval.metrics import auc  # noqa: E402
from atom.eval.report import gate_report  # noqa: E402
from atom.simulator.cw_4_train_loop import CODEC_K, run_one_seed_body  # noqa: E402
from atom.utils.logging import RunLogger  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[1]
BODY = "gru"
SEED = 0
K_FIXED = CODEC_K  # 5, held fixed while the rank varies
RANKS = [50, 69, 133, 261]  # codec_dim = rank - 5 -> {45, 64, 128, 256}
ITEM_ENCODER = "linear"  # isolating the rank axis; the item-encoder axis is measured separately
HELD = set(HELD_OUT_ITEMS)
TRANSFORM_PATH = REPO / "transforms" / "problem_mean_L15.npz"
CLEAN_FULL_IDX = OUTCOME_CLASSES.index("CLEAN_FULL")

# Full-val AUCs of the linear-encoder row of the all-items (WITH-H) codec-width grid, quoted
# only as WITH-H vs WITHOUT-H context in this report's notes -- never recomputed here.
CW_4G_WITH_H_LINEAR_AUC = {45: 0.807398061413771, 64: 0.8067680246206321, 128: 0.8131156432903827, 256: 0.8162285017846824}


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
    args = ap.parse_args()

    device = "cpu"
    logger = RunLogger(config={"task": "cw_4g_3_item_holdout_check", "body": BODY, "seed": SEED,
                               "ranks": RANKS, "item_encoder": ITEM_ENCODER, "k_fixed": K_FIXED,
                               "held_out_items": sorted(HELD)},
                       data_version="CSEDM-S19+F19-challenge-train-pool+val")

    logger.heartbeat(step=0, force=True, stage="loading base manifest + building the WITHOUT-H arm")
    base_manifest = load_manifest(args)
    without_h_attempts = strip_held_out_items(base_manifest["attempts"], HELD, base_manifest["train_learners"])
    without_h_episodes = build_episode_level_table(without_h_attempts)
    n_ep_with_h_train = base_manifest["episodes"][
        base_manifest["episodes"]["learner"].isin(base_manifest["train_learners"])]["episode_id"].nunique()
    n_ep_without_h_train = without_h_episodes[
        without_h_episodes["learner"].isin(base_manifest["train_learners"])]["episode_id"].nunique()
    n_episodes_removed = n_ep_with_h_train - n_ep_without_h_train

    code_hashes_needed = set(without_h_attempts["code_hash"].dropna())
    diff_hashes_needed = set(without_h_attempts.loc[without_h_attempts["diff_hash"].notna(), "diff_hash"])
    logger.heartbeat(step=1, force=True, stage="WITHOUT-H arm ready", n_episodes_removed_train=n_episodes_removed,
                     removed_frac=n_episodes_removed / max(1, n_ep_with_h_train))

    transform = CodecTransform.load(TRANSFORM_PATH)
    assert transform.k_max >= max(RANKS), (
        f"transform k_max={transform.k_max} too small for the widest rank {max(RANKS)} -- "
        "regenerate the shared transform with a larger k_max first (cw_4g_codec_rank_item_mlp.py "
        "reads the same committed transform).")

    results = {}
    for rank in RANKS:
        codec_dim = rank - K_FIXED
        assert codec_dim + K_FIXED == rank, "k must stay fixed while rank varies"

        t0 = time.time()
        code_of_hash = build_codec_vector_lookup(code_hashes_needed, args.codealone_meta,
                                                 args.codealone_states, transform, k=K_FIXED, rank=rank)
        diff_of_hash = build_codec_vector_lookup(diff_hashes_needed, args.diff_meta,
                                                 args.diff_states, transform, k=K_FIXED, rank=rank)
        problem_codec, pid_to_idx = rebuild_problem_codec_at_rank(rank, transform=transform)
        cell_manifest = {**base_manifest, "attempts": without_h_attempts, "episodes": without_h_episodes,
                         "code_of_hash": code_of_hash, "diff_of_hash": diff_of_hash,
                         "problem_codec": problem_codec, "pid_to_idx": pid_to_idx}
        codec_build_s = time.time() - t0

        t0 = time.time()
        r = run_one_seed_body(BODY, SEED, cell_manifest, args, device,
                              dims={"codec_dim": codec_dim}, item_encoder=ITEM_ENCODER)
        wall = time.time() - t0

        probs, target, pid = r["val"]["probs_real"], r["val"]["target"], r["val"]["problem_id"]
        held_mask = np.isin(pid, list(HELD))
        held_auc, held_n = cell_auc_on_mask(probs, target, held_mask)
        nonheld_auc, nonheld_n = cell_auc_on_mask(probs, target, ~held_mask)
        full_auc, full_n = cell_auc_on_mask(probs, target, np.ones_like(held_mask, dtype=bool))

        results[f"codec{codec_dim}"] = {
            "codec_dim": codec_dim, "rank": rank, "codec_build_s": round(codec_build_s, 1),
            "train_wall_s": round(wall, 1),
            "auc_held_out_items": held_auc, "n_held_out_eval": held_n,
            "auc_non_held_out_items": nonheld_auc, "n_non_held_out_eval": nonheld_n,
            "auc_full_val_without_h_arm": full_auc, "n_full_val": full_n,
            "cw_4g_with_h_full_sample_auc": CW_4G_WITH_H_LINEAR_AUC[codec_dim],
        }
        logger.heartbeat(step=0, force=True, stage=f"cell codec_dim={codec_dim} done",
                         auc_held=round(held_auc, 4) if held_auc == held_auc else held_auc,
                         auc_full=round(full_auc, 4) if full_auc == full_auc else full_auc,
                         n_held=held_n, wall_s=round(wall, 1))

    held_aucs = [r["auc_held_out_items"] for r in results.values()]
    held_valid = all(a == a for a in held_aucs)  # all finite (no nan from a degenerate stratum)
    held_gain_45_to_256 = (held_aucs[-1] - held_aucs[0]) if held_valid else float("nan")
    full_aucs = [r["auc_full_val_without_h_arm"] for r in results.values()]
    full_gain_45_to_256 = full_aucs[-1] - full_aucs[0]

    passed = len(results) == len(RANKS) and all(r["n_held_out_eval"] > 0 for r in results.values())
    logger.summary(status="ok" if passed else "gate-fail", n_cells=len(results),
                  held_gain_45_to_256=held_gain_45_to_256, full_gain_45_to_256=full_gain_45_to_256)

    meta = {"run_id": logger.run_id, "git_sha": logger.git_sha, "body": BODY, "seed": SEED,
           "k_fixed": K_FIXED, "held_out_items": sorted(HELD), "n_episodes_removed_train": n_episodes_removed,
           "n_train_learners": len(base_manifest["train_learners"]), "n_val_learners": len(base_manifest["val_learners"])}
    metrics = {
        "grid (codec_dim), WITHOUT-H arm: AUC on P(clean-full), held-out-items stratum / non-held-out stratum / full val": results,
        "rank gain 45->256, held-out-items stratum": held_gain_45_to_256,
        "rank gain 45->256, full val (this WITHOUT-H arm)": full_gain_45_to_256,
        "rank gain 45->256, all-items (WITH-H) grid full-sample fit (quoted, not recomputed)":
            CW_4G_WITH_H_LINEAR_AUC[256] - CW_4G_WITH_H_LINEAR_AUC[45],
    }
    notes = [
        f"**WITHOUT-H arm construction**, as in cw_4c_2_did_and_prefix_curve.py: {n_episodes_removed} train-pool episodes "
        f"({n_episodes_removed / max(1, n_ep_with_h_train):.1%} of {n_ep_with_h_train}) stripped from "
        "TRAIN learners' histories AND targets (`strip_held_out_items`); val learners are untouched, so "
        "their attempts on the 7 held-out items are scored by a model that never trained on those items "
        "from ANY learner -- the eval population this check needs.",
        f"**Held-out-items stratum, rank gain 45->256:** {held_gain_45_to_256:+.4f}"
        + (" (finite at every rank)" if held_valid else " -- AT LEAST ONE RANK HAD A DEGENERATE/EMPTY "
          "held-out-item stratum (nan); read the per-cell `n_held_out_eval` before trusting this number.")
        + f". **Full-val stratum, same arm:** {full_gain_45_to_256:+.4f}. **The all-items (WITH-H) grid's "
        f"full-sample gain (context, quoted not recomputed):** {CW_4G_WITH_H_LINEAR_AUC[256] - CW_4G_WITH_H_LINEAR_AUC[45]:+.4f}.",
        "**Reading, stated directly:** if the held-out-items gain is comparable to the full-val gain in "
        "this same arm, wider ranks resolve something that transfers to items never trained on -- not "
        "pool fingerprinting. If the held-out-items gain is much smaller, absent, or reversed while the "
        "full-val gain (within this same arm) still rises, that is fairly direct evidence the rank gain "
        "is concentrated in memorising the trained-on item pool rather than in transferable resolution.",
        "**Per-cell sample sizes matter here** -- the held-out-items stratum draws only from the ~7-item "
        "subset of val attempts, a much smaller population than the all-items grid's 21,849 rows; a "
        "small `n_held_out_eval` at any rank should widen how much this check's own AUC delta is trusted, "
        "reported per cell above rather than only as a headline gain.",
        f"**WITHOUT-H's own full-val AUC is expected to sit below the all-items (WITH-H) grid** (disclosed, "
        "not a bug -- the held-out items' training episodes are removed, the same pattern this arm shows "
        "in cw_4c_2_did_and_prefix_curve.py).",
        "**Scope, disclosed:** linear item-encoder only (isolating the rank axis from the "
        "item-encoder-MLP axis, measured separately); single seed=0, as in the other "
        "lever-search ablations, not the 5-seed protocol used for the DiD claim itself.",
    ]

    out_path = REPO / "reports" / "cw_4g_3_item_holdout_check.md"
    gate_report(out_path, "CodeWorkout codec width: item-holdout check (WITHOUT-H arm x codec rank)",
               meta, metrics, passed,
               "All 4 rank cells complete on the WITHOUT-H arm, non-empty held-out-item eval "
               "population each. This is a diagnostic (not a claim-bearing) gate -- the finding is "
               "in the reported held-out-items-vs-full-val gain comparison, not a pass/fail threshold.",
               notes)

    print("\n" + "=" * 100)
    print("Codec width, item-holdout check, WITHOUT-H arm")
    print("=" * 100)
    print(f"{'cell':<12}{'codec_dim':>10}{'auc_held':>10}{'n_held':>8}{'auc_full':>10}{'wall_s':>9}")
    for name, r in results.items():
        print(f"{name:<12}{r['codec_dim']:>10}{r['auc_held_out_items']:>10.4f}{r['n_held_out_eval']:>8}"
             f"{r['auc_full_val_without_h_arm']:>10.4f}{r['train_wall_s']:>9.1f}")
    print(f"\nheld-out-items gain 45->256: {held_gain_45_to_256:+.4f} | full-val gain (this arm): {full_gain_45_to_256:+.4f}")
    print(f"cw_4g_3_item_holdout_check: {'PASS' if passed else 'FAIL'}")
    print(f"report: reports/cw_4g_3_item_holdout_check.md")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
