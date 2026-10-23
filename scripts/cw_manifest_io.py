"""Shared helpers for loading the CodeWorkout training manifest.

Used by `cw_4f_clean_full_auc`, `cw_4g_3_item_holdout_check`, `cw_4g_codec_rank_item_mlp`,
`cw_4i_latent_dim_isolation`, `cw_sigma`, `cw_sim_head_refresh` and `cw_sim_tv_head_refresh`.
`load_manifest` joins the train and validation attempts, the episode-level table, the
code/diff codec vector lookups and the problem-level item codec; lockbox learners are
excluded.
"""
from __future__ import annotations
import pathlib
import sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from atom.codec.transform import CodecTransform  # noqa: E402
from atom.data.cw_manifest import (  # noqa: E402
    attach_code_diff_hashes,
    build_codec_vector_lookup,
    build_episode_level_table,
    fit_global_class_freq,
    fit_per_problem_base_rate,
    load_attempts_with_episodes,
    load_hash_maps,
    per_problem_base_rate_probs,
)
from atom.data.cw_outcome_label import N_OUTCOME, OUTCOME_CLASSES, outcome_label  # noqa: E402
from atom.simulator.cw_4_train_loop import (  # noqa: E402
    CODEC_DIM,
    CODEC_K,
    LATENT_DIM,
    SHRINKAGE_K,
    aggregate_across_seeds,
    run_one_seed_body,
)
REPO = pathlib.Path(__file__).resolve().parents[1]


def load_manifest(args) -> dict:
    """Everything downstream needs, joined once: pooled S19+F19 train and validation
    attempts (the release's official learner split, `splits/cw_learner_split.csv`), the
    episode-level table, code/diff codec vector lookups and the problem-level item codec,
    over all 50 items (the item holdout is handled by
    `scripts/cw_4c_2_did_and_prefix_curve.py`, not here). **The 123-subject F19-test
    lockbox is excluded entirely**: it is evaluated once, at the end, and this function
    never sees it."""
    attempts = load_attempts_with_episodes(raw_dir=args.raw_dir)  # pools S19 + F19, both releases kept
    n_before_split_filter = len(attempts)

    split = pd.read_csv(REPO / "splits" / "cw_learner_split.csv")
    train_learners = set(split.loc[split["split"] == "train", "learner"])
    val_learners = set(split.loc[split["split"] == "val", "learner"])
    lockbox_learners = set(split.loc[split["split"] == "lockbox", "learner"])
    attempts = attempts[attempts["learner"].isin(train_learners | val_learners)].reset_index(drop=True)

    code_map, diff_map = load_hash_maps(args.code_map, args.diff_map)
    attempts, hash_stats = attach_code_diff_hashes(attempts, code_map, diff_map)
    hash_stats["n_attempts_before_lockbox_exclusion"] = n_before_split_filter
    hash_stats["n_lockbox_learners_excluded"] = len(lockbox_learners)

    labels = outcome_label(attempts["compile_result"], attempts["score_frac"])
    attempts = attempts.assign(outcome_code=labels["outcome_code"], outcome_valid=labels["outcome_valid"])

    transform = CodecTransform.load(REPO / "transforms" / "problem_mean_L15.npz")
    code_hashes_needed = set(attempts["code_hash"].dropna())
    diff_hashes_needed = set(attempts.loc[attempts["diff_hash"].notna(), "diff_hash"])
    # `rank=50` is pinned explicitly. Without it the call inherits `transform.apply`'s
    # default `rank=transform.k_max`; the shared transform has k_max=300, which would
    # give 295-dim vectors and a shape mismatch in `code_vec[row_n, row_t] = ...`.
    # `rank=50` gives the production codec (45 dims after stripping k=5 components)
    # regardless of the transform's k_max.
    code_of_hash = build_codec_vector_lookup(code_hashes_needed, args.codealone_meta,
                                             args.codealone_states, transform, k=CODEC_K, rank=50)
    diff_of_hash = build_codec_vector_lookup(diff_hashes_needed, args.diff_meta,
                                             args.diff_states, transform, k=CODEC_K, rank=50)

    episodes = build_episode_level_table(attempts)

    item = np.load(REPO / "artifacts" / "cw_problem_codec_mean_L15.npz", allow_pickle=True)
    problem_codec = item["codec"].astype(np.float32)
    pid_to_idx = {int(p): i for i, p in enumerate(item["problem_id"])}

    # Every attempt's learner must fall in exactly one of the three frozen buckets -- a
    # learner present in `attempts` but absent from the split file would silently drop
    # out of both train and val with no error; catch it here instead.
    unassigned = set(attempts["learner"].unique()) - train_learners - val_learners
    if unassigned:
        raise ValueError(f"{len(unassigned)} learner(s) in the attempt table are not in "
                         f"the frozen split file (e.g. {sorted(unassigned)[:3]})")

    return {
        "attempts": attempts, "episodes": episodes, "code_of_hash": code_of_hash,
        "diff_of_hash": diff_of_hash, "problem_codec": problem_codec, "pid_to_idx": pid_to_idx,
        "train_learners": train_learners, "val_learners": val_learners, "hash_stats": hash_stats,
    }
