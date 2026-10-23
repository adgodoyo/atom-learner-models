"""Batch building for the belief filter: turn a slice of event-tuple rows into padded
[N, T, ...] tensors that the filter and heads consume. Kept in one place so that every
training and evaluation script builds identically shaped batches; only the choice of
learners differs between callers.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch


def load_production_codec(path: str | Path, expected_dim: int) -> tuple[np.ndarray, dict[int, int]]:
    """Load the frozen production item codec. Returns (codec_matrix [n_items, dim], pid_to_idx)."""
    npz = np.load(Path(path), allow_pickle=True)
    codec_matrix = npz["codec"].astype(np.float32)
    pid_to_idx = {int(p): i for i, p in enumerate(npz["problem_id"])}
    if codec_matrix.shape[1] != expected_dim:
        raise AssertionError(f"codec dim {codec_matrix.shape[1]} != expected {expected_dim}")
    return codec_matrix, pid_to_idx


def check_coverage(problem_ids, pid_to_idx: dict[int, int]) -> None:
    """Raise loudly if any problem_id lacks a codec vector (the codec is built to cover every item)."""
    missing = set(int(p) for p in problem_ids) - set(pid_to_idx)
    if missing:
        raise RuntimeError(f"{len(missing)} problem_ids missing from the production codec "
                            f"(expected 0: the codec covers every item): {sorted(missing)[:5]}...")


def build_padded_batch(ev: pd.DataFrame, codec_matrix: np.ndarray, pid_to_idx: dict[int, int],
                       device: str, include_delta_days: bool = False) -> dict[str, object]:
    """ev: event rows for the selected learners (any chronological order; sorted here by t).
    Returns [N, T, ...] padded tensors + valid_mask [N, T] + lengths [N]; column index t
    equals the true within-learner event index (right-padding only, per learner).

    Fully vectorised (one groupby + fancy-index scatter, no per-learner Python loop): at
    full scale (4000 learners x 1.4M rows) a naive `ev[ev["learner"] == learner]` filter
    run once per learner is O(n_learners x n_rows) and takes minutes before training starts.

    `include_delta_days` (off by default, so callers that do not need it are unaffected): also
    returns "delta_days" [N, T] -- elapsed real time since the learner's OWN previous event,
    0.0 at each learner's first position (no history to decay). Computed from `ev["timestamp"]`
    via the SAME sort + scatter indices as everything else, so it is guaranteed aligned with
    item_codec/correct/etc. at every cell, not a second, possibly-drifting derivation.
    """
    learners = sorted(ev["learner"].unique())
    learner_idx_of = {u: i for i, u in enumerate(learners)}
    n = len(learners)
    lengths_by_learner = ev.groupby("learner").size()
    t_max = int(lengths_by_learner.max())
    codec_dim = codec_matrix.shape[1]

    ev_sorted = ev.sort_values(["learner", "t"], kind="mergesort")
    row_n = ev_sorted["learner"].map(learner_idx_of).to_numpy()
    row_t = ev_sorted["t"].to_numpy()
    row_codec_idx = ev_sorted["problem_id"].map(pid_to_idx).to_numpy()

    item_codec = np.zeros((n, t_max, codec_dim), dtype=np.float32)
    correct = np.zeros((n, t_max), dtype=np.float32)
    hint_bin = np.zeros((n, t_max), dtype=np.float32)
    saw_answer = np.zeros((n, t_max), dtype=np.float32)
    valid_mask = np.zeros((n, t_max), dtype=bool)

    item_codec[row_n, row_t] = codec_matrix[row_codec_idx]
    correct[row_n, row_t] = ev_sorted["correct"].to_numpy()
    hint_bin[row_n, row_t] = (ev_sorted["hint_count"].to_numpy() > 0).astype(np.float32)
    saw_answer[row_n, row_t] = ev_sorted["saw_answer"].to_numpy()
    valid_mask[row_n, row_t] = True
    lengths = lengths_by_learner.reindex(learners).to_numpy().astype(np.int64)

    to_t = lambda a, dtype=torch.float32: torch.tensor(a, dtype=dtype, device=device)  # noqa: E731
    out = {
        "learners": learners,
        "item_codec": to_t(item_codec),
        "correct": to_t(correct),
        "hint_bin": to_t(hint_bin),
        "saw_answer": to_t(saw_answer),
        "valid_mask": to_t(valid_mask, torch.bool),
        "lengths": to_t(lengths, torch.long),
    }

    if include_delta_days:
        prev_ts = ev_sorted.groupby("learner")["timestamp"].shift(1)
        delta_seconds = (ev_sorted["timestamp"] - prev_ts).dt.total_seconds()
        delta_days_row = (delta_seconds / 86400.0).fillna(0.0).to_numpy()
        delta_days = np.zeros((n, t_max), dtype=np.float32)
        delta_days[row_n, row_t] = delta_days_row
        out["delta_days"] = to_t(delta_days)

    return out


def build_padded_batch_v2(ev: pd.DataFrame, codec_matrix: np.ndarray, pid_to_idx: dict[int, int],
                          device: str, include_time_min: bool = False) -> dict[str, object]:
    """v2 counterpart of `build_padded_batch`, for the rebuilt FoundationalASSIST manifest schema.

    `ev` must already be restricted to `has_valid_time == True` rows (the manifest's `t` is
    only defined there; the caller decides the split/cell, this function just pads it). Score-
    missing rows are KEPT as real sequence positions (their gap/order structure matters: a
    missing score is a labelling gap, not a missing event) but flagged via
    `score_missing` / `label_valid` rather than coerced to 0 -- the same "flag, don't impute"
    discipline `fa_manifest.py` established for the raw manifest, now carried into the
    padded-tensor world the filter trains on. `correct[n,t]` holds a 0.0 PLACEHOLDER at
    score-missing cells (never used unmasked: `label_valid` excludes them from every loss/
    AUC computation, and `episode_v2.build_prev_feat_v2` independently zeroes `prev_correct`
    wherever `score_missing` was set at t-1).

    `include_time_min` (off by default, so callers that do not need it are bit-for-bit
    unaffected) additionally returns `time_min` [N, T]: elapsed MINUTES since each learner's
    own first event, 0.0 at t=0. The event transformer's attention bias needs the pairwise elapsed time between any
    TWO of a learner's events, so it needs a per-position clock rather than v1's
    `include_delta_days` step-to-step difference -- pairwise gaps are then differences of
    this one column, which cannot drift out of alignment with a separately accumulated
    version. Computed through the SAME sort + scatter indices as every other channel, so it
    is aligned with item_codec/correct/etc. at every cell by construction.
    """
    learners = sorted(ev["learner"].unique())
    learner_idx_of = {u: i for i, u in enumerate(learners)}
    n = len(learners)
    lengths_by_learner = ev.groupby("learner").size()
    t_max = int(lengths_by_learner.max())
    codec_dim = codec_matrix.shape[1]

    ev_sorted = ev.sort_values(["learner", "t"], kind="mergesort")
    row_n = ev_sorted["learner"].map(learner_idx_of).to_numpy()
    row_t = ev_sorted["t"].to_numpy().astype(np.int64)
    row_codec_idx = ev_sorted["problem_id"].map(pid_to_idx).to_numpy()

    item_codec = np.zeros((n, t_max, codec_dim), dtype=np.float32)
    correct = np.zeros((n, t_max), dtype=np.float32)
    score_missing = np.zeros((n, t_max), dtype=bool)
    hint_bin = np.zeros((n, t_max), dtype=np.float32)
    saw_answer = np.zeros((n, t_max), dtype=np.float32)
    valid_mask = np.zeros((n, t_max), dtype=bool)

    score_missing_col = ev_sorted["score_missing"].to_numpy(dtype=bool)
    score_filled = ev_sorted["score"].astype("float32").fillna(0.0).to_numpy()

    item_codec[row_n, row_t] = codec_matrix[row_codec_idx]
    correct[row_n, row_t] = score_filled
    score_missing[row_n, row_t] = score_missing_col
    hint_bin[row_n, row_t] = (ev_sorted["hint_count"].to_numpy() > 0).astype(np.float32)
    saw_answer[row_n, row_t] = ev_sorted["saw_answer"].to_numpy().astype(np.float32)
    valid_mask[row_n, row_t] = True
    lengths = lengths_by_learner.reindex(learners).to_numpy().astype(np.int64)

    to_t = lambda a, dtype=torch.float32: torch.tensor(a, dtype=dtype, device=device)  # noqa: E731
    valid_mask_t = to_t(valid_mask, torch.bool)
    score_missing_t = to_t(score_missing, torch.bool)
    out = {
        "learners": learners,
        "item_codec": to_t(item_codec),
        "correct": to_t(correct),
        "score_missing": score_missing_t,
        "label_valid": valid_mask_t & ~score_missing_t,
        "hint_bin": to_t(hint_bin),
        "saw_answer": to_t(saw_answer),
        "valid_mask": valid_mask_t,
        "lengths": to_t(lengths, torch.long),
    }

    if include_time_min:
        first_ts = ev_sorted.groupby("learner")["timestamp"].transform("min")
        elapsed_min_row = ((ev_sorted["timestamp"] - first_ts).dt.total_seconds() / 60.0
                           ).clip(lower=0.0).to_numpy()
        time_min = np.zeros((n, t_max), dtype=np.float32)
        time_min[row_n, row_t] = elapsed_min_row
        out["time_min"] = to_t(time_min)

    return out


def mask_from_rows(rows: pd.DataFrame, learners: list[str], t_max: int, device: str) -> torch.Tensor:
    """Convert a (learner, t) row set (e.g. calib_eval_split's eval_rows) into a boolean
    [N, T] mask aligned to a padded batch built from the SAME `learners` order -- the bridge
    between the frozen protocol's DataFrame output and the padded-tensor world the filter
    operates in.
    """
    learner_of_row = {u: i for i, u in enumerate(learners)}
    mask = np.zeros((len(learners), t_max), dtype=bool)
    idx = rows["learner"].map(learner_of_row).to_numpy()
    t = rows["t"].to_numpy()
    mask[idx, t] = True
    return torch.tensor(mask, dtype=torch.bool, device=device)
