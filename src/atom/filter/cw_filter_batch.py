"""Batch construction bridging the CodeWorkout episode/attempt tables (`cw_sessions.py`) to
the two-timescale filter (thesis Section 5.3): `AttemptEncoder`'s per-episode attempt
padding, and the cross-episode filter's per-learner episode padding. Mirrors
`filter/batch.py`'s vectorised (groupby + scatter, no per-row Python loop) construction;
a naive per-learner filter costs minutes before training starts at CW's scale (47,758
episodes over 195,205 attempts).

Two batches, two different row units:
  `build_attempt_batch`: one row per EPISODE, T = that batch's own max attempt count.
      Feeds `AttemptEncoder` directly.
  `build_episode_batch`: one row per LEARNER, T = that batch's own max episode count.
      Feeds the cross-episode filter (`filter/factory.py::build_filter`) -- `item_codec`
      is the CURRENT episode's problem codec, RAW/unshifted (the filter body shifts it
      internally, exactly like FA's own convention); `prev_feat` is the PREVIOUS
      episode's summary vector, shifted by this module via `episode_v2.
      build_prev_item_codec` (reused as-is: it is dimension-agnostic, a generic
      zero-padded one-position shift over any `[N, T, D]` tensor, so no new shift logic
      is written here -- "import, don't fork" applies to a five-line helper too).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from atom.filter.episode_v2 import build_prev_item_codec


def build_attempt_batch(attempts: pd.DataFrame, code_vec_of_hash: dict[str, np.ndarray],
                        diff_vec_of_hash: dict[str, np.ndarray], code_dim: int, diff_dim: int,
                        device: str) -> dict[str, object]:
    """`attempts` must carry `episode_id`, `code_hash` (the code encoder's `text_hash(code)`,
    hash-keyed into `code_vec_of_hash`), `diff_hash` (same convention for
    `cw_diffs.build_code_diffs`'s diff text, NaN/absent at an episode's first attempt --
    those rows get an all-zero diff vector, matching `cw_diffs`' own "episode-first
    attempts get no diff row" convention), `compile_result`, `score_frac`,
    `n_compile_errors`. Rows must already be sorted (learner, timestamp) ascending
    WITHIN each episode (`cw_sessions.sessionize` output order) -- this function does not
    re-sort by time, only groups by `episode_id` preserving input order, so the caller
    owns chronological correctness.

    Returns a dict with `episode_ids` (the aligned row order every other output shares),
    `code_vec`/`diff_vec`: `[N_episodes, T_max, dim]`, `scalar_feats`:
    `[N_episodes, T_max, 4]`, `lengths`: `[N_episodes]` -- directly consumable by
    `AttemptEncoder.forward`.
    """
    from atom.filter.cw_attempt_encoder import build_scalar_feats

    ev = attempts.copy()
    ev["_attempt_idx"] = ev.groupby("episode_id", sort=False).cumcount()
    episode_ids = ev["episode_id"].drop_duplicates().tolist()
    ep_idx_of = {e: i for i, e in enumerate(episode_ids)}
    n = len(episode_ids)
    lengths_by_ep = ev.groupby("episode_id", sort=False).size()
    t_max = int(lengths_by_ep.max())

    row_n = ev["episode_id"].map(ep_idx_of).to_numpy()
    row_t = ev["_attempt_idx"].to_numpy()

    code_vec = np.zeros((n, t_max, code_dim), dtype=np.float32)
    diff_vec = np.zeros((n, t_max, diff_dim), dtype=np.float32)
    compile_ok = np.zeros((n, t_max), dtype=np.float32)
    score_frac = np.zeros((n, t_max), dtype=np.float32)
    n_compile_errors = np.zeros((n, t_max), dtype=np.float32)
    # attempt_idx, NOT attempt_frac: the raw, causal 0-based position.
    # A frac normalised by the episode's OWN total length would need that length before
    # the episode is over -- a label leak (see cw_attempt_encoder.py's module docstring
    # and build_scalar_feats).
    attempt_idx = np.zeros((n, t_max), dtype=np.float32)

    code_vecs_stacked = np.stack([code_vec_of_hash[h] for h in ev["code_hash"]])
    code_vec[row_n, row_t] = code_vecs_stacked

    has_diff = ev["diff_hash"].notna().to_numpy()
    if has_diff.any():
        diff_hashes = ev.loc[has_diff, "diff_hash"]
        diff_vecs_stacked = np.stack([diff_vec_of_hash[h] for h in diff_hashes])
        diff_vec[row_n[has_diff], row_t[has_diff]] = diff_vecs_stacked

    compile_ok[row_n, row_t] = (ev["compile_result"].to_numpy() == "Success").astype(np.float32)
    # score_frac: the raw platform-provided Score column (cw_events.py's
    # build_attempts, `runs["Score"]`) -- a per-attempt quantity with a FIXED, per-problem
    # test-suite denominator set by the platform's own grading, never anything computed
    # across an episode by this codebase, so it carries no episode-length leak.
    score_frac[row_n, row_t] = ev["score_frac"].to_numpy(dtype=np.float32)
    n_compile_errors[row_n, row_t] = ev["n_compile_errors"].to_numpy(dtype=np.float32)
    attempt_idx[row_n, row_t] = row_t.astype(np.float32)
    ep_lengths = lengths_by_ep.reindex(episode_ids).to_numpy()  # metadata for padding/gather
    # ONLY -- never fed into the per-position feature vector `x` the GRU recurs over, so
    # it cannot leak into a causal prefix (see the "lengths" return value below).

    to_t = lambda a, dtype=torch.float32: torch.tensor(a, dtype=dtype, device=device)  # noqa: E731
    scalar_feats = build_scalar_feats(
        to_t(compile_ok), to_t(score_frac), to_t(n_compile_errors), to_t(attempt_idx))

    return {
        "episode_ids": episode_ids,
        "code_vec": to_t(code_vec),
        "diff_vec": to_t(diff_vec),
        "scalar_feats": scalar_feats,
        "lengths": to_t(ep_lengths, torch.long),
        # One entry per row of the INPUT `attempts` frame (same order), for
        # `cw_train_step.losses_for_cw_hierarchical_batch`: `attempt_episode_row_idx`
        # indexes into THIS batch's own `episode_ids` order (matches `prefix_states`'
        # row 0, since `AttemptEncoder` consumes `code_vec`/`diff_vec` in that same
        # order); `attempt_within_idx` is the attempt's own causal position inside its
        # episode's prefix-state sequence -- exactly `row_n`/`row_t` above, kept rather
        # than recomputed by a caller so the two never drift out of sync with the
        # scatter above.
        "attempt_episode_row_idx": to_t(row_n, torch.long),
        "attempt_within_idx": to_t(row_t, torch.long),
    }


def build_episode_batch(episodes: pd.DataFrame, problem_codec: np.ndarray,
                        pid_to_idx: dict[int, int], episode_summary: torch.Tensor,
                        episode_ids_of_summary: list, device: str) -> dict[str, object]:
    """`episodes`: one row per episode (`cw_sessions.build_episode_table` schema: `episode_id`,
    `learner`, `problem_id`, `start_timestamp`). `episode_summary`/`episode_ids_of_summary`:
    `build_attempt_batch`'s own row order and `AttemptEncoder`'s pooled summary output for
    that order -- this function re-sorts into (learner, start_timestamp) order internally,
    so callers never need to align the two orderings by hand.

    Returns `item_codec` (RAW/unshifted, the CURRENT episode's problem codec -- the filter
    body shifts it internally), `prev_feat` (the PREVIOUS episode's summary, already
    shifted here), `time_min`, `valid_mask`, `learners` (row order), and `learner_idx`/
    `t_idx` (int arrays, one entry per input episode row, in `episodes`' OWN row order --
    the (learner, t) coordinates `gather_belief_for_batch` needs to read each episode's
    belief back out at).
    """
    summary_of_ep = {e: episode_summary[i] for i, e in enumerate(episode_ids_of_summary)}
    summary_dim = episode_summary.shape[-1]

    ev = episodes.sort_values(["learner", "start_timestamp"], kind="mergesort").reset_index(drop=True)
    learners = sorted(ev["learner"].unique())
    learner_idx_of = {u: i for i, u in enumerate(learners)}
    n = len(learners)
    lengths_by_learner = ev.groupby("learner", sort=False).size()
    t_max = int(lengths_by_learner.max())
    codec_dim = problem_codec.shape[1]

    row_n = ev["learner"].map(learner_idx_of).to_numpy()
    row_t = ev.groupby("learner", sort=False).cumcount().to_numpy()
    row_codec_idx = ev["problem_id"].map(pid_to_idx).to_numpy()

    item_codec = np.zeros((n, t_max, codec_dim), dtype=np.float32)
    valid_mask = np.zeros((n, t_max), dtype=bool)

    item_codec[row_n, row_t] = problem_codec[row_codec_idx]
    valid_mask[row_n, row_t] = True

    first_ts = ev.groupby("learner", sort=False)["start_timestamp"].transform("min")
    elapsed_min_row = ((ev["start_timestamp"] - first_ts).dt.total_seconds() / 60.0
                       ).clip(lower=0.0).to_numpy()
    time_min = np.zeros((n, t_max), dtype=np.float32)
    time_min[row_n, row_t] = elapsed_min_row

    to_t = lambda a, dtype=torch.float32: torch.tensor(a, dtype=dtype, device=device)  # noqa: E731

    # The scatter must stay inside the autograd graph. Routing it through numpy
    # (`torch.stack([...]).detach().cpu().numpy()` into an array, then back to a fresh
    # leaf tensor) severs the graph between `episode_summary` (AttemptEncoder.summary_head's
    # output) and every downstream loss, so `summary_head` receives zero gradient and the
    # cross-episode channel reads RANDOM PROJECTIONS of GRU states, not a learned summary
    # (an earlier implementation had exactly this defect). Here the
    # scatter is tensor-only: index into `episode_summary` directly (no detach), stack
    # (still differentiable), then advanced-index-assign into a `requires_grad`-free zero
    # tensor -- PyTorch records this as `IndexPutBackward`, so gradient flows back to
    # `episode_summary` and hence to `summary_head`'s parameters. The gradient canary test
    # checks that `summary_head`'s own parameters move: a check that a module received
    # SOME gradient cannot tell a live wire from a fresh leaf.
    summary_stacked = torch.stack([summary_of_ep[e] for e in ev["episode_id"]])  # still in the graph
    row_n_t = to_t(row_n, torch.long)
    row_t_t = to_t(row_t, torch.long)
    raw_summary_t = torch.zeros((n, t_max, summary_dim), dtype=summary_stacked.dtype, device=device)
    raw_summary_t[row_n_t, row_t_t] = summary_stacked
    return {
        "learners": learners,
        "item_codec": to_t(item_codec),
        "prev_feat": build_prev_item_codec(raw_summary_t),  # shifted here, per s1v2's contract
        "time_min": to_t(time_min),
        "valid_mask": to_t(valid_mask, torch.bool),
        "learner_idx": to_t(row_n, torch.long),
        "t_idx": to_t(row_t, torch.long),
    }
