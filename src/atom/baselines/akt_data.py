"""Data prep for the AKT baseline: FA event tuples -> pyKT AKT's tensor contract.

pyKT's AKT (`pykt.models.akt.AKT`) expects, per pyKT's own `train_model.py::model_forward`,
three aligned `[batch, seq_len]` LongTensors:
  - concepts  (their "q_data"):   a knowledge-component id at each step
  - pids      (their "pid_data"): the actual problem/question id at each step
  - responses (their "target"):   0/1 correctness at each step
`model(concepts, responses, pids)` returns `preds[:, t]` = P(correct at step t | steps
0..t-1's full (concept, pid, response) history + step t's own concept/pid, no peek at step
t's own response) -- this no-peek property is AKT's own published encoder/decoder split
(their "blocks_1" encodes q+a causally, "blocks_2" cross-attends q only), not something this
module re-derives; `tests/test_akt_baseline.py` spot-checks it survives OUR batch
construction (a flipped last-step response must not move position-0's prediction).

FA has no single-concept-per-problem field -- Skills.csv is problem_id -> skill_id,
MANY-TO-MANY (10.5%, 355/3368 problems, map to more than one skill; measured, not assumed).
This module uses the deterministic minimum skill_id per problem as a single canonical
concept -- a disclosed simplification (a small, real information loss on that 10.5%, not a
data bug), not the multi-concept row-expansion pyKT's own preprocessing does for its
official benchmark CSVs, which exists to reproduce pyKT's exact published leaderboard
numbers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch


def build_concept_map(skills: pd.DataFrame) -> dict[int, int]:
    """problem_id -> dense 0-indexed concept id, via the minimum skill_id per problem
    (deterministic, disclosed simplification for the 10.5% of multi-skill problems)."""
    min_skill = skills.groupby("problem_id")["skill_id"].min()
    uniq_skills = sorted(min_skill.unique())
    skill_to_idx = {s: i for i, s in enumerate(uniq_skills)}
    return {int(pid): skill_to_idx[int(s)] for pid, s in min_skill.items()}


def build_pid_map(problem_ids) -> dict[int, int]:
    """problem_id -> dense 0-indexed question id."""
    uniq = sorted({int(p) for p in problem_ids})
    return {p: i for i, p in enumerate(uniq)}


def check_coverage(problem_ids, concept_map: dict[int, int], pid_map: dict[int, int]) -> None:
    """Raise loudly if any problem_id lacks a concept or pid mapping."""
    missing_c = {int(p) for p in problem_ids} - set(concept_map)
    missing_p = {int(p) for p in problem_ids} - set(pid_map)
    if missing_c or missing_p:
        raise RuntimeError(
            f"{len(missing_c)} problem_ids missing a concept mapping, "
            f"{len(missing_p)} missing a pid mapping (expected 0 -- maps must be built from "
            f"the full FA problem/skill universe, not a subsample)"
        )


def subsample_by_event_budget(ev: pd.DataFrame, budget: int, seed: int = 0) -> pd.DataFrame:
    """Deterministic learner-order subsample of `ev`, capped at approximately `budget` total
    events (e.g. 100k interactions for a local pilot). Takes WHOLE learners in a
    seeded-shuffled order, stopping once the cumulative event count would reach budget --
    never truncates a learner's own history mid-stream here (that's build_akt_batch's job;
    truncating in both places would double-count against the budget)."""
    learners = sorted(ev["learner"].unique())
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(learners))
    counts = ev.groupby("learner").size()
    picked: list = []
    total = 0
    for i in order:
        learner = learners[i]
        n = int(counts[learner])
        if total > 0 and total + n > budget:
            continue  # skip (not break): keep sampling smaller learners toward the budget
        picked.append(learner)
        total += n
        if total >= budget:
            break
    return ev[ev["learner"].isin(picked)]


def build_akt_batch(ev: pd.DataFrame, concept_map: dict[int, int], pid_map: dict[int, int],
                     max_seq_len: int, device: str) -> dict[str, object]:
    """ev -> padded [N, T] tensors (concepts, pids, responses, valid_mask) + lengths [N].
    T = min(longest learner history in `ev`, max_seq_len); learners longer than max_seq_len
    keep their MOST RECENT max_seq_len events (right-aligned truncation, not pyKT's official
    multi-window chunking of long histories -- a disclosed pilot simplification; see
    `build_akt_batch_windowed` for the full-scale path).

    Same vectorised groupby + fancy-index scatter pattern as filter/batch.py's
    build_padded_batch (a row-by-row loop is roughly 9 minutes vs 0.77 s vectorised).
    """
    ev = ev.sort_values(["learner", "t"], kind="mergesort")
    ev = ev.groupby("learner", sort=False).tail(max_seq_len)
    # Re-index t to 0..len-1 WITHIN the (possibly truncated) kept rows, so padding lines up.
    ev = ev.assign(t_local=ev.groupby("learner", sort=False).cumcount())

    learners = sorted(ev["learner"].unique())
    learner_idx_of = {u: i for i, u in enumerate(learners)}
    n = len(learners)
    lengths_by_learner = ev.groupby("learner").size()
    t_max = int(lengths_by_learner.max())

    row_n = ev["learner"].map(learner_idx_of).to_numpy()
    row_t = ev["t_local"].to_numpy()
    row_concept = ev["problem_id"].map(concept_map).to_numpy()
    row_pid = ev["problem_id"].map(pid_map).to_numpy()

    concepts = np.zeros((n, t_max), dtype=np.int64)
    pids = np.zeros((n, t_max), dtype=np.int64)
    responses = np.zeros((n, t_max), dtype=np.int64)
    valid_mask = np.zeros((n, t_max), dtype=bool)

    concepts[row_n, row_t] = row_concept
    pids[row_n, row_t] = row_pid
    responses[row_n, row_t] = ev["correct"].to_numpy()
    valid_mask[row_n, row_t] = True
    lengths = lengths_by_learner.reindex(learners).to_numpy().astype(np.int64)

    def to_t(a, dtype):
        return torch.tensor(a, dtype=dtype, device=device)

    return {
        "learners": learners,
        "concepts": to_t(concepts, torch.long),
        "pids": to_t(pids, torch.long),
        "responses": to_t(responses, torch.long),
        "valid_mask": to_t(valid_mask, torch.bool),
        "lengths": to_t(lengths, torch.long),
    }


# -- Multi-window chunking (full-scale run) ----------------------------------
#
# pyKT's OWN documented mechanism for a learner history longer than the model's window
# (read directly from the installed package, `pykt/preprocess/split_datasets.py`, not
# invented here) is `generate_sequences`: split each learner's chronological history into
# CONSECUTIVE, NON-OVERLAPPING chunks of at most `maxlen` events; each chunk becomes its own
# independent training/eval row (AKT carries no state across chunk boundaries -- a later
# chunk's position 0 is treated exactly like a fresh, zero-history sequence, same as pyKT's
# own convention of never scoring position 0). The last, possibly-short chunk is
# right-padded. This -- NOT `generate_window_sequences` (pyKT's OTHER mechanism: a sliding,
# one-new-position-at-a-time window that reruns the full model once per position beyond
# `maxlen`) -- is what pyKT itself uses to build TRAIN and VALID data (see that module's
# `main()`: `generate_sequences(splitdf, ...)` feeds the K-Fold train/valid split).
# `generate_window_sequences` is reserved by pyKT for their OWN test-set leaderboard
# evaluation, where maximising context per scored position matters more than compute cost --
# it is O(sequence length) separate full-context forward passes per long learner, which at
# FA's full scale (some learners run past 400 events) would be prohibitively expensive for a
# baseline run that is not attempting to reproduce a published leaderboard number. The
# non-overlapping-chunk mechanism is therefore the right precedent for the full train+val
# run, in place of the pilot's simple right-truncation.
def chunk_learner_history(ev: pd.DataFrame, max_seq_len: int, min_seq_len: int = 3) -> pd.DataFrame:
    """Split each learner's chronological history into non-overlapping windows of at most
    `max_seq_len` events (pyKT's own `generate_sequences` mechanism). Adds `window` (0-indexed
    chunk index within the learner) and `chunk_id` (a composite `"{learner}::w{window}"` key)
    columns; every OTHER column and its values are preserved unchanged, and no event is
    reordered.

    A learner's LAST chunk can be short (a genuine remainder, not a full window); if it has
    fewer than `min_seq_len` events it is dropped entirely -- pyKT's own `dropnum` behaviour,
    same default (3) as pyKT's `generate_sequences`/`main()`. A chunk of exactly `max_seq_len`
    events is NEVER dropped by this rule even if `max_seq_len < min_seq_len` in some
    degenerate test config -- only a genuine (shorter-than-a-full-window) remainder can be
    "too small to keep".

    No event is silently discarded otherwise: every earlier chunk of a long learner's history
    is kept in full (unlike the pilot's right-truncation, which kept only the most recent
    `max_seq_len` events and dropped everything older).
    """
    ev = ev.sort_values(["learner", "t"], kind="mergesort").reset_index(drop=True)
    pos_in_learner = ev.groupby("learner", sort=False).cumcount()
    ev = ev.assign(window=(pos_in_learner // max_seq_len).astype(int))

    last_window = ev.groupby("learner", sort=False)["window"].transform("max")
    chunk_size = ev.groupby(["learner", "window"], sort=False)["t"].transform("size")
    is_remainder = chunk_size < max_seq_len
    drop_mask = (ev["window"] == last_window) & is_remainder & (chunk_size < min_seq_len)

    ev = ev.loc[~drop_mask].copy()
    ev["chunk_id"] = ev["learner"].astype(str) + "::w" + ev["window"].astype(str)
    return ev


def chunking_stats(ev: pd.DataFrame, chunked: pd.DataFrame, max_seq_len: int) -> dict:
    """Diagnostics for the run report: how much of `chunk_learner_history`'s job actually
    mattered on this data (fraction of learners needing >1 window, events dropped as tiny
    remainders, total chunk count) -- the same disclosure discipline as the pilot's
    truncated_frac notes, for the mechanism that replaces truncation at full scale."""
    raw_counts = ev.groupby("learner").size()
    total_learners = int(raw_counts.shape[0])
    multi_window_learners = int((raw_counts > max_seq_len).sum())
    raw_events = int(len(ev))
    kept_events = int(len(chunked))
    n_chunks = int(chunked["chunk_id"].nunique())
    return {
        "total_learners": total_learners,
        "multi_window_learners": multi_window_learners,
        "multi_window_frac": (multi_window_learners / total_learners) if total_learners else 0.0,
        "raw_events": raw_events,
        "kept_events": kept_events,
        "dropped_events": raw_events - kept_events,
        "dropped_frac": ((raw_events - kept_events) / raw_events) if raw_events else 0.0,
        "n_chunks": n_chunks,
    }


def build_akt_batch_windowed(ev: pd.DataFrame, concept_map: dict[int, int], pid_map: dict[int, int],
                             max_seq_len: int, device: str, min_seq_len: int = 3) -> dict[str, object]:
    """Full-scale replacement for build_akt_batch's pilot-only right-truncation: chunk each
    learner's FULL history via `chunk_learner_history` (pyKT's own multi-window mechanism,
    see that function's docstring), then feed the result through `build_akt_batch` UNCHANGED
    -- each chunk simply becomes what `build_akt_batch` already treats as one "learner" row,
    so its existing tested padding/masking logic (a row shorter than max_seq_len gets
    right-padded and masked) needs no modification at all to handle a ragged last chunk.

    Returns the same keys as `build_akt_batch`, PLUS `orig_learners`: the true learner id
    each batch row's chunk came from (batch["learners"] itself holds chunk_ids, e.g.
    "<learner>::w1", not the bare learner id -- needed for any downstream per-learner
    analysis, not needed for a pooled AUC gate).
    """
    chunked = chunk_learner_history(ev, max_seq_len, min_seq_len)
    orig_learner_by_chunk = (
        chunked.drop_duplicates("chunk_id").set_index("chunk_id")["learner"].to_dict()
    )
    ev_for_batch = chunked.drop(columns=["learner"]).rename(columns={"chunk_id": "learner"})
    batch = build_akt_batch(ev_for_batch, concept_map, pid_map, max_seq_len, device)
    batch["orig_learners"] = [orig_learner_by_chunk[c] for c in batch["learners"]]
    return batch


def build_support_mask(ev: pd.DataFrame, max_seq_len: int, batch_learners: list, t_max: int,
                       min_seq_len: int = 3, col: str = "support_code") -> np.ndarray:
    """An [N, T] grid of `support_code` (the FA 4-class support label), aligned
    cell-for-cell to `build_akt_batch_windowed`'s own chunk-id/position grid over the SAME
    (ev, max_seq_len) -- by re-running `chunk_learner_history` (the identical deterministic
    call `build_akt_batch_windowed` makes internally) rather than a second implementation
    that could silently drift out of alignment. `batch_learners` must be the `learners` list
    that call returned (chunk_ids, sorted). Returns int8, -1 where no chunk row exists (never
    a real support code, which is 0-3) -- e.g. `(mask == 0)` selects the support=NONE stratum.

    `col`: which integer column to lay onto the grid (default `support_code`).
    `scripts/fa_01c_item_holdout.py` passes a 0/1 item-stratum column to get a
    train-item-vs-heldout-item grid on the IDENTICAL chunk/position alignment, which is
    the reason to reuse this function rather than write a parallel one.
    """
    chunked = chunk_learner_history(ev, max_seq_len, min_seq_len)
    chunked = chunked.sort_values(["chunk_id", "t"], kind="mergesort")
    chunked = chunked.assign(t_local=chunked.groupby("chunk_id", sort=False).cumcount())
    learner_idx_of = {u: i for i, u in enumerate(batch_learners)}
    out = np.full((len(batch_learners), t_max), -1, dtype=np.int8)
    row_n = chunked["chunk_id"].map(learner_idx_of).to_numpy()
    row_t = chunked["t_local"].to_numpy()
    out[row_n, row_t] = chunked[col].to_numpy()
    return out
