"""Data-loading glue for the CodeWorkout hierarchical model.

Joins the attempts and episodes (`cw_events`, `cw_sessions`) to the frozen codec vector
caches, and provides the two closed-form baselines the primary gate must beat
(per-problem base rate, learner-prefix-with-shrinkage), plus the item-holdout
difference-in-differences (DiD) arms.

Nothing here trains anything -- this module only builds the manifest a training script
reads. Kept separate from `scripts/cw_4_hierarchical_model.py` so the join/baseline
logic is unit-testable on small synthetic frames, independent of the real ~195k-row
corpus and the real, remotely stored vector caches.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

from atom.data.cw_events import build_attempts, load_raw_tables
from atom.data.cw_sessions import sessionize

TAU_SESSION_MINUTES = 60.0  # The frozen session gap, fit on S19-train in `cw_sessions` -- never refit here.


def load_attempts_with_episodes(raw_dir=None, tau_session_minutes: float = TAU_SESSION_MINUTES) -> pd.DataFrame:
    """S19 + F19 attempts (no code text -- `include_code=False`, only `code_state_id` is
    needed to join the codec hash maps), sessionised at the frozen tau. One row per
    attempt, carrying `episode_id`, `learner`, `t` (per-learner causal attempt index,
    `cw_events.build_attempts`'s own convention), `problem_id`, `code_state_id`,
    `compile_result`, `score_frac`, `n_compile_errors`, `timestamp`, `release`.
    """
    frames = []
    for release in ["S19", "F19"]:
        main, _ = load_raw_tables(release, base_dir=raw_dir, include_code=False)
        a = build_attempts(main)
        a["release"] = release
        frames.append(a)
    attempts = pd.concat(frames, ignore_index=True)
    return sessionize(attempts, tau_session_minutes)


def load_hash_maps(code_map_path, diff_map_path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reads the codec stage's two small (no-text) lookup artifacts back into DataFrames:
    `code_state_id -> code_hash` and `code_state_id -> diff_hash` (episode-local diff
    against the previous attempt in the SAME episode; absent where there is no previous
    attempt, or the diff was empty -- see `cw_diffs.build_code_diffs`)."""
    c = np.load(code_map_path, allow_pickle=True)
    code_map = pd.DataFrame({
        "code_state_id": c["code_state_id"], "code_hash": c["code_hash"],
    })
    d = np.load(diff_map_path, allow_pickle=True)
    diff_map = pd.DataFrame({
        "code_state_id": d["code_state_id"], "diff_hash": d["diff_hash"],
    })
    return code_map, diff_map


def attach_code_diff_hashes(attempts: pd.DataFrame, code_map: pd.DataFrame,
                            diff_map: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Left-joins `code_hash` (required) and `diff_hash` (optional -- NaN where absent,
    `cw_filter_batch.build_attempt_batch`'s own "missing diff -> zero vector" convention)
    onto `attempts` via `code_state_id`.

    A small number of attempts have no `code_hash` at all (a disclosed gap in the codec
    stage: 10 / 195,205 rows with missing/whitespace-only code, excluded before hashing). Dropping
    ONLY those rows would break `AttemptEncoder`'s contiguous 0..T-1 causal indexing for
    every later attempt in the same episode, so this drops the WHOLE episode a bad row
    belongs to, not just the row -- a disclosed, deliberate over-exclusion, sized in the
    returned stats dict rather than silently absorbed.
    """
    ev = attempts.merge(code_map, on="code_state_id", how="left")
    ev = ev.merge(diff_map, on="code_state_id", how="left")

    bad_episodes = set(ev.loc[ev["code_hash"].isna(), "episode_id"].unique())
    n_bad_attempts = int(ev["code_hash"].isna().sum())
    clean = ev.loc[~ev["episode_id"].isin(bad_episodes)].reset_index(drop=True)
    stats = {
        "n_attempts_missing_code_hash": n_bad_attempts,
        "n_episodes_dropped_for_missing_code": len(bad_episodes),
        "n_attempts_before": int(len(attempts)),
        "n_attempts_after": int(len(clean)),
    }
    return clean, stats


def build_codec_vector_lookup(hashes_needed: set, meta_path, states_path, transform, k: int = 5,
                              batch_size: int = 20000, rank: int | None = None) -> dict:
    """Gathers the raw mean-pool@L15 rows the codec stage cached for exactly the hashes this
    run needs (never the full ~171k/~140k corpus -- an S19-only train+val slice touches
    a fraction of it), applies the FROZEN codec transform (`transform.apply(X, k=5)`,
    the SAME ruler `cw_problem_codec_mean_L15.npz` used for the problem-level codec), and
    returns `hash -> (rank-k,) float32` ready for `cw_filter_batch.build_attempt_batch`.

    Batched (not one `transform.apply` per hash): the raw cache is float16 on disk via
    `np.lib.format.open_memmap` (`SingleLayerVectorCache`, `cw_batch_extract.py`) -- a
    single `[N, 2560]` gather + one batched `apply` is both faster and avoids opening the
    memmap `len(hashes_needed)` separate times.

    `rank`: `None` (default) uses `transform.apply`'s own default, `rank=transform.k_max`.
    The codec-rank sweep (`cw_4g_codec_rank_item_mlp.py`) passes it explicitly (e.g. 69/133/261 for codec_dim 64/128/256 at k=5) so
    output width is decoupled from however large the shared transform's own k_max is.
    """
    meta = json.loads(Path(meta_path).read_text())
    row_of_hash = meta["row_of_hash"]
    states = np.lib.format.open_memmap(states_path, mode="r")

    hashes = sorted(h for h in hashes_needed if h in row_of_hash)
    missing = hashes_needed - set(hashes)
    if missing:
        raise KeyError(f"{len(missing)} hash(es) requested are absent from {meta_path} "
                       f"(e.g. {sorted(missing)[:3]}) -- caller's hash set must be a "
                       f"subset of what the codec stage actually encoded")

    out: dict = {}
    for i in range(0, len(hashes), batch_size):
        chunk = hashes[i:i + batch_size]
        rows = [row_of_hash[h] for h in chunk]
        raw = states[rows].astype(np.float64)
        coords = transform.apply(raw, k=k, rank=rank).astype(np.float32)
        for h, vec in zip(chunk, coords):
            out[h] = vec
    return out


def build_episode_level_table(attempts_with_episode: pd.DataFrame) -> pd.DataFrame:
    """One row per episode -- the columns `cw_filter_batch.build_episode_batch` needs
    (`episode_id`, `learner`, `problem_id`, `start_timestamp`) plus `y_ever` (max
    `correct` across the episode's attempts -- `cw_sessions.build_episode_table`'s
    convention, `y_ever=("correct","max")`, duplicated rather than reused because that
    function operates on the UNFILTERED attempt table and would disagree with whatever
    episodes survived `attach_code_diff_hashes`'s drop). `y_ever` is the target for the
    `Y_ever`-at-episode-start auxiliary (`cw_train_step.y_ever_loss_for_batch`)."""
    g = attempts_with_episode.sort_values(["episode_id", "timestamp"], kind="mergesort").groupby(
        "episode_id", sort=True)
    out = g.agg(learner=("learner", "first"), problem_id=("problem_id", "first"),
               start_timestamp=("timestamp", "min"), y_ever=("correct", "max")).reset_index()
    out["y_ever"] = out["y_ever"].astype(bool)
    return out


# -- Baselines the gate requires: per-problem base rate AND learner-prefix-with-shrinkage,
# -- both fit on TRAIN only. ---

def fit_global_class_freq(train_df: pd.DataFrame, class_col: str, n_classes: int) -> np.ndarray:
    """Population class frequency, fit on TRAIN only -- the fallback both baselines
    below shrink toward. Same shape/contract as `support_baselines.fit_global_support_freq`,
    duplicated (not imported) because that function hardcodes `SUPPORT_CLASSES`' count as
    its default and this module's caller always passes `n_classes` explicitly anyway."""
    counts = np.bincount(train_df[class_col].to_numpy().astype(int), minlength=n_classes).astype(float)
    total = counts.sum()
    if total == 0:
        raise ValueError("fit_global_class_freq: train_df has zero valid rows")
    return counts / total


def fit_per_problem_base_rate(train_df: pd.DataFrame, class_col: str, problem_col: str,
                              n_classes: int, global_freq: np.ndarray, shrinkage_k: float = 5.0
                              ) -> dict:
    """Per-problem class-frequency baseline (the "beat per-problem base rate" gate): a constant classifier per `problem_id`, fit on TRAIN attempts only,
    shrunk toward `global_freq` with pseudo-count `shrinkage_k` -- same shrinkage
    convention as `support_baselines.learner_prefix_probs`, applied over problems instead
    of learner-history position. A problem absent from `train_df` (should not happen for
    the learner split, which is by LEARNER not by item, but guarded rather than assumed) falls
    back to `global_freq` at lookup time, never a materialised entry here.
    """
    table: dict = {}
    for pid, sub in train_df.groupby(problem_col):
        counts = np.bincount(sub[class_col].to_numpy().astype(int), minlength=n_classes).astype(float)
        table[int(pid)] = (counts + shrinkage_k * global_freq) / (counts.sum() + shrinkage_k)
    return table


def per_problem_base_rate_probs(df: pd.DataFrame, table: dict, global_freq: np.ndarray,
                                problem_col: str) -> np.ndarray:
    """Applies `fit_per_problem_base_rate`'s table to any split (train/val/...)."""
    return np.stack([table.get(int(pid), global_freq) for pid in df[problem_col]])


# -- Item-holdout DiD arms: the WITHOUT-H arm never
# -- sees the held-out items, in either direction; the volume-matched arm removes the same
# -- amount of training data but item-blind, isolating the "less data" confound from the
# -- "these specific items" effect. Both operate on whole EPISODES -- CW's own episode
# -- boundary (`cw_sessions.sessionize`) is (learner, problem_id, tau-gap), so a problem
# -- is never split across two episodes and this never needs partial-episode surgery. -----

def strip_held_out_items(attempts: pd.DataFrame, held_out_items, learners: set) -> pd.DataFrame:
    """Drop every attempt whose `problem_id` is in `held_out_items` AND whose `learner` is
    in `learners` (the TRAINING population only -- eval populations must keep held-out
    items, that is what the DiD measures). A **target-only** removal would leak (the
    held-out outcome would still enter as history when predicting later events), so this
    removes the attempt from both roles at once by dropping the row entirely ("removed
    from targets AND histories for every learner")."""
    is_target_learner = attempts["learner"].isin(learners)
    is_held = attempts["problem_id"].isin(set(held_out_items))
    return attempts[~(is_target_learner & is_held)].reset_index(drop=True)


def volume_matched_removal(attempts: pd.DataFrame, learners: set, n_episodes_to_remove: int,
                           seed: int = 0) -> pd.DataFrame:
    """The third DiD arm: remove `n_episodes_to_remove` whole episodes at
    RANDOM (item-blind -- unlike `strip_held_out_items`, no problem_id is targeted) from
    `learners`' own attempts, matching the WITHOUT-H arm's exact training-volume reduction
    so a DiD reading of "held-out items hurt" isn't confounded by "less data hurts,
    independent of which items". Deterministic (`random.Random(seed)`), whole episodes
    only (same reasoning as `strip_held_out_items` -- CW's episode boundary never splits a
    problem across two episodes)."""
    target = attempts[attempts["learner"].isin(learners)]
    other = attempts[~attempts["learner"].isin(learners)]
    episode_ids = sorted(target["episode_id"].unique())
    if n_episodes_to_remove > len(episode_ids):
        raise ValueError(f"cannot remove {n_episodes_to_remove} episodes from only "
                         f"{len(episode_ids)} available")
    rng = random.Random(seed)
    removed = set(rng.sample(episode_ids, n_episodes_to_remove))
    kept_target = target[~target["episode_id"].isin(removed)]
    return (pd.concat([other, kept_target], ignore_index=True)
           .sort_values(["learner", "timestamp"], kind="mergesort").reset_index(drop=True))


def strict_cell_learners(full_attempts: pd.DataFrame, held_out_items, candidate_learners: set) -> set:
    """Learners (within `candidate_learners`) who touch >=1 held-out item AND have ZERO
    of their OWN events after their chronologically-last held-out-item attempt --
    computed on the FULL (unstripped) attempts, since this is a property of a learner's
    real history, independent of which arm's model later scores them. Used by
    `cw_4c_2_did_and_prefix_curve.py`; `strict_cell_row_mask` below restricts the scored
    rows to the held-out items of these learners."""
    held = set(held_out_items)
    a = full_attempts[full_attempts["learner"].isin(candidate_learners)].sort_values(
        ["learner", "timestamp"], kind="mergesort").reset_index(drop=True)
    if len(a) == 0:
        return set()
    a["_pos"] = a.groupby("learner", sort=False).cumcount()
    n_total = a.groupby("learner")["_pos"].size()
    held_mask = a["problem_id"].isin(held)
    if not held_mask.any():
        return set()
    last_held_pos = a.loc[held_mask].groupby("learner")["_pos"].max()
    events_after = (n_total.reindex(last_held_pos.index) - 1 - last_held_pos).astype(int)
    return set(events_after[events_after == 0].index)


def strict_cell_row_mask(problem_id: np.ndarray, learner: np.ndarray, prior_same_problem_count: np.ndarray,
                         attempt_within_idx: np.ndarray, held_out_items, qualifying_learners: set) -> dict:
    """Row-level mask for the strict cell. Restricting *learners*
    (`strict_cell_learners`, above) is not enough: scoring ALL of a qualifying learner's
    attempts, rather than only their held-item rows, inflates the cell far beyond what a
    7-item cell can contain. The cell is therefore defined at ROW level: a row qualifies iff its learner is in
    `qualifying_learners` AND its `problem_id` is held.

    Reports **attempt-1** (within-episode causal position 0) and **first-ever-exposure**
    (`prior_same_problem_count == 0` -- this learner's first attempt at this SPECIFIC
    held problem, across ANY episode) as two SEPARATE strata: they diverge whenever a
    learner revisits a held item in a later, separate episode, in which case attempt-1
    of that later episode is a within-episode first attempt but NOT the learner's
    first-ever exposure to the problem.

    Inputs are the arrays `row_identity`/`cw_4_train_loop.evaluate_on_learners` already
    emit (`problem_id`, `learner`, `prior_same_problem_count`, `attempt_within_idx`), so
    a caller applies this mask directly to an eval dict -- no re-join needed."""
    held = set(held_out_items)
    qualifying_mask = np.isin(learner, list(qualifying_learners))
    held_mask = np.isin(problem_id, list(held))
    base = qualifying_mask & held_mask
    return {
        "base_mask": base,
        "attempt1_mask": base & (attempt_within_idx == 0),
        "first_exposure_mask": base & (prior_same_problem_count == 0),
    }


def attach_prior_attempt_code(attempts: pd.DataFrame) -> pd.DataFrame:
    """The terminal-code production head's own within-episode ordering,
    kept independent of `cw_filter_batch.build_attempt_batch`'s tensor-shaped
    `attempt_within_idx` (same DEFINITION -- causal 0-based position within the episode
    -- but this one lives on the DataFrame the prompt encoder reads row-by-row, so the
    two never need to be reconciled through a shared function only one of them calls).

    Adds `attempt_within_idx` (int, groupby-cumcount within `episode_id`) and
    `prior_code` (the SAME episode's immediately preceding attempt's `code` column,
    `None`/NaN at `attempt_within_idx == 0` -- an explicit empty/START row, which `cw_alt.encode_example` reads as `prior_code=None` and encodes via the
    fixed `PRIOR_CODE_START_TEXT`, never as an accidentally-empty string).

    Caller must have already sorted `attempts` so that within each `episode_id`, rows
    are in causal (timestamp) order -- exactly `cw_sessions.sessionize`'s own output
    order (sorted `learner, timestamp`, which is `episode_id`-contiguous by
    construction: an episode is a maximal same-learner same-problem run)."""
    out = attempts.copy()
    out["attempt_within_idx"] = out.groupby("episode_id", sort=False).cumcount()
    out["prior_code"] = out.groupby("episode_id", sort=False)["code"].shift(1)
    return out


# The OTHER required gate baseline, learner-prefix-with-shrinkage, needs no CW-specific
# wrapper at all: `atom.eval.support_baselines.learner_prefix_probs(ev, global_freq,
# shrinkage_k, class_col="outcome_code")` is already fully generic over the class column
# (shared with the FA outcome label) and CW's `attempts` frame already carries `learner` +
# a per-learner-chronological `t` in the exact shape that module expects -- callers use
# it directly rather than through a duplicated wrapper.
