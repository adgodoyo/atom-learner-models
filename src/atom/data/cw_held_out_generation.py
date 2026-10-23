"""Validation learners' first attempts on the 7 frozen held-out CW problems.

The problems are `atom.data.cw_challenge_split.HELD_OUT_ITEMS`. The stratum asks "does
the simulator write plausible code for a problem it never trained on?", the generative
analogue of the CodeWorkout classifier's held-out-item AUC cell.

`infra.cw_5_full_run.load_cw5_manifest` EXCLUDES these 7 problems from targets AND
histories entirely (by design -- training must never see them), so its own `attempts`
table cannot supply this stratum's rows. This module builds them from the RAW,
unfiltered attempts table instead (real learners really did attempt these problems in
S19/F19; they are simply outside the generative simulator's training/evaluation scope),
then joins each row to the belief its OWN learner already had -- via the already-computed,
already-tested dense val-prefix belief export (value-join format,
`scripts/cw_sim_planner_handoff.py`'s own output) -- rather than re-deriving a second
belief-computation path on a differently-shaped manifest.

The first-attempt selection below is a deliberate, small duplicate of
`infra.cw_5_full_run.select_first_attempts_on_headline`'s own logic (same
groupby/earliest-by-`t` shape) rather than an import -- this module lives in
`src/atom/data/`, the infra-independent library layer; `infra/` scripts import FROM
`src/atom`, never the reverse, so pulling a 4-line selection helper across that
boundary is not worth inverting the dependency direction for.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def load_held_out_problem_attempts(val_learners: set, raw_dir: str | None = None) -> pd.DataFrame:
    """Raw attempts restricted to `HELD_OUT_ITEMS` + `val_learners`, one row per
    (learner, problem_id) -- the EARLIEST such attempt by `t` (causal per-learner
    attempt index), mirroring `infra/cw_5_full_run.py::select_first_attempts_on_headline`'s
    own "first attempt" convention. `prior_code` is `None` for every returned row by
    construction (a learner's first-ever attempt on a problem starts a fresh episode,
    `attempt_within_idx == 0`) -- there is no meaningful "prior submission on this exact
    problem" for a never-attempted-before item, matching the "first attempt" framing."""
    from atom.data.cw_challenge_split import HELD_OUT_ITEMS
    from atom.data.cw_events import build_attempts, load_raw_tables
    from atom.data.cw_manifest import TAU_SESSION_MINUTES, attach_prior_attempt_code
    from atom.data.cw_outcome_label import outcome_label
    from atom.data.cw_sessions import sessionize

    frames = []
    for release in ("S19", "F19"):
        main, code_states = load_raw_tables(release, base_dir=raw_dir, include_code=True)
        a = build_attempts(main, code_states)
        a["release"] = release
        frames.append(a)
    attempts = pd.concat(frames, ignore_index=True)
    attempts = sessionize(attempts, TAU_SESSION_MINUTES)
    attempts = attempts[attempts["learner"].isin(val_learners)].reset_index(drop=True)
    attempts = attempts[attempts["problem_id"].isin(HELD_OUT_ITEMS)].reset_index(drop=True)
    return select_first_attempts(attempts)


def select_first_attempts(attempts: pd.DataFrame) -> pd.DataFrame:
    """Given ALREADY-FILTERED (val learners x held-out problems) raw attempts, adds
    `attempt_within_idx`/`prior_code`/`outcome_code`, drops outcome-invalid rows, and
    keeps exactly one row per (learner, problem_id) -- the earliest by `t`. Split out
    from `load_held_out_problem_attempts` so the selection/labelling logic is testable
    without real CodeWorkout raw tables."""
    from atom.data.cw_manifest import attach_prior_attempt_code
    from atom.data.cw_outcome_label import outcome_label

    empty_cols = list(attempts.columns) + ["attempt_within_idx", "prior_code", "outcome_code"]
    if len(attempts) == 0:
        return pd.DataFrame(columns=empty_cols)

    attempts = attach_prior_attempt_code(attempts)
    labels = outcome_label(attempts["compile_result"], attempts["score_frac"])
    attempts = attempts.assign(outcome_code=labels["outcome_code"], outcome_valid=labels["outcome_valid"])
    attempts = attempts[attempts["outcome_valid"]].reset_index(drop=True)
    if len(attempts) == 0:
        return pd.DataFrame(columns=empty_cols)

    attempts = attempts.sort_values(["learner", "problem_id", "t"], kind="mergesort")
    return attempts.groupby(["learner", "problem_id"], sort=False).head(1).reset_index(drop=True)


def join_nearest_prior_belief(held_out_attempts: pd.DataFrame, dense_beliefs: dict,
                              episodes_table: pd.DataFrame) -> pd.DataFrame:
    """Attaches `mu`/`log_sigma` to each held-out attempt: the belief from that SAME
    learner's most recent episode (per `dense_beliefs`, a value-join dict --
    `learner`/`episode_id`/`mu`/`log_sigma`) whose `start_timestamp` is <= the held-out
    attempt's own `timestamp` -- the causal belief a well-formed filter forward pass
    would already have produced going INTO this attempt. A learner-scoped
    nearest-backward join (`pd.merge_asof(..., by="learner", direction="backward")`),
    never a global nearest-timestamp join, which would silently borrow another
    learner's belief.

    Rows with NO qualifying prior episode (the held-out attempt precedes every episode
    `dense_beliefs` has for that learner) are DROPPED -- counted in the returned
    DataFrame's own `.attrs["n_dropped_no_prior_belief"]` / `.attrs["n_input_rows"]`,
    never silently included with a placeholder belief."""
    # `dense_beliefs["episode_id"]` (an npz-persisted string array, the value-join
    # convention) and `episodes_table["episode_id"]` (an int64 groupby key,
    # `build_episode_level_table`'s own output) are the SAME identity in different
    # dtypes -- coerced to string on both sides before joining, never left to pandas'
    # own type-coercion guess (which refuses outright rather than risk a wrong join,
    # exactly the caution this codebase's own join canaries ask for).
    ep = episodes_table[["episode_id", "start_timestamp"]].drop_duplicates("episode_id").copy()
    ep["episode_id"] = ep["episode_id"].astype(str)
    beliefs = pd.DataFrame({
        "learner": np.asarray(dense_beliefs["learner"]),
        "belief_episode_id": np.asarray(dense_beliefs["episode_id"]).astype(str),
        "mu": list(np.asarray(dense_beliefs["mu"])),
        "log_sigma": list(np.asarray(dense_beliefs["log_sigma"])),
    }).merge(ep.rename(columns={"episode_id": "belief_episode_id", "start_timestamp": "belief_start_timestamp"}),
            on="belief_episode_id", how="left", validate="many_to_one")
    if beliefs["belief_start_timestamp"].isna().any():
        raise AssertionError(
            "some dense-belief episode_ids are absent from episodes_table -- "
            "the belief join is incomplete, refusing to proceed on a partial join")

    n_input = len(held_out_attempts)
    if n_input == 0:
        out = held_out_attempts.copy()
        out["mu"] = pd.Series(dtype=object)
        out["log_sigma"] = pd.Series(dtype=object)
        out["belief_episode_id"] = pd.Series(dtype=object)
        out["belief_start_timestamp"] = pd.Series(dtype="float64")
        out.attrs["n_dropped_no_prior_belief"] = 0
        out.attrs["n_input_rows"] = 0
        return out

    left = held_out_attempts.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    right = beliefs.sort_values("belief_start_timestamp", kind="mergesort").reset_index(drop=True)
    merged = pd.merge_asof(left, right, left_on="timestamp", right_on="belief_start_timestamp",
                           by="learner", direction="backward")
    n_dropped = int(merged["belief_episode_id"].isna().sum())
    out = merged[merged["belief_episode_id"].notna()].reset_index(drop=True)
    out.attrs["n_dropped_no_prior_belief"] = n_dropped
    out.attrs["n_input_rows"] = n_input
    return out
