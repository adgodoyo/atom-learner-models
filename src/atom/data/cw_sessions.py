"""CodeWorkout sessionisation, episode targets, and splits.

Builds on the attempt table (`cw_events.build_attempts`) to produce the FINAL episode
grouping -- `cw_events.build_problem_episodes` is explicitly a preliminary, problem-boundary
-only grouping for its hand-read sample, not this stage's target (see its docstring).

Episode boundary: a new episode starts when the learner changes, the
problem changes, or the gap since the learner's previous attempt on the SAME problem exceeds
`tau_session`. `tau_session` must be selected using only the S19-TRAIN release (never a
validation metric) -- `fit_tau_session` enforces this by taking exactly the gap series the
caller passes it and nothing else; the gate script is responsible for filtering to train
before calling it.

Episode targets are outcomes (Y_first/Y_ever/Y_terminal), scores (first/best/terminal),
effort (attempt count, duration, compile burden) and right-censoring (an episode
ending at the learner's very last observed event in the release cannot distinguish "genuinely
stopped" from "log ended here").

Splits provided here: S19 = train/val by learner (development), F19 = external cohort test
(temporal + new-cohort generalisation), plus an item-held-out split over the 50-problem
catalog. (The release's official learner split and the frozen 7-item holdout are in
`cw_challenge_split`.) Near-duplicate PROMPT TEXT would ideally be grouped before the item
split; no problem-statement text ships with the S19/F19 ProgSnap2
releases (checked: MainTable.csv, DatasetMetadata.csv, Subject.csv -- none carry it; only the
TIKTOC 17-problem subset has "prompt" text, via `cw.py`'s granular loader). The item split
here is therefore per-PROBLEM-ID, a disclosed simplification.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

EPISODE_COLUMNS = [
    "episode_id", "release", "learner", "problem_id", "n_attempts",
    "start_timestamp", "end_timestamp", "duration_seconds",
    "y_first", "y_ever", "y_terminal",
    "first_score", "best_score", "terminal_score",
    "n_compile_errors_total", "time_to_first_success",
    "is_right_censored",
]


def same_problem_gaps(attempts: pd.DataFrame) -> pd.Series:
    """Gap (minutes) between consecutive attempts by the SAME learner on the SAME problem --
    the series plotted (log scale) to look for a between-session valley. Only positive gaps (zero/negative would be a same-timestamp or out-of-order
    artifact, not a session boundary question).
    """
    ev = attempts.sort_values(["learner", "problem_id", "timestamp"], kind="mergesort")
    same_learner = ev["learner"].eq(ev["learner"].shift())
    same_problem = ev["problem_id"].eq(ev["problem_id"].shift())
    mask = same_learner & same_problem
    gap_min = (ev["timestamp"] - ev["timestamp"].shift()).dt.total_seconds() / 60.0
    return gap_min[mask & (gap_min > 0)]


def sessionize(attempts: pd.DataFrame, tau_session_minutes: float) -> pd.DataFrame:
    """Final episode grouping: new episode on learner change, problem change, or a
    same-learner/same-problem gap exceeding `tau_session_minutes`. Returns `attempts` sorted
    by (learner, timestamp) -- NOT (learner, problem, timestamp) -- with an `episode_id`
    column added, since episodes must stay chronological within a learner across problems.
    """
    ev = attempts.sort_values(["learner", "timestamp"], kind="mergesort").reset_index(drop=True)
    same_learner = ev["learner"].eq(ev["learner"].shift())
    same_problem = ev["problem_id"].eq(ev["problem_id"].shift())
    gap_min = (ev["timestamp"] - ev["timestamp"].shift()).dt.total_seconds() / 60.0
    within_session = same_learner & same_problem & (gap_min <= tau_session_minutes)
    new_episode = ~within_session
    ev["episode_id"] = (new_episode.cumsum() - 1).astype("int64")
    return ev


def audit_episode_partition(attempts_with_episode: pd.DataFrame) -> dict:
    """Sessionisation gate: 'no attempt in two episodes' + structural sanity, generalising
    `cw_events.assert_event_graph`-style audit to the tau-gapped grouping. All must be 0."""
    ev = attempts_with_episode
    bad_problem = int(ev.groupby("episode_id")["problem_id"].nunique().gt(1).sum())
    bad_learner = int(ev.groupby("episode_id")["learner"].nunique().gt(1).sum())
    dt = ev.groupby("episode_id")["timestamp"].diff()
    backwards_steps = int((dt < pd.Timedelta(0)).sum())
    n_attempts_covered = int(ev.groupby("episode_id").size().sum())
    return {
        "episodes_with_multiple_problem_ids": bad_problem,
        "episodes_with_multiple_learners": bad_learner,
        "backwards_timestamp_steps": backwards_steps,
        "attempts_not_covered_exactly_once": int(len(ev) - n_attempts_covered),
    }


def build_episode_table(attempts_with_episode: pd.DataFrame) -> pd.DataFrame:
    """One row per episode: targets, effort, censoring. A learner's LAST episode within a
    release is right-censored -- its end coincides with the observed log's end, so we
    cannot tell "genuinely stopped" from "log ended here"."""
    ev = attempts_with_episode.sort_values(["episode_id", "timestamp"], kind="mergesort")
    g = ev.groupby("episode_id", sort=True)

    ep = g.agg(
        release=("release", "first"),
        learner=("learner", "first"),
        problem_id=("problem_id", "first"),
        n_attempts=("id", "size"),
        start_timestamp=("timestamp", "min"),
        end_timestamp=("timestamp", "max"),
        first_score=("score_frac", "first"),
        best_score=("score_frac", "max"),
        terminal_score=("score_frac", "last"),
        y_first=("correct", "first"),
        y_ever=("correct", "max"),
        y_terminal=("correct", "last"),
        n_compile_errors_total=("n_compile_errors", "sum"),
    ).reset_index()
    ep["duration_seconds"] = (ep["end_timestamp"] - ep["start_timestamp"]).dt.total_seconds()
    for col in ("y_first", "y_ever", "y_terminal"):
        ep[col] = ep[col].astype(bool)

    # time_to_first_success: attempt index (0-based, within episode) of the first correct
    # attempt, NA if the episode never succeeds.
    ev = ev.copy()
    ev["_attempt_idx"] = ev.groupby("episode_id", sort=False).cumcount()
    first_success = (ev[ev["correct"] == 1]
                      .groupby("episode_id", sort=False)["_attempt_idx"].min())
    ep["time_to_first_success"] = ep["episode_id"].map(first_success)

    # Right-censoring: this episode's end_timestamp is the learner's last observed event in
    # this release -- the log ends here, not necessarily the learner's engagement.
    learner_last_ts = attempts_with_episode.groupby("learner")["timestamp"].max()
    ep["is_right_censored"] = ep["end_timestamp"] == ep["learner"].map(learner_last_ts)

    return ep[EPISODE_COLUMNS]


def reconcile_compile_score(attempts: pd.DataFrame) -> dict:
    """The compile<->score reconciliation numbers, REPRODUCED from the attempt table rather
    than taken on trust. Reference figures:
    compile error -> Score 0 at 100% both releases; zero-scores explained by compile error
    94.0% S19 / 91.9% F19; clean-compile-but-zero 1,781 (4.3%) S19 / 4,410 (5.9%) F19;
    clean-compile split full 39.7%/32.7%, partial 56.0%/61.4%.
    """
    has_error = attempts["n_compile_errors"] > 0
    zero = attempts["score_frac"] == 0.0
    clean = ~has_error
    n_clean = int(clean.sum())
    n_zero = int(zero.sum())

    return {
        "n_attempts": int(len(attempts)),
        "n_compile_error_attempts": int(has_error.sum()),
        "compile_error_implies_zero_score_rate": float(attempts.loc[has_error, "score_frac"].eq(0.0).mean()),
        "n_zero_score": n_zero,
        "zero_score_explained_by_compile_error_frac": float((has_error & zero).sum() / n_zero) if n_zero else float("nan"),
        "n_clean_compile": n_clean,
        "clean_compile_zero_n": int((clean & zero).sum()),
        "clean_compile_zero_frac": float((clean & zero).sum() / n_clean) if n_clean else float("nan"),
        "clean_compile_full_n": int((clean & attempts["score_frac"].eq(1.0)).sum()),
        "clean_compile_full_frac": float((clean & attempts["score_frac"].eq(1.0)).sum() / n_clean) if n_clean else float("nan"),
        "clean_compile_partial_n": int((clean & attempts["score_frac"].gt(0) & attempts["score_frac"].lt(1.0)).sum()),
        "clean_compile_partial_frac": float((clean & attempts["score_frac"].gt(0) & attempts["score_frac"].lt(1.0)).sum() / n_clean) if n_clean else float("nan"),
    }


def make_item_split(problem_ids, heldout_frac: float, seed: int = 0) -> pd.DataFrame:
    """Deterministic held-out split over the (shared, S19==F19) 50-problem catalog.

    Per-problem-ID, NOT per-near-duplicate-group (that grouping needs prompt text this
    release does not ship -- disclosed limitation, see module docstring). Returns
    DataFrame[problem_id, item_split] with item_split in {"train", "heldout"}.
    """
    uniq = np.array(sorted(set(int(p) for p in problem_ids)))
    n = len(uniq)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_heldout = int(round(n * heldout_frac))
    split = np.full(n, "train", dtype=object)
    split[perm[:n_heldout]] = "heldout"
    return pd.DataFrame({"problem_id": uniq, "item_split": split})


def tiktoc_stratum_ids() -> list[int]:
    """The 17 TIKTOC-protocol problem ids, read from the TIKTOC authors' own
    granular dataset rather than hardcoded, so this can never silently drift from theirs."""
    from atom.data.cw import load_granular
    return sorted(int(p) for p in load_granular()["ProblemID"].unique())
