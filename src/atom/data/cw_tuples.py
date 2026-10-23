"""CW event tuples + episode reconstruction from the TIKTOC granular subset.

An "episode" is a learner's contiguous run of attempts on one problem (a coding session on
that problem) -- the CodeWorkout literature's natural unit, and the granularity TIKTOC's own
GranularKT models operate on. Detected structurally (a new episode starts whenever
problem_id changes from the learner's own previous event, in chronological order), not
assumed from AssignmentID grouping -- a learner could in principle return to an earlier
problem later, and this definition handles that correctly either way.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from atom.data.tuples import hash_learner

EVENT_COLUMNS = [
    "learner", "id", "t", "problem_id", "assignment_id", "timestamp",
    "episode_id", "attempt_idx", "score_frac", "correct", "n_testcases",
]


def build_cw_events(df: pd.DataFrame) -> pd.DataFrame:
    """df: TIKTOC granular dataframe (atom.data.cw.load_granular's output).
    Returns the canonical CW event stream, chronologically ordered within learner
    (mergesort = stable, so same-timestamp submissions keep their original relative order),
    with episode_id/attempt_idx derived structurally per the module docstring.
    """
    ev = pd.DataFrame({
        "learner": df["SubjectID"].map(hash_learner),
        "id": df["CodeStateID"],
        "problem_id": df["ProblemID"].astype("int64"),
        "assignment_id": df["AssignmentID"].astype("int64"),
        "timestamp": df["ServerTimestamp"],
        "score_frac": df["Score"].astype("float64"),
        "correct": (df["Score"] == 1.0).astype("int8"),
        "n_testcases": df["binary_correctness"].map(len).astype("int32"),
    })
    ev = ev.sort_values(["learner", "timestamp"], kind="mergesort").reset_index(drop=True)
    ev["t"] = ev.groupby("learner", sort=False).cumcount().astype("int32")

    same_learner = ev["learner"].eq(ev["learner"].shift())
    same_problem = ev["problem_id"].eq(ev["problem_id"].shift())
    new_episode = ~(same_learner & same_problem)
    ev["episode_id"] = (new_episode.cumsum() - 1).astype("int64")  # 0-indexed, unique across learners
    ev["attempt_idx"] = ev.groupby("episode_id", sort=False).cumcount().astype("int32")

    return ev[EVENT_COLUMNS]


def build_episode_table(events: pd.DataFrame) -> pd.DataFrame:
    """One row per episode: learner, problem, attempt count, span, final outcome."""
    g = events.groupby("episode_id", sort=True)
    ep = g.agg(
        learner=("learner", "first"),
        problem_id=("problem_id", "first"),
        assignment_id=("assignment_id", "first"),
        n_attempts=("attempt_idx", "size"),
        start_timestamp=("timestamp", "min"),
        end_timestamp=("timestamp", "max"),
        final_score=("score_frac", "last"),
    ).reset_index()
    ep["solved"] = ep["final_score"] == 1.0
    return ep


def audit_episode_structure(events: pd.DataFrame) -> dict:
    """Structural checks standing in for the gate's 'episode reconstruction' claim: every
    episode has exactly one problem_id, timestamps never go backwards within an episode,
    and attempt_idx is a contiguous 0..n-1 run per episode (recomputed fresh here and
    compared against the stored column, so it also catches drift if something upstream
    re-sorts events without recomputing attempt_idx)."""
    bad_problem = int(events.groupby("episode_id")["problem_id"].nunique().gt(1).sum())
    dt = events.groupby("episode_id")["timestamp"].diff()
    backwards_steps = int((dt < pd.Timedelta(0)).sum())
    expected_attempt_idx = events.groupby("episode_id").cumcount()
    attempt_idx_mismatches = int((events["attempt_idx"].to_numpy() != expected_attempt_idx.to_numpy()).sum())
    return {
        "episodes_with_multiple_problem_ids": bad_problem,
        "backwards_timestamp_steps": backwards_steps,
        "attempt_idx_mismatches": attempt_idx_mismatches,
    }


def sample_trajectories(events: pd.DataFrame, episodes: pd.DataFrame, n: int = 10,
                         seed: int = 0) -> list[dict]:
    """Deterministically sample n episodes and return their full attempt sequence, for a
    human-readable spot check (the data check verifies 10 trajectories by hand).
    Biased toward multi-attempt episodes -- a 1-attempt episode has no ordering to verify,
    so it would make a trivial, uninformative spot check.
    """
    rng = np.random.default_rng(seed)
    multi_attempt = episodes[episodes["n_attempts"] > 1]
    pool = multi_attempt if len(multi_attempt) >= n else episodes
    chosen = rng.choice(pool["episode_id"].to_numpy(), size=min(n, len(pool)), replace=False)

    out = []
    for eid in sorted(int(x) for x in chosen):
        rows = events[events["episode_id"] == eid].sort_values("attempt_idx")
        out.append({
            "episode_id": eid,
            "learner": rows["learner"].iloc[0],
            "problem_id": int(rows["problem_id"].iloc[0]),
            "n_attempts": len(rows),
            "scores": [round(float(s), 3) for s in rows["score_frac"]],
            "timestamps": [str(t) for t in rows["timestamp"]],
        })
    return out
