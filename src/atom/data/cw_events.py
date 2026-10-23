"""CodeWorkout full-release event-graph ingest (S19 + F19).

Distinct from `atom.data.cw`/`cw_tuples`, which load the TIKTOC authors' own pre-merged
17-problem granular subset. This module ingests the raw CodeWorkout ProgSnap2 releases
(50 problems, ~410-500 learners each) directly from the vendor zips and reconstructs
attempts from the **event graph**. An attempt is:

    Run.Program parent
      + exactly one Compile child
      + zero or more Compile.Error children
      + CodeState code text
      + fractional Score
      + timestamp/problem/assignment metadata

Joining is by EventID/ParentEventID only -- never by timestamp proximity. F19
serialises ParentEventID with a trailing float suffix (e.g. "11161.0" pointing at EventID
"11161"); S19 does not. Both are canonicalised here before any join.

Licence discipline (same as `cw.py`): raw code text stays on the local machine and is
dropped by default. `include_code=True` keeps it for local-only uses (codec encoding);
nothing downstream commits it.
"""

from __future__ import annotations

import os
import zipfile
from pathlib import Path

import pandas as pd

from atom.data.tuples import hash_learner

RELEASE_ZIPS = {
    "S19": ("S19_All_Release_2_10_22.zip", "Data/"),
    "F19": ("F19_Release_All_05_23_22.zip", "All/Data/"),
}

_MAIN_TABLE_COLUMNS = [
    "SubjectID", "AssignmentID", "ProblemID", "CodeStateID", "EventType",
    "Score", "Compile.Result", "CompileMessageType", "EventID", "ParentEventID",
    "ServerTimestamp",
]


def default_cw_raw_dir() -> Path:
    """Resolve the CodeWorkout raw-release directory: $ATOM_CW_RAW_DIR, else
    ../Datasets/CodeWorkout_CMU."""
    env = os.environ.get("ATOM_CW_RAW_DIR")
    if env:
        return Path(env)
    repo_root = Path(__file__).resolve().parents[3]  # atom/src/atom/data/cw_events.py -> atom/
    return repo_root.parent / "Datasets" / "CodeWorkout_CMU"


def canonicalize_parent_id(series: pd.Series) -> pd.Series:
    """Strip a trailing '.0' float-formatting artifact from ParentEventID (F19-specific;
    a no-op on S19, whose IDs are never float-formatted -- see module docstring)."""
    return series.where(series.isna(), series.str.replace(r"\.0$", "", regex=True))


def load_raw_tables(release: str, base_dir=None, include_code: bool = False) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Read MainTable.csv (+ optionally CodeStates.csv) straight out of the vendor zip,
    all-string dtype (IDs are treated as opaque tokens, never numeric) except the parsed
    fields callers need as numbers. Returns (main_table, code_states_or_None).
    """
    if release not in RELEASE_ZIPS:
        raise ValueError(f"release must be one of {list(RELEASE_ZIPS)}, got {release!r}")
    zip_name, prefix = RELEASE_ZIPS[release]
    d = Path(base_dir) if base_dir else default_cw_raw_dir()
    zip_path = d / zip_name

    with zipfile.ZipFile(zip_path) as z:
        with z.open(f"{prefix}MainTable.csv") as f:
            main = pd.read_csv(f, dtype=str)
        code_states = None
        if include_code:
            with z.open(f"{prefix}CodeStates/CodeStates.csv") as f:
                code_states = pd.read_csv(f, dtype=str)

    main["ParentEventID"] = canonicalize_parent_id(main["ParentEventID"])
    main["ServerTimestamp"] = pd.to_datetime(main["ServerTimestamp"])
    main["release"] = release
    return main, code_states


def assert_event_graph(main: pd.DataFrame) -> dict:
    """The ingest gate's join assertions: every Compile's parent is a
    Run.Program EventID, every Run.Program has EXACTLY ONE Compile child, and every
    Compile.Error's parent is a Compile EventID. All three must be 0 for the gate to pass.
    """
    runs = main[main["EventType"] == "Run.Program"]
    compiles = main[main["EventType"] == "Compile"]
    errors = main[main["EventType"] == "Compile.Error"]

    run_ids = set(runs["EventID"])
    compile_ids = set(compiles["EventID"])

    compile_parent_ok = compiles["ParentEventID"].isin(run_ids)
    run_child_counts = compiles["ParentEventID"].value_counts()
    runs_with_not_exactly_one_compile = int((run_child_counts.reindex(list(run_ids), fill_value=0) != 1).sum())
    error_parent_ok = errors["ParentEventID"].isin(compile_ids)

    return {
        "compiles_with_bad_parent": int((~compile_parent_ok).sum()),
        "runs_with_not_exactly_one_compile": runs_with_not_exactly_one_compile,
        "errors_with_bad_parent": int((~error_parent_ok).sum()),
        "n_runs": len(runs),
        "n_compiles": len(compiles),
        "n_errors": len(errors),
    }


def build_attempts(main: pd.DataFrame, code_states: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per attempt (= one Run.Program event), joined by the event graph: the
    Compile.Result/CompileMessageType and the count of Compile.Error children come from
    the Run.Program's single Compile child (found via ParentEventID, never timestamp
    proximity). Learner ids are hashed (de-identified) before this leaves the loader.
    """
    runs = main[main["EventType"] == "Run.Program"].copy()
    compiles = main[main["EventType"] == "Compile"].set_index("EventID")
    errors = main[main["EventType"] == "Compile.Error"]
    error_counts = errors.groupby("ParentEventID").size()

    # Each Compile's ParentEventID IS the run's EventID -- build the reverse map once.
    compile_by_run_id = compiles.reset_index().set_index("ParentEventID")

    attempts = pd.DataFrame({
        "release": runs["release"].to_numpy(),
        "learner": runs["SubjectID"].map(hash_learner).to_numpy(),
        "id": runs["EventID"].to_numpy(),
        "problem_id": runs["ProblemID"].astype("int64").to_numpy(),
        "assignment_id": runs["AssignmentID"].astype("float64").to_numpy(),
        "code_state_id": runs["CodeStateID"].to_numpy(),
        "timestamp": runs["ServerTimestamp"].to_numpy(),
        "score_frac": runs["Score"].astype("float64").to_numpy(),
    })
    attempts["correct"] = (attempts["score_frac"] == 1.0).astype("int8")

    compile_event_id = runs["EventID"].map(compile_by_run_id["EventID"])
    attempts["compile_result"] = runs["EventID"].map(compile_by_run_id["Compile.Result"]).to_numpy()
    attempts["n_compile_errors"] = compile_event_id.map(error_counts).fillna(0).astype("int32").to_numpy()

    attempts = attempts.sort_values(["learner", "timestamp"], kind="mergesort").reset_index(drop=True)
    attempts["t"] = attempts.groupby("learner", sort=False).cumcount().astype("int32")

    if code_states is not None:
        # Joined via attempts["code_state_id"] (post-sort), never via `runs`'s original
        # row order -- `attempts` has already been re-sorted/re-indexed above.
        code_map = code_states.set_index("CodeStateID")["Code"]
        attempts["code"] = attempts["code_state_id"].map(code_map)

    return attempts


def release_stats(main: pd.DataFrame, attempts: pd.DataFrame) -> dict:
    """Canonical per-release counts used for the ingest gate."""
    return {
        "raw_rows": int(len(main)),
        "attempts": int(len(attempts)),
        "compiles": int((main["EventType"] == "Compile").sum()),
        "compile_errors": int((main["EventType"] == "Compile.Error").sum()),
        "learners": int(attempts["learner"].nunique()),
        "problems": int(attempts["problem_id"].nunique()),
    }


def build_problem_episodes(attempts: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Preliminary, problem-boundary episodes (a new episode starts whenever the learner
    switches problem) used ONLY for the ingest's hand-read stratified sample. This is NOT
    the final sessionised episode target (`cw_sessions` fits `tau_session` from the
    S19-train gap distribution and pre-registers it) -- do not reuse this grouping
    downstream of the ingest check.

    Returns (attempts_with_episode_id, episode_table) -- the attempt-level frame carries
    `episode_id` so callers can pull an episode's attempt rows without recomputing the
    grouping (see `sample_stratified_episodes`).
    """
    ev = attempts.sort_values(["learner", "timestamp"], kind="mergesort").reset_index(drop=True)
    same_learner = ev["learner"].eq(ev["learner"].shift())
    same_problem = ev["problem_id"].eq(ev["problem_id"].shift())
    new_episode = ~(same_learner & same_problem)
    ev["episode_id"] = (new_episode.cumsum() - 1).astype("int64")

    g = ev.groupby("episode_id", sort=True)
    ep = g.agg(
        release=("release", "first"),
        learner=("learner", "first"),
        problem_id=("problem_id", "first"),
        n_attempts=("id", "size"),
        start_timestamp=("timestamp", "min"),
        end_timestamp=("timestamp", "max"),
        first_score=("score_frac", "first"),
        final_score=("score_frac", "last"),
        min_score=("score_frac", "min"),
        max_score=("score_frac", "max"),
    ).reset_index()
    ep["solved"] = ep["final_score"] == 1.0
    ep["has_regression"] = ep["final_score"] < ep["max_score"]

    returned = ep.groupby(["learner", "problem_id"])["episode_id"].transform("count") > 1
    ep["is_return_episode"] = returned

    return ev, ep


def sample_stratified_episodes(attempts_with_episode_id: pd.DataFrame, episodes: pd.DataFrame,
                                n: int = 10, seed: int = 0) -> list[dict]:
    """Deterministically sample episodes across the length/success/regression/return
    strata (the ingest check reads 10 episodes by hand across these strata). Each stratum contributes at least one episode where available; the remainder
    (if n exceeds the number of strata) is filled from the full pool.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    long_thresh = episodes["n_attempts"].quantile(0.9)

    strata = {
        "long": episodes[episodes["n_attempts"] >= long_thresh],
        "short": episodes[episodes["n_attempts"] == 1],
        "solved": episodes[episodes["solved"]],
        "unsolved": episodes[~episodes["solved"]],
        "regression": episodes[episodes["has_regression"]],
        "return": episodes[episodes["is_return_episode"]],
    }

    chosen_ids: list[int] = []
    for name, pool in strata.items():
        if len(pool) == 0:
            continue
        pick = rng.choice(pool["episode_id"].to_numpy(), size=1, replace=False)
        if int(pick[0]) not in chosen_ids:
            chosen_ids.append(int(pick[0]))

    remaining = n - len(chosen_ids)
    if remaining > 0:
        pool = episodes[~episodes["episode_id"].isin(chosen_ids)]
        if len(pool) > 0:
            extra = rng.choice(pool["episode_id"].to_numpy(), size=min(remaining, len(pool)), replace=False)
            chosen_ids.extend(int(x) for x in extra)

    out = []
    for eid in sorted(chosen_ids[:n]):
        rows = attempts_with_episode_id[attempts_with_episode_id["episode_id"] == eid]
        ep_row = episodes[episodes["episode_id"] == eid].iloc[0]
        out.append({
            "episode_id": eid,
            "release": ep_row["release"],
            "learner": ep_row["learner"],
            "problem_id": int(ep_row["problem_id"]),
            "n_attempts": int(ep_row["n_attempts"]),
            "solved": bool(ep_row["solved"]),
            "has_regression": bool(ep_row["has_regression"]),
            "is_return_episode": bool(ep_row["is_return_episode"]),
            "scores": [round(float(s), 3) for s in rows["score_frac"]],
            "compile_errors": [int(x) for x in rows["n_compile_errors"]],
            "timestamps": [str(t) for t in rows["timestamp"]],
        })
    return out