"""CodeWorkout code-diff construction (previous attempt -> current attempt, unified diff).

Each attempt's code-edit input represents "the edit from the previous attempt". The
posterior encoder consumes attempts episode-locally, so "previous attempt" is scoped to
the same episode, not the same learner+problem across a session gap: a return episode's
first attempt gets the learned NO_PREVIOUS_CODE embedding even though an earlier episode
on the same problem has real prior code, because that prior code belongs to a different
attempt-token sequence. This module only emits diffs for attempts that do have a previous
attempt in the same episode; the NO_PREVIOUS_CODE substitution itself happens at model
time, not here.

Input is the `attempts` frame carrying an `episode_id` column
(`atom.data.cw_sessions.sessionize`'s output) plus a `code` column (raw text,
`include_code=True` at the `cw_events.build_attempts` call site). The sessioniser's
partition invariant (one problem and one learner per episode_id) is what makes grouping
by `episode_id` alone sufficient and correct here.
"""

from __future__ import annotations

import difflib

import pandas as pd

REQUIRED_COLUMNS = {"episode_id", "timestamp", "code_state_id", "code"}


def unified_diff_text(prev_code: str, code: str, context_lines: int = 3) -> str:
    """Unified diff, hunk lines only -- no `---`/`+++` file-header lines (there is no
    filename to put there, and a constant boilerplate pair on every diff would waste
    embedding capacity on a token sequence that carries no information)."""
    lines = list(difflib.unified_diff(
        prev_code.splitlines(keepends=False),
        code.splitlines(keepends=False),
        lineterm="",
        n=context_lines,
    ))
    if lines:
        lines = lines[2:]  # drop the '--- '/'+++ ' header pair
    return "\n".join(lines)


def build_code_diffs(attempts: pd.DataFrame, context_lines: int = 3) -> pd.DataFrame:
    """One row per attempt that has a PREVIOUS attempt in the same episode.

    Returns columns: episode_id, code_state_id, prev_code_state_id, timestamp,
    diff_text, is_empty_diff (identical consecutive code -- a real, disclosed
    possibility: e.g. a re-run with no edit). Rows for episode-first attempts
    (no previous code in that episode) are excluded by construction.

    A small, real data gap (measured on the full S19+F19 corpus: 10/195,205 attempts,
    0.005%) have missing or empty `code` -- most likely a
    CodeStateID that never joined to CodeStates.csv, not a genuine codeless attempt.
    Such rows are dropped BEFORE the groupby-shift, so they can never appear as either
    the current or the previous member of a diff pair: a row immediately after a
    dropped one is treated as if the dropped row were absent (diffs against whatever
    real code came before it, or gets no previous-attempt row at all if it was first),
    rather than fabricating a diff against unencodable text.
    """
    missing = REQUIRED_COLUMNS - set(attempts.columns)
    if missing:
        raise ValueError(f"attempts is missing required columns: {sorted(missing)}")

    has_code = attempts["code"].notna() & (attempts["code"].astype(str).str.strip() != "")
    df = attempts.loc[has_code].sort_values(["episode_id", "timestamp"]).reset_index(drop=True)
    grouped = df.groupby("episode_id", sort=False)
    prev_code = grouped["code"].shift(1)
    prev_id = grouped["code_state_id"].shift(1)

    has_prev = prev_code.notna()
    out = df.loc[has_prev, ["episode_id", "timestamp", "code_state_id"]].copy()
    out["prev_code_state_id"] = prev_id.loc[has_prev].to_numpy()

    cur_codes = df.loc[has_prev, "code"].tolist()
    prv_codes = prev_code.loc[has_prev].tolist()
    out["diff_text"] = [
        unified_diff_text(p, c, context_lines) for p, c in zip(prv_codes, cur_codes)
    ]
    out["is_empty_diff"] = out["diff_text"] == ""
    return out.reset_index(drop=True)
