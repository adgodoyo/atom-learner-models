"""FoundationalASSIST loader.

Loads the three FA files with canonical, snake_cased schemas and the correctness label
used everywhere downstream. The reference statistics checked on load are:

  interactions = len(Interactions.csv)                 = 1,753,384
  users        = nunique(user_id)                      = 5,000
  problems     = len(Problems.csv)  [ROW count]        = 3,395   (3,368 unique ids + 27 dup rows)
  correct rate = mean(discrete_score == 1 over ALL rows) = 0.613  (8,952 NaN scores counted as 0)

Licence rule: raw question/answer text stays on the local machine. This loader reads
from the local Datasets/ folder and returns DataFrames in memory; callers must not
persist raw text into version control or remote storage — only derived artifacts (codec vectors, hashed
IDs, labels) may be exported.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

# Problems.csv has human-readable headers; normalise to snake_case.
_PROBLEM_RENAME = {
    "Problem Set Id": "problem_set_id",
    "Problem Part": "problem_part",
    "Problem Type": "problem_type",
    "Answer Types": "answer_types",
    "Problem Body": "problem_body",
    "Fill-in Options": "fillin_options",
    "Fill-in Answers": "fillin_answers",
    "Multiple Choice Options": "mc_options",
    "Multiple Choice Answers": "mc_answers",
    "problem_id": "problem_id",
}


def default_fa_dir() -> Path:
    """Resolve the FoundationalASSIST directory: $ATOM_FA_DIR, else ../Datasets/FoundationalAssist."""
    env = os.environ.get("ATOM_FA_DIR")
    if env:
        return Path(env)
    repo_root = Path(__file__).resolve().parents[3]  # atom/src/atom/data/fa.py -> atom/
    return repo_root.parent / "Datasets" / "FoundationalAssist"


def _data_dir(base_dir: str | os.PathLike | None) -> Path:
    base = Path(base_dir) if base_dir else default_fa_dir()
    return base / "Data" if (base / "Data").exists() else base


def load_interactions(base_dir=None, parse_time: bool = True) -> pd.DataFrame:
    """Load Interactions.csv, drop the pandas index column, add the `correct` label.

    `correct` = 1 iff discrete_score == 1 (solved first try, no hint/answer); the 8,952
    NaN scores become 0. This is the canonical correctness signal of the loader.
    """
    df = pd.read_csv(_data_dir(base_dir) / "Interactions.csv")
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")], errors="ignore")
    df["correct"] = (df["discrete_score"] == 1).astype("int8")
    if parse_time:
        df["end_time"] = pd.to_datetime(df["end_time"], errors="coerce", utc=True)
    return df


def load_problems(base_dir=None, clean_body: bool = False) -> pd.DataFrame:
    """Load Problems.csv with snake_cased columns. If clean_body, add `problem_body_clean`."""
    df = pd.read_csv(_data_dir(base_dir) / "Problems.csv").rename(columns=_PROBLEM_RENAME)
    if clean_body:
        from atom.data.fa_clean import clean_problem_body

        df["problem_body_clean"] = df["problem_body"].map(clean_problem_body)
    return df


def load_skills(base_dir=None) -> pd.DataFrame:
    return pd.read_csv(_data_dir(base_dir) / "Skills.csv")


def fa_stats(interactions=None, problems=None, skills=None, base_dir=None) -> dict:
    """Canonical FA statistics used for the loading check and the data report
    (`p1_05_data_report.py`)."""
    if interactions is None:
        interactions = load_interactions(base_dir, parse_time=False)
    if problems is None:
        problems = load_problems(base_dir)
    if skills is None:
        skills = load_skills(base_dir)

    return {
        "interactions": int(len(interactions)),
        "users": int(interactions["user_id"].nunique()),
        "problems": int(len(problems)),  # ROW count is the gate definition
        "unique_problem_ids": int(problems["problem_id"].nunique()),
        "skills": int(skills["skill_id"].nunique()),
        "correct_rate": float(interactions["correct"].mean()),
        "discrete_score_nan": int(interactions["discrete_score"].isna().sum()),
        "hint_rate": float((interactions["hint_count"] > 0).mean()),
        "saw_answer_rate": float(interactions["saw_answer"].astype(bool).mean()),
        "min_problems_per_user": int(interactions.groupby("user_id").size().min()),
        "max_problems_per_user": int(interactions.groupby("user_id").size().max()),
    }
