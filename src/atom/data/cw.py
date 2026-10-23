"""CodeWorkout loader, via the TIKTOC repo's pre-built granular dataset.

`dataset_granular_all.pkl` (produced by the TIKTOC authors' own preprocessing, reused
rather than reimplemented) is already a merge of ProgSnap2's MainTable +
CodeStates with per-test-case binary correctness: one row per code submission, with a
per-submission list of test-case pass/fail outcomes (`binary_correctness`) and the mean
pass rate (`Score`). `dataset_granular_1st.pkl` (variant='1st', one row per (student,
problem) -- their first-submission file, the evaluation set of the TIKTOC-transfer
evaluation of the CodeWorkout generative simulator)
carries the identical list under a DIFFERENT raw name, `binary_score` -- `load_granular`
renames it to `binary_correctness` on the way out so callers never need to know which
variant they asked for.

Licence rule (same discipline as FA): raw code text stays on the local machine. By default `load_granular` drops the heavy/raw columns (`Code`, AST/embedding columns,
`prompt`) immediately after reading the pickle -- only IDs, labels and timestamps flow
downstream into event tuples and version-controlled artifacts. `include_code=True` (used by
the codec-semantics probes, which need the code text to encode it -- the same local-only
use as FA's `load_problems`/`load_interactions` feeding the codec) keeps
`Code` and `prompt` too; the AST/embedding columns are dropped either way since nothing
downstream uses TIKTOC's own precomputed embeddings.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

# Only these survive the read by default -- everything else (Code, Code-ast, code-astnn,
# code-embedding, embedding, astnn, prompt, prompt-embedding, input) is raw text or a
# derived embedding of it, and is dropped immediately (see module docstring).
_LIGHT_COLUMNS = [
    "SubjectID", "AssignmentID", "ProblemID", "CodeStateID", "Score",
    "ServerTimestamp", "timestep",
]
_CODE_COLUMNS = ["Code", "prompt"]

# The per-test-case pass/fail list carries a DIFFERENT raw column name per variant
# (verified directly against both released pickles -- 'all' has
# `binary_correctness`, '1st' has `binary_score`; there is no column named
# `binary_correctness` in `dataset_granular_1st.pkl` at all). Always exposed downstream
# as `binary_correctness` so every existing consumer (cw_pairs.py, cw_probes.py,
# cw_tuples.py, the test fixtures) needs no variant-awareness of its own.
_LABEL_COLUMN_BY_VARIANT = {"all": "binary_correctness", "1st": "binary_score"}


def default_cw_dir() -> Path:
    """Resolve the TIKTOC data directory: $ATOM_CW_DIR, else ../Datasets/tiktoc-main/data."""
    env = os.environ.get("ATOM_CW_DIR")
    if env:
        return Path(env)
    repo_root = Path(__file__).resolve().parents[3]  # atom/src/atom/data/cw.py -> atom/
    return repo_root.parent / "Datasets" / "tiktoc-main" / "data"


def load_granular(base_dir=None, variant: str = "all", include_code: bool = False) -> pd.DataFrame:
    """Load the TIKTOC granular dataset (`dataset_granular_{variant}.pkl`, unzipped from the
    matching .zip if the .pkl isn't already on disk). variant='all' = every submission
    (the richer, attempt-level file); variant='1st' = first
    submission per (student, problem) only (the TIKTOC authors' other released file).
    include_code: also keep `Code`/`prompt` (raw text) -- see module docstring.
    """
    if variant not in ("all", "1st"):
        raise ValueError(f"variant must be 'all' or '1st', got {variant!r}")
    d = Path(base_dir) if base_dir else default_cw_dir()
    pkl_path = d / f"dataset_granular_{variant}.pkl"
    if not pkl_path.exists():
        zip_path = d / f"dataset_granular_{variant}.zip"
        import zipfile
        with zipfile.ZipFile(zip_path) as z:
            z.extract(pkl_path.name, d)

    df = pd.read_pickle(pkl_path)
    label_col = _LABEL_COLUMN_BY_VARIANT[variant]
    columns = _LIGHT_COLUMNS + [label_col]
    if include_code:
        columns = columns + _CODE_COLUMNS
    df = df[columns].copy()
    if label_col != "binary_correctness":
        df = df.rename(columns={label_col: "binary_correctness"})
    df["AssignmentID"] = df["AssignmentID"].astype("int64")
    df["ServerTimestamp"] = pd.to_datetime(df["ServerTimestamp"])
    return df


def cw_stats(df: pd.DataFrame) -> dict:
    """Canonical CW statistics used for the data-loading check (cross-checked vs the TIKTOC paper's
    Table 2: 246 students, 17 problems with test cases, ~3,714 (student, problem) attempt
    pairs -- see scripts/p1_03_cw_tuples.py for the measured-vs-paper reconciliation)."""
    pairs = df.groupby(["SubjectID", "ProblemID"]).ngroups
    return {
        "submissions": int(len(df)),
        "students": int(df["SubjectID"].nunique()),
        "problems": int(df["ProblemID"].nunique()),
        "assignments": int(df["AssignmentID"].nunique()),
        "student_problem_pairs": int(pairs),
        "mean_score": float(df["Score"].mean()),
        "exact_solve_rate": float((df["Score"] == 1.0).mean()),
    }
