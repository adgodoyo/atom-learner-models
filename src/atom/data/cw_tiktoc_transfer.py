"""Ingest for the TIKTOC-transfer evaluation of the CodeWorkout generative simulator.

Adapts TIKTOC's own `dataset_granular_1st.pkl` (246 students, 17
problems, one row per (SubjectID, ProblemID) -- their first-submission file) into the
same `attempts`-shaped schema `atom.data.cw_events.build_attempts` produces from S19/F19,
so every downstream generative-simulator function (`sessionize`, `build_episode_level_table`,
`build_attempt_batch`, `filt_forward`, `test_vector_eval_predictions`,
`test_vector_auc_panel`) runs on TIKTOC rows completely unmodified.

Two real disclosed differences from the S19/F19 path, both consequences of TIKTOC's
schema having no event graph:

1. **No compile-event data at all** (checked directly -- no column matches any
   compile-related name). `compile_result`/`n_compile_errors` are a PROXY, not a
   measurement: every row is treated as `compile_result="Success"`/`n_compile_errors=0`,
   since a row with a real `binary_score` test-execution result could not have failed to
   compile. This only feeds `scalar_feats`' `compile_ok` input to the belief filter, never
   the test-vector ground truth itself (which comes straight from `binary_score`).
2. **Every episode has length 1** (verified: max group size over (SubjectID, ProblemID)
   is 1 across all 3,714 rows -- TIKTOC's own first-submission file, not the richer
   attempt-level `_all` variant). `sessionize` degenerates correctly (new episode fires on
   every problem change, which is every row here) with no special-casing needed, and
   `diff_of_hash` is never applicable -- every attempt is its episode's first, so
   `build_attempt_batch`'s own zero-diff-vector convention for a first attempt applies
   universally, not as an edge case.

`learner` ids are namespaced `"tiktoc:" + SubjectID` rather than run through
`atom.data.tuples.hash_learner` -- TIKTOC's SubjectID is a different, already-opaque id
space from S19/F19's raw CodeWorkout ids, and hashing it would not add de-identification,
only remove the ability to tell the two cohorts apart if a hash ever coincided.

Ground-truth test-vector labels are built directly from `binary_correctness` (the
column `atom.data.cw.load_granular` renames TIKTOC's `binary_score` to on load) -- no
JDK, no `score_submission`, no re-execution: TIKTOC already ran their own tests. Slot-
count agreement against our own `test_cases-1-26-24.csv` was verified directly for all
13 headline problems (exact match, single count per problem, zero row-level variation);
the 4 non-headline problems (34/37/39/40) have NO rows at all in our own CSV's
`coding_prompt_id` universe (confirmed -- `load_test_cases` returns an empty list for
each), which is why they are not in `cw_5_functional_equivalence.HEADLINE_PROBLEM_IDS` -- so their slot order is trusted from
TIKTOC's own internal consistency only, disclosed, never independently cross-checked.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from atom.data.cw import load_granular
from atom.data.cw_manifest import attach_prior_attempt_code, build_episode_level_table
from atom.data.cw_outcome_label import outcome_label
from atom.data.cw_sessions import sessionize

TAU_SESSION_MINUTES = 60.0  # The frozen session gap (`cw_sessions`); kept for interface parity even though
# every TIKTOC episode degenerates to length 1 regardless of tau (module docstring).

MAX_TEST_CASES = 32


def build_tiktoc_attempts(base_dir=None) -> pd.DataFrame:
    """TIKTOC's first-submission file as an `attempts`-shaped DataFrame, schema-identical
    to `atom.data.cw_events.build_attempts`'s output (same columns/dtypes/sort order),
    so every downstream generative-simulator function accepts it unmodified. See module docstring for
    the two disclosed proxy fields (`compile_result`, `n_compile_errors`)."""
    df = load_granular(base_dir=base_dir, variant="1st", include_code=True)

    attempts = pd.DataFrame({
        "release": np.full(len(df), "TIKTOC"),
        "learner": ("tiktoc:" + df["SubjectID"].astype(str)).to_numpy(),
        "id": df["CodeStateID"].to_numpy(),  # TIKTOC has no separate EventID; one row = one event
        "problem_id": df["ProblemID"].astype("int64").to_numpy(),
        "assignment_id": df["AssignmentID"].astype("float64").to_numpy(),
        "code_state_id": df["CodeStateID"].to_numpy(),
        "timestamp": df["ServerTimestamp"].to_numpy(),
        "score_frac": df["Score"].astype("float64").to_numpy(),
    })
    attempts["correct"] = (attempts["score_frac"] == 1.0).astype("int8")
    # Proxy (module docstring, disclosed difference 1): no compile-event data exists in
    # this schema at all, and a row with a real test-execution result must have compiled.
    attempts["compile_result"] = "Success"
    attempts["n_compile_errors"] = np.zeros(len(df), dtype="int32")

    attempts = attempts.sort_values(["learner", "timestamp"], kind="mergesort").reset_index(drop=True)
    attempts["t"] = attempts.groupby("learner", sort=False).cumcount().astype("int32")
    # Joined by VALUE (code_state_id, unique per row in this one-row-per-submission
    # schema), never by positional reindex against the pre-sort frame -- matches
    # build_attempts's own value-keyed `code_states.set_index("CodeStateID")["Code"]`
    # convention exactly.
    code_map = df.set_index("CodeStateID")["Code"]
    attempts["code"] = attempts["code_state_id"].map(code_map)

    return attempts


def build_tiktoc_test_vector_labels(base_dir=None, max_test_cases: int = MAX_TEST_CASES) -> dict:
    """TIKTOC's own `binary_correctness` (renamed from `binary_score` by `load_granular`)
    as the same npz-shaped dict `scripts/cw_sim_test_vector_labels.py` produces from
    executing S19/F19 code -- `test_vector_eval_predictions`/`test_vector_auc_panel`
    consume either source identically. No execution: TIKTOC already ran their tests."""
    df = load_granular(base_dir=base_dir, variant="1st", include_code=False)

    n_rows = len(df)
    n_cases = df["binary_correctness"].map(len).astype("int64").to_numpy()
    if (n_cases > max_test_cases).any():
        raise ValueError(
            f"a TIKTOC row has more than max_test_cases={max_test_cases} test cases "
            f"(max seen: {int(n_cases.max())}) -- raise max_test_cases, do not truncate"
        )

    test_vector = np.zeros((n_rows, max_test_cases), dtype=bool)
    valid_mask = np.zeros((n_rows, max_test_cases), dtype=bool)
    for i, (vec, n) in enumerate(zip(df["binary_correctness"], n_cases)):
        valid_mask[i, :n] = True
        test_vector[i, :n] = np.asarray(vec, dtype=bool)

    headline_problem_ids = np.asarray(
        [1, 3, 5, 12, 13, 17, 20, 21, 22, 24, 25, 46, 71], dtype=np.int64,
    )  # cw_5_functional_equivalence.HEADLINE_PROBLEM_IDS, duplicated as a literal here
       # to avoid this JDK-free module importing the JDK-dependent script.

    return {
        "learner": ("tiktoc:" + df["SubjectID"].astype(str)).to_numpy(),
        "problem_id": df["ProblemID"].astype("int64").to_numpy(),
        "code_state_id": df["CodeStateID"].to_numpy(),
        "release": np.full(n_rows, "TIKTOC"),
        "compiled": np.ones(n_rows, dtype=bool),  # a row with a binary_score result ran
        "test_vector": test_vector,
        "valid_mask": valid_mask,
        "n_cases": n_cases,
        "max_test_cases": np.asarray(max_test_cases),
        "headline_problem_ids": headline_problem_ids,
        "label_construct": np.asarray("tiktoc_native_first_submission"),
    }


def build_tiktoc_attempts_all(base_dir=None) -> pd.DataFrame:
    """TIKTOC's full attempt-level file (`dataset_granular_all`) as an attempts-shaped
    DataFrame -- restores real multi-attempt episodes, unlike `build_tiktoc_attempts`'s
    "_1st"-only first-submission file (module docstring's disclosed difference 2, which
    this function does NOT share). Rebuilding history from this variant restores real
    prior_code/diff across a student's own multiple attempts, the evidence channel that
    carries most of the in-corpus AUC signal. Schema-identical to `build_tiktoc_attempts`'s own output; same disclosed
    `compile_result="Success"` proxy -- neither TIKTOC variant carries genuine
    compile-event data, confirmed directly (13,083 rows, zero null/empty
    `binary_correctness`, so every row ran far enough to be tested)."""
    df = load_granular(base_dir=base_dir, variant="all", include_code=True)

    attempts = pd.DataFrame({
        "release": np.full(len(df), "TIKTOC"),
        "learner": ("tiktoc:" + df["SubjectID"].astype(str)).to_numpy(),
        "id": df["CodeStateID"].to_numpy(),
        "problem_id": df["ProblemID"].astype("int64").to_numpy(),
        "assignment_id": df["AssignmentID"].astype("float64").to_numpy(),
        "code_state_id": df["CodeStateID"].to_numpy(),
        "timestamp": df["ServerTimestamp"].to_numpy(),
        "score_frac": df["Score"].astype("float64").to_numpy(),
    })
    attempts["correct"] = (attempts["score_frac"] == 1.0).astype("int8")
    attempts["compile_result"] = "Success"
    attempts["n_compile_errors"] = np.zeros(len(df), dtype="int32")

    attempts = attempts.sort_values(["learner", "timestamp"], kind="mergesort").reset_index(drop=True)
    attempts["t"] = attempts.groupby("learner", sort=False).cumcount().astype("int32")
    code_map = df.set_index("CodeStateID")["Code"]
    attempts["code"] = attempts["code_state_id"].map(code_map)

    return attempts


def build_tiktoc_manifest_all(base_dir=None) -> dict:
    """TIKTOC as a dict shaped like the generative simulator's training manifest
    (`infra.cw_5_full_run.load_cw5_manifest`), built from the FULL
    `granular_all` attempt history rather than `build_tiktoc_manifest`'s first-
    submission-only "_1st" file -- restores real within-episode `prior_code`/diff for
    any target with an earlier same-problem attempt in its own episode, plus (for every
    target, regardless) a genuinely richer BELIEF at scoring time: the filter now
    integrates the student's fuller multi-attempt trajectory on every EARLIER problem,
    not a single first-submission snapshot per problem.

    Targets: the SAME (learner, problem_id, code_state_id) triples
    `build_tiktoc_attempts` ("_1st") designates as each student's first submission --
    NOT independently re-derived by earliest-timestamp-in-"_all". Verified: of
    3,714 "_1st" targets, 2 (subject,problem) pairs have NO rows at all in "_all", and a
    further 19 have a "_1st"-designated CodeStateID that does not appear anywhere in
    "_all" (a genuine inconsistency between TIKTOC's own two released files -- their
    earliest "_all" row for those 19 pairs is a DIFFERENT CodeStateID entirely, not a
    timestamp-tie artifact). Both gaps are real and disclosed, sized in the returned
    stats, never silently substituted with a different row: 21/3,714 (0.57%) "_1st"
    targets are unscoreable under "_all" and excluded from `is_target`, not replaced.

    Empirically (measured on the real data, NOT a logical guarantee -- a toy
    fixture can trivially violate it, see `tests/test_cw_tiktoc_transfer.py`), every one
    of the 3,693 FOUND targets happens to also be its own episode's first attempt
    (`attempt_within_idx == 0`): the 19 "_1st"/"_all" disagreements are exactly the
    cases excluded by the `is_target` lookup above, so among targets that DO survive the
    join, "_1st"'s designated submission and "_all"'s own earliest-in-episode row
    coincide in every real instance measured. Nothing here enforces that coincidence."""
    all_attempts = build_tiktoc_attempts_all(base_dir=base_dir)
    first_attempts = build_tiktoc_attempts(base_dir=base_dir)
    target_lookup = set(zip(first_attempts["learner"], first_attempts["problem_id"],
                           first_attempts["code_state_id"]))

    all_attempts = sessionize(all_attempts, TAU_SESSION_MINUTES)

    labels = outcome_label(all_attempts["compile_result"], all_attempts["score_frac"])
    n_before = len(all_attempts)
    all_attempts = all_attempts.assign(outcome_code=labels["outcome_code"], outcome_valid=labels["outcome_valid"])
    all_attempts = all_attempts[all_attempts["outcome_valid"]].reset_index(drop=True)
    n_dropped_invalid = n_before - len(all_attempts)
    all_attempts = attach_prior_attempt_code(all_attempts)

    row_key = list(zip(all_attempts["learner"], all_attempts["problem_id"], all_attempts["code_state_id"]))
    all_attempts["is_target"] = [k in target_lookup for k in row_key]
    n_targets_found = int(all_attempts["is_target"].sum())
    n_targets_missing = len(target_lookup) - n_targets_found

    episodes = build_episode_level_table(all_attempts)
    val_learners = set(all_attempts["learner"].unique())

    return {
        "attempts": all_attempts, "episodes": episodes, "val_learners": val_learners,
        "n_attempts_before_outcome_filter": n_before,
        "n_dropped_invalid_outcome": n_dropped_invalid,
        "n_targets_1st": len(target_lookup),
        "n_targets_found_in_all": n_targets_found,
        "n_targets_missing_from_all": n_targets_missing,
    }


def build_tiktoc_manifest(base_dir=None) -> dict:
    """TIKTOC as a dict shaped like the generative simulator's training manifest
    (`attempts`, `episodes`, `val_learners`) -- a drop-in for every downstream
    generative-simulator function
    (`score_arm_for_batch`, `forward_outcome`, `test_vector_eval_predictions`,
    `row_generation_plan`, `run_batched_generation_pass`) that only ever reads columns
    `build_tiktoc_attempts` already produces schema-identically to `build_attempts`.

    `val_learners` = every TIKTOC learner: TIKTOC is entirely an external transfer-eval
    set (246 students never in S19/F19's own train/val split), so there is no train
    population to hold anything out from -- unlike the training manifest, this one is not
    used for anything but scoring.

    Sessionize + `outcome_label` run UNMODIFIED on TIKTOC's attempts (both are generic
    over any attempts table with the right columns; `outcome_label`'s `compile_result`
    input is the disclosed `"Success"` proxy `build_tiktoc_attempts` already sets,
    documented there, not re-derived here). Rows failing `outcome_valid` are dropped
    exactly as `load_cw5_manifest` drops them -- TIKTOC's own docstring's proxy makes
    every row `compile_result="Success"` with a real `score_frac`, so `outcome_valid`
    should hold for (nearly) all rows; any drop is sized in the returned stats, never
    silently absorbed.

    `attach_prior_attempt_code` (`cw_manifest`'s own, the SAME function `cw_5_full_run.py`
    calls on S19/F19) adds `attempt_within_idx`/`prior_code` -- `build_tiktoc_attempts`
    does not derive these itself, and `build_class_conditioned_examples` requires the
    `prior_code` column to exist. Every TIKTOC episode is length 1
    (module docstring), so this degenerates correctly with no special-casing: every row
    gets `attempt_within_idx=0`, `prior_code=NaN`, matching the established 'every row is a
    first attempt' convention (`cw_held_out_generation.py`)."""
    attempts = build_tiktoc_attempts(base_dir=base_dir)
    attempts = sessionize(attempts, TAU_SESSION_MINUTES)

    labels = outcome_label(attempts["compile_result"], attempts["score_frac"])
    n_before = len(attempts)
    attempts = attempts.assign(outcome_code=labels["outcome_code"], outcome_valid=labels["outcome_valid"])
    attempts = attempts[attempts["outcome_valid"]].reset_index(drop=True)
    n_dropped_invalid = n_before - len(attempts)
    attempts = attach_prior_attempt_code(attempts)

    episodes = build_episode_level_table(attempts)
    val_learners = set(attempts["learner"].unique())

    return {
        "attempts": attempts, "episodes": episodes, "val_learners": val_learners,
        "n_attempts_before_outcome_filter": n_before,
        "n_dropped_invalid_outcome": n_dropped_invalid,
    }
