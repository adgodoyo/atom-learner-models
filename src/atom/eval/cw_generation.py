"""Numeric-only evaluation helpers for free generation by the CodeWorkout generative simulator.

Raw problem statements, prior submissions, reference submissions and generated code
must remain on transient storage. This module turns one raw generation record into an
all-numeric row before anything is eligible for persistence, and summarises those rows
into marginal cells, learner-clustered paired contrasts and copy baselines.
"""

from __future__ import annotations

import hashlib
import pathlib
from typing import Any

import numpy as np
from rapidfuzz.distance import Levenshtein

from atom.eval.metrics import learner_clustered_bootstrap_ci


ARMS = ("personalised", "dw0", "shuffled_u", "u_mean")
# "argmax": the belief-blind "most likely class" variant beside "ancestral" (sampled).
# It is additive, so a caller that never produces argmax records is unaffected when it
# passes its own `modes`; completed_row_indices/summarise_generation_rows both key off
# the requested modes generically.
MODES = ("conditional_imitation", "ancestral", "argmax")
ARM_TO_CODE = {name: i for i, name in enumerate(ARMS)}
MODE_TO_CODE = {name: i for i, name in enumerate(MODES)}
CODE_TO_ARM = {i: name for name, i in ARM_TO_CODE.items()}
CODE_TO_MODE = {i: name for name, i in MODE_TO_CODE.items()}

NUMERIC_FIELDS = (
    "row_idx",
    "learner_hash",
    "arm_code",
    "mode_code",
    "true_class",
    "conditioned_class",
    "exact_match",
    "normalised_edit_distance",
    "codec_similarity",
    "generated_chars",
    "has_prior",
    "copy_normalised_edit_distance",
    "copy_exact_match",
)


def normalised_edit_distance(a: str, b: str) -> float:
    """Character Levenshtein distance divided by the longer stripped string.

    Uses `rapidfuzz` (C-accelerated), not `atom.eval.belief_swap.edit_distance`:
    that function is a plain O(len(a)*len(b)) pure-Python DP, adequate only because FA
    answers are short (median 3 characters). CodeWorkout code completions are hundreds
    to thousands of characters, and the pure-Python version takes more than 18 minutes
    without finishing a single 10,746-row stratum. Do not swap this back to
    `edit_distance`."""
    a, b = str(a).strip(), str(b).strip()
    return float(Levenshtein.distance(a, b) / max(len(a), len(b), 1))


def dist_n(texts: list[str], n: int) -> float | None:
    """Distinct-n-gram diversity (Li et al. 2016): unique whitespace-token n-grams
    divided by total n-grams, pooled across every string in `texts` -- a CELL-level
    statistic (how varied is this arm/mode's OWN generated set), not a per-row/paired
    metric like every other score in this module. Word/whitespace tokenisation, not the
    model's own subword vocabulary -- code identifiers and punctuation are the natural
    "words" a diversity score should count, and this keeps the metric readable/portable
    without depending on which tokenizer produced the text. Returns `None` if `texts`
    yields zero n-grams (empty input, or every string shorter than `n` tokens) rather
    than raising or silently returning 0.0/1.0, both of which would be misread as a real
    measurement of a degenerate case with no n-grams to be diverse over."""
    if n < 1:
        raise ValueError(f"dist_n: n must be >= 1, got {n}")
    total = 0
    seen: set[tuple[str, ...]] = set()
    for text in texts:
        tokens = str(text).split()
        for i in range(len(tokens) - n + 1):
            seen.add(tuple(tokens[i:i + n]))
            total += 1
    if total == 0:
        return None
    return float(len(seen) / total)


def stable_learner_hash(learner: Any) -> int:
    """Stable, non-reversible 60-bit cluster id suitable for numeric persistence."""
    digest = hashlib.sha256(str(learner).encode("utf-8")).hexdigest()
    return int(digest[:15], 16)


def numeric_generation_row(record: dict, codec_similarity: float) -> dict[str, float | int]:
    """Strip all text and direct identifiers from one raw generation record."""
    generated = str(record["generated_code"]).strip()
    real = str(record["real_completion"]).strip()
    prior = record.get("prior_code")
    has_prior = isinstance(prior, str)
    return {
        "row_idx": int(record["row_idx"]),
        "learner_hash": stable_learner_hash(record["learner"]),
        "arm_code": ARM_TO_CODE[str(record["arm"])],
        "mode_code": MODE_TO_CODE[str(record["generation_mode"])],
        "true_class": int(record["true_class_code"]),
        "conditioned_class": int(record["conditioned_class_code"]),
        "exact_match": float(generated == real),
        "normalised_edit_distance": normalised_edit_distance(generated, real),
        "codec_similarity": float(codec_similarity),
        "generated_chars": int(len(generated)),
        "has_prior": int(has_prior),
        "copy_normalised_edit_distance": (
            normalised_edit_distance(str(prior), real) if has_prior else float("nan")
        ),
        "copy_exact_match": float(str(prior).strip() == real) if has_prior else float("nan"),
    }


def save_numeric_generation_rows(rows: list[dict], path: str | pathlib.Path) -> pathlib.Path:
    """Atomically persist numeric rows.  The resulting NPZ contains no object arrays."""
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    integer_fields = {
        "row_idx", "learner_hash", "arm_code", "mode_code", "true_class",
        "conditioned_class", "generated_chars", "has_prior",
    }
    payload = {
        field: np.asarray(
            [row[field] for row in rows],
            dtype=np.uint64 if field == "learner_hash" else (np.int64 if field in integer_fields else np.float64),
        )
        for field in NUMERIC_FIELDS
    }
    with open(tmp, "wb") as f:
        np.savez_compressed(f, **payload)
    tmp.replace(path)
    return path


def load_numeric_generation_rows(path: str | pathlib.Path) -> list[dict]:
    """Load a file written by :func:`save_numeric_generation_rows`."""
    with np.load(path, allow_pickle=False) as data:
        missing = set(NUMERIC_FIELDS) - set(data.files)
        if missing:
            raise ValueError(f"generation score file is missing fields: {sorted(missing)}")
        n = len(data["row_idx"])
        if any(len(data[field]) != n for field in NUMERIC_FIELDS):
            raise ValueError("generation score columns have inconsistent lengths")
        return [
            {field: data[field][i].item() for field in NUMERIC_FIELDS}
            for i in range(n)
        ]


def completed_row_indices(rows: list[dict], arms: tuple[str, ...] = ARMS,
                          modes: tuple[str, ...] = MODES) -> set[int]:
    """Rows with exactly one record for every arm × mode cell (safe resume boundary).

    `arms`/`modes` are explicit, not merely additive: a caller scoring a RESTRICTED
    cell set (e.g. a run that generates 3 arms and never `u_mean`) must pass the
    SAME `arms`/`modes` it generated with, or every row reads as incomplete (fails
    safe: an unexpectedly-small `expected` set never falsely marks a row complete, and
    an unexpectedly-large one -- e.g. re-reading OLD 2-mode/4-arm data under the
    now-3-mode global default -- correctly reports zero completed rows rather than a
    silently wrong subset)."""
    cells: dict[int, set[tuple[int, int]]] = {}
    counts: dict[int, int] = {}
    for row in rows:
        idx = int(row["row_idx"])
        cells.setdefault(idx, set()).add((int(row["arm_code"]), int(row["mode_code"])))
        counts[idx] = counts.get(idx, 0) + 1
    expected = {(ARM_TO_CODE[a], MODE_TO_CODE[md]) for a in arms for md in modes}
    return {idx for idx, got in cells.items() if got == expected and counts[idx] == len(expected)}


def _json_float(value: float) -> float | None:
    return None if not np.isfinite(value) else float(value)


def _cell(rows: list[dict]) -> dict:
    return {
        "n": len(rows),
        "exact_match_rate": _json_float(np.mean([r["exact_match"] for r in rows])) if rows else None,
        "mean_normalised_edit_distance": (
            _json_float(np.mean([r["normalised_edit_distance"] for r in rows])) if rows else None
        ),
        "mean_codec_similarity": (
            _json_float(np.mean([r["codec_similarity"] for r in rows])) if rows else None
        ),
        "mean_generated_chars": (
            _json_float(np.mean([r["generated_chars"] for r in rows])) if rows else None
        ),
    }


def _paired_ci(rows: list[dict], metric: str, mode_code: int, other_arm_code: int,
               n_boot: int, seed: int, allowed_rows: set[int] | None = None) -> dict | None:
    by_key = {
        (int(r["row_idx"]), int(r["arm_code"])): r
        for r in rows
        if int(r["mode_code"]) == mode_code
        and (allowed_rows is None or int(r["row_idx"]) in allowed_rows)
    }
    row_ids = sorted({idx for idx, arm in by_key if arm == ARM_TO_CODE["personalised"]}
                     & {idx for idx, arm in by_key if arm == other_arm_code})
    if not row_ids:
        return None
    a = np.asarray([by_key[(idx, ARM_TO_CODE["personalised"])][metric] for idx in row_ids], dtype=float)
    b = np.asarray([by_key[(idx, other_arm_code)][metric] for idx in row_ids], dtype=float)
    learners = np.asarray(
        [by_key[(idx, ARM_TO_CODE["personalised"])]["learner_hash"] for idx in row_ids],
        dtype=np.uint64,
    )
    result = learner_clustered_bootstrap_ci(a, b, learners, n_boot=n_boot, seed=seed)
    result["ci95"] = [_json_float(x) for x in result["ci95"]]
    result["point_delta"] = _json_float(result["point_delta"])
    return result


def summarise_generation_rows(rows: list[dict], n_boot: int = 1000, seed: int = 0,
                              arms: tuple[str, ...] = ARMS,
                              modes: tuple[str, ...] = MODES) -> dict:
    """Summarise marginal cells, paired arm contrasts, missed cell, and copy baselines.

    The all-arms-missed subset is selected using *every requested arm's* conditional
    exact-match indicator together.  No single arm gets to choose the subset on which it
    is judged.

    `arms`/`modes` must match what the rows were actually
    generated with -- passed straight to `completed_row_indices` (never a silent
    mismatch; see that function's own docstring) and used to bound the "missed" subset
    and the paired-contrast loop to the REQUESTED cells (e.g. 3 arms, never `u_mean`)
    rather than the global default set.
    """
    completed = completed_row_indices(rows, arms=arms, modes=modes)
    complete_rows = [r for r in rows if int(r["row_idx"]) in completed]
    other_arm_codes = sorted(ARM_TO_CODE[a] for a in arms if a != "personalised")
    requested_mode_codes = {MODE_TO_CODE[md] for md in modes}

    cells = {}
    paired = {}
    for mode_code, mode in CODE_TO_MODE.items():
        if mode_code not in requested_mode_codes:
            continue
        cells[mode] = {}
        paired[mode] = {}
        for arm_code, arm in CODE_TO_ARM.items():
            sub = [r for r in complete_rows
                   if int(r["mode_code"]) == mode_code and int(r["arm_code"]) == arm_code]
            cells[mode][arm] = _cell(sub)
        for metric in ("exact_match", "normalised_edit_distance", "codec_similarity"):
            paired[mode][metric] = {
                CODE_TO_ARM[other]: _paired_ci(
                    complete_rows, metric, mode_code, other, n_boot,
                    seed + 100 * mode_code + 10 * list(("exact_match", "normalised_edit_distance", "codec_similarity")).index(metric) + other,
                )
                for other in other_arm_codes
            }

    conditional = [r for r in complete_rows if int(r["mode_code"]) == MODE_TO_CODE["conditional_imitation"]]
    exact_by_row: dict[int, dict[int, float]] = {}
    for r in conditional:
        exact_by_row.setdefault(int(r["row_idx"]), {})[int(r["arm_code"])] = float(r["exact_match"])
    missed = {
        idx for idx, arm_values in exact_by_row.items()
        if len(arm_values) == len(arms) and all(value == 0.0 for value in arm_values.values())
    }
    missed_cells = {
        arm: _cell([r for r in conditional
                    if int(r["row_idx"]) in missed and int(r["arm_code"]) == arm_code])
        for arm_code, arm in CODE_TO_ARM.items()
    }
    missed_paired = {
        metric: {
            CODE_TO_ARM[other]: _paired_ci(
                complete_rows, metric, MODE_TO_CODE["conditional_imitation"], other,
                n_boot, seed + 500 + 10 * i + other, allowed_rows=missed,
            )
            for other in other_arm_codes
        }
        for i, metric in enumerate(("normalised_edit_distance", "codec_similarity"))
    }

    one_per_row = {}
    for r in complete_rows:
        one_per_row.setdefault(int(r["row_idx"]), r)
    eligible = [r for r in one_per_row.values() if int(r["has_prior"]) == 1]
    copy_baseline = {
        "n_selected_rows": len(one_per_row),
        "n_prior_eligible": len(eligible),
        "n_first_attempt_excluded": len(one_per_row) - len(eligible),
        "mean_prior_to_current_normalised_edit_distance": (
            _json_float(np.mean([r["copy_normalised_edit_distance"] for r in eligible]))
            if eligible else None
        ),
        "prior_equals_current_rate": (
            _json_float(np.mean([r["copy_exact_match"] for r in eligible])) if eligible else None
        ),
    }

    return {
        "schema_version": 1,
        "scope": "partial_generic_student_no_personalisation",
        "n_numeric_records": len(rows),
        "n_completed_generation_rows": len(completed),
        "n_expected_records_per_completed_row": len(arms) * len(modes),
        "cells": cells,
        "paired_ci_learner_clustered_personalised_minus_other": paired,
        "all_arms_missed_conditional": {
            "selection_rule": "conditional exact_match == 0 for all four arms",
            "arm_symmetric": True,
            "n_rows": len(missed),
            "cells": missed_cells,
            "paired_ci_learner_clustered_personalised_minus_other": missed_paired,
        },
        "copy_baseline": copy_baseline,
        "codebleu": "deferred_to_B4",
        "tiktoc_test_vector_f1": "deferred_to_B4",
    }
