"""Recover the 50 CW problem statements from their natural public source.

A per-problem vector `s(problem)` built as the mean of learners' CODE vectors would draw
64.1% of its submissions from the held-out F19 cohort, leaking that cohort into every
model that reads it. `s(problem)` is therefore built from the REAL problem-statement
TEXT instead -- the same kind of object FA's `s(q)` already is.

This module owns step 1, **identify**: every CW problem's method name is recoverable
from the STUDENT'S OWN CODE, because CodeWorkout hands every learner an unmodified
method-signature stub (`public <type> <name>(<params>)`) to fill in -- the signature is
platform furniture, not the student's own writing, so extracting it is not "reading the
student's work" in any sense that raises a construct-validity question. Restricted to
S19-TRAIN by the caller (any statistic fitted from data is fit on train only --
identification is itself a statistic, even though its target, the method name, is a
platform constant rather than a per-learner one; keeping to train costs nothing here and
keeps the rule uniform).
"""

from __future__ import annotations

import re

import pandas as pd

# CodeWorkout/CodingBat stubs are always `public [static] <returnType> <name>(...)`,
# any Java return type (primitive, array, generic) -- the modifier is what's load-
# bearing (only the platform-provided stub line is `public`; helper methods students
# add themselves are typically `private`/unmodified, and even if a student added another
# `public` method, taking the FIRST match keeps this aligned with the stub, which always
# comes first in the CodeWorkout template).
METHOD_SIGNATURE_RE = re.compile(r"\bpublic\b\s+(?:static\s+)?[\w\[\]<>]+(?:\s*<[^>]*>)?\s+(\w+)\s*\(")


def extract_method_name(code: str) -> str | None:
    """The first `public ... name(` match, or `None` if the code has no recognisable
    Java method signature (malformed/truncated submissions, a small and expected tail)."""
    m = METHOD_SIGNATURE_RE.search(code)
    return m.group(1) if m else None


def identify_problem_method_names(attempts_with_code: pd.DataFrame) -> pd.DataFrame:
    """One row per `problem_id`: the MODAL extracted method name across every valid-code
    attempt in `attempts_with_code` (caller's job to have already restricted this to
    S19-TRAIN), the agreement fraction (modal count / attempts with SOME extracted name),
    and the submission count backing it -- exactly the three things the recovery task's
    own audit trail needs. `attempts_with_code` must carry `problem_id`/`code`.
    """
    valid = attempts_with_code[
        attempts_with_code["code"].notna() & (attempts_with_code["code"].astype(str).str.strip() != "")
    ].copy()
    valid["method_name"] = valid["code"].map(extract_method_name)

    rows = []
    for pid, sub in valid.groupby("problem_id"):
        named = sub["method_name"].dropna()
        if named.empty:
            rows.append({"problem_id": int(pid), "method_name": None, "agreement": 0.0,
                        "n_submissions": int(len(sub)), "n_named": 0})
            continue
        counts = named.value_counts()
        modal_name = counts.index[0]
        rows.append({
            "problem_id": int(pid), "method_name": modal_name,
            "agreement": float(counts.iloc[0] / len(named)),
            "n_submissions": int(len(sub)), "n_named": int(len(named)),
        })
    return pd.DataFrame(rows).sort_values("problem_id").reset_index(drop=True)


def normalize_statement(text: str) -> str:
    """The comparison form the pre-registered 17/17 gate uses: lowercase, collapse
    whitespace, strip the standard `"Write a function in Java that implements the
    following logic:"` wrapper the pipeline always re-applies -- so wording drift
    between CodeWorkout's lightly-edited release text and the recovered CodingBat
    original doesn't register as a mismatch, while a genuinely different problem still
    will (the two share almost no content words once the wrapper is gone)."""
    t = re.sub(r"\s+", " ", text.strip().lower())
    t = re.sub(r"^write a function in java that implements the following logic:\s*", "", t)
    t = re.sub(r"^write a function in java that\s*", "", t)  # a few CodeWorkout wordings omit "...logic:"
    return t.strip()
