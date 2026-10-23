"""The 4-class hierarchical outcome label for CodeWorkout attempts.

The four classes, derived from (compile_result, score_frac) on one attempt row
(`cw_events.build_attempts`'s schema):

    0 COMPILE_ERROR   compile_result == "Error"                     (score_frac forced 0)
    1 CLEAN_ZERO      compile_result == "Success" AND score_frac == 0
    2 CLEAN_PARTIAL   compile_result == "Success" AND 0 < score_frac < 1
    3 CLEAN_FULL      compile_result == "Success" AND score_frac == 1

`P(Score==1) == P(CLEAN_FULL)` by construction, so every gate/base-rate written against
a scalar success head applies unchanged to this class's probability. The
`P(success) <= P(clean compile)` canary is also true by construction here (CLEAN_FULL is
a subset of "Success") and needs no runtime check.

Unlike FA's 5-class label (`fa_outcome_label.py`), CW has no missing-score complication:
measured directly on the full S19+F19 corpus (195,205 attempts) -- `compile_result` is
never null (the event-graph join in `cw_events` has zero join-assertion failures) and `score_frac`
is never null when `compile_result == "Success"` (0/78,262 "Error" rows and 0/116,943
"Success" rows). So every attempt row is `outcome_valid`; the field is still carried
(rather than omitted) to keep this module's call signature symmetric with FA's, and to
absorb a future release where that invariant might not hold without a silent behaviour
change.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

OUTCOME_CLASSES = ("COMPILE_ERROR", "CLEAN_ZERO", "CLEAN_PARTIAL", "CLEAN_FULL")
N_OUTCOME = len(OUTCOME_CLASSES)

# The class-conditioning phrase for the terminal-code production head (mirrors FA's
# `fa_outcome_label.OUTCOME_CLASS_TOKENS` -- one short natural-language phrase per class,
# named for what the token itself communicates, not the internal class name).
# `CW_OUTCOME_CLASS_TOKENS[code]` is always the token for `OUTCOME_CLASSES[code]`.
CW_OUTCOME_CLASS_TOKENS = ("compile error", "compiles, no tests passed",
                          "compiles, some tests passed", "compiles, all tests passed")
assert len(CW_OUTCOME_CLASS_TOKENS) == N_OUTCOME


def outcome_label(compile_result: pd.Series, score_frac: pd.Series) -> pd.DataFrame:
    """Returns a DataFrame with `outcome_code` (int, -1 where invalid -- never silently 0)
    and `outcome_valid` (bool), same index as the inputs."""
    compile_np = compile_result.to_numpy()
    score_np = score_frac.to_numpy(dtype="float64")

    is_error = compile_np == "Error"
    is_clean = compile_np == "Success"
    score_known = ~pd.isna(score_np)

    outcome_valid = is_error | (is_clean & score_known)
    outcome_code = np.full(len(compile_result), -1, dtype=int)
    outcome_code[is_error] = 0
    clean_scored = is_clean & score_known
    outcome_code[clean_scored & (score_np == 0)] = 1
    outcome_code[clean_scored & (score_np > 0) & (score_np < 1)] = 2
    outcome_code[clean_scored & (score_np == 1)] = 3

    return pd.DataFrame({"outcome_code": outcome_code, "outcome_valid": outcome_valid})


def outcome_label_report(compile_result: pd.Series, score_frac: pd.Series) -> dict:
    """Disclosure counts for a report/gate script -- never silently absorbed."""
    labels = outcome_label(compile_result, score_frac)
    counts = labels["outcome_code"].value_counts().reindex(range(N_OUTCOME), fill_value=0)
    n_clean = int((compile_result.to_numpy() == "Success").sum())
    return {
        "n_total": int(len(labels)),
        "n_valid": int(labels["outcome_valid"].sum()),
        "n_invalid": int((~labels["outcome_valid"]).sum()),
        "class_counts": {OUTCOME_CLASSES[i]: int(counts.iloc[i]) for i in range(N_OUTCOME)},
        "class_frac": {OUTCOME_CLASSES[i]: (int(counts.iloc[i]) / len(labels) if len(labels) else 0.0)
                       for i in range(N_OUTCOME)},
        "clean_full_frac_of_clean": (
            int(counts.iloc[3]) / n_clean if n_clean else 0.0),  # == P(Score==1) among Success
    }
