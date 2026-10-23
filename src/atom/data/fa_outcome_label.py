"""The 5-class hierarchical outcome label for FoundationalASSIST events.

Replaces the FOUR-way `support_code` (`atom.data.fa_manifest.SUPPORT_CLASSES`) plus the
separate binary `score` with ONE 5-class label. Measured on a 5% training subsample, the
factored heads did poorly: success (via `h_q`) scored below its constant floor and support
lost to a learner-prefix lookup table, while the belief filter's FiLM correctness head read
the same belief directly at 0.786 AUC. The platform's scoring policy
(P(correct|support!=NONE) < 2%) means three of the four support-conditional success logits
were near-vacuous -- this label collapses them.

The five classes, derived from (support_code, score):
    0 UNAIDED_CORRECT   support_code == NONE  AND score == 1
    1 UNAIDED_WRONG     support_code == NONE  AND score == 0
    2 HINT_ONLY         support_code == HINT_ONLY        (score IGNORED -- see below)
    3 REVEAL_ONLY       support_code == REVEAL_ONLY       (score IGNORED)
    4 HINT_AND_REVEAL   support_code == HINT_AND_REVEAL   (score IGNORED)

Score only SPLITS the unaided class. For the three help-used classes, score is not used to
split further: P(score==1 | support_code != NONE) < 2% in
every one of the three help strata (platform scoring policy -- using a hint/reveal forfeits
credit almost always), so splitting them by score would create three more classes each
carrying almost no second-class mass. The <2% "help-but-credited" rows are not dropped or
relabelled -- they ABSORB into their support class regardless of score, exactly as the
platform's own scoring policy already treats them (a credited hint-use is still a hint-use).
Their count is measured and disclosed (`outcome_label_report`) so this is a measured
absorption, not a silent one.

Validity: `outcome_valid = NOT (support_code == NONE AND score_missing)`. The unaided
classes need the score to tell them apart, so a missing score there is genuinely
unresolvable (excluded, matching the binary success head's `success_valid` convention). The three help
classes never depend on score, so a missing score there is IRRELEVANT to the label -- those
rows stay valid. This is a strict widening of the separate success/support validity contract,
not a narrowing: everything a binary success head could score, this can too.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

OUTCOME_CLASSES = (
    "UNAIDED_CORRECT", "UNAIDED_WRONG", "HINT_ONLY", "REVEAL_ONLY", "HINT_AND_REVEAL",
)
N_OUTCOME = len(OUTCOME_CLASSES)

# The frozen text token each class enters the prompt as (the primary conditioning
# route): ordinary English words the backbone's own pretraining
# already carries "incorrect / hint / revealed" semantics for -- the whole reason this
# route is tried before the learned-soft-token fallback. Indexed by `outcome_code`, so
# `OUTCOME_CLASS_TOKENS[code]` is always the token for `OUTCOME_CLASSES[code]`.
OUTCOME_CLASS_TOKENS = ("correct", "incorrect", "hint used", "answer revealed", "hint used, answer revealed")
assert len(OUTCOME_CLASS_TOKENS) == N_OUTCOME

# support_code -> outcome_code for the three help classes (score-independent, see module
# docstring). support_code 0 (NONE) is handled separately since it depends on score too.
_HELP_SUPPORT_TO_OUTCOME = {1: 2, 2: 3, 3: 4}  # HINT_ONLY, REVEAL_ONLY, HINT_AND_REVEAL


def outcome_label(support_code: pd.Series, score: pd.Series,
                  score_missing: pd.Series) -> pd.DataFrame:
    """Returns a DataFrame with `outcome_code` (int, -1 where invalid -- never silently
    0) and `outcome_valid` (bool), same index as the inputs. `score` may contain NaN on
    invalid rows; only read where `~score_missing`, matching every other reader of this
    column in this codebase (e.g. `fa_train_step.build_targets`)."""
    support_code = support_code.to_numpy()
    score_missing = score_missing.to_numpy().astype(bool)
    score_np = score.to_numpy()

    unaided = support_code == 0
    outcome_valid = ~(unaided & score_missing)

    outcome_code = np.full(len(support_code), -1, dtype=int)
    # unaided, score known: 0 if correct, 1 if wrong.
    unaided_scored = unaided & ~score_missing
    outcome_code[unaided_scored] = (score_np[unaided_scored] != 1).astype(int)  # 0 or 1
    # help classes: score-independent, absorb regardless of score/score_missing.
    for code, outcome in _HELP_SUPPORT_TO_OUTCOME.items():
        outcome_code[support_code == code] = outcome

    return pd.DataFrame({"outcome_code": outcome_code, "outcome_valid": outcome_valid})


def outcome_label_report(support_code: pd.Series, score: pd.Series,
                         score_missing: pd.Series) -> dict:
    """Disclosure counts for a report/gate script -- never silently absorbed. `n_help_credited`
    is the "<2% help-but-credited mass", measured and reported rather than hidden."""
    labels = outcome_label(support_code, score, score_missing)
    support_np = support_code.to_numpy()
    score_np = score.to_numpy()
    help_mask = support_np != 0
    n_help = int(help_mask.sum())
    n_help_credited = int(((score_np == 1) & help_mask & ~score_missing.to_numpy()).sum())
    counts = labels["outcome_code"].value_counts().reindex(range(N_OUTCOME), fill_value=0)
    return {
        "n_total": int(len(labels)),
        "n_valid": int(labels["outcome_valid"].sum()),
        "n_invalid_unaided_score_missing": int((~labels["outcome_valid"]).sum()),
        "class_counts": {OUTCOME_CLASSES[i]: int(counts.iloc[i]) for i in range(N_OUTCOME)},
        "n_help_total": n_help,
        "n_help_credited": n_help_credited,
        "help_credited_frac": (n_help_credited / n_help) if n_help else 0.0,
    }
