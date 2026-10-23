"""Shared held-out evaluation protocol for the FA baselines and every personalised model.

The protocol is fixed in one place so that no comparison can pick a favourable evaluation
window per model. It fixes two things:

1. WHICH events are scored. For each val/test learner, events with t >= that learner's
   calib_threshold (a fixed fraction, `calib_frac`, of their own chronological length) are
   the "eval half" -- the shared held-out cells every model reports its headline AUC on.
   Events with t < threshold are the "calib half": available context, never themselves
   scored. `calib_eval_split` is the single source of truth for this split; every model
   calls it with the same `calib_frac` rather than re-deriving its own eval window.

2. HOW MUCH history each model may condition on -- a deliberate, disclosed asymmetry.
   IRT fits one ability scalar from the calib half and never updates it while scoring the
   eval half (frozen theta: a snapshot ability estimate). A sequential filter may update
   its belief state as it processes eval-half events one at a time; the no-peek rule (each
   prediction uses only that learner's own strictly-earlier events, enforced by the
   t-ordering of the split machinery) holds throughout. When a filter's AUC is compared
   against IRT's frozen-theta AUC on these same cells, the two numbers therefore measure
   different things by design (snapshot ability vs. online belief tracking), and that
   should be stated next to the comparison.
"""

from __future__ import annotations

import pandas as pd


def calib_eval_split(
    ev: pd.DataFrame, calib_frac: float, min_interactions: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (calib_rows, eval_rows) for the val+test learners in `ev`.

    calib_rows: t < per-learner threshold (available context; never scored as predictions).
    eval_rows: t >= threshold -- the shared cells every model's headline AUC is computed on.
    Learners with fewer than `min_interactions` total events are dropped from both (too
    short to support a meaningful split); none are expected to trigger on FA (the minimum
    per-learner interaction count there is 215).
    """
    n_per_learner_row = ev.groupby("learner")["t"].transform("count")
    calib_thresh = (n_per_learner_row * calib_frac).astype(int)
    is_calib = ev["t"] < calib_thresh

    heldout = ev[ev["split"].isin(["val", "test"])].copy()
    heldout["is_calib"] = is_calib[heldout.index]

    learner_counts = ev.groupby("learner").size()  # LEARNER-indexed (unlike transform above)
    ok_learners = set(learner_counts[learner_counts >= min_interactions].index) & set(
        heldout["learner"])
    heldout = heldout[heldout["learner"].isin(ok_learners)]

    return heldout[heldout["is_calib"]], heldout[~heldout["is_calib"]]
