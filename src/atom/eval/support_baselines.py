"""The two behavioural baselines the simulator's class head must beat: learner-prefix
frequencies with shrinkage, and previous-class-plus-elapsed-time-bucket ("lag-1+gap").

Neither baseline needs a trained model: both are closed-form statistics over the manifest
itself, computable independently of the simulator.

Both baselines are **causal within one learner's own chronological sequence**
(`b_t = f(H_<t)`, the same boundary the simulator itself respects): a row's prediction
only ever uses that SAME learner's STRICTLY PRIOR events, never its own label or anything
later. Any GLOBAL statistic (the population prior, the lag/gap lookup table) is fitted on
TRAIN ONLY. The FA learner split is disjoint (a val learner has no train-split history of
their own), so this is the only way a baseline this cheap avoids borrowing information
the model itself is not allowed to use either.

Every function works on any integer-coded class column, not just `support_code`: each
takes an optional `class_col` (default `"support_code"`) and infers `n_classes` from
`len(global_freq)` wherever `global_freq` is already a parameter (only
`fit_global_support_freq`, which builds `global_freq` itself, takes it explicitly). The
5-class outcome label (`atom.data.fa_outcome_label`) reuses this module rather than a
near-duplicate. Callers using a different class column (e.g. `outcome_code`) are
responsible for filtering to that column's OWN validity mask FIRST (e.g. `outcome_valid`):
this module has no invalid-code sentinel handling, by design. `np.bincount`/one-hot both
require non-negative codes, so a caller passing an unfiltered `-1` sentinel would fail
loudly (bincount) or silently wrap (one-hot indexing) rather than being absorbed here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from atom.data.fa_manifest import SUPPORT_CLASSES

N_SUPPORT = len(SUPPORT_CLASSES)

# Elapsed-time buckets in minutes, upper edges (last is +inf): <5min, 5-30min, 30min-2h,
# 2h-1day, >1day. Coarse and round rather than tuned: the filter-body bake-off's finding
# (elapsed time carries ~96% of the event-transformer's attention-bias value) concerns a
# continuous RBF kernel, not this baseline; a handful of round buckets is enough to test whether SUPPORT_CLASSES has
# any lag/gap structure at all, which is all this baseline exists to falsify.
GAP_BUCKET_EDGES_MIN = (5.0, 30.0, 120.0, 1440.0)
N_GAP_BUCKETS = len(GAP_BUCKET_EDGES_MIN) + 1  # + one bucket for >last edge
FIRST_EVENT_BUCKET = -1  # sentinel gap bucket for a learner's first event (gap undefined)
FIRST_EVENT_SUPPORT = -1  # sentinel prev-support code for a learner's first event


def fit_global_support_freq(train_ev: pd.DataFrame, class_col: str = "support_code",
                            n_classes: int = N_SUPPORT) -> np.ndarray:
    """Population class frequency, fit on TRAIN only. (n_classes,), sums to 1."""
    counts = np.bincount(train_ev[class_col].to_numpy().astype(int),
                         minlength=n_classes).astype(float)
    total = counts.sum()
    if total == 0:
        raise ValueError("fit_global_support_freq: train_ev has zero rows")
    return counts / total


def _causal_prior_counts(ev: pd.DataFrame, class_col: str, n_classes: int) -> np.ndarray:
    """(N, n_classes) counts of each class among a row's OWN learner's STRICTLY PRIOR
    events (never including the row's own label). `ev` must already be sorted by
    (learner, t) ascending -- callers own that ordering so this function never silently
    re-sorts (and possibly disagrees with a caller's own row alignment)."""
    codes = ev[class_col].to_numpy().astype(int)
    onehot = np.eye(n_classes)[codes]
    oh_df = pd.DataFrame(onehot, columns=range(n_classes), index=ev.index)
    oh_df["learner"] = ev["learner"].to_numpy()
    cum_inclusive = oh_df.groupby("learner")[list(range(n_classes))].cumsum().to_numpy()
    return cum_inclusive - onehot  # subtract the row's own one-hot -> strictly prior


def learner_prefix_probs(ev: pd.DataFrame, global_freq: np.ndarray, shrinkage_k: float = 5.0,
                         class_col: str = "support_code") -> np.ndarray:
    """Causal, population-shrunk per-learner class frequency baseline.

    `ev` must be sorted by (learner, t) ascending. A learner's own prefix count is
    shrunk toward `global_freq` (fit on TRAIN, see module docstring) with pseudo-count
    `shrinkage_k`: `(prior_counts + k * global_freq) / (n_prior + k)`. At n_prior=0 (a
    learner's first event) this reduces to exactly `global_freq` -- the correct fallback,
    not a special case needing its own branch. `n_classes` is inferred from
    `len(global_freq)`.

    Returns (N, n_classes) rows, each already a valid probability simplex (prior_counts
    sums to n_prior, k * global_freq sums to k, denominator is n_prior + k).
    """
    if shrinkage_k <= 0:
        raise ValueError(f"shrinkage_k must be > 0, got {shrinkage_k}")
    prior_counts = _causal_prior_counts(ev, class_col, len(global_freq))
    n_prior = prior_counts.sum(axis=1, keepdims=True)
    return (prior_counts + shrinkage_k * global_freq[None, :]) / (n_prior + shrinkage_k)


def gap_bucket(gap_log1p_min: np.ndarray, first_event: np.ndarray) -> np.ndarray:
    """Elapsed-time-since-last-event bucket index, or FIRST_EVENT_BUCKET where
    `first_event` is True (gap is undefined there, never a real, if-large, elapsed time)."""
    gap_min = np.expm1(np.where(first_event, 0.0, gap_log1p_min))
    bucket = np.digitize(gap_min, GAP_BUCKET_EDGES_MIN)
    return np.where(first_event, FIRST_EVENT_BUCKET, bucket)


def _prev_class_code(ev: pd.DataFrame, class_col: str) -> np.ndarray:
    """Previous event's class within the SAME learner, or FIRST_EVENT_SUPPORT for a
    learner's own first event. `ev` must be sorted by (learner, t) ascending."""
    prev = ev.groupby("learner")[class_col].shift(1)
    return prev.fillna(FIRST_EVENT_SUPPORT).to_numpy().astype(int)


def fit_lag1_gap_table(train_ev: pd.DataFrame, global_freq: np.ndarray, shrinkage_k: float = 5.0,
                       class_col: str = "support_code") -> dict[tuple[int, int], np.ndarray]:
    """(prev_class_code, gap_bucket) -> shrunk class distribution, fit on TRAIN only.
    Every cell falls back toward `global_freq` at the same `shrinkage_k` as
    `learner_prefix_probs`, so a cell an unseen (or rare) combination lands on at eval
    time is never a fabricated zero-count distribution. `n_classes` is inferred from
    `len(global_freq)`."""
    n_classes = len(global_freq)
    prev_code = _prev_class_code(train_ev, class_col)
    bucket = gap_bucket(train_ev["gap_log1p_min"].to_numpy(),
                        train_ev["first_event"].to_numpy())
    codes = train_ev[class_col].to_numpy().astype(int)

    table: dict[tuple[int, int], np.ndarray] = {}
    keys = np.stack([prev_code, bucket], axis=1)
    for key in {tuple(k) for k in keys.tolist()}:
        mask = (prev_code == key[0]) & (bucket == key[1])
        counts = np.bincount(codes[mask], minlength=n_classes).astype(float)
        table[key] = (counts + shrinkage_k * global_freq) / (counts.sum() + shrinkage_k)
    return table


def lag1_gap_probs(ev: pd.DataFrame, table: dict[tuple[int, int], np.ndarray],
                   global_freq: np.ndarray, class_col: str = "support_code") -> np.ndarray:
    """Apply a `fit_lag1_gap_table` lookup to `ev` (any split). A (prev_code, bucket)
    combination never seen in TRAIN (so absent from `table`) falls back to `global_freq`
    directly -- consistent with what an all-zero-count cell would have shrunk to anyway,
    just without needing every possible key materialised in the table up front."""
    prev_code = _prev_class_code(ev, class_col)
    bucket = gap_bucket(ev["gap_log1p_min"].to_numpy(), ev["first_event"].to_numpy())
    out = np.empty((len(ev), len(global_freq)))
    for i, key in enumerate(zip(prev_code.tolist(), bucket.tolist())):
        out[i] = table.get(key, global_freq)
    return out
