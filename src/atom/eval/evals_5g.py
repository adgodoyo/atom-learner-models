"""Stratified personalisation diagnostics for the mathematics simulator (FA-SIM3), computed
in-run at the end of its evaluation pass.

Factored out as pure functions so the panel is unit-testable without a trained model or a
GPU. Every function operates on per-row arrays (the same arrays a simulator run holds in
memory after its eval pass) and returns a JSON-serialisable dict, using
`learner_clustered_bootstrap_ci` (`atom.eval.metrics`) wherever a paired comparison is
involved.

Four sub-instruments:
  (i)   per-problem-type split of personalised-vs-shuffled-u, both channels
  (ii)  history-length curve of the own-vs-stranger gain
  (iii) LOO learner-blind oracle ceiling, by item training exposure
  (iv)  belief-swap under both scorings (AND-of-directions null 0.25; directed null 0.5)
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from atom.eval.metrics import learner_clustered_bootstrap_ci


def per_type_split(problem_types: Any, learner_ids: Any,
                   personalised_mixture_nll: Any, shuffled_u_mixture_nll: Any,
                   personalised_exact_match: Any, shuffled_u_exact_match: Any,
                   n_boot: int, seed: int) -> dict:
    """(i) Per-problem-type personalised-vs-shuffled-u, on BOTH the outcome channel
    (mixture NLL) and the text channel (exact match), learner-clustered paired CIs per
    cell. `problem_types` may contain NaN (unmapped problem_id) -- those rows are
    reported under `"__unmapped__"` rather than silently dropped."""
    types = pd.Series(problem_types).fillna("__unmapped__").to_numpy()
    out: dict[str, dict] = {}
    for t in pd.unique(types):
        mask = types == t
        n = int(mask.sum())
        if n < 2:
            out[str(t)] = {"n": n, "note": "fewer than 2 rows, CI skipped"}
            continue
        out[str(t)] = {
            "n": n,
            "nll_delta_personalised_minus_shuffled_u": learner_clustered_bootstrap_ci(
                personalised_mixture_nll[mask], shuffled_u_mixture_nll[mask],
                learner_ids[mask], n_boot=n_boot, seed=seed),
            "exact_match_delta_personalised_minus_shuffled_u": learner_clustered_bootstrap_ci(
                personalised_exact_match[mask], shuffled_u_exact_match[mask],
                learner_ids[mask], n_boot=n_boot, seed=seed),
        }
    return out


DEFAULT_HISTORY_BINS = (0, 10, 25, 50, 100, 10_000)  # [0,10) cold start .. [100,inf) mature


def history_curve(t_values: Any, learner_ids: Any,
                  personalised_mixture_nll: Any, shuffled_u_mixture_nll: Any,
                  bins: tuple = DEFAULT_HISTORY_BINS, n_boot: int = 1000, seed: int = 0) -> dict:
    """(ii) History-length curve of the own-vs-stranger gain -- own-vs-stranger gain
    defined as `shuffled_u_mixture_nll - personalised_mixture_nll` (positive = the
    learner's own belief helps). This is the opposite sign convention from `learner_clustered_bootstrap_ci`'s own
    personalised-minus-other, which is why this function computes the CI on the
    OTHER-minus-personalised order directly rather than negating afterwards). Mechanical
    fact asserted, not merely measured: the gain is identically 0 at t=0 (the belief IS
    the prior) -- callers should expect (not be surprised by) a near-zero first bin."""
    t = np.asarray(t_values)
    out = {}
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (t >= lo) & (t < hi)
        n = int(mask.sum())
        label = f"[{lo},{hi if hi < 10_000 else 'inf'})"
        if n < 2:
            out[label] = {"n": n, "note": "fewer than 2 rows, CI skipped"}
            continue
        out[label] = {
            "n": n,
            "gain_ci": learner_clustered_bootstrap_ci(
                shuffled_u_mixture_nll[mask], personalised_mixture_nll[mask],
                learner_ids[mask], n_boot=n_boot, seed=seed),
        }
    return out


def loo_oracle_ceiling(problem_ids: Any, true_class: Any, answer_text: Any,
                       personalised_exact_match: Any, train_exposure_by_problem: dict | None = None,
                       n_exposure_quantiles: int = 4) -> dict:
    """(iii) Leave-one-out learner-blind oracle: for every row whose (problem_id,
    true_class) cell holds >=2 val rows, the LOO modal answer of the OTHER rows in that
    cell is what a model with zero person information could score from population
    statistics alone -- exact-matched against the row's own answer. Reported beside
    ATOM's own exact-match mean, on the SAME covered subset (apples-to-apples), plus the
    gradient by item training exposure if `train_exposure_by_problem` (problem_id ->
    train-row count) is supplied.

    Tie-break for the modal answer: pandas' `Series.mode()`, which returns ties in
    sorted order -- the first is taken, a deterministic (not random) choice, disclosed
    here rather than left implicit."""
    df = pd.DataFrame({
        "problem_id": np.asarray(problem_ids), "true_class": np.asarray(true_class),
        "answer_text": np.asarray(answer_text, dtype=object),
        "personalised_exact_match": np.asarray(personalised_exact_match, dtype=float),
    })
    df["cell"] = list(zip(df["problem_id"], df["true_class"]))
    cell_sizes = df.groupby("cell")["answer_text"].transform("size")
    covered = df[cell_sizes >= 2].copy()
    if len(covered) == 0:
        return {"n_covered": 0, "coverage_frac": 0.0, "note": "no multi-learner cells"}

    loo_correct = np.empty(len(covered), dtype=float)
    for pos, (idx, row) in enumerate(covered.iterrows()):
        others = covered[(covered["cell"] == row["cell"]) & (covered.index != idx)]["answer_text"]
        modal = others.mode()
        loo_correct[pos] = float(len(modal) > 0 and modal.iloc[0] == row["answer_text"])
    covered = covered.assign(loo_correct=loo_correct)

    result = {
        "n_covered": int(len(covered)), "coverage_frac": float(len(covered) / len(df)),
        "n_cells": int(covered["cell"].nunique()),
        "loo_oracle_exact_match_mean": float(covered["loo_correct"].mean()),
        "atom_personalised_exact_match_mean_same_subset":
            float(covered["personalised_exact_match"].mean()),
        "gap_oracle_minus_atom": float(
            covered["loo_correct"].mean() - covered["personalised_exact_match"].mean()),
    }
    if train_exposure_by_problem:
        covered = covered.assign(
            train_exposure=covered["problem_id"].map(train_exposure_by_problem).fillna(0))
        try:
            covered = covered.assign(
                exposure_quartile=pd.qcut(covered["train_exposure"], n_exposure_quantiles,
                                          labels=False, duplicates="drop"))
            by_q = {}
            for q, g in covered.groupby("exposure_quartile"):
                by_q[f"q{int(q)}"] = {
                    "n": int(len(g)),
                    "median_train_exposure": float(g["train_exposure"].median()),
                    "loo_oracle_exact_match_mean": float(g["loo_correct"].mean()),
                    "atom_personalised_exact_match_mean": float(g["personalised_exact_match"].mean()),
                    "gap": float(g["loo_correct"].mean() - g["personalised_exact_match"].mean()),
                }
            result["by_train_exposure_quartile"] = by_q
        except ValueError as e:  # noqa: BLE001 -- too few distinct exposure values to bin
            result["by_train_exposure_quartile"] = {"note": f"qcut failed: {e}"}
    return result


def belief_swap_directed_accuracy(nll_i_own: Any, nll_i_other: Any,
                                  nll_j_own: Any, nll_j_other: Any) -> dict:
    """(iv), the directed half. AND-of-directions scoring (both `i` and `j` must favour
    their own belief for the pair to count) has a mechanical null of 0.25, not 0.5, so
    an AND accuracy below 0.5 is not by itself below chance. The DIRECTED
    scoring (`i` favours own OR `j` favours own, scored as two independent Bernoulli
    trials, one per direction) has the clean 0.5 null with sqrt(2) more statistical
    power, since it does not require both directions to agree. Returns both the AND
    accuracy (comparable with the earlier simulator's report) and the directed accuracy."""
    nll_i_own = np.asarray(nll_i_own, dtype=float)
    nll_i_other = np.asarray(nll_i_other, dtype=float)
    nll_j_own = np.asarray(nll_j_own, dtype=float)
    nll_j_other = np.asarray(nll_j_other, dtype=float)
    i_favours_own = nll_i_own < nll_i_other
    j_favours_own = nll_j_own < nll_j_other
    and_correct = i_favours_own & j_favours_own
    directed = np.concatenate([i_favours_own, j_favours_own]).astype(float)
    return {
        "n_pairs": int(len(nll_i_own)),
        "and_both_directions": {
            "null": 0.25, "accuracy": float(and_correct.mean()) if len(and_correct) else float("nan"),
        },
        "directed": {
            "null": 0.5, "n_directed_comparisons": int(len(directed)),
            "accuracy": float(directed.mean()) if len(directed) else float("nan"),
        },
    }
