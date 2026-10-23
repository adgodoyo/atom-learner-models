#!/usr/bin/env python
"""Validates the planner's EPIG estimator (§5.7) on a synthetic toy with an exact posterior.

The toy uses a discrete latent with a uniform prior over M states directly as the "K samples",
so the comparison with a brute-force analytic mutual information has zero Monte Carlo error on
either side. It checks:

  (i)  the value V(b) is the K-sample Jensen gap (`src/atom/planner/evsi.py::_value`) --
       epistemic-only by construction (Jensen's inequality for entropy); the argument that the
       aleatoric floor cancels holds only for a Bayes-consistent posterior. (Covered by the
       planner test suite, not exercised by this toy.)
  (ii) the refilter-free estimator (`src/atom/planner/epig.py::sample_based_epig`) computed
       directly from K posterior samples, with no simulated reveal and no re-run of the
       recurrent filter. A single joint Jensen-gap term computes I((Y_e,Y_t);S), not I(Y_e;Y_t);
       the correct refilter-free form is the co-information identity I(Y_e;Y_t) =
       I(Y_e;S) + I(Y_t;S) - I((Y_e,Y_t);S), all three terms plain Jensen gaps over the same
       K samples. The script also measures how the one-term form overstates MI and how it ranks
       candidates (by BALD(e) - EPIG(e,t)).
  (iii) the information-bound invariant EVSI(e) <= H(p_e), logged per candidate.

Check: invariant violations = 0 on the exact-posterior toy, and the estimator matches the
analytic MI there. tests/test_planner_epig.py pins the same properties; this script re-derives
them independently and writes `reports/pl_1_epig_estimator.md`.
Run: `python scripts/pl_1_epig_estimator.py` (CPU, seconds).
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from atom.eval.report import gate_report  # noqa: E402
from atom.planner.epig import (  # noqa: E402
    binary_jensen_gap,
    information_bound_violations,
    joint_jensen_gap,
    sample_based_epig,
    total_entropy,
)
from atom.utils.logging import RunLogger  # noqa: E402
from atom.utils.seeds import seed_everything  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[1]

TOY_CONFIGS = [
    ([0.9, 0.1], [0.8, 0.2]),
    ([0.9, 0.1], [0.2, 0.8]),
    ([0.7, 0.3, 0.5, 0.5], [0.6, 0.4, 0.5, 0.5]),
    ([0.95, 0.05, 0.5], [0.5, 0.5, 0.95]),
    ([0.5, 0.5, 0.5, 0.5], [0.9, 0.1, 0.9, 0.1]),
]


def _analytic_mi(p_e_states: np.ndarray, p_t_states: np.ndarray) -> float:
    """Independent brute-force reference (no shared code with epig.py)."""
    m = len(p_e_states)
    joint = np.zeros((2, 2))
    for pe, pt in zip(p_e_states, p_t_states):
        joint[1, 1] += pe * pt
        joint[1, 0] += pe * (1 - pt)
        joint[0, 1] += (1 - pe) * pt
        joint[0, 0] += (1 - pe) * (1 - pt)
    joint /= m
    marg_e, marg_t = joint.sum(axis=1), joint.sum(axis=0)

    def h(p):
        p = np.clip(p, 1e-12, 1.0)
        return float(-np.sum(p * np.log(p)))

    return h(marg_e) + h(marg_t) - h(joint.flatten())


def main() -> int:
    seed_everything(0)
    logger = RunLogger(config={"task": "pl_1_epig_estimator"}, data_version="none", seed=0)

    max_abs_error = 0.0
    max_literal_overstatement = 0.0
    n_bound_violations = 0
    per_config = []

    for p_e_states, p_t_states in TOY_CONFIGS:
        probs_e_k = torch.tensor(p_e_states, dtype=torch.float64)
        probs_t_k = torch.tensor(p_t_states, dtype=torch.float64).unsqueeze(1)

        estimate = float(sample_based_epig(probs_e_k, probs_t_k)[0].item())
        literal_term = float(joint_jensen_gap(probs_e_k, probs_t_k)[0].item())
        analytic = _analytic_mi(np.array(p_e_states), np.array(p_t_states))
        h_e = float(total_entropy(probs_e_k).item())
        violated = bool(information_bound_violations(torch.tensor([estimate]), torch.tensor([h_e])).any())

        max_abs_error = max(max_abs_error, abs(estimate - analytic))
        max_literal_overstatement = max(max_literal_overstatement, literal_term - analytic)
        n_bound_violations += int(violated)
        per_config.append((p_e_states, p_t_states, estimate, literal_term, analytic, h_e, violated))

    # Ranking behaviour of the one-term form: ranking candidates by the literal one-term
    # formula, for a fixed target, is exactly ranking by binary_jensen_gap(e) -
    # sample_based_epig(e,t) -- BALD own-uncertainty minus true EPIG, not a merely-different
    # quantity. Verified via pairwise differences (the additive constant, binary_jensen_gap
    # of the fixed target, must cancel exactly), not just the identity asserted.
    rng_rank = np.random.default_rng(7)
    m_rank = 5
    p_t_fixed = rng_rank.uniform(0.05, 0.95, size=m_rank)
    probs_t_fixed_k = torch.tensor(p_t_fixed, dtype=torch.float64).unsqueeze(1)
    rank_candidates = [rng_rank.uniform(0.02, 0.98, size=m_rank) for _ in range(6)]
    joint_terms_rank, bald_minus_epig_rank = [], []
    for p_e_states in rank_candidates:
        probs_e_k = torch.tensor(p_e_states, dtype=torch.float64)
        joint_terms_rank.append(float(joint_jensen_gap(probs_e_k, probs_t_fixed_k)[0].item()))
        bald = float(binary_jensen_gap(probs_e_k).item())
        epig = float(sample_based_epig(probs_e_k, probs_t_fixed_k)[0].item())
        bald_minus_epig_rank.append(bald - epig)
    joint_terms_rank = np.array(joint_terms_rank)
    bald_minus_epig_rank = np.array(bald_minus_epig_rank)
    diffs_a = joint_terms_rank[:, None] - joint_terms_rank[None, :]
    diffs_b = bald_minus_epig_rank[:, None] - bald_minus_epig_rank[None, :]
    ranking_pairwise_diff_max = float(np.abs(diffs_a - diffs_b).max())
    rankings_match = list(np.argsort(-joint_terms_rank)) == list(np.argsort(-bald_minus_epig_rank))

    # sweep: many random EXACT-posterior toys, invariant only
    rng = np.random.default_rng(2)
    n_swept = 500
    for _ in range(n_swept):
        m = rng.integers(2, 12)
        probs_e_k = torch.tensor(rng.uniform(0.02, 0.98, size=m))
        probs_t_k = torch.tensor(rng.uniform(0.02, 0.98, size=m)).unsqueeze(1)
        evsi = sample_based_epig(probs_e_k, probs_t_k)
        h_e = total_entropy(probs_e_k)
        n_bound_violations += int(information_bound_violations(evsi, h_e, tol=1e-8).sum().item())

    estimator_matches = max_abs_error < 1e-6
    literal_term_overstates = max_literal_overstatement > 1e-6
    zero_violations = n_bound_violations == 0
    passed = estimator_matches and zero_violations

    logger.log_metrics(max_abs_error=max_abs_error, max_literal_overstatement=max_literal_overstatement,
                       n_bound_violations=n_bound_violations, n_configs=len(TOY_CONFIGS), n_swept=n_swept,
                       ranking_pairwise_diff_max=ranking_pairwise_diff_max, rankings_match=rankings_match,
                       passed=passed)
    logger.summary(status="ok" if passed else "fail", passed=passed)

    metrics = {
        "(ii) sample_based_epig max |estimate - analytic MI| over hand-picked configs": max_abs_error,
        "(ii) literal one-term formula (joint_jensen_gap alone) max overstatement vs analytic MI":
            max_literal_overstatement,
        "(ii) literal single-term formula overstates MI on >=1 config (confirms the correction is needed)":
            "YES" if literal_term_overstates else "NO",
        "(iii) information-bound violations, 5 configs + 500 random exact-posterior toys": n_bound_violations,
        "SHARPER FINDING: literal-term pairwise-difference vs (BALD(e)-EPIG(e,t)) pairwise-difference, max mismatch":
            ranking_pairwise_diff_max,
        "SHARPER FINDING: candidate ranking by literal term == ranking by BALD(e)-EPIG(e,t), for a fixed target":
            "YES" if rankings_match else "NO",
    }
    for i, (pe, pt, est, lit, ana, h_e, viol) in enumerate(per_config):
        metrics[f"config {i}: p_e={pe} p_t={pt}"] = (
            f"3-term={est:.6f} literal={lit:.6f} analytic={ana:.6f} H(p_e)={h_e:.6f} "
            f"bound_violated={viol}")

    gate_report(
        path="reports/pl_1_epig_estimator.md",
        title="PL-1 - Refilter-free EPIG estimator on a synthetic Bayes-consistent toy",
        meta={"run_id": logger.run_id, "git_sha": logger.git_sha, "config_hash": logger.config_hash,
             "n_hand_picked_configs": len(TOY_CONFIGS), "n_random_swept_configs": n_swept},
        metrics=metrics,
        passed=passed,
        gate_desc="Estimator check: invariant violations = 0 on a synthetic Bayes-consistent toy; "
                 "estimator matches analytic MI there. Fails if: violations occur where the posterior is exact.",
        notes=[
            "(i) V(b) promoted to the K-sample Jensen gap: src/atom/planner/evsi.py::_value, one line, "
            "not exercised by this toy script (that change is orthogonal to the estimator being validated "
            "here) -- covered by the planner test suite.",
            "(ii) The one-term formula "
            "(H(mean_k joint) - mean_k H(joint), here `joint_jensen_gap`) computes "
            "I((Y_e,Y_t);S) -- the sample index's MI with the PAIR -- not I(Y_e;Y_t). This script confirms "
            "the discrepancy directly (see the literal-term-overstates-MI metric) and validates the correct "
            "refilter-free completion instead: I(Y_e;Y_t) = I(Y_e;S) + I(Y_t;S) - I((Y_e,Y_t);S), all three "
            "terms plain Jensen gaps over the SAME K samples (src/atom/planner/epig.py::sample_based_epig).",
            "Toy construction: a discrete latent with a UNIFORM prior over M states, used directly as the K "
            "samples (K=M) -- an EXACT posterior by construction, not an approximation, so both sides of "
            "every comparison (estimator vs analytic MI) are computed with zero Monte Carlo error.",
            "Same properties pinned permanently in tests/test_planner_epig.py (8 tests): exact match to "
            "analytic MI, the literal-term overstatement, non-negativity (exact, not just in expectation, "
            "for any finite K -- proven via the same co-information identity applied to the finite K-sample "
            "empirical distribution), the K=1 and uninformative-candidate degenerate cases, and the bound "
            "invariant on both the hand-picked configs and 500 random exact-posterior sweeps.",
            "SHARPER FINDING: the literal one-term formula is not merely a "
            "different quantity -- its RANKING behaviour is characterisable. Since I((Y_e,Y_t);S) = "
            "binary_jensen_gap(e) + binary_jensen_gap(t) - sample_based_epig(e,t) and the middle term does "
            "not depend on the candidate e, ranking candidates by the literal formula for a FIXED target "
            "set is exactly ranking by binary_jensen_gap(e) - sample_based_epig(e,t) -- a candidate's OWN "
            "epistemic uncertainty (BALD) MINUS its true informativeness about the target (EPIG). Not 'EPIG "
            "degenerates to BALD' but 'BALD with EPIG subtracted': greedy under the literal formula would be "
            "biased AWAY from genuinely informative candidates, in proportion to how informative they "
            "actually are -- a nameable failure mode, not merely a noisy one, and directly relevant to "
            "the EPIG-vs-BALD contrast. Verified above via pairwise differences across several "
            "candidates against one fixed target (the additive constant cancels exactly), not just asserted "
            "from the identity; permanently pinned as "
            "tests/test_planner_epig.py::test_literal_term_ranks_candidates_by_bald_minus_epig.",
            "PHRASING CAUTION (keep these two claims apart in any write-up): "
            "this toy proves the ESTIMATOR is correct under an EXACT discrete posterior (the toy's states "
            "are used directly as the K samples, no amortised approximation anywhere in the comparison). It "
            "says NOTHING about whether K draws from an amortised GAUSSIAN belief -- what `sample_u` "
            "actually produces on a real, trained S1 filter -- carry real epistemic information about the "
            "person being modelled. 'The estimator is mathematically sound' and 'the belief it is applied "
            "to carries a real signal' are different claims; only the first is established here. The second "
            "is tested by the planted-world positive control (pl_2_positive_control.py, planted latents "
            "through the ACTUAL filter/heads) and by the real-data runs.",
        ],
    )
    print(f"EPIG estimator check: {'PASS' if passed else 'FAIL'} -- report: reports/pl_1_epig_estimator.md")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
