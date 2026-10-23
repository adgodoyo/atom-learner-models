"""Validates
`src/atom/planner/epig.py`'s refilter-free estimator against an ANALYTIC mutual information
on a synthetic Bayes-consistent toy -- a discrete latent $S$ with a UNIFORM prior over $M$
states, used directly as the "K samples" (K=M, unweighted mean over samples = the exact
expectation, zero Monte Carlo error), so `sample_based_epig`'s output can be compared to a
brute-force analytic $I(Y_e;Y_t)$ with no approximation on either side of the comparison.

This is also where the module docstring's claimed correction to the naive one-term
joint formula is verified, not just asserted: `test_literal_joint_term_alone_overstates_mi`
demonstrates the single-term formula's failure mode directly, and
`test_sample_based_epig_matches_analytic_mi_exactly` demonstrates the 3-term completion's
fix, on the SAME toy configurations.
"""

import numpy as np
import pytest
import torch

from atom.planner.epig import (
    binary_jensen_gap,
    categorical_jensen_gap,
    information_bound_violations,
    joint_jensen_gap,
    joint_jensen_gap_binary_categorical,
    sample_based_epig,
    sample_based_epig_binary_categorical,
    total_entropy,
)


def _analytic_mi(p_e_states: np.ndarray, p_t_states: np.ndarray) -> float:
    """Brute-force, independent reference: exact I(Y_e;Y_t) for a uniform M-state discrete
    latent, Y_e|state ~ Bernoulli(p_e_states), Y_t|state ~ Bernoulli(p_t_states),
    conditionally independent given the state. No sampling, no shared code with epig.py."""
    m = len(p_e_states)
    joint = np.zeros((2, 2))
    for pe, pt in zip(p_e_states, p_t_states):
        joint[1, 1] += pe * pt
        joint[1, 0] += pe * (1 - pt)
        joint[0, 1] += (1 - pe) * pt
        joint[0, 0] += (1 - pe) * (1 - pt)
    joint /= m
    marg_e = joint.sum(axis=1)
    marg_t = joint.sum(axis=0)

    def h(p):
        p = np.clip(p, 1e-12, 1.0)
        return float(-np.sum(p * np.log(p)))

    return h(marg_e) + h(marg_t) - h(joint.flatten())


def _analytic_joint_term(p_e_states: np.ndarray, p_t_states: np.ndarray) -> float:
    """The independent reference for the naive one-term joint formula,
    I((Y_e,Y_t);S) -- used to show it is a DIFFERENT quantity from I(Y_e;Y_t) in general."""
    m = len(p_e_states)
    joint = np.zeros((2, 2))
    h_cond = 0.0
    for pe, pt in zip(p_e_states, p_t_states):
        q = np.array([[(1 - pe) * (1 - pt), (1 - pe) * pt], [pe * (1 - pt), pe * pt]])
        joint += q / m
        qc = np.clip(q, 1e-12, 1.0)
        h_cond += -np.sum(q * np.log(qc)) / m
    jc = np.clip(joint, 1e-12, 1.0)
    h_total = -np.sum(joint * np.log(jc))
    return h_total - h_cond


def _analytic_mi_binary_categorical(p_e_states: np.ndarray, p_t_states: np.ndarray) -> float:
    """Independent exact I(binary candidate; categorical target)."""
    joint = np.zeros((2, p_t_states.shape[1]), dtype=np.float64)
    for pe, pt in zip(p_e_states, p_t_states):
        joint[1] += pe * pt
        joint[0] += (1.0 - pe) * pt
    joint /= len(p_e_states)
    pe = joint.sum(axis=1)
    pt = joint.sum(axis=0)

    def h(p):
        p = np.clip(p, 1e-12, 1.0)
        return float(-np.sum(p * np.log(p)))

    return h(pe) + h(pt) - h(joint.reshape(-1))


# Several hand-picked, non-degenerate toy configurations: uniform discrete latent, given
# per-state Bernoulli probabilities for the candidate and (potentially several) targets.
TOY_CONFIGS = [
    ([0.9, 0.1], [0.8, 0.2]),
    ([0.9, 0.1], [0.2, 0.8]),  # target anti-correlated with the candidate's state ordering
    ([0.7, 0.3, 0.5, 0.5], [0.6, 0.4, 0.5, 0.5]),
    ([0.95, 0.05, 0.5], [0.5, 0.5, 0.95]),
    ([0.5, 0.5, 0.5, 0.5], [0.9, 0.1, 0.9, 0.1]),  # candidate carries zero state information
]


def test_sample_based_epig_matches_analytic_mi_exactly():
    """The corrected 3-term estimator (module docstring's co-information identity) must
    reproduce the brute-force analytic I(Y_e;Y_t) on a toy where the posterior is EXACT
    (a uniform discrete latent used directly as the K samples) -- the estimator's acceptance gate."""
    for p_e_states, p_t_states in TOY_CONFIGS:
        probs_e_k = torch.tensor(p_e_states, dtype=torch.float64)
        probs_t_k = torch.tensor(p_t_states, dtype=torch.float64).unsqueeze(1)
        estimate = float(sample_based_epig(probs_e_k, probs_t_k)[0].item())
        analytic = _analytic_mi(np.array(p_e_states), np.array(p_t_states))
        assert estimate == pytest.approx(analytic, abs=1e-9), (p_e_states, p_t_states, estimate, analytic)


def test_binary_candidate_categorical_target_epig_matches_analytic_mi_exactly():
    """Five-class targets: the generalisation is the same MI, not a new heuristic."""
    pe = np.array([0.9, 0.2, 0.7, 0.1], dtype=np.float64)
    pt = np.array([
        [0.70, 0.10, 0.05, 0.10, 0.05],
        [0.05, 0.55, 0.20, 0.10, 0.10],
        [0.45, 0.15, 0.15, 0.15, 0.10],
        [0.05, 0.15, 0.15, 0.25, 0.40],
    ], dtype=np.float64)
    got = sample_based_epig_binary_categorical(torch.tensor(pe), torch.tensor(pt).unsqueeze(1))[0]
    want = _analytic_mi_binary_categorical(pe, pt)
    assert float(got) == pytest.approx(want, abs=1e-9)


def test_categorical_generalisation_reduces_exactly_to_binary_epig():
    rng = np.random.default_rng(91)
    pe = torch.tensor(rng.uniform(0.05, 0.95, size=17), dtype=torch.float64)
    pt = torch.tensor(rng.uniform(0.05, 0.95, size=(17, 4)), dtype=torch.float64)
    pt_cat = torch.stack((1.0 - pt, pt), dim=-1)
    assert torch.allclose(
        sample_based_epig_binary_categorical(pe, pt_cat),
        sample_based_epig(pe, pt), atol=1e-12, rtol=0,
    )
    assert torch.allclose(
        joint_jensen_gap_binary_categorical(pe, pt_cat),
        joint_jensen_gap(pe, pt), atol=1e-12, rtol=0,
    )


def test_categorical_epig_zero_for_an_aleatoric_candidate():
    pe = torch.full((23,), 0.5, dtype=torch.float64)
    raw = torch.arange(23 * 3 * 5, dtype=torch.float64).reshape(23, 3, 5) + 1.0
    pt = raw / raw.sum(dim=-1, keepdim=True)
    score = sample_based_epig_binary_categorical(pe, pt)
    assert torch.allclose(score, torch.zeros_like(score), atol=1e-12, rtol=0)
    assert torch.all(categorical_jensen_gap(pt) >= 0)


def test_literal_joint_term_alone_overstates_mi():
    """Documents the correction: the naive one-term formula (the joint
    Jensen gap alone, `joint_jensen_gap`) computes I((Y_e,Y_t);S), NOT I(Y_e;Y_t) -- and on
    at least one of the toy configs above it must OVERSTATE the true MI, confirming the
    single term is not, by itself, a valid EPIG estimator (why the estimator uses the 3-term
    completion instead)."""
    found_overstatement = False
    for p_e_states, p_t_states in TOY_CONFIGS:
        probs_e_k = torch.tensor(p_e_states, dtype=torch.float64)
        probs_t_k = torch.tensor(p_t_states, dtype=torch.float64).unsqueeze(1)
        literal_term = float(joint_jensen_gap(probs_e_k, probs_t_k)[0].item())
        analytic_joint = _analytic_joint_term(np.array(p_e_states), np.array(p_t_states))
        # our joint_jensen_gap is correct FOR THAT (different) quantity
        assert literal_term == pytest.approx(analytic_joint, abs=1e-9)
        analytic_mi = _analytic_mi(np.array(p_e_states), np.array(p_t_states))
        if literal_term > analytic_mi + 1e-9:
            found_overstatement = True
    assert found_overstatement, "expected at least one toy config to show the single-term overstatement"


def test_literal_term_ranks_candidates_by_bald_minus_epig():
    """Sharpens "the literal formula computes the
    wrong quantity" into a citable characterisation of ITS OWN ranking behaviour. Since
    `joint_jensen_gap(e,t) = binary_jensen_gap(e) + binary_jensen_gap(t) - sample_based_epig(e,t)`
    (the co-information identity), and `binary_jensen_gap(t)` does not depend on e, ranking
    CANDIDATES e by the literal one-term formula for a FIXED target set is exactly ranking
    by `binary_jensen_gap(e) - sample_based_epig(e,t)` -- own epistemic uncertainty (BALD)
    MINUS the true informativeness about the target (EPIG). Not "EPIG degenerates to BALD"
    but "BALD with EPIG subtracted": greedy selection under the literal formula would have
    been biased AWAY from genuinely informative candidates, in proportion to how
    informative they actually are -- a specific, characterisable pathology, not merely a
    noisy or arbitrary one. Verified here via the pairwise differences across several
    candidates against one fixed target (the additive constant, `binary_jensen_gap(t)`,
    cancels in any difference or ranking) -- not just asserted from the identity."""
    rng = np.random.default_rng(7)
    m = 5
    p_t_states = rng.uniform(0.05, 0.95, size=m)
    probs_t_k = torch.tensor(p_t_states, dtype=torch.float64).unsqueeze(1)

    candidates = [rng.uniform(0.02, 0.98, size=m) for _ in range(6)]
    joint_terms, bald_minus_epig = [], []
    for p_e_states in candidates:
        probs_e_k = torch.tensor(p_e_states, dtype=torch.float64)
        joint_terms.append(float(joint_jensen_gap(probs_e_k, probs_t_k)[0].item()))
        bald = float(binary_jensen_gap(probs_e_k).item())
        epig = float(sample_based_epig(probs_e_k, probs_t_k)[0].item())
        bald_minus_epig.append(bald - epig)

    joint_terms = np.array(joint_terms)
    bald_minus_epig = np.array(bald_minus_epig)
    # the additive constant (I(Y_t;S), fixed target) must cancel EXACTLY in every pairwise
    # difference -- this is the precise claim, stronger than "the rankings happen to agree"
    diffs_joint = joint_terms[:, None] - joint_terms[None, :]
    diffs_bald_epig = bald_minus_epig[:, None] - bald_minus_epig[None, :]
    assert np.abs(diffs_joint - diffs_bald_epig).max() < 1e-9

    rank_by_literal_term = np.argsort(-joint_terms)
    rank_by_bald_minus_epig = np.argsort(-bald_minus_epig)
    assert list(rank_by_literal_term) == list(rank_by_bald_minus_epig)


def test_sample_based_epig_is_never_negative():
    """Non-negativity must hold EXACTLY (not just approximately) for any finite K, since
    the 3-term completion literally computes I(Y_e;Y_t) for a well-defined finite joint
    distribution -- true MI is never negative. Checked on random (non-toy-exact) inputs
    too, not just the hand-picked configs."""
    rng = np.random.default_rng(0)
    for _ in range(200):
        k = rng.integers(2, 30)
        n_targets = rng.integers(1, 5)
        probs_e_k = torch.tensor(rng.uniform(0.01, 0.99, size=k))
        probs_t_k = torch.tensor(rng.uniform(0.01, 0.99, size=(k, n_targets)))
        scores = sample_based_epig(probs_e_k, probs_t_k)
        assert torch.all(scores >= -1e-9)


def test_binary_jensen_gap_matches_predict_on_targets_convention():
    """Sanity: binary_jensen_gap on a single item's K-sample probabilities must be
    non-negative and bounded by that item's own total entropy (the same relationship
    replay.py::predict_on_targets already asserts for its own jensen_gap/total_entropy
    pair)."""
    rng = np.random.default_rng(1)
    probs_k = torch.tensor(rng.uniform(0.01, 0.99, size=50))
    jg = binary_jensen_gap(probs_k)
    h = total_entropy(probs_k)
    assert jg >= -1e-9
    assert jg <= h + 1e-9


def test_information_bound_never_violated_on_the_exact_toy():
    """The estimator's acceptance gate: EVSI(e) <= H(p_e) must hold with ZERO violations where the posterior is
    exact (the same toy configs as the MI-matching test)."""
    for p_e_states, p_t_states in TOY_CONFIGS:
        probs_e_k = torch.tensor(p_e_states, dtype=torch.float64)
        probs_t_k = torch.tensor(p_t_states, dtype=torch.float64).unsqueeze(1)
        evsi = sample_based_epig(probs_e_k, probs_t_k)
        h_e = total_entropy(probs_e_k)
        violations = information_bound_violations(evsi, h_e)
        assert not bool(violations.any()), (p_e_states, p_t_states)


def test_information_bound_never_violated_on_random_exact_posteriors():
    """Same invariant, swept over many random uniform-discrete-latent toys (still an EXACT
    posterior, not an amortised approximation) -- the gate's "0 violations" claim must
    survive more than five hand-picked configs."""
    rng = np.random.default_rng(2)
    violations_found = 0
    for _ in range(500):
        m = rng.integers(2, 12)
        p_e_states = rng.uniform(0.02, 0.98, size=m)
        p_t_states = rng.uniform(0.02, 0.98, size=m)
        probs_e_k = torch.tensor(p_e_states)
        probs_t_k = torch.tensor(p_t_states).unsqueeze(1)
        evsi = sample_based_epig(probs_e_k, probs_t_k)
        h_e = total_entropy(probs_e_k)
        violations_found += int(information_bound_violations(evsi, h_e, tol=1e-8).sum().item())
    assert violations_found == 0


def test_sample_based_epig_at_k_equals_one_is_exactly_zero():
    """Degenerate edge case: with a single sample there is no epistemic spread at all --
    all three Jensen-gap terms must be exactly 0, so the estimate is exactly 0."""
    probs_e_k = torch.tensor([0.37])
    probs_t_k = torch.tensor([[0.81, 0.12]])
    scores = sample_based_epig(probs_e_k, probs_t_k)
    assert torch.allclose(scores, torch.zeros_like(scores), atol=1e-9)


def test_sample_based_epig_zero_when_candidate_uninformative():
    """If the candidate's probability is IDENTICAL under every sample (it carries no
    epistemic signal about which sample is true), it can convey nothing about any target,
    however uncertain the target itself is -- EPIG must be ~0 regardless of the target's
    own epistemic spread."""
    probs_e_k = torch.full((20,), 0.6, dtype=torch.float64)
    rng = np.random.default_rng(3)
    probs_t_k = torch.tensor(rng.uniform(0.05, 0.95, size=(20, 3)))
    scores = sample_based_epig(probs_e_k, probs_t_k)
    assert torch.allclose(scores, torch.zeros_like(scores), atol=1e-9)
