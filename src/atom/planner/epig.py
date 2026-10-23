"""Refilter-free EPIG estimator for the planner (thesis Section 5.7, Equation 20).

The acquisition score is computed directly from K posterior samples of the shared latent,
with no simulated reveal and no re-run of the recurrent filter.

**Why three Jensen gaps rather than one.** A tempting one-term estimator is
$I(Y_e;Y_t) \\approx H(\\mathrm{mean}_k\\,p_k(y_e,y_t)) -
\\mathrm{mean}_k H(p_k(y_e,y_t))$ -- the Jensen gap of the JOINT (y_e,y_t) pair's predictive
distribution across the K samples. That single term is not, in general, the mutual
information $I(Y_e;Y_t)$: it is $I((Y_e,Y_t);S)$, the mutual information between the PAIR
and the sample index $S$, which can differ arbitrarily from $I(Y_e;Y_t)$ (worked
counter-example: $S$ uniform over 4 states, $Y_e = S \\bmod 2$, $Y_t = \\lfloor S/2 \\rfloor$
-- both deterministic given $S$, so the joint term equals $H(S) = 2$ bits, while $Y_e,Y_t$
are in fact INDEPENDENT fair coins here, $I(Y_e;Y_t) = 0$). The correct refilter-free
completion follows from the co-information identity for two variables conditionally
independent given a third (true here: $Y_e \\perp Y_t \\mid S$, since each is a Bernoulli
draw driven only by the SAME shared sample $S$, independently, given which sample was
drawn):
    $$I(Y_e;Y_t) = I(Y_e;S) + I(Y_t;S) - I((Y_e,Y_t);S)$$
Every term on the right is a Jensen gap over the SAME K samples -- $I(Y_e;S)$ and $I(Y_t;S)$
are exactly the ordinary (marginal) BALD/Jensen-gap quantity `predict_on_targets` already
computes per item, and $I((Y_e,Y_t);S)$ is the one-term joint quantity above. The estimator
is therefore still computed from the K posterior samples directly, with no refiltering; it
just needs three Jensen gaps instead of one. Verified against a brute-force analytic MI on a toy
with an EXACT discrete posterior in tests/test_planner_epig.py, both that this completion
matches the analytic value and that the literal single-term formula overstates it in
general.

Every function here is generic over plain K-sample probability tensors -- no dependency on
the filter/heads modules -- so the same code path serves both the toy validation
(scripts/pl_1_epig_estimator.py) and the real replay harness, unchanged.

**How the one-term formula fails (verified in
tests/test_planner_epig.py::test_literal_term_ranks_candidates_by_bald_minus_epig).** The
literal one-term formula is not merely "a different, mislabelled quantity" -- its ranking
behaviour is characterisable. Since $I((Y_e,Y_t);S) = \mathrm{binary\_jensen\_gap}(e) +
\mathrm{binary\_jensen\_gap}(t) - \mathrm{sample\_based\_epig}(e,t)$ and the middle term does
not depend on the candidate $e$, ranking CANDIDATES by the literal formula for a fixed target
set is exactly ranking by $\mathrm{binary\_jensen\_gap}(e) - \mathrm{sample\_based\_epig}(e,t)$
-- a candidate's OWN epistemic uncertainty (BALD) minus its TRUE informativeness about the
target (EPIG). Not "EPIG degenerates to BALD" but "BALD with EPIG subtracted": greedy
selection under the literal formula would be biased AWAY from genuinely informative
candidates, in direct proportion to how informative they actually are: a structurally
opposite, nameable failure mode rather than a merely mislabelled quantity.

**Scope of the validation (two claims to keep apart).** The toy validation (tests/test_planner_epig.py, scripts/pl_1_epig_estimator.py)
proves the ESTIMATOR is correct: `sample_based_epig` matches the analytic $I(Y_e;Y_t)$ under
an EXACT discrete posterior, because the toy's states are used directly as the K samples (no
amortised approximation anywhere in that comparison). It says NOTHING about whether K draws
from an amortised Gaussian belief -- the actual output of `sample_u` on a real, trained
filter -- carry real epistemic information about the person being modelled. That is a
separate, substantive empirical question, and it is what the synthetic positive control
(planted latents, through the actual filter/heads; scripts/pl_2_positive_control.py, thesis
Section 4.5) is designed to test. "The
estimator is mathematically sound" and "the belief it is applied to carries a real signal"
are two different claims; this module only established the first.
"""

from __future__ import annotations

import torch


def _entropy_of_probs(*probs: torch.Tensor) -> torch.Tensor:
    """Shannon entropy (nats) of a categorical distribution given as outcome probabilities
    (broadcastable tensors summing to 1 along the implicit outcome axis) -- 2 tensors
    (p, 1-p) for the binary case, 4 for the joint case below."""
    total = torch.zeros_like(probs[0])
    for p in probs:
        pc = p.clamp(1e-12, 1.0)
        total = total - p * pc.log()
    return total


def total_entropy(probs_k: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """H(mean_k p_k) -- the item's own TOTAL (aleatoric + epistemic) predictive entropy
    under the K-sample mixture. This is H(p_e) in the information-bound invariant."""
    mean_p = probs_k.mean(dim=dim)
    return _entropy_of_probs(mean_p, 1 - mean_p)


def binary_jensen_gap(probs_k: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """H(mean_k p_k) - mean_k H(p_k) -- the ordinary (marginal) BALD/Jensen gap for ONE
    item's K-sample predictive probabilities: the epistemic component of its own total
    predictive entropy. Non-negative by Jensen's inequality (entropy is concave), exactly,
    for any finite K -- not merely in expectation."""
    mean_p = probs_k.mean(dim=dim)
    h_total = _entropy_of_probs(mean_p, 1 - mean_p)
    h_cond = _entropy_of_probs(probs_k, 1 - probs_k).mean(dim=dim)
    return (h_total - h_cond).clamp(min=0.0)


def joint_jensen_gap(probs_e_k: torch.Tensor, probs_t_k: torch.Tensor) -> torch.Tensor:
    """The one-term formula of the module docstring: the Jensen gap of the JOINT
    (y_e,y_t) pair's predictive distribution -- I((Y_e,Y_t);S), the sample index's mutual
    information with the PAIR, not I(Y_e;Y_t) itself (see module docstring). `probs_e_k`:
    [K]. `probs_t_k`: [K, n_targets] (or [K], broadcastable). Returns shape matching
    `probs_t_k`'s non-K dims.
    """
    p_e = probs_e_k.unsqueeze(-1) if probs_t_k.dim() > probs_e_k.dim() else probs_e_k
    p_t = probs_t_k
    q11, q10 = p_e * p_t, p_e * (1 - p_t)
    q01, q00 = (1 - p_e) * p_t, (1 - p_e) * (1 - p_t)
    h_total = _entropy_of_probs(q11.mean(0), q10.mean(0), q01.mean(0), q00.mean(0))
    h_cond = _entropy_of_probs(q11, q10, q01, q00).mean(0)
    return (h_total - h_cond).clamp(min=0.0)


def sample_based_epig(probs_e_k: torch.Tensor, probs_t_k: torch.Tensor) -> torch.Tensor:
    """The corrected, refilter-free estimator of I(Y_e;Y_t) per target (module docstring's
    co-information identity): `binary_jensen_gap(e) + binary_jensen_gap(t) -
    joint_jensen_gap(e,t)`. Non-negative EXACTLY (not just in expectation) for any finite K,
    because it is literally computing I(Y_e;Y_t) for the well-defined finite joint
    distribution over (Y_e, Y_t, K-sample-index) that these K samples induce, and mutual
    information between any two random variables is never negative.

    `probs_e_k`: [K] -- candidate e's own predicted-correct probability under each of the
    K shared posterior samples. `probs_t_k`: [K, n_targets] -- each target's predicted-
    correct probability under the SAME K samples (the identical draw `probs_e_k` came
    from). Returns [n_targets].
    """
    jg_e = binary_jensen_gap(probs_e_k)
    jg_t = binary_jensen_gap(probs_t_k, dim=0)
    jg_joint = joint_jensen_gap(probs_e_k, probs_t_k)
    return (jg_e + jg_t - jg_joint).clamp(min=0.0)


def categorical_entropy(probs: torch.Tensor) -> torch.Tensor:
    """Shannon entropy over the last (class) axis, in nats."""
    p = probs.clamp_min(1e-12)
    return -(probs * p.log()).sum(dim=-1)


def categorical_jensen_gap(probs_kc: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """I(Y;S) for a categorical predictive over the finite K-sample committee.

    ``probs_kc`` has a class axis last and a committee/sample axis at ``dim``. The returned
    shape is every non-sample, non-class axis (for planner targets: ``[n_targets]``).
    """
    h_total = categorical_entropy(probs_kc.mean(dim=dim))
    h_cond = categorical_entropy(probs_kc).mean(dim=dim)
    return (h_total - h_cond).clamp(min=0.0)


def joint_jensen_gap_binary_categorical(
    probs_e_k: torch.Tensor, probs_t_kc: torch.Tensor
) -> torch.Tensor:
    """I((Y_e,Y_t);S) for a binary candidate and categorical targets.

    ``probs_e_k`` is ``[K]`` and ``probs_t_kc`` is ``[K,T,C]``. Conditional independence
    given the sample index gives a ``2*C``-class joint distribution per target.
    """
    if probs_e_k.ndim != 1 or probs_t_kc.ndim != 3:
        raise ValueError("expected probs_e_k [K] and probs_t_kc [K,T,C]")
    if probs_e_k.shape[0] != probs_t_kc.shape[0]:
        raise ValueError("candidate and target probabilities must share the same K samples")
    pe = probs_e_k[:, None, None]
    joint = torch.cat((pe * probs_t_kc, (1.0 - pe) * probs_t_kc), dim=-1)
    return categorical_jensen_gap(joint, dim=0)


def sample_based_epig_binary_categorical(
    probs_e_k: torch.Tensor, probs_t_kc: torch.Tensor
) -> torch.Tensor:
    """Refilter-free I(binary candidate; categorical target) for every target.

    This is the exact generalisation of :func:`sample_based_epig` used when targets are
    priced over a categorical answer space:
    ``I(Y_e;S) + I(Y_t;S) - I((Y_e,Y_t);S)`` over the same finite committee.
    """
    jg_e = binary_jensen_gap(probs_e_k)
    jg_t = categorical_jensen_gap(probs_t_kc, dim=0)
    jg_joint = joint_jensen_gap_binary_categorical(probs_e_k, probs_t_kc)
    return (jg_e + jg_t - jg_joint).clamp(min=0.0)


def joint_jensen_gap_categorical_categorical(
    probs_e_kc: torch.Tensor, probs_t_ktc: torch.Tensor
) -> torch.Tensor:
    """``I((Y_e,Y_t);S)`` for categorical candidate and target predictions.

    ``probs_e_kc`` is ``[K,C_e]`` and ``probs_t_ktc`` is ``[K,T,C_t]``.  The
    conditional-independence construction used by the refilter-free planner gives a
    ``C_e * C_t`` joint simplex for each target; flattening that last pair of axes keeps
    the estimator identical to the binary/categorical special case.
    """
    if probs_e_kc.ndim != 2 or probs_t_ktc.ndim != 3:
        raise ValueError("expected probs_e_kc [K,C] and probs_t_ktc [K,T,C]")
    if probs_e_kc.shape[0] != probs_t_ktc.shape[0]:
        raise ValueError("candidate and target probabilities must share the same K samples")
    joint = (
        probs_e_kc[:, None, :, None] * probs_t_ktc[:, :, None, :]
    ).reshape(probs_e_kc.shape[0], probs_t_ktc.shape[1], -1)
    return categorical_jensen_gap(joint, dim=0)


def sample_based_epig_categorical_categorical(
    probs_e_kc: torch.Tensor, probs_t_ktc: torch.Tensor
) -> torch.Tensor:
    """Refilter-free EPIG for categorical candidate and categorical targets."""
    jg_e = categorical_jensen_gap(probs_e_kc, dim=0)
    jg_t = categorical_jensen_gap(probs_t_ktc, dim=0)
    jg_joint = joint_jensen_gap_categorical_categorical(probs_e_kc, probs_t_ktc)
    return (jg_e + jg_t - jg_joint).clamp(min=0.0)


def information_bound_violations(evsi_scores: torch.Tensor, own_total_entropy: torch.Tensor,
                                 tol: float = 1e-6) -> torch.Tensor:
    """EVSI(e) <= H(p_e): a true mutual information can never exceed the (Shannon) entropy
    of either marginal. `own_total_entropy` is `total_entropy(probs_e_k)` for the SAME
    candidate `evsi_scores` was computed for. Returns a bool tensor, True where violated
    beyond floating-point tolerance. A violation here IS the diagnosis (the committee has
    been mis-assembled), not something to clip away."""
    return evsi_scores > (own_total_entropy + tol)
