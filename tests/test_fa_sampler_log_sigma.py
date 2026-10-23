"""Tests targeting the failure modes of a sampler fed log_sigma where sigma is expected,
not just the formula in isolation. `reparameterize`'s own contract tests
(`test_binding.py`) cannot catch a caller passing log_sigma where sigma was expected --
that mistake is silently "valid" arithmetic (sigma*eps still type-checks when sigma is
secretly log_sigma), so the tests here are chosen specifically to fail loudly on THAT
class of mistake:

1. monotonicity -- the test that catches the inversion. Under that mistake, the
   effective noise scale is |log_sigma| itself, which is NOT monotonic in log_sigma (it
   falls from 5 to 0 across [-5, 0], then rises again across [0, 2]) -- a real, structural
   difference from the intended exp(log_sigma), which is monotonic everywhere.
2. empirical std matches exp(log_sigma) at representative points across the clamp range.
3. the caller-convention guard actually fires (uncertainty.py's new non-negative check).
4. reparameterize_from_log_sigma and sample_u agree once their K-axis layouts are aligned
   -- sample_u is documented as "a thin layout wrapper", so this is the identity gate that
   makes that documentation a checked claim, not a comment.
5. gradients reach both mu and log_sigma through the new canonical function.
"""

from __future__ import annotations

import torch

from atom.binding.uncertainty import reparameterize, reparameterize_from_log_sigma
from atom.filter.gru import sample_u
from atom.simulator.sampler_semantics import (
    CURRENT_SAMPLER_SEMANTICS,
    require_current_sampler_semantics,
)


def _empirical_std(mu: torch.Tensor, log_sigma: torch.Tensor, k: int,
                   generator: torch.Generator) -> torch.Tensor:
    u = reparameterize_from_log_sigma(mu, log_sigma, k, generator=generator)  # (..., k, d)
    return u.std(dim=-2, unbiased=False)


def test_empirical_std_matches_exp_log_sigma_at_clamp_extremes_and_zero():
    # KnowledgeFilter's own clamp bounds (gru.py): [-5, 2]. log_sigma=0 is the KL's own
    # trivial minimum -- exactly where the log_sigma-as-sigma mistake makes the sampled
    # spread vanish, so it must be covered explicitly.
    mu = torch.zeros(2000, 4)
    for log_sigma_value in (-5.0, -2.0, 0.0, 2.0):
        log_sigma = torch.full((2000, 4), log_sigma_value)
        g = torch.Generator().manual_seed(0)
        std = _empirical_std(mu, log_sigma, k=64, generator=g)
        expected = torch.exp(torch.tensor(log_sigma_value))
        assert torch.allclose(std.mean(), expected, atol=0.05, rtol=0.05), (
            f"log_sigma={log_sigma_value}: empirical std {std.mean().item():.4f} vs "
            f"expected exp(log_sigma)={expected.item():.4f}")


def test_monotonicity_increasing_log_sigma_always_increases_empirical_spread():
    """THE test that catches the inversion. Fixed common-random-numbers eps across every
    log_sigma value in the sweep, so this is a deterministic monotonicity check on one noise
    draw, not a statistical claim needing many samples (the same common-random-numbers
    pattern the planner replays use). Under the mistake (effective scale = |log_sigma|),
    this sweep would be NON-monotonic: falling from
    log_sigma=-5 to log_sigma=0, then rising again from 0 to +2 -- a U-shape, not a line.
    """
    mu = torch.zeros(1, 8)
    eps = torch.randn(1, 1, 8, generator=torch.Generator().manual_seed(0))
    log_sigma_sweep = torch.linspace(-5.0, 2.0, steps=15)
    spreads = []
    for log_sigma_value in log_sigma_sweep:
        log_sigma = torch.full((1, 8), float(log_sigma_value))
        u = reparameterize_from_log_sigma(mu, log_sigma, k=1, eps=eps)
        spreads.append((u - mu.unsqueeze(-2)).abs().mean().item())
    for a, b in zip(spreads, spreads[1:]):
        assert b >= a - 1e-6, (
            f"spread must be non-decreasing in log_sigma (same eps throughout); "
            f"got {spreads}")
    # not just non-decreasing -- the endpoints must differ by exp(2)/exp(-5), a ~1096x
    # ratio, not the mistaken ~0.4x ratio (|2|/|-5|).
    assert spreads[-1] / spreads[0] > 100


def test_reparameterize_rejects_negative_sigma_as_a_caller_convention_guard():
    """The permanent guard added to `reparameterize` itself: sigma is a scale and can
    never be negative, so a caller passing raw log_sigma (which routinely goes negative --
    the clamp floor is -5) is caught immediately, not after a full training run."""
    mu = torch.zeros(2, 3)
    log_sigma_shaped_input = torch.tensor([[-1.0, 0.5, -0.2], [0.1, -3.0, 2.0]])
    try:
        reparameterize(mu, log_sigma_shaped_input, k=1)
        raise AssertionError("expected ValueError for negative sigma entries")
    except ValueError as e:
        assert "negative" in str(e)


def test_reparameterize_from_log_sigma_never_trips_its_own_guard():
    # the whole point of the canonical function: log_sigma is free to be negative (it
    # usually is), because it gets exponentiated before reaching reparameterize's guard.
    mu = torch.zeros(2, 3)
    log_sigma = torch.tensor([[-5.0, 0.0, 2.0], [-1.0, -0.001, 1.5]])
    out = reparameterize_from_log_sigma(mu, log_sigma, k=4)
    assert out.shape == (2, 4, 3)


def test_sample_u_and_reparameterize_from_log_sigma_agree_after_aligning_k_axis():
    """`sample_u` (gru.py) is documented as a thin layout wrapper over
    `reparameterize_from_log_sigma` -- this is the identity gate that checks that claim.
    Both entry points, given the SAME common-random-numbers eps, must produce bit-identical
    samples once sample_u's leading-K layout is moved to reparameterize_from_log_sigma's
    trailing-before-d layout (or vice versa)."""
    torch.manual_seed(0)
    mu = torch.randn(5, 6, 7)
    log_sigma = torch.randn(5, 6, 7).clamp(-5, 2)
    k = 8
    noise = torch.randn(k, 7)  # sample_u's own [K, L] CRN contract

    u_sample_u = sample_u(mu, log_sigma, k, noise=noise)  # [k, 5, 6, 7]
    eps_for_canonical = noise.view(1, 1, k, 7)  # broadcast against (*mu.shape[:-1], k, 7)
    u_canonical = reparameterize_from_log_sigma(mu, log_sigma, k, eps=eps_for_canonical)  # [5,6,k,7]

    assert torch.equal(u_sample_u, u_canonical.movedim(-2, 0))


def test_sample_u_non_crn_path_matches_intended_sigma_not_the_bug():
    # regression guard: sample_u's own path must show empirical std ~ exp(log_sigma), not
    # |log_sigma| -- checked directly rather than taken on trust.
    mu = torch.zeros(4000, 3)
    log_sigma = torch.full((4000, 3), -1.5)
    u = sample_u(mu, log_sigma, k=1)  # [1, 4000, 3]
    empirical_std = u[0].std(dim=0)
    assert torch.allclose(empirical_std, torch.exp(torch.tensor(-1.5)).expand(3),
                          atol=0.05)
    # the mistaken reading (sigma == log_sigma directly) would give std ~ 1.5, not ~0.223
    assert (empirical_std - 1.5).abs().mean() > 1.0


def test_gradients_flow_to_mu_and_log_sigma_through_the_canonical_function():
    mu = torch.zeros(2, 3, requires_grad=True)
    log_sigma = torch.zeros(2, 3, requires_grad=True)
    u = reparameterize_from_log_sigma(mu, log_sigma, k=4,
                                      generator=torch.Generator().manual_seed(0))
    u.sum().backward()
    assert mu.grad is not None and mu.grad.abs().sum() > 0
    assert log_sigma.grad is not None and log_sigma.grad.abs().sum() > 0


def test_reparameterize_from_log_sigma_shape_matches_reparameterize_contract():
    # same output layout as reparameterize itself (K second-to-last) -- the "one canonical
    # implementation" is a thin exp() wrapper, not a parallel API with its own conventions.
    mu = torch.randn(3, 4)
    log_sigma = torch.randn(3, 4).clamp(-5, 2)
    out = reparameterize_from_log_sigma(mu, log_sigma, k=6)
    assert out.shape == (3, 6, 4)


# -- the sampler_semantics guard (`atom/simulator/sampler_semantics.py`) ----------------

def test_require_current_sampler_semantics_rejects_missing_field():
    try:
        require_current_sampler_semantics({"model_state_dict": {}}, context="fake.pt")
        raise AssertionError("expected ValueError for a checkpoint with no sampler_semantics")
    except ValueError as e:
        assert "sampler_semantics" in str(e) and "fake.pt" in str(e)


def test_require_current_sampler_semantics_rejects_stale_version():
    try:
        require_current_sampler_semantics({"sampler_semantics": "some_old_version"})
        raise AssertionError("expected ValueError for a stale sampler_semantics version")
    except ValueError as e:
        assert "does not match" in str(e)


def test_require_current_sampler_semantics_accepts_the_current_version():
    require_current_sampler_semantics({"sampler_semantics": CURRENT_SAMPLER_SEMANTICS})  # no raise
