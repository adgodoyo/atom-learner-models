"""Tests for `atom.binding.uncertainty`, including the structured (antithetic) committee
sampling used in joint-set training."""

from __future__ import annotations

import torch

from atom.binding.uncertainty import build_structured_committee_eps


def test_structured_committee_eps_shape():
    eps = build_structured_committee_eps(5, 8, 16, generator=torch.Generator().manual_seed(0))
    assert eps.shape == (5, 8, 16)


def test_structured_committee_eps_first_two_are_deterministic_sigma_points_at_magnitude_2():
    eps = build_structured_committee_eps(4, 8, 32, generator=torch.Generator().manual_seed(1))
    torch.testing.assert_close(eps[:, 0], -eps[:, 1])
    torch.testing.assert_close(eps[:, 0].norm(dim=-1), torch.full((4,), 2.0), atol=1e-5, rtol=1e-5)


def test_structured_committee_eps_remaining_draws_are_antithetic_pairs():
    eps = build_structured_committee_eps(3, 8, 10, generator=torch.Generator().manual_seed(2))
    # draws 2/3, 4/5, 6/7 are the three antithetic pairs (K-2=6 -> 3 pairs).
    for lo, hi in [(2, 3), (4, 5), (6, 7)]:
        torch.testing.assert_close(eps[:, lo], -eps[:, hi])
    # distinct pairs are not the same draw (generic randomness, not a hard guarantee, but a
    # collision at float precision across 3 independent N(0,I) draws is not a real risk).
    assert not torch.allclose(eps[:, 2], eps[:, 4])


def test_structured_committee_eps_odd_remainder_gets_one_plain_draw_not_dropped():
    # K=5 -> 2 sigma-points + remaining=3 -> 1 antithetic pair (2 draws) + 1 plain draw = 5.
    eps = build_structured_committee_eps(2, 5, 4, generator=torch.Generator().manual_seed(3))
    assert eps.shape == (2, 5, 4)
    torch.testing.assert_close(eps[:, 2], -eps[:, 3])
    # the 5th draw is not required to pair with anything -- just present and finite.
    assert torch.isfinite(eps[:, 4]).all()


def test_structured_committee_eps_rejects_k_below_2():
    import pytest
    with pytest.raises(ValueError, match="k>=2"):
        build_structured_committee_eps(2, 1, 4)


def test_structured_committee_eps_is_deterministic_given_a_seeded_generator():
    gen_a = torch.Generator().manual_seed(42)
    gen_b = torch.Generator().manual_seed(42)
    eps_a = build_structured_committee_eps(3, 8, 6, generator=gen_a)
    eps_b = build_structured_committee_eps(3, 8, 6, generator=gen_b)
    torch.testing.assert_close(eps_a, eps_b)
