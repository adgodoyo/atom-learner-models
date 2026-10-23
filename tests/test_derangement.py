"""FA-SIM2's shuffled-`u` control needs a genuine derangement, not merely a random
permutation -- FA-5's own shuffle canary was vacuous because `torch.randperm(2)` returns
the identity ~50% of the time. These tests exist to prove `derangement_permutation`
actually fixes that, not just that it runs.
"""

from __future__ import annotations

import pytest
import torch

from atom.binding.derangement import assert_derangement, derangement_permutation


def test_derangement_has_no_fixed_points_at_every_size():
    for n in [2, 3, 4, 5, 8, 16, 50, 200]:
        perm = derangement_permutation(n, generator=torch.Generator().manual_seed(0))
        assert not torch.any(perm == torch.arange(n)), f"fixed point found at n={n}"
        assert sorted(perm.tolist()) == list(range(n)), f"not a valid permutation at n={n}"


def test_derangement_n_equals_2_never_returns_identity_even_across_many_seeds():
    """The exact failure mode this module exists to prevent: torch.randperm(2) IS the
    identity permutation on ~50% of draws (only two permutations exist for n=2). Run
    many seeds and confirm derangement_permutation(2, ...) is NEVER the identity."""
    for seed in range(200):
        perm = derangement_permutation(2, generator=torch.Generator().manual_seed(seed))
        assert perm.tolist() == [1, 0], f"seed {seed} produced {perm.tolist()}, not the only valid derangement of n=2"


def test_torch_randperm_2_would_have_been_vacuous_about_half_the_time():
    """Would-have-caught regression check: confirm the PRE-FIX approach (plain
    torch.randperm) really does return the identity on a large fraction of draws at
    n=2, proving this test class has power to distinguish a real derangement generator
    from a vacuous one."""
    identity_count = 0
    n_trials = 500
    for seed in range(n_trials):
        perm = torch.randperm(2, generator=torch.Generator().manual_seed(seed))
        if torch.equal(perm, torch.arange(2)):
            identity_count += 1
    frac = identity_count / n_trials
    assert frac > 0.3, (
        f"plain torch.randperm(2) was only the identity {frac:.1%} of the time across "
        f"{n_trials} seeds -- expected close to 50%; if this genuinely changed, the "
        "vacuous-shuffle bug this module exists to fix may no longer reproduce, which "
        "would need investigating, not silently accepting")


def test_derangement_raises_for_n_le_1():
    with pytest.raises(ValueError):
        derangement_permutation(0)
    with pytest.raises(ValueError):
        derangement_permutation(1)


def test_derangement_reproducible_with_same_generator_seed():
    a = derangement_permutation(50, generator=torch.Generator().manual_seed(7))
    b = derangement_permutation(50, generator=torch.Generator().manual_seed(7))
    assert torch.equal(a, b)


def test_assert_derangement_passes_on_a_real_derangement():
    perm = derangement_permutation(20, generator=torch.Generator().manual_seed(1))
    assert_derangement(perm)  # must not raise


def test_assert_derangement_fires_on_the_identity():
    identity = torch.arange(10)
    with pytest.raises(AssertionError, match="NOT a derangement"):
        assert_derangement(identity)


def test_assert_derangement_fires_on_a_partial_fixed_point():
    perm = torch.tensor([1, 0, 2, 3])  # index 2 and 3 are fixed points
    with pytest.raises(AssertionError, match="NOT a derangement"):
        assert_derangement(perm)
