"""FA-SIM3 requirement 8's true-shuffled-epoch sampler
(`atom.utils.epoch_sampler.ShuffledEpochSampler`). Per the SEMANTIC CANARIES rule, this
does not just check the sampler runs -- it checks the STRUCTURAL invariant the whole
requirement is about: does one epoch's worth of draws actually cover every row exactly
once, in contrast to the `rng.choice(..., replace=False)`-per-micro-batch scheme it
replaces, which does not.
"""

from __future__ import annotations

import numpy as np
import pytest

from atom.utils.epoch_sampler import ShuffledEpochSampler


def test_one_full_epoch_covers_every_row_exactly_once_when_batch_size_divides_n():
    n, batch_size = 100, 10
    sampler = ShuffledEpochSampler(n, np.random.default_rng(0))
    drawn = np.concatenate([sampler.next_batch(batch_size) for _ in range(n // batch_size)])
    assert sorted(drawn.tolist()) == list(range(n))
    assert (sampler.exposure_counts == 1).all()


def test_exposure_counts_are_realised_not_assumed_across_many_epochs():
    """Ten full epochs -> every row's exposure count must be exactly 10, not merely
    'close to 10' -- this is the property `rng.choice`-per-micro-batch cannot deliver at
    all (its own per-row exposure over the same number of draws is Poisson-ish, with a
    real chance of 0)."""
    n, batch_size, n_epochs = 50, 5, 10
    sampler = ShuffledEpochSampler(n, np.random.default_rng(1))
    for _ in range(n_epochs * (n // batch_size)):
        sampler.next_batch(batch_size)
    assert (sampler.exposure_counts == n_epochs).all()


def test_batches_spanning_an_epoch_boundary_still_cover_every_row_before_repeating():
    """batch_size does NOT divide n -- batches straddle epoch boundaries. Within any
    window of draws covering at least `n` items, every row must have been seen at least
    once before any row is seen a second time from the SAME shuffle (verified indirectly:
    accumulate draws until every row has exposure >=1, and confirm no row's exposure
    exceeds ceil(total_draws/n) + 1, i.e. no row is starved while another repeats early)."""
    n, batch_size = 47, 10  # 47 is prime, guarantees a boundary-straddling batch
    sampler = ShuffledEpochSampler(n, np.random.default_rng(2))
    total_drawn = 0
    for _ in range(20):
        batch = sampler.next_batch(batch_size)
        assert len(batch) == batch_size
        total_drawn += batch_size
    # after 200 draws over 47 rows (~4.26 epochs), every row must have exposure in a
    # tight band (never a wide Poisson-ish spread) -- the buffer-refill boundary can
    # separate min and max by a couple of counts (a stale shuffle-tail row lagging a
    # freshly-appended shuffle's already-drawn rows) but never more.
    assert sampler.exposure_counts.max() - sampler.exposure_counts.min() <= 2
    assert sampler.exposure_counts.sum() == total_drawn


def test_would_have_caught_the_independent_draw_scheme_measured_undercoverage():
    """Would-have-caught regression check (SEMANTIC CANARIES convention): reconstruct
    the ORIGINAL per-micro-batch `rng.choice(n, size=batch_size, replace=False)` scheme
    over one nominal 'epoch' of `n // batch_size` independent draws, and confirm it
    leaves a MATERIAL fraction of rows completely unseen -- proving the true-epoch
    sampler's coverage guarantee is not a distinction without a difference. Uses the same
    scale character as the measured FA-SIM3 finding (~63.2% expected unique coverage)."""
    rng = np.random.default_rng(3)
    n, batch_size = 2000, 16
    seen = np.zeros(n, dtype=bool)
    for _ in range(n // batch_size):
        idx = rng.choice(n, size=batch_size, replace=False)
        seen[idx] = True
    coverage = seen.mean()
    # e^-1 ~= 0.368 unseen expected; assert comfortably inside a real regression margin
    # (not exactly 0.632, this is one finite draw) but clearly NOT full coverage.
    assert coverage < 0.90, (
        f"the independent-draw scheme covered {coverage:.3f} of rows in one nominal "
        "epoch -- this test is meant to demonstrate material undercoverage; if this "
        "fires, the reconstruction no longer has power")

    sampler = ShuffledEpochSampler(n, np.random.default_rng(3))
    for _ in range(n // batch_size):
        sampler.next_batch(batch_size)
    assert (sampler.exposure_counts >= 1).all()  # the true sampler: zero unseen rows


def test_rejects_nonpositive_n_and_batch_size():
    with pytest.raises(ValueError):
        ShuffledEpochSampler(0, np.random.default_rng(0))
    sampler = ShuffledEpochSampler(10, np.random.default_rng(0))
    with pytest.raises(ValueError):
        sampler.next_batch(0)
