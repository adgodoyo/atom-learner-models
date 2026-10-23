"""True shuffled-epoch batch sampling, used by the simulator trainers.

Corpus-agnostic: it lives in `atom.utils`, not `atom.simulator`, so the CodeWorkout
trainer can reuse it without importing the FA-specific module.

Why it exists: drawing each micro-batch independently via
`rng.choice(n, size=batch_size, replace=False)` gives each of the `n` rows only a
`1 - (1 - batch_size/n)^(n/batch_size) ≈ 1 - e^-1 ≈ 63.2%` chance of appearing at all
over one nominal "epoch" of `n/batch_size` draws (the birthday-problem rate); on the
FA-SIM3 training rows that independent-draw scheme left ~87,988 of 239,181 rows unseen in
one nominal epoch.
A true epoch guarantees every row is drawn exactly once per full pass.
"""

from __future__ import annotations

import numpy as np


class ShuffledEpochSampler:
    """Draws batches without replacement across successive full shuffles of `range(n)`,
    tracking realised per-item exposure counts. A batch may span an epoch boundary (the
    tail of one shuffle followed by the head of the next) rather than dropping a partial
    batch -- every returned batch is exactly `batch_size`, and no row is ever skipped."""

    def __init__(self, n: int, rng: np.random.Generator):
        if n <= 0:
            raise ValueError(f"n must be positive, got {n}")
        self.n = n
        self.rng = rng
        self.exposure_counts = np.zeros(n, dtype=np.int64)
        self._buffer = np.array([], dtype=np.int64)

    def _refill(self) -> None:
        self._buffer = np.concatenate([self._buffer, self.rng.permutation(self.n)])

    def next_batch(self, batch_size: int) -> np.ndarray:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        while len(self._buffer) < batch_size:
            self._refill()
        batch, self._buffer = self._buffer[:batch_size], self._buffer[batch_size:]
        # np.add.at, not `self.exposure_counts[batch] += 1`: a batch straddling a
        # refill boundary can rarely contain the SAME row twice (two independent
        # permutations are not guaranteed disjoint at the seam), and plain fancy-index
        # `+=` silently drops repeated updates to the same location instead of
        # accumulating them (numpy's well-known duplicate-index gotcha) -- caught by
        # this module's own test asserting `exposure_counts.sum() == total draws`.
        np.add.at(self.exposure_counts, batch, 1)
        return batch
