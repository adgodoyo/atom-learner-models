"""Prefix-geometry stratification: NEAR / FAR / REACH-BACK, used by the filter-body
bake-off (Appendix A).

The definitions are those of `scripts/p3_06_far_from_prefix.py`, lifted out of its inline
`_learner_distances` so that `scripts/fa_02a_strata_v2.py` can re-derive the SAME strata on
the rebuilt FA manifests without copying the code or editing the original script, whose
measurement stays reproducible byte-for-byte.

For each scored cell (learner, t), in the frozen ambient Z (the production codec) --
audit-only, never a prediction input, never supervision:

    d_min(t) = distance from s_q(t) to the NEAREST item in that learner's own prefix 0..t-1
    d_rec(t) = distance from s_q(t) to the MEAN of the learner's last `recency_window`
               prefix items

Distance is Euclidean on L2-normalised codec vectors -- monotonic with cosine distance, the
established metric for this codec (`codec.probe_panel.normalize_rows`, the
"deflated-cosine" convention).

    NEAR       = d_rec bottom tercile
    FAR        = d_rec top tercile
    REACH-BACK = d_rec top tercile AND d_min bottom tercile

REACH-BACK is the discriminating cell: far from recent practice but near something the
learner did long ago. A flat recurrent state cannot serve it (the relevant events are
averaged into h_t alongside everything else); distance-biased attention can, which is exactly
what the causal event-transformer filter body adds. Note that the stratification is a
property of the DATA, not of any model -- so it can be, and is, frozen before the arm being
judged exists.
"""

from __future__ import annotations

import numpy as np

TERCILE_LO_PCT, TERCILE_HI_PCT = 100 / 3, 200 / 3


def normalise_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Row-wise L2 normalisation (zero rows stay zero) -- `codec.probe_panel.normalize_rows`'
    convention, restated here so this module has no dependency on the codec package."""
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), eps)


def learner_prefix_distances(item_codec: np.ndarray, length: int,
                             recency_window: int) -> tuple[np.ndarray, np.ndarray]:
    """(d_min, d_rec) over one learner's own valid prefix range [0, length).

    Position 0 has no prefix and returns NaN in both -- callers must assert no scored cell
    lands there (the eval half starts well after each learner's calib threshold, so none
    does).
    """
    zn = normalise_rows(item_codec[:length])
    d_full = np.linalg.norm(zn[:, None, :] - zn[None, :, :], axis=2)
    idx = np.arange(length)
    causal = idx[None, :] < idx[:, None]
    d_min = np.where(causal, d_full, np.inf).min(axis=1)
    d_min[0] = np.nan

    d_rec = np.full(length, np.nan, dtype=np.float64)
    for t in range(1, length):
        centroid = zn[max(0, t - recency_window):t].mean(axis=0)
        d_rec[t] = np.linalg.norm(zn[t] - centroid)
    return d_min, d_rec


def tercile_strata(d_min: np.ndarray, d_rec: np.ndarray) -> dict[str, np.ndarray | float]:
    """Tercile boundaries computed ONCE over the pooled scored population, plus the three
    boolean masks. Returned rather than saved so the caller owns the artifact's schema.
    """
    d_rec_lo, d_rec_hi = np.percentile(d_rec, [TERCILE_LO_PCT, TERCILE_HI_PCT])
    d_min_lo = np.percentile(d_min, TERCILE_LO_PCT)
    near = d_rec <= d_rec_lo
    far = d_rec >= d_rec_hi
    return {
        "near_mask": near, "far_mask": far, "reach_back_mask": far & (d_min <= d_min_lo),
        "d_rec_tercile_lo": float(d_rec_lo), "d_rec_tercile_hi": float(d_rec_hi),
        "d_min_tercile_lo": float(d_min_lo),
    }
