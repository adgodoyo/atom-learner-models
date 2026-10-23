"""The probe panel (the "Fisher ruler"): ~200 anchor items chosen by
farthest-point sampling in the frozen item-geometry space Z (mean/L15/k5, the same space
the FA/CW/XES item codec vectors occupy, `artifacts/*_codec_*.npz`).

A learner's **panel profile** kappa(b) in [0,1]^|P| is the model's predicted-correctness
vector under belief b evaluated at every panel item -- distance between two learners is
disagreement between their profiles. Consumers of that profile live elsewhere; this module
only selects and freezes WHICH items make up the panel, and reports how evenly it covers Z.

Farthest-point sampling (FPS): greedy, deterministic given a seed. Start from one seeded
point; repeatedly add the point with the largest distance to its nearest already-selected
anchor, until n anchors are chosen. Distance is Euclidean on L2-normalised codec vectors
(monotonic with cosine distance, the codec's "deflated-cosine" convention), not raw
Euclidean.
"""

from __future__ import annotations

import numpy as np


def normalize_rows(Z: np.ndarray) -> np.ndarray:
    return Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12)


def farthest_point_sample(Z: np.ndarray, n: int, seed: int) -> np.ndarray:
    """Return the row-indices of `n` points in `Z` chosen by greedy farthest-point sampling
    (on L2-normalised `Z`, i.e. cosine geometry). Deterministic given `seed`, which only
    picks the FIRST anchor (every subsequent pick is the deterministic
    argmax-distance-to-nearest-selected-anchor). `n` is clamped to `len(Z)` if larger.
    """
    Zn = normalize_rows(Z)
    n = min(n, len(Zn))
    rng = np.random.default_rng(seed)
    selected = [int(rng.integers(len(Zn)))]
    min_dist = np.linalg.norm(Zn - Zn[selected[0]], axis=1)
    for _ in range(n - 1):
        nxt = int(np.argmax(min_dist))
        selected.append(nxt)
        d = np.linalg.norm(Zn - Zn[nxt], axis=1)
        min_dist = np.minimum(min_dist, d)
    return np.array(selected, dtype=np.int64)


def coverage_stat(Z: np.ndarray, anchor_idx: np.ndarray) -> dict[str, float]:
    """Mean/median/max distance from every point in `Z` to its NEAREST panel anchor
    (cosine geometry, i.e. Euclidean on L2-normalised vectors) -- the panel's gate metric.
    Lower mean = more even coverage of the space; the max is the single worst-served point.
    """
    Zn = normalize_rows(Z)
    anchors = Zn[anchor_idx]
    # Distances to all anchors, per point, then take the min per point -- done in chunks to
    # avoid an (N, n_anchors) blow-up on very large N (XES-scale corpora).
    n = Zn.shape[0]
    nearest = np.empty(n, dtype=np.float64)
    chunk = 2000
    for start in range(0, n, chunk):
        block = Zn[start:start + chunk]
        d = np.linalg.norm(block[:, None, :] - anchors[None, :, :], axis=2)
        nearest[start:start + chunk] = d.min(axis=1)
    return {
        "mean_dist_to_nearest_anchor": float(nearest.mean()),
        "median_dist_to_nearest_anchor": float(np.median(nearest)),
        "max_dist_to_nearest_anchor": float(nearest.max()),
    }
