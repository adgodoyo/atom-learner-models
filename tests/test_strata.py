"""Tests for `atom.eval.strata` -- the NEAR/FAR/REACH-BACK definitions FA-2 freezes before
body (c) trains. The definitions were extracted from `scripts/p3_06_far_from_prefix.py`'s
inline `_learner_distances`, so the first test pins that equivalence against an independent,
deliberately naive recomputation: if the extraction changed the definition, the v1 and v2
strata would stop being the same construct and the FA-2 screen would silently answer a
different question than P3-6 asked.
"""

from __future__ import annotations

import numpy as np

from atom.eval.strata import learner_prefix_distances, normalise_rows, tercile_strata


def _naive_distances(codec: np.ndarray, length: int, window: int):
    """Straightforward loop-based restatement of the P3-6 definition."""
    zn = np.array([v / max(np.linalg.norm(v), 1e-12) for v in codec[:length]])
    d_min = np.full(length, np.nan)
    d_rec = np.full(length, np.nan)
    for t in range(1, length):
        d_min[t] = min(np.linalg.norm(zn[t] - zn[j]) for j in range(t))
        d_rec[t] = np.linalg.norm(zn[t] - zn[max(0, t - window):t].mean(axis=0))
    return d_min, d_rec


def test_matches_an_independent_naive_recomputation():
    rng = np.random.default_rng(0)
    codec = rng.normal(size=(30, 8))
    d_min, d_rec = learner_prefix_distances(codec, 30, recency_window=5)
    n_min, n_rec = _naive_distances(codec, 30, 5)
    assert np.allclose(d_min[1:], n_min[1:])
    assert np.allclose(d_rec[1:], n_rec[1:])
    assert np.isnan(d_min[0]) and np.isnan(d_rec[0])


def test_distances_are_causal_only():
    """A future item must not be able to change an earlier cell's distances -- the strata are
    computed over each learner's own PREFIX."""
    rng = np.random.default_rng(1)
    codec = rng.normal(size=(20, 8))
    d_min, d_rec = learner_prefix_distances(codec, 20, recency_window=5)
    perturbed = codec.copy()
    perturbed[12:] = rng.normal(size=(8, 8)) * 50
    p_min, p_rec = learner_prefix_distances(perturbed, 20, recency_window=5)
    assert np.allclose(d_min[1:12], p_min[1:12])
    assert np.allclose(d_rec[1:12], p_rec[1:12])


def test_recency_window_is_respected():
    """d_rec at t must depend only on the last `window` prefix items."""
    rng = np.random.default_rng(2)
    codec = rng.normal(size=(20, 8))
    _, d_rec = learner_prefix_distances(codec, 20, recency_window=3)
    far_past = codec.copy()
    far_past[:10] = rng.normal(size=(10, 8)) * 10
    _, p_rec = learner_prefix_distances(far_past, 20, recency_window=3)
    assert np.allclose(d_rec[14:], p_rec[14:])


def test_strata_are_terciles_and_reach_back_is_their_intersection():
    rng = np.random.default_rng(3)
    d_rec, d_min = rng.uniform(size=3000), rng.uniform(size=3000)
    s = tercile_strata(d_min, d_rec)
    for name in ("near_mask", "far_mask"):
        assert 0.31 < s[name].mean() < 0.36, name
    assert np.array_equal(s["reach_back_mask"],
                          s["far_mask"] & (d_min <= s["d_min_tercile_lo"]))
    assert not (s["near_mask"] & s["far_mask"]).any()


def test_normalise_rows_leaves_zero_rows_at_zero():
    x = np.array([[0.0, 0.0], [3.0, 4.0]])
    out = normalise_rows(x)
    assert np.array_equal(out[0], np.zeros(2))
    assert np.allclose(np.linalg.norm(out[1]), 1.0)
