"""Tests for src/atom/codec/probe_panel.py (P2-8: farthest-point sampling + coverage stat)."""

import numpy as np

from atom.codec.probe_panel import coverage_stat, farthest_point_sample, normalize_rows


def test_normalize_rows_unit_norm():
    Z = np.array([[3.0, 4.0], [1.0, 0.0], [0.0, 0.0]])
    Zn = normalize_rows(Z)
    assert np.allclose(np.linalg.norm(Zn[:2], axis=1), 1.0)
    assert np.allclose(Zn[2], 0.0)  # zero vector stays zero, no div-by-zero blowup


def test_farthest_point_sample_returns_n_unique_indices():
    rng = np.random.default_rng(0)
    Z = rng.normal(size=(100, 8))
    idx = farthest_point_sample(Z, n=10, seed=0)
    assert len(idx) == 10
    assert len(set(idx.tolist())) == 10  # no repeats
    assert idx.min() >= 0 and idx.max() < 100


def test_farthest_point_sample_clamps_n_to_available_points():
    Z = np.random.default_rng(1).normal(size=(5, 4))
    idx = farthest_point_sample(Z, n=200, seed=0)
    assert len(idx) == 5
    assert set(idx.tolist()) == {0, 1, 2, 3, 4}


def test_farthest_point_sample_deterministic_given_seed():
    Z = np.random.default_rng(2).normal(size=(50, 6))
    idx1 = farthest_point_sample(Z, n=10, seed=42)
    idx2 = farthest_point_sample(Z, n=10, seed=42)
    assert (idx1 == idx2).all()


def test_farthest_point_sample_spreads_out_on_clustered_data():
    # Three tight clusters far apart -- FPS should pick from more than one cluster in its
    # first few picks, not all from whichever cluster the random seed happened to land in.
    rng = np.random.default_rng(3)
    c1 = rng.normal(loc=[10, 0], scale=0.1, size=(20, 2))
    c2 = rng.normal(loc=[-10, 0], scale=0.1, size=(20, 2))
    c3 = rng.normal(loc=[0, 10], scale=0.1, size=(20, 2))
    Z = np.vstack([c1, c2, c3])
    idx = farthest_point_sample(Z, n=3, seed=0)
    clusters_hit = {i // 20 for i in idx}
    assert len(clusters_hit) == 3  # one anchor per cluster, not 3 from the same cluster


def test_coverage_stat_zero_when_every_point_is_an_anchor():
    Z = np.random.default_rng(4).normal(size=(30, 5))
    idx = np.arange(30)
    stats = coverage_stat(Z, idx)
    assert stats["mean_dist_to_nearest_anchor"] < 1e-9
    assert stats["max_dist_to_nearest_anchor"] < 1e-9


def test_coverage_stat_positive_and_sane_with_partial_panel():
    rng = np.random.default_rng(5)
    Z = rng.normal(size=(200, 8))
    idx = farthest_point_sample(Z, n=20, seed=0)
    stats = coverage_stat(Z, idx)
    assert stats["mean_dist_to_nearest_anchor"] > 0
    assert stats["mean_dist_to_nearest_anchor"] <= stats["max_dist_to_nearest_anchor"]
    assert stats["median_dist_to_nearest_anchor"] <= stats["max_dist_to_nearest_anchor"]
