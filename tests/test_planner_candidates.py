"""Unit tests for src/atom/planner/candidates.py, including the replay's `recency_bounded_pool`
-- synthetic tensors, no real checkpoint/data dependency (fast, CPU).
"""

import numpy as np

from atom.planner.candidates import fit_pid_clusters, recency_bounded_pool, stratified_candidates


def test_recency_bounded_pool_is_the_lookback_window_immediately_before_target():
    pool_idx = recency_bounded_pool(target_start=100, pool_lookback=20)
    assert list(pool_idx) == list(range(80, 100))


def test_recency_bounded_pool_clips_at_zero_when_lookback_exceeds_target_start():
    pool_idx = recency_bounded_pool(target_start=5, pool_lookback=20)
    assert list(pool_idx) == list(range(0, 5))


def test_recency_bounded_pool_does_not_exclude_give_ups():
    """Filtering the pool on a future channel (saw_answer) is itself a leak --
    the repaired pool construction must include EVERY event in the lookback window,
    give-ups included, regardless of any outcome channel. There is nothing in this
    function's signature to filter by outcome at all -- this test pins that down."""
    pool_idx = recency_bounded_pool(target_start=50, pool_lookback=10)
    assert len(pool_idx) == 10
    # No outcome array is even accepted -- the pool is purely positional.
    import inspect
    params = inspect.signature(recency_bounded_pool).parameters
    assert "saw_answer" not in params
    assert "correct" not in params


def test_stratified_candidates_still_works_on_a_recency_bounded_pool():
    pool_idx = recency_bounded_pool(target_start=30, pool_lookback=12)
    pool_clusters = np.array([i % 3 for i in range(len(pool_idx))])
    selected, n_clusters = stratified_candidates(pool_idx, pool_clusters, candidate_w=6)
    assert len(selected) == 6
    assert set(selected).issubset(set(pool_idx.tolist()))
    assert n_clusters == 3


def test_fit_pid_clusters_returns_one_cluster_id_per_problem():
    codec_matrix = np.random.RandomState(0).randn(9, 4)
    pid_to_idx = {100 + i: i for i in range(9)}
    clusters = fit_pid_clusters(codec_matrix, pid_to_idx, n_clusters=3, seed=0)
    assert set(clusters.keys()) == set(pid_to_idx.keys())
    assert set(clusters.values()).issubset({0, 1, 2})