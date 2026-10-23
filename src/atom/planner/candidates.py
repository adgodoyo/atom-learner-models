"""Candidate-pool construction for the planner replay harness (thesis Section 5.7).

Provides a recency-bounded candidate pool and a codec-cluster-stratified selection of
candidates from it. Every replay script that builds a candidate menu imports from here, so
all of them use one construction.

Audit-only use of the semantic space: clustering the frozen problem codec never touches a
prediction head or supervises anything; it only decides which real, already-logged events
become candidates in a replay.

**Why the pool is never filtered on outcomes.** Excluding give-ups (`saw_answer==1`) from
the pool would itself be a leak: `saw_answer` is unknowable at selection time (it is the
outcome of asking, not information available before asking), so letting it decide which
candidates are offered makes the pool's composition depend on the future.
`recency_bounded_pool` therefore bounds the pool by recency (so the menu sits close to the
targets rather than early in the history) and consults no channel of a candidate's real
outcome. Thesis Appendix A.4 describes the earlier pool constructions this replaces.
"""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans


def recency_bounded_pool(target_start: int, pool_lookback: int) -> np.ndarray:
    """The `pool_lookback` events immediately preceding `target_start` (the target window's
    first index) -- recency-bounded so the menu sits next to the targets, but with NO filtering on any
    channel of what actually happened at those events (see module docstring: filtering the
    pool on `saw_answer` was itself a leak). Every event in this range is an eligible
    candidate, give-ups included; `_materialize` (evsi.py) neutralises what revealing one
    conveys, so the pool no longer needs to pre-empt that by excluding them.
    """
    pool_start = max(0, target_start - pool_lookback)
    return np.arange(pool_start, target_start)


def fit_pid_clusters(codec_matrix: np.ndarray, pid_to_idx: dict[int, int],
                     n_clusters: int, seed: int) -> dict[int, int]:
    """KMeans on the frozen problem codec matrix. Returns {problem_id: cluster_id}."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    idx_to_pid = {i: p for p, i in pid_to_idx.items()}
    cluster_labels = kmeans.fit_predict(codec_matrix)
    return {idx_to_pid[i]: int(cluster_labels[i]) for i in range(len(cluster_labels))}


def stratified_candidates(pool_idx: np.ndarray, pool_clusters: np.ndarray,
                          candidate_w: int) -> tuple[np.ndarray, int]:
    """Round-robin across codec clusters (sorted cluster id, chronological order WITHIN
    each cluster -- boolean masking preserves the ascending order pool_idx is built in) so
    the selected set spans as many distinct clusters as possible rather than being
    dominated by whichever cluster happens to be adjacent in time. Returns (selected_idx
    sorted ascending, n_distinct_clusters_selected)."""
    clusters = sorted(set(pool_clusters.tolist()))
    queues = {c: list(pool_idx[pool_clusters == c]) for c in clusters}
    selected: list[int] = []
    contributed: set[int] = set()
    while len(selected) < candidate_w and any(queues.values()):
        for c in clusters:
            if queues[c]:
                selected.append(queues[c].pop(0))
                contributed.add(c)
                if len(selected) == candidate_w:
                    break
    return np.array(sorted(selected)), len(contributed)
