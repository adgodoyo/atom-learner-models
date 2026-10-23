"""Shared helpers for `cw_sim_smoke_local`.

Provides `_pick_smoke_batch`, which selects a small batch of training learners for the
generative simulator's local smoke test.
"""
from __future__ import annotations
import pathlib
import sys
REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))


def _pick_smoke_batch(manifest: dict, n_learners: int = 3, min_episodes: int = 3, max_episodes: int = 15):
    """>=`min_episodes` per learner (so the cross-episode `summary` chain has a live
    downstream reader within this tiny batch), sorted by total attempt count ASCENDING
    but taking learners past the very smallest few -- large enough that
    `--micro-batch-size` (default well below this batch's row count) actually produces
    more than one chunk, small enough to stay memory-safe on a 24 GB laptop."""
    import pandas as pd

    train_learners = sorted(manifest["train_learners"])
    ep_counts = manifest["episodes"][manifest["episodes"]["learner"].isin(train_learners)].groupby("learner").size()
    attempt_counts = manifest["attempts"][manifest["attempts"]["learner"].isin(train_learners)].groupby("learner").size()
    candidates = ep_counts[(ep_counts >= min_episodes) & (ep_counts <= max_episodes)].index
    if len(candidates) < n_learners:
        raise RuntimeError(f"only {len(candidates)} learners have {min_episodes}-{max_episodes} "
                           f"episodes -- widen the range or lower n_learners")
    candidates = sorted(candidates, key=lambda learner: attempt_counts.get(learner, 0))
    # skip the very smallest handful (want >1 micro-batch chunk), take the next few.
    skip = max(0, len(candidates) // 3)
    batch_learners = set(candidates[skip:skip + n_learners])
    attempts_batch = manifest["attempts"][manifest["attempts"]["learner"].isin(batch_learners)]
    episodes_batch = manifest["episodes"][manifest["episodes"]["learner"].isin(batch_learners)]
    return attempts_batch, episodes_batch, batch_learners
