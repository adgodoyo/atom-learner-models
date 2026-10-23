"""Prefix-truncation invariance of the FA batch builders: is the belief built only from what
this learner answered BEFORE this question? Structurally yes (`test_s1_filter_v2.py`'s peek
canary reads max|Delta mu| =
0.00e+00, and the GRU body is unidirectional), but the peek canary tests the MODEL only, fed
hand-truncated tensors -- it cannot catch a leak that arrives through a FEATURE's normaliser
computed from a raw event table, because such a leak can leave the model's own forward pass
untouched while still corrupting what gets handed to it. The CodeWorkout analogue is an
`attempt_frac` feature with an episode-length denominator (`test_cw_filter_batch.py`'s
truncation test at the `build_attempt_batch` layer, not just `AttemptEncoder`): the canary
that catches it lives at the BATCH-BUILDER layer, one below the model.

This file is FA's analogue of that batch-builder-layer test, aimed at the two functions that
turn a raw per-learner event table into filter inputs: `build_padded_batch_v2` (every
production channel) and `compute_session_and_exposure` (the time features `is_new_session`/
`pos_in_session`/`exposure_index` -- the feature class most prone to such a leak).
Standard, matching the CodeWorkout test: truncate a
learner's event stream after position k, rebuild, and assert the retained prefix is
BIT-IDENTICAL to the full-history build -- `max|Delta| == 0.0`, not approximately.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from atom.filter.batch import build_padded_batch_v2
from atom.filter.fa_time_features import compute_session_and_exposure

CODEC_DIM = 4


def _codec_and_lookup(n_problems: int = 5):
    codec_matrix = np.arange(n_problems * CODEC_DIM, dtype=np.float32).reshape(n_problems, CODEC_DIM)
    pid_to_idx = {100 + i: i for i in range(n_problems)}
    return codec_matrix, pid_to_idx


def _two_learner_events():
    """Learner "a": 5 events (the one we truncate). Learner "b": 5 events, UNTOUCHED in every
    truncated variant -- keeps `t_max`/`learners` identical across full and truncated calls,
    so the two runs' output tensors are directly index-comparable without any reshaping."""
    base = pd.date_range("2026-01-01", periods=5, freq="10min")
    rows = []
    for learner, offset in (("a", 0), ("b", 100)):
        for t in range(5):
            rows.append({
                "learner": learner, "t": t, "problem_id": 100 + ((t + offset) % 5),
                "score": float((t + offset) % 2), "score_missing": False,
                "hint_count": t % 2, "saw_answer": 0, "timestamp": base[t],
            })
    return pd.DataFrame(rows)


def test_build_padded_batch_v2_truncation_invariance():
    """FA's own version of `test_cw_filter_batch.py`'s `build_attempt_batch` truncation
    test: for every truncation point k, every channel `build_padded_batch_v2` produces for
    learner "a"'s retained prefix 0..k-1 must be BIT-IDENTICAL between the full-history build
    and the build from an event table truncated after row k."""
    ev = _two_learner_events()
    codec_matrix, pid_to_idx = _codec_and_lookup()
    full = build_padded_batch_v2(ev, codec_matrix, pid_to_idx, device="cpu", include_time_min=True)
    a_idx = full["learners"].index("a")

    channels = ("item_codec", "correct", "score_missing", "hint_bin", "saw_answer",
               "valid_mask", "time_min")
    for k in range(1, 5):  # truncate learner "a" after event index k-1 (keep events 0..k-1)
        ev_trunc = ev[(ev["learner"] != "a") | (ev["t"] < k)]
        trunc = build_padded_batch_v2(ev_trunc, codec_matrix, pid_to_idx, device="cpu",
                                      include_time_min=True)
        assert trunc["learners"] == full["learners"], "truncating one learner must not change the roster"
        a_idx_t = trunc["learners"].index("a")
        for ch in channels:
            full_prefix = full[ch][a_idx, :k]
            trunc_prefix = trunc[ch][a_idx_t, :k]
            if full_prefix.dtype == torch.bool:
                assert torch.equal(full_prefix, trunc_prefix), f"{ch} mismatch at k={k}"
            else:
                max_abs_delta = (full_prefix.float() - trunc_prefix.float()).abs().max().item()
                assert max_abs_delta == 0.0, f"{ch} truncation at k={k}: expected 0.0, got {max_abs_delta:.2e}"


def test_compute_session_and_exposure_truncation_invariance():
    """The time-feature channels, at the layer that actually builds them from a raw event table
    (not the model, not even `build_prev_feat_v2_time` -- one layer below both, matching CW's
    finding that the leak lived in the batch builder, not the encoder). One learner, one
    session boundary mid-stream, one repeated problem_id -- so all three derived columns
    (`is_new_session`, `pos_in_session`, `exposure_index`) have real structure to preserve or
    break under truncation."""
    ev = pd.DataFrame({
        "learner": ["u"] * 6, "t": [0, 1, 2, 3, 4, 5],
        "problem_id": [10, 20, 10, 30, 30, 20],
        "first_event": [True, False, False, False, False, False],
        "gap_log1p_min": [float("nan"), 0.1, 0.1, 5.0, 0.1, 0.1],  # boundary between t=2,3
    })
    full = compute_session_and_exposure(ev, session_gap_log1p_threshold=1.0)

    for k in range(1, 6):  # truncate after event index k-1 (keep events 0..k-1)
        trunc = compute_session_and_exposure(ev.iloc[:k], session_gap_log1p_threshold=1.0)
        for col in ("is_new_session", "pos_in_session", "exposure_index"):
            full_prefix = full[col].iloc[:k].tolist()
            trunc_prefix = trunc[col].tolist()
            assert full_prefix == trunc_prefix, (
                f"{col} truncation at k={k}: full={full_prefix} trunc={trunc_prefix}"
            )


def _session_frac_leaky(ev: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """NOT production code -- a hand-built demonstration of the CW-class bug
    (`attempt_frac = position / (episode_length - 1)`), injected here only to prove the
    truncation-invariance test above has power against it. `session_length` requires knowing
    how many events the CURRENT session eventually contains -- exactly the aggregate-over-
    the-future quantity `compute_session_and_exposure`'s real `pos_in_session`/`exposure_index`
    (both plain `cumcount()`, forward-only) never compute."""
    out = compute_session_and_exposure(ev, session_gap_log1p_threshold=threshold)
    session_id = out.groupby("learner")["is_new_session"].cumsum()
    session_length = out.groupby(["learner", session_id])["is_new_session"].transform("count")
    out["session_frac"] = out["pos_in_session"] / (session_length - 1).clip(lower=1)
    return out


def test_would_have_caught_a_length_normalised_feature_leak():
    """Direct demonstration the canary fires: replays the CW `attempt_frac` bug shape
    (`session_frac = pos_in_session / (session_length - 1)`, `session_length` an aggregate
    over the CURRENT session's eventual total size) against FA's own event geometry, and
    confirms truncation invariance FAILS for that quantity -- proving `compute_session_and_
    exposure`'s real, shipped channels (`pos_in_session`, `exposure_index`, both cumcount)
    are safe not by accident of this test being too weak to notice, but because they are
    genuinely built differently from the leaky construction this test WOULD catch."""
    ev = pd.DataFrame({
        "learner": ["u"] * 4, "t": [0, 1, 2, 3], "problem_id": [10, 20, 30, 40],
        "first_event": [True, False, False, False],
        "gap_log1p_min": [float("nan"), 0.1, 0.1, 0.1],  # one long session, no boundary
    })
    full = _session_frac_leaky(ev, threshold=1.0)

    # Truncate after event index 1 (keep events 0..1) -- the FULL session has 4 events
    # (session_length=4), the TRUNCATED "session" has only 2 (session_length=2), so
    # session_frac at position 1 changes: 1/(4-1)=0.333... in the full build vs 1/(2-1)=1.0
    # in the truncated build, even though nothing about events 0..1 themselves changed.
    trunc = _session_frac_leaky(ev.iloc[:2], threshold=1.0)

    max_abs_delta = (full["session_frac"].iloc[:2].to_numpy()
                     - trunc["session_frac"].to_numpy()).__abs__().max()
    assert max_abs_delta > 0.0, (
        "the leaky, length-normalised feature should have broken truncation invariance -- "
        "if this assertion fails, the test above is too weak to catch this bug class"
    )
    # And the genuinely shipped columns, computed via the SAME calls, remain invariant --
    # isolating that the leak is specific to the injected aggregate-over-the-future
    # construction, not a property of compute_session_and_exposure itself.
    for col in ("is_new_session", "pos_in_session", "exposure_index"):
        assert full[col].iloc[:2].tolist() == trunc[col].tolist()
