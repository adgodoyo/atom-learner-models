"""Time, session and repeat-exposure features for the FoundationalASSIST filter, split across
the two places the architecture allows them. Screened as an ablation: thesis Appendix A.2
reports them worth -0.0029 nats, real and too small to ship.

`build_prev_feat_v2` carries five channels and nothing temporal. Two different kinds of
"time" feature are legal here, and they are legal in DIFFERENT places, for a reason
`event_transformer.py` already established for body (c)'s attention bias: at belief time
(computing mu_t, before item t is answered), the model does not yet know WHEN the learner
will next engage -- so any per-step time feature entering the RECURRENT input must describe
the gap ENDING at the PREVIOUS event, never the gap ending at t itself (which depends on
time_min[t], not yet known when b_t is formed). This module extends `build_prev_feat_v2`
with exactly that shifted convention, as a separate module (`episode_v2.py` untouched; `S1FilterV2`
already accepts `prev_feat_dim` as a constructor arg, so no filter-body change is needed
either).

Repeat-exposure ("has this learner seen THIS item before, and how many times") is different:
it is a property of the CURRENT item, exactly like item_codec itself, which the emission path
(`FiLMHeadsSupport`) already reads directly without violating `b_t = f(H_<t)` -- "current
item in the emission path only" is the architecture's rule (enforced by s1_heads.py's peek
canary), and repeat-exposure adds no new item-identity leak beyond what item_codec[t] already
carries. `build_current_repeat_features` is therefore an EMISSION-path input, consumed by
`heads/fa_time_heads.py::FiLMHeadsSupportTime`, never the recurrent filter.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from atom.filter.episode_v2 import PREV_FEAT_DIM_V2, build_prev_feat_v2

PREV_FEAT_DIM_V2_TIME = PREV_FEAT_DIM_V2 + 3  # + prev_gap_log1p_min, prev_new_session, prev_pos_in_session
N_REPEAT_FEAT = 2  # [is_repeat_exposure, log1p(exposure_index)] -- emission path only

# A screen-level choice, not swept: the candidate session-gap thresholds specified for
# CodeWorkout sessionisation are {30, 60, 120} min. 30 min is the shortest/most conservative of that set
# (more, shorter, sessions), picked once and disclosed rather than tuned, since this is an
# information-gathering screen, not a final session-boundary design.
DEFAULT_SESSION_GAP_MIN = 30.0


def scatter_field(ev: pd.DataFrame, learners: list[str], t_max: int, col: str,
                  fill: float = 0.0, dtype=np.float32) -> np.ndarray:
    """Same sort + scatter convention `build_padded_batch_v2` uses for every other channel,
    for a manifest column not already threaded through it (`batch.py` stays untouched). `ev` must carry `learner`/`t`/`col`; `learners`/`t_max` must be the SAME
    values the caller's padded batch was built with, or cells silently misalign."""
    learner_idx_of = {u: i for i, u in enumerate(learners)}
    ev_sorted = ev.sort_values(["learner", "t"], kind="mergesort")
    row_n = ev_sorted["learner"].map(learner_idx_of).to_numpy()
    row_t = ev_sorted["t"].to_numpy().astype(np.int64)
    out = np.full((len(learners), t_max), fill, dtype=dtype)
    out[row_n, row_t] = ev_sorted[col].to_numpy().astype(dtype)
    return out


def compute_session_and_exposure(ev: pd.DataFrame,
                                 session_gap_log1p_threshold: float) -> pd.DataFrame:
    """Adds three CURRENT-event columns to a copy of `ev` (`is_new_session`, `pos_in_session`,
    `exposure_index`), each a deterministic function of that event's own manifest fields and
    earlier history -- legitimate to know once the event happens, and legal as RECURRENT
    input only after the caller shifts them (see module docstring), legal as EMISSION input
    (`exposure_index`) immediately, since it describes the current item.

    `is_new_session`: the event starts a new session -- first event ever, or its own
    `gap_log1p_min` (already computed by `fa_manifest.py`, log1p of true minutes since the
    learner's previous event) exceeds the threshold.
    `pos_in_session`: 0-indexed position within that session (resets at every boundary).
    `exposure_index`: 0 at this learner's first exposure to this `problem_id`, 1 at the
    second, etc. -- the count of PRIOR exposures, so 0 <=> `first_exposure`.
    """
    ev = ev.sort_values(["learner", "t"], kind="mergesort").copy()
    ev["is_new_session"] = (
        ev["first_event"].astype(bool)
        | (ev["gap_log1p_min"].fillna(0.0) > session_gap_log1p_threshold)
    )
    session_id = ev.groupby("learner")["is_new_session"].cumsum()
    ev["pos_in_session"] = ev.groupby(["learner", session_id], sort=False).cumcount()
    ev["exposure_index"] = ev.groupby(["learner", "problem_id"], sort=False).cumcount()
    return ev


def _shift_right(x: torch.Tensor) -> torch.Tensor:
    """One-position-behind shift, zero at each learner's t=0 -- identical convention to
    `episode_v2.build_prev_item_codec`/`build_prev_feat_v2`."""
    out = torch.zeros_like(x)
    if x.shape[1] > 1:
        out[:, 1:] = x[:, :-1]
    return out


def build_prev_feat_v2_time(
    correct: torch.Tensor, hint_bin: torch.Tensor, saw_answer: torch.Tensor,
    score_missing: torch.Tensor, gap_log1p_min: torch.Tensor, is_new_session: torch.Tensor,
    pos_in_session: torch.Tensor,
) -> torch.Tensor:
    """Extends `build_prev_feat_v2`'s 5 channels with 3 more, ALL shifted by the same one
    position (prev_feat[t] describes event t-1, never t's own gap/session -- at belief time
    the learner's next engagement moment is not yet known, the same reason
    `event_transformer.py` shifts `node_time_min` before using it in body (c)'s bias). All
    inputs [N, T], unshifted (this function does the shifting). Returns [N, T, 8].
    """
    base = build_prev_feat_v2(correct, hint_bin, saw_answer, score_missing)
    prev_gap = _shift_right(torch.nan_to_num(gap_log1p_min, nan=0.0))
    prev_new_session = _shift_right(is_new_session.float())
    prev_pos = _shift_right(torch.log1p(pos_in_session.float()))
    return torch.cat(
        [base, prev_gap.unsqueeze(-1), prev_new_session.unsqueeze(-1), prev_pos.unsqueeze(-1)],
        dim=-1,
    )


def build_current_repeat_features(first_exposure: torch.Tensor,
                                  exposure_index: torch.Tensor) -> torch.Tensor:
    """CURRENT-item features for the EMISSION path only (see module docstring for why this
    is legal): [is_repeat_exposure, log1p(exposure_index)]. Both inputs [N, T], UNSHIFTED --
    these describe item t itself, exactly like item_codec[t]. Returns [N, T, 2]."""
    is_repeat = (~first_exposure.bool()).float()
    log_idx = torch.log1p(exposure_index.float())
    return torch.stack([is_repeat, log_idx], dim=-1)
