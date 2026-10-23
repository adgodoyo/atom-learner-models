"""Time/session/repeat-exposure features -- synthetic unit tests, fast, no data dependency.
The load-bearing test is `test_peek_canary_time_arm_matches_baseline`: the SAME mandatory
canary `test_s1_filter_v2.py` runs for the baseline architecture, re-run against the
8-channel time-augmented recurrent input (the peek canary must still read
max|Δμ| = 0.00e+00 after the change).
"""

from __future__ import annotations

import pandas as pd
import torch

from atom.filter.episode_v2 import build_prev_feat_v2
from atom.filter.fa_time_features import (
    PREV_FEAT_DIM_V2_TIME,
    build_current_repeat_features,
    build_prev_feat_v2_time,
    compute_session_and_exposure,
    scatter_field,
)
from atom.filter.s1v2 import S1FilterV2
from atom.heads.fa_time_heads import FiLMHeadsSupportTime


def test_scatter_field_aligns_by_learner_and_t():
    ev = pd.DataFrame({"learner": ["b", "a", "a"], "t": [0, 1, 0], "x": [9.0, 7.0, 5.0]})
    out = scatter_field(ev, learners=["a", "b"], t_max=2, col="x")
    assert out[0, 0] == 5.0 and out[0, 1] == 7.0   # learner "a": t=0 -> 5, t=1 -> 7
    assert out[1, 0] == 9.0 and out[1, 1] == 0.0   # learner "b": t=0 -> 9, t=1 -> fill (0)


def test_compute_session_and_exposure_boundaries_and_counts():
    # One learner, 4 events: gap after event 0 is small (same session), after event 1 is
    # huge (new session), problem_id repeats at events 0 and 3 (second exposure).
    ev = pd.DataFrame({
        "learner": ["u"] * 4, "t": [0, 1, 2, 3], "problem_id": [10, 20, 30, 10],
        "first_event": [True, False, False, False],
        "gap_log1p_min": [float("nan"), 0.1, 5.0, 0.1],  # threshold = 1.0 below
    })
    out = compute_session_and_exposure(ev, session_gap_log1p_threshold=1.0)
    assert out["is_new_session"].tolist() == [True, False, True, False]
    assert out["pos_in_session"].tolist() == [0, 1, 0, 1]
    assert out["exposure_index"].tolist() == [0, 0, 0, 1]  # problem 10 seen again at t=3


def test_build_prev_feat_v2_time_shifts_new_channels_and_extends_base():
    correct = torch.tensor([[1.0, 0.0, 1.0]])
    hint = torch.zeros(1, 3)
    saw = torch.zeros(1, 3)
    score_missing = torch.zeros(1, 3)
    gap = torch.tensor([[0.0, 2.0, 5.0]])
    new_session = torch.tensor([[1.0, 0.0, 1.0]])
    pos_in_session = torch.tensor([[0.0, 1.0, 0.0]])

    out = build_prev_feat_v2_time(correct, hint, saw, score_missing, gap, new_session,
                                  pos_in_session)
    assert out.shape == (1, 3, PREV_FEAT_DIM_V2_TIME)

    base = build_prev_feat_v2(correct, hint, saw, score_missing)
    assert torch.equal(out[..., :5], base)   # first 5 channels byte-identical to the baseline

    # t=0: no history, all three new channels zero.
    assert torch.equal(out[0, 0, 5:], torch.zeros(3))
    # t=1: carries t=0's own gap (0.0), new_session (1.0), log1p(pos_in_session=0)=0.0.
    assert torch.allclose(out[0, 1, 5:], torch.tensor([0.0, 1.0, 0.0]))
    # t=2: carries t=1's own gap (2.0), new_session (0.0), log1p(pos_in_session=1).
    assert torch.allclose(out[0, 2, 5:], torch.tensor([2.0, 0.0, torch.log1p(torch.tensor(1.0))]))


def test_build_current_repeat_features_first_exposure_and_index():
    first_exposure = torch.tensor([[True, False, True]])
    exposure_index = torch.tensor([[0.0, 1.0, 0.0]])
    out = build_current_repeat_features(first_exposure, exposure_index)
    assert out.shape == (1, 3, 2)
    assert torch.equal(out[..., 0], torch.tensor([[0.0, 1.0, 0.0]]))  # is_repeat
    assert torch.allclose(out[0, :, 1], torch.log1p(exposure_index[0]))


def test_film_heads_support_time_shape_and_repeat_feature_moves_output():
    heads = FiLMHeadsSupportTime(codec_dim=6, latent_dim=4, hidden=8)
    heads.eval()
    item_codec = torch.randn(2, 5, 6)
    u = torch.randn(3, 2, 5, 4)
    repeat_a = torch.zeros(2, 5, 2)
    repeat_b = torch.ones(2, 5, 2)
    with torch.no_grad():
        c_a, s_a = heads(item_codec, u, repeat_a)
        c_b, s_b = heads(item_codec, u, repeat_b)
    assert c_a.shape == (3, 2, 5)
    assert s_a.shape == (3, 2, 5, 4)
    assert not torch.equal(c_a, c_b)  # the repeat feature actually reaches the trunk


# -- the mandatory peek canary, re-run for the time-augmented recurrent input ----------------

def test_peek_canary_time_arm_matches_baseline():
    """Same construction as test_s1_filter_v2.py's canary: identical history, two different
    CURRENT questions. The recurrent input is now 8-dim (time-augmented) instead of 5, but
    every added channel is shifted by the SAME one position as the original 5 -- b_t at every
    position, including the final (current-question) one, must be BIT-IDENTICAL."""
    torch.manual_seed(1)
    n, tlen, codec_dim = 2, 6, 8
    history = torch.randn(n, tlen, codec_dim)
    correct = torch.randint(0, 2, (n, tlen)).float()
    hint = torch.zeros(n, tlen)
    saw = torch.zeros(n, tlen)
    score_missing = torch.zeros(n, tlen)
    gap = torch.rand(n, tlen) * 3.0
    new_session = torch.randint(0, 2, (n, tlen)).float()
    pos_in_session = torch.arange(tlen).float().unsqueeze(0).expand(n, tlen).clone()
    prev_feat = build_prev_feat_v2_time(correct, hint, saw, score_missing, gap, new_session,
                                        pos_in_session)
    assert prev_feat.shape[-1] == PREV_FEAT_DIM_V2_TIME

    item_q1 = history.clone()
    item_q2 = history.clone()
    item_q2[:, -1] = torch.randn(n, codec_dim)  # only the CURRENT (last) position differs

    torch.manual_seed(0)
    model = S1FilterV2(codec_dim=codec_dim, episode_hidden=10, filter_hidden=12, latent_dim=4,
                       query_independent=True, prev_feat_dim=PREV_FEAT_DIM_V2_TIME)
    model.eval()
    with torch.no_grad():
        mu1, ls1 = model(item_q1, prev_feat)
        mu2, ls2 = model(item_q2, prev_feat)

    max_abs_diff = (mu1 - mu2).abs().max().item()
    assert max_abs_diff == 0.0, f"peek canary FAILED: max|Δμ|={max_abs_diff:.2e}, expected 0.00e+00"
    assert torch.equal(mu1, mu2)
    assert torch.equal(ls1, ls2)
