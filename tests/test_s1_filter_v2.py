"""Query-independent S1v2 filter -- synthetic unit tests, fast, no data dependency.

The load-bearing test is `test_query_independence_peek_canary`: the mandatory peek
canary, "the same history paired with two questions produces the same b_t but
different question-conditioned head outputs" -- checked for `query_independent=True`, and
then checked to FAIL (the leaky variant must visibly leak) for `query_independent=False`,
proving the query-independent design has a real, non-vanishing structural effect.
"""

from __future__ import annotations

import torch

from atom.filter.episode_v2 import (
    PREV_FEAT_DIM_V2,
    EpisodeEncoderV2,
    build_prev_feat_v2,
    build_prev_item_codec,
)
from atom.filter.s1v2 import S1FilterV2
from atom.heads.s1_heads import FiLMHeads


def test_episode_encoder_v2_shape():
    enc = EpisodeEncoderV2(codec_dim=8, hidden=16)
    prev_item = torch.randn(3, 5, 8)
    prev_feat = torch.randn(3, 5, PREV_FEAT_DIM_V2)
    out = enc(prev_item, prev_feat)
    assert out.shape == (3, 5, 16)


def test_build_prev_item_codec_shifts_by_one_zero_at_start():
    item = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]])  # [1, 3, 2]
    prev = build_prev_item_codec(item)
    assert torch.equal(prev[0, 0], torch.zeros(2))       # t=0: no history
    assert torch.equal(prev[0, 1], torch.tensor([1.0, 2.0]))  # t=1: carries t=0's item
    assert torch.equal(prev[0, 2], torch.tensor([3.0, 4.0]))  # t=2: carries t=1's item


def test_build_prev_feat_v2_shifts_and_flags_missing_score():
    correct = torch.tensor([[1.0, 0.0, 1.0]])
    hint = torch.tensor([[0.0, 1.0, 0.0]])
    saw = torch.tensor([[0.0, 0.0, 1.0]])
    score_missing = torch.tensor([[0.0, 1.0, 0.0]])  # t=1's score is missing
    prev = build_prev_feat_v2(correct, hint, saw, score_missing)
    assert prev.shape == (1, 3, 5)
    assert torch.equal(prev[0, 0], torch.zeros(5))  # t=0: no history
    # t=1: carries t=0's (correct=1, hint=0, saw=0), valid=1, prev_score_missing=0.
    assert torch.equal(prev[0, 1], torch.tensor([1.0, 0.0, 0.0, 1.0, 0.0]))
    # t=2: carries t=1's outcome, but t=1's score WAS missing -> prev_correct zeroed,
    # prev_score_missing=1 (not silently coerced to "wrong").
    assert torch.equal(prev[0, 2], torch.tensor([0.0, 1.0, 0.0, 1.0, 1.0]))


def test_s1_filter_v2_end_to_end_shape():
    model = S1FilterV2(codec_dim=10, episode_hidden=12, filter_hidden=14, latent_dim=6)
    item = torch.randn(3, 9, 10)
    prev = torch.randn(3, 9, PREV_FEAT_DIM_V2)
    mu, log_sigma = model(item, prev)
    assert mu.shape == (3, 9, 6)
    assert log_sigma.shape == (3, 9, 6)


# -- the mandatory peek canary --------------------------------------------------------

def _fixed_history_two_questions(seed: int):
    torch.manual_seed(seed)
    n, tlen, codec_dim = 2, 6, 8
    history = torch.randn(n, tlen, codec_dim)
    correct = torch.randint(0, 2, (n, tlen)).float()
    hint = torch.zeros(n, tlen)
    saw = torch.zeros(n, tlen)
    score_missing = torch.zeros(n, tlen)
    prev_feat = build_prev_feat_v2(correct, hint, saw, score_missing)
    # Two candidate CURRENT questions at the final position t=tlen-1, identical history
    # everywhere strictly before it.
    item_q1 = history.clone()
    item_q2 = history.clone()
    item_q2[:, -1] = torch.randn(n, codec_dim)  # only the LAST (current) position differs
    return item_q1, item_q2, prev_feat


def test_query_independence_peek_canary_fixed_variant_ignores_current_question():
    """query_independent=True: b_t at the final position must be BIT-IDENTICAL across two
    different current questions, since the recurrent input never sees position t's item."""
    torch.manual_seed(0)
    model = S1FilterV2(codec_dim=8, episode_hidden=10, filter_hidden=12, latent_dim=4,
                       query_independent=True)
    model.eval()
    item_q1, item_q2, prev_feat = _fixed_history_two_questions(seed=1)

    with torch.no_grad():
        mu1, ls1 = model(item_q1, prev_feat)
        mu2, ls2 = model(item_q2, prev_feat)

    # b_t at every position, including the final (current-question) one, is unaffected --
    # only strictly-earlier positions ever entered the recurrent input.
    assert torch.equal(mu1, mu2)
    assert torch.equal(ls1, ls2)

    # But the emission/head layer, which legitimately takes the current question directly,
    # MUST differ -- "current item in the emission path only".
    heads = FiLMHeads(codec_dim=8, latent_dim=4, hidden=12)
    heads.eval()
    with torch.no_grad():
        u = torch.randn(1, *mu1.shape)
        logits1, _ = heads(item_q1, u)
        logits2, _ = heads(item_q2, u)
    assert not torch.equal(logits1[..., -1], logits2[..., -1])


def test_query_independence_peek_canary_leaky_variant_fails_it():
    """query_independent=False (the leaky v1-equivalent architecture): the SAME canary must
    now FAIL -- b_t at the final position visibly changes with the current question, proving
    this test catches the defect and that the query-independent design is not a no-op.
    Gate: 'fails if the canary gap vanishes' -- i.e. this assertion itself must hold."""
    torch.manual_seed(0)
    model = S1FilterV2(codec_dim=8, episode_hidden=10, filter_hidden=12, latent_dim=4,
                       query_independent=False)
    model.eval()
    item_q1, item_q2, prev_feat = _fixed_history_two_questions(seed=1)

    with torch.no_grad():
        mu1, _ = model(item_q1, prev_feat)
        mu2, _ = model(item_q2, prev_feat)

    # The final position's belief DOES change with the current question -- the leak.
    assert not torch.allclose(mu1[:, -1], mu2[:, -1])
    # Earlier positions are unaffected either way (neither variant peeks at the future).
    assert torch.equal(mu1[:, :-1], mu2[:, :-1])


# -- causality (no-peek), mirrored from test_s1_filter.py for the v2 architecture ---------

def test_belief_at_t_is_unaffected_by_outcome_at_t_v2():
    torch.manual_seed(0)
    model = S1FilterV2(codec_dim=6, episode_hidden=8, filter_hidden=10, latent_dim=4)
    model.eval()

    item = torch.randn(1, 5, 6)
    correct_a = torch.tensor([[1.0, 0.0, 1.0, 0.0, 1.0]])
    correct_b = correct_a.clone()
    correct_b[0, 2] = 1.0 - correct_b[0, 2]
    hint = torch.zeros(1, 5)
    saw = torch.zeros(1, 5)
    score_missing = torch.zeros(1, 5)

    with torch.no_grad():
        prev_a = build_prev_feat_v2(correct_a, hint, saw, score_missing)
        prev_b = build_prev_feat_v2(correct_b, hint, saw, score_missing)
        mu_a, ls_a = model(item, prev_a)
        mu_b, ls_b = model(item, prev_b)

    assert torch.allclose(mu_a[0, :3], mu_b[0, :3])
    assert torch.allclose(ls_a[0, :3], ls_b[0, :3])
    assert not torch.allclose(mu_a[0, 3:], mu_b[0, 3:])


def test_belief_at_t_is_unaffected_by_future_items_v2():
    """Held to the exact-zero bar the pre-registered
    prefix-truncation-invariance gate requires (`max|Δμ| == 0.0`, not `torch.allclose`) --
    this is the MODEL-forward layer of that instrument; `tests/test_fa_10c_prefix_truncation_
    invariance.py` covers the BATCH-BUILDER layer beneath it, which this test cannot reach
    (it is fed hand-truncated tensors, not a raw event table)."""
    torch.manual_seed(1)
    model = S1FilterV2(codec_dim=6, episode_hidden=8, filter_hidden=10, latent_dim=4)
    model.eval()

    item_full = torch.randn(1, 6, 6)
    correct = torch.randint(0, 2, (1, 6)).float()
    hint = torch.zeros(1, 6)
    saw = torch.zeros(1, 6)
    score_missing = torch.zeros(1, 6)

    with torch.no_grad():
        prev_full = build_prev_feat_v2(correct, hint, saw, score_missing)
        mu_full, _ = model(item_full, prev_full)

        item_trunc = item_full[:, :4]
        prev_trunc = build_prev_feat_v2(correct[:, :4], hint[:, :4], saw[:, :4], score_missing[:, :4])
        mu_trunc, _ = model(item_trunc, prev_trunc)

    max_abs_delta = (mu_full[:, :4] - mu_trunc).abs().max().item()
    assert max_abs_delta == 0.0, f"expected exact 0.0, got {max_abs_delta:.2e}"
