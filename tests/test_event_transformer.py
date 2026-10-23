"""Unit tests for the causal event-transformer filter body (c): the four contract clauses
of the filter bake-off, the causal mask, the peek canary, the upstream gradient canary, and
the three attention-bias ablations.

Every clause that can be checked without training is checked here, on random weights and
synthetic tensors -- so that a violation shows up in `make test` in seconds rather than in a
bake-off number nobody can explain. Contract clause 3 ("add, don't modify: existing tests
pass unedited") is why this body has its own test file.
"""

from __future__ import annotations

import pytest
import torch

from atom.filter.batch import build_padded_batch_v2
from atom.filter.episode_v2 import EpisodeEncoderV2, build_prev_feat_v2
from atom.filter.event_transformer import (
    BIAS_MODES,
    CausalEventTransformerFilter,
    DistanceTimeAttentionBias,
    S1FilterV2C,
    count_body_params,
)
from atom.filter.gru import KnowledgeFilter
from atom.heads.fa2_heads import N_SUPPORT, FiLMHeadsSupport

CODEC_DIM, EPISODE_HIDDEN, FILTER_HIDDEN, LATENT_DIM = 45, 64, 64, 32
N, T = 6, 12


def _inputs(n: int = N, t: int = T, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    item_codec = torch.randn(n, t, CODEC_DIM, generator=g)
    correct = (torch.rand(n, t, generator=g) > 0.4).float()
    hint = (torch.rand(n, t, generator=g) > 0.9).float()
    saw = (torch.rand(n, t, generator=g) > 0.8).float()
    missing = torch.zeros(n, t, dtype=torch.bool)
    prev_feat = build_prev_feat_v2(correct, hint, saw, missing)
    # Monotone per-learner clock in minutes, with a wide spread of gaps.
    time_min = torch.cumsum(torch.rand(n, t, generator=g) * 500.0, dim=1)
    time_min[:, 0] = 0.0
    valid_mask = torch.ones(n, t, dtype=torch.bool)
    return item_codec, prev_feat, time_min, valid_mask


def _model(bias_mode: str = "distance_time", query_independent: bool = True, seed: int = 0):
    torch.manual_seed(seed)
    return S1FilterV2C(CODEC_DIM, EPISODE_HIDDEN, FILTER_HIDDEN, LATENT_DIM,
                       bias_mode=bias_mode, query_independent=query_independent)


# -- contract clause 1: the episode encoder is SHARED, not re-implemented ------------------

def test_uses_the_shared_episode_encoder_class():
    assert isinstance(_model().episode_encoder, EpisodeEncoderV2)


def test_latent_dim_is_carried_through_untouched():
    """Contract clause 2: latent_dim stays 32."""
    mu, log_sigma = _model()(*_inputs())
    assert mu.shape == (N, T, LATENT_DIM)
    assert log_sigma.shape == (N, T, LATENT_DIM)


# -- contract clause 4a: the causal mask ---------------------------------------------------

def test_future_events_cannot_change_an_earlier_belief():
    """The defining property of the causal mask: perturb everything from position s onward
    and every belief strictly before s must be bit-identical.
    """
    model = _model().eval()
    item_codec, prev_feat, time_min, valid_mask = _inputs()
    with torch.no_grad():
        mu, _ = model(item_codec, prev_feat, time_min, valid_mask)

    s = 7
    perturbed = item_codec.clone()
    perturbed[:, s:] = torch.randn_like(perturbed[:, s:])
    prev_feat_p = prev_feat.clone()
    prev_feat_p[:, s:] = torch.rand_like(prev_feat_p[:, s:])
    time_p = time_min.clone()
    time_p[:, s:] = time_p[:, s:] + 10_000.0
    with torch.no_grad():
        mu_p, _ = model(perturbed, prev_feat_p, time_p, valid_mask)

    assert torch.equal(mu[:, :s], mu_p[:, :s])
    assert not torch.allclose(mu[:, s:], mu_p[:, s:])   # the change must be visible somewhere


def test_padding_is_never_attended():
    """Positions beyond a learner's own length must not influence any real position."""
    model = _model().eval()
    item_codec, prev_feat, time_min, valid_mask = _inputs()
    length = 8
    valid_mask[0, length:] = False
    with torch.no_grad():
        mu, _ = model(item_codec, prev_feat, time_min, valid_mask)
        junk = item_codec.clone()
        junk[0, length:] = 1e3 * torch.randn_like(junk[0, length:])
        mu_junk, _ = model(junk, prev_feat, time_min, valid_mask)
    assert torch.equal(mu[0, :length], mu_junk[0, :length])


def test_first_position_is_finite_despite_masking():
    """Position 0 must stay attendable, or its softmax row is all -inf -> NaN."""
    mu, log_sigma = _model()(*_inputs())
    assert torch.isfinite(mu).all() and torch.isfinite(log_sigma).all()


# -- contract clause 4b: the peek canary, both arms ----------------------------------------

@pytest.mark.parametrize("bias_mode", BIAS_MODES)
def test_peek_canary_swapping_the_current_question_never_moves_the_belief(bias_mode):
    """b_t = f(H_<t): swap the CURRENT item at every position and the belief must be
    bit-identical. Run for all four bias modes, because the attention bias is the one path
    through which body (c) could reintroduce a query-dependent belief.
    """
    model = _model(bias_mode=bias_mode).eval()
    item_codec, prev_feat, time_min, valid_mask = _inputs()
    with torch.no_grad():
        mu, log_sigma = model(item_codec, prev_feat, time_min, valid_mask)
        swapped = item_codec[torch.randperm(N)]
        mu_s, log_sigma_s = model(swapped, prev_feat, time_min, valid_mask)
    # Position t's belief may not depend on item t; it does legitimately depend on items <t,
    # which the permutation also changed -- so compare only the FIRST position, where the
    # whole history is empty and only the (swapped) current item differs.
    assert torch.equal(mu[:, 0], mu_s[:, 0])
    assert torch.equal(log_sigma[:, 0], log_sigma_s[:, 0])


@pytest.mark.parametrize("bias_mode", BIAS_MODES)
def test_peek_canary_fa1_form_swap_the_final_question_holding_history_fixed(bias_mode):
    """The exact canary `scripts/fa_01_s1v2.py` runs on the GRU, now on body (c): swap ONLY
    the final position's item codec across learners, leaving every earlier event untouched,
    and that position's belief must be bit-identical. Stronger than permuting the whole
    tensor, because here the history genuinely is fixed and the ONLY difference is the
    current question.
    """
    model = _model(bias_mode=bias_mode).eval()
    item_codec, prev_feat, time_min, valid_mask = _inputs()
    swapped = item_codec.clone()
    swapped[:, -1] = item_codec[torch.randperm(N), -1]
    with torch.no_grad():
        mu, log_sigma = model(item_codec, prev_feat, time_min, valid_mask)
        mu_s, log_sigma_s = model(swapped, prev_feat, time_min, valid_mask)
    assert torch.equal(mu[:, -1], mu_s[:, -1])
    assert torch.equal(log_sigma[:, -1], log_sigma_s[:, -1])


def test_peek_canary_is_two_sided_the_leaky_arm_is_caught():
    """A canary that can only pass is a receipt, not a canary (Law 2a). With
    query_independent=False the same architecture must visibly leak.
    """
    leaky = _model(query_independent=False).eval()
    item_codec, prev_feat, time_min, valid_mask = _inputs()
    with torch.no_grad():
        mu, _ = leaky(item_codec, prev_feat, time_min, valid_mask)
        mu_s, _ = leaky(item_codec[torch.randperm(N)], prev_feat, time_min, valid_mask)
    assert (mu[:, 0] - mu_s[:, 0]).abs().max() > 1e-6


def test_only_shifted_geometry_reaches_the_bias():
    """The final position's own timestamp must not affect its own belief -- the time-channel
    counterpart of the item-codec peek canary.
    """
    model = _model(bias_mode="time").eval()
    item_codec, prev_feat, time_min, valid_mask = _inputs()
    with torch.no_grad():
        mu, _ = model(item_codec, prev_feat, time_min, valid_mask)
        later = time_min.clone()
        later[:, -1] = later[:, -1] + 1e5
        mu_later, _ = model(item_codec, prev_feat, later, valid_mask)
    assert torch.equal(mu[:, -1], mu_later[:, -1])


# -- the three pre-registered bias ablations are real, distinct arms -----------------------

def test_bias_modes_allocate_exactly_the_parameters_they_claim():
    n_heads, n_rbf = 4, 8
    expected = {"none": 0, "distance": n_heads * n_rbf, "time": n_heads * n_rbf,
                "distance_time": 2 * n_heads * n_rbf}
    for mode, n in expected.items():
        bias = DistanceTimeAttentionBias(n_heads, n_rbf, mode)
        assert sum(p.numel() for p in bias.parameters()) == n, mode


def test_bias_is_zero_at_init_so_every_mode_starts_as_plain_causal_attention():
    item_codec, _, time_min, valid_mask = _inputs()
    bias = DistanceTimeAttentionBias(4, 8, "distance_time")
    out = bias(item_codec, time_min, valid_mask)
    assert torch.equal(out, torch.zeros_like(out))


@pytest.mark.parametrize("bias_mode", ["distance", "time", "distance_time"])
def test_each_bias_mode_actually_moves_attention_once_its_weights_are_nonzero(bias_mode):
    bias = DistanceTimeAttentionBias(4, 8, bias_mode)
    with torch.no_grad():
        for p in bias.parameters():
            p.normal_(std=1.0)
    item_codec, _, time_min, valid_mask = _inputs()
    out = bias(item_codec, time_min, valid_mask)
    assert out.abs().max() > 1e-6


def test_distance_bias_ignores_time_and_time_bias_ignores_distance():
    """Each ablation must isolate its own channel -- otherwise the three arms do not
    decompose the mechanism they were pre-registered to decompose.
    """
    item_codec, _, time_min, valid_mask = _inputs()
    other_codec, _, other_time, _ = _inputs(seed=99)

    dist_bias = DistanceTimeAttentionBias(4, 8, "distance")
    time_bias = DistanceTimeAttentionBias(4, 8, "time")
    with torch.no_grad():
        for m in (dist_bias, time_bias):
            for p in m.parameters():
                p.normal_(std=1.0)

    assert torch.equal(dist_bias(item_codec, time_min, valid_mask),
                       dist_bias(item_codec, other_time, valid_mask))
    assert torch.equal(time_bias(item_codec, time_min, valid_mask),
                       time_bias(other_codec, time_min, valid_mask))


def test_invalid_nodes_get_zero_geometric_bias():
    """An all-zeros start token is not 'an item at the origin'; it must contribute no bias."""
    bias = DistanceTimeAttentionBias(4, 8, "distance_time")
    with torch.no_grad():
        for p in bias.parameters():
            p.normal_(std=1.0)
    item_codec, _, time_min, valid_mask = _inputs()
    node_valid = valid_mask.clone()
    node_valid[:, 0] = False
    out = bias(item_codec, time_min, node_valid)
    assert torch.equal(out[:, :, :, 0], torch.zeros_like(out[:, :, :, 0]))
    assert torch.equal(out[:, :, 0, :], torch.zeros_like(out[:, :, 0, :]))


# -- matched parameter budget --------------------------------------------------------------

def test_body_c_parameter_budget_matches_the_gru_within_ten_percent():
    gru = KnowledgeFilter(EPISODE_HIDDEN, FILTER_HIDDEN, LATENT_DIM)
    trf = CausalEventTransformerFilter(EPISODE_HIDDEN, FILTER_HIDDEN, LATENT_DIM,
                                       n_heads=4, d_ff=64, n_rbf=8)
    n_gru, n_trf = count_body_params(gru), count_body_params(trf)
    assert abs(n_trf - n_gru) / n_gru <= 0.10, f"GRU {n_gru} vs transformer {n_trf}"


def test_both_bodies_expose_the_same_head_names_for_the_fa4_gradient_canary():
    gru = KnowledgeFilter(EPISODE_HIDDEN, FILTER_HIDDEN, LATENT_DIM)
    trf = CausalEventTransformerFilter(EPISODE_HIDDEN, FILTER_HIDDEN, LATENT_DIM)
    for body in (gru, trf):
        assert hasattr(body, "mu_head") and hasattr(body, "log_sigma_head")


# -- the simulator's extended gradient canary must run against body (c) --------------------

def test_gradient_reaches_the_mu_and_log_sigma_heads_and_they_move():
    """Assert non-zero gradient at the filter's mu-head AND
    log-sigma-head from the output losses alone, and that both change after one step. The
    log-sigma limb only receives gradient through the reparameterisation, so a
    non-reparameterised draw would freeze the belief's width silently.
    """
    from atom.filter.gru import sample_u

    model = _model()
    heads = FiLMHeadsSupport(CODEC_DIM, LATENT_DIM, FILTER_HIDDEN)
    item_codec, prev_feat, time_min, valid_mask = _inputs()
    target = (torch.rand(N, T) > 0.5).float()

    mu, log_sigma = model(item_codec, prev_feat, time_min, valid_mask)
    u = sample_u(mu, log_sigma, k=2)
    correctness_logits, _ = heads(item_codec, u)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        correctness_logits, target.unsqueeze(0).expand(2, -1, -1))

    body = model.knowledge_filter
    before = {n: p.detach().clone() for n, p in body.named_parameters()
              if n.startswith(("mu_head", "log_sigma_head"))}
    opt = torch.optim.SGD(list(model.parameters()) + list(heads.parameters()), lr=0.1)
    opt.zero_grad()
    loss.backward()

    for name in ("mu_head.weight", "log_sigma_head.weight"):
        grad = dict(body.named_parameters())[name].grad
        assert grad is not None and grad.abs().sum() > 0, f"dead gradient at {name}"

    opt.step()
    for name, old in before.items():
        assert not torch.equal(old, dict(body.named_parameters())[name].detach()), name


# -- the support head ----------------------------------------------------------------------

def test_support_head_shapes_and_no_dead_engagement_parameters():
    heads = FiLMHeadsSupport(CODEC_DIM, LATENT_DIM, FILTER_HIDDEN)
    u = torch.randn(3, N, T, LATENT_DIM)
    correctness, support = heads(torch.randn(N, T, CODEC_DIM), u)
    assert correctness.shape == (3, N, T)
    assert support.shape == (3, N, T, N_SUPPORT)
    assert not any(n.startswith("engagement") for n, _ in heads.named_parameters())


def test_support_code_reconstructs_from_the_padded_batch_channels():
    """The bake-off derives support_code as hint_bin + 2*saw_answer from the padded batch
    rather than adding a channel to `build_padded_batch_v2`. That identity IS
    `fa_manifest.support_label`'s definition, so it is pinned by a test rather than assumed.
    """
    import numpy as np
    import pandas as pd

    ev = pd.DataFrame({
        "learner": ["a", "a", "a", "b", "b"],
        "t": [0, 1, 2, 0, 1],
        "problem_id": [1, 2, 1, 2, 1],
        "score": [1.0, 0.0, 1.0, 0.0, 1.0],
        "score_missing": [False] * 5,
        "hint_count": [0, 3, 0, 1, 0],
        "saw_answer": [0, 1, 1, 0, 0],
        "support_code": [0, 3, 2, 1, 0],
        "timestamp": pd.to_datetime(
            ["2024-01-01 00:00", "2024-01-01 00:05", "2024-01-01 01:00",
             "2024-01-02 00:00", "2024-01-02 00:30"], utc=True),
    })
    codec = np.arange(2 * CODEC_DIM, dtype=np.float32).reshape(2, CODEC_DIM)
    batch = build_padded_batch_v2(ev, codec, {1: 0, 2: 1}, "cpu", include_time_min=True)

    derived = (batch["hint_bin"] + 2 * batch["saw_answer"]).long()
    for _, r in ev.iterrows():
        n = batch["learners"].index(r["learner"])
        assert derived[n, int(r["t"])].item() == r["support_code"]

    # time_min: minutes since each learner's own first event, 0 at t=0.
    assert batch["time_min"][0].tolist()[:3] == [0.0, 5.0, 60.0]
    assert batch["time_min"][1].tolist()[:2] == [0.0, 30.0]


def test_include_time_min_defaults_off_so_fa1_callers_are_unaffected():
    import numpy as np
    import pandas as pd

    ev = pd.DataFrame({
        "learner": ["a", "a"], "t": [0, 1], "problem_id": [1, 1],
        "score": [1.0, 0.0], "score_missing": [False, False],
        "hint_count": [0, 0], "saw_answer": [0, 0],
        "timestamp": pd.to_datetime(["2024-01-01", "2024-01-02"], utc=True),
    })
    codec = np.zeros((1, CODEC_DIM), dtype=np.float32)
    assert "time_min" not in build_padded_batch_v2(ev, codec, {1: 0}, "cpu")


# -- determinism ----------------------------------------------------------------------------

def test_forward_is_deterministic_under_a_fixed_seed():
    args = _inputs()
    a, _ = _model(seed=3).eval()(*args)
    b, _ = _model(seed=3).eval()(*args)
    assert torch.equal(a, b)


# -- the bake-off handoff surface -----------------------------------------------------------

def test_factory_builds_every_body_behind_one_uniform_forward():
    """The body is a config switch: the bake-off must be able to swap arms without
    branching, so every body has to accept the same call."""
    from atom.filter.factory import BODIES, build_filter

    args = _inputs()
    for body in BODIES:
        model, forward = build_filter(body, CODEC_DIM, EPISODE_HIDDEN, FILTER_HIDDEN, LATENT_DIM)
        mu, log_sigma = forward(*args)
        assert mu.shape == (N, T, LATENT_DIM), body
        assert log_sigma.shape == (N, T, LATENT_DIM), body
        assert torch.isfinite(mu).all() and torch.isfinite(log_sigma).all(), body


def test_factory_gru_arm_is_fa1s_shipped_filter_unchanged():
    from atom.filter.factory import build_filter
    from atom.filter.s1v2 import S1FilterV2

    model, _ = build_filter("gru", CODEC_DIM, EPISODE_HIDDEN, FILTER_HIDDEN, LATENT_DIM)
    assert isinstance(model, S1FilterV2)


def test_factory_rejects_an_unknown_body_rather_than_silently_defaulting():
    from atom.filter.factory import build_filter

    with pytest.raises(ValueError, match="unknown filter body"):
        build_filter("transformer", CODEC_DIM, EPISODE_HIDDEN, FILTER_HIDDEN, LATENT_DIM)


def test_factory_preserves_query_independence_on_every_body():
    from atom.filter.factory import BODIES, build_filter

    item_codec, prev_feat, time_min, valid_mask = _inputs()
    swapped = item_codec.clone()
    swapped[:, -1] = item_codec[torch.randperm(N), -1]
    for body in BODIES:
        model, forward = build_filter(body, CODEC_DIM, EPISODE_HIDDEN, FILTER_HIDDEN, LATENT_DIM)
        model.eval()
        with torch.no_grad():
            mu = forward(item_codec, prev_feat, time_min, valid_mask)[0]
            mu_s = forward(swapped, prev_feat, time_min, valid_mask)[0]
        assert torch.equal(mu[:, -1], mu_s[:, -1]), body
