"""Regression tests for `filter/factory.py::build_filter`'s `prev_feat_dim` override:
both bodies still default to FA's 5-channel `prev_feat` width, and both accept the
6-channel CodeWorkout width as an override."""

from __future__ import annotations

import torch

from atom.filter.factory import build_filter

CODEC_DIM = 6
EP_HIDDEN = 8
FILTER_HIDDEN = 8  # must be divisible by build_filter's default n_heads=4
LATENT_DIM = 4
N, T = 3, 5


def test_gru_body_default_prev_feat_dim_is_five():
    model, forward = build_filter("gru", CODEC_DIM, EP_HIDDEN, FILTER_HIDDEN, LATENT_DIM)
    item_codec = torch.randn(N, T, CODEC_DIM)
    prev_feat = torch.randn(N, T, 5)  # FA's own width
    mu, log_sigma = forward(item_codec, prev_feat, None, None)
    assert mu.shape == (N, T, LATENT_DIM)
    assert log_sigma.shape == (N, T, LATENT_DIM)


def test_gru_body_accepts_cw4_prev_feat_dim_override():
    model, forward = build_filter("gru", CODEC_DIM, EP_HIDDEN, FILTER_HIDDEN, LATENT_DIM,
                                  prev_feat_dim=6)
    item_codec = torch.randn(N, T, CODEC_DIM)
    prev_feat = torch.randn(N, T, 6)  # CodeWorkout width
    mu, log_sigma = forward(item_codec, prev_feat, None, None)
    assert mu.shape == (N, T, LATENT_DIM)


def test_c_body_accepts_cw4_prev_feat_dim_override():
    model, forward = build_filter("c_none", CODEC_DIM, EP_HIDDEN, FILTER_HIDDEN, LATENT_DIM,
                                  prev_feat_dim=6)
    item_codec = torch.randn(N, T, CODEC_DIM)
    prev_feat = torch.randn(N, T, 6)
    time_min = torch.zeros(N, T)
    valid_mask = torch.ones(N, T, dtype=torch.bool)
    mu, log_sigma = forward(item_codec, prev_feat, time_min, valid_mask)
    assert mu.shape == (N, T, LATENT_DIM)


def test_wrong_prev_feat_width_without_override_raises():
    """Sanity check that the override is load-bearing, not a silent no-op: a 6-wide
    prev_feat against the DEFAULT (5-wide) projection must fail, not broadcast."""
    model, forward = build_filter("gru", CODEC_DIM, EP_HIDDEN, FILTER_HIDDEN, LATENT_DIM)
    item_codec = torch.randn(N, T, CODEC_DIM)
    prev_feat = torch.randn(N, T, 6)
    try:
        forward(item_codec, prev_feat, None, None)
    except RuntimeError:
        return
    raise AssertionError("expected a shape-mismatch RuntimeError")
