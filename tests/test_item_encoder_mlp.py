"""Unit tests for the `sigma_parameterization` passthrough of FA's `S1FilterV2ItemMLP`
(`atom.filter.item_encoder_mlp`). It mirrors CW's identical wrapper
(`atom.simulator.cw_item_mlp.S1FilterV2ItemMLP`), so this mirrors
`tests/test_cw_item_mlp.py::test_cw_item_mlp_threads_precision_accumulation_without_changing_mu_contract`
exactly, on the FA-side class used in the width-parameterisation ablation."""

from __future__ import annotations

import torch

from atom.filter.item_encoder_mlp import S1FilterV2ItemMLP


def test_default_sigma_parameterization_is_byte_identical_to_pre_a3():
    """`sigma_parameterization` defaults to "direct" -- every existing FA caller (A1, A2,
    every fa_sim*_full_run.py) constructs this class without the kwarg, so their forward
    pass must be untouched by this addition."""
    torch.manual_seed(0)
    m_old = S1FilterV2ItemMLP(codec_dim=8, episode_hidden=6, filter_hidden=7, latent_dim=4)
    torch.manual_seed(0)
    m_new = S1FilterV2ItemMLP(codec_dim=8, episode_hidden=6, filter_hidden=7, latent_dim=4,
                              sigma_parameterization="direct")
    item_codec = torch.randn(2, 5, 8)
    prev_feat = torch.randn(2, 5, 5)
    with torch.no_grad():
        mu_old, ls_old = m_old(item_codec, prev_feat)
        mu_new, ls_new = m_new(item_codec, prev_feat)
    assert torch.equal(mu_old, mu_new)
    assert torch.equal(ls_old, ls_new)


def test_precision_accumulating_threads_through_to_knowledge_filter():
    """The flag actually reaches `KnowledgeFilter` (not silently dropped) -- log_sigma at
    the first position is the prior (log_sigma=0, precision=1) and monotonically narrows
    thereafter, exactly `KnowledgeFilter`'s own contract (`tests/test_s1_filter.py`)."""
    m = S1FilterV2ItemMLP(
        codec_dim=8, episode_hidden=6, filter_hidden=7, latent_dim=4,
        sigma_parameterization="precision_accumulating",
    )
    mu, log_sigma = m(torch.randn(2, 5, 8), torch.randn(2, 5, 5))
    assert mu.shape == log_sigma.shape == (2, 5, 4)
    assert torch.equal(log_sigma[:, 0], torch.zeros_like(log_sigma[:, 0]))
    assert torch.all(log_sigma[:, 1:] <= log_sigma[:, :-1])


def test_unknown_sigma_parameterization_raises():
    import pytest
    with pytest.raises(ValueError):
        S1FilterV2ItemMLP(codec_dim=8, episode_hidden=6, filter_hidden=7, latent_dim=4,
                          sigma_parameterization="nonsense")
