"""CW-4g: unit tests for the item-encoder MLP variants -- correctness of the swap, and
the one invariant the task's own text elevates to a gate: the MLP must be a SHARED
function of the codec vector, never a per-item lookup, or CW's item-agnosticism claim
(no head consumes a learned item ID) is silently forfeited."""

from __future__ import annotations

import torch

from atom.filter.item_encoder_mlp import ItemEncoderMLP
from atom.simulator.cw_item_mlp import OutcomeHeadItemMLP, S1FilterV2ItemMLP


def test_item_encoder_mlp_is_a_shared_function_never_a_per_item_lookup():
    """The invariant, directly: encoding the SAME codec vector twice (as two different
    "items" in a batch) gives the SAME output, and there is no parameter whose shape
    scales with the number of items -- a per-item lookup table would fail both."""
    torch.manual_seed(0)
    enc = ItemEncoderMLP(codec_dim=64, hidden=128)
    n_params_before = sum(p.numel() for p in enc.parameters())

    codec_a = torch.randn(1, 64)
    batch = torch.cat([codec_a, codec_a, torch.randn(1, 64)], dim=0)  # items 0,1 share a vector
    with torch.no_grad():
        out = enc(batch)
    assert torch.allclose(out[0], out[1]), (
        "identical codec vectors for two different 'items' must produce identical "
        "encodings -- any divergence would mean the module has learned per-item state")

    # No parameter tensor's shape depends on a number-of-items axis -- every weight is
    # (codec_dim, hidden)/(hidden,)/(hidden, hidden) shaped, never (n_items, ...).
    for p in enc.parameters():
        assert p.shape[0] != 3, "a per-item lookup would size a dimension by batch/item count"
    n_params_after = sum(p.numel() for p in enc.parameters())
    assert n_params_before == n_params_after, "forward passes must never grow parameter count"


def test_item_encoder_mlp_generalises_to_an_unseen_codec_vector():
    """A vector never seen at training time still gets a representation from the SAME
    weights -- the concrete form of "price a question never seen" this module exists to
    keep intact."""
    torch.manual_seed(1)
    enc = ItemEncoderMLP(codec_dim=32, hidden=16)
    novel = torch.randn(1, 32) * 100  # far outside any plausible training distribution
    with torch.no_grad():
        out = enc(novel)
    assert torch.isfinite(out).all()
    assert out.shape == (1, 16)


def test_s1_filter_v2_item_mlp_matches_linear_variant_shapes_and_respects_prev_feat_dim():
    torch.manual_seed(0)
    m = S1FilterV2ItemMLP(codec_dim=64, episode_hidden=32, filter_hidden=32,
                          latent_dim=16, prev_feat_dim=16)
    item_codec = torch.randn(2, 4, 64)
    prev_feat = torch.randn(2, 4, 16)
    mu, log_sigma = m(item_codec, prev_feat)
    assert mu.shape == (2, 4, 16) and log_sigma.shape == (2, 4, 16)
    assert torch.isfinite(mu).all() and torch.isfinite(log_sigma).all()


def test_cw_item_mlp_threads_precision_accumulation_without_changing_mu_contract():
    m = S1FilterV2ItemMLP(
        codec_dim=8, episode_hidden=6, filter_hidden=7, latent_dim=4,
        prev_feat_dim=3, sigma_parameterization="precision_accumulating",
    )
    mu, log_sigma = m(torch.randn(2, 5, 8), torch.randn(2, 5, 3))
    assert mu.shape == log_sigma.shape == (2, 5, 4)
    assert torch.equal(log_sigma[:, 0], torch.zeros_like(log_sigma[:, 0]))
    assert torch.all(log_sigma[:, 1:] <= log_sigma[:, :-1])


def test_outcome_head_item_mlp_matches_outcome_head_shape_contract():
    torch.manual_seed(0)
    h = OutcomeHeadItemMLP(codec_dim=128, latent_dim=32, hidden=64, n_classes=4, prefix_dim=64)
    item_codec = torch.randn(5, 128)
    u = torch.randn(5, 32)
    prefix = torch.randn(5, 64)
    logits = h(item_codec, u, prefix=prefix)
    assert logits.shape == (5, 4)
    assert torch.isfinite(logits).all()


def test_outcome_head_item_mlp_prefix_contract_matches_outcome_head():
    """Same two-way raise as `OutcomeHead` itself -- a silently-ignored prefix or a
    missing one are both the leak/omission CLAUSE 2 exists to prevent."""
    import pytest

    h_no_prefix = OutcomeHeadItemMLP(codec_dim=45, latent_dim=32, hidden=64, n_classes=4)
    with pytest.raises(ValueError):
        h_no_prefix(torch.randn(2, 45), torch.randn(2, 32), prefix=torch.randn(2, 64))

    h_with_prefix = OutcomeHeadItemMLP(codec_dim=45, latent_dim=32, hidden=64, n_classes=4, prefix_dim=64)
    with pytest.raises(ValueError):
        h_with_prefix(torch.randn(2, 45), torch.randn(2, 32))
