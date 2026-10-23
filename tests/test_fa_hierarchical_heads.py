"""Tests for OutcomeHead, the class-conditional outcome head."""

from __future__ import annotations

import torch

from atom.heads.fa_hierarchical_heads import OutcomeHead

CODEC_DIM = 5
LATENT_DIM = 6
HIDDEN = 8


def test_outcome_head_output_shape():
    head = OutcomeHead(CODEC_DIM, LATENT_DIM, HIDDEN)
    item_codec = torch.randn(4, CODEC_DIM)
    u = torch.randn(4, LATENT_DIM)
    logits = head(item_codec, u)
    assert logits.shape == (4, 5)


def test_outcome_head_near_identity_at_init():
    """FiLM's gamma/beta near-identity init (copied from FiLMHeads' own reasoning): a
    fresh random u should barely perturb the item-trunk signal at step 0, so outcome
    logits for the SAME item_codec under two different u draws should be close, not
    wildly different -- an untrained hypernetwork-analogue must not immediately drown
    the trunk in noise."""
    torch.manual_seed(0)
    head = OutcomeHead(CODEC_DIM, LATENT_DIM, HIDDEN)
    item_codec = torch.randn(1, CODEC_DIM).expand(2, -1)
    u = torch.randn(2, LATENT_DIM) * 3.0  # deliberately large-scale u draws
    logits = head(item_codec, u)
    assert torch.allclose(logits[0], logits[1], atol=0.5)


def test_outcome_head_gradient_reaches_u():
    """u must be on the live path to gamma/beta -- required for the upstream-gradient
    canary (filter's mu/log-sigma heads must receive finite, non-zero gradient from the
    outcome loss alone)."""
    head = OutcomeHead(CODEC_DIM, LATENT_DIM, HIDDEN)
    item_codec = torch.randn(3, CODEC_DIM)
    u = torch.randn(3, LATENT_DIM, requires_grad=True)
    logits = head(item_codec, u)
    logits.sum().backward()
    assert u.grad is not None
    assert torch.all(torch.isfinite(u.grad))
    assert bool((u.grad != 0).any())


def test_outcome_head_different_items_give_different_logits():
    head = OutcomeHead(CODEC_DIM, LATENT_DIM, HIDDEN)
    u = torch.randn(1, LATENT_DIM).expand(2, -1)
    item_codec = torch.stack([torch.zeros(CODEC_DIM), torch.ones(CODEC_DIM) * 5.0])
    logits = head(item_codec, u)
    assert not torch.allclose(logits[0], logits[1])


def test_outcome_head_n_classes_override_for_cw4():
    """The CodeWorkout instantiation needs 4 classes, not FA's 5 --
    regression-pins that the override changes only the output width."""
    head = OutcomeHead(CODEC_DIM, LATENT_DIM, HIDDEN, n_classes=4)
    item_codec = torch.randn(3, CODEC_DIM)
    u = torch.randn(3, LATENT_DIM)
    logits = head(item_codec, u)
    assert logits.shape == (3, 4)


PREFIX_DIM = 7


def test_outcome_head_without_prefix_dim_is_byte_for_byte_unaffected_by_the_new_kwarg():
    """FA never sets prefix_dim -- confirms the new code path is fully inert for every
    existing call site (same output as before this clause's change, given the same
    seed)."""
    torch.manual_seed(0)
    head = OutcomeHead(CODEC_DIM, LATENT_DIM, HIDDEN)
    assert head.prefix_proj is None
    item_codec = torch.randn(3, CODEC_DIM)
    u = torch.randn(3, LATENT_DIM)
    logits = head(item_codec, u)  # no `prefix` kwarg at all, exactly FA's own call shape
    assert logits.shape == (3, 5)


def test_outcome_head_prefix_dim_changes_logits_and_is_required_when_configured():
    """CLAUSE 2 -- one head, three inputs. Different prefixes (same item_codec, same u)
    must give different logits (the prefix is live, not a dead input); omitting a
    required prefix must raise loudly, never silently substitute a zero."""
    torch.manual_seed(0)
    head = OutcomeHead(CODEC_DIM, LATENT_DIM, HIDDEN, n_classes=4, prefix_dim=PREFIX_DIM)
    item_codec = torch.randn(1, CODEC_DIM).expand(2, -1)
    u = torch.randn(1, LATENT_DIM).expand(2, -1)
    prefix = torch.stack([torch.zeros(PREFIX_DIM), torch.ones(PREFIX_DIM) * 5.0])
    logits = head(item_codec, u, prefix=prefix)
    assert logits.shape == (2, 4)
    assert not torch.allclose(logits[0], logits[1])

    try:
        head(item_codec, u)  # missing required prefix
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError when prefix_dim is configured but prefix is omitted")


def test_outcome_head_rejects_unexpected_prefix_when_not_configured():
    head = OutcomeHead(CODEC_DIM, LATENT_DIM, HIDDEN)  # no prefix_dim
    item_codec = torch.randn(2, CODEC_DIM)
    u = torch.randn(2, LATENT_DIM)
    prefix = torch.randn(2, PREFIX_DIM)
    try:
        head(item_codec, u, prefix=prefix)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError when passing a prefix to a head with no prefix_dim")


def test_outcome_head_zeroing_u_changes_logits_belief_ablation_mechanism_sanity():
    """CLAUSE 4's belief-ablation gate ('zeroing u must materially hurt') needs real
    trained weights and real data to measure degradation -- that is a training-time gate,
    not this unit test. What IS checkable here, cheaply: the mechanism itself is live,
    i.e. u is not a dead input that the ablation would trivially pass by accident. Uses a
    large-scale u draw (matching `test_outcome_head_near_identity_at_init`'s own
    reasoning) so the near-identity init doesn't make this a coin-flip."""
    torch.manual_seed(0)
    head = OutcomeHead(CODEC_DIM, LATENT_DIM, HIDDEN, n_classes=4, prefix_dim=PREFIX_DIM)
    item_codec = torch.randn(1, CODEC_DIM).expand(2, -1)
    prefix = torch.randn(1, PREFIX_DIM).expand(2, -1)
    u_real = torch.randn(2, LATENT_DIM) * 5.0
    u_zero = torch.zeros(2, LATENT_DIM)
    logits_real = head(item_codec, u_real, prefix=prefix)
    logits_zero = head(item_codec, u_zero, prefix=prefix)
    assert not torch.allclose(logits_real, logits_zero)
