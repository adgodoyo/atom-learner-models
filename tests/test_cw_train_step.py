import pandas as pd
import pytest
import torch

from atom.filter.factory import build_filter
from atom.heads.fa_hierarchical_heads import OutcomeHead
from atom.simulator.cw_train_step import (
    belief_ablated_logits,
    build_cw_targets,
    gather_prefix_for_attempts,
    grad_norm_at_belief_junction,
    losses_for_cw_hierarchical_batch,
    y_ever_loss_for_batch,
)

CODEC_DIM = 2
EP_HIDDEN = 4
FILTER_HIDDEN = 4
LATENT_DIM = 3
PREFIX_HIDDEN = 5  # AttemptEncoder's own hidden size in this test


def _filter_batch():
    """2 episodes, 1 learner: episode 0 at t=0, episode 1 at t=1 (matches
    test_cw_filter_batch.py's own fixture shape)."""
    item_codec = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])  # [1 learner, T=2, codec_dim=2]
    prev_feat = torch.zeros(1, 2, 1)  # prev_feat_dim=1, unused content for this test
    return {"item_codec": item_codec, "prev_feat": prev_feat,
            "time_min": torch.zeros(1, 2), "valid_mask": torch.ones(1, 2, dtype=torch.bool)}


def _attempts_targets():
    return pd.DataFrame({
        "compile_result": ["Error", "Success", "Success"],
        "score_frac": [0.0, 1.0, 0.5],
    })


def test_gather_prefix_for_attempts_selects_correct_episode_and_step():
    prefix_states = torch.arange(2 * 3 * PREFIX_HIDDEN, dtype=torch.float32).reshape(2, 3, PREFIX_HIDDEN)
    ep_row_idx = torch.tensor([0, 0, 1])
    within_idx = torch.tensor([0, 1, 0])
    out = gather_prefix_for_attempts(prefix_states, ep_row_idx, within_idx)
    assert torch.equal(out[0], prefix_states[0, 0])
    assert torch.equal(out[1], prefix_states[0, 1])
    assert torch.equal(out[2], prefix_states[1, 0])


def test_losses_for_cw_hierarchical_batch_runs_end_to_end_and_shares_u_within_episode():
    torch.manual_seed(0)
    model, filt_forward = build_filter("gru", CODEC_DIM, EP_HIDDEN, FILTER_HIDDEN, LATENT_DIM,
                                       prev_feat_dim=1)
    outcome_head = OutcomeHead(CODEC_DIM, LATENT_DIM, hidden=6, n_classes=4, prefix_dim=PREFIX_HIDDEN)
    filter_batch = _filter_batch()

    # 3 attempts: 2 in episode 0 (row 0), 1 in episode 1 (row 1) -- matches _attempts_targets()
    prefix_states = torch.randn(2, 2, PREFIX_HIDDEN)  # [N_episodes_in_attempt_batch=2, T_max=2, hidden]
    attempt_episode_row_idx = torch.tensor([0, 0, 1])
    attempt_within_idx = torch.tensor([0, 1, 0])
    episode_learner_idx = torch.tensor([0, 0])  # both episodes belong to the one learner
    episode_t_idx = torch.tensor([0, 1])        # episode 0 -> t=0, episode 1 -> t=1

    targets = build_cw_targets(_attempts_targets(), device="cpu")
    l_outcome, kl, logits = losses_for_cw_hierarchical_batch(
        filt_forward, outcome_head, filter_batch, episode_learner_idx, episode_t_idx,
        prefix_states, attempt_episode_row_idx, attempt_within_idx, targets, beta_kl=0.01)

    assert logits.shape == (3, 4)
    assert torch.isfinite(l_outcome) and l_outcome.item() >= 0
    assert torch.isfinite(kl)

    # the two attempts in episode 0 (rows 0,1) must have been scored against the SAME u
    # and the SAME item_codec (only the prefix differs) -- confirmed indirectly: if we
    # re-run with the prefix for row 1 forced equal to row 0's, the logits must match
    # exactly (same u, same item_codec, same prefix -> identical FiLM output).
    prefix_states_equal = prefix_states.clone()
    prefix_states_equal[0, 1] = prefix_states_equal[0, 0]
    _, _, logits_eq = losses_for_cw_hierarchical_batch(
        filt_forward, outcome_head, filter_batch, episode_learner_idx, episode_t_idx,
        prefix_states_equal, attempt_episode_row_idx, attempt_within_idx, targets, beta_kl=0.01)
    assert torch.allclose(logits_eq[0], logits_eq[1])


def test_likelihood_default_single_is_bit_identical_to_pre_sigma_body():
    """Omitting `likelihood` must execute the historical sample-zero computation."""
    from atom.binding import reparameterize_from_log_sigma
    from atom.simulator.fa_alt2 import outcome_loss
    from atom.simulator.fa_train_step import gather_belief_for_batch, gather_item_codec_for_batch

    torch.manual_seed(0)
    _model, filt_forward = build_filter(
        "gru", CODEC_DIM, EP_HIDDEN, FILTER_HIDDEN, LATENT_DIM, prev_feat_dim=1
    )
    outcome_head = OutcomeHead(CODEC_DIM, LATENT_DIM, hidden=6, n_classes=4, prefix_dim=PREFIX_HIDDEN)
    filter_batch = _filter_batch()
    prefix_states = torch.randn(2, 2, PREFIX_HIDDEN)
    attempt_episode_row_idx = torch.tensor([0, 0, 1])
    attempt_within_idx = torch.tensor([0, 1, 0])
    episode_learner_idx = torch.tensor([0, 0])
    episode_t_idx = torch.tensor([0, 1])
    targets = build_cw_targets(_attempts_targets(), device="cpu")

    torch.manual_seed(1234)
    got_loss, got_kl, got_logits = losses_for_cw_hierarchical_batch(
        filt_forward,
        outcome_head,
        filter_batch,
        episode_learner_idx,
        episode_t_idx,
        prefix_states,
        attempt_episode_row_idx,
        attempt_within_idx,
        targets,
        beta_kl=0.01,
        k=8,
    )

    torch.manual_seed(1234)
    mu_sel, log_sigma_sel = gather_belief_for_batch(
        filt_forward, filter_batch, episode_learner_idx, episode_t_idx
    )
    item_codec_sel = gather_item_codec_for_batch(
        filter_batch, episode_learner_idx, episode_t_idx
    )
    u_ep = reparameterize_from_log_sigma(mu_sel, log_sigma_sel, k=8)[:, 0, :]
    u_attempt = u_ep[attempt_episode_row_idx]
    item_codec_attempt = item_codec_sel[attempt_episode_row_idx]
    prefix_attempt = gather_prefix_for_attempts(
        prefix_states, attempt_episode_row_idx, attempt_within_idx
    )
    ref_logits = outcome_head(item_codec_attempt, u_attempt, prefix=prefix_attempt)
    ref_loss = outcome_loss(
        ref_logits, targets["outcome_target"], targets["outcome_valid"]
    )

    assert torch.equal(got_loss, ref_loss)
    assert torch.equal(got_logits, ref_logits)
    assert torch.isfinite(got_kl)


def test_kmixture_uses_all_draws_and_returned_logits_match_its_loss():
    """The metric logits and scalar loss must describe the same K-mixture."""
    torch.manual_seed(7)
    _model, filt_forward = build_filter(
        "gru", CODEC_DIM, EP_HIDDEN, FILTER_HIDDEN, LATENT_DIM, prev_feat_dim=1
    )
    outcome_head = OutcomeHead(CODEC_DIM, LATENT_DIM, hidden=6, n_classes=4, prefix_dim=PREFIX_HIDDEN)
    targets = build_cw_targets(_attempts_targets(), device="cpu")

    loss, kl, mixture_logits = losses_for_cw_hierarchical_batch(
        filt_forward,
        outcome_head,
        _filter_batch(),
        torch.tensor([0, 0]),
        torch.tensor([0, 1]),
        torch.randn(2, 2, PREFIX_HIDDEN),
        torch.tensor([0, 0, 1]),
        torch.tensor([0, 1, 0]),
        targets,
        beta_kl=0.01,
        k=8,
        likelihood="kmixture",
    )
    probs = torch.softmax(mixture_logits, dim=-1)
    expected = -torch.log(
        probs.gather(1, targets["outcome_target"].unsqueeze(1)).squeeze(1).clamp(min=1e-8)
    ).mean()

    assert mixture_logits.shape == (3, 4)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(3), atol=1e-6)
    assert torch.allclose(loss, expected, atol=1e-6)
    assert torch.isfinite(kl)


def test_unknown_likelihood_is_rejected():
    torch.manual_seed(0)
    _model, filt_forward = build_filter(
        "gru", CODEC_DIM, EP_HIDDEN, FILTER_HIDDEN, LATENT_DIM, prev_feat_dim=1
    )
    outcome_head = OutcomeHead(CODEC_DIM, LATENT_DIM, hidden=6, n_classes=4, prefix_dim=PREFIX_HIDDEN)
    with pytest.raises(ValueError, match="unknown likelihood"):
        losses_for_cw_hierarchical_batch(
            filt_forward,
            outcome_head,
            _filter_batch(),
            torch.tensor([0, 0]),
            torch.tensor([0, 1]),
            torch.randn(2, 2, PREFIX_HIDDEN),
            torch.tensor([0, 0, 1]),
            torch.tensor([0, 1, 0]),
            build_cw_targets(_attempts_targets(), device="cpu"),
            beta_kl=0.01,
            likelihood="not-a-likelihood",
        )


def test_gradients_flow_to_filter_and_outcome_head():
    torch.manual_seed(0)
    model, filt_forward = build_filter("gru", CODEC_DIM, EP_HIDDEN, FILTER_HIDDEN, LATENT_DIM,
                                       prev_feat_dim=1)
    outcome_head = OutcomeHead(CODEC_DIM, LATENT_DIM, hidden=6, n_classes=4, prefix_dim=PREFIX_HIDDEN)
    filter_batch = _filter_batch()
    prefix_states = torch.randn(2, 2, PREFIX_HIDDEN, requires_grad=True)
    attempt_episode_row_idx = torch.tensor([0, 0, 1])
    attempt_within_idx = torch.tensor([0, 1, 0])
    episode_learner_idx = torch.tensor([0, 0])
    episode_t_idx = torch.tensor([0, 1])
    targets = build_cw_targets(_attempts_targets(), device="cpu")

    l_outcome, kl, _ = losses_for_cw_hierarchical_batch(
        filt_forward, outcome_head, filter_batch, episode_learner_idx, episode_t_idx,
        prefix_states, attempt_episode_row_idx, attempt_within_idx, targets, beta_kl=0.01)
    (l_outcome + 0.01 * kl).backward()

    assert prefix_states.grad is not None and torch.isfinite(prefix_states.grad).all()
    for p in model.parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all()
    for p in outcome_head.parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all()


def test_grad_norm_at_belief_junction_is_zero_before_backward_and_positive_after():
    """B-1 item i-e (review amendment 4): the measurement point is `mu_head`/
    `log_sigma_head`, not the LoRA basis -- `L_outcome` alone (no backbone/LoRA in this
    test at all) must already register a positive norm here, proving the outcome loss's
    contribution to the shared belief is visible at this junction."""
    torch.manual_seed(0)
    model, filt_forward = build_filter("gru", CODEC_DIM, EP_HIDDEN, FILTER_HIDDEN, LATENT_DIM,
                                       prev_feat_dim=1)
    outcome_head = OutcomeHead(CODEC_DIM, LATENT_DIM, hidden=6, n_classes=4, prefix_dim=PREFIX_HIDDEN)
    filter_batch = _filter_batch()
    prefix_states = torch.randn(2, 2, PREFIX_HIDDEN)
    attempt_episode_row_idx = torch.tensor([0, 0, 1])
    attempt_within_idx = torch.tensor([0, 1, 0])
    episode_learner_idx = torch.tensor([0, 0])
    episode_t_idx = torch.tensor([0, 1])
    targets = build_cw_targets(_attempts_targets(), device="cpu")

    assert grad_norm_at_belief_junction(model) == 0.0  # nothing has run backward yet

    l_outcome, kl, _ = losses_for_cw_hierarchical_batch(
        filt_forward, outcome_head, filter_batch, episode_learner_idx, episode_t_idx,
        prefix_states, attempt_episode_row_idx, attempt_within_idx, targets, beta_kl=0.0)
    l_outcome.backward()

    norm = grad_norm_at_belief_junction(model)
    assert norm > 0.0
    assert torch.isfinite(torch.tensor(norm))


def test_belief_ablated_logits_differ_from_real_belief_mechanism_sanity():
    """CLAUSE 4's mechanism sanity at the full-wiring level (the unit-level check
    already lives in test_fa_hierarchical_heads.py) -- confirms the ablation path is
    correctly wired end-to-end through gather_item_codec_for_batch/gather_prefix, not
    just at the bare OutcomeHead call."""
    torch.manual_seed(0)
    model, filt_forward = build_filter("gru", CODEC_DIM, EP_HIDDEN, FILTER_HIDDEN, LATENT_DIM,
                                       prev_feat_dim=1)
    outcome_head = OutcomeHead(CODEC_DIM, LATENT_DIM, hidden=6, n_classes=4, prefix_dim=PREFIX_HIDDEN)
    filter_batch = _filter_batch()
    prefix_states = torch.randn(2, 2, PREFIX_HIDDEN)
    attempt_episode_row_idx = torch.tensor([0, 0, 1])
    attempt_within_idx = torch.tensor([0, 1, 0])
    episode_learner_idx = torch.tensor([0, 0])
    episode_t_idx = torch.tensor([0, 1])
    targets = build_cw_targets(_attempts_targets(), device="cpu")

    _, _, real_logits = losses_for_cw_hierarchical_batch(
        filt_forward, outcome_head, filter_batch, episode_learner_idx, episode_t_idx,
        prefix_states, attempt_episode_row_idx, attempt_within_idx, targets, beta_kl=0.01)
    ablated_logits = belief_ablated_logits(
        outcome_head, filter_batch, episode_learner_idx, episode_t_idx,
        prefix_states, attempt_episode_row_idx, attempt_within_idx, LATENT_DIM)

    assert ablated_logits.shape == real_logits.shape
    assert not torch.allclose(real_logits, ablated_logits)


def test_y_ever_loss_for_batch_runs_end_to_end_and_gradients_flow():
    torch.manual_seed(0)
    model, filt_forward = build_filter("gru", CODEC_DIM, EP_HIDDEN, FILTER_HIDDEN, LATENT_DIM,
                                       prev_feat_dim=1)
    y_ever_head = OutcomeHead(CODEC_DIM, LATENT_DIM, hidden=6, n_classes=1)
    filter_batch = _filter_batch()
    episode_learner_idx = torch.tensor([0, 0])  # the 2 episodes in _filter_batch()
    episode_t_idx = torch.tensor([0, 1])
    y_ever_target = torch.tensor([1.0, 0.0])

    loss, logits = y_ever_loss_for_batch(filt_forward, y_ever_head, filter_batch,
                                         episode_learner_idx, episode_t_idx, y_ever_target)
    assert logits.shape == (2,)
    assert torch.isfinite(loss) and loss.item() >= 0

    loss.backward()
    for p in model.parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all()
    for p in y_ever_head.parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all()


def test_y_ever_loss_for_batch_rejects_empty_batch_gracefully():
    """A zero-episode batch (e.g. an all-invalid minibatch) must not crash BCE with a
    shape mismatch or a NaN from an empty mean."""
    torch.manual_seed(0)
    model, filt_forward = build_filter("gru", CODEC_DIM, EP_HIDDEN, FILTER_HIDDEN, LATENT_DIM,
                                       prev_feat_dim=1)
    y_ever_head = OutcomeHead(CODEC_DIM, LATENT_DIM, hidden=6, n_classes=1)
    filter_batch = _filter_batch()
    episode_learner_idx = torch.zeros(0, dtype=torch.long)
    episode_t_idx = torch.zeros(0, dtype=torch.long)
    y_ever_target = torch.zeros(0)

    loss, logits = y_ever_loss_for_batch(filt_forward, y_ever_head, filter_batch,
                                         episode_learner_idx, episode_t_idx, y_ever_target)
    assert logits.shape == (0,)
    assert loss.item() == 0.0  # same "exact zero, not 0/0 NaN" convention as outcome_loss
