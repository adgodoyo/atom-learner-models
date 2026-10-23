"""End-to-end batch and upstream-gradient canary for the CodeWorkout generative simulator,
mirroring `test_fa_alt2.py::
test_e2b_upstream_gradient_canary_reaches_the_filter` but through CW's OWN pipeline:
`AttemptEncoder` (episode summary) -> the cross-episode filter -> `gather_belief` ->
`reparameterize_from_log_sigma` -> per-episode `u`, broadcast to that episode's attempts
(an asserted invariant) -> `Hypernetwork` -> LoRA-hooked backbone
forward on `cw_alt`'s OWN segment-reserved prompts -> `OutcomeHead` (the 4-class head)
+ the production head, both losses backward together.

This exercises `cw_alt.encode_example`/`forward_personalised` against a real (if tiny,
stub) forward+backward pass rather than constructing them in isolation, so the CodeWorkout
production pipeline is shown to be wired correctly before any GPU run.
`scripts/cw_5_b2_pilot.py` runs the same shape of canary once against the real backbone.
"""

from __future__ import annotations

import torch

from atom.binding import (Hypernetwork, attach_lora_basis, reparameterize_from_log_sigma)
from atom.binding.stub_causal_lm import CausalStubLM
from atom.data.cw_outcome_label import N_OUTCOME
from atom.filter.cw_attempt_encoder import AttemptEncoder, build_scalar_feats
from atom.filter.factory import build_filter
from atom.heads.fa_hierarchical_heads import OutcomeHead
from atom.simulator.cw_alt import collate, encode_example, forward_personalised, gather_belief
from atom.simulator.cw_train_step import gather_prefix_for_attempts
from atom.simulator.fa_alt2 import outcome_loss, production_loss

CODE_DIM, DIFF_DIM, PROJ_DIM, ATTEMPT_HIDDEN, SUMMARY_DIM = 5, 5, 6, 8, 5
CODEC_DIM, LATENT_DIM, EPISODE_HIDDEN, FILTER_HIDDEN = 5, 6, 8, 8
HIDDEN_DIM = 32
N_HEADS, D_FF, N_RBF = 2, 8, 4


class _FakeTok:
    pad_token_id = 0
    eos_token_id = 63

    def __call__(self, text: str, add_special_tokens: bool = True) -> dict:
        words = [w for w in text.replace("\n", " \n ").split(" ") if w != ""]
        ids = [(sum(ord(c) for c in w) % 40) + 2 for w in words]
        if add_special_tokens:
            ids = [1, *ids]
        return {"input_ids": ids}


def test_e2e_cw5_production_pipeline_gradient_reaches_every_trainable_module():
    """Two episodes (learners 0 and 1), two attempts each -- exercises attempt-1's
    empty/START prior-code segment on one row, a real prior-code
    revision on another, and both `outcome_loss` + `production_loss` combined."""
    torch.manual_seed(0)
    tok = _FakeTok()
    model = CausalStubLM(vocab_size=64, dim=HIDDEN_DIM, n_layers=2, n_heads=4,
                         pad_token_id=tok.pad_token_id, seed=0)
    bases, context = attach_lora_basis(model, m=8, r=4, seed=1)
    hyper = Hypernetwork(u_dim=LATENT_DIM, m=8, kind="mlp", seed=2)
    filt, filt_forward = build_filter("gru", codec_dim=CODEC_DIM, episode_hidden=EPISODE_HIDDEN,
                                      filter_hidden=FILTER_HIDDEN, latent_dim=LATENT_DIM,
                                      n_heads=N_HEADS, d_ff=D_FF, n_rbf=N_RBF)
    attempt_encoder = AttemptEncoder(code_dim=CODE_DIM, diff_dim=DIFF_DIM, proj_dim=PROJ_DIM,
                                     hidden=ATTEMPT_HIDDEN, summary_dim=SUMMARY_DIM)
    outcome_head = OutcomeHead(codec_dim=CODEC_DIM, latent_dim=LATENT_DIM, hidden=16,
                               n_classes=N_OUTCOME, prefix_dim=ATTEMPT_HIDDEN)

    # -- AttemptEncoder: 2 episodes, 2 attempts each --
    n_ep, t_max = 2, 2
    code_vec = torch.randn(n_ep, t_max, CODE_DIM)
    diff_vec = torch.randn(n_ep, t_max, DIFF_DIM)
    compile_ok = torch.tensor([[1.0, 1.0], [0.0, 1.0]])
    score_frac = torch.tensor([[0.0, 1.0], [0.0, 1.0]])
    n_compile_errors = torch.zeros(n_ep, t_max)
    attempt_idx = torch.tensor([[0.0, 1.0], [0.0, 1.0]])
    scalar_feats = build_scalar_feats(compile_ok, score_frac, n_compile_errors, attempt_idx)
    lengths = torch.tensor([2, 2], dtype=torch.long)
    prefix_states, summary = attempt_encoder(code_vec, diff_vec, scalar_feats, lengths)

    # -- cross-episode filter: one (learner, t) position per episode, feeding the SAME
    # `u` to both of that episode's attempts (the broadcast invariant, i-c) -- `prev_feat`
    # is built FROM `attempt_encoder`'s own `summary` output (matches production's
    # `build_episode_batch`/`build_prev_item_codec` wiring, `cw_filter_batch.py`): without
    # this, `summary_head` would have no path to any loss in this synthetic fixture at
    # all, a false negative in the gradient canary this test exists to run, not a defect
    # in the pipeline being tested (SUMMARY_DIM==5==the filter's own `prev_feat` width).
    item_codec_ep = torch.randn(2, 3, CODEC_DIM)  # [n_learners=2, T=3, codec_dim]
    base_prev_feat = torch.randn(2, 3, SUMMARY_DIM)
    prev_feat_ep = base_prev_feat + summary.unsqueeze(1)
    time_min = torch.arange(3, dtype=torch.float32).unsqueeze(0).expand(2, -1) * 10.0
    valid_mask = torch.ones(2, 3, dtype=torch.bool)
    mu_all, log_sigma_all = filt_forward(item_codec_ep, prev_feat_ep, time_min, valid_mask)
    episode_learner_idx = torch.tensor([0, 1])   # episode 0 -> learner 0, episode 1 -> learner 1
    episode_t_idx = torch.tensor([2, 2])
    mu_sel, log_sigma_sel = gather_belief(mu_all, log_sigma_all, episode_learner_idx, episode_t_idx)
    u_ep = reparameterize_from_log_sigma(mu_sel, log_sigma_sel, k=1)[:, 0, :]  # [2, LATENT_DIM]

    attempt_episode_row_idx = torch.tensor([0, 0, 1, 1])  # 4 attempts: 2 per episode
    attempt_within_idx = torch.tensor([0, 1, 0, 1])
    u_attempt = u_ep[attempt_episode_row_idx]
    item_codec_attempt = item_codec_ep[episode_learner_idx, episode_t_idx][attempt_episode_row_idx]
    prefix_attempt = gather_prefix_for_attempts(prefix_states, attempt_episode_row_idx, attempt_within_idx)

    # -- cw_alt: 4 attempts, mixing a first-attempt (empty/START prior-code) row with a
    # real-revision row, across two different outcome classes --
    examples = [
        encode_example("Given 2 ints a and b, return their sum.", None,
                       "public int f(int a, int b) { return a+b; }", tok, max_len=64, outcome_code=3),
        encode_example("Given 2 ints a and b, return their sum.",
                       "public int f(int a, int b) { return a+b; }",
                       "public int f(int a, int b) { return a+b; } // done", tok, max_len=64, outcome_code=3),
        encode_example("Return true if n is prime.", None,
                       "public boolean isPrime(int n) { return false; }", tok, max_len=64, outcome_code=1),
        encode_example("Return true if n is prime.", "public boolean isPrime(int n) { return false; }",
                       "public boolean isPrime(int n) { return n > 1; }", tok, max_len=64, outcome_code=2),
    ]
    batch = collate(examples, pad_id=tok.pad_token_id)
    assert batch["is_first_attempt"].tolist() == [True, False, True, False]

    outcome_target = torch.tensor([3, 3, 1, 2])
    outcome_valid = torch.tensor([True, True, True, True])

    trainable = (
        list(filt.parameters()) + list(hyper.parameters()) + list(outcome_head.parameters())
        + list(attempt_encoder.parameters())
        + [param for b in bases.values() for param in b.parameters()]
    )
    optimizer = torch.optim.Adam(trainable, lr=1e-2)

    mu_head_before = filt.knowledge_filter.mu_head.weight.detach().clone()
    summary_head_before = attempt_encoder.summary_head.weight.detach().clone()
    outcome_out_before = outcome_head.outcome.weight.detach().clone()

    def run_step():
        optimizer.zero_grad()
        prefix_states_, summary_ = attempt_encoder(code_vec, diff_vec, scalar_feats, lengths)
        prev_feat_ep_ = base_prev_feat + summary_.unsqueeze(1)
        mu_all_, log_sigma_all_ = filt_forward(item_codec_ep, prev_feat_ep_, time_min, valid_mask)
        mu_sel_, log_sigma_sel_ = gather_belief(mu_all_, log_sigma_all_, episode_learner_idx, episode_t_idx)
        u_ep_ = reparameterize_from_log_sigma(mu_sel_, log_sigma_sel_, k=1)[:, 0, :]
        u_attempt_ = u_ep_[attempt_episode_row_idx]
        prefix_attempt_ = gather_prefix_for_attempts(prefix_states_, attempt_episode_row_idx, attempt_within_idx)

        out = forward_personalised(model, hyper, context, u_attempt_, batch["input_ids"],
                                   batch["attention_mask"], batch["prompt_end_index"])
        outcome_logits = outcome_head(item_codec_attempt, u_attempt_, prefix=prefix_attempt_)
        l_outcome = outcome_loss(outcome_logits, outcome_target, outcome_valid)
        l_prod = production_loss(out["lm_logits"], batch["labels"], batch["production_valid"])
        loss = l_outcome + l_prod
        loss.backward()
        return loss

    loss1 = run_step()
    assert torch.isfinite(loss1)
    optimizer.step()  # step 1: LoRA B starts at zero (FA's own documented special case) --
    # u's only path to the production loss is scaled by B and is exactly zero until this
    # step moves B away from zero, so the FIRST step's gradient at the filter/hyper is only
    # guaranteed via the OUTCOME head's path (which has no such zero-init gate).

    for p_ in [filt.knowledge_filter.mu_head.weight, filt.knowledge_filter.log_sigma_head.weight,
              attempt_encoder.summary_head.weight, outcome_head.outcome.weight]:
        assert p_.grad is not None
        assert torch.isfinite(p_.grad).all()
        assert bool((p_.grad != 0).any()), "a trainable module received zero gradient"

    loss2 = run_step()
    assert torch.isfinite(loss2)
    optimizer.step()  # step 2: B has moved off zero -- gradient must now reach EVERYTHING,
    # including through the production loss's path back to u/mu/log-sigma.
    for p_ in [filt.knowledge_filter.mu_head.weight, filt.knowledge_filter.log_sigma_head.weight]:
        assert p_.grad is not None and torch.isfinite(p_.grad).all() and bool((p_.grad != 0).any())
    for b in bases.values():
        for param in b.parameters():
            assert param.grad is not None and torch.isfinite(param.grad).all()

    assert not torch.equal(filt.knowledge_filter.mu_head.weight, mu_head_before)
    assert not torch.equal(attempt_encoder.summary_head.weight, summary_head_before)
    assert not torch.equal(outcome_head.outcome.weight, outcome_out_before)
