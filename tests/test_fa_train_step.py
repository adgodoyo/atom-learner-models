"""Unit tests for `atom.simulator.fa_train_step`, the training step shared by the FA
simulator trainers. The key piece is `gather_belief_for_batch`: it must be exactly
interchangeable with the reference "run the filter over every learner, then gather"
pattern (it exists for cost reasons, see the module's own docstring); these tests prove
that equivalence.
"""

from __future__ import annotations

import pytest
import torch

from atom.binding import Hypernetwork, attach_lora_basis, reparameterize_from_log_sigma
from atom.binding.stub_causal_lm import CausalStubLM
from atom.eval.spread_error import spread_error_correlation
from atom.filter.factory import build_filter
from atom.heads.fa_alt2_heads import SuccessHead
from atom.heads.fa_hierarchical_heads import OutcomeHead
from atom.simulator.fa_alt2 import collate, encode_example, forward_personalised
from atom.simulator.fa_train_step import (build_targets, chunked_filter_forward,
                                          combine_hierarchical_loss, gather_belief_for_batch,
                                          gather_item_codec_for_batch, k_sample_success_predictions,
                                          losses_for_hierarchical_batch)

CODEC_DIM = 5
LATENT_DIM = 8
HIDDEN_DIM = 32
FA_TRAIN_STEP_BODIES = ("gru", "c_distance_time")


class _FakeTok:
    pad_token_id = 0
    eos_token_id = 63

    def __call__(self, text: str, add_special_tokens: bool = True) -> dict:
        words = [w for w in text.replace("\n", " \n ").split(" ") if w != ""]
        ids = [(sum(ord(c) for c in w) % 40) + 2 for w in words]
        if add_special_tokens:
            ids = [1, *ids]
        return {"input_ids": ids}


def _build_k_sample_pipeline(body: str, seed: int = 0):
    tok = _FakeTok()
    model = CausalStubLM(vocab_size=64, dim=HIDDEN_DIM, n_layers=2, n_heads=4,
                         pad_token_id=tok.pad_token_id, seed=seed)
    bases, context = attach_lora_basis(model, m=8, r=4, seed=seed + 1)
    hyper = Hypernetwork(u_dim=LATENT_DIM, m=8, kind="mlp", seed=seed + 2)
    filt, filt_forward = _build_filter(body, seed=seed)
    success_head = SuccessHead(HIDDEN_DIM)
    return tok, model, bases, context, hyper, filt, filt_forward, success_head


def _synthetic_filter_batch(n_learners: int = 6, t_len: int = 5):
    torch.manual_seed(0)
    return {
        "item_codec": torch.randn(n_learners, t_len, CODEC_DIM),
        "prev_feat": torch.randn(n_learners, t_len, 5),
        "time_min": torch.arange(t_len, dtype=torch.float32).unsqueeze(0).expand(n_learners, -1) * 10.0,
        "valid_mask": torch.ones(n_learners, t_len, dtype=torch.bool),
    }


def _build_filter(body: str, seed: int = 0):
    return build_filter(body, codec_dim=CODEC_DIM, episode_hidden=8, filter_hidden=8,
                        latent_dim=LATENT_DIM, n_heads=2, d_ff=8, n_rbf=4)


def _reference_gather(filt_forward, filter_batch, learner_idx, t_idx):
    """The reference pattern: filter forward over EVERY learner, then gather."""
    mu_all, log_sigma_all = chunked_filter_forward(filt_forward, filter_batch)
    return mu_all[learner_idx, t_idx], log_sigma_all[learner_idx, t_idx]


import pytest  # noqa: E402


@pytest.mark.parametrize("body", FA_TRAIN_STEP_BODIES)
def test_gather_belief_for_batch_matches_full_batch_gather_exactly(body):
    """The whole point of the fix: subsetting to only the referenced learners before the
    filter forward must not change a single number, for either filter body -- no filter
    body has cross-learner attention, so a learner's belief cannot depend on who else is
    in the batch."""
    filt, filt_forward = _build_filter(body)
    filt.eval()
    filter_batch = _synthetic_filter_batch(n_learners=6, t_len=5)
    # A batch that references a strict, repeating, non-sorted subset of learners --
    # exercises the torch.unique/remap path properly, not just the trivial all-learners case.
    learner_idx = torch.tensor([4, 1, 4, 0])
    t_idx = torch.tensor([2, 3, 4, 1])

    with torch.no_grad():
        mu_ref, log_sigma_ref = _reference_gather(filt_forward, filter_batch, learner_idx, t_idx)
        mu_new, log_sigma_new = gather_belief_for_batch(filt_forward, filter_batch, learner_idx, t_idx)

    assert torch.allclose(mu_ref, mu_new, atol=1e-6)
    assert torch.allclose(log_sigma_ref, log_sigma_new, atol=1e-6)


@pytest.mark.parametrize("body", FA_TRAIN_STEP_BODIES)
def test_gather_belief_for_batch_is_differentiable_and_reaches_the_filter(body):
    """The upstream-gradient canary requirement, re-checked here since this
    function replaces the training loop's only path to the filter's belief heads."""
    filt, filt_forward = _build_filter(body)
    filt.train()
    filter_batch = _synthetic_filter_batch(n_learners=6, t_len=5)
    learner_idx = torch.tensor([2, 0])
    t_idx = torch.tensor([3, 1])

    mu_sel, log_sigma_sel = gather_belief_for_batch(filt_forward, filter_batch, learner_idx, t_idx)
    (mu_sel.sum() + log_sigma_sel.sum()).backward()

    assert filt.knowledge_filter.mu_head.weight.grad is not None
    assert torch.isfinite(filt.knowledge_filter.mu_head.weight.grad).all()
    assert filt.knowledge_filter.log_sigma_head.weight.grad is not None


def test_gather_belief_for_batch_only_computes_referenced_learners():
    """The actual cost fix, asserted directly: subsetting a 6-learner filter_batch down to
    a 2-example batch referencing only 2 unique learners must call the filter body's
    forward with exactly 2 learners, not 6 -- this is what makes a ~3,500-learner
    filter_batch cost O(batch_size) per step instead of O(n_learners_total)."""
    filt, filt_forward = _build_filter("gru")
    filter_batch = _synthetic_filter_batch(n_learners=6, t_len=5)
    seen_batch_sizes = []

    def spying_forward(item_codec, prev_feat, time_min, valid_mask):
        seen_batch_sizes.append(item_codec.shape[0])
        return filt_forward(item_codec, prev_feat, time_min, valid_mask)

    learner_idx = torch.tensor([5, 5, 2])  # 2 unique learners out of 6
    t_idx = torch.tensor([0, 1, 2])
    with torch.no_grad():
        gather_belief_for_batch(spying_forward, filter_batch, learner_idx, t_idx)

    assert seen_batch_sizes == [2]


# -- gather_item_codec_for_batch + build_targets's outcome fields --------------

def test_gather_item_codec_for_batch_matches_direct_indexing():
    filter_batch = _synthetic_filter_batch(n_learners=6, t_len=5)
    learner_idx = torch.tensor([4, 1, 4, 0])
    t_idx = torch.tensor([2, 3, 4, 1])
    gathered = gather_item_codec_for_batch(filter_batch, learner_idx, t_idx)
    expected = filter_batch["item_codec"][learner_idx, t_idx]
    assert torch.equal(gathered, expected)
    assert gathered.shape == (4, CODEC_DIM)


def test_build_targets_emits_outcome_fields_alongside_the_old_ones():
    """Additive, not a replacement: old support/success keys must still be present (the
    hierarchical design keeps the old heads defined, unused, for reproducibility -- their targets must still
    build)."""
    import pandas as pd

    rows = pd.DataFrame({
        "support_code": [0, 0, 1, 2],
        "score": [1.0, 0.0, 0.0, float("nan")],
        "score_missing": [False, False, False, True],
    })
    targets = build_targets(rows, device="cpu")
    for key in ("support_target", "success_target", "success_valid", "support_valid",
               "outcome_target", "outcome_valid"):
        assert key in targets
    assert targets["outcome_target"].tolist() == [0, 1, 2, 3]  # UNAIDED_CORRECT/WRONG/HINT/REVEAL
    assert targets["outcome_valid"].tolist() == [True, True, True, True]


def test_build_targets_outcome_invalid_only_for_unaided_missing_score():
    import pandas as pd

    rows = pd.DataFrame({
        "support_code": [0, 1],
        "score": [float("nan"), float("nan")],
        "score_missing": [True, True],
    })
    targets = build_targets(rows, device="cpu")
    assert targets["outcome_valid"].tolist() == [False, True]  # help class valid regardless


# -- losses_for_hierarchical_batch + combine_hierarchical_loss -----------------

def test_combine_hierarchical_loss_matches_the_design_formula():
    l_outcome, l_a, kl = torch.tensor(2.0), torch.tensor(3.0), torch.tensor(0.5)
    total = combine_hierarchical_loss(l_outcome, l_a, kl, lambda_a=0.3, beta_kl=0.1)
    assert torch.allclose(total, torch.tensor(2.0 + 0.3 * 3.0 + 0.1 * 0.5))


@pytest.mark.parametrize("body", FA_TRAIN_STEP_BODIES)
def test_losses_for_hierarchical_batch_gradient_reaches_filter_and_outcome_head_at_step_one(body):
    """Upstream-gradient canary. Unlike the old two-head design's e2b canary
    (`test_fa_alt2.py::test_e2b_upstream_gradient_canary_reaches_the_filter`), which
    needed a full optimizer step before ANY simulator-loss gradient reached the filter's
    mu/log-sigma heads (every path there went through `h_q`, i.e. through the backbone,
    i.e. through `LoRABasis.B`, which starts at exact zero) -- `OutcomeHead` reads
    `(item_codec, u)` directly, with NO dependency on the backbone/hypernetwork/LoRA
    basis at all. So `L_outcome`'s gradient must reach the filter's belief heads at
    STEP 1, no B-zero-init detour needed. `L_production`'s gradient still needs the
    B-off-zero step, same mechanism as before -- this test checks both halves of that
    asymmetry, which is the structural payoff of the hierarchical design.
    """
    tok, model, bases, context, hyper, filt, filt_forward, _success_head = _build_k_sample_pipeline(body)
    outcome_head = OutcomeHead(codec_dim=CODEC_DIM, latent_dim=LATENT_DIM, hidden=HIDDEN_DIM)
    filter_batch = _synthetic_filter_batch(n_learners=3, t_len=5)
    learner_idx, t_idx = torch.tensor([0, 1, 2]), torch.tensor([4, 4, 4])

    examples = [
        encode_example("What is 3+5?", "8", tok, max_len=32, outcome_code=0),
        encode_example("What is 6*7?", "42", tok, max_len=32, outcome_code=1),
        encode_example("Spell cat backwards.", "tac", tok, max_len=32, outcome_code=2),
    ]
    tok_batch = collate(examples, pad_id=tok.pad_token_id)
    targets = {
        "outcome_target": torch.tensor([0, 1, 2]),
        "outcome_valid": torch.tensor([True, True, True]),
    }

    trainable = (
        list(filt.parameters()) + list(hyper.parameters()) + list(outcome_head.parameters())
        + [p for b in bases.values() for p in b.parameters()]
    )
    optimizer = torch.optim.Adam(trainable, lr=1e-2)

    def run_step():
        optimizer.zero_grad()
        l_outcome, l_a, kl, _out = losses_for_hierarchical_batch(
            filt_forward, hyper, model, context, outcome_head, filter_batch,
            learner_idx, t_idx, tok_batch, targets, beta_kl=0.001,
        )
        total = combine_hierarchical_loss(l_outcome, l_a, kl, lambda_a=0.3, beta_kl=0.001)
        assert torch.isfinite(total), f"non-finite e2b loss: L_outcome={l_outcome} L_A={l_a}"
        total.backward()
        return total

    run_step()
    outcome_nonzero = any(
        p.grad is not None and torch.isfinite(p.grad).all() and bool((p.grad != 0).any())
        for p in outcome_head.parameters()
    )
    assert outcome_nonzero, "OutcomeHead received no gradient at step 1"
    mu_grad = filt.knowledge_filter.mu_head.weight.grad
    assert mu_grad is not None and torch.isfinite(mu_grad).all() and bool((mu_grad != 0).any()), (
        "filter mu_head received no gradient from L_outcome at step 1 -- OutcomeHead "
        "does not route through the backbone, so no B-zero-init detour should apply"
    )
    # production still needs the B-off-zero step, same mechanism as the old two-head design.
    for name, b in bases.items():
        assert b.B.grad is not None and torch.isfinite(b.B.grad).all()
        assert bool((b.B.grad != 0).any()), f"basis {name}: B has zero grad at step 1"
    optimizer.step()

    run_step()
    for name, b in bases.items():
        assert b.A.grad is not None and torch.isfinite(b.A.grad).all()
        assert bool((b.A.grad != 0).any()), f"basis {name}: A still zero-grad at step 2"


# -- `likelihood` flag (single-sample vs K-sample mixture likelihood) ----------------

@pytest.mark.parametrize("body", FA_TRAIN_STEP_BODIES)
def test_losses_for_hierarchical_batch_likelihood_default_is_byte_identical_to_pre_fix(body):
    """`likelihood='single'` is the default, and every existing caller (FA-SIM3's own
    script included) never passes it -- this pins that the new code path reproduces the
    OLD inline computation (`u = reparameterize_from_log_sigma(mu, log_sigma, k=1)[:, 0,
    :]` then `outcome_loss`'s masked cross-entropy) bit-for-bit, not merely "close"."""
    from atom.utils.seeds import seed_everything

    tok, model, bases, context, hyper, filt, filt_forward, _success_head = _build_k_sample_pipeline(body)
    outcome_head = OutcomeHead(codec_dim=CODEC_DIM, latent_dim=LATENT_DIM, hidden=HIDDEN_DIM)
    filter_batch = _synthetic_filter_batch(n_learners=3, t_len=5)
    learner_idx, t_idx = torch.tensor([0, 1, 2]), torch.tensor([4, 4, 4])
    examples = [
        encode_example("What is 3+5?", "8", tok, max_len=32, outcome_code=0),
        encode_example("What is 6*7?", "42", tok, max_len=32, outcome_code=1),
        encode_example("Spell cat backwards.", "tac", tok, max_len=32, outcome_code=2),
    ]
    tok_batch = collate(examples, pad_id=tok.pad_token_id)
    targets = {"outcome_target": torch.tensor([0, 1, 2]), "outcome_valid": torch.tensor([True, True, True])}

    seed_everything(0)
    l_outcome_new, l_a_new, kl_new, _out_new = losses_for_hierarchical_batch(
        filt_forward, hyper, model, context, outcome_head, filter_batch,
        learner_idx, t_idx, tok_batch, targets, beta_kl=0.001)  # likelihood omitted -> default

    # -- the OLD inline computation, reproduced verbatim (single-sample `losses_for_hierarchical_batch` body) --
    from atom.simulator.fa_alt2 import outcome_loss as _outcome_loss_ref
    from atom.simulator.fa_train_step import gather_belief_for_batch, gather_item_codec_for_batch

    seed_everything(0)
    mu_sel, log_sigma_sel = gather_belief_for_batch(filt_forward, filter_batch, learner_idx, t_idx)
    item_codec_sel = gather_item_codec_for_batch(filter_batch, learner_idx, t_idx)
    u_ref = reparameterize_from_log_sigma(mu_sel, log_sigma_sel, k=1)[:, 0, :]
    out_ref = forward_personalised(model, hyper, context, u_ref, tok_batch["input_ids"],
                                   tok_batch["attention_mask"], tok_batch["prompt_end_index"])
    outcome_logits_ref = outcome_head(item_codec_sel, u_ref)
    l_outcome_ref = _outcome_loss_ref(outcome_logits_ref, targets["outcome_target"], targets["outcome_valid"])

    assert torch.equal(l_outcome_new, l_outcome_ref), "default likelihood must be bit-identical to the single-sample reference"


@pytest.mark.parametrize("body", FA_TRAIN_STEP_BODIES)
def test_losses_for_hierarchical_batch_kmixture_runs_finite_and_differs_from_single(body):
    """Sanity, not a numeric pin (the kmixture branch draws K fresh samples so exact
    equality with anything is not the right check): the new `likelihood='kmixture'`
    branch produces a finite loss and, on a filter whose sigma is nontrivial (`_build_
    filter`'s init), differs from the single-sample loss under the SAME RNG state --
    proof the flag actually changes which computation runs, not a silent no-op."""
    from atom.utils.seeds import seed_everything

    tok, model, bases, context, hyper, filt, filt_forward, _success_head = _build_k_sample_pipeline(body)
    outcome_head = OutcomeHead(codec_dim=CODEC_DIM, latent_dim=LATENT_DIM, hidden=HIDDEN_DIM)
    filter_batch = _synthetic_filter_batch(n_learners=3, t_len=5)
    learner_idx, t_idx = torch.tensor([0, 1, 2]), torch.tensor([4, 4, 4])
    examples = [
        encode_example("What is 3+5?", "8", tok, max_len=32, outcome_code=0),
        encode_example("What is 6*7?", "42", tok, max_len=32, outcome_code=1),
        encode_example("Spell cat backwards.", "tac", tok, max_len=32, outcome_code=2),
    ]
    tok_batch = collate(examples, pad_id=tok.pad_token_id)
    targets = {"outcome_target": torch.tensor([0, 1, 2]), "outcome_valid": torch.tensor([True, True, True])}

    seed_everything(0)
    l_outcome_single, _l_a, _kl, _out = losses_for_hierarchical_batch(
        filt_forward, hyper, model, context, outcome_head, filter_batch,
        learner_idx, t_idx, tok_batch, targets, beta_kl=0.001, likelihood="single")

    seed_everything(0)
    l_outcome_kmix, _l_a2, _kl2, _out2 = losses_for_hierarchical_batch(
        filt_forward, hyper, model, context, outcome_head, filter_batch,
        learner_idx, t_idx, tok_batch, targets, beta_kl=0.001, k=8, likelihood="kmixture")

    assert torch.isfinite(l_outcome_kmix)
    assert not torch.equal(l_outcome_single, l_outcome_kmix), (
        "kmixture (K=8) must not silently reduce to the single-sample computation")


def test_outcome_kmixture_nll_per_row_matches_brute_force_probability_average():
    """Pure-function check on `outcome_kmixture_nll_per_row` (no model needed): builds a
    tiny linear `outcome_head` stand-in and K hand-chosen `u` vectors, computes the
    K-mixture NLL by hand (average softmax probabilities across K, then -log of the true
    class), and asserts the function matches to float precision -- proves the reduction
    is a PROBABILITY-space average, not a log-space (logsumexp) one, which is the exact
    distinction the sigma-fix's own spec draws against `joint_kc_mixture_nll`."""
    from atom.simulator.fa_alt2 import outcome_kmixture_nll_per_row

    torch.manual_seed(3)
    n, k_eff, codec_dim, latent_dim, n_classes = 4, 5, CODEC_DIM, LATENT_DIM, 3
    outcome_head = torch.nn.Linear(codec_dim + latent_dim, n_classes)
    item_codec = torch.randn(n, codec_dim)
    u_k = torch.randn(n, k_eff, latent_dim)
    target = torch.tensor([0, 1, 2, 0])

    def _head(ic, u):
        return outcome_head(torch.cat([ic, u], dim=-1))

    got = outcome_kmixture_nll_per_row(_head, item_codec, u_k, target)

    probs_sum = torch.zeros(n, n_classes)
    for j in range(k_eff):
        probs_sum = probs_sum + torch.softmax(_head(item_codec, u_k[:, j, :]), dim=-1)
    probs_mean = probs_sum / k_eff
    expected = -torch.log(probs_mean.gather(1, target.unsqueeze(1)).squeeze(1).clamp(min=1e-8))

    assert torch.allclose(got, expected, atol=1e-6)


# -- k_sample_success_predictions -------------------------------------

@pytest.mark.parametrize("body", FA_TRAIN_STEP_BODIES)
def test_k_sample_success_predictions_shape_and_range(body):
    from atom.simulator.fa_alt2 import collate

    tok, model, bases, context, hyper, filt, filt_forward, success_head = \
        _build_k_sample_pipeline(body)
    filter_batch = _synthetic_filter_batch(n_learners=4, t_len=5)
    examples = [encode_example(q, a, tok, max_len=16)
               for q, a in [("What is 2+2?", "4"), ("Name a planet.", "Mars"),
                           ("Spell dog backwards.", "god")]]
    tok_batch = collate(examples, pad_id=tok.pad_token_id, device="cpu")
    learner_idx = torch.tensor([0, 1, 2])
    t_idx = torch.tensor([1, 2, 3])
    support_target = torch.tensor([0, 1, 2])
    k = 6

    with torch.no_grad():
        probs = k_sample_success_predictions(
            filt_forward, hyper, model, context, success_head, filter_batch,
            learner_idx, t_idx, tok_batch, support_target, k=k)

    assert probs.shape == (k, 3)
    assert torch.isfinite(probs).all()
    assert (probs >= 0).all() and (probs <= 1).all()


@pytest.mark.parametrize("body", FA_TRAIN_STEP_BODIES)
def test_k_sample_success_predictions_vary_across_k_when_belief_has_nonzero_spread(body):
    """If the belief's own log_sigma is non-trivial, different K draws of u must actually
    produce different predictions -- otherwise K-sampling is decorative and spread-error
    could never be anything but degenerate downstream."""
    from atom.simulator.fa_alt2 import collate

    tok, model, bases, context, hyper, filt, filt_forward, success_head = \
        _build_k_sample_pipeline(body)
    # LoRABasis.B is zero-init by construction ("B zero-init = exact
    # no-op") -- at a fresh init, EVERY basis.apply(x, c) is exactly 0 regardless of c, so
    # different u samples produce byte-identical predictions, not merely similar ones.
    # That is correct behaviour for an untrained model, but it means this specific test
    # (spread requires personalisation to have an effect) needs a basis that has "learned
    # something", same as it would be mid-training -- perturb B away from zero directly.
    gen = torch.Generator().manual_seed(0)
    for b in bases.values():
        b.B.data = torch.randn(b.B.shape, generator=gen) * 0.1
    filter_batch = _synthetic_filter_batch(n_learners=3, t_len=4)
    examples = [encode_example("What is 2+2?", "4", tok, max_len=16)]
    tok_batch = collate(examples, pad_id=tok.pad_token_id, device="cpu")
    learner_idx = torch.tensor([0])
    t_idx = torch.tensor([1])
    support_target = torch.tensor([0])

    with torch.no_grad():
        probs = k_sample_success_predictions(
            filt_forward, hyper, model, context, success_head, filter_batch,
            learner_idx, t_idx, tok_batch, support_target, k=8,
            generator=torch.Generator().manual_seed(0))

    assert probs.shape == (8, 1)
    assert probs.std().item() > 1e-6  # NOT all 8 draws collapsed to the same prediction


@pytest.mark.parametrize("body", FA_TRAIN_STEP_BODIES)
def test_k_sample_success_predictions_feed_spread_error_correlation_end_to_end(body):
    """The actual gate (iv) pipeline, wired together: k_sample_success_predictions'
    output feeds directly into spread_error_correlation with no reshaping needed."""
    from atom.simulator.fa_alt2 import collate

    tok, model, bases, context, hyper, filt, filt_forward, success_head = \
        _build_k_sample_pipeline(body)
    filter_batch = _synthetic_filter_batch(n_learners=5, t_len=5)
    questions = ["What is 2+2?", "Name a planet.", "Spell dog backwards.",
                "What colour is the sky?", "Solve for x: x=5."]
    examples = [encode_example(q, "answer", tok, max_len=16) for q in questions]
    tok_batch = collate(examples, pad_id=tok.pad_token_id, device="cpu")
    learner_idx = torch.tensor([0, 1, 2, 3, 4])
    t_idx = torch.tensor([1, 2, 3, 4, 0])
    support_target = torch.tensor([0, 1, 2, 3, 0])
    target = torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0])

    with torch.no_grad():
        probs = k_sample_success_predictions(
            filt_forward, hyper, model, context, success_head, filter_batch,
            learner_idx, t_idx, tok_batch, support_target, k=8)
        out = spread_error_correlation(probs, target)

    assert out["n"] == 5
    assert "spread_error_corr" in out and "k_spread_non_degenerate" in out
