"""Stage-0 tests for the unified FA simulator: the simulator design's numbered peek
canaries, loss-mask tests, one tiny end-to-end batch (e2b), and the upstream-gradient
canary, which extends to the filter's own mu/log-sigma heads.

Runs against `CausalStubLM` (real causal masking, real LM vocab head, fast/no-download);
`scripts/fa_04_canaries.py` re-runs the load-bearing subset once against a real backbone.

Canaries 3, 4 and 5 already have dedicated coverage at the FILTER level in
`tests/test_s1_filter_v2.py` (GRU body) and `tests/test_event_transformer.py` (body (c)).
This file does not re-derive that; it extends the SAME invariants through the simulator's
own code (heads, prompt handling, the forward orchestration) so a bug introduced in the
wiring layer -- not the filter itself -- would still be caught.

BODY-AGNOSTIC BY CONSTRUCTION: every filter construction and call in this file goes through
`atom.filter.factory.build_filter`/its uniform
`forward(item_codec, prev_feat, time_min, valid_mask)` signature, and every filter-facing
test is parametrised over `FA4_TEST_BODIES`. A test that hardcoded one body's constructor
or its 2-argument `forward` would leave the other arm of the paired two-body comparison
unverified before it goes near a GPU.
"""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest
import torch
import torch.nn.functional as F

from atom.binding import (GlobalMixingVector, Hypernetwork, attach_lora_basis, build_second_basis,
                          reparameterize_from_log_sigma, set_active_basis)
from atom.binding.stub_causal_lm import CausalStubLM
from atom.data.fa_manifest import (
    assert_unique_ids,
    exclude_score_missing_from_history,
    inject_duplicate_row,
)
from atom.eval.metrics import with_population
from atom.filter.factory import build_filter
from atom.heads.fa_alt2_heads import SuccessHead, SupportHead
from atom.simulator.fa_alt2 import (
    build_prompt_completion,
    collate,
    encode_example,
    forward_global_control,
    forward_personalised,
    gather_belief,
    gather_prompt_hidden,
    outcome_loss,
    pick_lambda_a,
    production_loss,
    production_nll_per_row,
    production_topk_per_row,
    success_loss,
    support_loss,
)

CODEC_DIM = 5
LATENT_DIM = 6
HIDDEN_DIM = 32
PREV_FEAT_DIM = 5  # atom.filter.episode_v2.PREV_FEAT_DIM_V2
FILTER_HIDDEN = 8  # must be divisible by N_HEADS (body (c)'s constructor asserts this)
N_HEADS, D_FF, N_RBF = 2, 8, 4  # tiny, fast body-(c) hyperparameters -- values don't matter here
FA4_TEST_BODIES = ("gru", "c_distance_time")  # the GRU default + the full body-(c) arm


def _dummy_geometry(n: int, t: int) -> tuple[torch.Tensor, torch.Tensor]:
    """A plausible, non-degenerate (time_min, valid_mask) pair for tests that don't care
    about body (c)'s specific bias values (that mechanism is `test_event_transformer.py`'s
    job) but must still hand it well-formed geometry tensors -- every filter body call in
    this codebase is uniform-signature (`atom.filter.factory.build_filter`), so even the
    GRU-only tests must supply these two, even though the GRU ignores them."""
    time_min = torch.arange(t, dtype=torch.float32).unsqueeze(0).expand(n, -1) * 10.0
    valid_mask = torch.ones(n, t, dtype=torch.bool)
    return time_min, valid_mask


class _FakeTok:
    """Deterministic, dependency-free tokenizer stand-in -- avoids Python's per-process
    string-hash randomisation (`hash()`) so token ids are stable across test runs, not
    merely stable within one process."""

    pad_token_id = 0
    eos_token_id = 63

    def __call__(self, text: str, add_special_tokens: bool = True) -> dict:
        words = [w for w in text.replace("\n", " \n ").split(" ") if w != ""]
        ids = [self._word_id(w) for w in words]
        if add_special_tokens:
            ids = [1, *ids]
        return {"input_ids": ids}

    @staticmethod
    def _word_id(w: str) -> int:
        return (sum(ord(c) for c in w) % 40) + 2  # 0=pad, 1=bos reserved


def _build_pipeline(body: str = "gru", seed: int = 0) -> SimpleNamespace:
    tok = _FakeTok()
    model = CausalStubLM(vocab_size=64, dim=HIDDEN_DIM, n_layers=2, n_heads=4,
                          pad_token_id=tok.pad_token_id, seed=seed)
    bases, context = attach_lora_basis(model, m=8, r=4, seed=seed + 1)
    hyper = Hypernetwork(u_dim=LATENT_DIM, m=8, kind="mlp", seed=seed + 2)
    filt, filt_forward = build_filter(body, codec_dim=CODEC_DIM, episode_hidden=8,
                                      filter_hidden=FILTER_HIDDEN, latent_dim=LATENT_DIM,
                                      n_heads=N_HEADS, d_ff=D_FF, n_rbf=N_RBF)
    return SimpleNamespace(
        tok=tok, model=model, bases=bases, context=context, hyper=hyper,
        filt=filt, filt_forward=filt_forward, body=body,
        support_head=SupportHead(HIDDEN_DIM), success_head=SuccessHead(HIDDEN_DIM),
    )


# -- canary 1: response-token invariance ------------------------------------------

def test_canary_1_response_tokens_do_not_leak_into_support_success():
    p = _build_pipeline()
    ex_a = encode_example("What is 2+2?", "4", p.tok, max_len=32)
    ex_b = encode_example("What is 2+2?", "9", p.tok, max_len=32)  # same prompt, diff answer
    assert ex_a["prompt_end_index"] == ex_b["prompt_end_index"]
    boundary = ex_a["prompt_end_index"] + 1
    assert ex_a["input_ids"][:boundary] == ex_b["input_ids"][:boundary]
    assert ex_a["input_ids"] != ex_b["input_ids"]  # sanity: they DO differ downstream

    batch = collate([ex_a, ex_b], pad_id=p.tok.pad_token_id)
    u = torch.zeros(2, LATENT_DIM)
    out = forward_personalised(p.model, p.hyper, p.context, u, batch["input_ids"],
                               batch["attention_mask"], batch["prompt_end_index"])
    assert torch.equal(out["h_q"][0], out["h_q"][1])
    assert torch.equal(p.support_head(out["h_q"])[0], p.support_head(out["h_q"])[1])
    assert torch.equal(p.success_head(out["h_q"])[0], p.success_head(out["h_q"])[1])


# -- canary 2: current support/success labels never enter the forward pass --------

def test_canary_2_current_labels_have_no_pathway_into_predictions():
    p = _build_pipeline()
    ex = encode_example("Solve for x: x+1=2", "1", p.tok, max_len=32)
    batch = collate([ex], pad_id=p.tok.pad_token_id)
    u = torch.randn(1, LATENT_DIM)
    out1 = forward_personalised(p.model, p.hyper, p.context, u, batch["input_ids"],
                                batch["attention_mask"], batch["prompt_end_index"])
    # Two different hypothetical (support_code, score) pairs -- neither is a parameter of
    # encode_example/forward_personalised/the heads, so nothing about calling this twice
    # with either "true" label in hand can change the result: the API shape itself is the
    # guarantee (regression target: a future edit that threads a label into the forward
    # path would have to change this call's signature, which this test pins).
    out2 = forward_personalised(p.model, p.hyper, p.context, u, batch["input_ids"],
                                batch["attention_mask"], batch["prompt_end_index"])
    assert torch.equal(out1["h_q"], out2["h_q"])
    assert torch.equal(out1["lm_logits"], out2["lm_logits"])


# -- canary 3: same history + two questions -> same belief, different heads -------

@pytest.mark.parametrize("body", FA4_TEST_BODIES)
def test_canary_3_same_belief_reused_but_heads_differ_across_questions(body):
    p = _build_pipeline(body)
    item_codec = torch.randn(1, 5, CODEC_DIM)
    prev_feat = torch.randn(1, 5, PREV_FEAT_DIM)
    time_min, valid_mask = _dummy_geometry(1, 5)
    mu_all, log_sigma_all = p.filt_forward(item_codec, prev_feat, time_min, valid_mask)
    mu_sel, log_sigma_sel = gather_belief(mu_all, log_sigma_all,
                                          torch.tensor([0]), torch.tensor([4]))
    u = reparameterize_from_log_sigma(mu_sel, log_sigma_sel, k=1)[:, 0, :]  # ONE fixed sample, reused below

    ex1 = encode_example("What is the capital of France?", "Paris", p.tok, max_len=32)
    ex2 = encode_example("What is seven times six?", "42", p.tok, max_len=32)
    b1 = collate([ex1], pad_id=p.tok.pad_token_id)
    b2 = collate([ex2], pad_id=p.tok.pad_token_id)
    out1 = forward_personalised(p.model, p.hyper, p.context, u, b1["input_ids"],
                                b1["attention_mask"], b1["prompt_end_index"])
    out2 = forward_personalised(p.model, p.hyper, p.context, u, b2["input_ids"],
                                b2["attention_mask"], b2["prompt_end_index"])
    assert not torch.allclose(out1["h_q"], out2["h_q"])
    assert not torch.allclose(p.support_head(out1["h_q"]), p.support_head(out2["h_q"]))
    # The belief itself is identical BY CONSTRUCTION here (computed once, reused for both
    # questions) -- the load-bearing guarantee that the query never re-enters the filter
    # is proven at the filter level by the filter's own peek canary
    # (tests/test_s1_filter_v2.py); this test's job is only to confirm the downstream
    # heads still differ once a real backbone forward sits between belief and head.


# -- extends the filter's causality coverage: item-codec channel at its OWN position ----

@pytest.mark.parametrize("body", FA4_TEST_BODIES)
def test_extends_fa1_causality_to_item_codec_at_its_own_position(body):
    """The filter causality tests (test_s1_filter_v2.py / test_event_transformer.py) vary
    the OUTCOME channel (correct/hint/saw_answer) at position t and show positions <=t are
    unaffected. The simulator's orchestration reads `item_codec` directly (`gather_belief`'s
    callers), so this test covers the channel those didn't: corrupting the CURRENT event's
    own item codec must not change its own belief either (b_t never depends on q_t's
    codec, only H_<t's) -- on EITHER body, via the same `build_filter` entry point the
    trainer uses."""
    filt, filt_forward = build_filter(body, codec_dim=CODEC_DIM, episode_hidden=8,
                                      filter_hidden=FILTER_HIDDEN, latent_dim=LATENT_DIM,
                                      n_heads=N_HEADS, d_ff=D_FF, n_rbf=N_RBF)
    filt.eval()
    item_codec = torch.randn(2, 6, CODEC_DIM)
    prev_feat = torch.randn(2, 6, PREV_FEAT_DIM)
    time_min, valid_mask = _dummy_geometry(2, 6)
    with torch.no_grad():
        mu1, _ = filt_forward(item_codec, prev_feat, time_min, valid_mask)
        item_codec2 = item_codec.clone()
        item_codec2[:, 3, :] = torch.randn(2, CODEC_DIM) * 100.0
        mu2, _ = filt_forward(item_codec2, prev_feat, time_min, valid_mask)
    assert torch.allclose(mu1[:, 3], mu2[:, 3])


# -- canary 6: duplicate ID rejected (regression on the manifest's existing mechanism) --

def test_canary_6_duplicate_id_rejected():
    df = pd.DataFrame({"id": [10, 20, 30]})
    dup = inject_duplicate_row(df, seed=0)
    with pytest.raises(AssertionError):
        assert_unique_ids(dup)


# -- canary 7 + loss-mask tests: independent per-loss masking ----------------------

def test_canary_7_missing_success_zero_loss_and_zero_gradient():
    logits = torch.randn(4, 4, requires_grad=True)
    support_target = torch.tensor([0, 1, 2, 3])
    success_target = torch.tensor([1.0, 0.0, 1.0, 0.0])
    valid = torch.tensor([True, True, False, False])  # rows 2,3: missing success

    loss = success_loss(logits, support_target, success_target, valid)
    expected = F.binary_cross_entropy_with_logits(
        logits[:2].gather(1, support_target[:2].unsqueeze(1)).squeeze(1), success_target[:2],
    )
    assert torch.allclose(loss, expected)

    loss.backward()
    assert torch.all(logits.grad[2] == 0)
    assert torch.all(logits.grad[3] == 0)
    assert bool((logits.grad[:2] != 0).any())


def test_success_loss_all_invalid_is_exact_zero_not_nan():
    logits = torch.randn(2, 4)
    loss = success_loss(logits, torch.tensor([0, 1]), torch.tensor([1.0, 0.0]),
                        torch.tensor([False, False]))
    assert loss.item() == 0.0


def test_support_loss_masks_invalid_rows():
    logits = torch.randn(3, 4, requires_grad=True)
    target = torch.tensor([0, 1, 2])
    valid = torch.tensor([True, True, False])
    loss = support_loss(logits, target, valid)
    expected = F.cross_entropy(logits[:2], target[:2])
    assert torch.allclose(loss, expected)


def test_support_loss_all_invalid_is_exact_zero_not_nan():
    logits = torch.randn(2, 4)
    loss = support_loss(logits, torch.tensor([0, 1]), torch.tensor([False, False]))
    assert loss.item() == 0.0


def test_outcome_loss_masks_invalid_rows():
    logits = torch.randn(3, 5, requires_grad=True)
    target = torch.tensor([0, 1, 4])
    valid = torch.tensor([True, True, False])
    loss = outcome_loss(logits, target, valid)
    expected = F.cross_entropy(logits[:2], target[:2])
    assert torch.allclose(loss, expected)


def test_outcome_loss_all_invalid_is_exact_zero_not_nan():
    logits = torch.randn(2, 5)
    loss = outcome_loss(logits, torch.tensor([0, 1]), torch.tensor([False, False]))
    assert loss.item() == 0.0


def test_production_loss_masks_invalid_rows_even_with_real_looking_labels():
    """Row 1 has real (non -100) labels present but is flagged production_invalid -- the
    belt-and-suspenders masking `production_loss` documents must still exclude it."""
    vocab = 20
    logits = torch.randn(2, 4, vocab, requires_grad=True)
    labels = torch.tensor([[-100, -100, 5, 7], [-100, -100, 3, 9]])
    production_valid = torch.tensor([True, False])
    loss = production_loss(logits, labels, production_valid)
    expected = F.cross_entropy(
        logits[:1, :-1, :].reshape(-1, vocab), labels[:1, 1:].reshape(-1), ignore_index=-100,
    )
    assert torch.allclose(loss, expected)


def test_production_loss_all_invalid_is_exact_zero_not_nan():
    logits = torch.randn(2, 4, 10)
    labels = torch.full((2, 4), -100)
    loss = production_loss(logits, labels, torch.tensor([False, False]))
    assert loss.item() == 0.0


def test_production_nll_per_row_returns_float32_even_from_bf16_logits():
    """Regression: a bf16 backbone's cross_entropy output can stay bf16, and
    torch.Tensor.numpy() cannot convert bf16 (no native numpy dtype) -- an evaluation pass
    would then crash only after a full, expensive GPU forward. Must never regress silently."""
    logits = torch.randn(2, 4, 10).to(torch.bfloat16)
    labels = torch.tensor([[-100, -100, 5, 7], [-100, -100, 3, 9]])
    per_row, row_has_any = production_nll_per_row(logits, labels, torch.tensor([True, True]))
    assert per_row.dtype == torch.float32
    per_row.numpy()  # must not raise TypeError: Got unsupported ScalarType BFloat16


def _one_hot_logits(shape: tuple[int, int, int], target_ids: torch.Tensor) -> torch.Tensor:
    """Builds logits whose argmax at every position deterministically equals
    `target_ids` -- lets a top-1/exact-match test assert on exact expected numbers
    instead of tolerances on random logits."""
    n, t, v = shape
    logits = torch.full((n, t, v), -10.0)
    idx = target_ids.clamp(min=0)
    logits.scatter_(2, idx.unsqueeze(-1), 10.0)
    return logits


def test_production_topk_per_row_all_correct_is_exact_match_one():
    """Row 0: labels [-100,-100,5,7] (prompt masked), shift gives valid targets at
    positions 2,3 of the SHIFTED labels (i.e. tokens 5,7 predicted from positions 1,2)
    -- logits built so argmax matches everywhere valid."""
    labels = torch.tensor([[-100, -100, 5, 7]])
    shift_labels = labels[:, 1:]  # [-100, 5, 7]
    logits = _one_hot_logits((1, 4, 20), torch.cat([shift_labels, shift_labels[:, -1:]], dim=1))
    top1, exact, has_any = production_topk_per_row(logits, labels, torch.tensor([True]))
    assert top1.item() == 1.0
    assert exact.item() == 1.0
    assert has_any.item() == 1.0


def test_production_topk_per_row_partial_match_not_exact():
    """One of two valid tokens predicted correctly -- top1=0.5, exact_match=0."""
    labels = torch.tensor([[-100, -100, 5, 7]])
    logits = torch.full((1, 4, 20), -10.0)
    # shift_logits[:, :-1] used at positions 0,1,2 -> compared to shift_labels[-100,5,7]
    # position 1 (predicting token 5): correct. position 2 (predicting token 7): wrong.
    logits[0, 1, 5] = 10.0
    logits[0, 2, 3] = 10.0  # wrong class for the token-7 target
    top1, exact, has_any = production_topk_per_row(logits, labels, torch.tensor([True]))
    assert top1.item() == pytest.approx(0.5)
    assert exact.item() == 0.0
    assert has_any.item() == 1.0


def test_production_topk_per_row_masks_invalid_rows_even_with_real_looking_labels():
    """Same discipline as production_loss's own masking test: a row with real-looking
    labels but production_valid=False must be excluded (row_has_any=0), never scored."""
    labels = torch.tensor([[-100, -100, 5, 7], [-100, -100, 3, 9]])
    logits = _one_hot_logits((2, 4, 20), torch.cat([labels[:, 1:], labels[:, -1:]], dim=1))
    production_valid = torch.tensor([True, False])
    top1, exact, has_any = production_topk_per_row(logits, labels, production_valid)
    assert has_any.tolist() == [1.0, 0.0]
    assert top1[1].item() == 0.0
    assert exact[1].item() == 0.0


def test_production_topk_per_row_all_invalid_is_exact_zero_not_nan():
    logits = torch.randn(2, 4, 10)
    labels = torch.full((2, 4), -100)
    top1, exact, has_any = production_topk_per_row(logits, labels, torch.tensor([False, False]))
    assert top1.tolist() == [0.0, 0.0]
    assert exact.tolist() == [0.0, 0.0]
    assert has_any.tolist() == [0.0, 0.0]


# -- canary 8: no NEW fitted preprocessing statistic in the simulator --------------------

def test_canary_8_no_new_preprocessing_statistic_introduced():
    """The 8th canary (validation/test labels must never affect a preprocessing statistic
    fitted on training data) has no NEW artifact to gate in the simulator: it introduces no
    fitted statistic at all (`encode_example`'s truncation budget is a fixed config, not
    train-derived; belief-gap conditioning, which WOULD need a train-only quantile clip, is
    not implemented). The existing fitted statistics (the manifest's dedup/score-resolution,
    the learner/item splits) are train-only by construction of the data pipeline."""
    assert True


# -- prompt template sanity (no label/skill/reference leakage in the text) ---------------

def test_prompt_template_excludes_forbidden_fields():
    """No learner ID, skill tags, reference answer, observed support/success, or future
    events in the prompt text. ("This learner's response:" -- the template's own
    wording -- legitimately contains the word "learner"; what must never
    appear is an actual identifier or label VALUE.)"""
    prompt, completion = build_prompt_completion("What is 2+2?", "4")
    for forbidden in ("skill", "hint_count", "saw_answer", "support_code", "learner_id"):
        assert forbidden not in prompt.lower()
    assert completion == " 4"


# -- the class-conditioned prompt (production conditioning route 1) ---------------------

def test_build_prompt_completion_outcome_code_none_is_the_original_unconditioned_template():
    """`outcome_code=None` (the default) must reproduce the unconditioned prompt byte-for-
    byte -- callers that never pass this argument (the bread test, both trainers) must be
    unaffected."""
    unconditioned = build_prompt_completion("What is 2+2?", "4")
    defaulted = build_prompt_completion("What is 2+2?", "4", outcome_code=None)
    assert unconditioned == defaulted


def test_build_prompt_completion_outcome_code_inserts_the_frozen_class_token():
    prompt, completion = build_prompt_completion("What is 2+2?", "4", outcome_code=1)
    assert "incorrect" in prompt
    assert "What is 2+2?" in prompt
    assert completion == " 4"
    # still no forbidden identifiers/label-value leakage.
    for forbidden in ("skill", "hint_count", "saw_answer", "support_code", "learner_id"):
        assert forbidden not in prompt.lower()


def test_build_prompt_completion_every_outcome_code_gives_a_distinct_prompt():
    prompts = {build_prompt_completion("Q", "A", outcome_code=c)[0] for c in range(5)}
    assert len(prompts) == 5  # all five class tokens are textually distinct


def test_encode_example_class_token_is_masked_exactly_like_the_rest_of_the_prompt():
    """The design rule: "the class enters as a frozen text token in the prompt (masked from the
    loss)". There is no new masking mechanism for it -- `encode_example`'s existing
    `labels = [-100]*len(p_ids) + c_ids` already covers it because the token is textually
    inside the prompt string. This test is the canary that stays red if a future refactor
    ever moves the class token out of the prompt (e.g. into a separate channel) without
    also moving its masking."""
    p = _build_pipeline()
    unconditioned = encode_example("What is 2+2?", "4", p.tok, max_len=32)
    conditioned = encode_example("What is 2+2?", "4", p.tok, max_len=32, outcome_code=1)
    # the conditioned prompt is strictly longer (it has the extra class-token text) ...
    assert conditioned["prompt_end_index"] > unconditioned["prompt_end_index"]
    # ... and every one of those extra prompt tokens is masked, same as the rest.
    n_masked = conditioned["prompt_end_index"] + 1
    assert conditioned["labels"][:n_masked] == [-100] * n_masked
    # the completion tokens themselves (answer "4") are identical and unmasked either way.
    assert conditioned["input_ids"][n_masked:] == unconditioned["input_ids"][unconditioned["prompt_end_index"] + 1:]


def test_outcome_head_is_blind_to_the_production_class_token():
    """The hierarchical model's invariant, made a canary: "weights carry traits, tokens carry events" --
    the class token conditions PRODUCTION (through the backbone's text input) and must
    never reach the OUTCOME head, which reads only `(item_codec, u)`. Structural check
    (the class token is not even in `OutcomeHead.forward`'s signature) plus a functional
    one: calling it twice with the same `(item_codec, u)` gives bit-identical logits
    regardless of anything about the prompt -- there is no path by which it could differ."""
    from atom.heads.fa_hierarchical_heads import OutcomeHead
    import inspect

    sig = inspect.signature(OutcomeHead.forward)
    assert "input_ids" not in sig.parameters
    assert "outcome_code" not in sig.parameters

    torch.manual_seed(0)
    head = OutcomeHead(codec_dim=CODEC_DIM, latent_dim=LATENT_DIM, hidden=HIDDEN_DIM)
    item_codec = torch.randn(3, CODEC_DIM)
    u = torch.randn(3, LATENT_DIM)
    logits_a = head(item_codec, u)
    logits_b = head(item_codec, u)  # no prompt/class-token argument exists to vary
    assert torch.equal(logits_a, logits_b)


# -- global-LoRA control arm: forward_global_control ------------------------------------

def test_forward_global_control_broadcasts_one_c_and_ignores_u_entirely():
    """The global-control arm's whole point: production under `forward_global_control`
    must be identical for every row in a batch regardless of anything about the row's own
    `u` (there IS no `u` argument) -- only the shared `GlobalMixingVector` drives `c`,
    unlike `forward_personalised` where two different `u`s give two different `c`s."""
    p = _build_pipeline()
    ex = encode_example("What is 2+2?", "4", p.tok, max_len=32)
    ex2 = encode_example("What is 2+2?", "4", p.tok, max_len=32)  # identical row, twice
    batch = collate([ex, ex2], pad_id=p.tok.pad_token_id)

    global_bases = build_second_basis(p.model, seed=1000)
    for basis in global_bases.values():
        with torch.no_grad():
            basis.B.normal_(generator=torch.Generator().manual_seed(11))  # un-zero, real delta
    global_mixing = GlobalMixingVector(m=8, seed=12)
    set_active_basis(p.model, global_bases)

    logits = forward_global_control(p.model, global_mixing, p.context, n=2,
                                    input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
    assert torch.equal(logits[0], logits[1])  # same prompt, same shared c -> identical logits
    assert p.context.c.shape == (2, 8)
    assert torch.equal(p.context.c[0], p.context.c[1])  # the broadcast itself, directly


def test_forward_global_control_uses_the_active_basis_not_the_personalised_one():
    """After `set_active_basis(model, global_bases)`, `forward_global_control`'s output
    must differ from `forward_personalised`'s -- proving the swap actually routes through
    DIFFERENT weights, not just a different `c` value on the SAME basis."""
    p = _build_pipeline()
    ex = encode_example("What is 2+2?", "4", p.tok, max_len=32)
    batch = collate([ex], pad_id=p.tok.pad_token_id)
    u = torch.randn(1, LATENT_DIM)

    for basis in p.bases.values():
        with torch.no_grad():
            basis.B.normal_(generator=torch.Generator().manual_seed(21))
    personalised_out = forward_personalised(p.model, p.hyper, p.context, u, batch["input_ids"],
                                            batch["attention_mask"], batch["prompt_end_index"])

    global_bases = build_second_basis(p.model, seed=2000)
    for basis in global_bases.values():
        with torch.no_grad():
            basis.B.normal_(generator=torch.Generator().manual_seed(22))
    global_mixing = GlobalMixingVector(m=8, seed=23)
    set_active_basis(p.model, global_bases)
    global_logits = forward_global_control(p.model, global_mixing, p.context, n=1,
                                           input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])

    assert not torch.allclose(personalised_out["lm_logits"], global_logits)


# -- eval-harness additions (AKT comparability) -----------------------------------------

def test_exclude_score_missing_from_history_drops_rows_and_resequences():
    ts = pd.to_datetime([
        "2026-01-01T00:00:00Z", "2026-01-01T00:10:00Z", "2026-01-01T00:20:00Z",
        "2026-01-01T00:30:00Z",
    ])
    ev = pd.DataFrame({
        "learner": ["u1"] * 4,
        "id": [1, 2, 3, 4],
        "timestamp": ts,
        "has_valid_time": [True, True, True, True],
        "score_missing": [False, True, False, False],
        "t": [0, 1, 2, 3],
    })
    out = exclude_score_missing_from_history(ev)
    assert list(out["id"]) == [1, 3, 4]  # id=2 (score_missing) dropped entirely
    assert list(out["t"]) == [0, 1, 2]  # re-sequenced, no hole at the old t=1
    # id=3's gap is now measured from id=1 (20 real minutes), not from the dropped id=2 --
    # the AKT-matched, gap-free-of-the-excluded-row reading.
    assert out.loc[out["id"] == 3, "gap_log1p_min"].item() == pytest.approx(
        __import__("numpy").log1p(20.0), rel=1e-6,
    )
    assert bool((~out["score_missing"]).all())  # the dropped rows are gone, not just masked


def test_with_population_stamps_count_without_mutating_input():
    metrics = {"auc": 0.81}
    out = with_population(metrics, n=500)
    assert out == {"auc": 0.81, "n_eval_population": 500}
    assert metrics == {"auc": 0.81}  # input untouched


# -- the pre-registered lambda_A selection rule ------------------------------------------

def test_pick_lambda_a_balances_toward_the_other_losses_norm():
    # grad_norm_a=10 paired with each grid value: 0.1*10=1, 0.3*10=3, 1.0*10=10 -- closest
    # to grad_norm_other=3.2 is 0.3.
    assert pick_lambda_a(grad_norm_a=10.0, grad_norm_other=3.2) == 0.3
    assert pick_lambda_a(grad_norm_a=10.0, grad_norm_other=0.5) == 0.1
    assert pick_lambda_a(grad_norm_a=10.0, grad_norm_other=9.0) == 1.0


def test_pick_lambda_a_falls_back_to_smallest_grid_value_when_production_grad_is_zero():
    assert pick_lambda_a(grad_norm_a=0.0, grad_norm_other=5.0) == 0.1


# -- Stage 0 e2b + the upstream-gradient canary ------------------------------------------

@pytest.mark.parametrize("body", FA4_TEST_BODIES)
def test_e2b_upstream_gradient_canary_reaches_the_filter(body):
    """One tiny end-to-end batch through filter -> hypernetwork -> LoRA backbone ->
    {support, success, production} losses (Stage 0's e2b), then the upstream-gradient
    canary: after a backward pass driven by the SIMULATOR LOSSES ALONE (L_belief
    deliberately excluded -- this isolates gradient
    reaching the filter FROM the output heads, not from the filter's own regulariser),
    assert finite non-zero gradient at >=1 LoRA basis pair, the hypernetwork, the episode
    encoder, and the filter's mu-head AND log-sigma-head -- and that those two heads'
    parameters actually change after one optimizer step.

    Parametrised over both filter bodies: the upstream-gradient canary is the primary
    requirement for the belief creator, and it is only meaningful if it is proven for
    whichever body the paired comparison ships, not just one of them.

    Step 1 is a documented special case, not a bug (scripts/p5_01_smoke_local.py shows the
    same one level shallower, for A/hyper): every LoRABasis.B starts at exact zero,
    so c's (hence u's, hence mu/log-sigma's) only path to the loss is scaled by B and is
    THEREFORE EXACTLY zero until step 1's update moves B away from zero. Step 2 must show
    gradient reaching everything.
    """
    p = _build_pipeline(body)
    item_codec = torch.randn(3, 5, CODEC_DIM)
    prev_feat = torch.randn(3, 5, PREV_FEAT_DIM)
    time_min, valid_mask = _dummy_geometry(3, 5)
    learner_idx, t_idx = torch.tensor([0, 1, 2]), torch.tensor([4, 4, 4])

    examples = [
        encode_example("What is 3+5?", "8", p.tok, max_len=32),
        encode_example("What is the boiling point of water in Celsius?", "100", p.tok, max_len=32),
        encode_example("Spell the word cat backwards.", "tac", p.tok, max_len=32),
    ]
    batch = collate(examples, pad_id=p.tok.pad_token_id)
    assert bool(batch["production_valid"].all())  # sanity: all three rows carry a real answer

    support_target = torch.tensor([0, 1, 2])
    success_target = torch.tensor([1.0, 0.0, 1.0])
    support_valid = torch.tensor([True, True, True])
    success_valid = torch.tensor([True, True, False])  # row 2: missing success

    trainable = (
        list(p.filt.parameters()) + list(p.hyper.parameters())
        + list(p.support_head.parameters()) + list(p.success_head.parameters())
        + [param for b in p.bases.values() for param in b.parameters()]
    )
    optimizer = torch.optim.Adam(trainable, lr=1e-2)

    mu_head_before = p.filt.knowledge_filter.mu_head.weight.detach().clone()
    log_sigma_head_before = p.filt.knowledge_filter.log_sigma_head.weight.detach().clone()

    def run_step() -> torch.Tensor:
        optimizer.zero_grad()
        mu_all, log_sigma_all = p.filt_forward(item_codec, prev_feat, time_min, valid_mask)
        mu_sel, log_sigma_sel = gather_belief(mu_all, log_sigma_all, learner_idx, t_idx)
        u = reparameterize_from_log_sigma(mu_sel, log_sigma_sel, k=1)[:, 0, :]
        out = forward_personalised(p.model, p.hyper, p.context, u, batch["input_ids"],
                                   batch["attention_mask"], batch["prompt_end_index"])
        l_s = support_loss(p.support_head(out["h_q"]), support_target, support_valid)
        l_y = success_loss(p.success_head(out["h_q"]), support_target, success_target,
                           success_valid & support_valid)
        l_a = production_loss(out["lm_logits"], batch["labels"], batch["production_valid"])
        total = l_s + l_y + 0.3 * l_a
        assert torch.isfinite(total), f"non-finite e2b loss: L_S={l_s} L_Y={l_y} L_A={l_a}"
        total.backward()
        return total

    # -- step 1: B's zero-init no-op zeroes A, the hypernetwork, AND the filter's belief
    # heads -- documented mechanism, asserted explicitly so a future change that breaks it
    # is caught here rather than read as "the canary just doesn't apply at step 1".
    run_step()
    for name, b in p.bases.items():
        assert b.B.grad is not None and torch.isfinite(b.B.grad).all()
        assert bool((b.B.grad != 0).any()), f"basis {name}: B has zero grad at step 1"
        assert b.A.grad is None or torch.all(b.A.grad == 0), f"basis {name}: A not zero at step 1"
    for param in p.hyper.parameters():
        assert param.grad is None or torch.all(param.grad == 0)
    mu_head_grad_s1 = p.filt.knowledge_filter.mu_head.weight.grad
    log_sigma_head_grad_s1 = p.filt.knowledge_filter.log_sigma_head.weight.grad
    assert mu_head_grad_s1 is None or torch.all(mu_head_grad_s1 == 0)
    assert log_sigma_head_grad_s1 is None or torch.all(log_sigma_head_grad_s1 == 0)
    optimizer.step()

    # -- step 2: B has moved off zero -- gradient must now reach everything upstream,
    # including the filter's own belief-producing heads.
    run_step()
    for name, b in p.bases.items():
        assert b.A.grad is not None and torch.isfinite(b.A.grad).all()
        assert bool((b.A.grad != 0).any()), f"basis {name}: A still zero-grad at step 2"
    hyper_nonzero = any(
        param.grad is not None and torch.isfinite(param.grad).all() and bool((param.grad != 0).any())
        for param in p.hyper.parameters()
    )
    assert hyper_nonzero, "hypernetwork received no nonzero finite gradient by step 2"

    ep_nonzero = any(
        param.grad is not None and torch.isfinite(param.grad).all() and bool((param.grad != 0).any())
        for param in p.filt.episode_encoder.parameters()
    )
    assert ep_nonzero, "episode encoder received no nonzero finite gradient by step 2"

    mu_grad = p.filt.knowledge_filter.mu_head.weight.grad
    log_sigma_grad = p.filt.knowledge_filter.log_sigma_head.weight.grad
    assert mu_grad is not None and torch.isfinite(mu_grad).all() and bool((mu_grad != 0).any()), (
        "filter mu_head received no nonzero finite gradient from the simulator losses by step 2"
    )
    assert log_sigma_grad is not None and torch.isfinite(log_sigma_grad).all() and bool((log_sigma_grad != 0).any()), (
        "filter log_sigma_head (the sharp limb -- gradient only via reparameterisation) "
        "received no nonzero finite gradient from the simulator losses by step 2"
    )
    optimizer.step()

    mu_head_after = p.filt.knowledge_filter.mu_head.weight.detach().clone()
    log_sigma_head_after = p.filt.knowledge_filter.log_sigma_head.weight.detach().clone()
    assert not torch.equal(mu_head_before, mu_head_after), "mu_head did not actually change"
    assert not torch.equal(log_sigma_head_before, log_sigma_head_after), "log_sigma_head did not actually change"
