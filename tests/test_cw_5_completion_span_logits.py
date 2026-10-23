"""Identity gate for the `logits_to_keep` restriction in `forward_personalised`.

Run the full-sequence-logits and completion-span-only-logits paths on
the SAME model/inputs and assert every downstream per-row quantity
(`production_nll_per_row`, `production_topk_per_row`) is IDENTICAL to float tolerance --
not "similar". This is what makes the restriction a verified identity rather than an
assumed one: prompt positions carry `label=-100` and were always masked out of the loss,
but "masked out of the loss" and "never even projected through lm_head" are two different
code paths that could in principle disagree (a shape bug, an off-by-one on the cut point),
and only a byte-for-byte comparison rules that out.

Runs on CPU with the stub backbone (`CausalStubLM`, which mirrors
`transformers.models.gemma4.modeling_gemma4.Gemma4ForCausalLM.forward`'s own
`logits_to_keep` slicing exactly), so the restriction is verified before it touches the
real backbone.
"""

from __future__ import annotations

import torch

from atom.binding import Hypernetwork, attach_lora_basis
from atom.binding.stub_causal_lm import CausalStubLM
from atom.simulator.cw_alt import collate, encode_example
from atom.simulator.fa_alt2 import forward_personalised, production_nll_per_row, production_topk_per_row

HIDDEN_DIM = 32
LATENT_DIM = 6


class _FakeTok:
    pad_token_id = 0
    eos_token_id = 63

    def __call__(self, text: str, add_special_tokens: bool = True) -> dict:
        words = [w for w in text.replace("\n", " \n ").split(" ") if w != ""]
        ids = [(sum(ord(c) for c in w) % 40) + 2 for w in words]
        if add_special_tokens:
            ids = [1, *ids]
        return {"input_ids": ids}


def _build_rows(tok, n_rows: int = 20) -> list[dict]:
    """Deliberately uneven prompt lengths (short statement / no prior code vs long
    statement / long prior code) so `prompt_end_index` varies substantially across the
    batch -- the identity gate is vacuous if every row happens to share one boundary."""
    statements = [
        "Write a function that returns the sum of two integers a and b.",
        ("Given a list of integers nums and an integer target, write a function that "
         "returns the indices of the two numbers that add up to target. You may assume "
         "each input has exactly one solution, and you may not use the same element "
         "twice. Return the answer in any order."),
        "Reverse a string.",
        ("Implement a class representing a simple bank account with deposit, withdraw "
         "and balance-check methods, raising an error on overdraft attempts and logging "
         "every transaction to an internal history list for later audit."),
    ]
    rows = []
    for i in range(n_rows):
        statement = statements[i % len(statements)]
        prior_code = None if i % 3 == 0 else f"def f(a, b):\n    return a - b  # attempt {i}"
        code = f"def f(a, b):\n    return a + b  # row {i}"
        rows.append(encode_example(statement, prior_code, code, tok, max_len=48,
                                   min_completion_tokens=8, outcome_code=i % 4))
    return rows


def _build_personalised_model(seed: int = 0):
    torch.manual_seed(seed)
    tok = _FakeTok()
    model = CausalStubLM(vocab_size=64, dim=HIDDEN_DIM, n_layers=2, n_heads=4,
                         pad_token_id=tok.pad_token_id, seed=seed)
    bases, context = attach_lora_basis(model, m=8, r=4, seed=seed + 1)
    hyper = Hypernetwork(u_dim=LATENT_DIM, m=8, kind="mlp", seed=seed + 2)
    return tok, model, hyper, context


def test_completion_span_logits_identical_to_full_sequence_logits():
    tok, model, hyper, context = _build_personalised_model(seed=0)
    rows = _build_rows(tok, n_rows=20)
    encoded = collate(rows, pad_id=tok.pad_token_id, device="cpu")
    u = torch.randn(len(rows), LATENT_DIM, generator=torch.Generator().manual_seed(3))

    offset = int(encoded["prompt_end_index"].min().item())
    seq_len = encoded["input_ids"].shape[1]
    # Guard against a vacuous gate: the batch must actually span a real prompt-length
    # spread, else `offset` never cuts anything and every assertion below would pass
    # trivially even with a broken slice.
    assert offset > 0, "fixture must include at least one row whose completion starts late"
    assert offset < seq_len - 1, "fixture must leave a real completion span to restrict to"

    out_full = forward_personalised(model, hyper, context, u, encoded["input_ids"],
                                    encoded["attention_mask"], encoded["prompt_end_index"],
                                    logits_to_keep=0)
    out_restricted = forward_personalised(model, hyper, context, u, encoded["input_ids"],
                                          encoded["attention_mask"], encoded["prompt_end_index"],
                                          logits_to_keep=seq_len - offset)

    assert out_full["lm_logits"].shape[1] == seq_len
    assert out_restricted["lm_logits"].shape[1] == seq_len - offset
    assert out_restricted["lm_logits"].shape[1] < out_full["lm_logits"].shape[1]

    # h_q is read from the TRUNK's hidden states, never from the restricted lm_head
    # projection -- must be bit-identical (not merely close), since `logits_to_keep`
    # is documented to never touch it.
    assert torch.equal(out_full["h_q"], out_restricted["h_q"])

    labels_full = encoded["labels"]
    labels_sliced = encoded["labels"][:, offset:]

    nll_full, has_any_full = production_nll_per_row(out_full["lm_logits"], labels_full,
                                                     encoded["production_valid"])
    nll_restricted, has_any_restricted = production_nll_per_row(
        out_restricted["lm_logits"], labels_sliced, encoded["production_valid"])

    assert torch.equal(has_any_full, has_any_restricted)
    assert torch.allclose(nll_full, nll_restricted, atol=1e-6), (
        f"max abs diff = {(nll_full - nll_restricted).abs().max().item():.3e} -- "
        "the completion-span restriction must reproduce the full-sequence per-row "
        "production NLL EXACTLY, not approximately"
    )

    top1_full, exact_full, row_has_any_full = production_topk_per_row(
        out_full["lm_logits"], labels_full, encoded["production_valid"])
    top1_restricted, exact_restricted, row_has_any_restricted = production_topk_per_row(
        out_restricted["lm_logits"], labels_sliced, encoded["production_valid"])

    assert torch.equal(row_has_any_full, row_has_any_restricted)
    assert torch.equal(top1_full, top1_restricted), "top-1 accuracy per row must match exactly"
    assert torch.equal(exact_full, exact_restricted), "exact-match per row must match exactly"


def test_logits_to_keep_zero_default_matches_omitting_the_argument():
    """Callers that omit `logits_to_keep` (every FA site) must be unaffected -- confirms
    the keyword's default reproduces that exact call, not merely a numerically-close one
    (the "new keyword, default unchanged" bar applied to every keyword of this trainer)."""
    tok, model, hyper, context = _build_personalised_model(seed=1)
    rows = _build_rows(tok, n_rows=6)
    encoded = collate(rows, pad_id=tok.pad_token_id, device="cpu")
    u = torch.randn(len(rows), LATENT_DIM, generator=torch.Generator().manual_seed(4))

    out_default = forward_personalised(model, hyper, context, u, encoded["input_ids"],
                                       encoded["attention_mask"], encoded["prompt_end_index"])
    out_explicit_zero = forward_personalised(model, hyper, context, u, encoded["input_ids"],
                                             encoded["attention_mask"], encoded["prompt_end_index"],
                                             logits_to_keep=0)

    assert torch.equal(out_default["lm_logits"], out_explicit_zero["lm_logits"])
    assert torch.equal(out_default["h_q"], out_explicit_zero["h_q"])
