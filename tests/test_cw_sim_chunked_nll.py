"""CW-SIM 30-Aug audit item 1 -- `production_log_likelihood_sum_per_row`'s new
`row_chunk_size` argument must be an exact no-op on the returned numbers (chunking along
the row axis of an already-row-independent cross-entropy), only ever changing peak
memory. This is the canary the audit's fix depends on: a chunked rewrite that quietly
changed a number would be far worse than the OOM it replaces."""
from __future__ import annotations

import torch

from atom.simulator.fa_alt2 import production_log_likelihood_sum_per_row, production_nll_per_row


def _random_batch(b: int, t: int, v: int, seed: int):
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(b, t, v, generator=g)
    labels = torch.randint(0, v, (b, t), generator=g)
    production_valid = torch.ones(b, t - 1, dtype=torch.bool)
    # a realistic prompt/pad pattern: first few rows have a short invalid prefix and one
    # fully-invalid row (row_has_any must be correctly False for it under any chunk size)
    production_valid[0, :3] = False
    production_valid[1, :] = False
    return logits, labels, production_valid


def test_row_chunk_size_none_matches_default_call_exactly():
    logits, labels, valid = _random_batch(b=6, t=9, v=17, seed=0)
    a = production_log_likelihood_sum_per_row(logits, labels, valid)
    b = production_log_likelihood_sum_per_row(logits, labels, valid, row_chunk_size=None)
    for x, y in zip(a, b):
        torch.testing.assert_close(x, y, rtol=0, atol=0)


def test_chunking_is_bit_identical_to_unchunked_for_several_chunk_sizes():
    logits, labels, valid = _random_batch(b=7, t=11, v=23, seed=1)
    baseline = production_log_likelihood_sum_per_row(logits, labels, valid)
    for chunk in (1, 2, 3, 4, 7, 100):
        chunked = production_log_likelihood_sum_per_row(logits, labels, valid, row_chunk_size=chunk)
        for name, x, y in zip(("log_p_seq", "n_valid_tok", "row_has_any"), baseline, chunked):
            torch.testing.assert_close(x, y, rtol=0, atol=0, msg=f"{name} mismatch at chunk={chunk}")


def test_fully_invalid_row_reports_zero_and_row_has_any_false_under_chunking():
    logits, labels, valid = _random_batch(b=4, t=6, v=13, seed=2)
    log_p_seq, n_valid_tok, row_has_any = production_log_likelihood_sum_per_row(
        logits, labels, valid, row_chunk_size=1)
    assert row_has_any[1].item() == 0.0
    assert n_valid_tok[1].item() == 0
    assert log_p_seq[1].item() == 0.0


# `production_nll_per_row` -- item 1's fix extended to the TRAINING path (30-Aug
# shakeout's CUDA OOM in `production_backward_microbatched`). Same canary requirement:
# chunking must be exactly a no-op on the numbers AND must not break gradient flow
# (the caller's single later `.backward()` over the concatenated output must see the
# identical graph it would have unchunked).

def test_production_nll_per_row_chunking_matches_unchunked_values_exactly():
    logits, labels, valid = _random_batch(b=6, t=9, v=17, seed=3)
    baseline = production_nll_per_row(logits, labels, valid)
    for chunk in (1, 2, 3, 6, 100):
        chunked = production_nll_per_row(logits, labels, valid, row_chunk_size=chunk)
        for name, x, y in zip(("per_row", "row_has_any"), baseline, chunked):
            torch.testing.assert_close(x, y, rtol=0, atol=0, msg=f"{name} mismatch at chunk={chunk}")


def test_production_nll_per_row_chunking_preserves_gradient_flow_and_values():
    torch.manual_seed(0)
    b, t, v = 5, 7, 11
    logits_base = torch.randn(b, t, v, requires_grad=True)
    labels = torch.randint(0, v, (b, t))
    valid = torch.ones(b, t - 1, dtype=torch.bool)

    logits_unchunked = logits_base.clone().detach().requires_grad_(True)
    per_row_u, has_any_u = production_nll_per_row(logits_unchunked, labels, valid)
    per_row_u.sum().backward()

    logits_chunked = logits_base.clone().detach().requires_grad_(True)
    per_row_c, has_any_c = production_nll_per_row(logits_chunked, labels, valid, row_chunk_size=2)
    per_row_c.sum().backward()

    torch.testing.assert_close(per_row_u, per_row_c, rtol=0, atol=0)
    torch.testing.assert_close(has_any_u, has_any_c, rtol=0, atol=0)
    torch.testing.assert_close(logits_unchunked.grad, logits_chunked.grad, rtol=1e-6, atol=1e-6)
    assert logits_chunked.grad.abs().sum().item() > 0.0  # canary: gradient actually flowed
