#!/usr/bin/env python3
"""Shared helpers for the zero-shot LLM selector (§8.4): the constrained identifier read-out.

Why a constrained read-out. Free-form greedy continuation of ``CHOICE=`` parsed at only 0.425
against a pre-registered 0.99 validity floor, and the parsed choices collapsed onto two menu
slots in 1,296 states: slot 0 (``A``) 469 times and slot 23 (``X``) 82 times. Both are
artifacts, not decisions:

  * ``X`` was the placeholder in the instruction line (``CHOICE=X, where X is one of ...``)
    and is also identifier 23, so the model was copying the prompt's own literal;
  * the remaining failures are base-model echo -- ``gemma-4-E4B`` is not instruction-tuned,
    and at long prompts it continues the document rather than answering (parse rate falls
    0.81 -> 0.30 as prompts grow past 2k tokens).

A selector whose output space is the menu should not lose a decision to formatting. This module
takes the decision as the softmax over the identifier-token logits at the single position after
``CHOICE=`` (the same ``pi(j|s)`` the fine-tuned selector uses). It is greedy and deterministic,
parse failure is structurally impossible, and it costs one forward pass with no decode loop.
The free-form parse rate is retained and reported as a diagnostic of the backbone's zero-shot
format compliance. This read-out is an amendment to the pre-registered protocol, made after the
validity check failed and before any selector-versus-comparator quantity had been computed.
"""

from __future__ import annotations

import pathlib
import sys
from typing import Any, Sequence

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))


def identifier_token_ids(tokenizer: Any, identifiers: Sequence[str]) -> list[int]:
    """One token id per identifier, in menu order, asserted single-token."""
    ids = []
    for symbol in identifiers:
        enc = tokenizer(symbol, add_special_tokens=False)["input_ids"]
        if len(enc) != 1:
            raise AssertionError(f"identifier {symbol!r} is not single-token: {enc}")
        ids.append(int(enc[0]))
    if len(set(ids)) != len(ids):
        raise AssertionError("identifier token ids are not distinct")
    return ids


def _length_batches(
    prompts: Sequence[str],
    tokenizer: Any,
    max_rows: int,
    max_prompt_tokens: int,
    token_budget: int,
) -> tuple[list[int], list[list[int]]]:
    """Group rows into length-homogeneous batches under a PADDED-TOKEN budget.

    A fixed row count is the wrong unit on a 24 GB card holding 15 GB of weights: a batch
    of eight that fits on a small pilot runs out of memory (21.4 of 22.06 GiB) once enough
    learners make a batch of eight maximal 5,120-token prompts inevitable. A small pilot
    samples the length tail too thinly to reveal this.

    Batching by padded cost (`rows x longest-in-batch`) makes the worst case bounded by
    construction instead of bounded by luck.  Sorting by length also *reduces* padding,
    which makes the numerics more stable, not less: same-length batches are bit-identical,
    while mixed-length batches carry small padding noise.
    """
    lengths = [len(tokenizer(p, add_special_tokens=False)["input_ids"]) for p in prompts]
    order = sorted(range(len(prompts)), key=lambda i: -lengths[i])
    batches: list[list[int]] = []
    current: list[int] = []
    longest = 0
    for i in order:
        n = min(lengths[i], max_prompt_tokens)
        candidate_longest = max(longest, n)
        if current and (len(current) + 1 > max_rows
                        or (len(current) + 1) * candidate_longest > token_budget):
            batches.append(current)
            current, longest = [i], n
        else:
            current.append(i)
            longest = candidate_longest
    if current:
        batches.append(current)
    return order, batches


def read_out_choices(
    bundle: dict[str, Any],
    prompts: Sequence[str],
    menu_sizes: Sequence[int],
    all_token_ids: Sequence[int],
    batch_size: int,
    max_prompt_tokens: int,
    on_batch: Any | None = None,
    token_budget: int = 18000,
) -> tuple[list[int], list[dict[str, float]]]:
    """Greedy argmax over the live menu's identifier logits at the read-out position.

    Returns the chosen zero-based slot per prompt, plus per-row diagnostics: the chosen
    identifier's probability *within the menu*, the menu-restricted entropy (in nats), and
    the probability mass the menu holds against the full vocabulary — the last is the
    direct measure of how much of the model's belief the constraint is discarding, and it
    is reported rather than hidden.
    """
    import torch

    model, tokenizer, device = bundle["model"], bundle["tokenizer"], bundle["device"]
    old_pad, old_trunc = tokenizer.padding_side, tokenizer.truncation_side
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    token_ids = torch.tensor(list(all_token_ids), dtype=torch.long, device=device)
    slots_by_index: dict[int, int] = {}
    diag_by_index: dict[int, dict[str, float]] = {}
    order, batches = _length_batches(
        prompts, tokenizer, batch_size, max_prompt_tokens, token_budget)
    del order
    done = 0
    try:
        for group in batches:
            batch = [prompts[i] for i in group]
            sizes = [menu_sizes[i] for i in group]
            if on_batch is not None:
                on_batch(done, len(prompts))
            encoded = tokenizer(batch, return_tensors="pt", padding=True, truncation=True,
                                max_length=max_prompt_tokens).to(device)
            with torch.inference_mode():
                # `logits_to_keep=1` materialises the LM head for the LAST position only.
                # Without it the forward pass builds `batch × seq × 262,144` logits --
                # 6.61 GiB at batch 4 × ~3.4k tokens, enough to run out of memory.
                # `generate()` passes this flag for you; a direct forward call does not.
                out = model(**encoded, logits_to_keep=1)
            # Left padding ⇒ the final column is every row's true last token.
            last = out.logits[:, -1, :].float()
            full_logprobs = torch.log_softmax(last, dim=-1)
            for i, (row_index, n) in enumerate(zip(group, sizes)):
                live = token_ids[:n]
                menu_logits = last[i].index_select(0, live)
                probs = torch.softmax(menu_logits, dim=-1)
                slot = int(torch.argmax(menu_logits).item())
                slots_by_index[row_index] = slot
                p = probs.clamp_min(1e-12)
                diag_by_index[row_index] = {
                    "p_chosen": float(probs[slot].item()),
                    "menu_entropy_nats": float(-(p * p.log()).sum().item()),
                    "menu_mass_vs_vocab": float(
                        full_logprobs[i].index_select(0, live).exp().sum().item()),
                }
            done += len(group)
            if on_batch is not None:
                on_batch(done, len(prompts))
            del out, last, full_logprobs
    finally:
        tokenizer.padding_side, tokenizer.truncation_side = old_pad, old_trunc
    if len(slots_by_index) != len(prompts):
        raise RuntimeError("read-out lost a row")
    # Restore the caller's row order: batching is an execution detail and must never
    # permute results.
    return ([slots_by_index[i] for i in range(len(prompts))],
            [diag_by_index[i] for i in range(len(prompts))])


def summarise_diagnostics(diagnostics: Sequence[dict[str, float]]) -> dict[str, float]:
    if not diagnostics:
        return {}
    out = {}
    for key in ("p_chosen", "menu_entropy_nats", "menu_mass_vs_vocab"):
        arr = np.asarray([d[key] for d in diagnostics], dtype=float)
        out[f"{key}_mean"] = float(arr.mean())
        out[f"{key}_median"] = float(np.median(arr))
        out[f"{key}_min"] = float(arr.min())
    return out
