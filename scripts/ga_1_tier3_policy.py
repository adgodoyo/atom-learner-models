#!/usr/bin/env python3
"""Shared helpers for the fine-tuned LLM selector (Figure 12): the shuffled LLM selection policy,
used by both the trainer and the evaluator.

Wraps the zero-shot selector's renderer, alphabet and constrained `CHOICE=` read-out
(`ga_1_tier0_prompt`, `ga_1_tier0_readout`) unchanged and adds one thing: the CRN identifier
shuffle. The candidate list is permuted before it reaches `build_prompt`, so identifier `A`
labels a pseudorandom candidate, and the read-out slot is mapped back through the permutation.

Why the shuffle matters: the zero-shot policy emits `A` 90.8% of the time (99.7% without
history). Under the shuffle that behaviour becomes a uniformly random pick rather than "always
the first candidate", so a trained policy cannot score by learning a position and the zero-shot
arm becomes a fair baseline.
"""

from __future__ import annotations

import pathlib
import sys
from typing import Sequence

import numpy as np
import torch

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from ga_1_tier0_prompt import IDENTIFIERS, assert_prompt_wellformed, build_prompt  # noqa: E402
from ga_1_tier0_readout import _length_batches  # noqa: E402
from ga_1_tier3_states import apply_permutation, crn_permutation, episode_key  # noqa: E402


def render_state(events, candidates, budget, step, episode_text, statements, tokenizer,
                 max_prompt_tokens, with_history, perm):
    """Render one state's prompt under a given candidate permutation."""
    shuffled = apply_permutation(candidates, perm)
    prompt = build_prompt(events, shuffled, budget, step, episode_text, statements,
                          tokenizer, max_prompt_tokens=max_prompt_tokens,
                          with_history=with_history)
    assert_prompt_wellformed(prompt, shuffled, tokenizer, max_prompt_tokens)
    return prompt, shuffled


def menu_logits(bundle, prompts: Sequence[str], menu_sizes: Sequence[int],
                token_ids: Sequence[int], max_prompt_tokens: int,
                batch_rows: int, token_budget: int, grad: bool = False,
                on_batch=None) -> list[torch.Tensor]:
    """Logits over each row's live menu identifiers at the read-out position.

    `logits_to_keep=1` materialises the LM head for the final position only -- without it
    the forward builds a `batch x seq x 262,144` tensor (6.61 GiB at batch 4, enough to run
    out of memory). Batches are bounded by PADDED TOKEN COST, not row count, as in
    `ga_1_tier0_readout`: eight maximal prompts in one batch do not fit.
    """
    model, tokenizer, device = bundle["model"], bundle["tokenizer"], bundle["device"]
    old_pad, old_trunc = tokenizer.padding_side, tokenizer.truncation_side
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    ids = torch.tensor(list(token_ids), dtype=torch.long, device=device)
    out: dict[int, torch.Tensor] = {}
    done = 0
    try:
        _, batches = _length_batches(prompts, tokenizer, batch_rows, max_prompt_tokens,
                                     token_budget)
        for group in batches:
            batch = [prompts[i] for i in group]
            enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True,
                            max_length=max_prompt_tokens).to(device)
            ctx = torch.enable_grad() if grad else torch.inference_mode()
            with ctx:
                res = model(**enc, logits_to_keep=1)
                last = res.logits[:, -1, :].float()
                for i, row in enumerate(group):
                    out[row] = last[i].index_select(0, ids[:menu_sizes[row]])
            done += len(group)
            if on_batch is not None:
                on_batch(done, len(prompts))
    finally:
        tokenizer.padding_side, tokenizer.truncation_side = old_pad, old_trunc
    if len(out) != len(prompts):
        raise RuntimeError("menu_logits lost a row")
    return [out[i] for i in range(len(prompts))]


def rollout_picks(bundle, episodes: Sequence[dict], budget: int, episode_text, statements,
                  token_ids, *, with_history: bool, max_prompt_tokens: int,
                  batch_rows: int, token_budget: int, shuffle_seed: int,
                  arm: str, on_batch=None) -> dict[str, list[str]]:
    """Greedy shuffled roll-out for a whole cohort at one budget, batched per step.

    Returns {episode_key: [episode_id, ...]} in pick order, keyed by (learner, start) --
    a learner alone is not unique, because the same learner has one scenario row per
    start. All rows advance in lockstep so each step's prompts batch together, as in the
    zero-shot runner.
    """
    states = [{"ep": ep, "known": list(ep["prefix_events"]),
               "remaining": list(range(len(ep["candidates"]))), "picks": []}
              for ep in episodes]
    max_steps = max((min(budget, len(s["remaining"])) for s in states), default=0)
    for step in range(max_steps):
        active = [s for s in states if s["remaining"]]
        if not active:
            break
        prompts, perms, sizes = [], [], []
        for st in active:
            ep = st["ep"]
            cands = [ep["candidates"][i] for i in st["remaining"]]
            perm = crn_permutation(len(cands), arm, str(ep["learner"]),
                                   int(ep["start"]), budget, step, seed=shuffle_seed)
            prompt, _ = render_state(st["known"], cands, budget, step, episode_text,
                                     statements, bundle["tokenizer"], max_prompt_tokens,
                                     with_history, perm)
            prompts.append(prompt)
            perms.append(perm)
            sizes.append(len(cands))
        logits = menu_logits(bundle, prompts, sizes, token_ids, max_prompt_tokens,
                             batch_rows, token_budget, grad=False, on_batch=on_batch)
        for st, perm, lg in zip(active, perms, logits):
            slot = int(torch.argmax(lg).item())
            if not 0 <= slot < len(st["remaining"]):
                raise RuntimeError(f"read-out returned off-menu slot {slot}")
            chosen_local = perm[slot]                     # undo the shuffle
            chosen_global = st["remaining"].pop(chosen_local)
            chosen = st["ep"]["candidates"][chosen_global]
            st["known"].append(chosen)
            st["picks"].append(str(chosen["episode_id"]))
    return {episode_key(s["ep"]): s["picks"] for s in states}


def identifier_ids_for(tokenizer) -> list[int]:
    from ga_1_tier0_readout import identifier_token_ids
    return identifier_token_ids(tokenizer, IDENTIFIERS)


def shuffled_uniform_reference(n: int) -> np.ndarray:
    """Uniform distribution over a menu of size n -- the shuffle's own mechanical null."""
    return np.full(n, 1.0 / n, dtype=np.float64)
