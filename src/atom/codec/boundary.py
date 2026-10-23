"""Boundary-state extraction for code submissions.

A code submission's *trajectory* is the sequence of hidden states at its statement/block
boundaries -- one causal forward pass, read at multiple positions (never re-encoded),
following the refinement-contraction reading of the codec. This module finds those
boundary positions and reads them off an already-computed hidden-state stack. It is
production code for the episode encoder (the step-level trajectory input), not test
scaffolding -- the probe battery is simply another consumer.

Boundaries = end of every non-blank source line, plus every `;` and `}` (statement/block
ends; covers Java, the CW corpus's language, and C-family code generally). Evenly
subsampled to a cap so pathologically long submissions don't dominate memory/compute.
"""

from __future__ import annotations

import numpy as np


def boundary_char_offsets(code: str, cap: int = 32) -> list[int]:
    """Character offsets (exclusive end) of statement/block boundaries in `code`.

    Union of: end of every non-blank line, every ';', every '}'. Evenly subsampled to
    <= cap positions (first/last preserved) if more are found -- a long submission's
    boundaries are thinned, not truncated to its head.
    """
    offsets: set[int] = set()
    pos = 0
    for line in code.splitlines(keepends=True):
        pos += len(line)
        if line.strip():
            offsets.add(pos - (len(line) - len(line.rstrip())))
    for i, ch in enumerate(code):
        if ch in (";", "}"):
            offsets.add(i + 1)
    ordered = sorted(offsets)
    if len(ordered) <= cap or not ordered:
        return ordered
    idx = np.linspace(0, len(ordered) - 1, cap)
    keep = sorted(set(int(round(i)) for i in idx))
    return [ordered[i] for i in keep]


def boundary_token_positions(tokenizer, text: str, char_offsets: list[int],
                             char_base: int = 0, tok_base: int = 0) -> list[int]:
    """Map character offsets in `text` (a substring starting at `char_base` within a
    larger tokenized sequence, whose first token is `tok_base`) to token indices in that
    larger sequence: the token whose span first reaches or exceeds each offset.

    Uses the tokenizer's own fast-tokenizer offset mapping (verified against
    google/gemma-4-E4B, a GemmaTokenizer with is_fast=True) -- no independent
    re-tokenization, so positions land on the SAME token ids the causal forward already
    produced for `text` when it was encoded in-context.
    """
    enc = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    spans = enc["offset_mapping"]  # [(start_char, end_char), ...] within `text`
    positions = []
    ti = 0
    for off in char_offsets:
        while ti < len(spans) - 1 and spans[ti][1] < off:
            ti += 1
        positions.append(tok_base + ti)
    return positions


def extract_boundary_states(bundle: dict, code: str, prefix: str = "", cap: int = 32
                            ) -> tuple[np.ndarray, int]:
    """One causal forward pass over `prefix + code` (prefix="" = code-alone scope).

    Returns (states, n_found): states is (n_boundaries_found, n_layers, hidden_dim) --
    last-token hidden state at each boundary position, every layer, from THIS ONE pass
    (never a per-boundary re-encode). n_found <= cap (fewer if the code has fewer
    boundaries than the cap).
    """
    import torch

    model, tok, device = bundle["model"], bundle["tokenizer"], bundle["device"]
    full = prefix + code
    full_ids = tok(full, add_special_tokens=True).input_ids
    if prefix:
        prefix_ids = tok(prefix, add_special_tokens=True).input_ids
        tok_base = len(prefix_ids)
    else:
        # `tok_base=0` here (via `prefix_ids=[]`) would read every boundary one token
        # early -- `add_special_tokens=True` on `code` alone still
        # prepends a BOS token (confirmed on google/gemma-4-E4B), which
        # `boundary_token_positions`'s own no-special-tokens tokenization of `code`
        # doesn't count. `tok_base` must be that special-token count, not 0.
        content_ids = tok(code, add_special_tokens=False).input_ids
        tok_base = len(full_ids) - len(content_ids)

    offsets = boundary_char_offsets(code, cap=cap)
    if not offsets:
        offsets = [len(code)]  # degenerate (no ';'/'}'/newline): fall back to the last char
    positions = boundary_token_positions(tok, code, offsets, tok_base=tok_base)
    positions = sorted(set(p for p in positions if p < len(full_ids)))

    ids = torch.tensor([full_ids], device=device)
    with torch.no_grad():
        out = model(input_ids=ids, output_hidden_states=True)
    hs = out.hidden_states
    n_layers, dim = len(hs), hs[0].shape[-1]
    states = np.empty((len(positions), n_layers, dim), np.float32)
    for li, h in enumerate(hs):
        h0 = h[0].float()
        for bi, p in enumerate(positions):
            states[bi, li] = h0[p].cpu().numpy()
    del out, hs
    if device == "mps":
        torch.mps.empty_cache()
    return states, len(positions)


def boundary_mean(states: np.ndarray) -> np.ndarray:
    """Boundary-mean readout: mean over the boundary-state trajectory, per layer -> (L, D).

    A POOLED VIEW of the trajectory for probing only -- the episode encoder learns its own
    aggregation over the same per-boundary states; a pooled probe failing does not by
    itself show the encoder's richer aggregation would fail too.
    """
    return states.mean(axis=0)
