"""A tiny, deterministic LM stub with a real causal mask, for the fast pytest canary suite.

`stub_backbone.StubBackbone` cannot serve this purpose: its attention has no causal mask
(it only needs to exercise checkpoint/resume mechanics, never token-position causality)
and it has no vocabulary head, so it cannot test "changing the current response tokens
leaves earlier hidden states unchanged" or the other canaries that concern a causal LM's
own structure. This module is a second, purpose-built stub: real causal masking
(upper-triangular + padding), an embedding, and an LM vocabulary head, with projections
named `q_proj`/`v_proj` so `atom.binding.hooked_backbone.attach_lora_basis` exercises the
same code path it uses on the real backbone (checked in `scripts/p5_00_hook_validate.py`).

Never used for any reported result: the same orchestration functions are run once against
the real, cached backbone to confirm the structural properties hold there too.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn


class _CausalStubAttn(nn.Module):
    def __init__(self, dim: int, n_heads: int):
        super().__init__()
        if dim % n_heads != 0:
            raise ValueError(f"dim {dim} must be divisible by n_heads {n_heads}")
        self.n_heads, self.head_dim = n_heads, dim // n_heads
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        shape = (b, t, self.n_heads, self.head_dim)
        q = self.q_proj(x).view(*shape).transpose(1, 2)
        k = self.k_proj(x).view(*shape).transpose(1, 2)
        v = self.v_proj(x).view(*shape).transpose(1, 2)
        scores = (q @ k.transpose(-1, -2)) / (self.head_dim**0.5)
        scores = scores.masked_fill(mask, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(b, t, d)
        return self.o_proj(out)


class _CausalStubLayer(nn.Module):
    def __init__(self, dim: int, n_heads: int):
        super().__init__()
        self.self_attn = _CausalStubAttn(dim, n_heads)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn(self.norm1(x), mask)
        return x + self.mlp(self.norm2(x))


class CausalStubLM(nn.Module):
    """Frozen by convention (matches the real backbone's contract): the training loop
    never registers these params with the optimiser directly -- only the attached LoRA
    basis + hypernetwork + task heads are trainable. `attach_lora_basis` also forces
    `requires_grad=False` on everything it wraps, belt-and-suspenders (hooked_backbone.py).
    """

    def __init__(self, vocab_size: int = 64, dim: int = 32, n_layers: int = 2,
                 n_heads: int = 4, pad_token_id: int = 0, seed: int = 0):
        super().__init__()
        torch.manual_seed(seed)  # global-RNG seeding, deliberately (see StubBackbone's
        # own docstring on why a selective per-tensor re-init can silently skip bias
        # vectors and break "same seed -> identical init").
        self.vocab_size, self.dim, self.pad_token_id = vocab_size, dim, pad_token_id
        self.embed = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList([_CausalStubLayer(dim, n_heads) for _ in range(n_layers)])
        self.final_norm = nn.LayerNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None,
                output_hidden_states: bool = False, logits_to_keep: int | torch.Tensor = 0):
        """Matches the HF causal-LM call contract the binding code relies on:
        `model(input_ids=, attention_mask=, output_hidden_states=)` -> object with
        `.logits` [N,T,V] and (if requested) `.hidden_states` (tuple, last = final norm
        output fed to the LM head -- what `atom.simulator.fa_alt2.gather_prompt_hidden`
        reads `h_q` from, exactly as it would from a real HF model).

        `logits_to_keep`: mirrors `transformers.models.gemma4.
        modeling_gemma4.Gemma4ForCausalLM.forward`'s own slicing exactly
        (`slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else
        logits_to_keep`, applied to the hidden states BEFORE `lm_head`, never to
        `hidden_states` in the return value) -- so this stub can exercise the identity
        gate for `forward_personalised`'s completion-span restriction on a CPU without
        loading the real backbone. Default 0 returns logits for every position.
        """
        b, t = input_ids.shape
        device = input_ids.device
        causal = torch.triu(torch.ones(t, t, dtype=torch.bool, device=device), diagonal=1)
        mask = causal.view(1, 1, t, t)
        if attention_mask is not None:
            pad = (attention_mask == 0).view(b, 1, 1, t)
            mask = mask | pad
            # A LEFT-padded query position (itself padding) would otherwise have EVERY
            # key masked (causal only allows key<=self, and `pad` excludes self too as a
            # key) -> softmax(all -inf row) = NaN. That NaN becomes this position's own
            # hidden state and is never read directly, but it silently poisons every
            # LATER layer's REAL-token outputs anyway: once this position's value
            # projection is NaN, any future query's attention sum includes
            # `0 * NaN = NaN` for that key even though its attention WEIGHT is correctly
            # exactly 0 -- IEEE754's zero-times-NaN is NaN, not zero, so correct masking
            # upstream does not protect against a NaN value once it exists. This arises
            # only with batched, padded inputs (e.g. `greedy_generate_batch`); batch-of-1
            # calls have no padding. Remedy: a fully-masked query row keeps a harmless
            # self-attend fallback (never read, but numerically finite) instead of an
            # all-(-inf) row.
            fully_masked = mask.all(dim=-1, keepdim=True)
            eye = torch.eye(t, dtype=torch.bool, device=device).view(1, 1, t, t)
            mask = mask & ~(fully_masked & eye)
        h = self.embed(input_ids)
        hidden_states = [h] if output_hidden_states else None
        for layer in self.layers:
            h = layer(h, mask)
            if output_hidden_states:
                hidden_states.append(h)
        h = self.final_norm(h)
        if output_hidden_states:
            hidden_states[-1] = h  # final layer's output AFTER the closing norm (HF convention)
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(h[:, slice_indices, :])
        return SimpleNamespace(
            logits=logits,
            hidden_states=tuple(hidden_states) if output_hidden_states else None,
        )
