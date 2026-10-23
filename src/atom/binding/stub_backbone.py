"""A tiny, deterministic, HF-shaped stub backbone for the checkpoint/resume kill test.

The kill test proves checkpoint/resume MECHANICS (basis + hypernetwork + optimiser + RNG
round-tripping bit-for-bit through a real `kill -9`), not model quality or real-backbone
memory behaviour. Running it against gemma4-e4b would add a training step's worth of
activation memory on top of its 17 GB inference footprint for no benefit -- the mechanics
are identical on any q/v-projection model. The real backbone's structure (module names,
dtypes, device) is validated separately, once, in scripts/p5_00_hook_validate.py; this stub
is never used for anything that claims to be a result.

Named exactly like the real thing (`self_attn.q_proj`/`v_proj`) so `attach_lora_basis`
exercises the identical code path.
"""

from __future__ import annotations

import torch
from torch import nn


class StubAttn(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        attn = torch.softmax(q @ k.transpose(-1, -2) / (q.shape[-1] ** 0.5), dim=-1)
        return self.o_proj(attn @ v)


class StubLayer(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.self_attn = StubAttn(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class StubBackbone(nn.Module):
    """Frozen by convention: the training loop never registers its params with the
    optimiser (only the attached LoRA basis + hypernetwork are trainable, matching the
    real backbone's frozen-forever contract)."""

    def __init__(self, dim: int = 16, n_layers: int = 3, seed: int = 0):
        super().__init__()
        # Seed the GLOBAL torch RNG for construction, rather than selectively re-initing
        # weight matrices with a local generator: such a `p.dim() > 1` filter (meant
        # to skip LayerNorm's 1-D weight/bias) would also silently skip every Linear's 1-D
        # BIAS vector (nn.Sequential's Linears default to bias=True), which would keep
        # PyTorch's own global-RNG-seeded default init -- making two "same seed" instances
        # differ. Seeding globally before construction makes every default init
        # (weights AND biases) deterministic given `seed`, with no selective-filter bug
        # possible; this only ever runs at construction, in the same order on every fresh
        # or resumed invocation (see p5_00_pilot_train.py), so it does not interfere with
        # the training loop's own later, explicit RNG-state checkpointing.
        torch.manual_seed(seed)
        self.dim = dim
        self.embed = nn.Linear(1, dim, bias=False)
        self.layers = nn.ModuleList([StubLayer(dim) for _ in range(n_layers)])
        self.head = nn.Linear(dim, 1)
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq, 1) -> (batch, seq, 1)."""
        h = self.embed(x)
        for layer in self.layers:
            h = layer(h)
        return self.head(h)
