"""The shared LoRA basis: the population binding between belief space and the backbone's weights.

Shared basis: m LoRA pairs {A_i in R^(r x in_dim), B_i in R^(out_dim x r)}, one basis per
adapted projection (e.g. one q or v projection at one layer). The basis IS the manifold
materialised in weight space: all person-variation for that projection lives in its span.
A per-sample mixing vector c(u) in R^m (from the hypernetwork, see hypernetwork.py) selects
a point on that manifold: DeltaW(u) = alpha * sum_i c_i(u) B_i A_i.

`apply` computes x @ DeltaW(u)^T for a BATCH where every row can have a DIFFERENT c (a
different personalised adapter per sample) WITHOUT ever materialising the full
(out_dim x in_dim) DeltaW per sample -- the whole point of routing through the shared basis.
Peak intermediate is (batch, m, r) and (batch, m, out_dim); m=32, r=8 by default
both tiny compared to in_dim/out_dim at backbone scale.

Derivation (kept here, not just in a comment elsewhere, since the einsum order is the part
most likely to be "optimised" into something subtly wrong later):
    y_delta_b = alpha * sum_i c_{b,i} * B_i @ (A_i @ x_b)
    p_{b,i}   = A_i @ x_b                              (batch, m, r)
    q_{b,i}   = B_i @ p_{b,i}                           (batch, m, out_dim)
    y_delta_b = alpha * sum_i c_{b,i} * q_{b,i}          (batch, out_dim)
No (batch, out_dim, in_dim) tensor is ever formed.
"""

from __future__ import annotations

import math

import torch
from torch import nn


class LoRABasis(nn.Module):
    """Shared basis of `m` rank-`r` LoRA pairs for one linear projection.

    `alpha` follows the standard LoRA scaling convention (alpha/r); default alpha=r gives a
    scaling of 1.0 -- a documented class default, overridable via config.
    """

    def __init__(self, in_dim: int, out_dim: int, m: int = 32, r: int = 8,
                 alpha: float | None = None, seed: int = 0):
        super().__init__()
        self.in_dim, self.out_dim, self.m, self.r = in_dim, out_dim, m, r
        self.alpha = r if alpha is None else alpha
        g = torch.Generator().manual_seed(seed)
        # Kaiming-style init for A (the "down" projection), zero-init for B (the "up"
        # projection) -- standard LoRA convention: DeltaW starts at exactly zero, so an
        # untrained basis is a strict no-op regardless of what the hypernetwork outputs.
        self.A = nn.Parameter(torch.empty(m, r, in_dim))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5), generator=g)
        self.B = nn.Parameter(torch.zeros(m, out_dim, r))

    def delta_weight(self, c: torch.Tensor) -> torch.Tensor:
        """c: (..., m) -> DeltaW(u): (..., out_dim, in_dim). Only for inspection/tests --
        `apply` never materialises this at training/inference time (see module docstring)."""
        scale = self.alpha / self.r
        # (..., m, out_dim, r) @ (..., m, r, in_dim) -> (..., m, out_dim, in_dim), weighted
        # by c and summed over m.
        BA = torch.einsum("mor,mri->moi", self.B, self.A)  # (m, out_dim, in_dim)
        return scale * torch.einsum("...m,moi->...oi", c, BA)

    def apply(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """x: (N, in_dim), c: (N, m) -> personalised delta output (N, out_dim), i.e. the
        DeltaW(u) @ x term ONLY (add to the frozen base projection's own output separately).

        N is whatever the caller has already flattened batch/K-samples/etc into -- this
        function has no opinion on that structure, only that x and c share their leading
        dimension (one mixing vector per row of x).
        """
        if x.shape[0] != c.shape[0]:
            raise ValueError(f"x and c must share N (leading dim); got {x.shape[0]} vs {c.shape[0]}")
        scale = self.alpha / self.r
        p = torch.einsum("ni,mri->nmr", x, self.A)   # (N, m, r)
        q = torch.einsum("mor,nmr->nmo", self.B, p)  # (N, m, out_dim)
        return scale * torch.einsum("nm,nmo->no", c, q)
