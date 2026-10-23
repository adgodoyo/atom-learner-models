"""The hypernetwork H_xi: u -> c(u) in R^m, the map from person latent to LoRA mixing coefficients.

**Not assumed, ablated**: H_xi exists only to decouple two geometries (the filter's
tracking-a-person-over-time space vs the basis's mixing-weight-edits space). If they
already agree, H_xi solves a non-problem. The map is a config flag with three settings,
compared as an ablation:
  - identity: c = u  (requires u_dim == m; the latent IS the mixing vector -- simplest)
  - linear:   c = W u  (a learned rotation between the two geometries)
  - mlp:      c = MLP(u)  (full nonlinear decoupling; the default, following a "start
              expressive, ablate down" rule -- shrinking later is a one-line change,
              discovering you needed more is a messier backtrack)
Prior expectation: linear wins or ties against mlp; if so, that is reported as a finding in
its own right.
"""

from __future__ import annotations

import torch
from torch import nn


class Hypernetwork(nn.Module):
    def __init__(self, u_dim: int, m: int, kind: str = "mlp", hidden: int = 64, seed: int = 0):
        super().__init__()
        if kind not in ("identity", "linear", "mlp"):
            raise ValueError(f"kind must be identity/linear/mlp, got {kind!r}")
        if kind == "identity" and u_dim != m:
            raise ValueError(f"identity map requires u_dim == m; got u_dim={u_dim}, m={m}")
        self.kind, self.u_dim, self.m = kind, u_dim, m
        torch.manual_seed(seed)  # module-local init only; caller's global seed is untouched
        if kind == "identity":
            self.net: nn.Module | None = None
        elif kind == "linear":
            self.net = nn.Linear(u_dim, m, bias=False)
        else:
            self.net = nn.Sequential(nn.Linear(u_dim, hidden), nn.GELU(), nn.Linear(hidden, m))

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """u: (..., u_dim) -> c: (..., m)."""
        if self.kind == "identity":
            return u
        return self.net(u)
