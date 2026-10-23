"""Global-LoRA control: one trained LoRA mixing vector shared by every learner.

Unlike `Hypernetwork`, which maps a per-example belief sample `u` to a mixing vector
`c(u)`, this control ignores `u` entirely: one shared, trainable `(m,)` vector is broadcast
to every example regardless of learner. It is kept as its own small class rather than a
further `Hypernetwork` kind, because a hypernetwork's contract is the map `u -> c`; a
control that structurally cannot see `u` is a different model, and keeping them separate
keeps clear the question the control isolates: does conditioning on the person help at
all, as opposed to whether a particular map shape helps.

This class provides only the broadcast mechanism; training it is done by the simulator
training scripts.
"""

from __future__ import annotations

import torch
from torch import nn


class GlobalMixingVector(nn.Module):
    """A single learned `(m,)` mixing vector, broadcast to `(n, m)` for a batch of `n`
    examples -- the personalisation-free counterpart of `Hypernetwork(u) -> c(u)`."""

    def __init__(self, m: int, seed: int = 0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.c = nn.Parameter(torch.randn(m, generator=g) * 0.01)

    def forward(self, n: int) -> torch.Tensor:
        """Returns `(n, m)`: the same learned vector repeated for every row -- deliberately
        ignores anything about WHICH learner or example this is; that is the point."""
        return self.c.unsqueeze(0).expand(n, -1)
