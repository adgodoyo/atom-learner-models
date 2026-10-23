"""Tiny GRU filter + correctness head for the smoke test.

This is the miniature of the real ATOM filter (episode encoder -> recurrent belief ->
head). At step t the model may use the current item's features (known before the learner
answers) and the history of past outcomes, but never the current outcome it is predicting
-- the same no-peek discipline the real filter must honour.
"""

from __future__ import annotations

import torch
from torch import nn

_START = 2  # "previous outcome" token before the first step (0 = wrong, 1 = correct)


class TinyFilter(nn.Module):
    def __init__(self, codec_dim: int, hidden: int = 32):
        super().__init__()
        self.item_proj = nn.Linear(codec_dim, hidden)
        self.prev_embed = nn.Embedding(3, hidden)  # {wrong, correct, start}
        self.gru = nn.GRU(hidden, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, codec: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """codec: [N, T, D], y: [N, T] observed outcomes. Returns logits [N, T].

        Input at step t combines the current item's features with the *previous* step's
        outcome, so the belief entering the head at t has seen history strictly before t.
        """
        n, t = y.shape
        prev = torch.full((n, t), _START, dtype=torch.long, device=y.device)
        prev[:, 1:] = y[:, :-1].long()
        x = self.item_proj(codec) + self.prev_embed(prev)  # [N, T, H]
        out, _ = self.gru(x)                               # [N, T, H]
        return self.head(out).squeeze(-1)                  # [N, T]
