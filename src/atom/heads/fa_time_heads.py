"""Widens `FiLMHeadsSupport`'s trunk to also read the CURRENT item's repeat-exposure
features (`atom.filter.fa_time_features.build_current_repeat_features`), for the time-feature
ablation (thesis Appendix A.2). Legal in the emission path for the same reason
`item_codec[t]` itself is: it describes item t, not the recurrent belief. A subclass, not an
edit -- `fa2_heads.py`/`s1_heads.py` are imported unchanged (override trunk + forward,
nothing else).
"""

from __future__ import annotations

import torch
from torch import nn

from atom.heads.fa2_heads import FiLMHeadsSupport


class FiLMHeadsSupportTime(FiLMHeadsSupport):
    def __init__(self, codec_dim: int, latent_dim: int, hidden: int, n_repeat_feat: int = 2):
        super().__init__(codec_dim, latent_dim, hidden)
        # Widen the trunk to also read the repeat-exposure features; everything else
        # (gamma/beta/correctness/support) is untouched, so FiLM's near-identity init and
        # parameter count elsewhere are identical to the baseline arm.
        self.trunk = nn.Linear(codec_dim + n_repeat_feat, hidden)

    def forward(self, item_codec: torch.Tensor, u: torch.Tensor,
               repeat_feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """item_codec: [N, T, codec_dim]. u: [K, N, T, latent_dim]. repeat_feat: [N, T,
        n_repeat_feat] (current item's own repeat-exposure features -- see module docstring).
        Returns (correctness_logits [K, N, T], support_logits [K, N, T, 4])."""
        x = torch.cat([item_codec, repeat_feat], dim=-1)
        f = torch.relu(self.trunk(x)).unsqueeze(0)             # [1, N, T, H]
        personalised = torch.relu((1.0 + self.gamma(u)) * f + self.beta(u))
        return self.correctness(personalised).squeeze(-1), self.support(personalised)
