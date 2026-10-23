"""A nonlinear item encoder, and S1FilterV2/FiLMHeads variants built on it (thesis Section 5.2;
the linear-versus-nonlinear comparison is in Appendix A.1). These are separate classes, not
edits: `episode_v2.py`, `heads/s1_heads.py` and `gru.py` are consumed UNCHANGED --
`KnowledgeFilter` and `build_prev_item_codec` are imported, not redefined, and no other call
site is affected by this file.

WHY THIS EXISTS. With a single bare `nn.Linear` off the frozen codec (`item_proj` in
episode_v2.py, `trunk` in s1_heads.py), the item pathway reaches 20.7% of a reliable ceiling
on item-difficulty R^2, while a NONLINEAR probe on the SAME codec dims reaches 33.5% (paired
delta +0.1051, P(delta>0)=1.000). This module builds that nonlinearity into the model itself,
so the comparison can be made on the metric that actually matters: filter AUC, not a
difficulty-regression proxy.

`ItemEncoderMLP` replaces the bare `nn.Linear` ONLY where `item_codec` enters -- `prev_proj`
(episode_v2.py's OTHER input, the 5-channel prev_feat) and every downstream head (gamma/beta/
correctness/engagement in FiLMHeads, the GRU cell itself) stay exactly as shipped. This isolates
the one variable under test.
"""

from __future__ import annotations

import torch
from torch import nn

from atom.filter.episode_v2 import PREV_FEAT_DIM_V2, build_prev_item_codec
from atom.filter.gru import KnowledgeFilter
from atom.heads.s1_heads import N_ENGAGEMENT


class ItemEncoderMLP(nn.Module):
    """codec_dim -> hidden, nonlinear. The one thing this module adds over `nn.Linear`."""

    def __init__(self, codec_dim: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(codec_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EpisodeEncoderV2ItemMLP(nn.Module):
    """`EpisodeEncoderV2` with `item_proj` swapped for `ItemEncoderMLP`. `prev_proj` (the
    prev_feat pathway) and the additive combination are byte-identical to episode_v2.py."""

    def __init__(self, codec_dim: int, hidden: int, prev_feat_dim: int = PREV_FEAT_DIM_V2):
        super().__init__()
        self.item_proj = ItemEncoderMLP(codec_dim, hidden)
        self.prev_proj = nn.Linear(prev_feat_dim, hidden)

    def forward(self, prev_item_codec: torch.Tensor, prev_feat: torch.Tensor) -> torch.Tensor:
        return self.item_proj(prev_item_codec) + self.prev_proj(prev_feat)


class FiLMHeadsItemMLP(nn.Module):
    """`FiLMHeads` with `trunk` swapped for `ItemEncoderMLP`. gamma/beta/correctness/
    engagement and the FiLM near-identity init are byte-identical to s1_heads.py."""

    def __init__(self, codec_dim: int, latent_dim: int, hidden: int):
        super().__init__()
        self.trunk = ItemEncoderMLP(codec_dim, hidden)
        self.gamma = nn.Linear(latent_dim, hidden)
        self.beta = nn.Linear(latent_dim, hidden)
        nn.init.normal_(self.gamma.weight, std=0.01)
        nn.init.zeros_(self.gamma.bias)
        nn.init.normal_(self.beta.weight, std=0.01)
        nn.init.zeros_(self.beta.bias)
        self.correctness = nn.Linear(hidden, 1)
        self.engagement = nn.Linear(hidden, N_ENGAGEMENT)

    def forward(self, item_codec: torch.Tensor,
                u: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        f = torch.relu(self.trunk(item_codec))
        f = f.unsqueeze(0)
        gamma = 1.0 + self.gamma(u)
        beta = self.beta(u)
        personalised = torch.relu(gamma * f + beta)
        correctness_logits = self.correctness(personalised).squeeze(-1)
        engagement_logits = self.engagement(personalised)
        return correctness_logits, engagement_logits


class S1FilterV2ItemMLP(nn.Module):
    """`S1FilterV2` with the item pathway swapped for the MLP variant. `KnowledgeFilter`
    (the GRU cell) is imported UNCHANGED -- it was never the ablation's target."""

    def __init__(self, codec_dim: int, episode_hidden: int, filter_hidden: int,
                 latent_dim: int, query_independent: bool = True,
                 sigma_parameterization: str = "direct"):
        super().__init__()
        self.episode_encoder = EpisodeEncoderV2ItemMLP(codec_dim, episode_hidden)
        self.knowledge_filter = KnowledgeFilter(episode_hidden, filter_hidden, latent_dim,
                                                 sigma_parameterization=sigma_parameterization)
        self.query_independent = query_independent

    def forward(self, item_codec: torch.Tensor,
                prev_feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        recurrent_item = (
            build_prev_item_codec(item_codec) if self.query_independent else item_codec
        )
        e_ep = self.episode_encoder(recurrent_item, prev_feat)
        return self.knowledge_filter(e_ep)


def canary_item_mlp_differs_from_linear(codec_dim: int = 64, hidden: int = 256,
                                        seed: int = 0) -> None:
    """Build-time canary: at a fresh init, the MLP item pathway must NOT be reducible to a
    linear map (else this whole ablation is comparing a model to itself in disguise). Checks
    non-linearity directly: f(2x) != 2*f(x) for a linear layer with a nonzero bias this WOULD
    hold approximately only by bias cancellation, so the check compares f(a+b) - f(a) - f(b) +
    f(0) (the exact linearity defect) against a nonzero floor.
    """
    torch.manual_seed(seed)
    enc = ItemEncoderMLP(codec_dim, hidden)
    a = torch.randn(1, codec_dim)
    b = torch.randn(1, codec_dim)
    zero = torch.zeros(1, codec_dim)
    with torch.no_grad():
        defect = (enc(a + b) - enc(a) - enc(b) + enc(zero)).abs().max().item()
    assert defect > 1e-4, (
        f"ItemEncoderMLP linearity defect is {defect:.2e} -- indistinguishable from a linear "
        "map; the ReLU may be dead at this init.")
