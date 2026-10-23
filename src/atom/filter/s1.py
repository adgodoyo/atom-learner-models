"""v1 filter: episode encoder -> GRU belief filter, wired together. The v1 episode encoder
feeds the CURRENT item into the recurrent input, so this belief is query-dependent;
`s1v2.S1FilterV2` is the query-independent filter used in the thesis. Kept for
reproducibility of the scripts that load v1 checkpoints.
"""

from __future__ import annotations

import torch
from torch import nn

from atom.filter.episode import EpisodeEncoder
from atom.filter.gru import KnowledgeFilter


class S1Filter(nn.Module):
    def __init__(self, codec_dim: int, episode_hidden: int, filter_hidden: int,
                 latent_dim: int):
        super().__init__()
        self.episode_encoder = EpisodeEncoder(codec_dim, episode_hidden)
        self.knowledge_filter = KnowledgeFilter(episode_hidden, filter_hidden, latent_dim)

    def forward(self, item_codec: torch.Tensor,
                prev_feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """item_codec: [N, T, codec_dim], prev_feat: [N, T, 4] (see episode.build_prev_feat).
        Returns pre-observation (mu, log_sigma), each [N, T, latent_dim].
        """
        e_ep = self.episode_encoder(item_codec, prev_feat)
        return self.knowledge_filter(e_ep)
