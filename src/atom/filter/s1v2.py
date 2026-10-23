"""S1v2 filter: query-independent episode encoder -> GRU belief filter (thesis Section 5.3).

Reuses `KnowledgeFilter`/`sample_u` (gru.py) and `FiLMHeads` (heads/s1_heads.py) UNCHANGED --
the query dependence of v1 lived in neither. Only the recurrent input changes: `EpisodeEncoderV2` consumes the
PREVIOUS event's item codec (see episode_v2.py), so `b_t = f(H_<t)` strictly.

`query_independent` (default True) is a constructor flag, not a second model: with it False,
`forward` skips the one-position shift and feeds the CURRENT item's codec into the recurrent
input instead -- reproducing v1's defect exactly, inside the identical v2 architecture/
training code. This is the leaky-variant canary: train both flag settings on the SAME
clean v2 data, seed, and hyperparameters, and confirm (a) the mandatory peek canary --
same history, two different current questions -> identical b_t when True, DIFFERENT b_t when
False -- and (b) a non-vanishing AUC gap between them, proving the fix is structurally real,
not a no-op.
"""

from __future__ import annotations

import torch
from torch import nn

from atom.filter.episode_v2 import EpisodeEncoderV2, build_prev_item_codec
from atom.filter.gru import KnowledgeFilter


class S1FilterV2(nn.Module):
    def __init__(self, codec_dim: int, episode_hidden: int, filter_hidden: int,
                 latent_dim: int, query_independent: bool = True, prev_feat_dim: int | None = None):
        super().__init__()
        ep_kwargs = {} if prev_feat_dim is None else {"prev_feat_dim": prev_feat_dim}
        self.episode_encoder = EpisodeEncoderV2(codec_dim, episode_hidden, **ep_kwargs)
        self.knowledge_filter = KnowledgeFilter(episode_hidden, filter_hidden, latent_dim)
        self.query_independent = query_independent

    def forward(self, item_codec: torch.Tensor,
                prev_feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """item_codec: [N, T, codec_dim] (the CURRENT item at each position -- the caller's
        raw batch tensor, unshifted; the shift happens here, internally, so callers pass the
        same tensor FiLMHeads also consumes downstream). prev_feat: [N, T, 5] (see
        episode_v2.build_prev_feat_v2). Returns pre-observation (mu, log_sigma).
        """
        recurrent_item = (
            build_prev_item_codec(item_codec) if self.query_independent else item_codec
        )
        e_ep = self.episode_encoder(recurrent_item, prev_feat)
        return self.knowledge_filter(e_ep)
