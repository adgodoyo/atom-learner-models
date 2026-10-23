"""CodeWorkout item-encoder MLP variants (the nonlinear item encoder of the
item-encoder ablation, thesis Appendix A.1; Section 5.2).

Built by reusing `ItemEncoderMLP`/`EpisodeEncoderV2ItemMLP` from
`atom.filter.item_encoder_mlp` directly, never redefining them. On FoundationalASSIST,
replacing the filter's bare `nn.Linear` item pathway with this shared 2-layer MLP improved
AUC; this module ports the same nonlinearity to CodeWorkout's two item pathways -- the
filter's `EpisodeEncoderV2.item_proj` (via `S1FilterV2`, the "gru" body) and
`OutcomeHead.trunk` (the FiLM-conditioned 4-class head) -- at CodeWorkout's widths
(32/64/64, summary_dim 16). The FA-side `S1FilterV2ItemMLP`/`FiLMHeadsItemMLP` in
`item_encoder_mlp.py` are not reused: they hardcode FA's 5-channel `prev_feat_dim`.

Two new classes:

    S1FilterV2ItemMLP  -- a thin CW-side wrapper around the IMPORTED
                          `EpisodeEncoderV2ItemMLP` + `KnowledgeFilter`, existing to
                          plumb `prev_feat_dim` (CW's `summary_dim`) and the optional
                          width parameterisation through, which the FA-side
                          `S1FilterV2ItemMLP` wrapper does not need to.
    OutcomeHeadItemMLP -- `OutcomeHead` with `trunk` swapped for the IMPORTED
                          `ItemEncoderMLP`; gamma/beta/prefix_proj/outcome and the FiLM
                          near-identity init are byte-identical to
                          `fa_hierarchical_heads.OutcomeHead`.

**INVARIANT (asserted by test, `tests/test_cw_item_mlp.py`): the MLP is a SHARED function
of the codec vector, never a per-item lookup.** `ItemEncoderMLP` is stateless with respect
to item identity -- the same weights map ANY codec vector (including one for a problem
never seen during training) to a hidden representation, which is exactly what keeps CW's
item-agnosticism claim (no head consumes a learned item ID) intact while raising
resolution. A per-item embedding table would forfeit that claim and must not be
introduced as an "optimisation" of this module.
"""

from __future__ import annotations

import torch
from torch import nn

from atom.filter.episode_v2 import build_prev_item_codec
from atom.filter.gru import KnowledgeFilter
from atom.filter.item_encoder_mlp import EpisodeEncoderV2ItemMLP, ItemEncoderMLP


class S1FilterV2ItemMLP(nn.Module):
    """`S1FilterV2` with the item pathway swapped for the MLP variant. Imports
    `EpisodeEncoderV2ItemMLP` directly (never redefines it) -- that class already accepts
    `prev_feat_dim` as a constructor kwarg; only the FA-side `S1FilterV2ItemMLP` wrapper
    (in `item_encoder_mlp.py`) calls it without one, so this thin CodeWorkout wrapper
    exists to pass CW's `summary_dim` and the optional width parameterisation
    (`sigma_parameterization`) through. With the default `"direct"`
    parameterisation, `KnowledgeFilter` behaves exactly as its default."""

    def __init__(self, codec_dim: int, episode_hidden: int, filter_hidden: int,
                latent_dim: int, query_independent: bool = True, prev_feat_dim: int | None = None,
                sigma_parameterization: str = "direct"):
        super().__init__()
        ep_kwargs = {} if prev_feat_dim is None else {"prev_feat_dim": prev_feat_dim}
        self.episode_encoder = EpisodeEncoderV2ItemMLP(codec_dim, episode_hidden, **ep_kwargs)
        self.knowledge_filter = KnowledgeFilter(
            episode_hidden, filter_hidden, latent_dim,
            sigma_parameterization=sigma_parameterization,
        )
        self.query_independent = query_independent

    def forward(self, item_codec: torch.Tensor,
               prev_feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        recurrent_item = (
            build_prev_item_codec(item_codec) if self.query_independent else item_codec
        )
        e_ep = self.episode_encoder(recurrent_item, prev_feat)
        return self.knowledge_filter(e_ep)


class OutcomeHeadItemMLP(nn.Module):
    """`OutcomeHead` with `trunk` swapped for `ItemEncoderMLP`. gamma/beta/prefix_proj/
    outcome and the FiLM near-identity init are byte-identical to
    `fa_hierarchical_heads.OutcomeHead` -- only the item_codec -> hidden step gains a
    nonlinearity, the same scope as the FA-side `FiLMHeadsItemMLP` swap."""

    def __init__(self, codec_dim: int, latent_dim: int, hidden: int, n_classes: int,
                prefix_dim: int | None = None):
        super().__init__()
        self.trunk = ItemEncoderMLP(codec_dim, hidden)
        self.prefix_proj = nn.Linear(prefix_dim, hidden) if prefix_dim else None
        self.gamma = nn.Linear(latent_dim, hidden)
        self.beta = nn.Linear(latent_dim, hidden)
        nn.init.normal_(self.gamma.weight, std=0.01)
        nn.init.zeros_(self.gamma.bias)
        nn.init.normal_(self.beta.weight, std=0.01)
        nn.init.zeros_(self.beta.bias)
        self.outcome = nn.Linear(hidden, n_classes)

    def forward(self, item_codec: torch.Tensor, u: torch.Tensor,
               prefix: torch.Tensor | None = None) -> torch.Tensor:
        if self.prefix_proj is not None and prefix is None:
            raise ValueError("this head was constructed with prefix_dim set; forward() needs `prefix`")
        if self.prefix_proj is None and prefix is not None:
            raise ValueError("this head has no prefix_dim configured; got an unexpected `prefix`")
        trunk_pre = self.trunk(item_codec)
        if self.prefix_proj is not None:
            trunk_pre = trunk_pre + self.prefix_proj(prefix)
        f = torch.relu(trunk_pre)
        gamma = 1.0 + self.gamma(u)
        beta = self.beta(u)
        personalised = torch.relu(gamma * f + beta)
        return self.outcome(personalised)
