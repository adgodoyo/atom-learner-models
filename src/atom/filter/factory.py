"""One construction point for the filter body, so that the body is a config switch.

Training scripts on both corpora call `build_filter(cfg["body"], ...)` (GRU by default)
and get back a model plus a uniform forward function, without importing two classes or
knowing that one of them needs two extra geometry tensors.

The uniform signature is `forward(item_codec, prev_feat, time_min, valid_mask)`. Body (a)
ignores the last two, which is why the wrapper exists: the alternative is every downstream
caller branching on the body, which is exactly the kind of drift that lets a leaky belief
boundary (a belief that reads the current question) go unnoticed.

BODY NAMES are the arm names of the filter-body comparison (thesis Appendix A.2), so a
config value, a report row and a checkpoint directory all read the same:
    "gru"              body (a) -- S1FilterV2, the shipped query-independent filter
    "c_distance_time"  body (c), full: attention biased by semantic distance AND elapsed time
    "c_distance"       body (c), semantic-distance bias only        (ablation)
    "c_time"           body (c), elapsed-time bias only             (ablation)
    "c_none"           body (c), plain causal attention, no bias    (ablation)
"""

from __future__ import annotations

from typing import Callable

import torch
from torch import nn

from atom.filter.event_transformer import S1FilterV2C
from atom.filter.s1v2 import S1FilterV2

ARM_BIAS_MODE = {"c_distance_time": "distance_time", "c_distance": "distance",
                 "c_time": "time", "c_none": "none"}
BODIES = ("gru", *ARM_BIAS_MODE)

ForwardFn = Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
                     "tuple[torch.Tensor, torch.Tensor]"]


def build_filter(body: str, codec_dim: int, episode_hidden: int, filter_hidden: int,
                 latent_dim: int, n_heads: int = 4, d_ff: int = 64, n_rbf: int = 8,
                 query_independent: bool = True,
                 prev_feat_dim: int | None = None) -> tuple[nn.Module, ForwardFn]:
    """Returns (model, forward) where forward always takes
    `(item_codec, prev_feat, time_min, valid_mask)` and returns `(mu, log_sigma)`.

    All tensors are the caller's RAW, UNSHIFTED batch tensors: both bodies do their own
    one-position shift internally, so a caller cannot forget it and the peek canary can swap
    the current question in the caller's own tensor and watch it be correctly ignored.

    `prev_feat_dim`: overrides `EpisodeEncoderV2`'s recurrent-input width away
    from FA's own 5-channel layout. `None` (default) keeps FA's existing behaviour
    byte-for-byte; CodeWorkout passes its own prev_feat width explicitly.
    """
    if body == "gru":
        model = S1FilterV2(codec_dim, episode_hidden, filter_hidden, latent_dim,
                           query_independent=query_independent, prev_feat_dim=prev_feat_dim)
        return model, lambda ic, pf, tm, vm: model(ic, pf)
    if body not in ARM_BIAS_MODE:
        raise ValueError(f"unknown filter body {body!r}; expected one of {BODIES}")
    model = S1FilterV2C(codec_dim, episode_hidden, filter_hidden, latent_dim,
                        n_heads=n_heads, d_ff=d_ff, n_rbf=n_rbf,
                        bias_mode=ARM_BIAS_MODE[body], query_independent=query_independent,
                        prev_feat_dim=prev_feat_dim)
    return model, lambda ic, pf, tm, vm: model(ic, pf, tm, vm)
