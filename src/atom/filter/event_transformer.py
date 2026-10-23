"""Body (c): the causal event-transformer filter -- self-attention over the learner's own
event sequence with attention biased by a learned function of SEMANTIC DISTANCE in Z and
ELAPSED TIME. Drop-in alternative to `gru.py`'s `KnowledgeFilter` (body (a)) behind the
identical `(mu, log_sigma)` interface.

Attention over event nodes is message passing on a fully-connected temporal graph, and the
(distance, Δt) bias is what turns generic attention into a structured graph operation. The
two bodies are compared under identical heads and loss; the recurrent filter ships, and
thesis Appendix A.2 reports the comparison (the temporal half of the bias carries most of
this body's value).

FOUR CONTRACT CLAUSES for a fair comparison with the GRU body, and how each is met here:
  1. SHARED EPISODE ENCODER -- `S1FilterV2C` composes `EpisodeEncoderV2` (the same class
     `S1FilterV2` uses), not a re-implementation. Only the sequence-mixing body differs.
  2. `latent_dim` STAYS 32 -- passed through untouched to the mu/log-sigma heads; the
     bake-off script asserts it against configs/s1_filter.yaml.
  3. ADD-DON'T-MODIFY -- new file; `gru.py`, `s1v2.py`, `episode_v2.py`, `heads/s1_heads.py`
     are untouched, and the existing test suite passes unedited.
  4. CAUSAL MASK + PEEK CANARY MUST FIRE ON (c) TOO -- see "query independence" below;
     `tests/test_event_transformer.py` runs both against this module.

QUERY INDEPENDENCE (`b_t = f(H_<t)`, carried into (c) by construction). The token at
position t is the SHIFTED episode-encoder output `e_ep[t] = enc(item_codec[t-1],
prev_feat[t])`: it describes the previous COMPLETED event, never t's own question. Causal
attention then lets position t see tokens 0..t, i.e. events strictly before t, so
`b_t = f(H_<t)` exactly -- the same guarantee `S1FilterV2` gets from its one-position shift,
obtained here through the same shift plus a causal mask. This matters for the ATTENTION BIAS
too, and is the one place body (c) could have silently reintroduced a query leak: the bias
is computed over the SHIFTED item codecs and SHIFTED timestamps, so neither `s_q(t)` nor the
time at which t is answered ever enters the belief. (Two consequences worth stating: the
"semantic distance" in the bias is between PAST items, not from the current query to the
past -- query-conditioned retrieval is a separate mechanism, a "question-conditioned
local readout", which would live in the emission path and is not part of
this filter; and Δt is elapsed time between past events, which is also the only version
available at belief time, since when the learner will answer t is not yet known.)

`query_independent=False` reproduces the query leak inside this same architecture (no
shift), so the peek canary is a two-sided test on (c) exactly as it is on the GRU -- a
canary that can only ever pass proves nothing.

THE BIAS, and why it is shaped this way. `bias[b, h, i, j] = w_dist[h] . rbf(dist_ij) +
w_time[h] . rbf(log1p(Δt_ij))`: a learned per-head linear readout of a FIXED radial-basis
expansion of each scalar. That is a genuinely learned smooth function of distance and of
elapsed time (the basis is fixed; the function over it is not), it costs `2 * n_rbf *
n_heads` parameters (64 at the defaults) rather than the tens of thousands a full
pairwise-MLP would, and different heads can learn opposite profiles -- one specialising in
"near and recent", another in "similar but long ago", which is precisely the REACH-BACK cell.
Distance is Euclidean on L2-normalised codec vectors, the metric used for this codec
throughout (`codec.probe_panel.normalize_rows`), so the evaluation's near/far
stratification and the model's own bias speak the same geometry.

Nodes with no underlying event (position 0, which is the zero start token, and right-padding)
get ZERO geometric bias at both endpoints rather than a distance computed against a zero
vector -- an all-zeros codec is not "an item at the origin", and letting it act as one would
put a spurious constant in every learner's first rows. Padding is additionally masked out of
the attention entirely; position 0 is never masked (it must remain attendable, or its own
row would softmax over an all-`-inf` set and produce NaN).

BIAS MODES (`bias_mode`) are the three pre-registered ablations plus the full arm, and they
are what make the comparison a mechanism claim whichever body wins:
  "none"          -- plain causal attention. Isolates "is it attention, or is it the bias?"
  "distance"      -- semantic-distance bias only.
  "time"          -- elapsed-time bias only.
  "distance_time" -- both. The full body (c).

MATCHED PARAMETER BUDGET. One pre-LN layer at `d_model=64, n_heads=4, d_ff=64` costs 29,440
parameters against the GRU body's 29,120 (+1.1%) -- `count_body_params` reports both and the
bake-off script gates the ratio. Single-layer is a deliberate consequence of that budget, not
an oversight: depth would have to be bought by narrowing d_model, and REACH-BACK is a
one-hop retrieval ("attend to the far-back similar item"), which is what one attention layer
does well.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from atom.filter.episode_v2 import EpisodeEncoderV2, build_prev_item_codec

BIAS_MODES = ("none", "distance", "time", "distance_time")

# RBF grid bounds. Distance is Euclidean on L2-normalised vectors, so it lives in [0, 2] by
# construction. log1p(elapsed minutes) is capped at ~13.2 = log1p(60*24*365) -- a year; FA's
# measured median inter-event gap is 1.4 min and only ~3% of events follow a >7-day gap
# (thesis Appendix A.2), so the grid deliberately spends most of its resolution
# where the data is and treats "very long ago" as one broad regime.
DIST_RBF_MIN, DIST_RBF_MAX = 0.0, 2.0
TIME_RBF_MIN, TIME_RBF_MAX = 0.0, 13.2


def _l2_normalise(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Row-wise L2 normalisation, the torch counterpart of `codec.probe_panel.normalize_rows`
    (which is numpy-only, hence not directly callable inside the forward pass). Same
    convention -- zero rows stay zero rather than becoming NaN.
    """
    return x / x.norm(dim=-1, keepdim=True).clamp(min=eps)


class DistanceTimeAttentionBias(nn.Module):
    """Additive per-head attention bias from a fixed RBF expansion of (semantic distance,
    elapsed time), with learned per-head weights. See the module docstring for the rationale.
    """

    def __init__(self, n_heads: int, n_rbf: int, bias_mode: str):
        super().__init__()
        if bias_mode not in BIAS_MODES:
            raise ValueError(f"bias_mode must be one of {BIAS_MODES}, got {bias_mode!r}")
        self.bias_mode = bias_mode
        self.n_heads = n_heads
        self.n_rbf = n_rbf
        self.use_distance = bias_mode in ("distance", "distance_time")
        self.use_time = bias_mode in ("time", "distance_time")

        dist_centres = torch.linspace(DIST_RBF_MIN, DIST_RBF_MAX, n_rbf)
        time_centres = torch.linspace(TIME_RBF_MIN, TIME_RBF_MAX, n_rbf)
        # Width = one grid spacing: adjacent basis functions overlap at ~exp(-1/2), so the
        # expansion is smooth rather than a set of disjoint bins.
        self.register_buffer("dist_centres", dist_centres)
        self.register_buffer("time_centres", time_centres)
        self.register_buffer(
            "dist_width", torch.tensor((DIST_RBF_MAX - DIST_RBF_MIN) / max(n_rbf - 1, 1)))
        self.register_buffer(
            "time_width", torch.tensor((TIME_RBF_MAX - TIME_RBF_MIN) / max(n_rbf - 1, 1)))

        # Zero-init: at step 0 the bias is exactly 0, so training starts from plain causal
        # attention and any bias the model ends up using is one it actively learned. This
        # also makes "none" a true nested special case of the other three modes at init.
        self.w_dist = nn.Parameter(torch.zeros(n_heads, n_rbf)) if self.use_distance else None
        self.w_time = nn.Parameter(torch.zeros(n_heads, n_rbf)) if self.use_time else None

    def _rbf(self, x: torch.Tensor, centres: torch.Tensor, width: torch.Tensor) -> torch.Tensor:
        """x: [..., ] -> [..., n_rbf]."""
        z = (x.unsqueeze(-1) - centres) / width
        return torch.exp(-0.5 * z.pow(2))

    def forward(self, node_codec: torch.Tensor, node_time_min: torch.Tensor,
                node_valid: torch.Tensor) -> torch.Tensor | None:
        """node_codec: [B, T, D] (SHIFTED item codecs -- the item of the event each token
        describes). node_time_min: [B, T] minutes since the learner's first event, shifted
        the same way. node_valid: [B, T] bool -- False where the token has no underlying
        event (position 0's start token, and right-padding).

        Returns the additive bias [B, n_heads, T, T], or None for bias_mode="none".
        """
        if self.bias_mode == "none":
            return None

        b, t, _ = node_codec.shape
        bias = node_codec.new_zeros(b, self.n_heads, t, t)
        # Geometry is only meaningful where BOTH endpoints correspond to a real event.
        pair_valid = (node_valid.unsqueeze(2) & node_valid.unsqueeze(1)).unsqueeze(1)  # [B,1,T,T]

        if self.use_distance:
            # For L2-normalised rows, ||a - b||^2 == 2 - 2 a.b exactly, so the pairwise
            # distance is one matmul. Preferred over torch.cdist for two reasons: it is
            # uniformly supported and fast on the MPS backend, and the
            # clamp makes the sqrt safe against the small negative values floating-point
            # error puts on the diagonal.
            zn = _l2_normalise(node_codec)
            dist = (2.0 - 2.0 * (zn @ zn.transpose(-2, -1))).clamp(min=0.0).sqrt()  # [B,T,T]
            feats = self._rbf(dist, self.dist_centres, self.dist_width)  # [B, T, T, K]
            bias = bias + torch.einsum("bijk,hk->bhij", feats, self.w_dist)

        if self.use_time:
            # Elapsed time between the two past events, symmetric and non-negative. Under the
            # causal mask only j <= i is ever read, where this equals (t_i - t_j) >= 0.
            dt = (node_time_min.unsqueeze(2) - node_time_min.unsqueeze(1)).abs()
            feats = self._rbf(torch.log1p(dt), self.time_centres, self.time_width)
            bias = bias + torch.einsum("bijk,hk->bhij", feats, self.w_time)

        return bias * pair_valid


class CausalEventTransformerFilter(nn.Module):
    """Body (c). One pre-LN causal self-attention block over the episode-summary sequence,
    with the (distance, Δt) attention bias, then the same `mu_head`/`log_sigma_head` pair
    `gru.KnowledgeFilter` exposes.

    The head names are deliberately identical to the GRU body's: the gradient canary
    asserts non-zero gradient at the filter's `mu_head` and `log_sigma_head` and that both
    move after one optimizer step, and it must be able to find them by the same name on
    either arm.
    """

    def __init__(self, input_dim: int, hidden: int, latent_dim: int,
                 n_heads: int = 4, d_ff: int = 64, n_rbf: int = 8,
                 bias_mode: str = "distance_time",
                 log_sigma_min: float = -5.0, log_sigma_max: float = 2.0):
        super().__init__()
        if hidden % n_heads != 0:
            raise ValueError(f"hidden ({hidden}) must be divisible by n_heads ({n_heads})")
        self.hidden = hidden
        self.n_heads = n_heads
        self.head_dim = hidden // n_heads
        self.bias_mode = bias_mode

        # Identity when the episode encoder already emits `hidden` dims (the default config),
        # so the matched-budget comparison is not quietly inflated by a projection the GRU
        # arm does not pay for.
        self.in_proj = nn.Identity() if input_dim == hidden else nn.Linear(input_dim, hidden)

        self.ln_attn = nn.LayerNorm(hidden)
        self.qkv = nn.Linear(hidden, 3 * hidden)
        self.out_proj = nn.Linear(hidden, hidden)
        self.attn_bias = DistanceTimeAttentionBias(n_heads, n_rbf, bias_mode)

        self.ln_ff = nn.LayerNorm(hidden)
        self.ff = nn.Sequential(nn.Linear(hidden, d_ff), nn.GELU(), nn.Linear(d_ff, hidden))

        self.mu_head = nn.Linear(hidden, latent_dim)
        self.log_sigma_head = nn.Linear(hidden, latent_dim)
        self.log_sigma_min = log_sigma_min
        self.log_sigma_max = log_sigma_max

    def forward(self, e_ep: torch.Tensor, node_codec: torch.Tensor,
                node_time_min: torch.Tensor, node_valid: torch.Tensor,
                valid_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """e_ep: [B, T, input_dim] episode summaries (already query-independent).
        node_codec/node_time_min/node_valid: the shifted geometry inputs (see
        `DistanceTimeAttentionBias.forward`). valid_mask: [B, T] bool -- real sequence
        positions (right-padding excluded from attention).

        Returns (mu, log_sigma), each [B, T, latent_dim] -- same contract as
        `gru.KnowledgeFilter.forward`.
        """
        b, t, _ = e_ep.shape
        x = self.in_proj(e_ep)

        h = self.ln_attn(x)
        qkv = self.qkv(h).view(b, t, 3, self.n_heads, self.head_dim)
        q, k, v = (qkv[:, :, i].transpose(1, 2) for i in range(3))       # each [B, H, T, dh]

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)    # [B, H, T, T]
        bias = self.attn_bias(node_codec, node_time_min, node_valid)
        if bias is not None:
            scores = scores + bias

        causal = torch.ones(t, t, dtype=torch.bool, device=e_ep.device).tril()
        allowed = causal.view(1, 1, t, t) & valid_mask.view(b, 1, 1, t)
        # Position 0 is never masked as a key (it is the start token, and masking it would
        # softmax an all-`-inf` row for the first position of every learner).
        scores = scores.masked_fill(~allowed, float("-inf"))
        attn = torch.softmax(scores, dim=-1)

        ctx = (attn @ v).transpose(1, 2).reshape(b, t, self.hidden)
        x = x + self.out_proj(ctx)
        x = x + self.ff(self.ln_ff(x))

        mu = self.mu_head(x)
        log_sigma = self.log_sigma_head(x).clamp(self.log_sigma_min, self.log_sigma_max)
        return mu, log_sigma


class S1FilterV2C(nn.Module):
    """Body (c) wired into the query-independent filter contract: `EpisodeEncoderV2` (the SHARED class,
    contract clause 1) -> `CausalEventTransformerFilter`. Interface-compatible with
    `s1v2.S1FilterV2` apart from the two extra geometry tensors body (c) needs, and it takes
    the caller's RAW (unshifted) batch tensors for exactly the same reason `S1FilterV2` does:
    the shift is the model's business, so a caller cannot forget it, and the peek canary can
    swap the current question in the caller's tensor and see it correctly ignored.
    """

    def __init__(self, codec_dim: int, episode_hidden: int, filter_hidden: int,
                 latent_dim: int, n_heads: int = 4, d_ff: int = 64, n_rbf: int = 8,
                 bias_mode: str = "distance_time", query_independent: bool = True,
                 prev_feat_dim: int | None = None):
        super().__init__()
        ep_kwargs = {} if prev_feat_dim is None else {"prev_feat_dim": prev_feat_dim}
        self.episode_encoder = EpisodeEncoderV2(codec_dim, episode_hidden, **ep_kwargs)
        self.knowledge_filter = CausalEventTransformerFilter(
            episode_hidden, filter_hidden, latent_dim,
            n_heads=n_heads, d_ff=d_ff, n_rbf=n_rbf, bias_mode=bias_mode)
        self.query_independent = query_independent

    def forward(self, item_codec: torch.Tensor, prev_feat: torch.Tensor,
                time_min: torch.Tensor, valid_mask: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor]:
        """item_codec: [B, T, codec_dim] -- the CURRENT item at each position (unshifted).
        prev_feat: [B, T, 5] from `episode_v2.build_prev_feat_v2`. time_min: [B, T] minutes
        since each learner's first event, at the CURRENT event (unshifted). valid_mask:
        [B, T] bool.

        With `query_independent=True` (default) every path through this module -- recurrent
        input AND attention bias -- reads only shifted tensors, so `b_t = f(H_<t)`. With
        False, the shift is skipped everywhere, reproducing the query leak inside body (c) for
        the peek canary's negative arm.
        """
        if self.query_independent:
            node_codec = build_prev_item_codec(item_codec)
            node_time_min = _shift_right(time_min)
            node_valid = valid_mask & _shift_right(valid_mask)
        else:
            node_codec, node_time_min, node_valid = item_codec, time_min, valid_mask

        e_ep = self.episode_encoder(node_codec, prev_feat)
        return self.knowledge_filter(e_ep, node_codec, node_time_min, node_valid, valid_mask)


def _shift_right(x: torch.Tensor) -> torch.Tensor:
    """One-position shift along T, zero/False at position 0 -- the [N, T] counterpart of
    `episode_v2.build_prev_item_codec`. Dtype-preserving, so it serves the float time tensor
    and the boolean validity tensor alike.
    """
    out = torch.zeros_like(x)
    if x.shape[1] > 1:
        out[:, 1:] = x[:, :-1]
    return out


def count_body_params(module: nn.Module) -> int:
    """Trainable parameters of a filter BODY only -- the sequence-mixing stack plus the
    mu/log-sigma heads, excluding the shared episode encoder (which is identical across arms
    by contract clause 1, so including it would dilute the matched-budget comparison).
    """
    return sum(p.numel() for p in module.parameters() if p.requires_grad)