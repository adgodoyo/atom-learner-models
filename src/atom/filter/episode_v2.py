"""Query-independent episode encoder: the belief is a function of the history only,
`b_t = f(H_<t)` (thesis Section 5.3). The v1 encoder (`episode.py`) fed the CURRENT item's
codec into the recurrent input at position t, so `b_t = f(H_<t, q_t)`: two different
candidate questions at the same history produce two different beliefs, which is unsound
for the "shared person posterior" reading of K-sampling. This module is a separate
implementation, not a patch (v1's episode.py/s1.py/gru.py are untouched and still
importable for reproducibility).

The fix: the recurrent input at position t describes what happened at t-1 (the previous
COMPLETED event: which item, what outcome), never t's own item. `build_prev_item_codec`
performs the same one-position shift `build_prev_feat` already applied to
(correct, hint_bin, saw_answer) -- shifting item_codec too closes the gap. `KnowledgeFilter`
(gru.py) and `FiLMHeads` (heads/s1_heads.py) are reused UNCHANGED: the GRU cell itself was
never the defect, and FiLMHeads already takes the CURRENT item's codec as a direct,
non-recurrent input alongside a sampled u -- exactly "current item in the emission path
only", the shape the architecture requires, with no change needed.

`build_prev_feat_v2` additionally carries a `prev_score_missing` flag (5th channel): the
rebuilt manifest never imputes a missing platform score to 0, so the previous step's
`correct` value may be genuinely unknown. Zeroing it silently (as `prev_correct=0`) would
reintroduce the "missing coerced to failure" defect the manifest rebuild removes, one layer
down in the filter's own recurrent input -- the explicit flag lets the network learn to disregard
`prev_correct` when it is a placeholder, not a fact.
"""

from __future__ import annotations

import torch
from torch import nn

PREV_FEAT_DIM_V2 = 5  # [prev_correct, prev_hint_bin, prev_saw_answer, prev_valid, prev_score_missing]


class EpisodeEncoderV2(nn.Module):
    def __init__(self, codec_dim: int, hidden: int, prev_feat_dim: int = PREV_FEAT_DIM_V2):
        """`prev_feat_dim`: generalises the recurrent-input width beyond FA's own
        5-channel `build_prev_feat_v2` layout -- default preserves every FA call site
        byte-for-byte. CodeWorkout constructs its own prev_feat vector (`cw_filter_batch.py`)
        with a different width/semantics (its own outcome classes, not FA's
        correct/hint/saw_answer/valid/score_missing), so the projection layer's input
        size must be configurable rather than hardcoded to FA's own channel count."""
        super().__init__()
        self.item_proj = nn.Linear(codec_dim, hidden)
        self.prev_proj = nn.Linear(prev_feat_dim, hidden)

    def forward(self, prev_item_codec: torch.Tensor, prev_feat: torch.Tensor) -> torch.Tensor:
        """prev_item_codec: [N, T, codec_dim] -- the PREVIOUS event's item codec (already
        shifted by the caller; zeros at each learner's t=0). prev_feat: [N, T, 5] (see
        build_prev_feat_v2). Returns e_ep: [N, T, hidden], the query-independent evidence-
        update input to the GRU.
        """
        return self.item_proj(prev_item_codec) + self.prev_proj(prev_feat)


def build_prev_item_codec(item_codec: torch.Tensor) -> torch.Tensor:
    """Shift item_codec by one position along T, zeros at each learner's t=0 -- the same
    shift `build_prev_feat` already applies to (correct, hint_bin, saw_answer), extended to
    the item codec itself. item_codec: [N, T, codec_dim] -> prev_item_codec: [N, T, codec_dim].
    """
    prev = torch.zeros_like(item_codec)
    if item_codec.shape[1] > 1:
        prev[:, 1:] = item_codec[:, :-1]
    return prev


def build_prev_feat_v2(
    correct: torch.Tensor, hint_bin: torch.Tensor, saw_answer: torch.Tensor,
    score_missing: torch.Tensor,
) -> torch.Tensor:
    """Shift (correct, hint_bin, saw_answer, score_missing) by one step, with a start-of-
    sequence flag -- extends `episode.build_prev_feat` with a `prev_score_missing` channel.
    Where the previous step's score was missing, `prev_correct` is zeroed (a placeholder,
    not an observation) and `prev_score_missing=1` marks it as such. All inputs [N, T].
    """
    n, t = correct.shape
    device = correct.device
    prev_correct = torch.zeros(n, t, device=device)
    prev_hint = torch.zeros(n, t, device=device)
    prev_saw = torch.zeros(n, t, device=device)
    prev_valid = torch.zeros(n, t, device=device)
    prev_score_missing = torch.zeros(n, t, device=device)
    if t > 1:
        prev_missing_shifted = score_missing[:, :-1].float()
        prev_correct[:, 1:] = correct[:, :-1].float() * (1.0 - prev_missing_shifted)
        prev_hint[:, 1:] = hint_bin[:, :-1].float()
        prev_saw[:, 1:] = saw_answer[:, :-1].float()
        prev_valid[:, 1:] = 1.0
        prev_score_missing[:, 1:] = prev_missing_shifted
    return torch.stack(
        [prev_correct, prev_hint, prev_saw, prev_valid, prev_score_missing], dim=-1
    )
