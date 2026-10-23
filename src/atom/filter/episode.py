"""Episode encoder (fast loop, v1): per-event summary from the current item's codec features
plus the previous step's outcome/behaviour -- never the current step's own outcome (the
same no-peek discipline as src/atom/smoke/model.py::TinyFilter). Superseded by the
query-independent `episode_v2.py`; kept because the v1 filter (`s1.py`) imports it.

FA's event tuples are one row per interaction with no attempt-level sub-structure
(hint_count is already a per-interaction count, not a logged sequence of hint events), so
this reduces to a per-event MLP -- the degenerate case of "one attempt per episode".
CW's attempt-level feedback loops are where this module gains real internal recurrence;
the seam is deliberate so that upgrade doesn't require touching the filter downstream.
"""

from __future__ import annotations

import torch
from torch import nn

PREV_FEAT_DIM = 4  # [prev_correct, prev_hint_bin, prev_saw_answer, prev_valid]


class EpisodeEncoder(nn.Module):
    def __init__(self, codec_dim: int, hidden: int):
        super().__init__()
        self.item_proj = nn.Linear(codec_dim, hidden)
        self.prev_proj = nn.Linear(PREV_FEAT_DIM, hidden)

    def forward(self, item_codec: torch.Tensor, prev_feat: torch.Tensor) -> torch.Tensor:
        """item_codec: [N, T, codec_dim] (current item, known before attempting).
        prev_feat: [N, T, PREV_FEAT_DIM] (previous step's outcome/behaviour; zeros + a
        prev_valid=0 flag at each learner's t=0). Returns e_ep: [N, T, hidden].
        """
        return self.item_proj(item_codec) + self.prev_proj(prev_feat)


def build_prev_feat(correct: torch.Tensor, hint_bin: torch.Tensor,
                     saw_answer: torch.Tensor) -> torch.Tensor:
    """Shift (correct, hint_bin, saw_answer) by one step so position t carries only
    information about t-1, with a start-of-sequence flag. All inputs [N, T] float/bool-like.
    """
    n, t = correct.shape
    device = correct.device
    prev_correct = torch.zeros(n, t, device=device)
    prev_hint = torch.zeros(n, t, device=device)
    prev_saw = torch.zeros(n, t, device=device)
    prev_valid = torch.zeros(n, t, device=device)
    if t > 1:
        prev_correct[:, 1:] = correct[:, :-1].float()
        prev_hint[:, 1:] = hint_bin[:, :-1].float()
        prev_saw[:, 1:] = saw_answer[:, :-1].float()
        prev_valid[:, 1:] = 1.0
    return torch.stack([prev_correct, prev_hint, prev_saw, prev_valid], dim=-1)
