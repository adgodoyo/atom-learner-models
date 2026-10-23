"""Support and success heads read off the personalised backbone (a retired variant).

Thesis Appendix A.3 ("Where estimation lives") reports this design: reading outcome
predictions off the backbone's hidden state scored worse than the base rate on success and
lost to a per-learner frequency table on support, so estimation moved onto the belief
(`fa_hierarchical_heads.OutcomeHead`). These classes are kept for reproducibility.

Both heads read ONLY the personalised backbone's hidden state at the question-prompt's
final token (`h_q`) -- never `u`/`mu`/`sigma` directly (the support and success heads
remain global). Personalisation
reaches them exclusively through the LoRA-hooked backbone forward that produced `h_q`;
this is the same discipline `heads/s2_heads.py` already established for the correctness
head (a head FiLM-conditioned on `u` directly could learn support/success from
personalisation before the binding path is shown to work at all, which would say nothing
about whether weight-space personalisation is the mechanism doing the work).

Deliberately thin (LayerNorm + one Linear): predictive
improvement is meant to come from the personalised backbone, not from a large classifier
that can imitate the belief filter over pooled features.
"""

from __future__ import annotations

import torch
from torch import nn

N_SUPPORT = 4  # NONE, HINT_ONLY, REVEAL_ONLY, HINT_AND_REVEAL --
               # code order fixed by atom.data.fa_manifest.SUPPORT_CLASSES/support_label.


class SupportHead(nn.Module):
    """z^S_t: p_S(s|H_<t,q_t) = softmax(z^S_t)."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.proj = nn.Linear(hidden_dim, N_SUPPORT)

    def forward(self, h_q: torch.Tensor) -> torch.Tensor:
        """h_q: [..., hidden_dim] -> [..., 4] logits."""
        return self.proj(self.norm(h_q))


class SuccessHead(nn.Module):
    """z^Y_{t,s} for s in 0..3: one Bernoulli logit PER support mode.
    Training selects the logit at the OBSERVED support class (the caller's job, see
    `atom.simulator.fa_alt2.success_loss` -- the observed support code is used only to
    SELECT which of the 4 logits is supervised; it is never concatenated into `h_q` or
    fed forward anywhere, which is exactly what the support peek canary requires)."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.proj = nn.Linear(hidden_dim, N_SUPPORT)

    def forward(self, h_q: torch.Tensor) -> torch.Tensor:
        """h_q: [..., hidden_dim] -> [..., 4] logits, one per support mode."""
        return self.proj(self.norm(h_q))
