"""Backbone-read correctness/engagement head: reads the personalised backbone's hidden state.

Design decision this module encodes: personalisation reaches this
head ONLY through the frozen backbone's own weights (hypernetwork -> LoRA-basis -> hooked
q/v projections), never through a second direct conditioning on `u` at the head itself.
Concretely: the head reads a single pooled hidden vector produced by a personalised
backbone forward pass -- it never sees `u`, `mu`, or `sigma`. This is not a simplification;
it is the point. If the head were FiLM-conditioned on `u` directly (as `s1_heads.FiLMHeads`
is), gradients from the correctness loss could reach a good prediction without ever
touching the basis or hypernetwork, and the resulting AUC would say nothing about whether
the mechanism this thesis proposes (weight-space personalisation) works at all -- it would
just be the FiLM filter heads with an extra, redundant `u`-conditioned readout bolted on.

Two channels: engagement -> P(give-up), correctness -> P(solve | attempted), because on
FoundationalASSIST a hint or answer reveal forfeits credit, so the recorded `correct` label
is the product of two different events. The CALLER is responsible for masking the
correctness loss to `saw_answer == 0` rows and for forming the product
`(1 - p_giveup) * p_solve` scored against the observed `correct` label -- this module only
supplies the two logits, it does not know about masking or the
product, matching FiLMHeads' own division of labour (heads emit logits, callers own the
loss/scoring protocol).

Deliberately small (one shared hidden layer, two linear output heads), as the filter heads
are thin: the head tests WHETHER a correctness signal survives the hypernetwork -> basis ->
backbone path at all. A larger head could
memorise its way to a plausible AUC even with a broken binding path, defeating the point of
the gate.
"""

from __future__ import annotations

import torch
from torch import nn


class S2CorrectnessHead(nn.Module):
    def __init__(self, hidden_dim: int, head_hidden: int = 64):
        super().__init__()
        self.trunk = nn.Linear(hidden_dim, head_hidden)
        self.correctness = nn.Linear(head_hidden, 1)  # P(solve | attempted)
        self.engagement = nn.Linear(head_hidden, 1)   # P(give-up)

    def forward(self, pooled_hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """pooled_hidden: [..., hidden_dim] -- the personalised backbone's last-hidden-layer
        state at the final PROMPT-token position (never a completion-token position; the
        caller is responsible for that pooling choice; reading a completion token would let
        the head peek at the answer). Returns (correctness_logit, engagement_logit),
        each [...], i.e. the trailing hidden_dim axis is consumed and no axis is added --
        callers manage any leading K-sample axis themselves (mean = prediction, spread =
        person-epistemic uncertainty, per `atom.binding.uncertainty`), matching how
        `reparameterize`/`person_epistemic_spread` are used elsewhere in the binding code.
        """
        f = torch.relu(self.trunk(pooled_hidden))
        return self.correctness(f).squeeze(-1), self.engagement(f).squeeze(-1)


def product_probability(correctness_logit: torch.Tensor,
                        engagement_logit: torch.Tensor) -> torch.Tensor:
    """The two-process rule made executable: observed P(correct) is scored as
    (1 - P(give-up)) * P(solve | attempted), never the correctness channel alone -- the
    correctness channel by itself only means P(solve | attempted) once its OWN loss is
    masked to `saw_answer == 0` rows by the caller (this function does not mask; it only
    combines two already-independent probabilities into the one the observed label scores
    against)."""
    p_solve = torch.sigmoid(correctness_logit)
    p_giveup = torch.sigmoid(engagement_logit)
    return (1.0 - p_giveup) * p_solve
