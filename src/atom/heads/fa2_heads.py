"""Heads for the filter-body comparison: FiLM-conditioned correctness + the 4-class SUPPORT head.

WHY A SUPPORT HEAD IS IN THE COMPARISON. A correctness-only court is systematically biased
against body (c) (`filter/event_transformer.py`): its claimed advantage is attention over
semantic distance and elapsed time -- "what has this learner done near this question, and
how recently" -- which plausibly pays far more in PRODUCTION than in a single correctness
bit. Support is one of the three output families the FA simulator emits, and it needs no
backbone, so widening the court to include it is free.

WHAT THE SUPPORT LABEL IS, STATED PLAINLY. `fa_manifest.support_label` builds
`support_code = (hint_count > 0) + 2 * saw_answer`, i.e. NONE / HINT_ONLY / REVEAL_ONLY /
HINT_AND_REVEAL. It is therefore the JOINT of the two channels `s1_heads.FiLMHeads` already
predicts as independent marginals, not new information. That is the point rather than a
caveat: the simulator's output family is the joint 4-class categorical, which models the dependency
between asking for a hint and revealing the answer (empirically strong -- the marginals give
P(HINT)*P(REVEAL) = 0.052 * 0.218 = 1.1% for HINT_AND_REVEAL against a measured 4.0%). So
this head REPLACES the two independent engagement BCEs rather than sitting alongside them;
having two heads predict the same thing twice would be accidental complexity, and the joint
is the one the simulator ships. The consequence for comparability: the GRU arm here is
retrained under this objective, so its correctness AUC is a reference against the
correctness-only filter rather than an identity check.

WHY A SUBCLASS THAT REDECLARES THE TRUNK. `FiLMHeads` is imported and subclassed for its
parameters and its near-identity FiLM initialisation (the reasoning in its own docstring
applies unchanged), but `forward` is overridden rather than refactoring a shared
`_personalise` helper into the parent. Eight duplicated lines are the price of leaving
`s1_heads.py` byte-identical for the scripts that still load checkpoints against it, with
the existing tests passing unedited.
The returned tuple is a superset of the parent's first element, so the correctness path is
the same computation, not a re-implementation of it.
"""

from __future__ import annotations

import torch
from torch import nn

from atom.data.fa_manifest import SUPPORT_CLASSES
from atom.heads.s1_heads import FiLMHeads

N_SUPPORT = len(SUPPORT_CLASSES)  # 4 -- NONE / HINT_ONLY / REVEAL_ONLY / HINT_AND_REVEAL


class FiLMHeadsSupport(FiLMHeads):
    def __init__(self, codec_dim: int, latent_dim: int, hidden: int):
        super().__init__(codec_dim, latent_dim, hidden)
        # The two independent engagement marginals are SUBSUMED by the joint support head
        # (see the module docstring), so the inherited head is removed rather than left as
        # dead parameters that would sit in every checkpoint and in the head parameter count.
        del self.engagement
        self.support = nn.Linear(hidden, N_SUPPORT)

    def forward(self, item_codec: torch.Tensor,
                u: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """item_codec: [N, T, codec_dim] (the CURRENT item -- the emission path, where the
        query legitimately enters). u: [K, N, T, latent_dim]. Returns
        (correctness_logits [K, N, T], support_logits [K, N, T, 4]).
        """
        f = torch.relu(self.trunk(item_codec)).unsqueeze(0)   # [1, N, T, H], broadcast over K
        personalised = torch.relu((1.0 + self.gamma(u)) * f + self.beta(u))
        return self.correctness(personalised).squeeze(-1), self.support(personalised)
