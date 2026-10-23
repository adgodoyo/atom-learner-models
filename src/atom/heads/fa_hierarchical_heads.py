"""OutcomeHead: P(class | u, s(q)), 5 logits -- the joint outcome head (thesis Section 5.5).

Replaces BOTH `fa_alt2_heads.SupportHead` and `SuccessHead` (kept, unused, for
reproducibility -- see that module). The decisive structural change is not just "5 classes
instead of 4+2": this head reads (item_codec, u) -- the SAME belief-plus-question-codec
input `heads/s1_heads.FiLMHeads` uses -- instead of `h_q`, the personalised backbone's
hidden state. Measured directly at pilot scale, the `h_q` channel lost to a lookup table
(thesis Appendix A.3); this head never goes through the backbone at all, so outcome
prediction no longer depends on (or pays the cost of) a forward pass through the LLM.

FiLM pattern copied from `heads/s1_heads.FiLMHeads` (proven, not reinvented): near-identity
init so an untrained hypernetwork-derived `u` barely perturbs the item trunk at step 0.
Shape convention matches the simulator's per-ROW training loop (`fa_train_step.py`'s
`gather_belief_for_batch` + `reparameterize(..., k=1)[:, 0, :]`), NOT the filter heads' per-sequence
[N,T,...] convention -- item_codec: [N, codec_dim], u: [N, latent_dim] -> logits: [N, 5].

`n_classes` (the CodeWorkout instantiation) generalises the class
count -- default `N_OUTCOME` (5) keeps every FA call site byte-for-byte unaffected;
CodeWorkout passes `n_classes=4` (`cw_outcome_label.N_OUTCOME`) for its own
{compile-error, clean-zero, clean-partial, clean-full} label. The FiLM trunk/gamma/beta
machinery is class-count-agnostic by construction -- only the final linear layer's
output width changes.

`prefix_dim`: CodeWorkout's per-attempt design needs a THIRD input -- the within-episode causal prefix state from
`cw_attempt_encoder.AttemptEncoder` (attempt k's prediction reads the state entering
step k, never attempt k's own inputs; see that module's docstring for the leak this
avoids). The three-input head is one head, not a new one -- the head-economy rule is
about gradient conflict on a SHARED representation, not parameter count, and widening
one head's input adds neither. `prefix_dim=None` (default) omits the third projection
entirely and keeps every existing call -- `OutcomeHead(...)` then `head(item_codec, u)`
-- byte-for-byte identical (FA never constructs or passes a prefix). CodeWorkout constructs
`OutcomeHead(..., prefix_dim=attempt_hidden)` and always calls
`head(item_codec, u, prefix=h_k_prefix)`.
"""

from __future__ import annotations

import torch
from torch import nn

from atom.data.fa_outcome_label import N_OUTCOME


class OutcomeHead(nn.Module):
    def __init__(self, codec_dim: int, latent_dim: int, hidden: int, n_classes: int = N_OUTCOME,
                 prefix_dim: int | None = None):
        super().__init__()
        self.trunk = nn.Linear(codec_dim, hidden)
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
        """item_codec: [N, codec_dim]. u: [N, latent_dim]. `prefix`: [N, prefix_dim],
        required iff this head was constructed with `prefix_dim` set (raises loudly if
        the two disagree, in either direction -- a silently-ignored prefix or a missing
        one are both the exact leak/omission this head exists to prevent). Returns
        outcome_logits [N, n_classes]."""
        if self.prefix_proj is not None and prefix is None:
            raise ValueError("this head was constructed with prefix_dim set; forward() needs `prefix`")
        if self.prefix_proj is None and prefix is not None:
            raise ValueError("this head has no prefix_dim configured; got an unexpected `prefix`")
        trunk_pre = self.trunk(item_codec)         # [N, H]
        if self.prefix_proj is not None:
            trunk_pre = trunk_pre + self.prefix_proj(prefix)  # additive BEFORE the nonlinearity,
            # so item_codec and the causal prefix are two contributions to ONE shared hidden
            # representation rather than the prefix bypassing the trunk's own activation.
        f = torch.relu(trunk_pre)                   # [N, H]
        gamma = 1.0 + self.gamma(u)               # [N, H]
        beta = self.beta(u)                        # [N, H]
        personalised = torch.relu(gamma * f + beta)  # [N, H]
        return self.outcome(personalised)             # [N, n_classes]
