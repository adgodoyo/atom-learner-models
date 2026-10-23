"""Correctness/engagement heads for the belief filter, FiLM-conditioned on a sampled person
latent u -- the cheap stand-in for "hypernetwork -> LoRA-basis -> frozen-backbone forward"
when there is no backbone in the loop (cached codec features only). Personalisation still
flows through a *sample* of the belief, never through (mu, Sigma) directly, so the K-sample
machinery and the person-epistemic-spread reading ("spread across samples = person-epistemic
uncertainty") carry over unchanged when the hypernetwork replaces FiLM.

FiLM affine is initialised near-identity (gamma ~= 1, beta ~= 0): at init a fresh sample of
u should barely perturb the shared item trunk, so early training is dominated by "what does
this item look like" (the trunk) before personalisation differentiates learners -- avoids
FiLM immediately drowning the trunk signal in noise from an untrained hypernetwork-analogue.
Small-std (not exactly zero) weight init: an exact-zero gamma/beta weight matrix makes
d(gamma)/d(u) = d(beta)/d(u) = 0 at step 0, which would zero the gradient path back into u
(and hence into the filter) on the very first backward pass -- small noise avoids that dead
start while keeping FiLM's initial effect negligible.
"""

from __future__ import annotations

import torch
from torch import nn

N_ENGAGEMENT = 2  # [p(hint), p(saw_answer)]


class FiLMHeads(nn.Module):
    def __init__(self, codec_dim: int, latent_dim: int, hidden: int):
        super().__init__()
        self.trunk = nn.Linear(codec_dim, hidden)
        self.gamma = nn.Linear(latent_dim, hidden)
        self.beta = nn.Linear(latent_dim, hidden)
        nn.init.normal_(self.gamma.weight, std=0.01)
        nn.init.zeros_(self.gamma.bias)
        nn.init.normal_(self.beta.weight, std=0.01)
        nn.init.zeros_(self.beta.bias)
        self.correctness = nn.Linear(hidden, 1)
        self.engagement = nn.Linear(hidden, N_ENGAGEMENT)

    def forward(self, item_codec: torch.Tensor,
                u: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """item_codec: [N, T, codec_dim]. u: [K, N, T, latent_dim] (K samples of the
        belief). Returns (correctness_logits [K, N, T], engagement_logits [K, N, T, 2]).
        """
        f = torch.relu(self.trunk(item_codec))       # [N, T, H]
        f = f.unsqueeze(0)                            # [1, N, T, H] -- broadcast over K
        gamma = 1.0 + self.gamma(u)                   # [K, N, T, H]
        beta = self.beta(u)                            # [K, N, T, H]
        personalised = torch.relu(gamma * f + beta)    # [K, N, T, H]
        correctness_logits = self.correctness(personalised).squeeze(-1)  # [K, N, T]
        engagement_logits = self.engagement(personalised)                # [K, N, T, 2]
        return correctness_logits, engagement_logits
