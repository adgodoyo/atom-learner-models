"""Forgetting/staleness as a mean-reverting Ornstein-Uhlenbeck bridge (an ablation; not in
the shipped model):

    mu_z <- m + (mu_z - m) * exp(-lambda * dt)
    sigma_z^2 <- sigma_z^2 * exp(-2*lambda*dt) + Q * (1 - exp(-2*lambda*dt))

Applied at READ time, immediately before the GRU's belief is used to predict an event, using
the elapsed real time since that learner's previous event. Learned per-dimension rate
(lambda) and stationary variance (Q); a learned resting point (m) the belief decays toward
during silence -- "skills decay toward what your traits predict, not to zero", which
properly wants m predicted from a person's trait slice theta; the filter's u is one
undifferentiated vector, so here m is a single, population-level learned vector -- a
deliberate scope narrowing for this ablation (does explicit time-modelling help at all).
A personalised m(theta) is not required to answer that question.

Scope note (also deliberate): this adjusts the belief used for PREDICTION only. The GRU's own
recurrent state is untouched -- future steps do not "remember" that this decay happened. A
version that feeds the decayed belief back into the recurrence (so forgetting compounds
across a gap-heavy history) is a larger commitment, worth making only if this cheaper test
justifies it.

Result (thesis Appendix A.2, `p3_04_ou_ablation`): the kernel improved post-gap calibration
from 0.0183 to 0.0153 ECE and cost 0.061 AUC, and was retired; at these practice densities
(median inter-event gap 1.4 minutes, ~3% of events after a gap longer than a week) there is
little forgetting to model.
"""

from __future__ import annotations

import torch
from torch import nn


class OUKernel(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        self.log_lambda = nn.Parameter(torch.zeros(latent_dim))  # rate, > 0 via softplus
        self.log_q = nn.Parameter(torch.zeros(latent_dim))        # stationary variance, > 0
        self.m = nn.Parameter(torch.zeros(latent_dim))             # learned resting point

    def forward(self, mu: torch.Tensor, log_sigma: torch.Tensor,
                delta_days: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """mu/log_sigma: [..., L]. delta_days: [...] (elapsed days since the learner's own
        previous event; 0 at a sequence's first position -- no history to decay). Returns
        (mu_decayed, log_sigma_decayed), same shapes as the inputs.
        """
        lam = torch.nn.functional.softplus(self.log_lambda)              # [L]
        q = torch.nn.functional.softplus(self.log_q)                      # [L]
        decay = torch.exp(-lam * delta_days.unsqueeze(-1))                 # [..., L], in (0, 1]

        mu_decayed = self.m + (mu - self.m) * decay
        var = (2 * log_sigma).exp()
        var_decayed = var * decay.pow(2) + q * (1 - decay.pow(2))
        log_sigma_decayed = 0.5 * var_decayed.clamp(min=1e-8).log()
        return mu_decayed, log_sigma_decayed
