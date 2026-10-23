"""Slow loop (thesis Section 5.3): a GRU consumes the episode-summary sequence and emits
the belief b_t = N(mu_t, sigma_t^2) BEFORE step t's own outcome is folded in -- e_ep[t]
only carries the current item's (known-in-advance) codec and t-1's outcome/behaviour
(src/atom/filter/episode.py), so the GRU's output at position t is a legitimate
pre-observation belief, the same prequential discipline as the smoke-test TinyFilter and
the IRT calibration. (With `episode_v2.py` the episode summary carries t-1's item instead,
so the belief is query-independent.) Log-sigma is clamped for numerical stability (no magic
number left implicit: bounds are constructor args, defaulted in configs/s1_filter.yaml).

This is body (a), the GRU. Body (c), the causal event-transformer, is a separate module
(`event_transformer.py`) behind the same interface.

`sigma_parameterization` selects how the width is emitted: "direct" (log sigma read freely
from the GRU state) or "precision_accumulating" (precision is 1 plus a cumulative sum of
non-negative per-event contributions, so the belief can only narrow as evidence arrives;
thesis Appendix A.2).
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from atom.binding.uncertainty import reparameterize_from_log_sigma


class KnowledgeFilter(nn.Module):
    def __init__(self, input_dim: int, hidden: int, latent_dim: int,
                 log_sigma_min: float = -5.0, log_sigma_max: float = 2.0,
                 sigma_parameterization: str = "direct"):
        super().__init__()
        if sigma_parameterization not in ("direct", "precision_accumulating"):
            raise ValueError(
                "sigma_parameterization must be 'direct' or 'precision_accumulating', "
                f"got {sigma_parameterization!r}"
            )
        self.gru = nn.GRU(input_dim, hidden, batch_first=True)
        self.mu_head = nn.Linear(hidden, latent_dim)
        if sigma_parameterization == "direct":
            # Construction and state-dict keys are those of the original direct form.
            self.log_sigma_head = nn.Linear(hidden, latent_dim)
        else:
            # One non-negative precision contribution per observed event.  The event
            # encoder is the correct input: e_ep[:, t] describes event t-1, so the
            # cumulative sum at belief position t contains exactly H_<t.
            self.precision_head = nn.Linear(input_dim, latent_dim)
        self.log_sigma_min = log_sigma_min
        self.log_sigma_max = log_sigma_max
        self.sigma_parameterization = sigma_parameterization

    def forward(self, e_ep: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """e_ep: [N, T, input_dim]. Returns (mu, log_sigma), each [N, T, latent_dim]."""
        h, _ = self.gru(e_ep)
        mu = self.mu_head(h)
        if self.sigma_parameterization == "direct":
            log_sigma = self.log_sigma_head(h).clamp(self.log_sigma_min, self.log_sigma_max)
        else:
            increments = F.softplus(self.precision_head(e_ep))
            # Position zero is the prior: e_ep[:, 0] is only the START placeholder, not
            # an observed event.  Every later position adds the encoding of one newly
            # observed event, making precision monotone by construction.
            increments = torch.cat([torch.zeros_like(increments[:, :1]), increments[:, 1:]], dim=1)
            precision = 1.0 + torch.cumsum(increments, dim=1)
            log_sigma = (-0.5 * torch.log(precision)).clamp(
                self.log_sigma_min, self.log_sigma_max
            )
        return mu, log_sigma


def sample_u(mu: torch.Tensor, log_sigma: torch.Tensor, k: int,
            noise: torch.Tensor | None = None) -> torch.Tensor:
    """Reparameterised samples u = mu + sigma*eps. mu/log_sigma: [..., L] -> [K, ..., L].

    K=1-2 during training (the reparameterised path gradients flow through), K=8 at
    inference (person-epistemic spread = std across this axis).

    `noise` (common random numbers): optional pre-drawn standard-normal draws,
    shape [K, L]. When given, the SAME [K, L] pattern is broadcast over every leading dim
    of mu/log_sigma instead of drawing fresh eps -- reusing one frozen noise draw across
    many calls (e.g. every candidate evaluated for one learner) removes independent-
    sampling noise from comparisons BETWEEN those calls, without touching the reparameterised-
    sampling semantics for any single call. Default (None) is the original behaviour.

    A thin layout wrapper over `atom.binding.uncertainty.reparameterize_from_log_sigma`,
    the ONE place `.exp()` is applied to a filter's raw `log_sigma` output, so that no
    caller can sample with log sigma mistaken for sigma. The only work done here is
    reshaping this function's leading-K convention to/from `reparameterize_from_log_sigma`'s
    trailing-before-L convention.
    """
    extra_dims = mu.dim() - 1
    eps = None if noise is None else noise.view(*([1] * extra_dims), k, noise.shape[-1])
    u = reparameterize_from_log_sigma(mu, log_sigma, k, eps=eps)  # [..., k, L]
    return u.movedim(-2, 0)  # -> [k, ...]
