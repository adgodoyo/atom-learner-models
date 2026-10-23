"""Prequential ELBO machinery for the belief filter: masked BCE, Gaussian KL terms with
free bits against posterior collapse, and KL annealing.
"""

from __future__ import annotations

import torch


def masked_bce(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
               loss_fn: torch.nn.Module) -> torch.Tensor:
    """Mean BCE over `mask`-selected entries only (shapes must match)."""
    per = loss_fn(logits, target)
    m = mask.float()
    return (per * m).sum() / m.sum().clamp(min=1.0)


def gaussian_kl_free_bits(mu: torch.Tensor, log_sigma: torch.Tensor,
                           free_bits: float) -> torch.Tensor:
    """Per-(...)  KL(N(mu, sigma^2) || N(0, I)), free-bits clamped per dimension before
    summing over the latent axis (last dim). mu/log_sigma: [..., L] -> returns [...].

    Free bits: clamp each dimension's KL to >= free_bits before the sum, so the optimiser
    cannot cheaply shrink an already-near-zero dimension to exactly zero -- the standard
    guard against the KL term collapsing the whole posterior to the prior (the "free bits"
    of Kingma et al. 2016).
    """
    kl_per_dim = 0.5 * (mu.pow(2) + (2 * log_sigma).exp() - 1.0 - 2 * log_sigma)
    kl_per_dim = kl_per_dim.clamp(min=free_bits)
    return kl_per_dim.sum(dim=-1)


def gaussian_kl_free_bits_mu_only(mu: torch.Tensor, log_sigma: torch.Tensor,
                                   free_bits_mu: float) -> torch.Tensor:
    """Splits the per-dim Gaussian KL into a mu-term (0.5*mu^2) and a width-term
    (0.5*(sigma^2-1-2*log_sigma)), and free-bits clamps ONLY the mu-term -- the width term
    is NEVER floored, unlike `gaussian_kl_free_bits` above which clamps their sum.

    Why: a free-bits floor on the summed KL blocks the width term's gradient on log_sigma
    wherever the floor is active. Setting free_bits=0 unblocks it, but then the mu^2 term
    is *also* exposed everywhere with no floor, uniformly shrinking mu; this was measured
    to cost 26% of the identity signal (mu's between-learner variance share). Splitting
    the two terms keeps mu's own floor at its normal free_bits value (protecting identity)
    while leaving the width term as unconstrained as free_bits=0 does for tr(Sigma)
    tracking -- the two objectives are not in tension, only coupled by clamping their
    sum instead of each part."""
    kl_mu = 0.5 * mu.pow(2)
    kl_width = 0.5 * ((2 * log_sigma).exp() - 1.0 - 2 * log_sigma)
    kl_mu_clamped = kl_mu.clamp(min=free_bits_mu)
    return (kl_mu_clamped + kl_width).sum(dim=-1)


def gaussian_kl_two_diagonal(mu_q: torch.Tensor, log_sigma_q: torch.Tensor,
                              mu_p: torch.Tensor, log_sigma_p: torch.Tensor) -> torch.Tensor:
    """KL(N(mu_q, sigma_q^2) || N(mu_p, sigma_p^2)) for two diagonal Gaussians, summed over
    the last (latent) axis. The teacher-posterior regulariser calls this with q = the
    teacher q_psi(z|C,T) (which also sees target answers T) and p = the causal student
    q_phi(z|C) --
    the direction matters: minimising KL(q_psi || q_phi) over many different (C,T) draws
    pulls q_phi to be broad enough to COVER the spread of teacher posteriors it is asked to
    match (mode-covering), which is what produces the calibrated white start without an
    explicit forced target. `gaussian_kl_free_bits` above is the
    special case p = N(0, I); this is the general two-distribution form it does not cover.
    Shapes: mu_q/log_sigma_q/mu_p/log_sigma_p all `[..., L]` -> returns `[...]`."""
    var_q = (2 * log_sigma_q).exp()
    var_p = (2 * log_sigma_p).exp()
    kl_per_dim = (log_sigma_p - log_sigma_q) + (var_q + (mu_q - mu_p).pow(2)) / (2 * var_p) - 0.5
    return kl_per_dim.sum(dim=-1)


def gaussian_kl_two_diagonal_free_bits(mu_q: torch.Tensor, log_sigma_q: torch.Tensor,
                                       mu_p: torch.Tensor, log_sigma_p: torch.Tensor,
                                       free_bits: float) -> torch.Tensor:
    """`gaussian_kl_two_diagonal`, SUMMED over the latent axis then free-bits clamped as
    ONE scalar floor on the total -- the two-distribution generalisation of
    `gaussian_kl_free_bits`'s own per-DIMENSION convention is deliberately NOT reused here:
    Dreamer v2/v3's own published "free nats" (Hafner et al.) floors the TOTAL summed KL at
    a single scalar (their `tf.maximum(kl_value, free_nats)`, applied AFTER the sum over
    the stochastic state), not each latent dimension independently. This matters at
    `latent_dim=256`: a per-dimension 1-nat floor would need every one of 256 dimensions
    to individually exceed 1 nat before ANY gradient flows (a smoke test measured
    `grad_norm_reg=0.0` exactly under the per-dimension form: a dead regulariser), whereas
    the single-scalar floor only requires the SUM across all 256 dimensions to exceed
    1 nat -- true almost immediately once teacher and student diverge at all, matching the
    literature convention.

    Used by the Dreamer-style balanced KL of the teacher-student belief variant:
    `L_dyn = beta_dyn * KL(sg(teacher) || student)` and
    `L_rep = beta_rep * KL(teacher || sg(student))`, each floored independently at the
    1-nat free-bits value before its own beta weighting, so neither term can be cheaply
    driven to zero by collapsing student and teacher onto each other."""
    var_q = (2 * log_sigma_q).exp()
    var_p = (2 * log_sigma_p).exp()
    kl_per_dim = (log_sigma_p - log_sigma_q) + (var_q + (mu_q - mu_p).pow(2)) / (2 * var_p) - 0.5
    kl_total = kl_per_dim.sum(dim=-1)
    return kl_total.clamp(min=free_bits)


def kl_anneal_weight(epoch: int, max_weight: float, anneal_epochs: int) -> float:
    """Linear ramp 0 -> max_weight over `anneal_epochs`, then held at max_weight."""
    if anneal_epochs <= 0:
        return max_weight
    return max_weight * min(1.0, epoch / anneal_epochs)
