"""Collapse diagnostics for the belief filter: is the belief mean mu actually person- and
history-sensitive, or has personalisation silently collapsed to a population average?
"""

from __future__ import annotations

import numpy as np
import torch


def per_learner_mean_error(mask: torch.Tensor, correct: torch.Tensor,
                           probs: torch.Tensor) -> np.ndarray:
    """mask/correct/probs: [N, T]. Returns [N] mean |correct - probs| per learner over
    masked cells (nan where a learner has zero masked cells, e.g. too short a history for
    a given window -- callers should filter these before aggregating further).
    """
    err = torch.abs(correct - probs)
    mask_f = mask.float()
    count = mask_f.sum(dim=1)
    summed = (err * mask_f).sum(dim=1)
    mean_err = summed / count.clamp(min=1)
    mean_err = torch.where(count > 0, mean_err, torch.full_like(mean_err, float("nan")))
    return mean_err.detach().cpu().numpy()


def collapse_diagnostics(mu: torch.Tensor, valid_mask: torch.Tensor,
                         lengths: torch.Tensor) -> dict[str, float]:
    """mu: [N, T, L] pre-observation belief means. Returns two complementary readings.

    cross_learner: std, across the N learners, of each learner's LAST valid mu (their most-
    evidenced belief) -- mean over latent dims. within_learner: for each learner, std of mu
    over their OWN valid t's, averaged over learners and dims. Both the mean AND the full
    distribution of the within-learner reading are returned (percentiles + threshold-count
    caller-side) so a thin tail of near-static learners is visible, not just the aggregate.
    """
    n, _t, latent_dim = mu.shape
    last_idx = (lengths - 1).clamp(min=0)
    last_mu = mu[torch.arange(n, device=mu.device), last_idx]          # [N, L]
    cross_learner_std = last_mu.std(dim=0)                              # [L]

    within_stds = []
    mu_np = mu.detach().cpu().numpy()
    mask_np = valid_mask.detach().cpu().numpy()
    for i in range(n):
        seq = mu_np[i, mask_np[i]]                                      # [len_i, L]
        if seq.shape[0] > 1:
            within_stds.append(seq.std(axis=0))
    within_learner_std = np.stack(within_stds, axis=0) if within_stds else np.zeros((1, latent_dim))
    per_learner = within_learner_std.mean(axis=1)                        # [N]

    return {
        "cross_learner_mu_std_mean": float(cross_learner_std.mean()),
        "cross_learner_mu_std_min": float(cross_learner_std.min()),
        "within_learner_mu_std_mean": float(within_learner_std.mean()),
        "within_learner_mu_std_min_over_learners": float(per_learner.min()),
        "within_learner_mu_std_p5": float(np.percentile(per_learner, 5)),
        "within_learner_mu_std_p25": float(np.percentile(per_learner, 25)),
        "within_learner_mu_std_median": float(np.median(per_learner)),
        "within_learner_mu_std_p95": float(np.percentile(per_learner, 95)),
        "within_learner_per_learner": per_learner,  # not logged directly; threshold counts caller-side
    }
