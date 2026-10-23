"""Spread-error gate for K-sample person uncertainty, and the K-spread itself.

Pure and generic, with no model coupling: it takes (K, N) predicted-probability arrays and
an (N,) observed-outcome array and produces the correlation, however those K-sample
predictions were produced. The wiring that obtains K-sample success predictions from the
personalised backbone lives in `atom.simulator.fa_train_step.k_sample_success_predictions`,
which calls into this.

Precedent: the outcome-only filter measured a spread-error Spearman correlation of
**0.40** over FiLM-conditioned cached 45-d codec features. The bar here is that same 0.40,
with a **basis caveat**: the filter's spread is computed over cached features, whereas the
simulator's spread is computed through the full personalised backbone/LoRA-basis path, a
different geometric basis for the same nominal quantity. So 0.40 is a target by precedent,
not a like-for-like equivalence. Report the basis alongside the number, never the number
alone.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import spearmanr

from atom.binding import mean_prediction, person_epistemic_spread

SPREAD_ERROR_MIN_CORR = 0.40  # the outcome-only filter's measured value, re-used as the
                              # bar (see the module docstring's basis caveat); a
                              # precedent, not derived from first principles.
SPREAD_ERROR_MAX_P = 0.01
K_SPREAD_DEGENERATE_STD_EPS = 1e-6  # below this, K-sample spread is numerically flat


def _to_numpy(x: Any) -> np.ndarray:
    try:
        import torch

        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except ImportError:
        pass
    return np.asarray(x)


def spread_error_correlation(probs_k: Any, target: Any, dim: int = 0) -> dict[str, Any]:
    """`probs_k`: (K, N) predicted probabilities, one row per K-sample draw of the person
    latent `u` (K along `dim`, matching `atom.binding.reparameterize`'s own convention);
    `target`: (N,) observed 0/1 outcomes. Point prediction = K-mean, K-spread =
    K-std (population, unbiased=False -- `person_epistemic_spread`'s own convention, right
    for the small K used here: 1-2 train, 8 eval). Returns the Spearman
    correlation between K-spread and |error| (the "spread-error" gate number), plus a
    `non_degenerate` flag: if every example's K-spread collapsed to (near-)zero, K-sampling
    isn't actually producing different predictions and the correlation number is not
    meaningful regardless of its value -- checked explicitly rather than left for a reader
    to notice a suspiciously tight correlation.
    """
    t = _to_numpy(target).ravel().astype(float)
    probs_t = _as_tensor_like(probs_k)
    point = _to_numpy(mean_prediction(probs_t, dim=dim))
    spread = _to_numpy(person_epistemic_spread(probs_t, dim=dim))
    error = np.abs(t - point)

    non_degenerate = bool(np.std(spread) > K_SPREAD_DEGENERATE_STD_EPS)
    if non_degenerate:
        corr, p = spearmanr(spread, error)
        corr, p = float(corr), float(p)
    else:
        corr, p = float("nan"), float("nan")

    passed = bool(non_degenerate and corr >= SPREAD_ERROR_MIN_CORR and p < SPREAD_ERROR_MAX_P)
    return {
        "spread_error_corr": corr, "spread_error_p": p,
        "k_spread_mean": float(spread.mean()), "k_spread_non_degenerate": non_degenerate,
        "passed": passed, "n": int(len(t)),
    }


def _as_tensor_like(x: Any):
    """`mean_prediction`/`person_epistemic_spread` are torch-native (`.mean`/`.std`); this
    keeps `spread_error_correlation` accepting plain numpy too without duplicating their
    reduction logic."""
    import torch

    return x if isinstance(x, torch.Tensor) else torch.as_tensor(np.asarray(x))
