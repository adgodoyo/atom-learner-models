"""FA-6 gate (iv): spread-error correlation, pure/model-free. See
`atom.eval.spread_error`'s module docstring for the S1-precedent 0.40 bar and the
basis-caveat it carries. `atom.simulator.fa_train_step.k_sample_success_predictions`
(the model-coupled half that produces the (K, N) input this module scores) is tested
separately in `tests/test_fa_train_step.py`, alongside its own filter/backbone fixtures.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from atom.eval.spread_error import (K_SPREAD_DEGENERATE_STD_EPS, SPREAD_ERROR_MAX_P,
                                    SPREAD_ERROR_MIN_CORR, spread_error_correlation)


def test_spread_error_correlation_accepts_numpy_and_torch_identically():
    rng = np.random.default_rng(0)
    probs_np = rng.uniform(0.1, 0.9, size=(8, 50))
    target_np = (rng.uniform(size=50) < 0.5).astype(float)

    out_np = spread_error_correlation(probs_np, target_np)
    out_torch = spread_error_correlation(torch.as_tensor(probs_np), torch.as_tensor(target_np))

    assert math.isclose(out_np["spread_error_corr"], out_torch["spread_error_corr"], rel_tol=1e-9)
    assert out_np["n"] == out_torch["n"] == 50


def test_spread_error_correlation_detects_a_strong_planted_relationship():
    """Construct K-sample predictions where higher spread genuinely co-occurs with higher
    error, and confirm the correlation comes out strongly positive -- the metric must be
    able to detect the relationship it exists to measure, not just run without crashing."""
    rng = np.random.default_rng(1)
    n = 200
    target = (rng.uniform(size=n) < 0.5).astype(float)
    # low-spread examples: all K samples agree AND land near the true label (low error).
    # high-spread examples: K samples disagree widely (high spread) AND their mean drifts
    # far from the true label (high error) -- spread and error move together by construction.
    probs = np.empty((8, n))
    for i in range(n):
        if i % 2 == 0:
            center = target[i] * 0.9 + (1 - target[i]) * 0.1  # confident, near-correct
            probs[:, i] = np.clip(center + rng.normal(scale=0.02, size=8), 0.0, 1.0)
        else:
            probs[:, i] = rng.uniform(0.0, 1.0, size=8)  # wide, uninformative spread

    out = spread_error_correlation(probs, target)
    assert out["k_spread_non_degenerate"] is True
    assert out["spread_error_corr"] > 0.5


def test_spread_error_correlation_flags_degenerate_spread_and_reports_nan_not_a_number():
    """If every K sample predicts the exact same value (spread collapses to ~0 for every
    example), a correlation is not a meaningful statement about anything -- must be flagged
    explicitly, not silently returned as some arbitrary (possibly even high) number."""
    n = 30
    probs = np.tile(np.full(n, 0.5), (8, 1))  # every K-sample identical -> zero spread everywhere
    target = np.zeros(n)

    out = spread_error_correlation(probs, target)
    assert out["k_spread_non_degenerate"] is False
    assert math.isnan(out["spread_error_corr"])
    assert math.isnan(out["spread_error_p"])
    assert out["passed"] is False


def test_spread_error_correlation_gate_uses_the_s1_precedent_bar():
    assert SPREAD_ERROR_MIN_CORR == 0.40
    assert SPREAD_ERROR_MAX_P == 0.01
    assert K_SPREAD_DEGENERATE_STD_EPS > 0


def test_spread_error_correlation_pass_requires_both_corr_and_p_thresholds():
    rng = np.random.default_rng(2)
    n = 500  # large n so a real (if modest) correlation clears the p-value bar too
    target = (rng.uniform(size=n) < 0.5).astype(float)
    base_spread = rng.uniform(0.01, 0.3, size=n)
    point = np.clip(target * (1 - base_spread) + (1 - target) * base_spread, 0.0, 1.0)
    probs = point[None, :] + rng.normal(scale=base_spread / 2, size=(8, n))
    probs = np.clip(probs, 0.0, 1.0)

    out = spread_error_correlation(probs, target)
    if out["spread_error_corr"] >= SPREAD_ERROR_MIN_CORR and out["spread_error_p"] < SPREAD_ERROR_MAX_P:
        assert out["passed"] is True
    else:
        assert out["passed"] is False
