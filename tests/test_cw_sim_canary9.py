"""CW-SIM 30-Aug-audit R1: canary family #9 (belief dependence, applied to the
hypernetwork's own `c = hyper(u)` output) + the lambda probe guard (gradient reaches
the filter's mu_head/log_sigma_head). Pure-tensor tests, no real backbone/manifest
needed -- mirrors `tests/test_cw_sim_blocked_eval.py`'s own established split."""
from __future__ import annotations

import importlib.util
import pathlib

import numpy as np
import pytest
import torch
import torch.nn as nn

REPO = pathlib.Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("cw_sim_full_run", REPO / "infra" / "cw_sim_full_run.py")
sim = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sim)


def test_between_learner_variance_share_all_signal_is_one():
    # Each learner has a distinct, noise-free constant value -> ALL variance is between-learner.
    x = np.array([[1.0], [1.0], [5.0], [5.0], [9.0], [9.0]])
    learners = np.array([0, 0, 1, 1, 2, 2])
    share = sim._between_learner_variance_share(x, learners)
    assert share == pytest.approx(1.0, abs=1e-9)


def test_between_learner_variance_share_pure_within_noise_is_zero():
    # Same learner-mean for everyone (0.0) -> zero between-learner variance.
    rng = np.random.default_rng(0)
    x = rng.normal(size=(100, 3))
    x -= x.mean(axis=0, keepdims=True)  # exactly zero grand mean, and per-learner-group means -> 0 in expectation
    learners = rng.integers(0, 5, size=100)
    share = sim._between_learner_variance_share(x, learners)
    assert 0.0 <= share < 0.2  # small, not exactly the noise-free 1.0 case above


class _LinearHyper(nn.Module):
    """A `hyper(u)` stand-in whose output DOES depend on `u`: c = W @ u."""

    def __init__(self, latent_dim: int, m_dim: int, seed: int = 0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.W = nn.Parameter(torch.randn(m_dim, latent_dim, generator=g) * 2.0)
        self.m = m_dim

    def forward(self, u):
        return u @ self.W.T


class _DeafHyper(nn.Module):
    """A `hyper(u)` stand-in whose output IGNORES `u` (a broken/deaf adapter)."""

    def __init__(self, m_dim: int):
        super().__init__()
        self.m = m_dim

    def forward(self, u):
        return torch.zeros(u.shape[0], self.m)


def test_canary9_passes_when_adapter_output_tracks_belief_and_noise_is_small():
    torch.manual_seed(0)
    latent_dim, m_dim, n_learners, per_learner = 4, 6, 5, 8
    hyper = _LinearHyper(latent_dim, m_dim, seed=1)
    # Distinct, well-separated per-learner mu; small log_sigma -> small eps noise.
    learner_group = np.repeat(np.arange(n_learners), per_learner)
    base = torch.arange(n_learners).float().unsqueeze(1) * 5.0  # widely separated per learner
    mu_sel = base.repeat_interleave(per_learner, dim=0).repeat(1, latent_dim)
    mu_sel = mu_sel + 0.01 * torch.randn(n_learners * per_learner, latent_dim)
    log_sigma_sel = torch.full((n_learners * per_learner, latent_dim), -4.0)  # tiny sigma

    out = sim.belief_dependence_canary9({"hyper": hyper}, mu_sel, log_sigma_sel,
                                        learner_group, device="cpu", seed=0, k_eps=8)
    assert out["mu_between_learner_variance_share"] > 0.9
    assert out["across_learner_spread"] > out["across_eps_spread"]
    assert out["pass"] is True


def test_canary9_fails_when_adapter_is_deaf_to_belief():
    torch.manual_seed(0)
    latent_dim, m_dim, n_learners, per_learner = 4, 6, 5, 8
    hyper = _DeafHyper(m_dim)
    learner_group = np.repeat(np.arange(n_learners), per_learner)
    base = torch.arange(n_learners).float().unsqueeze(1) * 5.0
    mu_sel = base.repeat_interleave(per_learner, dim=0).repeat(1, latent_dim)
    log_sigma_sel = torch.zeros(n_learners * per_learner, latent_dim)

    out = sim.belief_dependence_canary9({"hyper": hyper}, mu_sel, log_sigma_sel,
                                        learner_group, device="cpu", seed=0, k_eps=8)
    # A deaf adapter's output never varies at all: both spreads are exactly zero, so
    # the pass rule's strict `>` correctly reports FAIL (mu still carries signal, but
    # nothing downstream of it does -- exactly what this canary exists to catch).
    assert out["across_learner_spread"] == pytest.approx(0.0, abs=1e-9)
    assert out["across_eps_spread"] == pytest.approx(0.0, abs=1e-9)
    assert out["pass"] is False


def test_canary9_fails_when_noise_dominates_signal():
    torch.manual_seed(0)
    latent_dim, m_dim, n_learners, per_learner = 4, 6, 4, 10
    hyper = _LinearHyper(latent_dim, m_dim, seed=2)
    learner_group = np.repeat(np.arange(n_learners), per_learner)
    # Learners barely distinguishable in mu, but log_sigma is huge -> eps noise dominates.
    mu_sel = 0.001 * torch.randn(n_learners * per_learner, latent_dim)
    log_sigma_sel = torch.full((n_learners * per_learner, latent_dim), 4.0)  # sigma ~ 55

    out = sim.belief_dependence_canary9({"hyper": hyper}, mu_sel, log_sigma_sel,
                                        learner_group, device="cpu", seed=0, k_eps=16)
    assert out["across_eps_spread"] > out["across_learner_spread"]
    assert out["pass"] is False


class _FakeFilter(nn.Module):
    def __init__(self, hidden: int, latent_dim: int):
        super().__init__()
        self.mu_head = nn.Linear(hidden, latent_dim)
        self.log_sigma_head = nn.Linear(hidden, latent_dim)


def test_lambda_probe_guard_passes_on_a_real_backward_pass():
    filt = _FakeFilter(hidden=8, latent_dim=4)
    x = torch.randn(3, 8)
    mu = filt.mu_head(x)
    log_sigma = filt.log_sigma_head(x)
    loss = mu.pow(2).sum() + log_sigma.pow(2).sum()
    loss.backward()
    grads = sim.assert_belief_junction_gradient({"filt": filt})
    assert grads["mu_head"] > 0.0
    assert grads["log_sigma_head"] > 0.0


def test_lambda_probe_guard_fails_when_log_sigma_head_never_received_gradient():
    filt = _FakeFilter(hidden=8, latent_dim=4)
    x = torch.randn(3, 8)
    mu = filt.mu_head(x)
    # log_sigma_head is never used in the loss -- reproduces the exact "non-
    # reparameterised draw silently freezes the belief's width" failure mode.
    loss = mu.pow(2).sum()
    loss.backward()
    with pytest.raises(RuntimeError, match="LAMBDA_PROBE_GUARD"):
        sim.assert_belief_junction_gradient({"filt": filt})


def test_lambda_probe_guard_fails_when_no_backward_ran_at_all():
    filt = _FakeFilter(hidden=8, latent_dim=4)
    with pytest.raises(RuntimeError, match="LAMBDA_PROBE_GUARD"):
        sim.assert_belief_junction_gradient({"filt": filt})


class _FakeS1FilterV2ItemMLP(nn.Module):
    """Mirrors the REAL CW-SIM filter's structure (`S1FilterV2ItemMLP` nests its heads
    under `.knowledge_filter`) -- caught a real bug where the guard's first draft
    assumed `mu_head`/`log_sigma_head` sat directly on `m["filt"]`, which is only true
    of the standalone GRU/event-transformer bodies, not this wrapper (found by an actual
    2-step run against the real model, not by these tests, which is exactly why they
    are added now: to pin the fix)."""

    def __init__(self, hidden: int, latent_dim: int, sigma_parameterization: str):
        super().__init__()
        kf = nn.Module()
        kf.mu_head = nn.Linear(hidden, latent_dim)
        if sigma_parameterization == "direct":
            kf.log_sigma_head = nn.Linear(hidden, latent_dim)
        else:
            kf.precision_head = nn.Linear(hidden, latent_dim)
        self.knowledge_filter = kf


@pytest.mark.parametrize("sigma_parameterization", ["direct", "precision_accumulating"])
def test_lambda_probe_guard_passes_on_the_real_nested_filter_structure(sigma_parameterization):
    filt = _FakeS1FilterV2ItemMLP(hidden=8, latent_dim=4, sigma_parameterization=sigma_parameterization)
    x = torch.randn(3, 8)
    kf = filt.knowledge_filter
    width_head = kf.log_sigma_head if sigma_parameterization == "direct" else kf.precision_head
    loss = kf.mu_head(x).pow(2).sum() + width_head(x).pow(2).sum()
    loss.backward()
    grads = sim.assert_belief_junction_gradient({"filt": filt})
    width_name = "log_sigma_head" if sigma_parameterization == "direct" else "precision_head"
    assert grads["mu_head"] > 0.0
    assert grads[width_name] > 0.0


def test_lambda_probe_guard_fails_on_nested_filter_when_precision_head_never_received_gradient():
    filt = _FakeS1FilterV2ItemMLP(hidden=8, latent_dim=4, sigma_parameterization="precision_accumulating")
    x = torch.randn(3, 8)
    loss = filt.knowledge_filter.mu_head(x).pow(2).sum()
    loss.backward()
    with pytest.raises(RuntimeError, match="LAMBDA_PROBE_GUARD"):
        sim.assert_belief_junction_gradient({"filt": filt})
