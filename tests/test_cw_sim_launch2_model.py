"""`infra.cw_sim_launch2_model.held_out_class_posterior_batch`: K-marginalised class posterior for held-out-item rows,
against a REAL (tiny-dim) `OutcomeHeadItemMLP`, verified against an explicit
per-row-per-K reference loop -- not just shape/range checks."""
from __future__ import annotations

import importlib.util
import pathlib

import numpy as np
import torch

REPO = pathlib.Path(__file__).resolve().parents[1]
import sys  # noqa: E402
sys.path.insert(0, str(REPO / "src"))

_spec = importlib.util.spec_from_file_location(
    "cw_sim_launch2_model", REPO / "infra" / "cw_sim_launch2_model.py")
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)

from atom.simulator.cw_item_mlp import OutcomeHeadItemMLP  # noqa: E402


def _tiny_head(seed=0):
    torch.manual_seed(seed)
    return OutcomeHeadItemMLP(codec_dim=6, latent_dim=4, hidden=8, n_classes=4, prefix_dim=5)


def test_output_shape_and_probability_simplex():
    head = _tiny_head()
    n = 7
    item = torch.randn(n, 6)
    mu = torch.randn(n, 4)
    log_sigma = torch.full((n, 4), -1.0)
    out = mod.held_out_class_posterior_batch(head, item, mu, log_sigma, k=5, seed=0, device="cpu")
    assert out.shape == (n, 4)
    np.testing.assert_allclose(out.sum(dim=-1).numpy(), np.ones(n), atol=1e-5)
    assert (out >= 0).all()


def test_deterministic_given_same_seed():
    head = _tiny_head()
    item = torch.randn(3, 6)
    mu = torch.randn(3, 4)
    log_sigma = torch.full((3, 4), -0.5)
    a = mod.held_out_class_posterior_batch(head, item, mu, log_sigma, k=8, seed=42, device="cpu")
    b = mod.held_out_class_posterior_batch(head, item, mu, log_sigma, k=8, seed=42, device="cpu")
    np.testing.assert_array_equal(a.numpy(), b.numpy())


def test_different_seeds_give_different_results():
    head = _tiny_head()
    item = torch.randn(3, 6)
    mu = torch.randn(3, 4)
    log_sigma = torch.full((3, 4), -0.1)
    a = mod.held_out_class_posterior_batch(head, item, mu, log_sigma, k=4, seed=1, device="cpu")
    b = mod.held_out_class_posterior_batch(head, item, mu, log_sigma, k=4, seed=2, device="cpu")
    assert not torch.allclose(a, b)


def test_matches_an_explicit_mean_of_softmax_reference_never_softmax_of_mean():
    head = _tiny_head()
    n, k = 4, 6
    item = torch.randn(n, 6) * 3.0
    mu = torch.randn(n, 4) * 3.0
    log_sigma = torch.full((n, 4), 1.5)  # large sigma -> the K draws spread widely
    out = mod.held_out_class_posterior_batch(head, item, mu, log_sigma, k=k, seed=7, device="cpu")

    gen = torch.Generator(device="cpu").manual_seed(7)
    ref = torch.zeros(n, 4)
    with torch.no_grad():
        for _ in range(k):
            eps = torch.randn(mu.shape, generator=gen)
            u = mu + torch.exp(log_sigma) * eps
            ref += torch.softmax(head(item, u, prefix=torch.zeros(n, 5)), dim=-1)
    ref /= k
    torch.testing.assert_close(out, ref)

    # Confirm it is NOT softmax-of-mean-u (a different, wrong quantity under a
    # nonlinear head) -- at this variance the two should clearly disagree.
    with torch.no_grad():
        wrong = torch.softmax(head(item, mu, prefix=torch.zeros(n, 5)), dim=-1)
    assert not torch.allclose(out, wrong, atol=1e-3)


def test_zero_prefix_used_when_prefix_dim_is_set():
    """Clause 5's own convention: a fixed-zero prefix for a genuine first attempt --
    this test just confirms the function does not crash/mismatch shape when the head
    was constructed WITH a prefix_dim (the production configuration)."""
    head = _tiny_head()
    out = mod.held_out_class_posterior_batch(
        head, torch.randn(2, 6), torch.randn(2, 4), torch.zeros(2, 4), k=2, seed=0, device="cpu")
    assert out.shape == (2, 4)
