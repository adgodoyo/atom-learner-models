"""Tests for the OU forgetting bridge (`atom.filter.ou_kernel`, the OU-kernel ablation)."""

import torch

from atom.filter.ou_kernel import OUKernel


def test_shape_preserved():
    kernel = OUKernel(latent_dim=5)
    mu = torch.randn(3, 4, 5)
    log_sigma = torch.randn(3, 4, 5)
    delta_days = torch.rand(3, 4) * 10
    mu_d, ls_d = kernel(mu, log_sigma, delta_days)
    assert mu_d.shape == mu.shape
    assert ls_d.shape == log_sigma.shape


def test_zero_gap_is_identity():
    """delta_days=0 -> decay=exp(0)=1 -> no change to mu or sigma at all."""
    torch.manual_seed(0)
    kernel = OUKernel(latent_dim=4)
    mu = torch.randn(2, 3, 4)
    log_sigma = torch.randn(2, 3, 4) * 0.1
    delta_days = torch.zeros(2, 3)
    mu_d, ls_d = kernel(mu, log_sigma, delta_days)
    assert torch.allclose(mu_d, mu, atol=1e-6)
    assert torch.allclose(ls_d, log_sigma, atol=1e-5)


def test_large_gap_decays_toward_resting_point_and_stationary_variance():
    torch.manual_seed(1)
    kernel = OUKernel(latent_dim=4)
    with torch.no_grad():
        kernel.log_lambda.fill_(2.0)  # a large-ish rate so decay is near-total by delta=50
    mu = torch.randn(1, 1, 4) * 5
    log_sigma = torch.randn(1, 1, 4)
    delta_days = torch.full((1, 1), 50.0)
    mu_d, ls_d = kernel(mu, log_sigma, delta_days)
    assert torch.allclose(mu_d, kernel.m.unsqueeze(0).unsqueeze(0), atol=1e-3)
    expected_var = torch.nn.functional.softplus(kernel.log_q)
    got_var = (2 * ls_d).exp().squeeze()
    assert torch.allclose(got_var, expected_var, atol=1e-3)


def test_variance_never_negative_across_a_range_of_gaps():
    torch.manual_seed(2)
    kernel = OUKernel(latent_dim=8)
    mu = torch.randn(5, 20, 8)
    log_sigma = torch.randn(5, 20, 8)
    delta_days = torch.rand(5, 20) * 200  # up to ~200 days
    _mu_d, ls_d = kernel(mu, log_sigma, delta_days)
    var_d = (2 * ls_d).exp()
    assert torch.all(var_d > 0)
    assert torch.isfinite(var_d).all()


def test_lambda_and_q_are_positive_regardless_of_raw_parameter_sign():
    kernel = OUKernel(latent_dim=3)
    with torch.no_grad():
        kernel.log_lambda.copy_(torch.tensor([-5.0, 0.0, 5.0]))
        kernel.log_q.copy_(torch.tensor([-5.0, 0.0, 5.0]))
    lam = torch.nn.functional.softplus(kernel.log_lambda)
    q = torch.nn.functional.softplus(kernel.log_q)
    assert torch.all(lam > 0)
    assert torch.all(q > 0)
