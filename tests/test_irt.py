"""IRT-2PL recovery tests on synthetic data with a known generative model."""

import numpy as np
import torch

from atom.eval.metrics import auc
from atom.models.irt import fit_2pl, predict_2pl


def _synthetic(n_persons=60, n_items=30, seed=0):
    rng = np.random.default_rng(seed)
    theta_true = rng.normal(0, 1, n_persons)
    a_true = np.exp(rng.normal(0, 0.3, n_items))  # positive, centred near 1
    b_true = rng.normal(0, 1, n_items)
    pi = rng.integers(0, n_persons, 4000)
    ii = rng.integers(0, n_items, 4000)
    p = 1 / (1 + np.exp(-a_true[ii] * (theta_true[pi] - b_true[ii])))
    y = rng.binomial(1, p)
    return pi, ii, y, theta_true, a_true, b_true, p


def test_joint_fit_recovers_high_insample_auc():
    pi, ii, y, *_ = _synthetic()
    params = fit_2pl(torch.tensor(pi), torch.tensor(ii), torch.tensor(y),
                     n_persons=60, n_items=30, epochs=400)
    pred = predict_2pl(params, torch.tensor(pi), torch.tensor(ii)).numpy()
    assert auc(y, pred) > 0.75  # the true generative model itself is imperfectly separable


def test_joint_fit_recovers_item_difficulty_ordering():
    pi, ii, y, theta_true, a_true, b_true, _ = _synthetic()
    params = fit_2pl(torch.tensor(pi), torch.tensor(ii), torch.tensor(y),
                     n_persons=60, n_items=30, epochs=400)
    b_hat = params.b.numpy()
    # Harder items (higher true b) should get higher fitted b -- rank correlation, not exact.
    corr = np.corrcoef(b_true, b_hat)[0, 1]
    assert corr > 0.6


def test_calibration_only_fits_theta_with_items_frozen():
    pi, ii, y, theta_true, a_true, b_true, _ = _synthetic(n_persons=80)
    # "Train" item params on the first 60 persons.
    train_mask = pi < 60
    item_params = fit_2pl(torch.tensor(pi[train_mask]), torch.tensor(ii[train_mask]),
                          torch.tensor(y[train_mask]), n_persons=60, n_items=30, epochs=400)

    # Calibrate theta for held-out persons 60-79 using ONLY their responses, items frozen.
    rng = np.random.default_rng(1)
    held_pi = rng.integers(60, 80, 1500) - 60  # re-index to 0..19 for the calib person table
    held_ii = rng.integers(0, 30, 1500)
    p_true = 1 / (1 + np.exp(-a_true[held_ii] * (theta_true[held_pi + 60] - b_true[held_ii])))
    held_y = rng.binomial(1, p_true)

    calib = fit_2pl(torch.tensor(held_pi), torch.tensor(held_ii), torch.tensor(held_y),
                    n_persons=20, n_items=30, epochs=300,
                    fixed_log_a=item_params.log_a, fixed_b=item_params.b)
    # theta is trainable, item params must be exactly unchanged (frozen).
    assert torch.equal(calib.log_a, item_params.log_a)
    assert torch.equal(calib.b, item_params.b)

    theta_hat = calib.theta.numpy()
    theta_true_held = theta_true[60:80]
    corr = np.corrcoef(theta_true_held, theta_hat)[0, 1]
    assert corr > 0.5  # recovered ability tracks the true (unseen-during-item-fit) ability
