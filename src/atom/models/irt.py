"""2-parameter logistic IRT: P(correct) = sigmoid(a_j * (theta_i - b_j)).

a_j > 0 (discrimination, via softplus of a free parameter so gradient descent can't push it
negative), b_j (difficulty), theta_i (person ability). Fit by full-batch MAP (BCE + Gaussian
priors on theta/log_a/b) with Adam -- the dataset is tiny for a model this size (a handful of
thousand scalar parameters), so no mini-batching is needed.

Two fitting modes, both going through `fit_2pl`:
  - joint: theta, log_a and b all trainable (used once, on TRAIN learners, to get item params
    that generalise);
  - calibration: item params passed in and held fixed (`fixed_log_a`/`fixed_b`), only theta is
    trained -- used per held-out learner, matching how a real IRT/CAT deployment estimates a
    new person's ability from their own responses only.

CPU by default: MPS does not support float64 and this model is far too small to need a GPU.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class IRTParams:
    theta: torch.Tensor    # (n_persons,)
    log_a: torch.Tensor    # (n_items,)
    b: torch.Tensor        # (n_items,)

    @property
    def a(self) -> torch.Tensor:
        return torch.nn.functional.softplus(self.log_a)


def fit_2pl(
    person_idx: torch.Tensor,
    item_idx: torch.Tensor,
    y: torch.Tensor,
    n_persons: int,
    n_items: int,
    epochs: int = 300,
    lr: float = 0.05,
    l2_theta: float = 0.05,
    l2_log_a: float = 0.05,
    l2_b: float = 0.01,
    fixed_log_a: torch.Tensor | None = None,
    fixed_b: torch.Tensor | None = None,
    seed: int = 42,
    device: str = "cpu",
) -> IRTParams:
    """MAP-fit a 2PL model. If fixed_log_a/fixed_b are given, only theta is trained."""
    torch.manual_seed(seed)
    person_idx = person_idx.to(device)
    item_idx = item_idx.to(device)
    y = y.to(device).float()

    theta = nn.Parameter(torch.zeros(n_persons, device=device))
    params = [theta]
    if fixed_log_a is None:
        log_a = nn.Parameter(torch.zeros(n_items, device=device))  # softplus(0) = a≈0.69
        params.append(log_a)
    else:
        log_a = fixed_log_a.to(device)
    if fixed_b is None:
        b = nn.Parameter(torch.zeros(n_items, device=device))
        params.append(b)
    else:
        b = fixed_b.to(device)

    opt = torch.optim.Adam(params, lr=lr)
    bce = nn.BCEWithLogitsLoss()
    for _ in range(epochs):
        opt.zero_grad()
        a = torch.nn.functional.softplus(log_a)
        logit = a[item_idx] * (theta[person_idx] - b[item_idx])
        loss = bce(logit, y) + l2_theta * theta.pow(2).mean()
        if fixed_log_a is None:
            loss = loss + l2_log_a * log_a.pow(2).mean()
        if fixed_b is None:
            loss = loss + l2_b * b.pow(2).mean()
        loss.backward()
        opt.step()

    return IRTParams(theta=theta.detach(), log_a=log_a.detach() if fixed_log_a is None else log_a,
                     b=b.detach() if fixed_b is None else b)


def predict_2pl(params: IRTParams, person_idx: torch.Tensor, item_idx: torch.Tensor) -> torch.Tensor:
    """P(correct) for each (person, item) pair."""
    a = params.a
    logit = a[item_idx] * (params.theta[person_idx] - params.b[item_idx])
    return torch.sigmoid(logit)
