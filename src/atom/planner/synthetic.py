"""Synthetic positive control for the planner (thesis Section 4.5): a planted-latent world
with a KNOWN answer key, plus an exact-Bayes belief that plugs into the SHIPPED planner
harness unmodified.

**Why this module exists.** On real data nobody knows what the right answer is, so estimator
and harness defects can pass for findings. The control is a world constructed so that the
correct ranking of candidate questions is known in advance, run through `atom.planner.evsi.run_episode`
and `atom.planner.replay.predict_on_targets` **without modification**, so a pass is
evidence about the shipped code path rather than about a reimplementation of it.

**The world.** Each learner has a planted latent theta ~ N(0, I_2) -- the "person". Item j
carries a loading a_j in R^2 and a difficulty b_j, and is answered correctly with
probability sigmoid(a_j . theta + b_j). Three item families, and the whole point of the
control is that two of them carry EXACTLY ZERO information about theta by construction:

  * DIAGNOSTIC  ||a_j|| > 0        -- the answer depends on the person. Informative.
  * ALEATORIC TRAP  a_j = 0, b_j = 0  -- p == 0.5 for EVERY learner. MAXIMUM predictive
    entropy, ZERO information. This is the coin-flip everyone is equally unsure about,
    and the trap that separates a
    decision-aware score (EPIG) from a raw-uncertainty score (max-entropy / BALD flavour).
  * EASY  a_j = 0, b_j > 0         -- p == sigmoid(b_j) for every learner. Low entropy,
    also zero information. Present so that "avoid the trap" cannot be satisfied by simply
    preferring low-entropy items.

Targets are DIAGNOSTIC items disjoint from the candidate pool, so target uncertainty is
genuinely reducible by learning theta -- there is something for EPIG to buy.

**Item parameters are carried BY VALUE in the codec vector, never by an index join**
(metadata carried from source tables, never re-derived through index gymnastics). `s_j = [a_j1, a_j2, b_j, nuisance...]`, and both the
oracle filter and the oracle heads decode (a, b) straight out of the codec they are handed.
There is no lookup table to get out of step with the tensors.

**The oracle belief is query-independent by construction** -- `b_t = f(H_<t)`, the same
property the FA filters are required to satisfy. The posterior at position t is built from
items 0..t-1 and their outcomes only; `item_codec[:, t]` (the probe/query) never enters it.
`OracleFilter` therefore passes the same peek canary as the query-independent FA filter,
exactly (max|d mu| == 0.0), and `assert_query_independent` below is that check.

**What this control does and does not establish.** It tests the ESTIMATOR AND THE HARNESS
on a belief that is Bayes-consistent by construction. It says nothing on its own about
whether K draws from an *amortised* Gaussian belief -- what a trained filter actually
produces -- carry real epistemic information about a person. That is the separate second
arm (`fit_amortised_arm` in scripts/pl_2_positive_control.py), and the distinction is the
one drawn in atom.planner.epig: "the estimator is mathematically sound" and "the belief it
is applied to carries a real signal" are different claims.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

# Codec layout, by value: dims 0..1 = loading a, dim 2 = difficulty b.
A_SLICE = slice(0, 2)
B_INDEX = 2
LATENT_DIM = 2

FAMILIES = ("diagnostic", "trap", "easy")


def make_item_bank(n_diagnostic: int, n_trap: int, n_easy: int, codec_dim: int,
                   diag_loading: float, easy_b: float, nuisance_std: float,
                   rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Returns (codec [M, codec_dim] float32, family_id [M] int -- index into FAMILIES).

    Diagnostic items get a loading of FIXED norm `diag_loading` in a random direction, so
    every diagnostic item is equally informative in expectation and the control cannot be
    passed by luck of one dominant item. Trap items are p == 0.5 EXACTLY (a = 0, b = 0);
    easy items are p == sigmoid(easy_b) exactly (a = 0). Nuisance dims are shared noise
    across all three families so family membership is not linearly trivial from the tail.
    """
    m = n_diagnostic + n_trap + n_easy
    codec = rng.normal(0.0, nuisance_std, size=(m, codec_dim)).astype(np.float32)
    codec[:, A_SLICE] = 0.0
    codec[:, B_INDEX] = 0.0

    family = np.concatenate([
        np.zeros(n_diagnostic, dtype=np.int64),
        np.ones(n_trap, dtype=np.int64),
        np.full(n_easy, 2, dtype=np.int64),
    ])

    if n_diagnostic:
        ang = rng.uniform(0.0, 2 * np.pi, size=n_diagnostic)
        codec[:n_diagnostic, 0] = (diag_loading * np.cos(ang)).astype(np.float32)
        codec[:n_diagnostic, 1] = (diag_loading * np.sin(ang)).astype(np.float32)
    # trap: a = 0, b = 0 -> p == 0.5 exactly (already zeroed above)
    if n_easy:
        codec[n_diagnostic + n_trap:, B_INDEX] = np.float32(easy_b)
    return codec, family


def true_prob(codec: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """p(y=1 | theta, item) for items [M, D] and one learner's theta [2] -> [M].
    Reads (a, b) BY VALUE out of the codec -- the same decode the oracle uses."""
    z = codec[:, A_SLICE] @ theta + codec[:, B_INDEX]
    return 1.0 / (1.0 + np.exp(-z))


def build_grid(half_width: float, n_per_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Product grid over theta in R^2 with a standard-normal prior evaluated on it.
    Returns (grid [G, 2], log_prior [G]) -- log_prior already normalised over the grid, so
    grid coarseness shows up as a measurable approximation, never as a silent bias."""
    ax = torch.linspace(-half_width, half_width, n_per_dim, dtype=torch.float64)
    gx, gy = torch.meshgrid(ax, ax, indexing="ij")
    grid = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=-1)  # [G, 2]
    log_prior = -0.5 * (grid ** 2).sum(-1)
    log_prior = log_prior - torch.logsumexp(log_prior, dim=0)
    return grid, log_prior


class OracleFilter(nn.Module):
    """Exact-Bayes posterior over the planted latent, exposed through the `S1Filter`
    interface (`forward(item_codec [N,T,D], prev_feat [N,T,4]) -> (mu, log_sigma)`), so
    `atom.planner.replay._filter_belief` consumes it with no special-casing.

    The returned (mu, log_sigma) are the exact posterior's MEAN and MARGINAL STANDARD
    DEVIATIONS on the grid -- i.e. a moment-matched diagonal Gaussian, because that is the
    shape `sample_u` draws from and the shape a real filter emits. The moment-match is the
    only approximation in this arm and it is MEASURED, not assumed: see
    `grid_exact_predictive` and the `moment_match_max_abs_dev` metric reported by
    scripts/pl_2_positive_control.py.

    Query-independence (`b_t = f(H_<t)`): the posterior at position t sums
    log-likelihood over positions i < t only. `item_codec[:, t]` -- the probe -- is never
    read into the belief. Checked by `assert_query_independent`.
    """

    def __init__(self, grid: torch.Tensor, log_prior: torch.Tensor):
        super().__init__()
        self.register_buffer("grid", grid)            # [G, 2] float64
        self.register_buffer("log_prior", log_prior)  # [G]   float64
        self.latent_dim = LATENT_DIM

    def forward(self, item_codec: torch.Tensor,
                prev_feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        grid = self.grid
        a = item_codec[..., A_SLICE].to(grid.dtype)          # [N, T, 2]
        b = item_codec[..., B_INDEX].to(grid.dtype)          # [N, T]

        # The observation MADE AT position i is (item i, y_i); y_i is visible one step
        # later as prev_correct[i+1] (build_prev_feat shifts by one). The final position
        # therefore has no observed outcome -- obs_valid is 0 there, by construction.
        prev_correct = prev_feat[..., 0].to(grid.dtype)
        prev_valid = prev_feat[..., 3].to(grid.dtype)
        obs_y = torch.zeros_like(prev_correct)
        obs_valid = torch.zeros_like(prev_valid)
        if prev_correct.shape[-1] > 1:
            obs_y[..., :-1] = prev_correct[..., 1:]
            obs_valid[..., :-1] = prev_valid[..., 1:]

        z = torch.einsum("ntl,gl->ntg", a, grid) + b.unsqueeze(-1)       # [N, T, G]
        ll = obs_y.unsqueeze(-1) * nn.functional.logsigmoid(z) \
            + (1.0 - obs_y).unsqueeze(-1) * nn.functional.logsigmoid(-z)
        ll = ll * obs_valid.unsqueeze(-1)

        # EXCLUSIVE cumulative sum along T: position t sees strictly i < t.
        cum = torch.cumsum(ll, dim=-2)
        cum = torch.cat([torch.zeros_like(cum[..., :1, :]), cum[..., :-1, :]], dim=-2)

        log_post = self.log_prior + cum                                   # [N, T, G]
        log_post = log_post - torch.logsumexp(log_post, dim=-1, keepdim=True)
        w = log_post.exp()                                                # [N, T, G]

        mu = torch.einsum("ntg,gl->ntl", w, grid)                         # [N, T, 2]
        second = torch.einsum("ntg,gl->ntl", w, grid ** 2)
        var = (second - mu ** 2).clamp_min(1e-12)
        return mu.to(item_codec.dtype), (0.5 * var.log()).to(item_codec.dtype)

    def posterior_weights(self, item_codec: torch.Tensor,
                          prev_feat: torch.Tensor) -> torch.Tensor:
        """The full grid posterior [N, T, G] -- what `forward` moment-matches away. Used
        only by `grid_exact_predictive`, to price that moment-match."""
        grid = self.grid
        a = item_codec[..., A_SLICE].to(grid.dtype)
        b = item_codec[..., B_INDEX].to(grid.dtype)
        prev_correct = prev_feat[..., 0].to(grid.dtype)
        prev_valid = prev_feat[..., 3].to(grid.dtype)
        obs_y = torch.zeros_like(prev_correct)
        obs_valid = torch.zeros_like(prev_valid)
        if prev_correct.shape[-1] > 1:
            obs_y[..., :-1] = prev_correct[..., 1:]
            obs_valid[..., :-1] = prev_valid[..., 1:]
        z = torch.einsum("ntl,gl->ntg", a, grid) + b.unsqueeze(-1)
        ll = obs_y.unsqueeze(-1) * nn.functional.logsigmoid(z) \
            + (1.0 - obs_y).unsqueeze(-1) * nn.functional.logsigmoid(-z)
        ll = ll * obs_valid.unsqueeze(-1)
        cum = torch.cumsum(ll, dim=-2)
        cum = torch.cat([torch.zeros_like(cum[..., :1, :]), cum[..., :-1, :]], dim=-2)
        log_post = self.log_prior + cum
        return (log_post - torch.logsumexp(log_post, dim=-1, keepdim=True)).exp()


class OracleHeads(nn.Module):
    """The TRUE likelihood, exposed through the `FiLMHeads` interface
    (`forward(item_codec [N,T,D], u [K,N,T,L]) -> (correctness_logits [K,N,T], engagement)`).
    Reads (a, b) by value from the codec, so heads and world can never disagree."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, item_codec: torch.Tensor,
                u: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        a = item_codec[..., A_SLICE]           # [N, T, 2]
        b = item_codec[..., B_INDEX]           # [N, T]
        logits = (u * a).sum(-1) + b           # [K, N, T]
        engagement = torch.zeros(*logits.shape, 2, dtype=logits.dtype, device=logits.device)
        return logits, engagement


def grid_exact_predictive(oracle: OracleFilter, revealed_item_codec: torch.Tensor,
                          revealed_correct: torch.Tensor, probe_item_codec: torch.Tensor) -> torch.Tensor:
    """p(y=1 | history) under the EXACT grid posterior -- no Gaussian moment-match, no
    K-sampling. Compared against `predict_on_targets`'s `mean_prob` to price the only
    approximation this arm makes. probe_item_codec: [n_probe, D] -> returns [n_probe]."""
    m = revealed_item_codec.shape[0]
    n_probe = probe_item_codec.shape[0]
    d = probe_item_codec.shape[1]
    t_total = m + 1
    dev = probe_item_codec.device

    item_codec = torch.zeros(n_probe, t_total, d, device=dev, dtype=probe_item_codec.dtype)
    correct = torch.zeros(n_probe, t_total, device=dev, dtype=probe_item_codec.dtype)
    if m > 0:
        item_codec[:, :m] = revealed_item_codec.unsqueeze(0)
        correct[:, :m] = revealed_correct.unsqueeze(0)
    item_codec[:, m] = probe_item_codec

    from atom.filter.episode import build_prev_feat
    zeros = torch.zeros_like(correct)
    prev_feat = build_prev_feat(correct, zeros, zeros)
    w = oracle.posterior_weights(item_codec, prev_feat)[:, m]      # [n_probe, G]

    grid = oracle.grid
    a = probe_item_codec[..., A_SLICE].to(grid.dtype)
    b = probe_item_codec[..., B_INDEX].to(grid.dtype)
    z = a @ grid.T + b.unsqueeze(-1)                              # [n_probe, G]
    return (w * torch.sigmoid(z)).sum(-1)


def assert_query_independent(filter_model, item_codec: torch.Tensor, prev_feat: torch.Tensor,
                             probe_position: int, replacement: torch.Tensor) -> float:
    """The peek canary, applied to the control's own belief (the same check the
    query-independent FA filter must pass). Replaces the codec AT `probe_position`
    and asserts the belief there is bit-identical. Returns max|d mu| -- must be EXACTLY
    0.0, not approximately."""
    with torch.no_grad():
        mu_a, ls_a = filter_model(item_codec, prev_feat)
        perturbed = item_codec.clone()
        perturbed[:, probe_position] = replacement
        mu_b, ls_b = filter_model(perturbed, prev_feat)
    d_mu = (mu_a[:, probe_position] - mu_b[:, probe_position]).abs().max().item()
    d_ls = (ls_a[:, probe_position] - ls_b[:, probe_position]).abs().max().item()
    return max(d_mu, d_ls)


def make_learner(rng: np.random.Generator, bank_codec: np.ndarray, theta: np.ndarray,
                 prefix_items: np.ndarray, cand_items: np.ndarray,
                 target_items: np.ndarray) -> dict:
    """One synthetic learner's FROZEN logged history. Every outcome -- prefix, candidate,
    target -- is drawn ONCE from the true model and then never redrawn: the planner
    reveals logged answers, exactly as in the FA replay, so answers come from the
    environment and never from the scorer."""
    p_all = true_prob(bank_codec, theta)
    y_all = (rng.random(bank_codec.shape[0]) < p_all).astype(np.float32)
    return {
        "theta": theta,
        "prefix_items": prefix_items,
        "cand_items": cand_items,
        "target_items": target_items,
        "prefix_y": y_all[prefix_items],
        "cand_y": y_all[cand_items],
        "target_y": y_all[target_items],
        "target_p_true": p_all[target_items],
    }
