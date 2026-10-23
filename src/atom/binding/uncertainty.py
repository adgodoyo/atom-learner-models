"""K-sample machinery for the belief b_t = N(mu_t, Sigma_t) and the person-epistemic uncertainty slice.

Draws K samples via the reparameterisation trick (the path gradients flow through from
every K-sample prediction back into the belief), and provides the person-epistemic slice
of the three-way uncertainty decomposition: mean across K = prediction; spread across K =
person-epistemic uncertainty.

The other two parts of the three-way split are NOT this module's job: aleatoric is
entropy WITHIN one sample's own predictive distribution (head-specific -- computed by
whichever head produces that distribution, e.g. a Bernoulli's own entropy for correctness);
law-epistemic is spread across independently-seeded REPLICAS of {basis+hypernet+filter}
(a training-time concern -- multiple full models, not multiple samples from one).
"""

from __future__ import annotations

import torch


def reparameterize(mu: torch.Tensor, sigma: torch.Tensor, k: int,
                   generator: torch.Generator | None = None) -> torch.Tensor:
    """mu, sigma: (..., d) -> u samples: (..., k, d), u = mu + sigma * eps, eps ~ N(0,I).

    Differentiable w.r.t. mu and sigma (standard reparameterisation trick) -- this is the
    only place noise enters the sampling, so gradients from every downstream K-sample
    prediction flow back into the belief (mu, sigma) through here.

    Takes SIGMA, not log-sigma -- every filter body's `KnowledgeFilter`/`S1FilterV2C`
    forward returns `(mu, log_sigma)` (`gru.py`, `event_transformer.py`), so a caller
    holding a filter's raw output must exponentiate before calling this. Callers that
    already hold true sigma (scripts that apply `.exp()` once at data-prep time) call this
    directly; every other caller wants `reparameterize_from_log_sigma` below instead. The
    distinction is load-bearing, not stylistic: passing log_sigma as sigma silently trains
    a model with the wrong noise scale.
    """
    if sigma.shape != mu.shape:
        raise ValueError(f"mu and sigma must share shape; got {mu.shape} vs {sigma.shape}")
    # Caller-convention guard: sigma is a scale, so it can never be negative -- a caller
    # that accidentally passes a filter's raw log_sigma here WILL hit a negative entry the
    # first time training pushes log_sigma below zero (routine: the clamp floor is -5, and
    # log_sigma=0 is the KL's own trivial minimum, so any exploration below it is expected).
    # This turns that mistake into an immediate crash instead of a silently-trained model.
    if torch.any(sigma < 0):
        raise ValueError(
            "sigma has negative entries -- this is almost certainly log_sigma passed where "
            "sigma was expected. Use reparameterize_from_log_sigma(mu, log_sigma, k) instead, "
            "or exponentiate before calling reparameterize().")
    shape = (*mu.shape[:-1], k, mu.shape[-1])
    eps = torch.randn(shape, generator=generator, dtype=mu.dtype, device=mu.device)
    return mu.unsqueeze(-2) + sigma.unsqueeze(-2) * eps


def reparameterize_from_log_sigma(mu: torch.Tensor, log_sigma: torch.Tensor, k: int,
                                  generator: torch.Generator | None = None,
                                  eps: torch.Tensor | None = None) -> torch.Tensor:
    """THE canonical, single entry point for every caller holding a filter's raw
    `log_sigma` output (every FA/CW simulator call site: `gather_belief_for_batch` ->
    `(mu, log_sigma)`, never sigma directly). Exponentiates once, here, so no call site has
    to remember to: a missing `.exp()` repeated independently across many call sites, instead
    of shared through one function, trains a simulator with `log_sigma` treated as sigma.

    `sample_u` (`atom/filter/gru.py`) is a thin layout wrapper over this function --
    same maths, different K-axis convention (leading K, plus optional common-random-
    numbers `noise` support for the planner's paired comparisons).

    `eps`: optional pre-drawn standard-normal noise, any shape broadcastable against
    `(*mu.shape[:-1], k, mu.shape[-1])` -- lets a caller inject common random numbers
    (as the planner's paired comparisons do) without duplicating the affine-combination formula; `generator` is
    ignored when `eps` is given.
    """
    sigma = log_sigma.exp()
    if eps is None:
        return reparameterize(mu, sigma, k, generator=generator)
    if sigma.shape != mu.shape:
        raise ValueError(f"mu and log_sigma must share shape; got {mu.shape} vs {log_sigma.shape}")
    shape = (*mu.shape[:-1], k, mu.shape[-1])
    eps_b = eps.expand(shape)
    return mu.unsqueeze(-2) + sigma.unsqueeze(-2) * eps_b


def build_structured_committee_eps(batch_size: int, k: int, latent_dim: int, *,
                                   generator: torch.Generator | None = None,
                                   device: torch.device | str = "cpu",
                                   dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Structured, TRAINING-ONLY epsilon for the K-sample committee, replacing i.i.d.
    `eps ~ N(0,I)` per draw. Rationale: an inflation probe showed the sampler transmits sigma faithfully (a frozen eps, scaled 7.8x
    in sigma, produces proportionally scaled spread) -- so a collapsed committee is sigma's
    own level collapsing, not a sampling defect. At a collapsed sigma, K=8 i.i.d. draws give
    the set-level objective's gradient toward WIDENING a noise-buried signal (most draws land
    near mu by chance; whether reaching further out would explain the target set better is
    only weakly estimated). Structured draws de-noise that gradient without adding any force
    toward dispersion: 2 of the K draws are DETERMINISTIC sigma-points, fixed at exactly
    magnitude 2 along one fresh random unit direction per row (`mu +/- 2*sigma*u`,
    sigma-point/unscented lineage -- guarantees the committee always samples the +/-2sigma
    tail, every step, rather than leaving it to chance); the remaining K-2 draws are
    ANTITHETIC PAIRS (`mu +/- sigma*eps`, eps ~ N(0,I), each eps reused for both signs of its
    pair) -- a standard variance-reduction construction that cancels first-order sampling
    noise around mu while still covering it. If K-2 is odd, the leftover draw is a single
    plain i.i.d. sample rather than being silently dropped.

    STRICT BOUNDARY, enforced by construction and by test (`test_joint_set_train_step.py`,
    `test_uncertainty.py`): this function is called ONLY from
    `joint_set_train_step.joint_set_training_step` when `structured_sampling=True`, which is
    itself called ONLY from the training loop. Every eval/gate path
    (`evaluate_gates`/`committee_p_correct`/`fa_coupling_probe.py`) calls
    `reparameterize_from_log_sigma` with `eps=None`, drawing honest i.i.d. samples -- none of
    those functions accept a structured-sampling argument at all, so there is no parameter
    through which this could leak into a reported number.

    Returns `eps` of shape `[batch_size, k, latent_dim]`, meant for
    `reparameterize_from_log_sigma(mu, log_sigma, k, eps=this)`.
    """
    if k < 2:
        raise ValueError(f"structured committee sampling needs k>=2 (2 sigma-point draws), got k={k}")
    draws: list[torch.Tensor] = []
    u = torch.randn(batch_size, latent_dim, generator=generator, device=device, dtype=dtype)
    u = u / u.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    draws.append(2.0 * u)
    draws.append(-2.0 * u)
    remaining = k - 2
    for _ in range(remaining // 2):
        eps = torch.randn(batch_size, latent_dim, generator=generator, device=device, dtype=dtype)
        draws.append(eps)
        draws.append(-eps)
    if remaining % 2 == 1:
        draws.append(torch.randn(batch_size, latent_dim, generator=generator, device=device, dtype=dtype))
    return torch.stack(draws, dim=1)


def person_epistemic_spread(preds: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """preds: predictions per K-sample, K along `dim` (e.g. (..., K) predicted
    probabilities) -> per-example std across K -- the person-epistemic part:
    disagreement between what different plausible YOUs would predict, not noise within one
    prediction. Population std (unbiased=False): K is small (1-2 train, 8 inference),
    where the usual n-1 correction is a bigger relative distortion than
    the bias it corrects.
    """
    return preds.std(dim=dim, unbiased=False)


def mean_prediction(preds: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """The K-sample MEAN is the point prediction -- named here (rather
    than callers just writing `.mean(dim)` inline) so the "mean = prediction, spread =
    person-epistemic" pairing from the spec stays one visible unit, not two places that can
    drift apart under future edits.
    """
    return preds.mean(dim=dim)
