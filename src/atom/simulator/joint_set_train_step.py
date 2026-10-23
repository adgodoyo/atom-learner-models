"""The joint-set training step: one shared implementation for both FA and CW belief
training. Every function here is corpus-agnostic: callers pass their own
filter/outcome_head/codec/target-count (5 for FA, 4 for CW); nothing here assumes a
specific class count or item representation. The thesis reports this objective among the
posterior-family variants of Appendix A.2.

THE OBJECTIVE. A SINGLE-outcome likelihood cannot identify the belief's concentration: a
"clone committee" (K samples that agree on everything) always scores at least as well as a
genuinely diverse one on any one held-out answer, so nothing in that objective rewards
diversity, and three single-outcome width interventions did not make the width track
evidence. A SET-level likelihood addresses this:
predicting a JOINT SET of 10-20 later answers under ONE shared z per draw means a clone
committee cannot hedge a set on which the learner's own answers really vary -- and
item-level between-learner variance measured at 61.5% of the maximal-heterogeneity
benchmark shows such sets exist on real data.

THE REGULARISER. A KL(q_phi(z|C) || N(0,I)) anchor never references the calibration
targets this design cares about. The teacher-posterior KL replaces it: a training-only
encoder `q_psi(z|C,T)` that (unlike the causal filter `q_phi(z|C)`) is ALLOWED to see the
target answers, paired with `KL(q_psi || q_phi)` (see `atom.filter.losses.
gaussian_kl_two_diagonal`'s docstring for why this direction, not the reverse, produces a
calibrated white start). The teacher is trained by its OWN set-likelihood term (using
z ~ q_psi) and is DETACHED for the KL term's gradient -- without both of those, minimising
the KL alone would let the teacher collapse trivially onto whatever the student already
says, and it would stop being a meaningful target. This gradient-stop is an implementation
choice beyond the loss term itself, and is stated here so it is not mistaken for part of
the objective.

MEAN-SCALE PENALTY. Nothing else bounds `mu_c`/`mu_psi`'s absolute scale, which was a
candidate explanation for calibration failures of the plain objective. `mu_scale_penalty`
below is a plain L2 penalty on the MEAN only (`mu_c.pow(2).sum(-1)` /
`mu_psi.pow(2).sum(-1)`, never touching `log_sigma_c`/`log_sigma_psi`), applied at EVERY
context, in place of an empty-context-only `gaussian_kl_two_diagonal(mu_c, log_sigma_c, 0, 0)`
anchor. Because nothing may directly reward dispersion, this is an L2-on-mean, not a full
KL(q||N(0,I)): a full KL's variance term actively penalises `log_sigma` moving away from
0, i.e. it fights the very widening this design exists to recover, while a scale bound on
the mean alone leaves `log_sigma` completely free to respond to evidence. Both the
student's own mean (`mu_c`) and the teacher's (`mu_psi`) are penalised, each contributing
gradient only to the network that produced it (no detach) -- the teacher's own scale is
exactly as unbounded as the student's, and it was not established which side drives
scale growth. "never touching log_sigma" is exact for the student term
(`mu_scale_penalty(mu_c)` has no algebraic path to `log_sigma_c`); the teacher term
inherits `TeacherEncoder.context_proj`'s pre-existing dependence of `mu_psi` on
`log_sigma_c`, which this penalty does not add to or remove. Measured effect: the penalty
tamed KL growth and improved coupling/log-loss, but the calibration gates it targeted did
not improve, so mean-scale unboundedness is not SUFFICIENT on its own.

STRUCTURED SAMPLING. A sampler inflation probe showed the sampler transmits sigma
faithfully (a frozen eps, sigma scaled 7.8x, produces proportionally scaled spread), so a
collapsed committee is sigma's own level collapsing -- and at a collapsed sigma, K=8
i.i.d. training draws give the set-level objective's gradient toward WIDENING a
noise-buried signal (most draws land near mu by chance, so whether reaching further out
would explain the target set better is only weakly estimated: sigma does not grow because
the signal telling it to grow is buried in the smallness of sigma).
`structured_sampling=True` below replaces K i.i.d. draws with `atom.binding.
build_structured_committee_eps`'s antithetic pairs + 2 deterministic +-2-sigma tail draws,
de-noising that gradient WITHOUT adding any force toward dispersion (the tail draws are
fixed-magnitude, not reward-shaped). STRICT BOUNDARY: training only. Every eval/gate path
(`evaluate_gates`/`committee_p_correct`/the coupling probe) has no `structured_sampling`
parameter at all and always calls `reparameterize_from_log_sigma` with `eps=None` --
structured sampling cannot reach a reported number through this module's own API surface,
asserted by test (`test_joint_set_train_step.py`).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

from atom.binding import build_structured_committee_eps, reparameterize_from_log_sigma
from atom.filter.losses import gaussian_kl_two_diagonal


class TeacherEncoder(nn.Module):
    """q_psi(z|C,T): a permutation-invariant set encoder over the target answers, combined
    with the student's own (mu_C, log_sigma_C) as a deterministic summary of the causal
    context C (reusing the student's own causal encoding rather than building a second,
    independent context encoder -- the teacher's privileged information is T, not a
    different view of C). `codec_dim`/`latent_dim`/`n_classes` match the corpus's own
    OutcomeHead/filter dims (5 for FA, 4 for CW).

    TEACHER CLAMP. Measured UNCLAMPED, this head's committee width collapsed to ~200x narrower than the student's (implied log sigma
    ~ -6.7, past the STUDENT's own long-standing `[-5, +2]` floor in
    `atom.filter.gru.KnowledgeFilter`), uniformly across context -- and because the
    teacher-posterior KL term (`joint_set_training_step`'s `kl_teacher_student`) trains
    only the student, a collapsed teacher mechanically pulls the student narrower via the
    KL's dominant linear `+log_sigma_student` component, the likely dominant reason
    the student's width did not widen. Fix: symmetric with the student's own bound,
    clamping `log_sigma_head`'s raw output the same way `KnowledgeFilter.forward` already
    clamps its own (`gru.py`'s `log_sigma_min`/`log_sigma_max`, both -5.0/2.0 by default).
    Defaults preserve every existing call site's behaviour up to the bound itself (a
    teacher that was never straying outside [-5, +2] is byte-unaffected); a fresh
    `TeacherEncoder()` construction is REQUIRED for this to take effect -- it cannot be
    retrofitted onto a teacher checkpoint trained without the clamp, which is why
    clamped-teacher training warm-starts only the FILTER, never the teacher."""

    def __init__(self, codec_dim: int, latent_dim: int, n_classes: int, hidden: int,
                log_sigma_min: float = -5.0, log_sigma_max: float = 2.0):
        super().__init__()
        self.n_classes = n_classes
        self.log_sigma_min = log_sigma_min
        self.log_sigma_max = log_sigma_max
        self.target_embed = nn.Linear(codec_dim + n_classes, hidden)
        self.context_proj = nn.Linear(2 * latent_dim, hidden)
        self.trunk = nn.Linear(2 * hidden, hidden)
        self.mu_head = nn.Linear(hidden, latent_dim)
        self.log_sigma_head = nn.Linear(hidden, latent_dim)
        # Small init on the heads: at initialisation the teacher should sit close to the
        # student's own (mu_C, log_sigma_C) summary via context_proj, not a random point --
        # matches `OutcomeHead`'s own FiLM-gate small-init convention (fa_hierarchical_heads.py).
        nn.init.normal_(self.mu_head.weight, std=0.01)
        nn.init.zeros_(self.mu_head.bias)
        nn.init.normal_(self.log_sigma_head.weight, std=0.01)
        nn.init.zeros_(self.log_sigma_head.bias)

    def forward(self, mu_c: torch.Tensor, log_sigma_c: torch.Tensor,
                target_item_codec: torch.Tensor, target_outcome_onehot: torch.Tensor,
                target_valid_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """mu_c/log_sigma_c: `[B, L]`. target_item_codec: `[B, T, codec_dim]`.
        target_outcome_onehot: `[B, T, n_classes]`. target_valid_mask: `[B, T]` bool.
        Returns (mu_psi, log_sigma_psi), each `[B, L]` -- `log_sigma_psi` clamped to
        `[log_sigma_min, log_sigma_max]`, symmetric with the student's own floor."""
        per_target = torch.relu(
            self.target_embed(torch.cat([target_item_codec, target_outcome_onehot], dim=-1))
        )
        mask_f = target_valid_mask.float().unsqueeze(-1)
        pooled = (per_target * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0)
        ctx = torch.relu(self.context_proj(torch.cat([mu_c, log_sigma_c], dim=-1)))
        h = torch.relu(self.trunk(torch.cat([pooled, ctx], dim=-1)))
        log_sigma_psi = self.log_sigma_head(h).clamp(self.log_sigma_min, self.log_sigma_max)
        return self.mu_head(h), log_sigma_psi


class TeacherPosterior(nn.Module):
    """CW-compatible adapter over the shared teacher-posterior architecture.

    CW passes target-set tensors first and an optional pooled context codec; FA uses
    :class:`TeacherEncoder` directly.  Keeping this adapter here preserves one shared
    implementation while allowing each corpus trainer to retain its natural call shape.
    """

    def __init__(self, codec_dim: int, latent_dim: int, n_classes: int, hidden: int):
        super().__init__()
        self.encoder = TeacherEncoder(codec_dim, latent_dim, n_classes, hidden)
        self.context_codec_proj = nn.Linear(codec_dim, 2 * latent_dim)
        self.codec_dim = codec_dim
        self.latent_dim = latent_dim
        self.n_classes = n_classes

    def forward(self, target_codec: torch.Tensor, target_class: torch.Tensor,
                target_mask: torch.Tensor, *, context_codec: torch.Tensor | None = None):
        b, t, _ = target_codec.shape
        onehot = torch.zeros((b, t, self.n_classes), device=target_codec.device,
                             dtype=target_codec.dtype)
        onehot.scatter_(2, target_class.clamp(min=0).unsqueeze(-1), 1.0)
        onehot = onehot * target_mask.unsqueeze(-1).to(onehot.dtype)
        if context_codec is None:
            context_codec = torch.zeros((b, self.codec_dim), device=target_codec.device,
                                         dtype=target_codec.dtype)
        ctx = self.context_codec_proj(context_codec)
        mu_c, log_sigma_c = ctx.chunk(2, dim=-1)
        return self.encoder(mu_c, log_sigma_c, target_codec, onehot, target_mask)


def joint_set_kmixture_nll(logits_k: torch.Tensor, target_class: torch.Tensor,
                           target_mask: torch.Tensor) -> torch.Tensor:
    """Set-level K-mixture NLL from precomputed ``[B,K,T,C]`` logits."""
    if logits_k.ndim != 4:
        raise ValueError(f"expected logits [B,K,T,C], got {tuple(logits_k.shape)}")
    b, k, t, _ = logits_k.shape
    if target_class.shape != (b, t) or target_mask.shape != (b, t):
        raise ValueError("target tensors must have shape [B,T]")
    lp = torch.log_softmax(logits_k, dim=-1)
    gather = target_class.clamp(min=0).unsqueeze(1).unsqueeze(-1).expand(b, k, t, 1)
    log_p = lp.gather(-1, gather).squeeze(-1)
    log_joint = (log_p * target_mask.unsqueeze(1).to(log_p.dtype)).sum(-1)
    return -torch.logsumexp(log_joint, dim=1) + torch.log(torch.tensor(float(k), device=lp.device))


def teacher_posterior_kl(mu_teacher: torch.Tensor, log_sigma_teacher: torch.Tensor,
                         mu_student: torch.Tensor, log_sigma_student: torch.Tensor) -> torch.Tensor:
    """Per-draw KL(q_teacher || q_student), returned as ``[B]``."""
    return gaussian_kl_two_diagonal(mu_teacher.detach(), log_sigma_teacher.detach(),
                                    mu_student, log_sigma_student)


@dataclass
class PrefixTargetDraw:
    learner_idx: int
    prefix_t: int            # raw padded-sequence position to gather belief at (H_<prefix_t)
    target_t: np.ndarray     # sorted raw positions of the target set, all >= prefix_t


def sample_prefix_target_draws(valid_t_by_learner: dict[int, np.ndarray], rng: np.random.Generator,
                                prefix_pool: tuple[int, ...], target_set_size_range: tuple[int, int],
                                n_draws: int) -> list[PrefixTargetDraw]:
    """`valid_t_by_learner[learner_idx]`: that learner's own sorted array of OUTCOME-VALID
    raw sequence positions (score-missing rows excluded upstream -- they cannot be target
    answers, but the filter still sees them as events via the padded batch it was built
    from, so raw positions may have gaps). `prefix_pool` values are RAW positions to gather
    the filter's belief at; `mu_all[learner, p]` contains H_<p by `KnowledgeFilter`'s own
    construction (`src/atom/filter/gru.py`: "e_ep[:, t] describes event t-1"), so p=0 is
    the true cold start and any p <= min(target_t) is guaranteed non-leaky by construction
    -- this function enforces exactly that bound, never a stronger (lossier) one.

    A learner contributes a draw only if some prefix value in `prefix_pool` leaves at least
    `target_set_size_range[0]` valid positions at or after it; infeasible learners are
    skipped, not padded with a degenerate draw."""
    lo, hi = target_set_size_range
    learners = [ell for ell, v in valid_t_by_learner.items() if len(v) >= lo]
    if not learners:
        return []
    draws: list[PrefixTargetDraw] = []
    for _ in range(n_draws):
        ell = learners[int(rng.integers(0, len(learners)))]
        v = valid_t_by_learner[ell]
        feasible_p = [p for p in prefix_pool if int(np.sum(v >= p)) >= lo]
        if not feasible_p:
            feasible_p = [0]
        prefix_t = int(feasible_p[int(rng.integers(0, len(feasible_p)))])
        pool = v[v >= prefix_t]
        size = min(len(pool), int(rng.integers(lo, hi + 1)))
        chosen = rng.choice(len(pool), size=size, replace=False)
        target_t = np.sort(pool[chosen])
        draws.append(PrefixTargetDraw(learner_idx=ell, prefix_t=prefix_t, target_t=target_t))
    return draws


def pad_draws(draws: list[PrefixTargetDraw], item_codec_all: torch.Tensor, outcome_all: torch.Tensor,
              n_classes: int, device: str) -> dict[str, torch.Tensor]:
    """Materialise a batch of draws into padded tensors. `item_codec_all`: `[N_learners, T, codec_dim]`
    (the filter batch's own item codec, indexable by raw position). `outcome_all`:
    `[N_learners, T]` int64, the 5/4-class outcome code at every raw position (undefined --
    any value -- at score-missing/invalid positions, since `target_valid_mask` excludes
    them; callers must NOT rely on `outcome_all`'s value there)."""
    b = len(draws)
    t_max = max((len(d.target_t) for d in draws), default=0)
    codec_dim = item_codec_all.shape[-1]
    learner_idx = torch.tensor([d.learner_idx for d in draws], dtype=torch.long, device=device)
    prefix_t = torch.tensor([d.prefix_t for d in draws], dtype=torch.long, device=device)
    target_item_codec = torch.zeros(b, t_max, codec_dim, device=device)
    target_outcome = torch.zeros(b, t_max, dtype=torch.long, device=device)
    target_valid_mask = torch.zeros(b, t_max, dtype=torch.bool, device=device)
    for i, d in enumerate(draws):
        n = len(d.target_t)
        if n == 0:
            continue
        tt = torch.tensor(d.target_t, dtype=torch.long, device=device)
        target_item_codec[i, :n] = item_codec_all[d.learner_idx, tt]
        target_outcome[i, :n] = outcome_all[d.learner_idx, tt]
        target_valid_mask[i, :n] = True
    target_outcome_onehot = torch.zeros(b, t_max, n_classes, device=device)
    target_outcome_onehot.scatter_(2, target_outcome.unsqueeze(-1), 1.0)
    target_outcome_onehot = target_outcome_onehot * target_valid_mask.unsqueeze(-1).float()
    return {
        "learner_idx": learner_idx, "prefix_t": prefix_t,
        "target_item_codec": target_item_codec, "target_outcome": target_outcome,
        "target_outcome_onehot": target_outcome_onehot, "target_valid_mask": target_valid_mask,
    }


def set_kmixture_nll(outcome_head, item_codec_target: torch.Tensor, z_k: torch.Tensor,
                      outcome_target: torch.Tensor, target_valid_mask: torch.Tensor,
                      outcome_head_kwargs: dict | None = None) -> torch.Tensor:
    """The set-level K-mixture NLL: `-log mean_k Pi_{i in T} p(y_i | z_k, e_i)`, computed in
    log-space (`-logsumexp_k(sum_i log p(y_i|z_k,e_i)) + log K`) for numerical stability --
    the set-level generalisation of `atom.simulator.fa_alt2.outcome_kmixture_nll_per_row`'s
    single-outcome mixture, which this REPLACES in the joint-set objective (never invoked here). A clone
    committee (every z_k predicting the same class everywhere) pays the SAME per-target cost
    as any single sample would; a committee that HEDGES across genuinely different targets
    (because z_k's differ) can spread its probability mass to cover more of the true joint
    outcome pattern -- this is the mechanism the single-outcome mixture structurally lacked.

    `item_codec_target`: `[B, T, codec_dim]`. `z_k`: `[B, K, latent_dim]`. `outcome_target`:
    `[B, T]` int64 (value at invalid positions ignored). `target_valid_mask`: `[B, T]` bool.
    Returns per-draw NLL `[B]`, never batch-reduced (caller means/weights, matching every
    other loss function's convention in this codebase)."""
    b, k, latent_dim = z_k.shape
    t = item_codec_target.shape[1]
    head_kwargs = outcome_head_kwargs or {}
    codec_dim = item_codec_target.shape[-1]
    ic_flat = item_codec_target.unsqueeze(1).expand(b, k, t, codec_dim).reshape(b * k * t, codec_dim)
    z_flat = z_k.unsqueeze(2).expand(b, k, t, latent_dim).reshape(b * k * t, latent_dim)
    logits = outcome_head(ic_flat, z_flat, **head_kwargs).reshape(b, k, t, -1)
    log_probs = torch.log_softmax(logits, dim=-1)
    target_flat = outcome_target.unsqueeze(1).expand(b, k, t).clamp(min=0)
    log_p_target = log_probs.gather(-1, target_flat.unsqueeze(-1)).squeeze(-1)  # [B, K, T]
    mask = target_valid_mask.unsqueeze(1).float()  # [B, 1, T]
    log_p_set_given_k = (log_p_target * mask).sum(dim=-1)  # [B, K]
    log_k = torch.log(torch.tensor(float(k), device=z_k.device))
    return log_k - torch.logsumexp(log_p_set_given_k, dim=1)


def mu_scale_penalty(mu: torch.Tensor) -> torch.Tensor:
    """Mean-scale penalty (see module docstring): a plain L2 penalty on a mean vector's
    own scale, `[B]` (per-row squared norm, never batch-reduced -- caller means/weights,
    matching every other loss function's convention here). Deliberately touches only the
    mean, never `log_sigma` -- an output-scale BOUND, not a dispersion force -- so it can
    be applied at every context (not just `prefix_t == 0`) without fighting the widening
    this design exists to recover. Shared so the CW trainer can use it directly against
    its own `mu_c`/`mu_psi` outside :func:`joint_set_training_step`'s wrapper, matching how
    it consumes :class:`TeacherPosterior`/`joint_set_kmixture_nll`/
    `teacher_posterior_kl` at the primitive level rather than through this function."""
    return mu.pow(2).sum(dim=-1)


@dataclass
class JointSetStepOutput:
    loss: torch.Tensor
    set_nll_student: torch.Tensor
    set_nll_teacher: torch.Tensor
    kl_teacher_student: torch.Tensor
    mu_scale_penalty: torch.Tensor
    diagnostics: dict = field(default_factory=dict)


def joint_set_training_step(outcome_head, teacher: TeacherEncoder,
                             mu_c: torch.Tensor, log_sigma_c: torch.Tensor,
                             target_item_codec: torch.Tensor, target_outcome: torch.Tensor,
                             target_outcome_onehot: torch.Tensor, target_valid_mask: torch.Tensor,
                             prefix_t: torch.Tensor, k_train: int, beta_kl: float,
                             beta_scale: float, generator: torch.Generator | None = None,
                             outcome_head_kwargs: dict | None = None,
                             structured_sampling: bool = False) -> JointSetStepOutput:
    """One joint-set training step's LOSS, given an already-computed filter belief
    `(mu_c, log_sigma_c)` at each draw's prefix position (the caller runs the filter
    forward + `gather_belief`-style indexing itself, matching every other trainer in this
    codebase's convention of keeping the filter forward outside this function). `prefix_t`
    (`[B]`) is used only to count empty-context draws (`prefix_t == 0`) for the
    diagnostics -- it does not re-index anything here and enters no loss term.

    Returns every term SEPARATELY (never pre-summed) so a caller/report can log and gate on
    each one; `.loss` is the pre-registered weighted sum for `.backward()`."""
    mu_psi, log_sigma_psi = teacher(mu_c, log_sigma_c, target_item_codec,
                                    target_outcome_onehot, target_valid_mask)

    if structured_sampling:
        # Structured sampling: antithetic pairs + 2 deterministic
        # sigma-point tail draws, built fresh per call for student and teacher separately
        # (their sigmas differ, so a shared eps still gives each its own structured z).
        eps_student = build_structured_committee_eps(
            mu_c.shape[0], k_train, mu_c.shape[-1], generator=generator,
            device=mu_c.device, dtype=mu_c.dtype)
        z_student = reparameterize_from_log_sigma(mu_c, log_sigma_c, k=k_train, eps=eps_student)
        eps_teacher = build_structured_committee_eps(
            mu_psi.shape[0], k_train, mu_psi.shape[-1], generator=generator,
            device=mu_psi.device, dtype=mu_psi.dtype)
        z_teacher = reparameterize_from_log_sigma(mu_psi, log_sigma_psi, k=k_train, eps=eps_teacher)
    else:
        z_student = reparameterize_from_log_sigma(mu_c, log_sigma_c, k=k_train, generator=generator)
        z_teacher = reparameterize_from_log_sigma(mu_psi, log_sigma_psi, k=k_train, generator=generator)

    nll_student = set_kmixture_nll(outcome_head, target_item_codec, z_student, target_outcome,
                                   target_valid_mask, outcome_head_kwargs)
    nll_teacher = set_kmixture_nll(outcome_head, target_item_codec, z_teacher, target_outcome,
                                   target_valid_mask, outcome_head_kwargs)

    # Gradient-stop on the teacher side of the KL: only the student is pulled toward
    # covering the teacher's spread (see module docstring for why this direction).
    kl = gaussian_kl_two_diagonal(mu_psi.detach(), log_sigma_psi.detach(), mu_c, log_sigma_c)

    # Mean-scale penalty: L2-on-mean scale bound, EVERY context, both mu_c and mu_psi, no
    # detach on either side (each penalises only the network that produced it -- see module
    # docstring for why this replaces an empty-context-only KL anchor).
    scale_penalty_per_row = mu_scale_penalty(mu_c) + mu_scale_penalty(mu_psi)
    scale_penalty = scale_penalty_per_row.mean()

    empty_mask = (prefix_t == 0).float()
    loss = nll_student.mean() + nll_teacher.mean() + beta_kl * kl.mean() + beta_scale * scale_penalty
    return JointSetStepOutput(
        loss=loss, set_nll_student=nll_student.mean(), set_nll_teacher=nll_teacher.mean(),
        kl_teacher_student=kl.mean(), mu_scale_penalty=scale_penalty,
        diagnostics={"n_empty_context_draws": int(empty_mask.sum().item())},
    )
