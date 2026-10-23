"""Loss wiring for the CodeWorkout per-attempt hierarchical outcome head (thesis
Sections 5.3 and 5.5). Reuses FA's corpus-agnostic filter machinery UNCHANGED (`gather_belief_for_batch`,
`chunked_filter_forward`, `gather_item_codec_for_batch` -- all already generic over
what "item_codec"/"prev_feat" mean, per `fa_train_step.py`'s own docstrings); this
module supplies only the CW-specific glue `losses_for_batch`/`losses_for_hierarchical_batch`
cannot: attempts and episodes are two different row units, so a batch of attempts being
scored must be mapped onto (a) the cross-episode belief `u` at their OWN episode's
(learner, t) position (one `u` shared by every attempt in that episode -- the belief is
per-EPISODE, not per-attempt) and (b) their OWN within-episode causal prefix state
(one PER ATTEMPT, built causally by `cw_attempt_encoder.AttemptEncoder` so an attempt
never sees its own submission).

`losses_for_cw_hierarchical_batch` has no production term (unlike FA's
`losses_for_hierarchical_batch`): the outcome model has no backbone/LoRA/hypernetwork in
its loop. Its loss is exactly `L_outcome` plus the filter's KL term. Code production is
trained separately, through `production_forward_for_batch` below.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from atom.binding import reparameterize_from_log_sigma
from atom.data.cw_outcome_label import outcome_label
from atom.simulator.cw_alt import forward_personalised
from atom.simulator.fa_alt2 import outcome_loss
from atom.simulator.fa_train_step import gather_belief_for_batch, gather_item_codec_for_batch


def build_cw_targets(attempts, device: str) -> dict:
    """`attempts` must carry `compile_result`/`score_frac` (the CW attempt schema).
    Mirrors `fa_train_step.build_targets`'s division of labour (label construction lives
    in its own data module, this just tensorises it)."""
    labels = outcome_label(attempts["compile_result"], attempts["score_frac"])
    outcome_target = torch.tensor(labels["outcome_code"].to_numpy(), dtype=torch.int64, device=device)
    outcome_valid = torch.tensor(labels["outcome_valid"].to_numpy(), device=device)
    return {"outcome_target": outcome_target, "outcome_valid": outcome_valid}


def gather_prefix_for_attempts(prefix_states: torch.Tensor, attempt_episode_row_idx: torch.Tensor,
                               attempt_within_idx: torch.Tensor) -> torch.Tensor:
    """`prefix_states`: `[N_episodes_in_attempt_batch, T_max_attempts, hidden]` --
    `AttemptEncoder`'s own output, one row per episode in ITS OWN batch order.
    `attempt_episode_row_idx`/`attempt_within_idx`: `[N_attempts]` each, mapping every
    attempt being scored to (which episode row, which attempt position within it).
    Returns `[N_attempts, hidden]` -- one causal prefix state per attempt."""
    return prefix_states[attempt_episode_row_idx, attempt_within_idx]


def _outcome_kmixture_nll_and_logits(
    outcome_head,
    item_codec: torch.Tensor,
    u_k: torch.Tensor,
    outcome_target: torch.Tensor,
    prefix: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-row ``-log mean_k p(y|u_k)`` and mixture log-probabilities.

    A CW-local implementation, independent of the FA code path. Averaging normalised
    class probabilities is the same estimator; log space makes it numerically stable and
    makes the metric logits exactly match the loss.
    """
    log_probs_k = torch.stack(
        [
            F.log_softmax(outcome_head(item_codec, u_k[:, j, :], prefix=prefix), dim=-1)
            for j in range(u_k.shape[1])
        ],
        dim=1,
    )
    mixture_logits = torch.logsumexp(log_probs_k, dim=1) - torch.log(
        torch.tensor(float(u_k.shape[1]), dtype=log_probs_k.dtype, device=log_probs_k.device)
    )
    per_row = F.nll_loss(mixture_logits, outcome_target.long(), reduction="none")
    return per_row, mixture_logits


def losses_for_cw_hierarchical_batch(
    filt_forward, outcome_head, filter_batch, episode_learner_idx: torch.Tensor,
    episode_t_idx: torch.Tensor, prefix_states: torch.Tensor,
    attempt_episode_row_idx: torch.Tensor, attempt_within_idx: torch.Tensor,
    targets: dict, beta_kl: float, k: int = 1, free_bits: float = 0.05,
    likelihood: str = "single",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One belief `u` per EPISODE (`episode_learner_idx`/`episode_t_idx`: `[N_episodes]`,
    the DISTINCT episodes referenced by the attempts in this training batch, in the same
    order `attempt_episode_row_idx` indexes into) is drawn once and broadcast to every
    attempt in that episode -- an episode's outcome-conditioning belief does not change
    attempt-to-attempt within it, only the causal prefix does. `s(problem)` is gathered
    the same way (same episode -> same problem).

    Returns `(l_outcome, kl, outcome_logits)` -- `outcome_logits`: `[N_attempts,
    n_classes]`, for a caller that also wants predictions (e.g. the belief-ablation gate,
    which re-runs this with `u` forced to zero and compares).

    `free_bits`: the KL term's free-bits floor (default 0.05).

    `likelihood`: `"single"` (default) scores one reparameterised sample of the belief.
    `"kmixture"` implements the outcome
    objective `-log mean_k p(y | mu + sigma*eps_k)`; the episode-level K draws are broadcast to
    attempts exactly as the single draw is. The returned `outcome_logits` are log
    mixture probabilities in this branch, so downstream NLL/AUC metrics describe the
    same predictive distribution the loss trained rather than sample zero.
    """
    from atom.filter.losses import gaussian_kl_free_bits

    if likelihood not in ("single", "kmixture"):
        raise ValueError(
            f"losses_for_cw_hierarchical_batch: unknown likelihood {likelihood!r}; "
            "expected 'single' or 'kmixture'"
        )

    mu_sel, log_sigma_sel = gather_belief_for_batch(filt_forward, filter_batch,
                                                    episode_learner_idx, episode_t_idx)
    item_codec_sel = gather_item_codec_for_batch(filter_batch, episode_learner_idx, episode_t_idx)
    item_codec_attempt = item_codec_sel[attempt_episode_row_idx]  # [N_attempts, codec_dim]
    prefix_attempt = gather_prefix_for_attempts(prefix_states, attempt_episode_row_idx, attempt_within_idx)

    u_ep_k = reparameterize_from_log_sigma(mu_sel, log_sigma_sel, k=k)
    if likelihood == "single":
        # Single-sample path: draw `k`, consume sample zero, broadcast once, then masked
        # cross-entropy through `outcome_loss`.
        u_attempt = u_ep_k[:, 0, :][attempt_episode_row_idx]
        outcome_logits = outcome_head(item_codec_attempt, u_attempt, prefix=prefix_attempt)
        l_outcome = outcome_loss(outcome_logits, targets["outcome_target"], targets["outcome_valid"])
    else:
        u_attempt_k = u_ep_k[attempt_episode_row_idx]  # [N_attempts, K, latent_dim]
        per_row, outcome_logits = _outcome_kmixture_nll_and_logits(
            outcome_head,
            item_codec_attempt,
            u_attempt_k,
            targets["outcome_target"],
            prefix_attempt,
        )
        valid = targets["outcome_valid"].float()
        l_outcome = per_row.sum() * 0.0 if not bool(valid.any()) else (per_row * valid).sum() / valid.sum()
    kl = gaussian_kl_free_bits(mu_sel, log_sigma_sel, free_bits=free_bits).mean()
    return l_outcome, kl, outcome_logits


def belief_ablated_logits(outcome_head, filter_batch, episode_learner_idx: torch.Tensor,
                          episode_t_idx: torch.Tensor, prefix_states: torch.Tensor,
                          attempt_episode_row_idx: torch.Tensor, attempt_within_idx: torch.Tensor,
                          latent_dim: int) -> torch.Tensor:
    """Belief-ablation arm: `u` forced to exactly zero at every attempt, every other
    input (item_codec, prefix) held identical to a real forward pass. Does NOT run the
    filter at all (no `mu`/`log_sigma` needed) -- `u <- 0 at eval` is the ablation, not
    a draw from a zeroed belief. Callers
    compare this batch's 4-class metrics against `losses_for_cw_hierarchical_batch`'s own
    `outcome_logits` on the SAME rows; a real belief must degrade materially when
    replaced by this, on both the attempt-1 stratum (where `u` is the ONLY signal) and
    all-attempts."""
    item_codec_sel = gather_item_codec_for_batch(filter_batch, episode_learner_idx, episode_t_idx)
    item_codec_attempt = item_codec_sel[attempt_episode_row_idx]
    prefix_attempt = gather_prefix_for_attempts(prefix_states, attempt_episode_row_idx, attempt_within_idx)
    u_zero = torch.zeros(item_codec_attempt.shape[0], latent_dim,
                         device=item_codec_attempt.device, dtype=item_codec_attempt.dtype)
    return outcome_head(item_codec_attempt, u_zero, prefix=prefix_attempt)


def y_ever_loss_for_batch(filt_forward, y_ever_head, filter_batch, episode_learner_idx: torch.Tensor,
                          episode_t_idx: torch.Tensor, y_ever_target: torch.Tensor, k: int = 1
                          ) -> tuple[torch.Tensor, torch.Tensor]:
    """`Y_ever`-at-episode-start: whether the episode is ever solved, an auxiliary target
    used by a direct-versus-rollout comparison downstream.
    Deliberately EPISODE-level, not attempt-level -- one prediction per episode from
    `(item_codec, u)` alone, no within-episode prefix, matching "at episode start" (the
    attempt-1 cell, where the causal prefix is a fixed constant anyway, so omitting
    it here is not a simplification, it is the correct input set). `y_ever_head` is a
    plain `OutcomeHead(..., n_classes=1)` -- reused, not reinvented, since the FiLM
    trunk-on-(item_codec,u) architecture is exactly what this needs at width 1.

    Re-runs its OWN filter gather (does not share `losses_for_cw_hierarchical_batch`'s)
    -- `losses_for_cw_hierarchical_batch`'s tested 3-tuple return is left unchanged
    rather than widened to smuggle out `u_ep`, which would touch every existing call
    site for a second filter forward pass this small a net does not need to avoid.
    """
    if episode_learner_idx.numel() == 0:
        # Same convention as `outcome_loss`/`support_loss`: an empty batch (e.g. every
        # attempt in a minibatch was outcome-invalid) contributes an exact zero, never a
        # 0/0 NaN from an empty BCE mean.
        return y_ever_target.sum() * 0.0, y_ever_target.new_zeros(0)
    mu_sel, log_sigma_sel = gather_belief_for_batch(filt_forward, filter_batch, episode_learner_idx, episode_t_idx)
    item_codec_sel = gather_item_codec_for_batch(filter_batch, episode_learner_idx, episode_t_idx)
    u_ep = reparameterize_from_log_sigma(mu_sel, log_sigma_sel, k=k)[:, 0, :]
    logits = y_ever_head(item_codec_sel, u_ep).squeeze(-1)
    loss = F.binary_cross_entropy_with_logits(logits, y_ever_target.float())
    return loss, logits


def grad_norm_at_belief_junction(filt) -> float:
    """The per-loss gradient-norm logging point for CodeWorkout code-production training
    is the SHARED BELIEF JUNCTION -- the filter's `mu_head` plus its sigma-head parameters --
    never the LoRA basis FA's own `pick_lambda_a` measures at
    (`fa_alt2.pick_lambda_a`'s docstring: "the L2 norms of dL_A/d(basis)"). That
    convention is wrong for CW's outcome head: `OutcomeHead` reads `(item_codec, u)`
    DIRECTLY (`losses_for_cw_hierarchical_batch`), never through the backbone/LoRA path
    at all, so `L_outcome`'s gradient at the basis is EXACTLY zero by construction --
    basis-side balancing would be blind to it, silently treating the outcome loss as
    though it contributed nothing to the shared representation it actually trains hardest.

    Caller's job: run ONE backward pass per loss on a fixed batch (mirroring
    `pick_lambda_a`'s own "two backward passes on the SAME fixed batch" protocol,
    `filt.zero_grad()` between them), then call this once per pass to read off the
    combined L2 norm across both heads -- the two heads together ARE the belief every
    downstream consumer (the outcome head, the hypernetwork via `u`, the planner) reads
    from, so neither alone is the right unit.
    `pick_lambda_a` itself is unchanged and reused as-is (`fa_alt2.pick_lambda_a`) --
    only WHERE the two norms it compares come from differs for CW.
    """
    total_sq = 0.0
    sigma_head = (filt.knowledge_filter.log_sigma_head
                  if hasattr(filt.knowledge_filter, "log_sigma_head")
                  else filt.knowledge_filter.precision_head)
    for p in (filt.knowledge_filter.mu_head.weight, filt.knowledge_filter.mu_head.bias,
              sigma_head.weight, sigma_head.bias):
        if p.grad is not None:
            total_sq += float(p.grad.norm(2).item()) ** 2
    return total_sq ** 0.5


def production_forward_for_batch(filt_forward, hyper, context, model, encoded_batch: dict,
                                 filter_batch, episode_learner_idx: torch.Tensor,
                                 episode_t_idx: torch.Tensor, attempt_episode_row_idx: torch.Tensor,
                                 k: int = 1) -> dict:
    """The code-production head's own belief gather. Like `y_ever_loss_for_batch` (above)
    it RE-RUNS its own filter gather rather than widen `losses_for_cw_hierarchical_batch`'s
    tested 3-tuple return; an extra forward pass of a net this small is cheap.

    Broadcasts one `u` per EPISODE to every attempt in it (the same invariant `losses_
    for_cw_hierarchical_batch` already asserts), imports `atom.simulator.cw_alt.
    forward_personalised` (itself re-exported unchanged from `fa_alt2`) to run the
    LoRA-hooked backbone on `encoded_batch` (a `cw_alt.collate` output, ROW-ALIGNED with
    `attempt_episode_row_idx` -- caller's job, per `build_cw5_examples`'s own docstring
    in `infra/cw_5_full_run.py`).

    Returns `{"lm_logits", "u_attempt", "mu_sel", "log_sigma_sel"}` -- the belief tensors
    are returned too so a caller building the K-marginalised eval (`joint_kc_mixture_nll`)
    or the gradient-norm-balance measurement (`grad_norm_at_belief_junction`) does not
    need a THIRD filter forward for that."""
    mu_sel, log_sigma_sel = gather_belief_for_batch(filt_forward, filter_batch,
                                                    episode_learner_idx, episode_t_idx)
    u_ep = reparameterize_from_log_sigma(mu_sel, log_sigma_sel, k=k)[:, 0, :]
    u_attempt = u_ep[attempt_episode_row_idx]
    out = forward_personalised(model, hyper, context, u_attempt, encoded_batch["input_ids"],
                               encoded_batch["attention_mask"], encoded_batch["prompt_end_index"])
    return {"lm_logits": out["lm_logits"], "u_attempt": u_attempt,
           "mu_sel": mu_sel, "log_sigma_sel": log_sigma_sel}
