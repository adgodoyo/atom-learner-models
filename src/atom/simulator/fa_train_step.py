"""Shared training-step logic for the FA simulator trainers: filter forward, belief
gathering, loss computation and lambda_A selection, in one tested implementation used by
the FA trainers in `infra/` (e.g. `fa_05_pilot_train.py`, `fa_sim3_full_run.py`).

Running the filter over EVERY learner in `filter_batch` and then keeping only the current
mini-batch's rows is cheap at pilot scale (200 learners) but not at full scale: a global
event-level fraction sample (`select_lm_subsample` with no `n_learners` cap) touches
nearly ALL 3,500 train learners even at a 5% fraction, so that pattern would run a full
3,500-learner filter forward on every training step. `gather_belief_for_batch` avoids this by
subsetting `filter_batch` down to only the learners the current mini-batch actually
references BEFORE running the filter forward -- mathematically identical (no filter body
has cross-learner attention, so subsetting learners cannot change any remaining learner's
own belief), but shrinks the filter's per-step cost from O(n_learners_total) to
O(batch_size).
"""

from __future__ import annotations

import torch

from atom.binding import reparameterize_from_log_sigma
from atom.simulator.fa_alt2 import (
    forward_personalised,
    gather_belief,
    outcome_loss,
    production_loss,
    success_loss,
    support_loss,
)

FILTER_CHUNK_SIZE = 50  # see chunked_filter_forward's own docstring


def chunked_filter_forward(filt_forward, filter_batch, chunk_size: int = FILTER_CHUNK_SIZE):
    """Run the filter over `filter_batch`'s learners in chunks along the batch (learner)
    dimension, concatenating the results -- mathematically identical to one full-batch
    call (each learner's belief only attends within its OWN sequence; there is no
    cross-learner attention in any filter body), but caps the size of any single
    intermediate tensor. This matters most for body (c) at pilot scale (its
    event-transformer computes dense [B, T, T, n_rbf]-shaped attention-bias tensors, and at
    B=200 x T~421 this is >1GB per tensor -- it ran out of memory even with gradient
    checkpointing on the filter forward, since the recompute itself still needs to build
    that same tensor). Once callers go through `gather_belief_for_batch` (below), `B` here
    is normally at most a mini-batch's worth of unique learners, so this chunking loop is
    mostly a safety net rather than a load-bearing fix -- kept because it is an exact,
    free-to-keep operation (concatenation is exactly reversible/differentiable) and because
    a large batch size could still reference more than `chunk_size` unique learners.
    """
    item_codec, prev_feat = filter_batch["item_codec"], filter_batch["prev_feat"]
    time_min, valid_mask = filter_batch["time_min"], filter_batch["valid_mask"]
    n_learners = item_codec.shape[0]
    if chunk_size >= n_learners:
        bounds = [(0, n_learners)]
    else:
        bounds = [(i, min(i + chunk_size, n_learners)) for i in range(0, n_learners, chunk_size)]
    mu_parts, log_sigma_parts = [], []
    for lo, hi in bounds:
        chunk_args = (item_codec[lo:hi], prev_feat[lo:hi], time_min[lo:hi], valid_mask[lo:hi])
        if torch.is_grad_enabled():
            mu_c, log_sigma_c = torch.utils.checkpoint.checkpoint(
                filt_forward, *chunk_args, use_reentrant=False)
        else:
            mu_c, log_sigma_c = filt_forward(*chunk_args)
        mu_parts.append(mu_c)
        log_sigma_parts.append(log_sigma_c)
    return torch.cat(mu_parts, dim=0), torch.cat(log_sigma_parts, dim=0)


def gather_belief_for_batch(filt_forward, filter_batch, learner_idx: torch.Tensor,
                            t_idx: torch.Tensor, chunk_size: int = FILTER_CHUNK_SIZE
                            ) -> tuple[torch.Tensor, torch.Tensor]:
    """Learner-subset belief gather (see module docstring): subset `filter_batch` to only the learners
    `learner_idx` references, run the filter over just that subset, then gather -- instead
    of running the filter over every learner in `filter_batch` and discarding all but this
    batch's rows. `learner_idx`/`t_idx` are indices into the ORIGINAL (full) `filter_batch`
    (the convention every caller already uses); this function handles the remap to the
    subset's local indices internally, so callers pass the same arguments they always did.
    """
    # Subsetting is along dim 0 (the learner axis) ONLY -- every tensor keeps its full
    # original T (the padded history length shared across all of filter_batch's
    # learners), so no learner's padding WIDTH changes between "run over everyone" and
    # "run over just this subset." That is what makes bit-identity with the
    # full-batch-then-gather pattern structural rather than approximate (see this
    # module's tests): position-dependent machinery inside a filter body (causal masks,
    # RBF time/distance biases -- body (c) especially) reads positions relative to a
    # fixed T, and T is exactly what this function leaves untouched.
    # DO NOT "optimise" this further by also truncating T down to the subset's own max
    # valid length: that WOULD change relative padding/positions and would need the same
    # causal-mask/peek-canary re-verification `chunked_filter_forward`'s own chunking
    # never needed (chunking only splits along the already-invariant learner axis, same
    # as here) -- it buys little (T is bounded by history length, not batch composition)
    # and introduces a masking-correctness risk that this function currently has none
    # of.
    unique_learners, local_idx = torch.unique(learner_idx, return_inverse=True)
    sub_batch = {
        "item_codec": filter_batch["item_codec"][unique_learners],
        "prev_feat": filter_batch["prev_feat"][unique_learners],
        "time_min": filter_batch["time_min"][unique_learners],
        "valid_mask": filter_batch["valid_mask"][unique_learners],
    }
    mu_sub, log_sigma_sub = chunked_filter_forward(filt_forward, sub_batch, chunk_size)
    return gather_belief(mu_sub, log_sigma_sub, local_idx, t_idx)


def gather_item_codec_for_batch(filter_batch, learner_idx: torch.Tensor,
                                t_idx: torch.Tensor) -> torch.Tensor:
    """OutcomeHead reads (item_codec, u) -- s(q), the CURRENT row's own question
    codec vector, gathered at the same (learner, t) position `gather_belief_for_batch`
    gathers the belief from. `filter_batch["item_codec"]` is already the frozen codec
    embedding at every (learner, t) cell in the padded history (`filter/batch.py`), so
    this is a plain gather, no new codec lookup -- same convention as
    `atom.simulator.fa_alt2.gather_belief`, kept as its own function (not folded into
    `gather_belief_for_batch`) since OutcomeHead needs it independent of the filter
    forward -- the codec vector does not depend on the belief, only the position."""
    return filter_batch["item_codec"][learner_idx, t_idx]


def build_targets(rows, device: str) -> dict:
    import numpy as np

    from atom.data.fa_outcome_label import outcome_label

    support_target = torch.tensor(rows["support_code"].astype("int64").to_numpy(), device=device)
    success_valid_np = (~rows["score_missing"]).to_numpy()
    success_target = torch.tensor(
        np.where(success_valid_np, rows["score"].fillna(0).to_numpy(), 0.0),
        dtype=torch.float32, device=device)
    success_valid = torch.tensor(success_valid_np, device=device)
    support_valid = torch.ones(len(rows), dtype=torch.bool, device=device)

    # The 5-class outcome label, additive alongside the support/success targets above
    # (those stay defined for reproducibility, unused by the hierarchical trainer -- see
    # fa_outcome_label.py's module docstring).
    outcome = outcome_label(rows["support_code"], rows["score"], rows["score_missing"])
    outcome_target = torch.tensor(outcome["outcome_code"].to_numpy(), dtype=torch.int64, device=device)
    outcome_valid = torch.tensor(outcome["outcome_valid"].to_numpy(), device=device)

    return {"support_target": support_target, "success_target": success_target,
            "success_valid": success_valid, "support_valid": support_valid,
            "outcome_target": outcome_target, "outcome_valid": outcome_valid}


def losses_for_batch(filt_forward, hyper, model, context, support_head, success_head,
                     filter_batch, learner_idx, t_idx, tok_batch, targets, beta_kl: float,
                     k: int = 1):
    from atom.filter.losses import gaussian_kl_free_bits

    mu_sel, log_sigma_sel = gather_belief_for_batch(filt_forward, filter_batch, learner_idx, t_idx)
    u = reparameterize_from_log_sigma(mu_sel, log_sigma_sel, k=k)[:, 0, :]
    out = forward_personalised(model, hyper, context, u, tok_batch["input_ids"],
                               tok_batch["attention_mask"], tok_batch["prompt_end_index"])
    l_s = support_loss(support_head(out["h_q"]), targets["support_target"], targets["support_valid"])
    l_y = success_loss(success_head(out["h_q"]), targets["support_target"], targets["success_target"],
                       targets["success_valid"] & targets["support_valid"])
    l_a = production_loss(out["lm_logits"], tok_batch["labels"], tok_batch["production_valid"])
    kl = gaussian_kl_free_bits(mu_sel, log_sigma_sel, free_bits=0.05).mean()
    return l_s, l_y, l_a, kl, out


def losses_for_hierarchical_batch(filt_forward, hyper, model, context, outcome_head,
                                  filter_batch, learner_idx, t_idx, tok_batch, targets,
                                  beta_kl: float, k: int = 1, likelihood: str = "single"):
    """The hierarchical-emission counterpart of
    `losses_for_batch`, above. Same filter/reparameterise/backbone-forward skeleton;
    the two-head (support+success) tail is replaced by ONE `OutcomeHead` reading
    `(item_codec, u)` directly (never `h_q` -- see `fa_hierarchical_heads.py`'s module
    docstring for the reason).

    `tok_batch` must already be built with `encode_example(..., outcome_code=<the
    OBSERVED class>)` (teacher-forced conditioning); this function does
    not build prompts itself, matching `losses_for_batch`'s own division of labour
    (prompt construction is the caller's/dataset's job, not the loss function's).

    `likelihood`: forwarded to `atom.simulator.fa_alt2.outcome_loss_likelihood` UNCHANGED.
    `'single'` (default) is the single-sample loss for every caller that does not pass
    `likelihood` -- `outcome_loss_likelihood('single', k=1)` samples ONE `u`
    and runs the same masked cross-entropy `outcome_loss` did, through the same call
    site. `'kmixture'` scores the K-sample mixture instead (`k` becomes `K_train`); the
    production forward below always uses ONE representative `u` regardless (the K-mixture
    option changes only the outcome likelihood, never production).

    Returns `(l_outcome, l_a, kl, out)` -- one fewer term than `losses_for_batch`'s
    `(l_s, l_y, l_a, kl, out)` (one head, not two). `select_lambda_a` needs no changes
    to accept this: its `run_losses_fn` contract is exactly "return something whose
    first two entries sum to the non-production loss, then `l_a`, then `kl`, `out`" --
    a hierarchical trainer's closure supplies `(l_outcome, torch.zeros_like(l_outcome),
    l_a, kl, out)`, and `l_s + l_y` reduces to `l_outcome + 0` exactly."""
    from atom.filter.losses import gaussian_kl_free_bits
    from atom.simulator.fa_alt2 import outcome_loss_likelihood

    mu_sel, log_sigma_sel = gather_belief_for_batch(filt_forward, filter_batch, learner_idx, t_idx)
    item_codec_sel = gather_item_codec_for_batch(filter_batch, learner_idx, t_idx)
    l_outcome, u = outcome_loss_likelihood(
        outcome_head, item_codec_sel, mu_sel, log_sigma_sel,
        targets["outcome_target"], targets["outcome_valid"], likelihood=likelihood, k=k)
    out = forward_personalised(model, hyper, context, u, tok_batch["input_ids"],
                               tok_batch["attention_mask"], tok_batch["prompt_end_index"])
    l_a = production_loss(out["lm_logits"], tok_batch["labels"], tok_batch["production_valid"])
    kl = gaussian_kl_free_bits(mu_sel, log_sigma_sel, free_bits=0.05).mean()
    return l_outcome, l_a, kl, out


def combine_hierarchical_loss(l_outcome: torch.Tensor, l_a: torch.Tensor, kl: torch.Tensor,
                              lambda_a: float, beta_kl: float) -> torch.Tensor:
    """The hierarchical loss combination, `L_outcome + lambda_A * L_production +
    beta * KL`, written as its own function (not inlined at each call site) so the
    trainer, any test, and this docstring all point at ONE place. `lambda_A`/`beta` are selected upstream
    (`select_lambda_a`/the KL-annealing schedule) -- this function only sums."""
    return l_outcome + lambda_a * l_a + beta_kl * kl


def select_lambda_a(basis_params, run_losses_fn, grid) -> tuple[float, dict]:
    """The pre-registered lambda_A selection rule, made executable: measure ||d(L_A)/d(basis)|| and
    ||d(L_S+L_Y)/d(basis)|| on the SAME fixed batch via two independent backward passes
    (never mixed into one graph, so neither measurement is affected by the other's
    gradient), then hand both norms to `pick_lambda_a` -- selection logic lives there,
    purely, and is unit-tested there; this function only does the measurement."""
    from atom.simulator.fa_alt2 import pick_lambda_a

    l_s, l_y, l_a, _kl, _out = run_losses_fn()
    for p in basis_params:
        p.grad = None
    (l_s + l_y).backward(retain_graph=True)
    grad_norm_other = sum(p.grad.norm().item() ** 2 for p in basis_params if p.grad is not None) ** 0.5
    for p in basis_params:
        p.grad = None
    l_a.backward()
    grad_norm_a = sum(p.grad.norm().item() ** 2 for p in basis_params if p.grad is not None) ** 0.5
    for p in basis_params:
        p.grad = None
    lam = pick_lambda_a(grad_norm_a, grad_norm_other, grid)
    return lam, {"grad_norm_a": grad_norm_a, "grad_norm_other": grad_norm_other, "chosen_lambda_a": lam}


def k_sample_success_predictions(filt_forward, hyper, model, context, success_head,
                                 filter_batch, learner_idx, t_idx, tok_batch,
                                 support_target: torch.Tensor, k: int,
                                 generator: torch.Generator | None = None) -> torch.Tensor:
    """K-sample machinery for the spread-error check: draw `k` samples of the person latent `u` from
    the CURRENT belief and run each through the full personalised backbone, returning
    (K, N) predicted P(success) at the OBSERVED support class -- one column per example,
    one row per plausible-YOU sample. Selection-at-observed-class mirrors
    `fa_alt2.success_loss`'s own convention (the success head emits one logit per support
    mode; only the mode that actually happened is ever read, same as training).

    Unlike `losses_for_batch` (which draws k samples but only ever uses sample 0, `k=1`
    during training), this runs a SEPARATE backbone forward pass per sample -- K
    independent draws through the SAME frozen+LoRA-hooked backbone, at K times the compute
    of one point-prediction pass. Feed the (K, N) result to
    `atom.eval.spread_error.spread_error_correlation` for the actual gate number; kept
    separate from that function so the FA-specific "how do we get K predictions" wiring
    stays here, next to every other FA-specific forward-pass helper, while the pure
    correlation math stays reusable/testable without a model.
    """
    mu_sel, log_sigma_sel = gather_belief_for_batch(filt_forward, filter_batch, learner_idx, t_idx)
    u_k = reparameterize_from_log_sigma(mu_sel, log_sigma_sel, k=k, generator=generator)  # (N, K, d)
    n, kk, _d = u_k.shape
    probs = torch.empty(kk, n, device=u_k.device, dtype=torch.float32)
    for j in range(kk):
        out = forward_personalised(model, hyper, context, u_k[:, j, :], tok_batch["input_ids"],
                                   tok_batch["attention_mask"], tok_batch["prompt_end_index"])
        logits = success_head(out["h_q"])  # (N, N_SUPPORT)
        selected = logits.gather(1, support_target.view(-1, 1)).squeeze(1)  # (N,)
        probs[j] = torch.sigmoid(selected).float()
    return probs
