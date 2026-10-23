"""Training and evaluation loop for the CodeWorkout hierarchical attempt-level model:
the within-episode attempt encoder, the cross-episode belief filter and the 4-class
outcome head (the CodeWorkout classifier; thesis Sections 5.3-5.6 and Table 9).

Shared by every script that trains this model (e.g. `cw_sigma.py`,
`cw_4c_2_did_and_prefix_curve.py`, `cw_4i_latent_dim_isolation.py`), so that arms of a
comparison such as the item-holdout difference-in-differences run through one identical
training/evaluation code path; mismatched pipelines between arms would confound the
comparison.

Nothing here knows about the official learner split, the item holdout, or DiD arms --
those are the caller's job (manifest construction). This module only trains one body on
whatever `manifest["attempts"]`/`manifest["train_learners"]`/`manifest["val_learners"]`
it is given and evaluates it.
"""

from __future__ import annotations

import random
import time

import numpy as np
import pandas as pd
import torch

from atom.data.cw_outcome_label import N_OUTCOME, OUTCOME_CLASSES
from atom.eval.metrics import auc, bootstrap_mean_ci, multiclass_brier, multiclass_ece, multiclass_nll
from atom.filter.cw_attempt_encoder import AttemptEncoder
from atom.filter.cw_filter_batch import build_attempt_batch, build_episode_batch
from atom.filter.factory import build_filter
from atom.filter.losses import kl_anneal_weight
from atom.heads.fa_hierarchical_heads import OutcomeHead
from atom.simulator.cw_item_mlp import OutcomeHeadItemMLP, S1FilterV2ItemMLP
from atom.simulator.cw_train_step import (
    belief_ablated_logits,
    build_cw_targets,
    losses_for_cw_hierarchical_batch,
    y_ever_loss_for_batch,
)
from atom.utils.seeds import seed_everything

CODEC_DIM = 45           # matches item_codec's own dim (k=5 deflated, production convention)
CODEC_K = 5
LATENT_DIM = 32
EPISODE_HIDDEN = 64
FILTER_HIDDEN = 64
FACTORY_KW = {"n_heads": 4, "d_ff": 64, "n_rbf": 8}         # the FA filter bake-off's matched-budget values
ATTEMPT_PROJ_DIM = 32
ATTEMPT_HIDDEN = 64                                          # == the head's own prefix_dim
SUMMARY_DIM = 16                                              # == prev_feat_dim override
SHRINKAGE_K = 5.0

DEFAULT_DIMS = {"latent_dim": LATENT_DIM, "episode_hidden": EPISODE_HIDDEN,
                "filter_hidden": FILTER_HIDDEN, "attempt_proj_dim": ATTEMPT_PROJ_DIM,
                "attempt_hidden": ATTEMPT_HIDDEN, "summary_dim": SUMMARY_DIM,
                "codec_dim": CODEC_DIM}


def row_identity(df: pd.DataFrame, split_flag: str) -> dict:
    """Every eval row's identity, read DIRECTLY off `df`'s own columns in `df`'s OWN row
    order -- never re-derived through `attempt_episode_row_idx`/`attempt_within_idx`,
    which index into the PADDED EPISODE tensors (`prefix_states`, `u_ep`,
    `item_codec_sel` -- one row per EPISODE, 0..n_episodes-1), not into the attempts
    table. Using an episode index as an attempts-table row label
    (`df.loc[attempt_episode_row_idx, "problem_id"]`) assigns most rows the wrong
    problem_id while keeping a plausible mask SIZE, so no count-based check can catch
    it; an earlier version of this join had exactly that defect, and
    `assert_row_identity_join_canary` now guards against it.

    This is safe BECAUSE `build_attempt_batch`'s own docstring guarantee holds: it does
    `ev = attempts.copy()` with no re-sort, so `outcome_logits`/`targets` (built from
    `attempt_batch["attempt_episode_row_idx"]`/`build_cw_targets(attempts_batch, ...)`,
    both operating on `attempts_batch` directly) are ALREADY in `attempts_batch`'s own row
    order -- the correct join is therefore the IDENTITY join, not an index lookup at all.

    `df` must be the EXACT frame passed to `forward_batch` as `attempts_batch` (same row
    order, un-reset-index changes only). `split_flag` is caller-supplied context (e.g.
    "val", "train_pool"), not derived from the data.
    """
    n = len(df)
    if n == 0:
        return {"learner": np.array([]), "release": np.array([]), "episode_id": np.array([]),
               "attempt_id": np.array([]), "problem_id": np.array([]),
               "prior_same_problem_count": np.array([], dtype=int),
               "split_flag": np.array([], dtype=object)}
    sorted_df = df.sort_values(["learner", "t"], kind="mergesort")
    prior_same_problem = sorted_df.groupby(["learner", "problem_id"], sort=False).cumcount()
    prior_same_problem = pd.Series(prior_same_problem.to_numpy(), index=sorted_df.index).reindex(df.index)
    return {
        "learner": df["learner"].to_numpy(),
        "release": df["release"].to_numpy(),
        "episode_id": df["episode_id"].to_numpy(),
        "attempt_id": df["code_state_id"].to_numpy(),
        "problem_id": df["problem_id"].to_numpy(),
        "prior_same_problem_count": prior_same_problem.to_numpy(),
        "split_flag": np.full(n, split_flag, dtype=object),
    }


def assert_row_identity_join_canary(df: pd.DataFrame, identity: dict, n_sample: int = 50, seed: int = 0) -> None:
    """Join canary: re-join a random SAMPLE of emitted rows against the SOURCE table BY
    VALUE (attempt_id == code_state_id, the natural unique key the event graph assigns
    per submission) and assert every carried field agrees. Catches the misjoin shape
    described in `row_identity` -- a plausible mask/array SIZE with wrong per-row
    MEMBERSHIP -- which no shape or count assertion can see."""
    n = len(df)
    if n == 0:
        return
    rng = np.random.default_rng(seed)
    sample_pos = rng.choice(n, size=min(n_sample, n), replace=False)
    source_by_attempt_id = df.set_index("code_state_id", drop=False)
    for pos in sample_pos:
        attempt_id = identity["attempt_id"][pos]
        source_row = source_by_attempt_id.loc[attempt_id]
        if isinstance(source_row, pd.DataFrame):  # duplicate code_state_id -- should not happen, fail loudly
            raise AssertionError(f"attempt_id {attempt_id} is not unique in the source table "
                                 f"({len(source_row)} rows) -- join canary cannot resolve it")
        assert identity["learner"][pos] == source_row["learner"], f"pos={pos}: learner mismatch"
        assert identity["episode_id"][pos] == source_row["episode_id"], f"pos={pos}: episode_id mismatch"
        assert identity["problem_id"][pos] == source_row["problem_id"], f"pos={pos}: problem_id mismatch"
        assert identity["release"][pos] == source_row["release"], f"pos={pos}: release mismatch"


def forward_batch(attempt_encoder, filt_forward, outcome_head, y_ever_head, attempts_batch,
                  episodes_batch, code_of_hash, diff_of_hash, problem_codec, pid_to_idx, device,
                  latent_dim: int = LATENT_DIM, codec_dim: int = CODEC_DIM,
                  free_bits: float = 0.05, beta_kl: float = 0.0, k: int = 1,
                  likelihood: str = "single") -> dict:
    """One forward pass for a set of learners' worth of attempts + episodes: the trained
    outcome logits, the belief-ablated logits, the `Y_ever` auxiliary, and enough index
    tensors for a caller to stratify by episode history length.

    `latent_dim`: must match whatever width the `model` argument's pieces were actually
    built with -- only matters for the belief-ablation zero vector's shape, since every
    other tensor's width is read off the modules themselves. Default: `LATENT_DIM`.

    `codec_dim`: must match the actual width of `code_of_hash`/`diff_of_hash`'s vectors
    (and `problem_codec`'s own columns) the caller is passing in -- these come from
    applying the codec transform at whatever rank the caller is testing (e.g. the
    codec-rank ablation), not always the default 45.

    `free_bits`: threads into the KL term's free-bits floor (default 0.05).

    `beta_kl`/`k`: threaded to the internal `losses_for_cw_hierarchical_batch` call
    below so a caller can override them. `beta_kl` is accepted here only for signature
    parity with FA's own `losses_for_batch`/`losses_for_hierarchical_batch` (which take
    the identical dead parameter) -- `losses_for_cw_hierarchical_batch` never reads it,
    the real KL weight applied to the total loss is `train_one_body`'s own `beta_kl`
    argument (its `kl_anneal_weight(..., max_weight=beta_kl, ...)` ceiling), not this one.
    `k` DOES change behaviour: it is the sample count `reparameterize_from_log_sigma`
    draws, though `losses_for_cw_hierarchical_batch` only ever consumes sample index 0
    of the draw (`u_ep = reparameterize_from_log_sigma(...)[:, 0, :]`, the same
    single-sample-per-step convention every FA training call site uses) -- so at k>1 the
    extra samples are drawn and discarded under `likelihood="single"`; `kmixture`
    instead uses all K samples in the shared probability-mixture outcome loss. Defaults:
    `0.0`, `1`, `"single"`."""
    attempt_batch = build_attempt_batch(attempts_batch, code_of_hash, diff_of_hash,
                                        codec_dim, codec_dim, device)
    prefix_states, summary = attempt_encoder(
        attempt_batch["code_vec"], attempt_batch["diff_vec"], attempt_batch["scalar_feats"],
        attempt_batch["lengths"])
    episode_ids_of_summary = attempt_batch["episode_ids"]

    ep_batch = build_episode_batch(episodes_batch, problem_codec, pid_to_idx, summary,
                                   episode_ids_of_summary, device)
    ep_learner_idx, ep_t_idx = _distinct_episode_positions(episodes_batch, ep_batch, episode_ids_of_summary, device)

    targets = build_cw_targets(attempts_batch, device)
    l_outcome, kl, outcome_logits = losses_for_cw_hierarchical_batch(
        filt_forward, outcome_head, ep_batch, ep_learner_idx, ep_t_idx, prefix_states,
        attempt_batch["attempt_episode_row_idx"], attempt_batch["attempt_within_idx"], targets,
        beta_kl=beta_kl, k=k, free_bits=free_bits, likelihood=likelihood)
    ablated_logits = belief_ablated_logits(
        outcome_head, ep_batch, ep_learner_idx, ep_t_idx, prefix_states,
        attempt_batch["attempt_episode_row_idx"], attempt_batch["attempt_within_idx"], latent_dim)

    y_ever_of_ep = dict(zip(episodes_batch["episode_id"], episodes_batch["y_ever"]))
    y_ever_target = torch.tensor([float(y_ever_of_ep[e]) for e in episode_ids_of_summary], device=device)
    l_y_ever, y_ever_logits = y_ever_loss_for_batch(
        filt_forward, y_ever_head, ep_batch, ep_learner_idx, ep_t_idx, y_ever_target)

    return {
        "l_outcome": l_outcome, "kl": kl, "outcome_logits": outcome_logits, "ablated_logits": ablated_logits,
        "targets": targets, "ep_batch": ep_batch, "l_y_ever": l_y_ever, "y_ever_logits": y_ever_logits,
        "y_ever_target": y_ever_target, "attempt_batch": attempt_batch,
        # per-ATTEMPT episode history position (= that episode's own t_idx, i.e. how many
        # PRIOR episodes this learner already had) -- the history-length stratification key.
        "episode_history_per_attempt": ep_t_idx[attempt_batch["attempt_episode_row_idx"]],
        # The causal within-episode prefix state, one row per ATTEMPT in
        # `attempt_batch`'s own row order. Needed for OutcomeHead's `prefix` argument at
        # eval time (`gather_prefix_for_attempts(prefix_states, attempt_episode_row_idx,
        # attempt_within_idx)`, the same call training's own loss function makes).
        "prefix_states": prefix_states,
    }


def _distinct_episode_positions(episodes_batch, ep_batch, episode_ids_of_summary, device):
    ev_sorted_ids = episodes_batch.sort_values(
        ["learner", "start_timestamp"], kind="mergesort")["episode_id"].tolist()
    pos = dict(zip(ev_sorted_ids, zip(ep_batch["learner_idx"].tolist(), ep_batch["t_idx"].tolist())))
    l_idx = torch.tensor([pos[e][0] for e in episode_ids_of_summary], dtype=torch.long, device=device)
    t_idx = torch.tensor([pos[e][1] for e in episode_ids_of_summary], dtype=torch.long, device=device)
    return l_idx, t_idx


def train_one_body(body: str, manifest: dict, args, device: str, seed: int,
                   dims: dict | None = None, item_encoder: str = "linear",
                   beta_kl: float = 0.01, k: int = 1, free_bits: float = 0.05,
                   likelihood: str = "single", sigma_parameterization: str = "direct",
                   progress_callback=None) -> dict:
    """Train one model (attempt encoder, filter `body`, outcome head, `Y_ever` head).

    `dims` (width/summary_dim and codec-rank sweeps): overrides the module's own width
    constants (`latent_dim`, `episode_hidden`, `filter_hidden`, `attempt_proj_dim`,
    `attempt_hidden`, `summary_dim`, `codec_dim`) -- keys as in `DEFAULT_DIMS`. `None`
    (default) trains at exactly the `DEFAULT_DIMS` widths.

    `item_encoder`: `"linear"` (default) or `"mlp"` -- swaps BOTH item pathways (the
    filter's `EpisodeEncoderV2.item_proj` via `body="gru"` only, and both
    `OutcomeHead`/`y_ever_head`'s `trunk`) for `atom.simulator.cw_item_mlp`'s nonlinear
    variants. Only the nonlinearity changes; widths are `dims`' job, kept orthogonal here.

    `beta_kl`/`k`: `beta_kl` is the REAL, load-bearing lever -- it is the ceiling
    `kl_anneal_weight` anneals up to,
    which is the actual weight multiplying `out["kl"]` in this loop's `total` loss below
    (unlike `forward_batch`'s own same-named parameter, which is accepted only for
    signature parity with FA and never used in `losses_for_cw_hierarchical_batch`'s body).
    `k` is threaded to `forward_batch`. `free_bits` and `likelihood` are passed through
    to the loss (defaults `0.05`, `"single"`).
    `sigma_parameterization` defaults to the direct log-sigma head. The alternative,
    `"precision_accumulating"`, is restricted to the MLP item encoder with the GRU body
    and implements monotone precision accumulation inside `KnowledgeFilter` (thesis
    Appendix A.2, width parameterisation).

    `progress_callback`, when supplied, is called after each optimizer step as
    `(step, total_steps, epoch, outcome_loss, y_ever_loss)`; its default is inert and
    exists so detached runners can feed the shared RunLogger without forking this loop.
    Defaults for (`beta_kl`, `k`, `free_bits`, `likelihood`): (`0.01`, `1`, `0.05`,
    `"single"`)."""
    if item_encoder not in ("linear", "mlp"):
        raise ValueError(f"unknown item_encoder {item_encoder!r}; expected 'linear' or 'mlp'")
    if item_encoder == "mlp" and body != "gru":
        raise ValueError("item_encoder='mlp' is only wired for body='gru'")
    if sigma_parameterization not in ("direct", "precision_accumulating"):
        raise ValueError(f"unknown sigma_parameterization {sigma_parameterization!r}")
    if sigma_parameterization != "direct" and item_encoder != "mlp":
        raise ValueError("precision_accumulating is wired only for CW's shipped MLP/GRU filter")
    d = {**DEFAULT_DIMS, **(dims or {})}
    seed_everything(seed)
    attempt_encoder = AttemptEncoder(d["codec_dim"], d["codec_dim"], d["attempt_proj_dim"],
                                     d["attempt_hidden"], d["summary_dim"]).to(device)
    if item_encoder == "linear":
        filt, filt_forward = build_filter(body, codec_dim=d["codec_dim"], episode_hidden=d["episode_hidden"],
                                          filter_hidden=d["filter_hidden"], latent_dim=d["latent_dim"],
                                          prev_feat_dim=d["summary_dim"], **FACTORY_KW)
        outcome_head = OutcomeHead(codec_dim=d["codec_dim"], latent_dim=d["latent_dim"], hidden=d["attempt_hidden"],
                                   n_classes=N_OUTCOME, prefix_dim=d["attempt_hidden"]).to(device)
        y_ever_head = OutcomeHead(codec_dim=d["codec_dim"], latent_dim=d["latent_dim"], hidden=d["attempt_hidden"],
                                  n_classes=1).to(device)
    else:
        filt = S1FilterV2ItemMLP(codec_dim=d["codec_dim"], episode_hidden=d["episode_hidden"],
                                 filter_hidden=d["filter_hidden"], latent_dim=d["latent_dim"],
                                 prev_feat_dim=d["summary_dim"],
                                 sigma_parameterization=sigma_parameterization)
        filt_forward = lambda ic, pf, tm, vm: filt(ic, pf)  # noqa: E731 -- matches build_filter's own convention
        outcome_head = OutcomeHeadItemMLP(codec_dim=d["codec_dim"], latent_dim=d["latent_dim"], hidden=d["attempt_hidden"],
                                          n_classes=N_OUTCOME, prefix_dim=d["attempt_hidden"]).to(device)
        y_ever_head = OutcomeHeadItemMLP(codec_dim=d["codec_dim"], latent_dim=d["latent_dim"], hidden=d["attempt_hidden"],
                                         n_classes=1).to(device)
    filt.to(device)
    # Y_ever-at-episode-start (the auxiliary with a downstream consumer): reuses
    # OutcomeHead's own FiLM-on-(item_codec,u) architecture at n_classes=1, no prefix
    # (predicted AT episode start, where the causal prefix is a fixed constant anyway).

    problem_codec, pid_to_idx = manifest["problem_codec"], manifest["pid_to_idx"]
    trainable = (list(attempt_encoder.parameters()) + list(filt.parameters())
                + list(outcome_head.parameters()) + list(y_ever_head.parameters()))
    optimizer = torch.optim.Adam(trainable, lr=args.lr)

    attempts = manifest["attempts"]
    episodes = manifest["episodes"]
    train_learners = sorted(manifest["train_learners"])
    rng = np.random.default_rng(seed)

    n_batches_per_epoch = max(1, (len(train_learners) + args.learner_batch_size - 1) // args.learner_batch_size)
    total_steps = n_batches_per_epoch * args.epochs
    step = 0
    loss_history: list[float] = []
    y_ever_loss_history: list[float] = []
    t0 = time.time()
    for epoch in range(args.epochs):
        order = rng.permutation(len(train_learners))
        for bstart in range(0, len(train_learners), args.learner_batch_size):
            batch_learners = {train_learners[i] for i in order[bstart:bstart + args.learner_batch_size]}
            attempts_batch = attempts[attempts["learner"].isin(batch_learners)]
            episodes_batch = episodes[episodes["learner"].isin(batch_learners)]

            optimizer.zero_grad()
            out = forward_batch(attempt_encoder, filt_forward, outcome_head, y_ever_head, attempts_batch,
                               episodes_batch, manifest["code_of_hash"], manifest["diff_of_hash"],
                               problem_codec, pid_to_idx, device, latent_dim=d["latent_dim"],
                               codec_dim=d["codec_dim"], beta_kl=beta_kl, k=k,
                               free_bits=free_bits, likelihood=likelihood)
            beta = kl_anneal_weight(step, max_weight=beta_kl, anneal_epochs=max(1, int(0.3 * total_steps)))
            total = out["l_outcome"] + beta * out["kl"] + out["l_y_ever"]
            if not torch.isfinite(total):
                raise RuntimeError(f"[cw-train:{body}:seed{seed}] TRIPWIRE non-finite loss at step={step}: "
                                   f"L_outcome={out['l_outcome'].item()} KL={out['kl'].item()} "
                                   f"L_y_ever={out['l_y_ever'].item()}")
            total.backward()
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
            optimizer.step()
            loss_history.append(out["l_outcome"].item())
            y_ever_loss_history.append(out["l_y_ever"].item())
            step += 1
            if progress_callback is not None:
                progress_callback(step, total_steps, epoch + 1,
                                  loss_history[-1], y_ever_loss_history[-1])
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"[cw-train:{body}:seed{seed}] epoch={epoch + 1}/{args.epochs} "
                 f"L_outcome={loss_history[-1]:.4f} L_y_ever={y_ever_loss_history[-1]:.4f}", flush=True)

    train_s = time.time() - t0
    d = max(1, len(loss_history) // 10)
    first, last = float(np.mean(loss_history[:d])), float(np.mean(loss_history[-d:]))
    loss_decreased = last < first
    y_ever_first = float(np.mean(y_ever_loss_history[:d]))
    y_ever_last = float(np.mean(y_ever_loss_history[-d:]))

    return {
        "attempt_encoder": attempt_encoder, "filt": filt, "filt_forward": filt_forward,
        "outcome_head": outcome_head, "y_ever_head": y_ever_head,
        "loss_first": first, "loss_last": last, "loss_decreased": loss_decreased,
        "y_ever_loss_decreased": y_ever_last < y_ever_first,
        "train_s": train_s, "n_steps": step,
        # Retention/resume payload for standalone cloud trainers (e.g. `cw_sigma.py`).
        # Callers that do not resume ignore these keys; no computation changes.
        "optimizer_state_dict": optimizer.state_dict(),
        "torch_rng_state": torch.get_rng_state(),
        "torch_cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "numpy_rng_state": np.random.get_state(),
        "python_rng_state": random.getstate(),
    }


@torch.no_grad()
def evaluate_val(model, manifest: dict, device: str, latent_dim: int = LATENT_DIM,
                 codec_dim: int = CODEC_DIM, beta_kl: float = 0.0, k: int = 1,
                 free_bits: float = 0.05, likelihood: str = "single") -> dict:
    """Full forward over EVERY val learner at once (83 learners -- cheap, no LLM in this
    loop) with the REAL belief, the belief-ablated arm, the `Y_ever` auxiliary, and a
    shuffled-summary arm (episode-summary-shuffle-must-hurt gate).

    `latent_dim`: must match the `model`'s own width -- see `forward_batch`'s docstring.
    `codec_dim`: must match `manifest["code_of_hash"]`/`manifest["diff_of_hash"]`/
    `manifest["problem_codec"]`'s own width.

    `beta_kl`/`k`: threaded to both the real-belief forward pass and the
    shuffled-summary counterfactual below, so the two are scored under the same `k`
    (a fair shuffle-hurts comparison). Defaults: `0.0`, `1`."""
    attempts = manifest["attempts"]
    episodes = manifest["episodes"]
    val_learners = manifest["val_learners"]
    val_attempts = attempts[attempts["learner"].isin(val_learners)].reset_index(drop=True)
    val_episodes = episodes[episodes["learner"].isin(val_learners)].reset_index(drop=True)

    out = forward_batch(model["attempt_encoder"], model["filt_forward"], model["outcome_head"],
                        model["y_ever_head"], val_attempts, val_episodes, manifest["code_of_hash"],
                        manifest["diff_of_hash"], manifest["problem_codec"], manifest["pid_to_idx"], device,
                        latent_dim=latent_dim, codec_dim=codec_dim, beta_kl=beta_kl, k=k,
                        free_bits=free_bits, likelihood=likelihood)
    attempt_batch, targets = out["attempt_batch"], out["targets"]

    # -- episode-summary shuffle: corrupt WHICH episode's pooled summary informs the
    # cross-episode belief, holding everything else (weights, current item_codec, target
    # attempts) fixed -- a fixed, non-identity permutation (roll-by-1, same convention
    # the FA shuffled-u canary uses, guaranteeing no accidental fixed point at n>=2).
    prefix_states, summary = model["attempt_encoder"](
        attempt_batch["code_vec"], attempt_batch["diff_vec"], attempt_batch["scalar_feats"],
        attempt_batch["lengths"])
    shuf_idx = torch.roll(torch.arange(summary.shape[0], device=device), shifts=1)
    summary_shuffled = summary[shuf_idx]
    ep_batch_shuf = build_episode_batch(val_episodes, manifest["problem_codec"], manifest["pid_to_idx"],
                                        summary_shuffled, attempt_batch["episode_ids"], device)
    ep_l_idx, ep_t_idx = _distinct_episode_positions(val_episodes, ep_batch_shuf, attempt_batch["episode_ids"], device)
    shuf_l_outcome, _, shuf_logits = losses_for_cw_hierarchical_batch(
        model["filt_forward"], model["outcome_head"], ep_batch_shuf, ep_l_idx, ep_t_idx, prefix_states,
        attempt_batch["attempt_episode_row_idx"], attempt_batch["attempt_within_idx"], targets,
        beta_kl=beta_kl, k=k, free_bits=free_bits, likelihood=likelihood)

    probs_real = torch.softmax(out["outcome_logits"], dim=-1).cpu().numpy()
    probs_ablated = torch.softmax(out["ablated_logits"], dim=-1).cpu().numpy()
    probs_shuf = torch.softmax(shuf_logits, dim=-1).cpu().numpy()
    target_np = targets["outcome_target"].cpu().numpy()
    attempt1_mask = (attempt_batch["attempt_within_idx"] == 0).cpu().numpy()
    history_len = out["episode_history_per_attempt"].cpu().numpy()

    y_ever_probs = torch.sigmoid(out["y_ever_logits"]).cpu().numpy()
    y_ever_target_np = out["y_ever_target"].cpu().numpy()
    y_ever_auc = auc(y_ever_target_np, y_ever_probs) if len(set(y_ever_target_np.tolist())) == 2 else float("nan")

    # Row identity read directly off `val_attempts`'
    # OWN columns, in its OWN row order -- see `row_identity`'s docstring for why this
    # (not an `attempt_episode_row_idx` lookup) is the correct join.
    identity = row_identity(val_attempts, split_flag="val")
    assert_row_identity_join_canary(val_attempts, identity)

    return {
        "attempt1_mask": attempt1_mask, "target": target_np, "history_len": history_len,
        "probs_real": probs_real, "probs_ablated": probs_ablated, "probs_shuffled": probs_shuf,
        "val_outcome_loss": float(out["l_outcome"].item()), "val_shuffled_outcome_loss": float(shuf_l_outcome.item()),
        "y_ever_auc": y_ever_auc, "y_ever_n": len(y_ever_target_np),
        # exposed for callers doing their own stratification (item-holdout DiD, prefix
        # curve): which problem each scored attempt belongs to, and each attempt's own
        # causal position within its episode (0-based prefix length).
        "problem_id": identity["problem_id"],
        "attempt_within_idx": attempt_batch["attempt_within_idx"].cpu().numpy(),
        # Row identity carried from source, never re-derived through an index.
        "learner": identity["learner"], "release": identity["release"], "episode_id": identity["episode_id"],
        "attempt_id": identity["attempt_id"], "prior_same_problem_count": identity["prior_same_problem_count"],
        "prior_episode_count": history_len, "split_flag": identity["split_flag"],
        "target_class": target_np,
    }


@torch.no_grad()
def evaluate_on_learners(model, attempts, episodes, code_of_hash, diff_of_hash,
                         problem_codec, pid_to_idx, learners: set, device: str,
                         latent_dim: int = LATENT_DIM, codec_dim: int = CODEC_DIM) -> dict:
    """Generalises `evaluate_val`'s forward-pass-plus-probs core to an ARBITRARY learner
    set and an arbitrary attempts/episodes source (e.g. the item-holdout analysis:
    scoring a TRAIN-POOL learner's own held-out-item attempts under the model trained
    without those items -- legitimate because those specific rows were excluded from
    that model's training entirely, even though the same learner's OTHER episodes were
    not). No shuffle arm (the item-holdout analysis and the prefix curve don't need it)
    -- `evaluate_val` stays the one gate-relevant entry point; this is a read-only
    reporting helper.

    `attempts`/`episodes` should be the FULL (unstripped) tables -- at eval time any
    model may see a learner's real history freely (weights are frozen, nothing leaks
    back into training), so this always evaluates against real history regardless of
    which arm's model is scoring it.
    """
    ev_attempts = attempts[attempts["learner"].isin(learners)].reset_index(drop=True)
    ev_episodes = episodes[episodes["learner"].isin(learners)].reset_index(drop=True)
    if len(ev_attempts) == 0:
        return {"n": 0, "probs_real": np.zeros((0, N_OUTCOME)), "probs_ablated": np.zeros((0, N_OUTCOME)),
               "target": np.array([], dtype=int), "problem_id": np.array([]),
               "attempt_within_idx": np.array([]), "history_len": np.array([]),
               "learner": np.array([]), "release": np.array([]), "episode_id": np.array([]),
               "attempt_id": np.array([]), "prior_same_problem_count": np.array([], dtype=int),
               "split_flag": np.array([], dtype=object)}

    out = forward_batch(model["attempt_encoder"], model["filt_forward"], model["outcome_head"],
                        model["y_ever_head"], ev_attempts, ev_episodes, code_of_hash, diff_of_hash,
                        problem_codec, pid_to_idx, device, latent_dim=latent_dim, codec_dim=codec_dim)
    attempt_batch, targets = out["attempt_batch"], out["targets"]

    # As in `evaluate_val`: identity read directly off `ev_attempts`, never through
    # `attempt_episode_row_idx`.
    identity = row_identity(ev_attempts, split_flag="train_pool_or_val")
    assert_row_identity_join_canary(ev_attempts, identity)

    return {
        "n": len(ev_attempts),
        "probs_real": torch.softmax(out["outcome_logits"], dim=-1).cpu().numpy(),
        "probs_ablated": torch.softmax(out["ablated_logits"], dim=-1).cpu().numpy(),
        "target": targets["outcome_target"].cpu().numpy(),
        "problem_id": identity["problem_id"],
        "attempt_within_idx": attempt_batch["attempt_within_idx"].cpu().numpy(),
        "history_len": out["episode_history_per_attempt"].cpu().numpy(),
        "learner": identity["learner"], "release": identity["release"], "episode_id": identity["episode_id"],
        "attempt_id": identity["attempt_id"], "prior_same_problem_count": identity["prior_same_problem_count"],
        "split_flag": identity["split_flag"],
    }


def stratum_metrics(probs: np.ndarray, target: np.ndarray, mask: np.ndarray | None = None) -> dict:
    if mask is not None:
        probs, target = probs[mask], target[mask]
    n = len(target)
    if n == 0:
        return {"n_eval_population": 0, "nll": float("nan"), "brier": float("nan"), "ece": float("nan")}
    return {
        "n_eval_population": int(n),
        "nll": multiclass_nll(probs, target),
        "brier": multiclass_brier(probs, target, n_classes=N_OUTCOME),
        "ece": multiclass_ece(probs, target),
    }


def run_one_seed_body(body: str, seed: int, manifest: dict, args, device: str,
                      dims: dict | None = None, item_encoder: str = "linear",
                      beta_kl: float = 0.01, k: int = 1, free_bits: float = 0.05,
                      likelihood: str = "single", progress_callback=None) -> dict:
    model = train_one_body(body, manifest, args, device, seed=seed, dims=dims, item_encoder=item_encoder,
                           beta_kl=beta_kl, k=k, free_bits=free_bits, likelihood=likelihood,
                           progress_callback=progress_callback)
    d = {**DEFAULT_DIMS, **(dims or {})}
    val = evaluate_val(model, manifest, device, latent_dim=d["latent_dim"], codec_dim=d["codec_dim"],
                       beta_kl=beta_kl, k=k, free_bits=free_bits, likelihood=likelihood)

    all_metrics = stratum_metrics(val["probs_real"], val["target"])
    attempt1_metrics = stratum_metrics(val["probs_real"], val["target"], val["attempt1_mask"])
    ablated_all = stratum_metrics(val["probs_ablated"], val["target"])
    ablated_attempt1 = stratum_metrics(val["probs_ablated"], val["target"], val["attempt1_mask"])
    shuffled_all = stratum_metrics(val["probs_shuffled"], val["target"])

    row_nll_real = -np.log(np.clip(val["probs_real"][np.arange(len(val["target"])), val["target"]], 1e-12, 1.0))
    row_nll_ablated = -np.log(np.clip(val["probs_ablated"][np.arange(len(val["target"])), val["target"]], 1e-12, 1.0))
    ablation_effect_all, ci_lo_all, ci_hi_all = bootstrap_mean_ci(
        row_nll_ablated - row_nll_real, n_boot=2000, alpha=0.05, seed=seed)
    m1 = val["attempt1_mask"]
    ablation_effect_a1, ci_lo_a1, ci_hi_a1 = bootstrap_mean_ci(
        row_nll_ablated[m1] - row_nll_real[m1], n_boot=2000, alpha=0.05, seed=seed)

    # Free consistency canary (true by construction: clean-full is a subset of
    # clean/compiled) -- checked anyway, at zero cost.
    clean_full_idx = OUTCOME_CLASSES.index("CLEAN_FULL")
    p_clean_full = val["probs_real"][:, clean_full_idx]
    p_any_clean = 1.0 - val["probs_real"][:, OUTCOME_CLASSES.index("COMPILE_ERROR")]
    canary_success_le_clean = bool(np.all(p_clean_full <= p_any_clean + 1e-6))

    # History-length strata on the attempt-1 cell (computed for every seed at zero extra
    # cost, since it is a re-slice of numbers already computed).
    terciles = np.quantile(val["history_len"][m1], [1 / 3, 2 / 3]) if m1.any() else (0.0, 0.0)
    low_mask = m1 & (val["history_len"] <= terciles[0])
    high_mask = m1 & (val["history_len"] >= terciles[1])
    history_low_metrics = stratum_metrics(val["probs_real"], val["target"], low_mask)
    history_high_metrics = stratum_metrics(val["probs_real"], val["target"], high_mask)

    return {
        "model": model, "val": val, "all_attempts": all_metrics, "attempt1_stratum": attempt1_metrics,
        "belief_ablated_all_attempts": ablated_all, "belief_ablated_attempt1": ablated_attempt1,
        "belief_ablation_effect_nats_all_attempts": {"mean": ablation_effect_all, "ci_lo": ci_lo_all, "ci_hi": ci_hi_all},
        "belief_ablation_effect_nats_attempt1": {"mean": ablation_effect_a1, "ci_lo": ci_lo_a1, "ci_hi": ci_hi_a1},
        "belief_ablation_hurts_all_attempts": bool(ci_lo_all > 0.0),
        "belief_ablation_hurts_attempt1": bool(ci_lo_a1 > 0.0),
        "shuffled_summary_all_attempts": shuffled_all,
        "shuffle_hurts": bool(val["val_shuffled_outcome_loss"] > val["val_outcome_loss"]),
        "shuffle_real_val_outcome_loss": val["val_outcome_loss"],
        "shuffle_shuffled_val_outcome_loss": val["val_shuffled_outcome_loss"],
        "p_clean_full_le_p_any_clean_canary": canary_success_le_clean,
        "beats_per_problem_base_rate_nll": None,  # filled in by the caller (needs the shared baseline)
        "beats_learner_prefix_shrinkage_nll": None,
        "y_ever_auc": val["y_ever_auc"], "y_ever_n": val["y_ever_n"],
        "y_ever_loss_decreased": model["y_ever_loss_decreased"],
        "history_terciles": (float(terciles[0]), float(terciles[1])),
        "history_low_attempt1": history_low_metrics, "history_high_attempt1": history_high_metrics,
        "loss_first": model["loss_first"], "loss_last": model["loss_last"],
        "loss_decreased": model["loss_decreased"], "train_s": model["train_s"], "n_steps": model["n_steps"],
    }


def aggregate_across_seeds(per_seed: list[dict], key_path: tuple[str, ...], seed_list: list[int]) -> dict:
    """Mean + bootstrap CI of a scalar metric across seeds (n = number of seeds -- small;
    this is a seed-replication interval and should be read as such, not as a more
    powerful test than it is)."""
    vals = []
    for r in per_seed:
        v = r
        for k in key_path:
            v = v[k]
        vals.append(v)
    mean, lo, hi = bootstrap_mean_ci(np.array(vals), n_boot=2000, alpha=0.05, seed=seed_list[0])
    return {"mean": mean, "ci_lo": lo, "ci_hi": hi, "per_seed": vals, "n_seeds": len(vals)}
