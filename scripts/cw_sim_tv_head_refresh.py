#!/usr/bin/env python3
"""Retrain the test-vector head of the CodeWorkout generative simulator on a frozen filter.

Applies the recipe of `cw_sim_head_refresh.py` to `test_vector_head` instead of
`outcome_head`: `attempt_encoder` and `filt` are frozen completely (checked byte for
byte after training), Adam lr=1e-3, 60 epochs, learner batches of 32, masked BCE as in
`cw_sim_full_run.test_vector_batch_loss`, loss weight 1.0. Manifest, split, codec, item
encoder and dimensions are those of `cw_sigma.py`. First-attempt rows are up-weighted by
a fixed ratio, `n_later_attempt_matched / n_first_attempt_matched` on the train split,
computed before any gradient step, so both groups carry equal total loss mass.

Arms, each scored on validation first attempts with a learner-clustered bootstrap AUC
CI (`atom.eval.metrics.learner_clustered_auc_ci`; AUC is not a per-row mean, so the
paired-delta bootstrap does not apply):
  1. "old": the base checkpoint's own `test_vector_head`, loaded unmodified.
  2. "refreshed": the head retrained with the recipe above.
  3. "refreshed_null_prefix": the same recipe plus a learned null-prefix embedding
     (`nn.Parameter`, in the same optimiser) that replaces the fixed zero vector
     `AttemptEncoder` emits at `attempt_within_idx == 0`. The substitution is applied
     to the gathered prefix tensor after `AttemptEncoder.forward` returns, so the frozen
     encoder and filter are untouched.
  4. "base_rate": a per-(problem_id, test-case slot) pass rate from train first
     attempts, falling back to the problem-pooled rate for an unseen slot and to the
     corpus-wide rate for an unseen problem (fallback use is counted and saved).
  5. "blend_<head>": `sigmoid(w1*head_logit + w2*base_rate_logit)` with no intercept,
     fitted per head on train first attempts and applied unchanged to validation.
     Caveat: the base-rate feature is computed from the train split and the blend is
     also fitted on it, which gives that feature some in-sample optimism.

The base checkpoint (`--checkpoint`) must match `EXPECTED_BASE_SHA256`; this is asserted
before any training step.

Outputs (in `--out-dir`): `final.pt`, `final_null_prefix.pt`, `base_rate_table.npz`,
`blend_weights.json`, `manifest.json` (arms table, fallback counts, hashes). The
TIKTOC half of the arms table is computed by
`infra/cw_sim_diag2_tiktoc_tv_refresh_rescore.py` from these files (TIKTOC holds first
attempts only, so no slicing is needed there).
Serves Appendix A (test-vector head refresh) and Table 11.

    python scripts/cw_sim_tv_head_refresh.py \\
        --code-map cache/cw_03_vectors/cw_code_state_hash_map_full.npz \\
        --diff-map cache/cw_03_vectors/cw_diff_hash_map_full.npz \\
        --codealone-meta cache/cw_03_vectors/codealone/meta.json \\
        --codealone-states cache/cw_03_vectors/codealone/states_mean.npy \\
        --diff-meta cache/cw_03_vectors/diff/meta.json \\
        --diff-states cache/cw_03_vectors/diff/states_mean.npy \\
        --out-dir results/cw_sim_tv_head_refresh
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

REPO = pathlib.Path(__file__).resolve().parents[1]
for p in (REPO / "src", REPO / "infra", REPO / "scripts", REPO):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from cw_manifest_io import load_manifest  # noqa: E402
from cw_4g_codec_rank_item_mlp import rebuild_problem_codec_at_rank  # noqa: E402
from cw_classifier_common import CODEC_K, RANK, TRANSFORM_PATH  # noqa: E402
from cw_sim_belief_chunks import learner_aligned_chunks  # noqa: E402
from cw_sim_full_run import load_test_vector_labels, test_vector_auc_panel  # noqa: E402
from cw_sim_head_refresh import _git_sha, _sha256, assert_frozen, freeze  # noqa: E402

from atom.binding import reparameterize_from_log_sigma  # noqa: E402
from atom.codec.transform import CodecTransform  # noqa: E402
from atom.data.cw_manifest import build_codec_vector_lookup  # noqa: E402
from atom.eval.metrics import learner_clustered_auc_ci  # noqa: E402
from atom.eval.tiktoc_metrics import tiktoc_comparison_panel  # noqa: E402
from atom.filter.cw_attempt_encoder import AttemptEncoder  # noqa: E402
from atom.filter.cw_filter_batch import build_attempt_batch, build_episode_batch  # noqa: E402
from atom.simulator.cw_4_train_loop import _distinct_episode_positions  # noqa: E402
from atom.simulator.cw_item_mlp import OutcomeHeadItemMLP, S1FilterV2ItemMLP  # noqa: E402
from atom.simulator.cw_train_step import (gather_belief_for_batch,  # noqa: E402
                                          gather_item_codec_for_batch, gather_prefix_for_attempts)
from atom.utils.seeds import seed_everything  # noqa: E402

EXPECTED_BASE_SHA256 = "3143d05122957e855b2067f9bd9c61cd7aabf414b380ffc5fc2e46a0ccf7c3e9"


class _Args:
    pass


# --------------------------------------------------------------------------------------
# Shared primitives (oversample ratio, weighted loss, mu scoring, null-prefix swap)
# --------------------------------------------------------------------------------------

def compute_first_attempt_oversample_ratio(attempts: pd.DataFrame, tv_labels: dict,
                                           train_learners: set) -> tuple[float, dict]:
    """The fixed first-attempt weight, computed once from the TRAIN split's matched-row
    composition, before any gradient step. `ratio = n_later_matched / n_first_matched`
    (rounded to 2dp): the weight that would make first-attempt and later-attempt rows
    contribute EQUAL total mass to the loss."""
    lookup = tv_labels["lookup"]
    tr = attempts[attempts["learner"].isin(train_learners)]
    keys = list(zip(tr["learner"].astype(str), tr["problem_id"].astype(int), tr["code_state_id"].astype(str)))
    matched = np.array([k in lookup for k in keys])
    is_first = (tr["attempt_within_idx"].to_numpy() == 0)
    n_first = int((matched & is_first).sum())
    n_later = int((matched & ~is_first).sum())
    ratio = round(n_later / n_first, 2) if n_first > 0 else 1.0
    stats = {"n_first_attempt_matched_train": n_first, "n_later_attempt_matched_train": n_later, "ratio": ratio}
    return ratio, stats


def apply_null_prefix(prefix_sel: torch.Tensor, attempt_within_idx_sel: torch.Tensor,
                      null_prefix: "torch.Tensor | None") -> torch.Tensor:
    """Substitutes the LEARNED `null_prefix` for exactly the first-attempt rows in
    `prefix_sel` (the frozen, always-identical zero vector `AttemptEncoder` emits at
    position 0), leaving every later-attempt row's real causal prefix untouched.
    `None` is a pure pass-through, used by the standard-refresh arm."""
    if null_prefix is None:
        return prefix_sel
    is_first = (attempt_within_idx_sel == 0).unsqueeze(1)
    return torch.where(is_first, null_prefix.unsqueeze(0).expand_as(prefix_sel), prefix_sel)


def weighted_test_vector_batch_loss(filt_forward, test_vector_head, ep_batch, ep_learner_idx, ep_t_idx,
                                    prefix_states, attempt_batch, attempts_batch: pd.DataFrame,
                                    tv_labels: dict, oversample_ratio: float, device: str,
                                    null_prefix: "torch.Tensor | None" = None) -> tuple["torch.Tensor | None", dict]:
    """`cw_sim_full_run.test_vector_batch_loss`'s weighted sibling -- identical matching,
    gather, and forward computation, with two additions: (1) each matched row's
    contribution to the masked-BCE mean is multiplied by `oversample_ratio` if it is a
    first attempt, else 1.0; (2) if `null_prefix` is given, `apply_null_prefix`
    substitutes it into first-attempt rows' gathered prefix before the head reads it."""
    at = attempts_batch.reset_index(drop=True)
    lookup = tv_labels["lookup"]
    matched_pos, targets, valid_masks, weights = [], [], [], []
    for i, row in at.iterrows():
        key = (str(row["learner"]), int(row["problem_id"]), str(row["code_state_id"]))
        if key not in lookup:
            continue
        rec = lookup[key]
        matched_pos.append(i)
        targets.append(rec["test_vector"].astype(np.float32))
        valid_masks.append(rec["valid_mask"])
        weights.append(oversample_ratio if int(row.get("attempt_within_idx", 0)) == 0 else 1.0)
    if not matched_pos:
        return None, {"n_matched": 0, "n_first_attempt_matched": 0}

    mu_sel_all, log_sigma_sel_all = gather_belief_for_batch(filt_forward, ep_batch, ep_learner_idx, ep_t_idx)
    item_codec_ep = gather_item_codec_for_batch(ep_batch, ep_learner_idx, ep_t_idx)

    row_idx_t = attempt_batch["attempt_episode_row_idx"]
    matched_pos_t = torch.tensor(matched_pos, dtype=torch.long, device=device)
    matched_row_idx = row_idx_t[matched_pos_t]
    matched_within_idx = attempt_batch["attempt_within_idx"][matched_pos_t]

    u = reparameterize_from_log_sigma(mu_sel_all, log_sigma_sel_all, k=1)[:, 0, :][matched_row_idx]
    item_codec_sel = item_codec_ep[matched_row_idx]
    prefix_sel = gather_prefix_for_attempts(prefix_states, matched_row_idx, matched_within_idx)
    prefix_sel = apply_null_prefix(prefix_sel, matched_within_idx, null_prefix)

    logits = test_vector_head(item_codec_sel, u, prefix=prefix_sel)
    target_t = torch.tensor(np.stack(targets), dtype=torch.float32, device=device)
    mask_t = torch.tensor(np.stack(valid_masks), dtype=torch.float32, device=device)
    weight_t = torch.tensor(weights, dtype=torch.float32, device=device).unsqueeze(1)
    per_slot = torch.nn.functional.binary_cross_entropy_with_logits(logits, target_t, reduction="none")
    weighted_mask = mask_t * weight_t
    loss = (per_slot * weighted_mask).sum() / weighted_mask.sum().clamp(min=1)
    n_first = int(sum(1 for w in weights if oversample_ratio != 1.0 and abs(w - oversample_ratio) < 1e-9))
    return loss, {"n_matched": len(matched_pos), "n_first_attempt_matched": n_first}


def score_population_mu(attempt_encoder, filt_forward, test_vector_head, code_of_hash: dict,
                        diff_of_hash: dict, codec_dim: int, attempts: pd.DataFrame,
                        episodes_full: pd.DataFrame, problem_codec, pid_to_idx, tv_labels: dict,
                        device: str, target_rows_per_chunk: int, first_attempt_only: bool,
                        label: str, null_prefix: "torch.Tensor | None" = None) -> list[dict]:
    """Scores the head at the belief mean mu over `attempts`, in learner-aligned chunks. Returns a
    list of per-chunk prediction records carrying BOTH `pred_prob` (post-sigmoid) and
    `logit` (pre-sigmoid, needed by the base-rate blend) -- same row-order/shape
    convention `test_vector_eval_predictions` uses, extended by one field."""
    chunks = learner_aligned_chunks(attempts, target_rows_per_chunk)
    all_preds = []
    lookup = tv_labels["lookup"]
    t0 = time.time()
    for ci, learners in enumerate(chunks):
        at = attempts[attempts["learner"].isin(set(learners))].reset_index(drop=True)
        if first_attempt_only:
            at = at[at["attempt_within_idx"] == 0].reset_index(drop=True)
        if len(at) == 0:
            continue
        episodes_batch = episodes_full[episodes_full["learner"].isin(set(learners))]

        matched_pos, targets, valid_masks = [], [], []
        for i, row in at.iterrows():
            key = (str(row["learner"]), int(row["problem_id"]), str(row["code_state_id"]))
            if key not in lookup:
                continue
            rec = lookup[key]
            matched_pos.append(i)
            targets.append(rec["test_vector"].astype(np.float32))
            valid_masks.append(rec["valid_mask"])
        if not matched_pos:
            continue

        with torch.no_grad():
            attempt_batch = build_attempt_batch(at, code_of_hash, diff_of_hash, codec_dim, codec_dim, device)
            prefix_states, summary = attempt_encoder(
                attempt_batch["code_vec"], attempt_batch["diff_vec"], attempt_batch["scalar_feats"],
                attempt_batch["lengths"])
            episode_ids_of_summary = attempt_batch["episode_ids"]
            ep_batch = build_episode_batch(episodes_batch, problem_codec, pid_to_idx, summary,
                                           episode_ids_of_summary, device)
            ep_learner_idx, ep_t_idx = _distinct_episode_positions(episodes_batch, ep_batch,
                                                                    episode_ids_of_summary, device)
            mu_sel_all, _log_sigma = gather_belief_for_batch(filt_forward, ep_batch, ep_learner_idx, ep_t_idx)
            item_codec_ep = gather_item_codec_for_batch(ep_batch, ep_learner_idx, ep_t_idx)

            row_idx_t = attempt_batch["attempt_episode_row_idx"]
            matched_pos_t = torch.tensor(matched_pos, dtype=torch.long, device=device)
            matched_row_idx = row_idx_t[matched_pos_t]
            matched_within_idx = attempt_batch["attempt_within_idx"][matched_pos_t]

            u_mu = mu_sel_all[matched_row_idx]
            item_codec_sel = item_codec_ep[matched_row_idx]
            prefix_sel = gather_prefix_for_attempts(prefix_states, matched_row_idx, matched_within_idx)
            prefix_sel = apply_null_prefix(prefix_sel, matched_within_idx, null_prefix)
            logits_t = test_vector_head(item_codec_sel, u_mu, prefix=prefix_sel)
            probs = torch.sigmoid(logits_t).cpu().numpy()
            logit_np = logits_t.cpu().numpy()

        rows_matched = at.iloc[matched_pos]
        all_preds.append({
            "learner": rows_matched["learner"].to_numpy(), "problem_id": rows_matched["problem_id"].to_numpy(),
            "code_state_id": rows_matched["code_state_id"].to_numpy(),
            "pred_prob": probs, "logit": logit_np, "target": np.stack(targets), "valid_mask": np.stack(valid_masks),
        })
        print(f"[tv-refresh:{label}] chunk {ci + 1}/{len(chunks)} ({len(learners)} learners, "
             f"{len(matched_pos)} matched rows), {time.time() - t0:.0f}s elapsed", flush=True)
    return all_preds


def train_test_vector_head(attempts: pd.DataFrame, episodes: pd.DataFrame, code_of_hash: dict, diff_of_hash: dict,
                           problem_codec, pid_to_idx, train_learners: list, tv_labels: dict, d: dict,
                           filt_forward, attempt_encoder, attempt_encoder_snapshot: dict, filt, filt_snapshot: dict,
                           oversample_ratio: float, epochs: int, lr: float, seed: int, learner_batch_size: int,
                           use_null_prefix: bool, device: str, label: str) -> dict:
    """One full training run with the `cw_sim_head_refresh.py` recipe, optionally with
    the learned null-prefix embedding. Byte identity of attempt_encoder/filt is asserted
    at the end for every arm; the null-prefix substitution happens entirely on the gathered
    prefix tensor, never inside `attempt_encoder`/`filt`, so it cannot affect this check.
    """
    seed_everything(seed)
    test_vector_head = OutcomeHeadItemMLP(codec_dim=d["codec_dim"], latent_dim=d["latent_dim"],
                                          hidden=d["attempt_hidden"], n_classes=tv_labels["max_test_cases"],
                                          prefix_dim=d["attempt_hidden"]).to(device)
    null_prefix = None
    params = list(test_vector_head.parameters())
    if use_null_prefix:
        null_prefix = torch.nn.Parameter(torch.zeros(d["attempt_hidden"], device=device))
        params = params + [null_prefix]
    optimizer = torch.optim.Adam(params, lr=lr)

    rng = np.random.default_rng(seed)
    n_batches_per_epoch = max(1, (len(train_learners) + learner_batch_size - 1) // learner_batch_size)
    t0 = time.time()
    step = 0
    loss_history: list[float] = []
    for epoch in range(epochs):
        order = rng.permutation(len(train_learners))
        for bstart in range(0, len(train_learners), learner_batch_size):
            batch_learners = {train_learners[i] for i in order[bstart:bstart + learner_batch_size]}
            attempts_batch = attempts[attempts["learner"].isin(batch_learners)]
            episodes_batch = episodes[episodes["learner"].isin(batch_learners)]

            optimizer.zero_grad()
            attempt_batch = build_attempt_batch(attempts_batch, code_of_hash, diff_of_hash,
                                                d["codec_dim"], d["codec_dim"], device)
            prefix_states, summary = attempt_encoder(
                attempt_batch["code_vec"], attempt_batch["diff_vec"], attempt_batch["scalar_feats"],
                attempt_batch["lengths"])
            episode_ids_of_summary = attempt_batch["episode_ids"]
            ep_batch = build_episode_batch(episodes_batch, problem_codec, pid_to_idx, summary,
                                           episode_ids_of_summary, device)
            ep_learner_idx, ep_t_idx = _distinct_episode_positions(
                episodes_batch, ep_batch, episode_ids_of_summary, device)

            loss, stats = weighted_test_vector_batch_loss(
                filt_forward, test_vector_head, ep_batch, ep_learner_idx, ep_t_idx, prefix_states,
                attempt_batch, attempts_batch, tv_labels, oversample_ratio, device, null_prefix=null_prefix)
            if loss is None:
                continue
            if not torch.isfinite(loss):
                raise RuntimeError(f"[tv-refresh:{label}] TRIPWIRE non-finite loss at step={step}: {loss.item()}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optimizer.step()
            loss_history.append(loss.item())
            step += 1
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"[tv-refresh:{label}] epoch={epoch + 1}/{epochs} L={loss_history[-1]:.4f}", flush=True)

    d0 = max(1, n_batches_per_epoch)
    loss_first = float(np.mean(loss_history[:d0]))
    loss_last = float(np.mean(loss_history[-d0:]))
    elapsed = time.time() - t0
    print(f"[tv-refresh:{label}] done in {elapsed:.1f}s: L first-epoch-mean={loss_first:.4f} "
         f"last-epoch-mean={loss_last:.4f}", flush=True)

    assert_frozen(attempt_encoder, attempt_encoder_snapshot, "attempt_encoder")
    assert_frozen(filt, filt_snapshot, "filt")
    print(f"[tv-refresh:{label}] frozen-weights check PASS: attempt_encoder + filt byte-identical before/after", flush=True)

    return {"test_vector_head": test_vector_head, "null_prefix": null_prefix,
           "loss_first_epoch_mean": loss_first, "loss_last_epoch_mean": loss_last, "elapsed_s": elapsed}


# --------------------------------------------------------------------------------------
# Base-rate baseline + two-parameter logistic blend
# --------------------------------------------------------------------------------------

def compute_first_attempt_base_rate_table(attempts: pd.DataFrame, tv_labels: dict, train_learners: set,
                                          max_test_cases: int) -> tuple[dict, float, dict]:
    """Per-(problem_id, test-case slot) empirical pass rate from TRAIN first-attempts
    only, the item-statistics baseline. Returns `(table, global_rate, stats)`;
    `table[pid]` is a `[max_test_cases]` array, NaN at slots never seen in train."""
    lookup = tv_labels["lookup"]
    tr = attempts[attempts["learner"].isin(train_learners) & (attempts["attempt_within_idx"] == 0)]
    sum_by_pid: dict[int, np.ndarray] = {}
    mask_by_pid: dict[int, np.ndarray] = {}
    n_matched = 0
    for _, row in tr.iterrows():
        key = (str(row["learner"]), int(row["problem_id"]), str(row["code_state_id"]))
        if key not in lookup:
            continue
        rec = lookup[key]
        pid = int(row["problem_id"])
        t = rec["test_vector"].astype(np.float64)
        m = rec["valid_mask"].astype(np.float64)
        sum_by_pid.setdefault(pid, np.zeros(max_test_cases))
        mask_by_pid.setdefault(pid, np.zeros(max_test_cases))
        sum_by_pid[pid] += t * m
        mask_by_pid[pid] += m
        n_matched += 1
    table: dict[int, np.ndarray] = {}
    for pid in sum_by_pid:
        with np.errstate(invalid="ignore", divide="ignore"):
            table[pid] = np.where(mask_by_pid[pid] > 0, sum_by_pid[pid] / np.maximum(mask_by_pid[pid], 1), np.nan)
    total_sum = sum(sum_by_pid.values()) if sum_by_pid else np.zeros(max_test_cases)
    total_mask = sum(mask_by_pid.values()) if mask_by_pid else np.zeros(max_test_cases)
    global_rate = float(total_sum.sum() / total_mask.sum()) if total_mask.sum() > 0 else 0.5
    stats = {"n_first_attempt_matched_train": n_matched, "n_problems_covered": len(table),
             "global_rate": global_rate}
    return table, global_rate, stats


def lookup_base_rate(pid: int, slot: int, table: dict, global_rate: float) -> tuple[float, str]:
    """Returns `(rate, fallback_level)` -- `fallback_level` in {"slot", "problem",
    "global"}, so fallback usage is countable, never silently absorbed."""
    arr = table.get(pid)
    if arr is None:
        return global_rate, "global"
    v = arr[slot]
    if np.isnan(v):
        valid = arr[~np.isnan(arr)]
        if len(valid):
            return float(valid.mean()), "problem"
        return global_rate, "global"
    return float(v), "slot"


def base_rate_predictions(preds: list[dict], table: dict, global_rate: float, max_test_cases: int) -> tuple[list[dict], dict]:
    """Same row order/shape as `preds`, `pred_prob`/`logit` replaced by the train-derived
    base-rate lookup. Fallback usage counted and returned alongside."""
    out = []
    fallback_counts = {"slot": 0, "problem": 0, "global": 0}
    eps = 1e-6
    for p in preds:
        n = len(p["problem_id"])
        rate_arr = np.zeros((n, max_test_cases), dtype=np.float32)
        for i, pid in enumerate(p["problem_id"]):
            for slot in range(max_test_cases):
                rate, level = lookup_base_rate(int(pid), slot, table, global_rate)
                rate_arr[i, slot] = rate
                fallback_counts[level] += 1
        clipped = np.clip(rate_arr, eps, 1 - eps)
        logit_arr = np.log(clipped / (1 - clipped)).astype(np.float32)
        out.append({**p, "pred_prob": rate_arr, "logit": logit_arr})
    return out, fallback_counts


def fit_two_param_logistic_blend(preds_head: list[dict], preds_base_rate: list[dict],
                                 seed: int = 0) -> tuple[float, float, dict]:
    """`sigmoid(w1*head_logit + w2*base_rate_logit)`, NO intercept -- exactly two
    learned scalars, fit on TRAIN first-attempts only (`preds_head`/`preds_base_rate`
    must both be TRAIN-population predictions, same row order)."""
    from sklearn.linear_model import LogisticRegression

    logit_head = np.concatenate([p["logit"] for p in preds_head], axis=0) if preds_head else np.zeros((0, 0))
    logit_base = np.concatenate([p["logit"] for p in preds_base_rate], axis=0) if preds_base_rate else np.zeros((0, 0))
    target = np.concatenate([p["target"] for p in preds_head], axis=0) if preds_head else np.zeros((0, 0))
    mask = np.concatenate([p["valid_mask"] for p in preds_head], axis=0).astype(bool) if preds_head else np.zeros((0, 0), dtype=bool)
    if mask.sum() == 0:
        return 1.0, 0.0, {"n_fit_rows": 0, "degenerate": True}
    xh, xb, y = logit_head[mask], logit_base[mask], target[mask]
    if len(np.unique(y)) < 2:
        return 1.0, 0.0, {"n_fit_rows": int(mask.sum()), "degenerate": True}
    clf = LogisticRegression(fit_intercept=False, max_iter=1000, random_state=seed)
    clf.fit(np.stack([xh, xb], axis=1), y)
    w1, w2 = float(clf.coef_[0][0]), float(clf.coef_[0][1])
    return w1, w2, {"n_fit_rows": int(mask.sum()), "degenerate": False}


def blend_predictions(preds_head: list[dict], preds_base_rate: list[dict], w1: float, w2: float) -> list[dict]:
    out = []
    for ph, pb in zip(preds_head, preds_base_rate):
        logits = w1 * ph["logit"] + w2 * pb["logit"]
        probs = 1.0 / (1.0 + np.exp(-logits))
        out.append({**ph, "pred_prob": probs.astype(np.float32), "logit": logits.astype(np.float32)})
    return out


def ci_for_preds(preds: list[dict], n_boot: int = 1000, seed: int = 0) -> dict:
    """Pools `preds` (per-slot [n_rows, max_test_cases] arrays) into flat (target, prob,
    learner) triples, masked, and returns a learner-clustered AUC + 95% CI."""
    if not preds:
        return {"auc": None, "ci95": [None, None], "n_pairs": 0, "n_learners": 0}
    targets = np.concatenate([p["target"] for p in preds], axis=0)
    probs = np.concatenate([p["pred_prob"] for p in preds], axis=0)
    masks = np.concatenate([p["valid_mask"] for p in preds], axis=0).astype(bool)
    learners = np.concatenate([p["learner"] for p in preds], axis=0)
    learners_tiled = np.repeat(learners, targets.shape[1])
    flat_mask = masks.ravel()
    n_pairs = int(flat_mask.sum())
    if n_pairs == 0:
        return {"auc": None, "ci95": [None, None], "n_pairs": 0, "n_learners": 0}
    result = learner_clustered_auc_ci(targets.ravel()[flat_mask], probs.ravel()[flat_mask],
                                      learners_tiled[flat_mask], n_boot=n_boot, seed=seed)
    result["n_pairs"] = n_pairs
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="results/cw_sim_base_run/final.pt")
    ap.add_argument("--raw-dir", default=None)
    for name in ("code_map", "diff_map", "codealone_meta", "codealone_states", "diff_meta", "diff_states"):
        ap.add_argument("--" + name.replace("_", "-"), required=True)
    ap.add_argument("--test-vector-labels", default=str(REPO / "artifacts" / "cw_sim_test_vector_labels.npz"))
    ap.add_argument("--learner-batch-size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--target-rows-per-chunk", type=int, default=256)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--out-dir", default="results/cw_sim_tv_head_refresh")
    ns = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = REPO / ns.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    t_all0 = time.time()

    manifest_args = _Args()
    manifest_args.raw_dir = ns.raw_dir
    manifest_args.code_map, manifest_args.diff_map = ns.code_map, ns.diff_map
    manifest_args.codealone_meta, manifest_args.codealone_states = ns.codealone_meta, ns.codealone_states
    manifest_args.diff_meta, manifest_args.diff_states = ns.diff_meta, ns.diff_states

    print("[tv-refresh] loading manifest (S19+F19 train+val, lockbox excluded)...", flush=True)
    base_manifest = load_manifest(manifest_args)
    # `cw_4_hierarchical_model.load_manifest`'s attempts carry no row-level
    # `attempt_within_idx` and no `code` column (`include_code=False`), so
    # `cw_manifest.attach_prior_attempt_code` (which also needs `code`) is not reused;
    # only its `attempt_within_idx` definition is (groupby-cumcount within `episode_id`).
    base_manifest["attempts"] = base_manifest["attempts"].copy()
    base_manifest["attempts"]["attempt_within_idx"] = (
        base_manifest["attempts"].groupby("episode_id", sort=False).cumcount())
    transform = CodecTransform.load(TRANSFORM_PATH)
    code_hashes = set(base_manifest["attempts"]["code_hash"].dropna())
    diff_hashes = set(base_manifest["attempts"].loc[base_manifest["attempts"]["diff_hash"].notna(), "diff_hash"])
    code_of_hash = build_codec_vector_lookup(code_hashes, ns.codealone_meta, ns.codealone_states,
                                             transform, k=CODEC_K, rank=RANK)
    diff_of_hash = build_codec_vector_lookup(diff_hashes, ns.diff_meta, ns.diff_states,
                                             transform, k=CODEC_K, rank=RANK)
    problem_codec, pid_to_idx = rebuild_problem_codec_at_rank(RANK, transform=transform)
    if len(problem_codec) != 50 or len(pid_to_idx) != 50:
        raise AssertionError(f"the fixed problem set requires all 50 problems, got {len(problem_codec)}")
    manifest = {**base_manifest, "code_of_hash": code_of_hash, "diff_of_hash": diff_of_hash,
               "problem_codec": problem_codec, "pid_to_idx": pid_to_idx}
    attempts, episodes = manifest["attempts"], manifest["episodes"]
    train_learners = sorted(manifest["train_learners"])
    val_learners = set(manifest["val_learners"])
    print(f"[tv-refresh] n_attempts={len(attempts)} n_train_learners={len(train_learners)} "
         f"n_val_learners={len(val_learners)}", flush=True)

    tv_labels = load_test_vector_labels(pathlib.Path(ns.test_vector_labels))
    max_test_cases = tv_labels["max_test_cases"]
    print(f"[tv-refresh] test-vector labels: {len(tv_labels['lookup'])} rows, "
         f"{len(tv_labels['headline_problem_ids'])} headline problems, max_test_cases={max_test_cases}", flush=True)

    oversample_ratio, oversample_stats = compute_first_attempt_oversample_ratio(
        attempts, tv_labels, set(train_learners))
    print(f"[tv-refresh] fixed first-attempt oversample ratio: {oversample_ratio} "
         f"({json.dumps(oversample_stats)})", flush=True)

    ckpt_path = REPO / ns.checkpoint
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    d = ckpt["architecture"]
    ratchet_cwsim_sha256 = _sha256(ckpt_path)
    assert ratchet_cwsim_sha256 == EXPECTED_BASE_SHA256, (
        f"base checkpoint sha mismatch: got {ratchet_cwsim_sha256}, expected {EXPECTED_BASE_SHA256} -- "
        "refusing to train on an unexpected base checkpoint")

    attempt_encoder = AttemptEncoder(d["codec_dim"], d["codec_dim"], d["attempt_proj_dim"],
                                     d["attempt_hidden"], d["summary_dim"]).to(device)
    filt = S1FilterV2ItemMLP(codec_dim=d["codec_dim"], episode_hidden=d["episode_hidden"],
                             filter_hidden=d["filter_hidden"], latent_dim=d["latent_dim"],
                             prev_feat_dim=d["summary_dim"], sigma_parameterization="precision_accumulating"
                             ).to(device)
    attempt_encoder.load_state_dict(ckpt["attempt_encoder"])
    filt.load_state_dict(ckpt["filt"])
    attempt_encoder_snapshot = freeze(attempt_encoder)
    filt_snapshot = freeze(filt)
    filt_forward = lambda ic, pf, tm, vm: filt(ic, pf)  # noqa: E731 -- same convention as cw_sim_head_refresh.py

    assert "test_vector_head" in ckpt, "base checkpoint has no test_vector_head -- cannot build the 'old head' arm"
    old_head = OutcomeHeadItemMLP(codec_dim=d["codec_dim"], latent_dim=d["latent_dim"],
                                  hidden=d["attempt_hidden"], n_classes=ckpt.get("max_test_cases", max_test_cases),
                                  prefix_dim=d["attempt_hidden"]).to(device)
    old_head.load_state_dict(ckpt["test_vector_head"])
    old_head.eval()
    print("[tv-refresh] 'old head' arm loaded unmodified from the base checkpoint", flush=True)

    # ---- Train the two refreshed arms ------------------------------------------------
    arm_standard = train_test_vector_head(
        attempts, episodes, code_of_hash, diff_of_hash, problem_codec, pid_to_idx, train_learners, tv_labels, d,
        filt_forward, attempt_encoder, attempt_encoder_snapshot, filt, filt_snapshot, oversample_ratio,
        ns.epochs, ns.lr, ns.seed, ns.learner_batch_size, use_null_prefix=False, device=device, label="refreshed")
    arm_null_prefix = train_test_vector_head(
        attempts, episodes, code_of_hash, diff_of_hash, problem_codec, pid_to_idx, train_learners, tv_labels, d,
        filt_forward, attempt_encoder, attempt_encoder_snapshot, filt, filt_snapshot, oversample_ratio,
        ns.epochs, ns.lr, ns.seed, ns.learner_batch_size, use_null_prefix=True, device=device,
        label="refreshed+null-prefix")

    heads = {
        "old": (old_head, None), "refreshed": (arm_standard["test_vector_head"], None),
        "refreshed_null_prefix": (arm_null_prefix["test_vector_head"], arm_null_prefix["null_prefix"]),
    }

    # ---- Base-rate table (train first-attempts) ---------------------------------------
    base_rate_table, global_rate, base_rate_stats = compute_first_attempt_base_rate_table(
        attempts, tv_labels, set(train_learners), max_test_cases)
    print(f"[tv-refresh] base-rate table: {json.dumps(base_rate_stats)}", flush=True)

    # ---- Score TRAIN first-attempts under every head (for blend fitting) --------------
    train_attempts = attempts[attempts["learner"].isin(set(train_learners))].reset_index(drop=True)
    train_preds_by_head = {}
    for name, (head, null_prefix) in heads.items():
        train_preds_by_head[name] = score_population_mu(
            attempt_encoder, filt_forward, head, code_of_hash, diff_of_hash, d["codec_dim"], train_attempts,
            episodes, problem_codec, pid_to_idx, tv_labels, device, ns.target_rows_per_chunk,
            first_attempt_only=True, label=f"train-first-{name}", null_prefix=null_prefix)
    train_preds_base_rate, train_fallback_counts = base_rate_predictions(
        train_preds_by_head["old"], base_rate_table, global_rate, max_test_cases)

    blend_weights = {}
    for name in heads:
        w1, w2, fit_stats = fit_two_param_logistic_blend(train_preds_by_head[name], train_preds_base_rate, ns.seed)
        blend_weights[name] = {"w1_head": w1, "w2_base_rate": w2, **fit_stats}
        print(f"[tv-refresh] blend({name}): w1={w1:.4f} w2={w2:.4f} ({json.dumps(fit_stats)})", flush=True)

    # ---- Score VAL first-attempts under every arm --------------------------------------
    val_attempts = attempts[attempts["learner"].isin(val_learners)].reset_index(drop=True)
    val_preds_by_head = {}
    for name, (head, null_prefix) in heads.items():
        val_preds_by_head[name] = score_population_mu(
            attempt_encoder, filt_forward, head, code_of_hash, diff_of_hash, d["codec_dim"], val_attempts,
            episodes, problem_codec, pid_to_idx, tv_labels, device, ns.target_rows_per_chunk,
            first_attempt_only=True, label=f"val-first-{name}", null_prefix=null_prefix)
    val_preds_base_rate, val_fallback_counts = base_rate_predictions(
        val_preds_by_head["old"], base_rate_table, global_rate, max_test_cases)

    arms_table = {"base_rate": ci_for_preds(val_preds_base_rate, ns.n_boot, ns.seed)}
    for name in heads:
        arms_table[name] = ci_for_preds(val_preds_by_head[name], ns.n_boot, ns.seed)
        w1, w2 = blend_weights[name]["w1_head"], blend_weights[name]["w2_base_rate"]
        blended = blend_predictions(val_preds_by_head[name], val_preds_base_rate, w1, w2)
        arms_table[f"blend_{name}"] = ci_for_preds(blended, ns.n_boot, ns.seed)
    print(f"[tv-refresh] IN-CORPUS VAL first-attempt arms table: {json.dumps(arms_table, indent=2)}", flush=True)

    # ---- Continuity numbers: refreshed head, mu, ALL-ATTEMPTS + FIRST-ATTEMPT ---------
    preds_all_refreshed = score_population_mu(
        attempt_encoder, filt_forward, heads["refreshed"][0], code_of_hash, diff_of_hash, d["codec_dim"],
        val_attempts, episodes, problem_codec, pid_to_idx, tv_labels, device, ns.target_rows_per_chunk,
        first_attempt_only=False, label="in-corpus-all-refreshed")
    panel_all = test_vector_auc_panel(preds_all_refreshed)
    print(f"[tv-refresh] in-corpus ALL-ATTEMPTS refreshed-head mu: pooled_auc={panel_all['pooled_auc']} "
         f"n_pairs={panel_all['n_pairs']}", flush=True)

    # ---- Persist checkpoints + shared artifacts for the TIKTOC companion script -------
    def _ckpt_dict(name: str, head, null_prefix, recipe_extra: dict) -> dict:
        out = {
            "architecture": d,
            "attempt_encoder": {k: v.detach().cpu() for k, v in attempt_encoder.state_dict().items()},
            "filt": {k: v.detach().cpu() for k, v in filt.state_dict().items()},
            "test_vector_head": {k: v.detach().cpu() for k, v in head.state_dict().items()},
            "max_test_cases": max_test_cases, "sigma_parameterization": "precision_accumulating",
            "source": f"ratchet-cwsim/{name}",
            "ratchet_cwsim_checkpoint": ns.checkpoint, "ratchet_cwsim_checkpoint_sha256": ratchet_cwsim_sha256,
            "tv_head_refresh_recipe": {
                "epochs": ns.epochs, "lr": ns.lr, "seed": ns.seed, "learner_batch_size": ns.learner_batch_size,
                "test_vector_loss_weight": 1.0, "optimizer": "Adam", "filter_frozen": True,
                "first_attempt_oversample_ratio": oversample_ratio, "oversample_stats": oversample_stats,
                **recipe_extra,
            },
            "git_sha": _git_sha(),
        }
        if null_prefix is not None:
            out["null_prefix_embedding"] = null_prefix.detach().cpu()
        return out

    out_path_standard = out_dir / "final.pt"
    torch.save(_ckpt_dict("refreshed-tv-head", arm_standard["test_vector_head"], None,
                          {"loss_first_epoch_mean": arm_standard["loss_first_epoch_mean"],
                           "loss_last_epoch_mean": arm_standard["loss_last_epoch_mean"],
                           "null_prefix_embedding": False}), out_path_standard)
    out_path_null_prefix = out_dir / "final_null_prefix.pt"
    torch.save(_ckpt_dict("refreshed-tv-head-null-prefix", arm_null_prefix["test_vector_head"],
                          arm_null_prefix["null_prefix"],
                          {"loss_first_epoch_mean": arm_null_prefix["loss_first_epoch_mean"],
                           "loss_last_epoch_mean": arm_null_prefix["loss_last_epoch_mean"],
                           "null_prefix_embedding": True}), out_path_null_prefix)

    np.savez(out_dir / "base_rate_table.npz",
            problem_ids=np.array(list(base_rate_table.keys())),
            rates=np.stack(list(base_rate_table.values())) if base_rate_table else np.zeros((0, max_test_cases)),
            global_rate=global_rate, max_test_cases=max_test_cases)
    (out_dir / "blend_weights.json").write_text(json.dumps(blend_weights, indent=2))

    manifest_out = {
        "checkpoint_standard": str(out_path_standard.relative_to(REPO)), "sha256_standard": _sha256(out_path_standard),
        "checkpoint_null_prefix": str(out_path_null_prefix.relative_to(REPO)),
        "sha256_null_prefix": _sha256(out_path_null_prefix),
        "ratchet_cwsim_checkpoint_sha256": ratchet_cwsim_sha256,
        "oversample_ratio": oversample_ratio, "oversample_stats": oversample_stats,
        "base_rate_stats": base_rate_stats, "blend_weights": blend_weights,
        "train_fallback_counts": train_fallback_counts, "val_fallback_counts": val_fallback_counts,
        "canary_6": "PASS -- attempt_encoder + filt byte-identical before/after, both arms",
        "elapsed_s_total": time.time() - t_all0,
        "elapsed_s_standard": arm_standard["elapsed_s"], "elapsed_s_null_prefix": arm_null_prefix["elapsed_s"],
        "in_corpus_val_first_attempt_arms_table": arms_table,
        "in_corpus_val_all_attempts_refreshed_head": panel_all,
        # old-head mu AUCs measured earlier with the same scoring, for before/after comparison:
        "old_head_mu_reference": {"all_attempts_auc": 0.7437995572277897, "first_attempt_auc": 0.5916337930315183},
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest_out, indent=2, allow_nan=True))
    print(json.dumps(manifest_out, indent=2, allow_nan=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
