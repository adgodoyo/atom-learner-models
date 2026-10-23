#!/usr/bin/env python3
"""Trains and evaluates the CodeWorkout generative simulator (Table 11).

Belief mu -> hypernetwork -> LoRA on the frozen gemma4-e4b code backbone generates the
learner's next-attempt code, conditioned on problem, belief and true outcome class; a
test-vector head (sigmoid per slot, BCE masked to each problem's case count) predicts
per-test-case pass/fail on the 13 join-verified TIKTOC headline problems (of 17; see
`scripts/cw_5_functional_equivalence.py`). The belief comes from the CodeWorkout
classifier's filter (`ratchet-original`), frozen (`--filter-mode frozen`) or
warm-started and trained jointly with its outcome loss at `--outcome-loss-weight`
(`--filter-mode joint`). Inputs: CodeWorkout manifest and code caches, the
sha256-pinned classifier checkpoint, `artifacts/cw_sim_test_vector_labels.npz`.
Outputs under `--out-dir`: `step*.pt`/`final.pt`, `metrics.jsonl`, `eval_blocks/`,
`predictions.npz`, `result.json` (personalisation gap with paired learner-clustered CI;
with `--full-eval` also strata, test-vector AUC, CodeBLEU) and `generations.jsonl` (raw
code, local only, never uploaded). Run: `python infra/cw_sim_full_run.py --filter-mode
{frozen,joint} --max-steps N --out-dir DIR --code-map ... [--full-eval] [--resume]
[--bucket NAME]` (optional S3 upload of numeric outputs). CodeBLEU needs the vendored
x86-64 Linux tree-sitter grammar and is reported as unavailable elsewhere.
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import torch

REPO = pathlib.Path(__file__).resolve().parents[1]
for p in (REPO / "src", REPO / "infra", REPO / "scripts", REPO):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import cw_sim_core as cw5  # noqa: E402

from atom.eval.cw_generation import (  # noqa: E402
    numeric_generation_row, save_numeric_generation_rows, summarise_generation_rows,
)
from atom.eval.codebleu import codebleu_available, codebleu_one  # noqa: E402
from atom.simulator.cw_item_mlp import OutcomeHeadItemMLP  # noqa: E402

MAX_TEST_CASES = 32
RATCHET_CHECKPOINT_DEFAULT = REPO / "results" / "cw_sigma_precision_0305a22" / "aws" / "results" / "final.pt"
RATCHET_CHECKPOINT_SHA256 = "8d22b6914e041dfa88bf911b1a498fa0c87af30488b07f6e49a72adca731c544"
SIGMA_PARAMETERIZATION = "precision_accumulating"


def _between_learner_variance_share(x: np.ndarray, learner_group: np.ndarray) -> float:
    """Fraction of `x`'s total variance explained by between-learner group means.
    Duplicated here rather than imported, because infra scripts are not an importable
    package."""
    df = pd.DataFrame(x)
    df["_g"] = learner_group
    total_var = df.drop(columns="_g").var(ddof=0).to_numpy()
    between_var = df.groupby("_g").mean().var(ddof=0).to_numpy()
    return float(np.mean(between_var / np.clip(total_var, 1e-12, None)))


@torch.no_grad()
def belief_dependence_canary9(m: dict, mu_sel: torch.Tensor, log_sigma_sel: torch.Tensor,
                              learner_group: np.ndarray, device: str, seed: int,
                              k_eps: int = 16) -> dict:
    """Belief-dependence check: a personalised adapter is only personalised if its
    output depends on the belief. This model has a hypernetwork adapter (`m["hyper"]`)
    between the belief and every downstream effect, so the check tests the adapter's own
    output `c = hyper(u)` directly rather than using the belief as a proxy for it.

    `learner_group`: any array of grouping keys distinguishing learners within
    `mu_sel`'s rows -- the training loop's own `ep_learner_idx` (an integer index into
    this batch's learners, not a resolved learner-ID string) is sufficient, since the
    canary only needs a consistent partition, never real identities.

    PASS iff `mu`'s between-learner variance share > 0 AND the adapter's
    across-learner spread exceeds its across-epsilon spread: a run whose across-eps
    spread exceeds its across-learner spread is training a global adapter on noise,
    whatever its gain over ΔW=0."""
    learner_group = np.asarray(learner_group)
    c_mean = m["hyper"](mu_sel)
    c_np = c_mean.detach().cpu().numpy()
    mean_norm = max(float(np.linalg.norm(c_np, axis=-1).mean()), 1e-12)

    df = pd.DataFrame(c_np)
    df["_g"] = learner_group
    across_learner_spread = float(df.groupby("_g").mean().std(ddof=0).mean())

    gen = torch.Generator(device=device).manual_seed(seed)
    u_k = cw5.reparameterize_from_log_sigma(mu_sel, log_sigma_sel, k=k_eps, generator=gen)  # [N, K, D]
    n, k, d = u_k.shape
    c_k = m["hyper"](u_k.reshape(n * k, d)).reshape(n, k, -1)
    across_eps_spread = float(c_k.std(dim=1).mean().item())

    mean_sigma = float(log_sigma_sel.exp().mean().item())
    mu_np = mu_sel.detach().cpu().numpy()
    between_learner_share = _between_learner_variance_share(mu_np, learner_group)

    passed = bool(between_learner_share > 0.0 and across_learner_spread > across_eps_spread)
    return {
        "across_learner_spread": across_learner_spread, "across_eps_spread": across_eps_spread,
        "mean_adapter_output_norm": mean_norm,
        "across_learner_spread_over_norm": across_learner_spread / mean_norm,
        "across_eps_spread_over_norm": across_eps_spread / mean_norm,
        "mean_sigma": mean_sigma, "prior_sigma": 1.0,
        "mu_between_learner_variance_share": between_learner_share,
        "pass": passed,
        "rule": "share > 0 AND across-learner spread > across-eps spread (not training on noise)",
    }


def assert_belief_junction_gradient(m: dict) -> dict:
    """Belief-junction gradient guard: asserts that a non-zero, finite gradient reached
    the filter's mu head and its width-producing head from the simulator losses, after
    a real backward pass -- not merely that training ran without error. The width head
    is the sharper test: it only receives gradient through the reparameterisation, so a
    non-reparameterised draw would freeze the belief's width silently while everything
    else appears to converge.

    `m["filt"]` (`S1FilterV2ItemMLP`) nests the actual heads under `.knowledge_filter`
    (`src/atom/filter/gru.py::KnowledgeFilter`), not on itself (unlike the standalone
    GRU/event-transformer filter bodies, which expose `mu_head` directly). The width
    head's name depends on `sigma_parameterization`: `log_sigma_head` under `"direct"`,
    `precision_head` under `"precision_accumulating"` (this model's configuration,
    `SIGMA_PARAMETERIZATION` above) -- `precision_head` is exactly as sharp a
    reparameterisation-only dependency as `log_sigma_head` (both only affect `log_sigma`,
    never `mu`, per `KnowledgeFilter.forward`), so the same guard applies unchanged.

    Call ONCE, right after the first joint-mode training step's backward calls
    complete, before `optimizer.zero_grad()`/`.step()` clears anything."""
    kf = getattr(m["filt"], "knowledge_filter", m["filt"])
    width_head_name = "log_sigma_head" if hasattr(kf, "log_sigma_head") else "precision_head"
    grads = {}
    for name in ("mu_head", width_head_name):
        head = getattr(kf, name)
        norms = [p.grad.norm().item() for p in head.parameters() if p.grad is not None]
        grads[name] = float(sum(norms))
    ok = all(np.isfinite(v) and v > 0.0 for v in grads.values())
    if not ok:
        raise RuntimeError(
            f"[cw-sim] LAMBDA_PROBE_GUARD failed: belief-junction gradient must be "
            f"finite and > 0 at both mu_head and {width_head_name} after the first "
            f"joint training step, got {grads!r} -- refusing to proceed unattended"
        )
    return grads


def load_test_vector_labels(path: pathlib.Path) -> dict:
    """Labels are per attempt (as written by `scripts/cw_sim_test_vector_labels.py`):
    the lookup key is the full `(learner, problem_id, code_state_id)` triple, never
    `(learner, problem_id)` alone, which would match every attempt a learner made on a
    problem against one executed attempt's label. A file without a `code_state_id`
    array (a per-problem labels file) is refused outright."""
    d = np.load(path, allow_pickle=False)
    assert "code_state_id" in d, (
        f"{path} has no code_state_id column -- this is a per-problem labels file "
        "(one attempt's label applied to every attempt); regenerate with "
        "scripts/cw_sim_test_vector_labels.py before using it for training/eval"
    )
    # Materialise each array ONCE before the loop: `NpzFile.__getitem__` re-reads and
    # decompresses the WHOLE array from the zip archive on every access (it never caches),
    # so indexing `d["test_vector"][i]` etc. inside a per-row loop is O(n) full-array
    # decompressions per row -- O(n^2) total. Invisible on a small synthetic labels set;
    # prohibitive (unbounded CPU and memory growth) on the real per-attempt file
    # (22,697 rows).
    learner_arr, problem_id_arr, code_state_id_arr = d["learner"], d["problem_id"], d["code_state_id"]
    test_vector_arr, valid_mask_arr = d["test_vector"], d["valid_mask"]
    n_cases_arr, compiled_arr = d["n_cases"], d["compiled"]
    lookup = {}
    for i in range(len(learner_arr)):
        key = (str(learner_arr[i]), int(problem_id_arr[i]), str(code_state_id_arr[i]))
        lookup[key] = {
            "test_vector": test_vector_arr[i], "valid_mask": valid_mask_arr[i],
            "n_cases": int(n_cases_arr[i]), "compiled": bool(compiled_arr[i]),
        }
    return {
        "lookup": lookup, "max_test_cases": int(d["max_test_cases"]),
        "headline_problem_ids": set(int(x) for x in d["headline_problem_ids"]),
    }


def load_and_freeze_ratchet(m: dict, checkpoint_path: pathlib.Path, learner_split_sha256: str,
                            freeze: bool, device: str,
                            sigma_parameterization: str = "precision_accumulating") -> dict:
    """Loads the CodeWorkout classifier's `attempt_encoder`/`filt` weights, after
    verifying that its learner-split hash matches this run's (a different learner split
    would silently train on validation learners). `freeze=True` (frozen arm): both
    modules -> `.eval()` + `requires_grad_(False)`, EXCLUDED from any optimizer. `freeze=
    False` (joint arm, warm start only): loaded but left trainable, so both pilot arms
    share one initial condition.

    The checkpoint's `sigma_parameterization` and `sampler_semantics` fields are also
    checked: checkpoints carry a sampler-semantics version field and loaders refuse
    anything that does not match the current sampler. The
    sha256 pin (checked by the caller before this function runs) protects against
    loading the WRONG FILE; this protects against the hardcoded sha256 CONSTANT itself
    later being repointed at an incompatible checkpoint whose sigma parameterisation or
    sampler semantics do not match what `build_model` was called with -- a mismatch the
    sha check alone cannot catch, since it only verifies bytes, never meaning."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert ckpt.get("learner_split_sha256") == learner_split_sha256, (
        f"ratchet checkpoint's learner_split_sha256 ({ckpt.get('learner_split_sha256')}) does not "
        f"match this run's split ({learner_split_sha256}) -- refusing to load, this would be a leak")
    assert ckpt.get("sigma_parameterization") == sigma_parameterization, (
        f"ratchet checkpoint's sigma_parameterization ({ckpt.get('sigma_parameterization')}) does not "
        f"match this run's ({sigma_parameterization}) -- refusing to load an incompatible belief head"
    )
    assert ckpt.get("sampler_semantics") == cw5.CURRENT_SAMPLER_SEMANTICS, (
        f"ratchet checkpoint's sampler_semantics ({ckpt.get('sampler_semantics')}) does not match "
        f"this codebase's current sampler ({cw5.CURRENT_SAMPLER_SEMANTICS}) -- a checkpoint "
        "trained under different reparameterize_from_log_sigma semantics must not be loaded "
        "under the current sampler; refusing rather than silently mis-scoring"
    )
    m["attempt_encoder"].load_state_dict(ckpt["attempt_encoder_state_dict"], strict=True)
    m["filt"].load_state_dict(ckpt["filter_state_dict"], strict=True)
    m["attempt_encoder"].to(device)
    m["filt"].to(device)
    if freeze:
        m["attempt_encoder"].eval()
        m["filt"].eval()
        for p in m["attempt_encoder"].parameters():
            p.requires_grad_(False)
        for p in m["filt"].parameters():
            p.requires_grad_(False)
    return ckpt


def test_vector_batch_loss(m: dict, attempts_batch: pd.DataFrame, out: dict, manifest: dict,
                           tv_labels: dict, device: str) -> tuple[torch.Tensor | None, dict]:
    """BCE-with-logits, masked to each matched row's real test-case count, over whatever
    subset of `attempts_batch` has (a) `problem_id` in the 13 join-verified headline
    problems and (b) a ground-truth label in `tv_labels` (the offline TIKTOC-executed
    pass -- not every attempt was executed, see the labelling script's own disclosed
    cap). Returns `(None, {"n_matched": 0})` if nothing in this batch matched, so the
    caller can skip the backward cleanly rather than backward a zero tensor every step."""
    at = attempts_batch.reset_index(drop=True)
    lookup = tv_labels["lookup"]
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
        return None, {"n_matched": 0}

    ep_learner_idx, ep_t_idx = cw5._distinct_episode_positions(
        manifest["episodes"][manifest["episodes"]["learner"].isin(set(at["learner"]))],
        out["ep_batch"], out["attempt_batch"]["episode_ids"], device=device)
    mu_sel, log_sigma_sel = cw5.gather_belief_for_batch(m["filt_forward"], out["ep_batch"], ep_learner_idx, ep_t_idx)
    item_codec_ep = cw5.gather_item_codec_for_batch(out["ep_batch"], ep_learner_idx, ep_t_idx)

    row_idx_t = out["attempt_batch"]["attempt_episode_row_idx"]
    matched_pos_t = torch.tensor(matched_pos, dtype=torch.long, device=device)
    matched_row_idx = row_idx_t[matched_pos_t]

    u = cw5.reparameterize_from_log_sigma(mu_sel, log_sigma_sel, k=1)[:, 0, :][matched_row_idx]
    item_codec_sel = item_codec_ep[matched_row_idx]
    prefix_sel = cw5.gather_prefix_for_attempts(
        out["prefix_states"], matched_row_idx, out["attempt_batch"]["attempt_within_idx"][matched_pos_t])

    logits = m["test_vector_head"](item_codec_sel, u, prefix=prefix_sel)  # [n, MAX_TEST_CASES]
    target_t = torch.tensor(np.stack(targets), dtype=torch.float32, device=device)
    mask_t = torch.tensor(np.stack(valid_masks), dtype=torch.float32, device=device)
    per_slot = torch.nn.functional.binary_cross_entropy_with_logits(logits, target_t, reduction="none")
    loss = (per_slot * mask_t).sum() / mask_t.sum().clamp(min=1)
    return loss, {"n_matched": len(matched_pos)}


def test_vector_eval_predictions(m: dict, attempts_batch: pd.DataFrame, out: dict, manifest: dict,
                                 tv_labels: dict, device: str) -> dict | None:
    """Predictions for the TIKTOC-comparable per-test-case AUC panel. This is
    `test_vector_batch_loss`'s eval-time sibling: same matching, same
    forward, but no backward and no reduction -- every valid (row, test-case-slot) pair
    is returned individually (predicted probability, target, and the row's OWN
    identity read straight off `attempts_batch` -- membership asserted BY VALUE, never
    a positional index) so the caller can pool across chunks and compute pooled AND
    per-problem AUC (`test_vector_auc_panel`, below) exactly once, at the end. Returns
    `None` if nothing in this batch matched (same convention as `test_vector_batch_loss`,
    so both functions can share one caller-side "skip if nothing matched" branch)."""
    at = attempts_batch.reset_index(drop=True)
    lookup = tv_labels["lookup"]
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
        return None

    ep_learner_idx, ep_t_idx = cw5._distinct_episode_positions(
        manifest["episodes"][manifest["episodes"]["learner"].isin(set(at["learner"]))],
        out["ep_batch"], out["attempt_batch"]["episode_ids"], device=device)
    mu_sel, log_sigma_sel = cw5.gather_belief_for_batch(m["filt_forward"], out["ep_batch"], ep_learner_idx, ep_t_idx)
    item_codec_ep = cw5.gather_item_codec_for_batch(out["ep_batch"], ep_learner_idx, ep_t_idx)

    row_idx_t = out["attempt_batch"]["attempt_episode_row_idx"]
    matched_pos_t = torch.tensor(matched_pos, dtype=torch.long, device=device)
    matched_row_idx = row_idx_t[matched_pos_t]

    with torch.no_grad():
        u = cw5.reparameterize_from_log_sigma(mu_sel, log_sigma_sel, k=1)[:, 0, :][matched_row_idx]
        item_codec_sel = item_codec_ep[matched_row_idx]
        prefix_sel = cw5.gather_prefix_for_attempts(
            out["prefix_states"], matched_row_idx, out["attempt_batch"]["attempt_within_idx"][matched_pos_t])
        logits = m["test_vector_head"](item_codec_sel, u, prefix=prefix_sel)
        probs = torch.sigmoid(logits).cpu().numpy()

    rows_matched = at.iloc[matched_pos]
    return {
        "learner": rows_matched["learner"].to_numpy(),
        "problem_id": rows_matched["problem_id"].to_numpy(),
        "code_state_id": rows_matched["code_state_id"].to_numpy(),
        "pred_prob": probs, "target": np.stack(targets), "valid_mask": np.stack(valid_masks),
    }


def test_vector_auc_panel(all_preds: list[dict]) -> dict:
    """Pools `test_vector_eval_predictions` records across every eval chunk into the
    test-vector panel: per-test-case AUC pooled (every valid (row, slot) pair across
    every problem, one ROC) and per problem (one ROC per problem_id on the 13 headline
    problems), via the shared `atom.eval.metrics.auc`. A group whose matched rows are
    single-class reports None (from NaN), not a silent 0.5/1.0 default."""
    from atom.eval.metrics import auc as _shared_auc

    if not all_preds:
        return {"pooled_auc": None, "n_pairs": 0, "n_rows": 0, "per_problem": {}}

    probs = np.concatenate([p["pred_prob"] for p in all_preds], axis=0)
    targets = np.concatenate([p["target"] for p in all_preds], axis=0)
    masks = np.concatenate([p["valid_mask"] for p in all_preds], axis=0)
    problem_ids = np.concatenate([p["problem_id"] for p in all_preds], axis=0)

    flat_mask = masks.astype(bool)
    n_pairs = int(flat_mask.sum())
    pooled_auc = None
    if n_pairs > 0:
        val = _shared_auc(targets[flat_mask], probs[flat_mask])
        pooled_auc = None if np.isnan(val) else val

    per_problem = {}
    for pid in sorted(set(int(p) for p in problem_ids)):
        row_sel = problem_ids == pid
        p_mask = masks[row_sel].astype(bool)
        n = int(p_mask.sum())
        if n == 0:
            per_problem[str(pid)] = {"auc": None, "n_pairs": 0}
            continue
        val = _shared_auc(targets[row_sel][p_mask], probs[row_sel][p_mask])
        per_problem[str(pid)] = {"auc": None if np.isnan(val) else val, "n_pairs": n}

    return {"pooled_auc": pooled_auc, "n_pairs": n_pairs, "n_rows": int(masks.shape[0]),
           "per_problem": per_problem}


def value_keyed_rows_from_scores(scores: dict, attempts_batch: pd.DataFrame) -> dict:
    """Extracts one arm's scored rows keyed by value
    (`learner`/`problem_id`/`code_state_id`, read straight off `attempts_batch` at
    `kept_row_idx` -- SAME technique as `cw5.write_predictions_npz`) rather than by the
    positional `kept_row_idx` itself. This is what makes per-BLOCK eval results
    concatenable across blocks: two different blocks' `kept_row_idx` are positions into
    two DIFFERENT `attempts_batch` slices (different val-learner subsets) and are not
    comparable as integers, but every row's OWN identity is globally meaningful and
    trivially concatenates."""
    kept = scores["kept_row_idx"]
    rows = attempts_batch.iloc[kept]
    return {
        "learner": rows["learner"].to_numpy(), "problem_id": rows["problem_id"].to_numpy(),
        "code_state_id": rows["code_state_id"].to_numpy(),
        "true_class_nll": scores["true_class_nll"], "true_class_top1": scores["true_class_top1"],
        "true_class_exact_match": scores["true_class_exact_match"],
    }


def concat_value_keyed_rows(blocks: list[dict]) -> dict:
    if not blocks:
        return {k: np.zeros(0) for k in
               ("learner", "problem_id", "code_state_id", "true_class_nll",
                "true_class_top1", "true_class_exact_match")}
    return {k: np.concatenate([b[k] for b in blocks]) for k in blocks[0]}


def value_keyed_stratum_summary(rows: dict, stratum_mask: np.ndarray | None) -> dict:
    """`stratum_mask`, if given, is aligned to `rows`'s OWN row order (the caller builds
    it from `rows["learner"]`/`rows["problem_id"]` directly, e.g.
    `value_keyed_bottom_tercile_mask` below) -- never a mask built against some OTHER
    row ordering, which is exactly the position-vs-identity confusion this value-keyed
    design exists to avoid."""
    n = len(rows["learner"])
    sel = np.ones(n, dtype=bool) if stratum_mask is None else stratum_mask
    if sel.sum() == 0:
        return {"n": 0}
    return {
        "n": int(sel.sum()),
        "token_nll_mean": float(np.mean(rows["true_class_nll"][sel])),
        "exact_match_rate": float(np.mean(rows["true_class_exact_match"][sel])),
        "top1_rate": float(np.mean(rows["true_class_top1"][sel])),
    }


def value_keyed_bottom_tercile_mask(rows: dict, manifest: dict) -> np.ndarray:
    """Bottom third of VAL learners by mean CLEAN_FULL rate over the FULL manifest
    attempts table (not just this eval's matched rows -- a learner's ability tercile is
    a property of their whole history, matching `infra/cw_fc_grid.py::evaluate_iid`'s
    own convention), then applied to `rows` by joining on the `learner` VALUE."""
    val_attempts = manifest["attempts"][manifest["attempts"]["learner"].isin(manifest["val_learners"])]
    rate = val_attempts.groupby("learner")["outcome_code"].apply(lambda s: float((s == 3).mean()))
    cutoff = rate.quantile(1.0 / 3.0)
    bottom_learners = set(rate[rate <= cutoff].index)
    return pd.Series(rows["learner"]).isin(bottom_learners).to_numpy()


def paired_personalisation_gap_ci(personalised_rows: dict, shuffled_rows: dict,
                                  n_boot: int = 2000, seed: int = 0) -> dict | None:
    """The headline personalisation gap as a paired contrast with a learner-clustered
    CI on the row intersection. `personalised` and
    `shuffled_u` drop different rows (budget/derangement exclusions differ), so the
    contrast is only valid on rows BOTH arms scored, joined by VALUE (`learner`,
    `problem_id`, `code_state_id` -- never position, since the two arms' arrays are not
    row-aligned to begin with). Uses the shared
    `atom.eval.metrics.learner_clustered_bootstrap_ci` (repeated-measures-correct: many
    rows per learner, whose errors are correlated within-learner). Returns `None` if the
    intersection is empty."""
    from atom.eval.metrics import learner_clustered_bootstrap_ci

    def _key(rows, i):
        return (str(rows["learner"][i]), int(rows["problem_id"][i]), str(rows["code_state_id"][i]))

    personalised_index = {_key(personalised_rows, i): i for i in range(len(personalised_rows["learner"]))}
    shuffled_index = {_key(shuffled_rows, i): i for i in range(len(shuffled_rows["learner"]))}
    common_keys = sorted(set(personalised_index) & set(shuffled_index))
    if not common_keys:
        return None

    p_idx = np.array([personalised_index[k] for k in common_keys])
    s_idx = np.array([shuffled_index[k] for k in common_keys])
    a = personalised_rows["true_class_nll"][p_idx]
    b = shuffled_rows["true_class_nll"][s_idx]
    learner_ids = np.array([k[0] for k in common_keys])
    ci = learner_clustered_bootstrap_ci(a, b, learner_ids, n_boot=n_boot, seed=seed)
    ci["n_rows_intersection"] = len(common_keys)
    return ci


def _eval_block_dir(out_dir: pathlib.Path) -> pathlib.Path:
    return out_dir / "eval_blocks"


def _block_scores_path(out_dir: pathlib.Path, block: int, arm: str) -> pathlib.Path:
    return _eval_block_dir(out_dir) / f"block{block:04d}_{arm}.npz"


def _block_tv_path(out_dir: pathlib.Path, block: int) -> pathlib.Path:
    return _eval_block_dir(out_dir) / f"block{block:04d}_tv.npz"


def save_block_scores(scores: dict, out_dir: pathlib.Path, block: int, arm: str,
                      bucket: str | None, out_prefix: str) -> None:
    """Persist one (block, arm)'s scores as soon as it finishes -- `cw5.save_arm_scores`'s
    per-arm granularity made finer (per block too), so a crash mid-eval loses at most
    one block's worth of one arm, not the whole arm across every val learner."""
    path = _block_scores_path(out_dir, block, arm)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {k: scores[k] for k in cw5._ARM_SCORE_ARRAY_KEYS}
    payload.update({k: np.array(scores[k]) for k in cw5._ARM_SCORE_SCALAR_KEYS})
    np.savez(path, **payload)
    if bucket:
        import boto3
        boto3.client("s3").upload_file(str(path), bucket, f"{out_prefix}/eval_blocks/{path.name}")


def try_resume_block_scores(out_dir: pathlib.Path, block: int, arm: str,
                            bucket: str | None, out_prefix: str) -> dict | None:
    path = _block_scores_path(out_dir, block, arm)
    if not path.exists() and bucket:
        import boto3
        s3 = boto3.client("s3")
        key = f"{out_prefix}/eval_blocks/{path.name}"
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            s3.download_file(bucket, key, str(path))
        except Exception:
            return None
    if not path.exists():
        return None
    return cw5.load_arm_scores(path, arm)


_TV_PRED_KEYS = ("learner", "problem_id", "code_state_id", "pred_prob", "target", "valid_mask")


def save_block_tv_preds(tv_preds: dict | None, out_dir: pathlib.Path, block: int,
                        bucket: str | None, out_prefix: str) -> None:
    path = _block_tv_path(out_dir, block)
    path.parent.mkdir(parents=True, exist_ok=True)
    if tv_preds is None:
        np.savez(path, empty=np.array(True))
    else:
        np.savez(path, empty=np.array(False), **{k: tv_preds[k] for k in _TV_PRED_KEYS})
    if bucket:
        import boto3
        boto3.client("s3").upload_file(str(path), bucket, f"{out_prefix}/eval_blocks/{path.name}")


_TV_RESUME_NOT_FOUND = object()


def try_resume_block_tv_preds(out_dir: pathlib.Path, block: int, bucket: str | None, out_prefix: str):
    path = _block_tv_path(out_dir, block)
    if not path.exists() and bucket:
        import boto3
        s3 = boto3.client("s3")
        key = f"{out_prefix}/eval_blocks/{path.name}"
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            s3.download_file(bucket, key, str(path))
        except Exception:
            return _TV_RESUME_NOT_FOUND
    if not path.exists():
        return _TV_RESUME_NOT_FOUND
    # allow_pickle=True: `code_state_id` (and `learner`) arrive from a pandas
    # object-dtype column upstream (`value_keyed_rows_from_scores`'s identity fields),
    # so `np.savez` stores them as numpy object arrays here -- unlike the tiny
    # hand-built `np.array([...])` fixtures in the unit test, whose short uniform-length
    # strings numpy infers as fixed-width unicode and which therefore round-trip under
    # allow_pickle=False, so the unit test alone would not catch this.
    d = np.load(path, allow_pickle=True)
    if bool(d["empty"]):
        return None
    return {k: d[k] for k in _TV_PRED_KEYS}


def run_blocked_eval(m: dict, manifest: dict, val_learners_for_eval: set, args, device: str,
                     tv_labels: dict, out_dir: pathlib.Path, publish_fn, eval_arms: tuple[str, ...]
                     ) -> tuple[dict[str, list[dict]], list[dict]]:
    """Per-learner-block evaluation, so that the whole validation set is never scored
    in a single call. Every (block, arm) result is checkpointed as soon as it finishes
    (`save_block_scores`) and skippable on `--resume` (`try_resume_block_scores`); a
    heartbeat and a self-re-arming shutdown fuse (the same pattern as the
    generation heartbeat in `cw_sim_core`) are refreshed after (block, arm) units, at most
    every 20 s; an `OverrunTrigger` checks the projected total against `--estimated-minutes`
    at the same cadence. Returns `{arm: [value-keyed-row-dict per block]}` and the list
    of per-block test-vector prediction dicts (`None` entries already filtered out)."""
    val_learners_sorted = sorted(val_learners_for_eval)
    blocks = [val_learners_sorted[i:i + args.eval_block_learners]
             for i in range(0, len(val_learners_sorted), args.eval_block_learners)]
    n_blocks = len(blocks)
    n_units_total = max(n_blocks * len(eval_arms), 1)

    overrun = cw5.OverrunTrigger(args.estimated_minutes, out_dir, args.bucket or "", args.out_prefix, "cw-sim-eval")
    t0 = time.time()
    units_done = 0
    last_hb = 0.0

    rows_by_arm: dict[str, list[dict]] = {arm: [] for arm in eval_arms}
    tv_preds_all: list[dict] = []

    for b, block_learners in enumerate(blocks):
        block_set = set(block_learners)
        block_attempts = manifest["attempts"][manifest["attempts"]["learner"].isin(block_set)].reset_index(drop=True)
        block_episodes = manifest["episodes"][manifest["episodes"]["learner"].isin(block_set)]
        if len(block_attempts) == 0:
            continue

        rng = np.random.default_rng(args.seed + 1000 + b)
        n_ep = len(block_episodes)
        eps = torch.randn(n_ep, max(args.eval_k, 1), cw5.CW5_DIMS["latent_dim"],
                          generator=torch.Generator().manual_seed(args.seed + 1000 + b))

        for arm in eval_arms:
            resumed = try_resume_block_scores(out_dir, b, arm, args.bucket, args.out_prefix) if args.resume else None
            if resumed is not None:
                rows_by_arm[arm].append(value_keyed_rows_from_scores(resumed, block_attempts))
                units_done += 1
                print(f"[cw-sim] eval resume: block={b}/{n_blocks} arm={arm} loaded from checkpoint", flush=True)
                continue

            scores = cw5.score_arm_for_batch(
                m, block_attempts, block_episodes, manifest, cw5.CW5_DIMS, device,
                args.eval_k, args.eval_learner_batch_size, arm, rng=rng, eps=eps,
                row_chunk_size=args.eval_row_chunk_size)
            save_block_scores(scores, out_dir, b, arm, args.bucket, args.out_prefix)
            rows_by_arm[arm].append(value_keyed_rows_from_scores(scores, block_attempts))
            units_done += 1

            now = time.time()
            if now - last_hb >= 20:
                elapsed = now - t0
                remaining_units = n_units_total - units_done
                per_unit = elapsed / max(units_done, 1)
                projected_remaining_s = per_unit * remaining_units
                fuse_minutes = max(60, int(math.ceil(2.0 * projected_remaining_s / 60.0)))
                subprocess.run(["sudo", "shutdown", "-c"], check=False,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                subprocess.run(["sudo", "shutdown", "-h", f"+{fuse_minutes}"], check=False,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                hb = {"phase": "eval_blocked", "block": b, "n_blocks": n_blocks, "arm": arm,
                     "units_done": units_done, "units_total": n_units_total,
                     "elapsed_s": elapsed, "fuse_rearmed_minutes": fuse_minutes}
                hb_path = out_dir / "heartbeat.json"
                hb_path.write_text(json.dumps(hb))
                publish_fn(hb_path)
                print(f"[cw-sim] eval block={b}/{n_blocks} arm={arm} done "
                     f"({units_done}/{n_units_total} units, {elapsed:.0f}s elapsed); "
                     f"fuse re-armed +{fuse_minutes}min", flush=True)
                last_hb = now
            if overrun.check(units_done, n_units_total, time.time() - t0):
                raise SystemExit(1)

        tv_resumed = (try_resume_block_tv_preds(out_dir, b, args.bucket, args.out_prefix)
                     if args.resume else _TV_RESUME_NOT_FOUND)
        if tv_resumed is _TV_RESUME_NOT_FOUND:
            out = cw5.forward_outcome(m, block_attempts, block_episodes, manifest, cw5.CW5_DIMS,
                                      device, args.free_bits)
            tv_pred = test_vector_eval_predictions(m, block_attempts, out, manifest, tv_labels, device)
            save_block_tv_preds(tv_pred, out_dir, b, args.bucket, args.out_prefix)
        else:
            tv_pred = tv_resumed
        if tv_pred is not None:
            tv_preds_all.append(tv_pred)

    return rows_by_arm, tv_preds_all


def write_value_keyed_predictions_npz(all_rows_by_arm: dict[str, dict], out_path: pathlib.Path) -> None:
    """Persists per-row NLLs for every arm: one file, every arm's per-row
    identity + score, prefixed per arm exactly like `cw5.write_predictions_npz` (this
    is that function's value-keyed sibling -- the blocked-eval design tracks rows by
    identity rather than a shared positional index, so the arrays it has to persist are
    `value_keyed_rows_from_scores`'s output, not `score_arm_for_batch`'s raw
    `kept_row_idx`-indexed one)."""
    payload = {}
    for arm, rows in all_rows_by_arm.items():
        prefix = f"{arm}__"
        for key, arr in rows.items():
            payload[prefix + key] = arr
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **payload)


def build_args_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    for name in ("raw_dir", "code_map", "diff_map", "codealone_meta", "codealone_states",
                "diff_meta", "diff_states"):
        ap.add_argument("--" + name.replace("_", "-"), required=name != "raw_dir")
    ap.add_argument("--filter-mode", choices=["frozen", "joint"], required=True)
    ap.add_argument("--ratchet-checkpoint", default=str(RATCHET_CHECKPOINT_DEFAULT))
    ap.add_argument("--test-vector-labels", default=str(REPO / "artifacts" / "cw_sim_test_vector_labels.npz"))
    ap.add_argument("--token-loss-weight", type=float, default=1.0)
    ap.add_argument("--outcome-loss-weight", type=float, default=0.3)
    ap.add_argument("--test-vector-loss-weight", type=float, default=0.1)
    ap.add_argument("--kl-max-weight", type=float, default=0.01)
    ap.add_argument("--kl-anneal-epochs", type=float, default=2.0)
    ap.add_argument("--free-bits", type=float, default=0.05)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--max-steps", type=int, required=True)
    ap.add_argument("--learner-batch-size", type=int, default=4)
    ap.add_argument("--micro-batch-size", type=int, default=4)
    ap.add_argument("--train-row-chunk-size", type=int, default=None,
                    help="bounds peak memory in "
                         "production_nll_per_row's cross-entropy independent of "
                         "--micro-batch-size (gemma4-e4b's huge vocab makes full-"
                         "sequence logits the dominant OOM term); omit to disable")
    ap.add_argument("--eval-learner-batch-size", type=int, default=4)
    ap.add_argument("--eval-arms", default=None,
                    help="comma-separated subset of ARMS to score at eval (to keep a pilot cheap, "
                         "e.g. 'personalised,shuffled_u' -- omit for all four)")
    ap.add_argument("--eval-max-learners", type=int, default=None,
                    help="pilot-scale eval: restrict to a random subset of this many val "
                         "learners (seeded, disclosed); omit for the real full-val eval")
    ap.add_argument("--eval-block-learners", type=int, default=20,
                    help="val learners per checkpointed eval block")
    ap.add_argument("--eval-row-chunk-size", type=int, default=None,
                    help="row-chunk size inside "
                         "production_log_likelihood_sum_per_row's cross-entropy (bounds "
                         "peak memory independent of --eval-learner-batch-size)")
    ap.add_argument("--resume", action="store_true",
                    help="skip (block, arm) eval units already checkpointed "
                         "locally or in S3, load their scores instead of re-scoring")
    ap.add_argument("--estimated-minutes", type=float, default=120.0,
                    help="overrun trigger: eval self-terminates if "
                         "its own measured rate projects past 2x this estimate")
    ap.add_argument("--n-boot-personalisation-gap", type=int, default=2000,
                    help="bootstrap resamples for the paired, "
                         "learner-clustered personalisation-gap CI")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--backbone", default="gemma4-e4b")
    ap.add_argument("--eval-k", type=int, default=4)
    ap.add_argument("--n-generation-rows", type=int, default=20)
    ap.add_argument("--max-new-tokens", type=int, default=cw5.MAX_LEN,
                    help="default MAX_LEN (512, chosen from measured completion "
                         "lengths) -- a lower value silently truncates longer "
                         "completions before EOS, which biases exact-match DOWN for no "
                         "reason connected to model quality. A caller who overrides this "
                         "below MAX_LEN must disclose the truncation bias in the report.")
    ap.add_argument("--full-eval", action="store_true",
                    help="run the full mandatory panel (all 4 arms, all 3 strata, CodeBLEU, "
                         "test-vector AUC); off by default so the pilot stays cheap")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--bucket", default=None)
    ap.add_argument("--out-prefix", default="cw-sim")
    ap.add_argument("--fuse-minutes", type=float, default=60.0)
    ap.add_argument("--checkpoint-every-s", type=float, default=300.0)
    ap.add_argument("--self-terminate", action="store_true")
    return ap


## Evaluation summaries are value-keyed (`value_keyed_bottom_tercile_mask`,
## `value_keyed_stratum_summary`, `paired_personalisation_gap_ci`) rather than
## positional into a single `attempts_batch`, because the per-block checkpointed eval
## (`run_blocked_eval`) gives each block its own positional indexing.


def main() -> int:
    ap = build_args_parser()
    args = ap.parse_args()

    import subprocess
    subprocess.run(["sudo", "shutdown", "-c"], check=False)
    subprocess.run(["sudo", "shutdown", "-h", f"+{int(args.fuse_minutes)}"], check=False)
    print(f"[cw-sim] boot fuse armed: shutdown in {args.fuse_minutes} min unless cancelled", flush=True)

    git_sha = cw5._git_sha()
    cw5.seed_everything(args.seed)
    device = cw5.default_device()
    print(f"[cw-sim] git_sha={git_sha} device={device} filter_mode={args.filter_mode} "
         f"token_loss_weight={args.token_loss_weight} outcome_loss_weight={args.outcome_loss_weight} "
         f"test_vector_loss_weight={args.test_vector_loss_weight}", flush=True)

    manifest = cw5.load_cw5_manifest(args)
    learner_split_sha256 = cw5._sha256(REPO / "splits" / "cw_learner_split.csv")
    print(f"[cw-sim] {len(manifest['attempts'])} outcome-valid attempts, "
         f"{len(manifest['train_learners'])} train / {len(manifest['val_learners'])} val learners", flush=True)

    tv_labels = load_test_vector_labels(pathlib.Path(args.test_vector_labels))
    print(f"[cw-sim] test-vector labels: {len(tv_labels['lookup'])} rows, "
         f"{len(tv_labels['headline_problem_ids'])} headline problems", flush=True)

    backbone_bundle = cw5.load_backbone(args.backbone, device=device, allow_download=True)
    m = cw5.build_model(cw5.CW5_DIMS, backbone_bundle, args.seed, device,
                        item_encoder="mlp", sigma_parameterization=SIGMA_PARAMETERIZATION)
    ratchet_path = pathlib.Path(args.ratchet_checkpoint)
    ratchet_sha256 = cw5._sha256(ratchet_path)
    assert ratchet_sha256 == RATCHET_CHECKPOINT_SHA256, (
        f"ratchet checkpoint sha256 mismatch: {ratchet_sha256} != {RATCHET_CHECKPOINT_SHA256} "
        "(the pinned classifier checkpoint) -- refusing to proceed on an unverified checkpoint")
    load_and_freeze_ratchet(m, ratchet_path, learner_split_sha256,
                            freeze=(args.filter_mode == "frozen"), device=device,
                            sigma_parameterization=SIGMA_PARAMETERIZATION)
    print(f"[cw-sim] ratchet checkpoint verified (sha256={ratchet_sha256[:12]}...) and loaded "
         f"({'FROZEN' if args.filter_mode == 'frozen' else 'warm-started, trainable'})", flush=True)

    m["test_vector_head"] = OutcomeHeadItemMLP(
        codec_dim=cw5.CW5_DIMS["codec_dim"], latent_dim=cw5.CW5_DIMS["latent_dim"],
        hidden=cw5.CW5_DIMS["attempt_hidden"], n_classes=MAX_TEST_CASES,
        prefix_dim=cw5.CW5_DIMS["attempt_hidden"]).to(device)

    trainable = (list(m["hyper"].parameters())
                + [p for b in m["bases"].values() for p in b.parameters()]
                + list(m["test_vector_head"].parameters()))
    if args.filter_mode == "joint":
        trainable += (list(m["attempt_encoder"].parameters()) + list(m["filt"].parameters())
                     + list(m["outcome_head"].parameters()))
    optimizer = torch.optim.Adam(trainable, lr=args.lr)
    print(f"[cw-sim] trainable params: {sum(p.numel() for p in trainable):,}", flush=True)

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.jsonl"
    heartbeat_path = out_dir / "heartbeat.json"

    def publish(path):
        if args.bucket:
            import boto3
            boto3.client("s3").upload_file(
                str(path), args.bucket, f"{args.out_prefix.rstrip('/')}/{path.name}",
                ExtraArgs={"ServerSideEncryption": "AES256"})

    def checkpoint_and_upload(step: int, is_final: bool, headline_metrics: dict | None = None,
                              canary9: dict | None = None) -> pathlib.Path:
        # Called periodically, not only at the end of the run, so that nothing valuable
        # exists only on the training machine. The checkpoint carries the full
        # provenance schema: architecture, learner_split_sha256, seed, git_sha,
        # sampler_semantics, held_out_items, codec_path, train_regime and epochs.
        # `headline_metrics` is None for every
        # periodic checkpoint and the pre-eval final save (the numbers do not exist
        # yet); `main()` re-saves final.pt with them once eval completes, so the
        # final.pt on disk always carries the numbers it reported.
        state = {
            "filter_mode": args.filter_mode, "attempt_encoder": m["attempt_encoder"].state_dict(),
            "filt": m["filt"].state_dict(), "outcome_head": m["outcome_head"].state_dict(),
            "test_vector_head": m["test_vector_head"].state_dict(), "hyper": m["hyper"].state_dict(),
            "basis": {n: b.state_dict() for n, b in m["bases"].items()},
            "optimizer": optimizer.state_dict(), "architecture": cw5.CW5_DIMS,
            "token_loss_weight": args.token_loss_weight, "outcome_loss_weight": args.outcome_loss_weight,
            "test_vector_loss_weight": args.test_vector_loss_weight,
            "ratchet_checkpoint_sha256": ratchet_sha256, "learner_split_sha256": learner_split_sha256,
            "git_sha": git_sha, "seed": args.seed, "sampler_semantics": cw5.CURRENT_SAMPLER_SEMANTICS,
            "step": step, "max_test_cases": MAX_TEST_CASES,
            "held_out_items": list(cw5.HELD_OUT_ITEMS),
            "codec_path": "transforms/problem_mean_L15.npz",
            "train_regime": f"cw_sim_full_run.py --filter-mode {args.filter_mode}",
            "epochs": step / max(1, len(manifest["train_learners"]) // args.learner_batch_size),
            "headline_metrics": headline_metrics,
            # The belief-dependence check, run at every checkpoint -- computed by the
            # caller from the step's own training batch (belief_dependence_canary9) so
            # every checkpoint carries its own belief-dependence readout, not just the
            # training loss.
            "canary9_belief_dependence": canary9,
        }
        name = "final.pt" if is_final else f"step{step:08d}.pt"
        path = out_dir / name
        tmp = path.with_suffix(".pt.tmp")
        torch.save(state, tmp)
        tmp.replace(path)
        publish(path)
        if is_final:
            final_copy = out_dir / "final.pt"
            if path != final_copy:
                torch.save(state, final_copy)
                publish(final_copy)
        return path

    train_learners = sorted(manifest["train_learners"])
    sampler = cw5.ShuffledEpochSampler(len(train_learners), np.random.default_rng(args.seed))
    t0 = time.time()
    last_hb = 0.0
    last_checkpoint_t = t0
    n_tv_matched_total = 0

    for step in range(args.max_steps):
        elapsed_min = (time.time() - t0) / 60.0
        if elapsed_min >= args.fuse_minutes:
            print(f"[cw-sim] FUSE: {elapsed_min:.1f} min >= --fuse-minutes, stopping", flush=True)
            break

        batch_idx = sampler.next_batch(args.learner_batch_size)
        batch_learners = {train_learners[i] for i in batch_idx}
        attempts_batch = manifest["attempts"][manifest["attempts"]["learner"].isin(batch_learners)]
        episodes_batch = manifest["episodes"][manifest["episodes"]["learner"].isin(batch_learners)]
        if len(attempts_batch) == 0:
            continue

        epoch = step / max(1, len(train_learners) // args.learner_batch_size)
        optimizer.zero_grad()
        out = cw5.forward_outcome(m, attempts_batch, episodes_batch, manifest, cw5.CW5_DIMS, device, args.free_bits)

        l_outcome_val, kl_val = float(out["l_outcome"].item()), float(out["kl"].item())
        if args.filter_mode == "joint":
            kl_weight = cw5.kl_anneal_weight(epoch, args.kl_max_weight, args.kl_anneal_epochs)
            outcome_total = args.outcome_loss_weight * out["l_outcome"] + kl_weight * out["kl"]
            if not torch.isfinite(outcome_total):
                raise RuntimeError(f"[cw-sim] TRIPWIRE non-finite outcome loss at step={step}")
            outcome_total.backward(retain_graph=True)

        tv_loss, tv_stats = test_vector_batch_loss(m, attempts_batch, out, manifest, tv_labels, device)
        tv_loss_val = 0.0
        if tv_loss is not None:
            n_tv_matched_total += tv_stats["n_matched"]
            (args.test_vector_loss_weight * tv_loss).backward(retain_graph=True)
            tv_loss_val = float(tv_loss.item())

        ep_learner_idx, ep_t_idx = cw5._distinct_episode_positions(
            episodes_batch, out["ep_batch"], out["attempt_batch"]["episode_ids"], device=device)
        l_prod = cw5.production_backward_microbatched(
            m, attempts_batch, manifest, out["ep_batch"], ep_learner_idx, ep_t_idx,
            out["attempt_batch"]["attempt_episode_row_idx"], device, k=1,
            micro_batch_size=args.micro_batch_size, scale=args.token_loss_weight,
            row_chunk_size=args.train_row_chunk_size)
        if not np.isfinite(l_prod):
            raise RuntimeError(f"[cw-sim] TRIPWIRE non-finite production loss at step={step}: {l_prod}")

        torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
        if step == 0 and args.filter_mode == "joint":
            # Belief-junction gradient guard: once, on the first joint step, before
            # optimizer.step() changes any gradient state.
            guard = assert_belief_junction_gradient(m)
            print(f"[cw-sim] LAMBDA_PROBE_GUARD passed: {guard}", flush=True)
        optimizer.step()

        record = {"step": step, "elapsed_s": time.time() - t0, "l_outcome": l_outcome_val,
                 "kl": kl_val, "l_prod": l_prod, "l_test_vector": tv_loss_val,
                 "n_tv_matched": tv_stats["n_matched"], "n_rows": len(attempts_batch),
                 "filter_mode": args.filter_mode}
        with open(metrics_path, "a") as f:
            f.write(json.dumps(record) + "\n")

        now = time.time()
        if now - last_hb >= 20:
            hb = {"step": step, "max_steps": args.max_steps, "phase": "train",
                 "elapsed_s": now - t0, "n_tv_matched_total": n_tv_matched_total, "t": now}
            heartbeat_path.write_text(json.dumps(hb))
            publish(heartbeat_path)
            publish(metrics_path)
            last_hb = now
            print(f"[cw-sim] step={step}/{args.max_steps} l_prod={l_prod:.4f} "
                 f"l_outcome={l_outcome_val:.4f} l_tv={tv_loss_val:.4f} "
                 f"n_tv_matched={tv_stats['n_matched']}", flush=True)

        if (time.time() - last_checkpoint_t) >= args.checkpoint_every_s:
            with torch.no_grad():
                mu_ck, log_sigma_ck = cw5.gather_belief_for_batch(
                    m["filt_forward"], out["ep_batch"], ep_learner_idx, ep_t_idx)
            canary9 = belief_dependence_canary9(
                m, mu_ck, log_sigma_ck, ep_learner_idx.detach().cpu().numpy(), device, args.seed)
            checkpoint_and_upload(step, is_final=False, canary9=canary9)
            last_checkpoint_t = time.time()
            where = "saved+uploaded" if args.bucket else "saved LOCALLY ONLY (--bucket not set)"
            print(f"[cw-sim] periodic checkpoint {where} at step={step}; "
                 f"canary9 pass={canary9['pass']} "
                 f"across_learner/eps={canary9['across_learner_spread']:.4g}/"
                 f"{canary9['across_eps_spread']:.4g}", flush=True)

    completed_steps = step + 1 if "step" in dir() else 0
    elapsed = time.time() - t0
    print(f"[cw-sim] training done: {completed_steps} steps in {elapsed:.1f}s "
         f"({elapsed / max(1, completed_steps):.4f} s/step), "
         f"n_tv_matched_total={n_tv_matched_total}", flush=True)

    final_canary9 = None
    if "out" in dir() and "ep_learner_idx" in dir():
        with torch.no_grad():
            mu_final, log_sigma_final = cw5.gather_belief_for_batch(
                m["filt_forward"], out["ep_batch"], ep_learner_idx, ep_t_idx)
        final_canary9 = belief_dependence_canary9(
            m, mu_final, log_sigma_final, ep_learner_idx.detach().cpu().numpy(), device, args.seed)
        print(f"[cw-sim] final-checkpoint canary9: pass={final_canary9['pass']} "
             f"across_learner/eps={final_canary9['across_learner_spread']:.4g}/"
             f"{final_canary9['across_eps_spread']:.4g} "
             f"mu_between_learner_share={final_canary9['mu_between_learner_variance_share']:.4g}",
             flush=True)
    ckpt_path = checkpoint_and_upload(max(completed_steps - 1, 0), is_final=True, canary9=final_canary9)

    for v in (m["attempt_encoder"], m["filt"], m["outcome_head"], m["test_vector_head"], m["hyper"]):
        v.eval()

    val_learners_for_eval = manifest["val_learners"]
    if args.eval_max_learners is not None and args.eval_max_learners < len(val_learners_for_eval):
        rng_eval = np.random.default_rng(args.seed + 777)
        val_learners_for_eval = set(rng_eval.choice(
            sorted(val_learners_for_eval), size=args.eval_max_learners, replace=False))
        print(f"[cw-sim] eval restricted to a random {args.eval_max_learners}-learner subset "
             f"(of {len(manifest['val_learners'])} val learners), seed={args.seed + 777}", flush=True)
    print(f"[cw-sim] eval: {len(val_learners_for_eval)} val learners, "
         f"{args.eval_block_learners} learners/block, resume={args.resume}", flush=True)

    heartbeat_path.write_text(json.dumps({"phase": "eval_blocked", "t": time.time()}))
    publish(heartbeat_path)
    eval_arms = tuple(args.eval_arms.split(",")) if args.eval_arms else cw5.ARMS
    assert "personalised" in eval_arms, (
        f"--eval-arms={args.eval_arms!r} excludes 'personalised' -- "
        "the personalisation gap, the paired CI, and the generation pass "
        "all require the personalised arm as their reference point; every eval launch "
        "must include it"
    )
    rows_by_arm_blocks, tv_preds_all = run_blocked_eval(
        m, manifest, val_learners_for_eval, args, device, tv_labels, out_dir, publish, eval_arms)
    all_rows_by_arm = {arm: concat_value_keyed_rows(blocks) for arm, blocks in rows_by_arm_blocks.items()}

    arm_summaries = {arm: value_keyed_stratum_summary(rows, None) for arm, rows in all_rows_by_arm.items()}
    gap = None
    gap_ci = None
    if "personalised" in all_rows_by_arm and "shuffled_u" in all_rows_by_arm:
        p_summary, s_summary = arm_summaries["personalised"], arm_summaries["shuffled_u"]
        if p_summary.get("n", 0) > 0 and s_summary.get("n", 0) > 0:
            gap = p_summary["token_nll_mean"] - s_summary["token_nll_mean"]
        gap_ci = paired_personalisation_gap_ci(
            all_rows_by_arm["personalised"], all_rows_by_arm["shuffled_u"],
            n_boot=args.n_boot_personalisation_gap, seed=args.seed)
    overall_summary = {"arms": arm_summaries, "personalisation_gap_token_nll": gap,
                      "personalisation_gap_ci": gap_ci}
    predictions_path = out_dir / "predictions.npz"
    write_value_keyed_predictions_npz(all_rows_by_arm, predictions_path)
    publish(predictions_path)
    result = {"filter_mode": args.filter_mode, "token_loss_weight": args.token_loss_weight,
             "outcome_loss_weight": args.outcome_loss_weight, "overall": overall_summary}
    print(f"[cw-sim] overall personalisation gap (token NLL, personalised-shuffled_u): "
         f"{gap} (paired CI: {gap_ci})", flush=True)

    if args.full_eval:
        heartbeat_path.write_text(json.dumps({"phase": "eval_strata", "t": time.time()}))
        publish(heartbeat_path)
        result["bottom_tercile"] = {
            arm: value_keyed_stratum_summary(rows, value_keyed_bottom_tercile_mask(rows, manifest))
            for arm, rows in all_rows_by_arm.items()
        }
        headline_ids = list(tv_labels["headline_problem_ids"])
        result["tiktoc_headline_stratum"] = {
            arm: value_keyed_stratum_summary(
                rows, np.isin(rows["problem_id"].astype(int), headline_ids)
                if len(rows["problem_id"]) else np.zeros(0, dtype=bool))
            for arm, rows in all_rows_by_arm.items()
        }
        result["test_vector_auc"] = test_vector_auc_panel(tv_preds_all)
        print(f"[cw-sim] test-vector AUC: pooled={result['test_vector_auc']['pooled_auc']} "
             f"n_pairs={result['test_vector_auc']['n_pairs']}", flush=True)

        heartbeat_path.write_text(json.dumps({"phase": "generation", "t": time.time()}))
        publish(heartbeat_path)
        # The eval-block loop re-arms the OS shutdown fuse per (block, arm) unit
        # (`run_blocked_eval`'s own heartbeat), but that loop is entirely SKIPPED under
        # `--resume` when every unit is loaded from checkpoint -- so the fuse is left at
        # whatever it was last set to (the initial boot value on a full `--resume` hit)
        # for the entire generation pass below, which has no heartbeat of its own once
        # started, so the boot fuse can fire mid-generation on a fully resumed run.
        # Re-arm unconditionally here, generous enough for generation + CodeBLEU/codec
        # similarity on `--n-generation-rows` rows (this phase has no per-row checkpoint
        # to resume from on a re-launch, so the buffer must cover it in one shot).
        fuse_minutes = max(60, int(args.fuse_minutes))
        subprocess.run(["sudo", "shutdown", "-c"], check=False)
        subprocess.run(["sudo", "shutdown", "-h", f"+{fuse_minutes}"], check=False)
        print(f"[cw-sim] fuse re-armed +{fuse_minutes}min before generation pass", flush=True)
        # Generation is a SMALL diagnostic subsample (`--n-generation-rows`, default 20),
        # architecturally separate from the main blocked NLL panel above -- it only ever
        # reads the "personalised" arm's own mu/log_sigma/p_class_meanK
        # (`run_ancestral_generation_pass`'s own contract), so one small, un-blocked
        # `score_arm_for_batch` call on a random learner subsample is cheap and correct;
        # it does not need the full-val blocked machinery above.
        gen_rng = np.random.default_rng(args.seed + 999)
        n_gen_learners = min(args.n_generation_rows, len(val_learners_for_eval))
        gen_learners = set(gen_rng.choice(sorted(val_learners_for_eval), size=n_gen_learners, replace=False))
        gen_attempts = manifest["attempts"][manifest["attempts"]["learner"].isin(gen_learners)].reset_index(drop=True)
        gen_episodes = manifest["episodes"][manifest["episodes"]["learner"].isin(gen_learners)]
        gen_scores = {"personalised": cw5.score_arm_for_batch(
            m, gen_attempts, gen_episodes, manifest, cw5.CW5_DIMS, device, args.eval_k,
            args.eval_learner_batch_size, "personalised", row_chunk_size=args.eval_row_chunk_size)}
        gen_records = cw5.run_ancestral_generation_pass(
            m, gen_attempts, manifest, gen_scores, device,
            args.n_generation_rows, args.max_new_tokens, args.seed)
        gen_local_path = out_dir / "generations.jsonl"
        with open(gen_local_path, "w") as f:
            for r in gen_records:
                f.write(json.dumps(r) + "\n")
        print(f"[cw-sim] generations.jsonl remains at {gen_local_path} on this machine only "
             "-- copy it off before shutdown; it is never uploaded", flush=True)

        transform = cw5.CodecTransform.load(REPO / "transforms" / "problem_mean_L15.npz")
        numeric_rows = []
        codebleu_rows = []
        real_cache: dict = {}
        for rec in gen_records:
            sim = cw5.generation_codec_similarity_one(m, transform, device, rec, real_cache)
            numeric_rows.append(numeric_generation_row(rec, sim))
            if codebleu_available() and rec["generated_code"].strip():
                try:
                    cb = codebleu_one(rec["real_completion"], rec["generated_code"])
                    codebleu_rows.append({"row_idx": rec["row_idx"], "arm": rec["arm"],
                                          "generation_mode": rec["generation_mode"], **cb})
                except Exception as e:  # pragma: no cover -- disclosed, never silently dropped
                    print(f"[cw-sim] codebleu failed for row {rec['row_idx']}: {e}", flush=True)
        numeric_path = out_dir / "generation_numeric_rows.npz"
        save_numeric_generation_rows(numeric_rows, numeric_path)
        publish(numeric_path)
        gen_summary = summarise_generation_rows(numeric_rows)
        if codebleu_rows:
            gen_summary["codebleu"] = {
                "n_scored": len(codebleu_rows),
                "mean_by_arm": {
                    arm: float(np.mean([r["codebleu"] for r in codebleu_rows if r["arm"] == arm]))
                    for arm in cw5.ARMS if any(r["arm"] == arm for r in codebleu_rows)
                },
                "component_means": {
                    comp: float(np.mean([r[comp] for r in codebleu_rows]))
                    for comp in ("ngram_match", "weighted_ngram_match", "syntax_match", "dataflow_match")
                } if codebleu_rows else None,
            }
        else:
            gen_summary["codebleu"] = "unavailable_on_this_platform" if not codebleu_available() else "no_rows_scored"
        result["generation_summary"] = gen_summary

    # final.pt must carry the numbers it reported, not just the weights that produced
    # them -- re-save after eval so anyone loading final.pt sees its headline metrics,
    # not just architecture/config.
    checkpoint_and_upload(max(completed_steps - 1, 0), is_final=True, headline_metrics=overall_summary)

    result_path = out_dir / "result.json"
    result_path.write_text(json.dumps(result, indent=2, allow_nan=True))
    publish(result_path)
    heartbeat_path.write_text(json.dumps({"phase": "complete", "t": time.time()}))
    publish(heartbeat_path)
    print(json.dumps(result, indent=2, allow_nan=True))

    if args.self_terminate:
        import subprocess as sp
        sp.run(["sudo", "shutdown", "-h", "now"], check=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
