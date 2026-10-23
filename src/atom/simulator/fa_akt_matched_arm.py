"""The AKT-matched evaluation arm for the FA simulator.

Rule: every model's row states its evaluation-population count, and any simulator-vs-AKT
gap is quoted with both arms, the native one and one matched to AKT's history
(score-missing events excluded from history).

AKT v2 (`scripts/fa_01_akt_v2.py`) excludes 6,980 score-missing
events from its event stream ENTIRELY (pyKT's AKT couples a step's response into the joint
INPUT embedding, not only the loss target, so a maskable placeholder isn't well-defined for
it) -- while the simulator keeps them as recurrent input and masks only the
loss (`atom.filter.batch.build_padded_batch_v2`'s `label_valid` channel). That difference
in CONDITIONING, not just in which rows get scored, is why a plain "simulator val AUC vs
AKT val AUC" comparison overstates AKT's edge by an unmeasured amount -- fixing it needs a
second simulator evaluation run under AKT-shaped history, not a re-subset of one evaluation
after the fact (`atom.data.fa_manifest.exclude_score_missing_from_history`).

Two arms, same scored rows, same targets, same text -- only the BELIEF each row is scored
under differs:
  - native: `build_padded_batch_v2` on the untouched val split (score-missing rows kept as
    real sequence positions, masked out of the loss only -- the simulator's own design).
  - matched: `build_padded_batch_v2` on `exclude_score_missing_from_history`'s output
    (score-missing rows dropped entirely, `t` resequenced -- structurally identical to
    what AKT's own stream already does).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from atom.binding import reparameterize_from_log_sigma
from atom.data.fa_manifest import exclude_score_missing_from_history
from atom.eval.metrics import auc, brier, ece, with_population
from atom.filter.batch import build_padded_batch_v2
from atom.filter.episode_v2 import build_prev_feat_v2
from atom.simulator.fa_alt2 import collate, forward_personalised
from atom.simulator.fa_train_step import gather_belief_for_batch

# AKT v2's already-measured numbers (from `scripts/fa_01_akt_v2.py`), carried here as named
# constants (not re-derived) so a gate report can cite them alongside a fresh matched-arm
# run without re-running AKT itself.
AKT_V2_MARGINAL_VAL_AUC = 0.8016
AKT_V2_SUPPORT_NONE_STRATUM_AUC = 0.6982
AKT_V2_N_EXCLUDED_SCORE_MISSING = 6980


def build_matched_arm_filter_batch(ev_split: pd.DataFrame, bundle_learners: list,
                                   codec_matrix: np.ndarray, pid_to_idx: dict) -> tuple[dict, pd.DataFrame]:
    """The matched arm's own filter_batch: same learners as the native val filter_batch,
    built from `exclude_score_missing_from_history`'s output instead of the raw split.
    Returns (filter_batch, matched_ev) -- `matched_ev`'s own `id`/`t` columns are what
    `remap_t_idx_to_matched_arm` uses to translate each scored row from the native
    sequence's `t` to this resequenced one.
    """
    sub = ev_split[ev_split["learner"].isin(bundle_learners)].sort_values(
        ["learner", "t"], kind="mergesort")
    matched_ev = exclude_score_missing_from_history(sub)
    filter_batch = build_padded_batch_v2(matched_ev, codec_matrix, pid_to_idx, device="cpu",
                                         include_time_min=True)
    filter_batch["prev_feat"] = build_prev_feat_v2(
        filter_batch["correct"], filter_batch["hint_bin"], filter_batch["saw_answer"],
        filter_batch["score_missing"])
    return filter_batch, matched_ev


def remap_t_idx_to_matched_arm(examples_df: pd.DataFrame,
                               matched_ev: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """For each row of `examples_df` (must carry an `id` column), find its counterpart in
    `matched_ev` by `id` and return (row_has_match, new_t). A score-missing row never has a
    counterpart (by construction of `exclude_score_missing_from_history`) -- `row_has_match`
    flags this explicitly rather than letting a silent NaN/garbage `t` reach the model.
    Every non-score-missing row DOES have a counterpart, since `exclude_score_missing_from_history`
    keeps every row it doesn't drop.
    """
    lookup = matched_ev.set_index("id")["t"]
    mapped = examples_df["id"].map(lookup)
    row_has_match = ~mapped.isna()
    new_t = mapped.fillna(0).to_numpy().astype("int64")
    return row_has_match.to_numpy(), new_t


@torch.no_grad()
def success_eval(filt_forward, hyper, model, context, success_head,
                 filter_batch: dict, learner_idx: torch.Tensor, t_idx: torch.Tensor,
                 examples: list, targets: dict, pad_id: int, device: str,
                 batch_size: int = 8) -> dict:
    """Point-prediction (no K-sampling) success log-loss + ECE + AUC + Brier for a GIVEN
    filter_batch/learner_idx/t_idx/examples/targets combination -- the shared scoring code
    both the native and matched arms call, so they differ only in which belief feeds it,
    never in how the metric itself is computed. Log-loss/ECE use the same eps=1e-7-clamp,
    gather-selected-logit-at-the-observed-support-class math as
    `infra/fa_06_stage2_train.py::val_evaluate`'s own success block, kept identical on
    purpose so a native-arm number computed here is directly comparable to one read out of
    a real val_evaluate run. AUC/Brier are added ON TOP (not in val_evaluate) specifically
    so this gate has a metric that is actually SHARED with AKT v2's own reported row
    ("AUC 0.8016 - acc 0.744 - ECE 0.011 - Brier 0.173") --
    without them, "matched arm vs AKT" would compare a log-loss number against AKT's AUC,
    which is not the same statement even after the history-conditioning is matched.
    """
    n = len(examples)
    probs_kept, targets_kept = [], []
    for start in range(0, n, batch_size):
        idx = list(range(start, min(start + batch_size, n)))
        valid = targets["success_valid"][idx] & targets["support_valid"][idx]
        if not bool(valid.any()):
            continue
        tok_batch = collate([examples[i] for i in idx], pad_id=pad_id, device=device)
        l_idx, t_i = learner_idx[idx], t_idx[idx]
        mu_sel, log_sigma_sel = gather_belief_for_batch(filt_forward, filter_batch, l_idx, t_i)
        u = reparameterize_from_log_sigma(mu_sel, log_sigma_sel, k=1)[:, 0, :]
        out = forward_personalised(model, hyper, context, u, tok_batch["input_ids"],
                                   tok_batch["attention_mask"], tok_batch["prompt_end_index"])
        logits = success_head(out["h_q"])
        support_t = targets["support_target"][idx]
        selected = logits.gather(1, support_t.view(-1, 1)).squeeze(1)
        probs = torch.sigmoid(selected)
        probs_kept.append(probs[valid].cpu())
        targets_kept.append(targets["success_target"][idx][valid].cpu())

    if not probs_kept:
        return {"success_logloss": with_population({"value": float("nan")}, 0),
               "success_ece": with_population({"value": float("nan")}, 0),
               "success_auc": with_population({"value": float("nan")}, 0),
               "success_brier": with_population({"value": float("nan")}, 0)}
    probs_cat = torch.cat(probs_kept)
    target_cat = torch.cat(targets_kept)
    eps = 1e-7
    p = probs_cat.clamp(eps, 1 - eps)
    logloss = float(-(target_cat * torch.log(p) + (1 - target_cat) * torch.log(1 - p)).mean().item())
    ece_val = float(ece(target_cat.numpy(), probs_cat.numpy()))
    auc_val = float(auc(target_cat.numpy(), probs_cat.numpy()))
    brier_val = float(brier(target_cat.numpy(), probs_cat.numpy()))
    n_eval = len(probs_cat)
    return {"success_logloss": with_population({"value": logloss}, n_eval),
           "success_ece": with_population({"value": ece_val}, n_eval),
           "success_auc": with_population({"value": auc_val}, n_eval),
           "success_brier": with_population({"value": brier_val}, n_eval)}


def matched_arm_success_eval(filt_forward, hyper, model, context, success_head,
                             native_filter_batch: dict, ev_split: pd.DataFrame,
                             examples: list, examples_df: pd.DataFrame, targets: dict,
                             codec_matrix: np.ndarray, pid_to_idx: dict,
                             pad_id: int, device: str, batch_size: int = 8) -> dict:
    """The matched-arm comparison, assembled: native-arm and matched-arm success log-loss/ECE on the
    SAME scored rows (`examples`/`targets`, built once by the caller exactly as
    `infra/fa_06_stage2_train.py`'s own val pipeline does), plus AKT v2's already-measured
    numbers for the report to cite alongside both. `examples_df` must carry `learner`/`t`/
    `id` columns aligned 1:1 with `examples` (same convention `val_examples_df` already
    uses in `infra/fa_06_stage2_train.py`).
    """
    native_learner_to_idx = {u: i for i, u in enumerate(native_filter_batch["learners"])}
    native_learner_idx = torch.tensor(
        [native_learner_to_idx[u] for u in examples_df["learner"]], device=device)
    native_t_idx = torch.tensor(examples_df["t"].astype("int64").to_numpy(), device=device)
    native_metrics = success_eval(filt_forward, hyper, model, context, success_head,
                                  native_filter_batch, native_learner_idx, native_t_idx,
                                  examples, targets, pad_id, device, batch_size)

    bundle_learners = sorted(examples_df["learner"].unique())
    matched_filter_batch, matched_ev = build_matched_arm_filter_batch(
        ev_split, bundle_learners, codec_matrix, pid_to_idx)
    row_has_match, new_t = remap_t_idx_to_matched_arm(examples_df, matched_ev)
    matched_learner_to_idx = {u: i for i, u in enumerate(matched_filter_batch["learners"])}
    # a row with no matched-arm counterpart (score-missing -> excluded, never scored
    # anyway since success_valid already excludes it) gets a dummy but IN-RANGE index so
    # the batch shape stays aligned; success_eval's own success_valid mask (unaffected by
    # this remap) is what actually excludes it from the metric.
    matched_learner_idx = torch.tensor(
        [matched_learner_to_idx.get(u, 0) for u in examples_df["learner"]], device=device)
    matched_t_idx = torch.tensor(
        np.where(row_has_match, new_t, 0), dtype=torch.int64, device=device)
    matched_metrics = success_eval(filt_forward, hyper, model, context, success_head,
                                   matched_filter_batch, matched_learner_idx, matched_t_idx,
                                   examples, targets, pad_id, device, batch_size)

    return {
        "native_arm": native_metrics, "matched_arm": matched_metrics,
        "akt_v2_marginal_val_auc": AKT_V2_MARGINAL_VAL_AUC,
        "akt_v2_support_none_stratum_auc": AKT_V2_SUPPORT_NONE_STRATUM_AUC,
        "akt_v2_n_excluded_score_missing": AKT_V2_N_EXCLUDED_SCORE_MISSING,
    }
