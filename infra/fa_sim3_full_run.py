#!/usr/bin/env python
"""Trains FA-SIM3, the mathematics simulator, and evaluates it on validation learners.

FA-SIM3 is one personalised language model: a history-only belief filter (warm-started
from the outcome-only filter `fa_outcome_filter.pt`) feeds a hypernetwork that mixes a
shared LoRA basis on a frozen backbone; an outcome head predicts the 5-class support
outcome and the backbone produces the learner's answer text. Training is one true
shuffled epoch over the training bundle with loss
`l_outcome + alpha * l_a + beta_kl * kl` (alpha = `--token-loss-weight`).

Evaluation compares four arms (personalised, dw0 = no personalisation, shuffled_u =
another learner's belief from the same history bin, u_mean = belief mean only) with a
joint (k, c) K-sample mixture NLL, common random numbers across arms and
learner-clustered paired CIs. It also scores paired belief swaps, closed-set
multiple-choice options, within-item AUC, the history-curve / per-type / LOO-oracle
panel, and two generation modes (conditional imitation on the true class; ancestral,
class sampled from p(c|u)). Results are reported on three populations: a bridge
population from an earlier smaller simulator, the naturally sampled t>=10 stratum
(primary) and the oversampled cold-start t<10 stratum. Summary keys
requirement_8/10/12/14 hold, respectively, realised training exposure, rows dropped by
the derangement, rows dropped by the completion-budget check, and these three courts.

Inputs: staged train/validation bundles (scripts/fa_sim3_stage_bundles.py), the FA
manifest and splits, the frozen codec. Outputs: checkpoints, predictions.csv.gz,
belief_diagnostics.npz and summary.json uploaded to the S3 bucket given with `--bucket`;
generations.jsonl (raw text) stays on the local disk and is never uploaded. Run:
`python infra/fa_sim3_full_run.py --bucket <bucket>`. Serves Tables 10 and 12 and
Figures 7 and 8.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import subprocess
import tarfile
import time

import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[1]

from atom.binding import (Hypernetwork, attach_lora_basis, mean_prediction,  # noqa: E402
                          person_epistemic_spread, reparameterize_from_log_sigma)
from atom.binding.derangement import (assert_learner_matched_derangement,  # noqa: E402
                                      learner_history_matched_derangement)
from atom.codec.backbone import load_backbone  # noqa: E402
from atom.data.fa_manifest import load_v2_events  # noqa: E402
from atom.data.fa_outcome_label import N_OUTCOME, OUTCOME_CLASSES, outcome_label  # noqa: E402
from atom.eval.belief_swap import edit_distance, find_discriminating_pairs  # noqa: E402
from atom.eval.evals_5g import (DEFAULT_HISTORY_BINS, belief_swap_directed_accuracy,  # noqa: E402
                                history_curve, loo_oracle_ceiling, per_type_split)
from atom.eval.metrics import learner_clustered_bootstrap_ci  # noqa: E402
from atom.eval.spread_error import spread_error_correlation  # noqa: E402
from atom.eval.within_item import within_item_auc_paired  # noqa: E402
from atom.filter.batch import build_padded_batch_v2, load_production_codec  # noqa: E402
from atom.filter.episode_v2 import build_prev_feat_v2  # noqa: E402
from atom.filter.item_encoder_mlp import S1FilterV2ItemMLP  # noqa: E402
from atom.filter.losses import gaussian_kl_free_bits, kl_anneal_weight  # noqa: E402
from atom.heads.fa_hierarchical_heads import OutcomeHead  # noqa: E402
from atom.simulator.fa_alt2 import (build_reserved_prompt_ids, collate, encode_example,  # noqa: E402
                                    forward_personalised, joint_kc_mixture_nll,
                                    production_log_likelihood_sum_per_row,
                                    production_nll_per_row, production_topk_per_row)
from atom.simulator.fa_train_step import (chunked_filter_forward, gather_belief_for_batch,  # noqa: E402
                                          gather_item_codec_for_batch,
                                          losses_for_hierarchical_batch)
from atom.simulator.sampler_semantics import CURRENT_SAMPLER_SEMANTICS  # noqa: E402
from atom.utils.epoch_sampler import ShuffledEpochSampler  # noqa: E402
from atom.utils.logging import default_device  # noqa: E402
from atom.utils.seeds import seed_everything  # noqa: E402

B4_CONFIG = {"codec_dim": 64, "episode_hidden": 256, "filter_hidden": 256, "latent_dim": 256}
CODEC_PATH = REPO / "artifacts" / "fa_item_codec_dim64.npz"
EXPECTED_CODEC_SHA = "3dc74d29072c967416351c305bae8d02952049e69052507845f98e0b444f7fe0"
LOG_SIGMA_MIN, LOG_SIGMA_MAX = -5.0, 2.0
ARMS = ("personalised", "dw0", "shuffled_u", "u_mean")


def _codec_sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def warm_start_filter(filt: torch.nn.Module, checkpoint_path: pathlib.Path) -> dict:
    if not checkpoint_path.exists():
        return {"warm_started": False, "reason": f"no checkpoint at {checkpoint_path}"}
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    arch = ckpt.get("architecture", {})
    expected = {"codec_dim": B4_CONFIG["codec_dim"], "episode_hidden": B4_CONFIG["episode_hidden"],
               "filter_hidden": B4_CONFIG["filter_hidden"], "latent_dim": B4_CONFIG["latent_dim"]}
    mismatches = {k: (arch.get(k), v) for k, v in expected.items() if arch.get(k) != v}
    if mismatches:
        return {"warm_started": False, "reason": f"architecture mismatch: {mismatches}"}
    try:
        filt.load_state_dict(ckpt["model_state_dict"], strict=True)
    except Exception as e:  # noqa: BLE001
        return {"warm_started": False, "reason": f"load_state_dict failed: {e}"}
    return {"warm_started": True, "source": str(checkpoint_path),
           "source_git_sha": ckpt.get("git_sha"), "source_epochs": ckpt.get("epochs")}


def load_bundle_and_manifest(bundle_path, manifest_path, learner_split_path, item_split_path,
                             codec_path, split: str):
    extract_dir = pathlib.Path(f"/tmp/fa_sim3_bundle_{split}")
    extract_dir.mkdir(exist_ok=True)
    with tarfile.open(bundle_path, "r:gz") as tf:
        tf.extractall(extract_dir, filter="data")
    bundle = pd.read_json(extract_dir / "events.jsonl", lines=True)

    ev = load_v2_events(manifest_path, learner_split_path, item_split_path)
    ev_split = ev[ev["split"] == split]
    codec_matrix, pid_to_idx = load_production_codec(codec_path, B4_CONFIG["codec_dim"])

    learners = sorted(bundle["learner"].unique())
    sub = ev_split[ev_split["learner"].isin(learners)].sort_values(["learner", "t"], kind="mergesort")
    filter_batch = build_padded_batch_v2(sub, codec_matrix, pid_to_idx, device="cpu",
                                         include_time_min=True)
    filter_batch["prev_feat"] = build_prev_feat_v2(
        filter_batch["correct"], filter_batch["hint_bin"], filter_batch["saw_answer"],
        filter_batch["score_missing"])
    return bundle, filter_batch


def build_examples(bundle: pd.DataFrame, tok, max_len: int):
    labels = outcome_label(bundle["support_code"], bundle["score"], bundle["score_missing"])
    valid = labels["outcome_valid"].to_numpy()
    bundle = bundle.iloc[np.flatnonzero(valid)].reset_index(drop=True)
    outcome_codes = labels["outcome_code"].to_numpy()[valid]

    rows, examples = [], []
    for i, row in bundle.iterrows():
        ex = encode_example(row["question_text"], row["answer_text"], tok, max_len,
                            outcome_code=int(outcome_codes[i]))
        if not ex["production_valid"]:
            continue
        rows.append({"learner": row["learner"], "t": row["t"], "outcome_code": int(outcome_codes[i]),
                    "problem_id": row["problem_id"], "support_code": row["support_code"],
                    "score": row["score"], "score_missing": row["score_missing"]})
        examples.append(ex)
    return pd.DataFrame(rows), examples


def log_sigma_report(log_sigma: torch.Tensor, valid_mask: torch.Tensor) -> dict:
    v = log_sigma[valid_mask.bool()].detach().cpu().numpy().reshape(-1)
    return {
        "mean": float(v.mean()), "std": float(v.std()),
        "p05": float(np.percentile(v, 5)), "p50": float(np.percentile(v, 50)),
        "p95": float(np.percentile(v, 95)),
        "frac_at_clamp_min": float((v <= LOG_SIGMA_MIN + 1e-4).mean()),
        "frac_at_clamp_max": float((v >= LOG_SIGMA_MAX - 1e-4).mean()),
    }


def greedy_generate(model, tok, context, c_vec: torch.Tensor, prompt_ids: list[int],
                    max_new_tokens: int, device: str) -> str:
    context.set(c_vec)
    ids = list(prompt_ids)
    eos_id = tok.eos_token_id
    for _ in range(max_new_tokens):
        input_ids = torch.tensor([ids], device=device)
        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids)).logits
        next_id = int(logits[0, -1].argmax())
        ids.append(next_id)
        if eos_id is not None and next_id == eos_id:
            break
    gen_ids = ids[len(prompt_ids):]
    return tok.decode(gen_ids, skip_special_tokens=True).strip()


def k_marginalised_eval(filt_forward, hyper, model, context, val_filter_batch, val_learner_idx,
                        val_t_idx, tok_batch_by_class, k: int, batch_size: int, arm: str,
                        generator, derangement: torch.Tensor | None = None,
                        eps: torch.Tensor | None = None) -> dict:
    """K-marginalised evaluation of one arm. Uses `atom.simulator.fa_alt2.joint_kc_mixture_nll`,
    the joint (k, c) estimator: class probability and sequence likelihood share each u_k
    (averaging them over K separately would lose that coupling). `eps` (shape `[n, k_eval, latent_dim]`, pre-drawn ONCE by the caller and shared
    across every sampling arm) is sliced per-batch and forwarded to
    `reparameterize_from_log_sigma`, so `personalised` and `shuffled_u` draw IDENTICAL
    noise for the same row -- the only thing that should differ between them is WHOSE
    belief `(mu, log_sigma)` is sampled, never the noise itself. The
    common-completion-budget filter is assumed already applied upstream (by the caller,
    once, on the whole eval population) -- every row entering this function has an
    IDENTICAL completion token count across all 5 classes, which is what makes a single
    `n_valid_tok` per row well-defined for the joint mixture."""
    assert arm in ARMS, arm
    n = len(val_learner_idx)
    if arm == "shuffled_u":
        assert derangement is not None and len(derangement) == n
        belief_l_idx = val_learner_idx[derangement]
        belief_t_idx = val_t_idx[derangement]
    else:
        belief_l_idx, belief_t_idx = val_learner_idx, val_t_idx
    k_eff = 1 if arm == "u_mean" else k

    mixture_nll_all, true_class_nll_spread_all, true_class_nll_all = [], [], []
    exact_match_all, top1_all = [], []
    correctness_prob_meanK_all, correctness_spread_all, correctness_prob_perK_all = [], [], []
    outcome_prob_meanK_full_all = []
    mu_all, log_sigma_all = [], []
    m = hyper.m

    for bstart in range(0, n, batch_size):
        idx = list(range(bstart, min(bstart + batch_size, n)))
        l_idx, t_idx = val_learner_idx[idx], val_t_idx[idx]
        bl_idx, bt_idx = belief_l_idx[idx], belief_t_idx[idx]
        mu_sel, log_sigma_sel = gather_belief_for_batch(filt_forward, val_filter_batch, bl_idx, bt_idx)
        item_codec_sel = gather_item_codec_for_batch(val_filter_batch, l_idx, t_idx)
        mu_all.append(mu_sel.detach().cpu().numpy())
        log_sigma_all.append(log_sigma_sel.detach().cpu().numpy())

        if arm == "u_mean":
            u_k = mu_sel.unsqueeze(1)
        else:
            eps_batch = eps[bstart:bstart + len(idx)].to(mu_sel.device) if eps is not None else None
            u_k = reparameterize_from_log_sigma(mu_sel, log_sigma_sel, k=k_eff,
                                                generator=generator, eps=eps_batch)
        b = len(idx)

        # RAW SUM sequence log-likelihood per (c, k, row), never a mean --
        # `joint_kc_mixture_nll` needs a real sequence probability to combine in the mixture.
        log_p_seq_given_c_k = torch.empty(N_OUTCOME, k_eff, b)
        n_valid_tok_row = None  # one value per row -- identical across classes (budget filter)
        true_class_top1 = torch.zeros(b)
        true_class_exact = torch.zeros(b)
        with torch.no_grad():
            for c in range(N_OUTCOME):
                examples_c = [tok_batch_by_class[c][i] for i in idx]
                tb = collate(examples_c, pad_id=tok_batch_by_class["pad_id"], device=str(u_k.device))
                for j in range(k_eff):
                    if arm == "dw0":
                        c_vec = torch.zeros(b, m, device=u_k.device)
                        context.set(c_vec)
                        out_logits = model(input_ids=tb["input_ids"], attention_mask=tb["attention_mask"],
                                          output_hidden_states=False).logits
                    else:
                        out = forward_personalised(model, hyper, context, u_k[:, j, :],
                                                   tb["input_ids"], tb["attention_mask"],
                                                   tb["prompt_end_index"])
                        out_logits = out["lm_logits"]
                    log_p_seq, n_valid_tok_c, _row_has_any = production_log_likelihood_sum_per_row(
                        out_logits, tb["labels"], tb["production_valid"])
                    log_p_seq_given_c_k[c, j] = log_p_seq.cpu()
                    if j == 0:
                        if c == 0:
                            n_valid_tok_row = n_valid_tok_c.cpu().float()
                        top1, exact, _ = production_topk_per_row(out_logits, tb["labels"], tb["production_valid"])
                        top1_cpu, exact_cpu = top1.cpu(), exact.cpu()
                        mask_c = (tok_batch_by_class["true_class"][idx] == c)
                        true_class_top1[mask_c] = top1_cpu[mask_c]
                        true_class_exact[mask_c] = exact_cpu[mask_c]

        # the TRUE class's own K-averaged sequence NLL (no class-mixing at
        # all here -- length-normalised once, at the end, from the raw-sum quantity).
        log_p_class_kc = torch.empty(N_OUTCOME, k_eff, b)
        outcome_head_local = tok_batch_by_class["outcome_head"]
        with torch.no_grad():
            for j in range(k_eff):
                logits = outcome_head_local(item_codec_sel, u_k[:, j, :])
                log_p_class_kc[:, j, :] = torch.log_softmax(logits, dim=-1).transpose(0, 1).cpu()
        p_class = log_p_class_kc.exp()

        mixture_nll_raw, mixture_nll = joint_kc_mixture_nll(
            log_p_class_kc, log_p_seq_given_c_k, n_valid_tok_row)
        mixture_nll_all.append(mixture_nll.numpy())

        true_class = tok_batch_by_class["true_class"][idx]
        true_log_p_seq_k = torch.stack(
            [log_p_seq_given_c_k[true_class[i], :, i] for i in range(b)], dim=1)  # [K, B]
        true_log_p_seq = torch.logsumexp(true_log_p_seq_k, dim=0) - float(np.log(k_eff))
        true_class_nll_all.append((-true_log_p_seq / n_valid_tok_row.clamp(min=1)).numpy())
        # spread reported per-token (nats/token) for scale-comparability with mixture_nll.
        true_class_nll_spread_all.append(
            (-true_log_p_seq_k / n_valid_tok_row.clamp(min=1)).std(dim=0, unbiased=False).numpy())
        exact_match_all.append(true_class_exact.numpy())
        top1_all.append(true_class_top1.numpy())

        p_correct_k = p_class[0]
        correctness_prob_meanK_all.append(mean_prediction(p_correct_k, dim=0).numpy())
        correctness_spread_all.append(person_epistemic_spread(p_correct_k, dim=0).numpy())
        correctness_prob_perK_all.append(p_correct_k.numpy())
        # the FULL 5-class p(c|u,q) meanK per row, not just class 0 -- outcome
        # NLL/Brier/ECE for every class are then computable offline from persisted
        # artifacts alone, with no retrain.
        outcome_prob_meanK_full_all.append(p_class.mean(dim=1).transpose(0, 1).numpy())  # [b, C]

    return {
        "n_rows": n, "k": k_eff, "arm": arm,
        "mixture_nll": np.concatenate(mixture_nll_all),
        "true_class_nll_k_spread": np.concatenate(true_class_nll_spread_all),
        "true_class_production_nll": np.concatenate(true_class_nll_all),
        "exact_match": np.concatenate(exact_match_all),
        "top1_frac": np.concatenate(top1_all),
        "correctness_prob_meanK": np.concatenate(correctness_prob_meanK_all),
        "correctness_spread": np.concatenate(correctness_spread_all),
        "correctness_prob_perK": np.concatenate(correctness_prob_perK_all, axis=1),
        "outcome_prob_meanK_full": np.concatenate(outcome_prob_meanK_full_all, axis=0),
        "mu": np.concatenate(mu_all, axis=0), "log_sigma": np.concatenate(log_sigma_all, axis=0),
    }


def score_belief_swap_pairs(filt_forward, hyper, model, context, val_filter_batch,
                            eval_bundle: pd.DataFrame, true_class: torch.Tensor, tok,
                            max_len: int, device: str) -> dict:
    """Scores discriminating belief-swap pairs. Returns the four per-direction NLL arrays
    flat (as well as inside `pair_records`), so the caller can persist them directly and feed
    `atom.eval.evals_5g.belief_swap_directed_accuracy` without re-parsing records."""
    val_learner_to_idx = {u: i for i, u in enumerate(val_filter_batch["learners"])}
    l_idx_all = torch.tensor([val_learner_to_idx[u] for u in eval_bundle["learner"]], device=device)
    t_idx_all = torch.tensor(eval_bundle["t"].astype("int64").to_numpy(), device=device)

    pairs = find_discriminating_pairs(eval_bundle["problem_id"].to_numpy(),
                                      eval_bundle["learner"].to_numpy(),
                                      eval_bundle["answer_text"].to_numpy())
    if not pairs:
        return {"pair_records": [], "correct": np.array([]), "edit_distances": np.array([]),
               "nll_i_own": np.array([]), "nll_i_other": np.array([]),
               "nll_j_own": np.array([]), "nll_j_other": np.array([])}

    with torch.no_grad():
        mu_all, _log_sigma_all = gather_belief_for_batch(filt_forward, val_filter_batch, l_idx_all, t_idx_all)

    def nll_of_row_under_belief(row_idx: int, belief_idx: int) -> float:
        r = eval_bundle.iloc[row_idx]
        c = int(true_class[row_idx])
        ex = encode_example(r["question_text"], r["answer_text"], tok, max_len, outcome_code=c)
        tb = collate([ex], pad_id=tok.pad_token_id, device=device)
        u = mu_all[belief_idx:belief_idx + 1]
        with torch.no_grad():
            out = forward_personalised(model, hyper, context, u, tb["input_ids"],
                                       tb["attention_mask"], tb["prompt_end_index"])
            per_row, _ = production_nll_per_row(out["lm_logits"], tb["labels"], tb["production_valid"])
        return float(per_row.item())

    correct, edit_distances, pair_records = [], [], []
    nll_i_own_list, nll_i_other_list, nll_j_own_list, nll_j_other_list = [], [], [], []
    for i, j in pairs:
        nll_i_own = nll_of_row_under_belief(i, i)
        nll_i_other = nll_of_row_under_belief(i, j)
        nll_j_own = nll_of_row_under_belief(j, j)
        nll_j_other = nll_of_row_under_belief(j, i)
        i_favours_own = nll_i_own < nll_i_other
        j_favours_own = nll_j_own < nll_j_other
        is_correct = float(i_favours_own and j_favours_own)
        correct.append(is_correct)
        dist = edit_distance(str(eval_bundle.iloc[i]["answer_text"]), str(eval_bundle.iloc[j]["answer_text"]))
        edit_distances.append(dist)
        nll_i_own_list.append(nll_i_own); nll_i_other_list.append(nll_i_other)  # noqa: E702
        nll_j_own_list.append(nll_j_own); nll_j_other_list.append(nll_j_other)  # noqa: E702
        pair_records.append({
            "i": int(i), "j": int(j), "edit_distance": dist, "correct": is_correct,
            "nll_i_own": nll_i_own, "nll_i_other": nll_i_other,
            "nll_j_own": nll_j_own, "nll_j_other": nll_j_other,
        })
    return {"pair_records": pair_records, "correct": np.array(correct),
           "edit_distances": np.array(edit_distances, dtype=float),
           "nll_i_own": np.array(nll_i_own_list, dtype=float),
           "nll_i_other": np.array(nll_i_other_list, dtype=float),
           "nll_j_own": np.array(nll_j_own_list, dtype=float),
           "nll_j_other": np.array(nll_j_other_list, dtype=float)}


def sync_progress(s3, bucket: str, out_prefix: str, payload: dict) -> None:
    path = pathlib.Path(f"/tmp/fa_sim3_progress_{out_prefix.replace('/', '_')}.json")
    path.write_text(json.dumps(payload, indent=2, default=str))
    s3.upload_file(str(path), bucket, f"{out_prefix}/results/progress.json")


def _periodic_s3_checkpoint(s3, bucket, out_prefix, filt, hyper, outcome_head, bases, optimizer, step):
    state = {
        "filter": filt.state_dict(), "hyper": hyper.state_dict(), "outcome_head": outcome_head.state_dict(),
        "basis": {n: b.state_dict() for n, b in bases.items()}, "optimizer": optimizer.state_dict(),
        "step": step, "sampler_semantics": CURRENT_SAMPLER_SEMANTICS,
    }
    path = pathlib.Path("/tmp/fa_sim3_periodic.pt")
    torch.save(state, path)
    s3.upload_file(str(path), bucket, f"{out_prefix}/checkpoints/periodic_latest.pt")
    print(f"[fa-sim3] periodic checkpoint synced to S3 at step={step}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bucket", required=True)
    ap.add_argument("--fuse-minutes", type=float, default=480.0)
    ap.add_argument("--bundle-path", default="staging/bundles/fa_sim3_train_corrected.tar.gz")
    ap.add_argument("--val-bundle-path", default="staging/bundles/fa_sim3_val_coldstart.tar.gz")
    ap.add_argument("--sim2-bridge-bundle-path", default="staging/bundles/fa_sim1_val_corrected.tar.gz",
                    help="Bridge court: the 3,398-row evaluation population of the earlier, "
                         "smaller FA-SIM2, scored here. Estimator, EOS training, epoch "
                         "sampling and data scale all differ between the two runs, so no "
                         "difference on the new bundle is attributable to scale alone; "
                         "this is the one comparable population.")
    ap.add_argument("--sim2-bridge-max-rows", type=int, default=0,
                    help="Cap the bridge court's row count (0 = uncapped, the full "
                         "3,398-row bundle -- the full-run default). A pilot passes a "
                         "small value so this court scales down with every other pilot "
                         "setting instead of running at full size.")
    ap.add_argument("--manifest-path", default="artifacts/fa_manifest_v2.parquet")
    ap.add_argument("--learner-split-path", default="splits/fa_learner_split_v2.csv")
    ap.add_argument("--item-split-path", default="splits/fa_item_split_v2.csv")
    ap.add_argument("--codec-path", default=str(CODEC_PATH))
    ap.add_argument("--warm-start-checkpoint",
                    default="artifacts/checkpoints/fa_outcome_filter.pt")
    ap.add_argument("--problem-type-map", default="artifacts/fa_problem_type_map.csv")
    ap.add_argument("--backbone", default="gemma4-e4b")
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--micro-batch-size", type=int, default=2)
    ap.add_argument("--grad-accum-steps", type=int, default=8)
    ap.add_argument("--eval-batch-size", type=int, default=8)
    ap.add_argument("--max-steps-personalised", type=int, default=14949,
                    help="Optimizer steps for the personalised arm. Default = ceil("
                         "239,181-row 20pct train population / effective_batch=16) -- "
                         "ONE EPOCH over the 20%% population (the same epoch count as "
                         "FA-SIM2 over its 5%% population: 3,738 steps x 16 = 59,808 ~= "
                         "59,798 rows).")
    ap.add_argument("--k-eval", type=int, default=8)
    ap.add_argument("--k-eval-sweep", default="",
                    help="Comma-separated K values (e.g. '1,4,8,16') for a convergence-"
                         "curve side computation (personalised arm only, reusing the "
                         "SAME trained model and eval population, no retraining) -- does "
                         "the K-sample mixture estimate stabilise as K grows? Empty "
                         "(default) skips it; a pilot run sets this explicitly.")
    ap.add_argument("--n-val-rows-eval", type=int, default=100000)
    ap.add_argument("--n-generation-rows", type=int, default=300,
                    help="Per-arm free-generation subsample size -- 300 suffices because "
                         "the per-arm K=8 NLL/exact-match pass already covers the full "
                         "5,000-row evaluation population.")
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--token-loss-weight", type=float, default=0.3,
                    help="Weight alpha on the token-production loss term in "
                         "`micro_loss = l_outcome + alpha * l_a + beta_kl * kl`. Default 0.3 "
                         "is FA-SIM3's recipe (and the control arm of `infra/sim_alpha_sweep.py`, "
                         "which sweeps this weight). A full retrain at the selected "
                         "alpha* passes it here.")
    ap.add_argument("--kl-max-weight", type=float, default=0.01)
    ap.add_argument("--kl-anneal-steps", type=int, default=1000)
    ap.add_argument("--free-bits", type=float, default=0.05)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--out-prefix", default="fa-sim3-full-run")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--checkpoint-every-s", type=float, default=900.0)
    ap.add_argument("--generations-local-dir", default="",
                    help="Where to write generations.jsonl LOCALLY on the instance "
                        "(licence terms -- raw text is never uploaded to S3). Copy it off "
                        "over SSH after the run. Defaults to /tmp if unset.")
    ap.add_argument("--self-terminate", action="store_true")
    ap.add_argument("--git-sha", default=None,
                    help="Override for the checkpoint's git_sha field when the launch "
                         "rsync excludes .git/ -- pass the launching machine's `git rev-parse HEAD`.")
    args = ap.parse_args()

    codec_path = pathlib.Path(args.codec_path)
    actual_sha = _codec_sha256(codec_path)
    assert actual_sha == EXPECTED_CODEC_SHA, (
        f"codec SHA mismatch: {actual_sha} != {EXPECTED_CODEC_SHA} -- refusing to proceed")
    print(f"[fa-sim3] codec SHA verified: {actual_sha}", flush=True)
    git_sha = args.git_sha or _git_sha()
    print(f"[fa-sim3] git_sha={git_sha}", flush=True)

    seed_everything(args.seed)
    device = default_device()
    cuda_ok = torch.cuda.is_available()
    print(f"[fa-sim3] device={device} cuda_available={cuda_ok}", flush=True)

    subprocess.run(["sudo", "shutdown", "-c"], check=False)
    subprocess.run(["sudo", "shutdown", "-h", f"+{int(args.fuse_minutes)}"], check=False)
    print(f"[fa-sim3] boot fuse armed: {args.fuse_minutes:.0f} min", flush=True)

    import boto3
    s3 = boto3.client("s3")

    gen_dir = pathlib.Path(args.generations_local_dir or "/tmp/fa_sim3_generations")
    gen_dir.mkdir(parents=True, exist_ok=True)
    generations_path = gen_dir / "generations.jsonl"
    gen_f = generations_path.open("w")

    t_start = time.time()
    last_ckpt_t = t_start
    bundle, filter_batch = load_bundle_and_manifest(
        args.bundle_path, args.manifest_path, args.learner_split_path, args.item_split_path,
        args.codec_path, split="train")
    val_bundle, val_filter_batch = load_bundle_and_manifest(
        args.val_bundle_path, args.manifest_path, args.learner_split_path, args.item_split_path,
        args.codec_path, split="val")
    for fb in (filter_batch, val_filter_batch):
        for k in ("item_codec", "prev_feat", "time_min", "valid_mask"):
            fb[k] = fb[k].to(device)
    print(f"[fa-sim3] {len(bundle)} train events / {len(filter_batch['learners'])} train learners, "
         f"{len(val_bundle)} val events / {len(val_filter_batch['learners'])} val learners", flush=True)

    train_exposure_by_problem = bundle["problem_id"].value_counts().to_dict()

    backbone = load_backbone(args.backbone, allow_download=True)
    model, tok = backbone["model"], backbone["tokenizer"]
    bases, context = attach_lora_basis(model, m=32, r=8, seed=args.seed)
    model.enable_input_require_grads()
    model.config.use_cache = False

    filt = S1FilterV2ItemMLP(**B4_CONFIG).to(device)
    warm_start_result = warm_start_filter(filt, REPO / args.warm_start_checkpoint)
    filt.to(device)
    print(f"[fa-sim3] warm start: {warm_start_result}", flush=True)

    filt_forward = lambda ic, pf, tm, vm: filt(ic, pf)  # noqa: E731
    hyper = Hypernetwork(u_dim=B4_CONFIG["latent_dim"], m=32, kind="mlp", seed=args.seed + 1).to(device)
    outcome_head = OutcomeHead(codec_dim=B4_CONFIG["codec_dim"], latent_dim=B4_CONFIG["latent_dim"],
                               hidden=backbone["hidden_dim"]).to(device)

    basis_params = [p for b in bases.values() for p in b.parameters()]
    trainable = list(filt.parameters()) + list(hyper.parameters()) + list(outcome_head.parameters()) + basis_params
    optimizer = torch.optim.Adam(trainable, lr=args.lr)

    pad_id = tok.pad_token_id
    examples_df, examples = build_examples(bundle, tok, args.max_len)
    print(f"[fa-sim3] {len(examples_df)} personalised-arm train examples after outcome-label "
         f"filtering + truncation drop", flush=True)
    print(f"[fa-sim3] class_token_truncated rate (train): "
         f"{np.mean([e['class_token_truncated'] for e in examples]):.4f}", flush=True)

    learner_to_idx = {u: i for i, u in enumerate(filter_batch["learners"])}
    learner_idx_all = torch.tensor([learner_to_idx[u] for u in examples_df["learner"]], device=device)
    t_idx_all = torch.tensor(examples_df["t"].astype("int64").to_numpy(), device=device)
    outcome_target_all = torch.tensor(examples_df["outcome_code"].to_numpy(), dtype=torch.int64, device=device)
    outcome_valid_all = torch.ones(len(examples_df), dtype=torch.bool, device=device)

    n = len(examples_df)
    # TRUE SHUFFLED EPOCHS: independent per-micro-batch draws would cover only ~63.2% of
    # rows per nominal "epoch" (~87,988 of 239,181 never seen). Realised per-item exposure
    # counts are persisted below.
    epoch_sampler = ShuffledEpochSampler(n, np.random.default_rng(args.seed))
    steps = args.max_steps_personalised
    loss_history = {"total": [], "outcome": [], "production": [], "kl": []}
    log_sigma_first, log_sigma_last = None, None

    print(f"[fa-sim3] personalised training: {steps} optimizer steps, "
         f"micro_batch={args.micro_batch_size} x accumulate={args.grad_accum_steps} "
         f"(effective batch={args.micro_batch_size * args.grad_accum_steps}), "
         f"TRUE shuffled epochs over {n} rows", flush=True)
    for step in range(steps):
        optimizer.zero_grad()
        accum_loss = accum_outcome = accum_prod = accum_kl = 0.0
        for micro in range(args.grad_accum_steps):
            idx = epoch_sampler.next_batch(args.micro_batch_size)
            l_idx, t_idx = learner_idx_all[idx], t_idx_all[idx]
            tok_examples = [examples[i] for i in idx]
            tok_batch = collate(tok_examples, pad_id=pad_id, device=device)
            targets = {"outcome_target": outcome_target_all[idx], "outcome_valid": outcome_valid_all[idx]}

            beta_kl = kl_anneal_weight(step, max_weight=args.kl_max_weight, anneal_epochs=args.kl_anneal_steps)
            l_outcome, l_a, _kl_raw, out = losses_for_hierarchical_batch(
                filt_forward, hyper, model, context, outcome_head, filter_batch, l_idx, t_idx,
                tok_batch, targets, beta_kl, k=1)
            mu_sel, log_sigma_sel = gather_belief_for_batch(filt_forward, filter_batch, l_idx, t_idx)
            kl = gaussian_kl_free_bits(mu_sel, log_sigma_sel, free_bits=args.free_bits).mean()
            micro_loss = l_outcome + args.token_loss_weight * l_a + beta_kl * kl
            if not torch.isfinite(micro_loss):
                print(f"[fa-sim3] TRIPWIRE non-finite loss at step={step} micro={micro}: "
                     f"{micro_loss.item()}", flush=True)
                return 1
            (micro_loss / args.grad_accum_steps).backward()
            accum_loss += micro_loss.item() / args.grad_accum_steps
            accum_outcome += l_outcome.item() / args.grad_accum_steps
            accum_prod += l_a.item() / args.grad_accum_steps
            accum_kl += kl.item() / args.grad_accum_steps
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()

        loss_history["total"].append(accum_loss)
        loss_history["outcome"].append(accum_outcome)
        loss_history["production"].append(accum_prod)
        loss_history["kl"].append(accum_kl)

        if step == 0:
            with torch.no_grad():
                _mu_all, log_sigma_all = chunked_filter_forward(filt_forward, filter_batch)
                log_sigma_first = log_sigma_report(log_sigma_all, filter_batch["valid_mask"])
        if step % 500 == 0 or step == steps - 1:
            print(f"[hb] step={step}/{steps} loss={accum_loss:.4f} outcome={accum_outcome:.4f} "
                 f"prod={accum_prod:.4f} kl={accum_kl:.4f} beta_kl={beta_kl:.5f} "
                 f"elapsed={time.time()-t_start:.0f}s", flush=True)
            sync_progress(s3, args.bucket, args.out_prefix, {
                "stage": "personalised", "step": step, "steps": steps,
                "loss": loss_history["total"][-5:], "elapsed_s": time.time() - t_start})
        if time.time() - last_ckpt_t > args.checkpoint_every_s:
            _periodic_s3_checkpoint(s3, args.bucket, args.out_prefix, filt, hyper, outcome_head,
                                    bases, optimizer, step)
            last_ckpt_t = time.time()

    model.eval()
    filt.eval()
    with torch.no_grad():
        _mu_all, log_sigma_all = chunked_filter_forward(filt_forward, filter_batch)
        log_sigma_last = log_sigma_report(log_sigma_all, filter_batch["valid_mask"])
    print(f"[fa-sim3] log_sigma FIRST step: {log_sigma_first}", flush=True)
    print(f"[fa-sim3] log_sigma LAST  step: {log_sigma_last}", flush=True)
    loss_decreased = loss_history["total"][-1] < loss_history["total"][0]
    print(f"[fa-sim3] personalised training done. loss decreased={loss_decreased} "
         f"({loss_history['total'][0]:.4f} -> {loss_history['total'][-1]:.4f})", flush=True)

    # -- REALISED per-item exposure counts (true-epoch coverage) --
    exposure_counts = epoch_sampler.exposure_counts
    exposure_report = {
        "n_rows": int(n), "n_never_seen": int((exposure_counts == 0).sum()),
        "frac_never_seen": float((exposure_counts == 0).mean()),
        "min": int(exposure_counts.min()), "max": int(exposure_counts.max()),
        "mean": float(exposure_counts.mean()),
        "total_draws": int(exposure_counts.sum()),
        "expected_total_draws": int(steps * args.grad_accum_steps * args.micro_batch_size),
    }
    print(f"[fa-sim3] realised exposure counts: {exposure_report}", flush=True)

    # -- eval population (the cold-start-oversampled bundle) --------------------------------
    n_eval = min(args.n_val_rows_eval, len(val_bundle))
    val_learner_to_idx = {u: i for i, u in enumerate(val_filter_batch["learners"])}
    eval_bundle = val_bundle.iloc[:n_eval].reset_index(drop=True)
    labels = outcome_label(eval_bundle["support_code"], eval_bundle["score"], eval_bundle["score_missing"])
    valid = labels["outcome_valid"].to_numpy()
    eval_bundle = eval_bundle.iloc[np.flatnonzero(valid)].reset_index(drop=True)
    true_class = torch.tensor(labels["outcome_code"].to_numpy()[valid], dtype=torch.int64)
    print(f"[fa-sim3] eval population: {len(eval_bundle)} outcome-valid val rows "
         f"(of {n_eval} requested), {(eval_bundle['t'] < 10).sum()} with t<10", flush=True)

    eval_examples_by_class = []
    class_token_truncated_flags = None
    for c in range(N_OUTCOME):
        exs = [encode_example(r["question_text"], r["answer_text"], tok, args.max_len, outcome_code=c)
              for _, r in eval_bundle.iterrows()]
        eval_examples_by_class.append(exs)
        if c == 0:
            class_token_truncated_flags = np.array([e["class_token_truncated"] for e in exs])
    eval_l_idx = torch.tensor([val_learner_to_idx[u] for u in eval_bundle["learner"]], device=device)
    eval_t_idx = torch.tensor(eval_bundle["t"].astype("int64").to_numpy(), device=device)
    tok_batch_by_class = {c: eval_examples_by_class[c] for c in range(N_OUTCOME)}
    tok_batch_by_class["pad_id"] = pad_id
    tok_batch_by_class["true_class"] = true_class
    tok_batch_by_class["outcome_head"] = outcome_head
    print(f"[fa-sim3] class_token_truncated rate (val eval): "
         f"{class_token_truncated_flags.mean():.4f}", flush=True)

    # -- COMMON COMPLETION BUDGET across the 5 class-encodings
    # of a row. Different class tokens have different string lengths, so a row near max_len
    # can have a DIFFERENT completion-token count per class -- a class-marginalised mixture
    # cannot combine likelihoods of differently-censored sequences (this is exactly what
    # `joint_kc_mixture_nll` requires its `n_valid_tok` to be consistent per row). Measured
    # on this run's own eval bundle; mismatched rows are dropped from the ENTIRE eval
    # population (all 4 arms, every downstream stratification) BEFORE anything else runs,
    # so every arm and every panel scores the identical row set, and the count is reported.
    def _content_token_count(ex: dict) -> int:
        return sum(1 for t in ex["labels"] if t != -100)

    token_counts_by_class = np.array([
        [_content_token_count(tok_batch_by_class[c][i]) for c in range(N_OUTCOME)]
        for i in range(len(eval_bundle))
    ])
    budget_consistent = (token_counts_by_class == token_counts_by_class[:, :1]).all(axis=1)
    n_budget_mismatch = int((~budget_consistent).sum())
    print(f"[fa-sim3] completion budget: {n_budget_mismatch} of {len(eval_bundle)} eval rows "
         f"({n_budget_mismatch / len(eval_bundle):.4f}) have a completion budget that differs "
         "across the 5 class-encodings -- excluded from the entire eval population", flush=True)
    if n_budget_mismatch:
        keep = budget_consistent
        kept_positions = np.flatnonzero(keep).tolist()
        eval_bundle = eval_bundle.iloc[kept_positions].reset_index(drop=True)
        true_class = true_class[torch.from_numpy(keep)]
        class_token_truncated_flags = class_token_truncated_flags[keep]
        eval_l_idx = eval_l_idx[torch.from_numpy(keep)]
        eval_t_idx = eval_t_idx[torch.from_numpy(keep)]
        for c in range(N_OUTCOME):
            tok_batch_by_class[c] = [tok_batch_by_class[c][i] for i in kept_positions]
        tok_batch_by_class["true_class"] = true_class

    # -- LEARNER-LEVEL, HISTORY-MATCHED derangement for shuffled_u ---------------------------
    # (a row-level derangement would confound identity with history maturity -- a cold-start
    # row could receive a mature belief -- and could hand a multi-row learner ANOTHER of
    # their OWN rows.) Bins match
    # `atom.eval.evals_5g.history_curve`'s own `DEFAULT_HISTORY_BINS` so the shuffled belief
    # is never more/less mature than the row's own AND the same binning underlies both the
    # derangement and the 5g panel's own history stratification.
    t_values_eval = eval_bundle["t"].to_numpy()
    history_bin_ids = np.digitize(t_values_eval, DEFAULT_HISTORY_BINS[1:-1])
    derangement_rng = np.random.default_rng(args.seed + 100)
    learner_ids_np = eval_bundle["learner"].to_numpy()
    derangement_np, deranged_mask = learner_history_matched_derangement(
        learner_ids_np, history_bin_ids, rng=derangement_rng)
    assert_learner_matched_derangement(derangement_np, learner_ids_np, deranged_mask)
    n_underanged = int((~deranged_mask).sum())
    if n_underanged:
        print(f"[fa-sim3] WARNING: {n_underanged} of {len(deranged_mask)} eval rows could not "
             f"be learner-matched deranged (single-learner history bin) -- dropping them from "
             f"the ENTIRE eval population so every arm scores the same rows",
             flush=True)
        keep = deranged_mask
        eval_bundle = eval_bundle.iloc[np.flatnonzero(keep)].reset_index(drop=True)
        true_class = true_class[torch.from_numpy(keep)]
        class_token_truncated_flags = class_token_truncated_flags[keep]
        eval_l_idx = eval_l_idx[torch.from_numpy(keep)]
        eval_t_idx = eval_t_idx[torch.from_numpy(keep)]
        kept_positions = np.flatnonzero(keep).tolist()
        for c in range(N_OUTCOME):
            tok_batch_by_class[c] = [tok_batch_by_class[c][i] for i in kept_positions]
        tok_batch_by_class["true_class"] = true_class
        deranged_mask = np.ones(len(eval_bundle), dtype=bool)  # every remaining row IS deranged now
        derangement_np, _ = learner_history_matched_derangement(
            eval_bundle["learner"].to_numpy(), history_bin_ids[keep], rng=derangement_rng)
        assert_learner_matched_derangement(derangement_np, eval_bundle["learner"].to_numpy(), deranged_mask)
    derangement = torch.from_numpy(derangement_np)

    # -- FOUR arms ---------------------------------------------------------------------------
    n_eval_rows = len(eval_bundle)
    # CRN -- one shared eps[N, K, d] drawn ONCE and reused across every
    # sampling arm (personalised, shuffled_u), so paired-CI power is not spent on avoidable
    # Monte-Carlo noise between arms that ought to differ only in WHICH belief is sampled.
    crn_generator = torch.Generator(device="cpu").manual_seed(args.seed + 200)
    shared_eps = torch.randn(n_eval_rows, args.k_eval, B4_CONFIG["latent_dim"], generator=crn_generator)
    arm_results = {}
    for arm in ARMS:
        print(f"[fa-sim3] running K-marginalised eval, {arm} arm...", flush=True)
        arm_results[arm] = k_marginalised_eval(
            filt_forward, hyper, model, context, val_filter_batch, eval_l_idx, eval_t_idx,
            tok_batch_by_class, k=args.k_eval, batch_size=args.eval_batch_size, arm=arm,
            generator=None, derangement=derangement, eps=shared_eps)
        print(f"[fa-sim3] {arm}: mixture_nll_mean={arm_results[arm]['mixture_nll'].mean():.4f} "
             f"true_class_production_nll_mean={arm_results[arm]['true_class_production_nll'].mean():.4f} "
             f"elapsed={time.time()-t_start:.0f}s", flush=True)
        sync_progress(s3, args.bucket, args.out_prefix, {
            "stage": f"eval_{arm}_done", "elapsed_s": time.time() - t_start})

    # -- optional K-CONVERGENCE CURVE (personalised arm only, no retraining, same
    # eval population and CRN eps already computed above -- does the K-sample mixture
    # estimate stabilise as K grows?) ---------------------------------------------------------
    k_convergence_curve = {}
    if args.k_eval_sweep.strip():
        k_values = [int(k.strip()) for k in args.k_eval_sweep.split(",") if k.strip()]
        print(f"[fa-sim3] K-convergence sweep: {k_values}", flush=True)
        for k_sweep in k_values:
            eps_sweep = shared_eps[:, :k_sweep, :] if k_sweep <= args.k_eval else torch.randn(
                n_eval_rows, k_sweep, B4_CONFIG["latent_dim"],
                generator=torch.Generator(device="cpu").manual_seed(args.seed + 200))
            r = k_marginalised_eval(
                filt_forward, hyper, model, context, val_filter_batch, eval_l_idx, eval_t_idx,
                tok_batch_by_class, k=k_sweep, batch_size=args.eval_batch_size, arm="personalised",
                generator=None, derangement=derangement, eps=eps_sweep)
            k_convergence_curve[k_sweep] = {
                "mixture_nll_mean": float(r["mixture_nll"].mean()),
                "mixture_nll_std": float(r["mixture_nll"].std()),
                "true_class_nll_k_spread_mean": float(r["true_class_nll_k_spread"].mean()),
            }
            print(f"[fa-sim3] K={k_sweep}: {k_convergence_curve[k_sweep]}", flush=True)

    # -- paired belief-swap eval (per-direction NLLs persisted) -----------------------------
    print("[fa-sim3] running paired belief-swap eval...", flush=True)
    swap_result = score_belief_swap_pairs(
        filt_forward, hyper, model, context, val_filter_batch, eval_bundle, true_class, tok,
        args.max_len, device)
    from atom.eval.belief_swap import graded_accuracy_by_edit_distance
    swap_report = graded_accuracy_by_edit_distance(swap_result["correct"], swap_result["edit_distances"])
    swap_directed = belief_swap_directed_accuracy(
        swap_result["nll_i_own"], swap_result["nll_i_other"],
        swap_result["nll_j_own"], swap_result["nll_j_other"])
    print(f"[fa-sim3] belief-swap: {len(swap_result['pair_records'])} discriminating pairs, "
         f"AND accuracy={swap_report['overall']['accuracy']:.4f} "
         f"directed accuracy={swap_directed['directed']['accuracy']:.4f} "
         f"elapsed={time.time()-t_start:.0f}s", flush=True)
    sync_progress(s3, args.bucket, args.out_prefix, {"stage": "belief_swap_done",
                                                     "elapsed_s": time.time() - t_start})

    # -- MC CLOSED-SET OPTION SCORING: score each offered option under the learner-production
    # distribution and normalise within the option set. Every option is scored with the
    # outcome-conditioned prompt template the model was trained on
    # (`atom.simulator.fa_bread_test.mc_option_log_likelihoods` requires `outcome_code`; an
    # unconditioned template is a prompt shape absent from training and scores below the
    # uniform-option floor). The joint (k, c) scorer (`mc_option_joint_log_likelihoods`,
    # reusing `joint_kc_mixture_nll` unchanged) gives
    # `p(opt) = sum_c sum_k p(c|u_k) p(opt|c,u_k) / K`, matching `k_marginalised_eval`;
    # `u_k` is SAMPLED (K=`args.k_eval`) from this row's persisted `(mu, log_sigma)`, never
    # the point estimate alone. The metric is optional: if `Problems.csv` is absent, or if
    # `assert_above_uniform_option_floor` raises because top-1 falls below the uniform-option
    # floor, it is recorded as skipped with the reason instead of being reported.
    mc_option_scoring = {"skipped": True, "reason": None}
    try:
        from atom.data.fa import load_problems
        from atom.data.fa_mc_options import extract_single_select_mc
        from atom.simulator.fa_bread_test import (assert_above_uniform_option_floor,
                                                   mc_option_joint_log_likelihoods, mc_option_metrics)

        problems = load_problems()
        # `.merge()` inside `extract_single_select_mc` resets the index, so the row's
        # position in `eval_bundle` (which `arm_results[...]["mu"]`'s row order matches
        # exactly, since `eval_bundle` was reset to a clean 0..n-1 index just above) must
        # travel through the join as an explicit COLUMN, not be recovered from `.name`.
        eval_bundle_with_pos = eval_bundle.reset_index().rename(columns={"index": "_eval_row_pos"})
        mc_rows = extract_single_select_mc(problems, eval_bundle_with_pos)
        print(f"[fa-sim3] MC option scoring: {len(mc_rows)} single-select MC eval rows "
             f"({int(mc_rows['response_matched'].sum())} response-matched)", flush=True)
        personalised_mu = arm_results["personalised"]["mu"]  # [n_eval_rows, latent_dim]
        personalised_log_sigma = arm_results["personalised"]["log_sigma"]  # [n_eval_rows, latent_dim]
        # reparameterize_from_log_sigma below runs against mu_row/log_sigma_row on `device`
        # (CUDA), so the generator must be on the same device.
        mc_rng = torch.Generator(device=device).manual_seed(args.seed + 600)
        correct_nlls, correct_top1, response_nlls, response_top1 = [], [], [], []
        n_skipped_degenerate, n_skipped_budget_mismatch, n_options_scored = 0, 0, []
        for _, row in mc_rows.iterrows():
            options = row["options"]
            if len(options) < 2 or any(not isinstance(o, str) or not o.strip() for o in options):
                n_skipped_degenerate += 1
                continue
            pos = int(row["_eval_row_pos"])
            mu_row = torch.tensor(personalised_mu[pos], device=device)
            log_sigma_row = torch.tensor(personalised_log_sigma[pos], device=device)
            u_k = reparameterize_from_log_sigma(mu_row.unsqueeze(0), log_sigma_row.unsqueeze(0),
                                                k=args.k_eval, generator=mc_rng)[0]
            item_codec_row = gather_item_codec_for_batch(
                val_filter_batch, eval_l_idx[pos:pos + 1], eval_t_idx[pos:pos + 1])[0]
            try:
                out = mc_option_joint_log_likelihoods(model, hyper, context, outcome_head, u_k,
                                                       item_codec_row, tok, row["question_text"],
                                                       options, args.max_len, device)
            except ValueError:
                n_skipped_degenerate += 1
                continue
            if not out["budget_consistent"]:
                n_skipped_budget_mismatch += 1
                continue
            ll = out["log_likelihoods"]
            n_options_scored.append(len(options))
            m_correct = mc_option_metrics(ll, int(row["correct_option_index"]))
            correct_nlls.append(m_correct["nll"]); correct_top1.append(m_correct["top1_correct"])  # noqa: E702
            if bool(row["response_matched"]):
                m_resp = mc_option_metrics(ll, int(row["response_option_index"]))
                response_nlls.append(m_resp["nll"]); response_top1.append(m_resp["top1_correct"])  # noqa: E702
        uniform_floor = assert_above_uniform_option_floor(
            top1_rate=float(np.mean(correct_top1)) if correct_top1 else 0.0, n_options=n_options_scored)
        mc_option_scoring = {
            "skipped": False, "n_rows": len(mc_rows), "n_skipped_degenerate": n_skipped_degenerate,
            "n_skipped_budget_mismatch": n_skipped_budget_mismatch, "uniform_option_floor": uniform_floor,
            "correct_option": {"mean_nll": float(np.mean(correct_nlls)) if correct_nlls else None,
                               "top1_rate": float(np.mean(correct_top1)) if correct_top1 else None,
                               "n": len(correct_nlls)},
            "response_option_person_target": {
                "mean_nll": float(np.mean(response_nlls)) if response_nlls else None,
                "top1_rate": float(np.mean(response_top1)) if response_top1 else None,
                "n": len(response_nlls)},
        }
        print(f"[fa-sim3] MC option scoring (joint k,c): {mc_option_scoring}",
             flush=True)
    except Exception as e:  # noqa: BLE001 -- optional metric, never blocks the run
        mc_option_scoring = {"skipped": True, "reason": f"{type(e).__name__}: {e}"}
        print(f"[fa-sim3] MC option scoring SKIPPED: {mc_option_scoring['reason']}",
             flush=True)

    # -- THREE REPORTING COURTS. Scale, EOS training, epoch sampling and the eval population
    # all differ from FA-SIM2, so no difference on the new bundle alone is attributable to
    # scale. (a) FA-SIM2's 3,398-row evaluation population, scored here as the bridge.
    # (b)/(c) are sub-slices of the ALREADY-COMPUTED SIM3 `arm_results` above -- no
    # extra forward passes needed, because `fa_sim3_stage_bundles.py`'s own construction
    # makes `t<10` exactly the deliberately-oversampled 500-row stratum and `t>=10` exactly
    # the naturally-sampled 4,500-row REST stratum.
    print("[fa-sim3] court (a): scoring the FA-SIM2 bridge court...", flush=True)
    bridge_bundle, bridge_filter_batch = load_bundle_and_manifest(
        args.sim2_bridge_bundle_path, args.manifest_path, args.learner_split_path,
        args.item_split_path, args.codec_path, split="val")
    for k_ in ("item_codec", "prev_feat", "time_min", "valid_mask"):
        bridge_filter_batch[k_] = bridge_filter_batch[k_].to(device)
    bridge_learner_to_idx = {u: i for i, u in enumerate(bridge_filter_batch["learners"])}
    bridge_labels = outcome_label(bridge_bundle["support_code"], bridge_bundle["score"],
                                  bridge_bundle["score_missing"])
    bridge_valid = bridge_labels["outcome_valid"].to_numpy()
    bridge_bundle = bridge_bundle.iloc[np.flatnonzero(bridge_valid)].reset_index(drop=True)
    bridge_true_class = torch.tensor(bridge_labels["outcome_code"].to_numpy()[bridge_valid], dtype=torch.int64)
    if args.sim2_bridge_max_rows > 0 and len(bridge_bundle) > args.sim2_bridge_max_rows:
        print(f"[fa-sim3] bridge court capped: {len(bridge_bundle)} -> "
             f"{args.sim2_bridge_max_rows} rows (pilot scale, --sim2-bridge-max-rows)", flush=True)
        bridge_bundle = bridge_bundle.iloc[:args.sim2_bridge_max_rows].reset_index(drop=True)
        bridge_true_class = bridge_true_class[:args.sim2_bridge_max_rows]

    bridge_examples_by_class = {
        c: [encode_example(r["question_text"], r["answer_text"], tok, args.max_len, outcome_code=c)
           for _, r in bridge_bundle.iterrows()]
        for c in range(N_OUTCOME)
    }

    def _content_token_count_local(ex: dict) -> int:
        return sum(1 for t in ex["labels"] if t != -100)

    bridge_counts = np.array([
        [_content_token_count_local(bridge_examples_by_class[c][i]) for c in range(N_OUTCOME)]
        for i in range(len(bridge_bundle))
    ])
    bridge_keep = (bridge_counts == bridge_counts[:, :1]).all(axis=1)
    n_bridge_budget_dropped = int((~bridge_keep).sum())
    if n_bridge_budget_dropped:
        kept_positions = np.flatnonzero(bridge_keep).tolist()
        bridge_bundle = bridge_bundle.iloc[kept_positions].reset_index(drop=True)
        bridge_true_class = bridge_true_class[torch.from_numpy(bridge_keep)]
        for c in range(N_OUTCOME):
            bridge_examples_by_class[c] = [bridge_examples_by_class[c][i] for i in kept_positions]

    bridge_l_idx = torch.tensor([bridge_learner_to_idx[u] for u in bridge_bundle["learner"]], device=device)
    bridge_t_idx = torch.tensor(bridge_bundle["t"].astype("int64").to_numpy(), device=device)
    bridge_tok_batch_by_class = dict(bridge_examples_by_class)
    bridge_tok_batch_by_class["pad_id"] = pad_id
    bridge_tok_batch_by_class["true_class"] = bridge_true_class
    bridge_tok_batch_by_class["outcome_head"] = outcome_head

    bridge_bin_ids = np.digitize(bridge_bundle["t"].to_numpy(), DEFAULT_HISTORY_BINS[1:-1])
    bridge_derangement_np, bridge_deranged_mask = learner_history_matched_derangement(
        bridge_bundle["learner"].to_numpy(), bridge_bin_ids, rng=np.random.default_rng(args.seed + 400))
    n_bridge_underanged = int((~bridge_deranged_mask).sum())
    if n_bridge_underanged:
        keep = bridge_deranged_mask
        kept_positions = np.flatnonzero(keep).tolist()
        bridge_bundle = bridge_bundle.iloc[kept_positions].reset_index(drop=True)
        bridge_true_class = bridge_true_class[torch.from_numpy(keep)]
        bridge_l_idx = bridge_l_idx[torch.from_numpy(keep)]
        bridge_t_idx = bridge_t_idx[torch.from_numpy(keep)]
        for c in range(N_OUTCOME):
            bridge_tok_batch_by_class[c] = [bridge_tok_batch_by_class[c][i] for i in kept_positions]
        bridge_tok_batch_by_class["true_class"] = bridge_true_class
        bridge_derangement_np, _ = learner_history_matched_derangement(
            bridge_bundle["learner"].to_numpy(), bridge_bin_ids[keep],
            rng=np.random.default_rng(args.seed + 401))
    assert_learner_matched_derangement(
        bridge_derangement_np, bridge_bundle["learner"].to_numpy(), np.ones(len(bridge_bundle), dtype=bool))
    bridge_derangement = torch.from_numpy(bridge_derangement_np)
    bridge_eps = torch.randn(len(bridge_bundle), args.k_eval, B4_CONFIG["latent_dim"],
                             generator=torch.Generator(device="cpu").manual_seed(args.seed + 500))

    bridge_arm_results = {}
    for arm in ARMS:
        bridge_arm_results[arm] = k_marginalised_eval(
            filt_forward, hyper, model, context, bridge_filter_batch, bridge_l_idx, bridge_t_idx,
            bridge_tok_batch_by_class, k=args.k_eval, batch_size=args.eval_batch_size, arm=arm,
            generator=None, derangement=bridge_derangement, eps=bridge_eps)
    bridge_learner_ids = bridge_bundle["learner"].to_numpy()
    bridge_court = {
        "n_rows": len(bridge_bundle), "n_budget_mismatch_dropped": n_bridge_budget_dropped,
        "n_underanged_dropped": n_bridge_underanged,
        **{f"{arm}_mixture_nll_mean": float(bridge_arm_results[arm]["mixture_nll"].mean()) for arm in ARMS},
        "paired_ci_personalised_minus_shuffled_u": learner_clustered_bootstrap_ci(
            bridge_arm_results["personalised"]["mixture_nll"], bridge_arm_results["shuffled_u"]["mixture_nll"],
            bridge_learner_ids, n_boot=args.n_boot, seed=args.seed),
        "paired_ci_personalised_minus_dw0": learner_clustered_bootstrap_ci(
            bridge_arm_results["personalised"]["mixture_nll"], bridge_arm_results["dw0"]["mixture_nll"],
            bridge_learner_ids, n_boot=args.n_boot, seed=args.seed),
    }
    print(f"[fa-sim3] bridge court: {bridge_court}", flush=True)

    # (b)/(c): sub-slices of the SIM3 population already scored above.
    learner_ids_eval = eval_bundle["learner"].to_numpy()
    t_lt10_mask = (eval_bundle["t"] < 10).to_numpy()
    t_ge10_mask = ~t_lt10_mask

    def _stratum_court(mask: np.ndarray) -> dict:
        if int(mask.sum()) < 2:
            return {"n": int(mask.sum()), "note": "too few rows for a paired CI"}
        return {
            "n": int(mask.sum()),
            **{f"{arm}_mixture_nll_mean": float(arm_results[arm]["mixture_nll"][mask].mean()) for arm in ARMS},
            "paired_ci_personalised_minus_shuffled_u": learner_clustered_bootstrap_ci(
                arm_results["personalised"]["mixture_nll"][mask], arm_results["shuffled_u"]["mixture_nll"][mask],
                learner_ids_eval[mask], n_boot=args.n_boot, seed=args.seed),
            "paired_ci_personalised_minus_dw0": learner_clustered_bootstrap_ci(
                arm_results["personalised"]["mixture_nll"][mask], arm_results["dw0"]["mixture_nll"][mask],
                learner_ids_eval[mask], n_boot=args.n_boot, seed=args.seed),
        }

    natural_distribution_primary_court = _stratum_court(t_ge10_mask)
    cold_start_stratum_court = _stratum_court(t_lt10_mask)
    print(f"[fa-sim3] natural-distribution PRIMARY court (t>=10, {int(t_ge10_mask.sum())} "
         f"rows): {natural_distribution_primary_court}", flush=True)
    print(f"[fa-sim3] cold-start stratum court (t<10, {int(t_lt10_mask.sum())} rows): "
         f"{cold_start_stratum_court}", flush=True)

    # -- learner-clustered bootstrap CIs (default estimator, pooled SIM3 population --
    # includes the deliberate cold-start oversample; the PRIMARY headline is
    # `natural_distribution_primary_court` above, not this pooled number) --------------------
    ci_dw0_clustered = learner_clustered_bootstrap_ci(
        arm_results["personalised"]["mixture_nll"], arm_results["dw0"]["mixture_nll"],
        learner_ids_eval, n_boot=args.n_boot, seed=args.seed)
    ci_shuffled_clustered = learner_clustered_bootstrap_ci(
        arm_results["personalised"]["mixture_nll"], arm_results["shuffled_u"]["mixture_nll"],
        learner_ids_eval, n_boot=args.n_boot, seed=args.seed)
    ci_umean_clustered = learner_clustered_bootstrap_ci(
        arm_results["personalised"]["mixture_nll"], arm_results["u_mean"]["mixture_nll"],
        learner_ids_eval, n_boot=args.n_boot, seed=args.seed)
    print(f"[fa-sim3] learner-clustered paired CI (personalised - dw0) mixture_nll: "
         f"{ci_dw0_clustered}", flush=True)
    print(f"[fa-sim3] learner-clustered paired CI (personalised - shuffled_u) mixture_nll: "
         f"{ci_shuffled_clustered}", flush=True)
    print(f"[fa-sim3] learner-clustered paired CI (personalised - u_mean) mixture_nll: "
         f"{ci_umean_clustered}", flush=True)

    # -- within-item AUC with a CI ------------------------------------------------------------
    within_item = within_item_auc_paired(
        eval_bundle["problem_id"].to_numpy(),
        (true_class.numpy() == OUTCOME_CLASSES.index("UNAIDED_CORRECT")).astype(float),
        arm_results["personalised"]["correctness_prob_meanK"],
        n_boot=args.n_boot, seed=args.seed)
    within_item = {k: v for k, v in within_item.items() if k != "items"}
    print(f"[fa-sim3] within-item AUC (personalised, paired CI): {within_item}", flush=True)

    # -- hard-stratum Cog Acc, problem-type table, truncation stratification -----------------
    unaided_correct_mask = (true_class.numpy() == OUTCOME_CLASSES.index("UNAIDED_CORRECT"))
    unaided_wrong_mask = (true_class.numpy() == OUTCOME_CLASSES.index("UNAIDED_WRONG"))
    cog_acc = {
        "unaided_correct": {"n": int(unaided_correct_mask.sum()),
                            "exact_match": float(arm_results["personalised"]["exact_match"][unaided_correct_mask].mean())
                            if unaided_correct_mask.any() else None},
        "unaided_wrong": {"n": int(unaided_wrong_mask.sum()),
                          "exact_match": float(arm_results["personalised"]["exact_match"][unaided_wrong_mask].mean())
                          if unaided_wrong_mask.any() else None},
    }
    print(f"[fa-sim3] Cog Acc (hard-stratum exact-match): {cog_acc}", flush=True)

    problem_type_map = {}
    problem_types_for_eval = pd.Series([None] * len(eval_bundle))
    ptm_path = REPO / args.problem_type_map
    if ptm_path.exists():
        ptm = pd.read_csv(ptm_path).set_index("problem_id")["problem_type"].to_dict()
        problem_types_for_eval = eval_bundle["problem_id"].map(ptm)
        eval_types = problem_types_for_eval.to_numpy()
        for t in pd.unique(eval_types[~pd.isna(eval_types)]):
            mask = (eval_types == t)
            problem_type_map[str(t)] = {
                "n": int(mask.sum()),
                "exact_match": float(arm_results["personalised"]["exact_match"][mask].mean()),
                "mixture_nll_mean": float(arm_results["personalised"]["mixture_nll"][mask].mean()),
            }
    print(f"[fa-sim3] problem-type table: {problem_type_map}", flush=True)

    truncation_strata = {}
    for label, mask in (("previously_truncated", class_token_truncated_flags),
                        ("clean", ~class_token_truncated_flags)):
        if mask.any():
            truncation_strata[label] = {
                "n": int(mask.sum()),
                "personalised_mixture_nll_mean": float(arm_results["personalised"]["mixture_nll"][mask].mean()),
                "dw0_mixture_nll_mean": float(arm_results["dw0"]["mixture_nll"][mask].mean()),
            }
    print(f"[fa-sim3] truncation-stratum attribution: {truncation_strata}", flush=True)

    hint_only_code = OUTCOME_CLASSES.index("HINT_ONLY")  # noqa: F841 -- unused, kept for readability
    per_class_nll = {}
    for c, name in enumerate(OUTCOME_CLASSES):
        mask = (true_class.numpy() == c)
        if mask.any():
            per_class_nll[name] = float(arm_results["personalised"]["mixture_nll"][mask].mean())
    print(f"[fa-sim3] per-class mixture_nll (HINT_ONLY separation watch): {per_class_nll}", flush=True)

    se = spread_error_correlation(
        torch.tensor(arm_results["personalised"]["correctness_prob_perK"]),
        unaided_correct_mask.astype(float), dim=0)
    print(f"[fa-sim3] spread-error correlation: {se}", flush=True)

    print(f"[fa-sim3] VRAM peak: {torch.cuda.max_memory_allocated()/1e9:.2f} GB" if cuda_ok else
         "[fa-sim3] VRAM: n/a", flush=True)

    # -- history-curve / per-type / LOO-oracle panel (atom.eval.evals_5g), computed in-run ---
    print("[fa-sim3] computing EVALS 5g panel...", flush=True)
    panel_per_type = per_type_split(
        problem_types_for_eval.to_numpy(), learner_ids_eval,
        arm_results["personalised"]["mixture_nll"], arm_results["shuffled_u"]["mixture_nll"],
        arm_results["personalised"]["exact_match"], arm_results["shuffled_u"]["exact_match"],
        n_boot=args.n_boot, seed=args.seed)
    panel_history = history_curve(
        eval_bundle["t"].to_numpy(), learner_ids_eval,
        arm_results["personalised"]["mixture_nll"], arm_results["shuffled_u"]["mixture_nll"],
        n_boot=args.n_boot, seed=args.seed)
    panel_loo = loo_oracle_ceiling(
        eval_bundle["problem_id"].to_numpy(), true_class.numpy(), eval_bundle["answer_text"].to_numpy(),
        arm_results["personalised"]["exact_match"], train_exposure_by_problem=train_exposure_by_problem)
    print(f"[fa-sim3] EVALS 5g: per_type keys={list(panel_per_type.keys())} "
         f"history_bins={list(panel_history.keys())} "
         f"loo n_covered={panel_loo.get('n_covered')} gap={panel_loo.get('gap_oracle_minus_atom')} "
         f"belief_swap_directed_acc={swap_directed['directed']['accuracy']:.4f}", flush=True)

    # -- TWO GENERATION INSTRUMENTS, both through the segment-reserved encoder
    # (`build_reserved_prompt_ids`, never a raw `tok(prompt)`). (a) CONDITIONAL IMITATION:
    # true class, LABELLED as such, `u` SAMPLED for the personalised arm (so it differs
    # from u_mean, which uses `mu`). (b)
    # ANCESTRAL: class SAMPLED from `p(c|u_k)`, the deployable-simulation instrument (no
    # oracle access to the true class at generation time). Both persisted with an explicit
    # `generation_mode` tag and the CONDITIONED class alongside the true class.
    n_gen = min(args.n_generation_rows, n_eval_rows)
    gen_idx = np.linspace(0, n_eval_rows - 1, n_gen, dtype=int) if n_eval_rows else np.array([], dtype=int)

    # max_new_tokens sized from the MEASURED answer p99 (a fixed 32 truncates ~2.8% of
    # answers).
    answer_token_lens = np.array([
        len(tok(f" {a}", add_special_tokens=False)["input_ids"]) for a in eval_bundle["answer_text"]
    ]) if len(eval_bundle) else np.array([0])
    measured_p99 = int(np.ceil(np.percentile(answer_token_lens, 99)))
    effective_max_new_tokens = max(args.max_new_tokens, measured_p99)
    print(f"[fa-sim3] answer token length p99={measured_p99}, CLI default="
         f"{args.max_new_tokens} -> using max_new_tokens={effective_max_new_tokens}", flush=True)
    print(f"[fa-sim3] generating text for {n_gen} rows x {len(ARMS)} arms x 2 instruments "
         f"(licensed text -> local disk only, never S3)...", flush=True)
    # device-matched, not CPU: both reparameterize_from_log_sigma (mu_row/log_sigma_row are
    # on `device`) and torch.multinomial (class_probs is on `device`) need a generator on
    # the SAME device as the tensors they sample against (a CPU generator raises
    # "Expected a 'cuda' device type for generator").
    gen_rng = torch.Generator(device=device).manual_seed(args.seed + 300)
    for gi in gen_idx.tolist():
        r = eval_bundle.iloc[gi]
        true_c = int(true_class[gi])
        mu_row = torch.tensor(arm_results["personalised"]["mu"][gi:gi + 1], device=device)
        log_sigma_row = torch.tensor(arm_results["personalised"]["log_sigma"][gi:gi + 1], device=device)
        partner = int(derangement[gi]) if derangement is not None else gi
        mu_partner = torch.tensor(arm_results["personalised"]["mu"][partner:partner + 1], device=device)
        log_sigma_partner = torch.tensor(arm_results["personalised"]["log_sigma"][partner:partner + 1], device=device)
        item_codec_row = gather_item_codec_for_batch(val_filter_batch, eval_l_idx[gi:gi + 1], eval_t_idx[gi:gi + 1])

        def _u_for_arm(arm_name):
            if arm_name == "dw0":
                return None  # c_vec is zeroed regardless of u
            if arm_name == "u_mean":
                return mu_row
            if arm_name == "shuffled_u":
                return reparameterize_from_log_sigma(mu_partner, log_sigma_partner, k=1,
                                                     generator=gen_rng)[:, 0, :]
            return reparameterize_from_log_sigma(mu_row, log_sigma_row, k=1,  # personalised: a
                                                 generator=gen_rng)[:, 0, :]  # genuine SAMPLE, not mu

        for arm in ARMS:
            u = _u_for_arm(arm)
            c_vec = torch.zeros(1, hyper.m, device=device) if arm == "dw0" else hyper(u)

            prompt_ids_true, _ = build_reserved_prompt_ids(r["question_text"], tok, args.max_len,
                                                            outcome_code=true_c)
            gen_true = greedy_generate(model, tok, context, c_vec, prompt_ids_true,
                                       effective_max_new_tokens, device)
            gen_f.write(json.dumps({
                "row_idx": int(gi), "arm": arm, "generation_mode": "conditional_imitation",
                "learner": str(r["learner"]), "problem_id": str(r["problem_id"]),
                "true_class": OUTCOME_CLASSES[true_c], "conditioned_class": OUTCOME_CLASSES[true_c],
                "generated_text": gen_true, "real_answer": str(r["answer_text"]),
            }) + "\n")

            # dw0 has no `u` of its own (c_vec zeroed by definition) -- its ancestral class
            # draw uses this row's OWN personalised belief (design choice: "which class to
            # talk about" is a property of the learner; "how personalised the TEXT is" is the
            # arm's own axis, and dw0 tests the latter only).
            u_for_class = u if u is not None else reparameterize_from_log_sigma(
                mu_row, log_sigma_row, k=1, generator=gen_rng)[:, 0, :]
            with torch.no_grad():
                class_probs = torch.softmax(outcome_head(item_codec_row, u_for_class), dim=-1)[0]
            sampled_c = int(torch.multinomial(class_probs, 1, generator=gen_rng).item())
            prompt_ids_ancestral, _ = build_reserved_prompt_ids(r["question_text"], tok, args.max_len,
                                                                 outcome_code=sampled_c)
            gen_ancestral = greedy_generate(model, tok, context, c_vec, prompt_ids_ancestral,
                                            effective_max_new_tokens, device)
            gen_f.write(json.dumps({
                "row_idx": int(gi), "arm": arm, "generation_mode": "ancestral",
                "learner": str(r["learner"]), "problem_id": str(r["problem_id"]),
                "true_class": OUTCOME_CLASSES[true_c], "conditioned_class": OUTCOME_CLASSES[sampled_c],
                "generated_text": gen_ancestral, "real_answer": str(r["answer_text"]),
            }) + "\n")
    gen_f.close()
    print(f"[fa-sim3] generations written to {generations_path} "
         f"({n_gen * len(ARMS) * 2} rows -- x2 for the two generation instruments) "
         "-- LOCAL ONLY, never uploaded to S3", flush=True)

    # -- save checkpoint -----------------------------------------------------------------------
    state = {
        "filter": filt.state_dict(), "hyper": hyper.state_dict(), "outcome_head": outcome_head.state_dict(),
        "basis": {n: b.state_dict() for n, b in bases.items()},
        "architecture": {"class": "S1FilterV2ItemMLP/OutcomeHead", **B4_CONFIG},
        "codec_path": str(args.codec_path), "codec_sha256": actual_sha,
        "sampler_semantics": CURRENT_SAMPLER_SEMANTICS, "git_sha": git_sha,
        "seed": args.seed, "steps_personalised": steps,
        "micro_batch_size": args.micro_batch_size, "grad_accum_steps": args.grad_accum_steps,
        "warm_start": warm_start_result,
    }
    ckpt_path = pathlib.Path("/tmp/fa_sim3_final.pt")
    torch.save(state, ckpt_path)
    s3.upload_file(str(ckpt_path), args.bucket, f"{args.out_prefix}/checkpoints/final.pt")

    # -- predictions.csv.gz: scalars ONLY, no text --------------------------------------------
    pred_dict = {
        "learner": eval_bundle["learner"], "problem_id": eval_bundle["problem_id"],
        "t": eval_bundle["t"], "true_class": true_class.numpy(),
        "class_token_truncated": class_token_truncated_flags,
    }
    for arm in ARMS:
        pred_dict[f"mixture_nll_{arm}"] = arm_results[arm]["mixture_nll"]
        pred_dict[f"true_class_production_nll_{arm}"] = arm_results[arm]["true_class_production_nll"]
        pred_dict[f"exact_match_{arm}"] = arm_results[arm]["exact_match"]
        pred_dict[f"top1_frac_{arm}"] = arm_results[arm]["top1_frac"]
    pred_dict["correctness_prob_meanK_personalised"] = arm_results["personalised"]["correctness_prob_meanK"]
    pred_dict["correctness_spread_personalised"] = arm_results["personalised"]["correctness_spread"]
    predictions = pd.DataFrame(pred_dict)
    pred_path = pathlib.Path("/tmp/fa_sim3_predictions.csv.gz")
    predictions.to_csv(pred_path, index=False, compression="gzip")
    s3.upload_file(str(pred_path), args.bucket, f"{args.out_prefix}/results/predictions.csv.gz")

    # per-row (mu, log_sigma) for every arm + the belief-swap pair table INCLUDING the four
    # per-direction NLLs + the full 5-class p(c|u,q) per arm + realised per-item training
    # exposure counts + the derangement PROVENANCE (mapping AND which rows were genuinely
    # deranged, never just the mapping alone).
    npz_path = pathlib.Path("/tmp/fa_sim3_belief_diagnostics.npz")
    np.savez(npz_path,
            **{f"mu_{arm}": arm_results[arm]["mu"] for arm in ARMS},
            **{f"log_sigma_{arm}": arm_results[arm]["log_sigma"] for arm in ARMS},
            **{f"outcome_prob_meanK_full_{arm}": arm_results[arm]["outcome_prob_meanK_full"] for arm in ARMS},
            derangement=derangement.numpy(),
            deranged_mask=deranged_mask,
            train_exposure_counts=exposure_counts,
            belief_swap_pairs=np.array([[p["i"], p["j"]] for p in swap_result["pair_records"]]),
            belief_swap_edit_distances=swap_result["edit_distances"],
            belief_swap_correct=swap_result["correct"],
            belief_swap_nll_i_own=swap_result["nll_i_own"],
            belief_swap_nll_i_other=swap_result["nll_i_other"],
            belief_swap_nll_j_own=swap_result["nll_j_own"],
            belief_swap_nll_j_other=swap_result["nll_j_other"])
    s3.upload_file(str(npz_path), args.bucket, f"{args.out_prefix}/results/belief_diagnostics.npz")

    summary = {
        "task": "fa_sim3_full_run", "seed": args.seed, "git_sha": git_sha,
        "steps_personalised": steps, "k_eval": args.k_eval,
        "micro_batch_size": args.micro_batch_size, "grad_accum_steps": args.grad_accum_steps,
        "warm_start": warm_start_result,
        "loss_decreased": loss_decreased,
        "loss_history_head_tail": {"first5": loss_history["total"][:5], "last5": loss_history["total"][-5:]},
        "log_sigma_first": log_sigma_first, "log_sigma_last": log_sigma_last,
        "n_train_rows": len(bundle), "n_eval_rows": len(eval_bundle),
        "n_eval_rows_t_lt10": int((eval_bundle["t"] < 10).sum()),
        "class_token_truncated_rate_val": float(class_token_truncated_flags.mean()),
        "requirement_8_exposure_report": exposure_report,
        "k_convergence_curve": k_convergence_curve,
        "requirement_12_budget_mismatch_dropped": n_budget_mismatch,
        "requirement_10_underanged_dropped": n_underanged,
        **{f"{arm}_mixture_nll_mean": float(arm_results[arm]["mixture_nll"].mean()) for arm in ARMS},
        **{f"{arm}_true_class_production_nll_mean":
           float(arm_results[arm]["true_class_production_nll"].mean()) for arm in ARMS},
        "paired_ci_learner_clustered_personalised_minus_dw0_mixture_nll": ci_dw0_clustered,
        "paired_ci_learner_clustered_personalised_minus_shuffled_u_mixture_nll": ci_shuffled_clustered,
        "paired_ci_learner_clustered_personalised_minus_u_mean_mixture_nll": ci_umean_clustered,
        "within_item_auc": within_item,
        "belief_swap": {"n_pairs": len(swap_result["pair_records"]), "report": swap_report,
                        "directed": swap_directed},
        "mc_option_scoring": mc_option_scoring,
        "cog_acc_hard_stratum": cog_acc,
        "problem_type_table": problem_type_map,
        "truncation_stratum_attribution": truncation_strata,
        "per_class_mixture_nll": per_class_nll,
        "spread_error_correlation": se,
        "evals_5g_panel": {
            "per_type_split_personalised_vs_shuffled_u": panel_per_type,
            "history_curve_personalised_vs_shuffled_u": panel_history,
            "loo_oracle_ceiling": panel_loo,
        },
        "requirement_14_reporting_courts": {
            "run_identity": "20% CORRECTED RE-ESTABLISHMENT RUN, not a causal scale rung",
            "a_sim2_bridge_court": bridge_court,
            "b_natural_distribution_primary_court": natural_distribution_primary_court,
            "c_cold_start_stratum_court": cold_start_stratum_court,
        },
        "kl_max_weight": args.kl_max_weight, "kl_anneal_steps": args.kl_anneal_steps,
        "free_bits": args.free_bits, "sampler_semantics": CURRENT_SAMPLER_SEMANTICS,
        "n_generation_rows": n_gen, "max_new_tokens_effective": effective_max_new_tokens,
        "elapsed_s": time.time() - t_start,
        "DEVELOPMENTAL_BASELINE_NOT_A_THESIS_CLAIM": True,
    }
    summary_path = pathlib.Path("/tmp/fa_sim3_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    s3.upload_file(str(summary_path), args.bucket, f"{args.out_prefix}/results/summary.json")
    print(f"[fa-sim3] summary uploaded to bucket {args.bucket}: {args.out_prefix}/results/summary.json",
         flush=True)
    print(f"[fa-sim3] generations.jsonl remains at {generations_path} on THIS INSTANCE ONLY "
         f"-- pull over SSH before termination, never via S3", flush=True)
    print(f"[fa-sim3] DONE. elapsed={time.time()-t_start:.0f}s", flush=True)

    if args.self_terminate:
        print("[fa-sim3] self-terminating instance now", flush=True)
        subprocess.run(["sudo", "shutdown", "-h", "now"], check=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
