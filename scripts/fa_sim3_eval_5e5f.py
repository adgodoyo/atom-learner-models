#!/usr/bin/env python
"""Score FA-SIM3's free generations against the learners' real answers, per arm.

Inputs: `generations.jsonl`, `predictions.csv.gz` and the validation bundle of
infra/fa_sim3_full_run.py. On the `conditional_imitation` rows (`ancestral` rows imitate a
sampled outcome, so they are excluded) it computes exact match, normalised edit distance and
codec similarity of the free generation, next to the teacher-forced numbers, for the four arms,
with learner-clustered bootstrap CIs of personalised minus each control. The evaluation rows are
rebuilt with the training run's filters so `row_idx` joins back to the question text; the join
is checked by row count and by each row's recorded `true_class`. Outputs:
`results/fa_sim3_eval_5e5f/{summary.json,sim_scores.csv.gz}` (no text). Run:
`python scripts/fa_sim3_eval_5e5f.py [--limit-rows N]`. Serves Tables 10 and 12.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import tarfile
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[1]

from atom.binding import attach_lora_basis  # noqa: E402
from atom.codec.backbone import load_backbone  # noqa: E402
from atom.codec.extract import encode_one  # noqa: E402
from atom.codec.transform import CodecTransform  # noqa: E402
from atom.data.fa_manifest import load_v2_events  # noqa: E402
from atom.data.fa_outcome_label import N_OUTCOME, OUTCOME_CLASSES, outcome_label  # noqa: E402
from atom.eval.belief_swap import edit_distance  # noqa: E402
from atom.eval.metrics import learner_clustered_bootstrap_ci  # noqa: E402
from atom.filter.batch import build_padded_batch_v2, load_production_codec  # noqa: E402
from atom.simulator.fa_alt2 import build_prompt_completion, encode_example  # noqa: E402
from atom.utils.logging import default_device  # noqa: E402

B4_CODEC_DIM = 64
ARMS = ("personalised", "dw0", "shuffled_u", "u_mean")
OUTCOME_CLASS_TO_CODE = {name: i for i, name in enumerate(OUTCOME_CLASSES)}


def reconstruct_eval_bundle(val_bundle_path, manifest_path, learner_split_path,
                            item_split_path, codec_path, tok, max_len: int,
                            n_val_rows_eval: int) -> pd.DataFrame:
    """The same construction as `infra/fa_sim3_full_run.py::main`'s `eval_bundle`:
    truncate to n_eval -> outcome-valid filter -> common-completion-budget filter (each row is
    re-tokenised under all N_OUTCOME class encodings and kept only if every encoding has the
    same number of content tokens; this logic is copied from the training run, not
    approximated)."""
    extract_dir = pathlib.Path("/tmp/fa_sim3_eval5e5f_val_bundle")
    extract_dir.mkdir(exist_ok=True)
    with tarfile.open(val_bundle_path, "r:gz") as tf:
        tf.extractall(extract_dir, filter="data")
    val_bundle = pd.read_json(extract_dir / "events.jsonl", lines=True)

    n_eval = min(n_val_rows_eval, len(val_bundle))
    eval_bundle = val_bundle.iloc[:n_eval].reset_index(drop=True)
    labels = outcome_label(eval_bundle["support_code"], eval_bundle["score"], eval_bundle["score_missing"])
    valid = labels["outcome_valid"].to_numpy()
    eval_bundle = eval_bundle.iloc[np.flatnonzero(valid)].reset_index(drop=True)
    true_class = labels["outcome_code"].to_numpy()[valid]
    print(f"[eval5e5f] eval population before the completion-budget filter: {len(eval_bundle)} outcome-valid "
         f"rows (of {n_eval} requested)", flush=True)

    tok_batch_by_class = {}
    for c in range(N_OUTCOME):
        exs = [encode_example(r["question_text"], r["answer_text"], tok, max_len, outcome_code=c)
              for _, r in eval_bundle.iterrows()]
        tok_batch_by_class[c] = exs

    def _content_token_count(ex: dict) -> int:
        return sum(1 for t in ex["labels"] if t != -100)

    token_counts_by_class = np.array([
        [_content_token_count(tok_batch_by_class[c][i]) for c in range(N_OUTCOME)]
        for i in range(len(eval_bundle))
    ])
    budget_consistent = (token_counts_by_class == token_counts_by_class[:, :1]).all(axis=1)
    n_budget_mismatch = int((~budget_consistent).sum())
    print(f"[eval5e5f] completion-budget filter: {n_budget_mismatch} of {len(eval_bundle)} rows dropped "
         f"for cross-class budget mismatch (the training run reports 23)", flush=True)
    if n_budget_mismatch:
        keep = budget_consistent
        eval_bundle = eval_bundle.iloc[np.flatnonzero(keep)].reset_index(drop=True)
        true_class = true_class[keep]

    eval_bundle = eval_bundle.assign(true_class_code=true_class)
    return eval_bundle


def codec_similarity_one(model, tok, device, context, m: int, transform: CodecTransform,
                         prefix: str, real_full: str, gen_full: str,
                         layer: int, k_strip: int, real_cache: dict, real_key) -> float:
    """Cosine similarity of generated and real answers in the frozen codec space.

    The LoRA context is zeroed (an exact no-op, since B is zero-initialised), so this uses the
    plain backbone and needs no trained checkpoint. Real-answer encodings are cached per row,
    since the four arms share the same real answer."""
    context.set(torch.zeros(1, m, device=device))
    bundle = {"model": model, "tokenizer": tok, "device": device}

    if real_key not in real_cache:
        _, mean_states = encode_one(bundle, prefix, real_full)
        raw = mean_states[layer][None, :]
        coords = transform.apply(raw, k=k_strip, rank=None, whiten=False)[0]
        real_cache[real_key] = coords / (np.linalg.norm(coords) + 1e-12)
    real_vec = real_cache[real_key]

    _, mean_states = encode_one(bundle, prefix, gen_full)
    raw = mean_states[layer][None, :]
    gen_vec = transform.apply(raw, k=k_strip, rank=None, whiten=False)[0]
    gen_vec = gen_vec / (np.linalg.norm(gen_vec) + 1e-12)
    return float(np.dot(gen_vec, real_vec))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--generations-path", default="results/fa_sim3_full_run/generations.jsonl")
    ap.add_argument("--predictions-path", default="results/fa_sim3_full_run/predictions.csv.gz")
    ap.add_argument("--summary-path", default="results/fa_sim3_full_run/summary.json")
    ap.add_argument("--val-bundle-path", default="staging/bundles/fa_sim3_val_coldstart.tar.gz")
    ap.add_argument("--manifest-path", default="artifacts/fa_manifest_v2.parquet")
    ap.add_argument("--learner-split-path", default="splits/fa_learner_split_v2.csv")
    ap.add_argument("--item-split-path", default="splits/fa_item_split_v2.csv")
    ap.add_argument("--codec-path", default="artifacts/fa_item_codec_dim64.npz")
    ap.add_argument("--backbone", default="gemma4-e4b")
    ap.add_argument("--transform-path", default="transforms/problem_mean_L15.npz")
    ap.add_argument("--codec-layer", type=int, default=15)
    ap.add_argument("--codec-strip-k", type=int, default=5)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--n-val-rows-eval", type=int, default=100000)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="results/fa_sim3_eval_5e5f")
    ap.add_argument("--limit-rows", type=int, default=0, help="0 = all; >0 truncates for a quick smoke pass")
    args = ap.parse_args()

    t_start = time.time()
    device = default_device()
    print(f"[eval5e5f] device={device}", flush=True)

    backbone = load_backbone(args.backbone, allow_download=True)
    model, tok = backbone["model"], backbone["tokenizer"]
    _bases, context = attach_lora_basis(model, m=32, r=8, seed=args.seed)
    model.config.use_cache = False
    model.eval()
    print(f"[eval5e5f] backbone loaded (fresh LoRA hooks, zeroed at every call)", flush=True)

    eval_bundle = reconstruct_eval_bundle(
        REPO / args.val_bundle_path, REPO / args.manifest_path,
        REPO / args.learner_split_path, REPO / args.item_split_path, REPO / args.codec_path,
        tok, args.max_len, args.n_val_rows_eval)

    predictions = pd.read_csv(REPO / args.predictions_path)
    assert len(predictions) == len(eval_bundle), (
        f"row-count mismatch: predictions.csv.gz has {len(predictions)} rows, reconstructed "
        f"eval_bundle has {len(eval_bundle)} -- reconstruction does not match the run's own "
        f"eval population, refusing to join on row_idx")
    print(f"[eval5e5f] row-count canary PASSED: {len(eval_bundle)} == {len(predictions)}", flush=True)

    gens_all = pd.DataFrame([json.loads(l) for l in open(REPO / args.generations_path)])
    gens = gens_all[gens_all["generation_mode"] == "conditional_imitation"].reset_index(drop=True)
    print(f"[eval5e5f] {len(gens)} conditional_imitation generation rows loaded "
         f"({gens['row_idx'].nunique()} unique row_idx x {gens['arm'].nunique()} arms), "
         f"{len(gens_all) - len(gens)} ancestral rows excluded (different question, not scored here)",
         flush=True)

    gens = gens.merge(
        eval_bundle[["question_text", "true_class_code"]].reset_index().rename(columns={"index": "row_idx"}),
        on="row_idx", how="left")
    assert gens["question_text"].notna().all(), "some generation rows failed to join question_text"

    # -- SEMANTIC CANARY: the join is what it claims to be, not just the right row count ----
    joined_class_name = gens["true_class_code"].map(lambda c: OUTCOME_CLASSES[int(c)])
    mismatch = joined_class_name != gens["true_class"]
    n_mismatch = int(mismatch.sum())
    print(f"[eval5e5f] join-correctness canary: {n_mismatch} of {len(gens)} rows have a "
         f"true_class mismatch between the reconstructed eval_bundle and generations.jsonl's "
         f"own recorded value", flush=True)
    assert n_mismatch == 0, (
        f"{n_mismatch} rows joined to the WRONG eval_bundle row (true_class disagrees with "
        f"what generations.jsonl itself recorded) -- refusing to score a silently-misaligned join")

    if args.limit_rows:
        keep_rows = sorted(gens["row_idx"].unique())[:args.limit_rows]
        gens = gens[gens["row_idx"].isin(keep_rows)].reset_index(drop=True)
        print(f"[eval5e5f] --limit-rows {args.limit_rows}: truncated to {len(gens)} rows "
             f"({len(keep_rows)} unique row_idx x {gens['arm'].nunique()} arms) for a smoke pass",
             flush=True)

    transform = CodecTransform.load(REPO / args.transform_path)
    print(f"[eval5e5f] codec transform loaded: layer={transform.layer}, "
         f"pooling={transform.pooling}, k_max={transform.k_max}", flush=True)

    # -- free-generation metrics: exact match, normalised edit distance, codec similarity ---
    real_cache: dict = {}
    exact_free, edit_free, sim_free = [], [], []
    n = len(gens)
    for i, r in enumerate(gens.itertuples()):
        gen_text = str(r.generated_text).strip()
        real_text = str(r.real_answer).strip()
        exact_free.append(1.0 if gen_text == real_text else 0.0)
        denom = max(len(gen_text), len(real_text), 1)
        edit_free.append(edit_distance(gen_text, real_text) / denom)

        prompt, _ = build_prompt_completion(r.question_text, "", outcome_code=int(r.true_class_code))
        real_full = f"{prompt} {real_text}"
        gen_full = f"{prompt} {gen_text}"
        sim = codec_similarity_one(model, tok, str(device), context, m=32, transform=transform,
                                   prefix=prompt, real_full=real_full, gen_full=gen_full,
                                   layer=args.codec_layer, k_strip=args.codec_strip_k,
                                   real_cache=real_cache, real_key=int(r.row_idx))
        sim_free.append(sim)
        if (i + 1) % 100 == 0 or (i + 1) == n:
            elapsed = time.time() - t_start
            print(f"[eval5e5f] {i + 1}/{n} rows, elapsed={elapsed:.0f}s, "
                 f"eta={elapsed / (i + 1) * (n - i - 1):.0f}s", flush=True)

    gens["exact_match_free"] = exact_free
    gens["normalised_edit_distance_free"] = edit_free
    gens["sim_vs_real_free"] = sim_free
    print(f"[eval5e5f] free-generation metrics computed for all {n} rows, "
         f"real-answer cache hits saved {n - len(real_cache)} redundant encodes, "
         f"elapsed={time.time()-t_start:.0f}s", flush=True)

    # -- assemble the matrix: teacher-forced (from predictions.csv.gz) + free-generation ----
    matrix = {}
    for arm in ARMS:
        row = {
            "exact_match_teacher_forced": float(predictions[f"exact_match_{arm}"].mean()),
            "token_change_distance_teacher_forced": float(1.0 - predictions[f"top1_frac_{arm}"].mean()),
        }
        sub = gens[gens["arm"] == arm]
        row["exact_match_free_generation"] = float(sub["exact_match_free"].mean())
        row["normalised_edit_distance_free_generation"] = float(sub["normalised_edit_distance_free"].mean())
        row["codec_semantic_similarity_free_generation"] = float(sub["sim_vs_real_free"].mean())
        matrix[arm] = row

    learner_ids_p = gens.loc[gens["arm"] == "personalised"].sort_values("row_idx")["learner"].to_numpy()
    ci = {}
    for metric_col, metric_name, lower_is_better in (
        ("exact_match_free", "exact_match_free_generation", False),
        ("normalised_edit_distance_free", "normalised_edit_distance_free_generation", True),
        ("sim_vs_real_free", "codec_semantic_similarity_free_generation", False),
    ):
        vals_p = gens.loc[gens["arm"] == "personalised"].sort_values("row_idx")[metric_col].to_numpy()
        ci[metric_name] = {}
        for arm in ("dw0", "shuffled_u", "u_mean"):
            vals_other = gens.loc[gens["arm"] == arm].sort_values("row_idx")[metric_col].to_numpy()
            result = learner_clustered_bootstrap_ci(vals_p, vals_other, learner_ids_p,
                                                     n_boot=args.n_boot, seed=args.seed)
            result["lower_is_better"] = lower_is_better
            ci[metric_name][arm] = result
            print(f"[eval5e5f] {metric_name} learner-clustered CI (personalised - {arm}): "
                 f"{result}", flush=True)

    out = {
        "task": "fa_sim3_eval_5e5f", "evals_rows": ["5e", "5f"],
        "generation_mode_scored": "conditional_imitation",
        "n_generation_rows": int(gens["row_idx"].nunique()), "n_arms": len(ARMS),
        "matrix": matrix,
        "paired_ci_learner_clustered": ci,
        "join_correctness_canary": {"n_rows_checked": int(len(gens_all[gens_all['generation_mode']=='conditional_imitation'])),
                                    "n_mismatch": n_mismatch},
        "elapsed_s": time.time() - t_start,
        "DEVELOPMENTAL_BASELINE_NOT_A_THESIS_CLAIM": True,
    }
    out_dir = REPO / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(out, indent=2, default=str))
    gens.drop(columns=["generated_text", "real_answer", "question_text"]).to_csv(
        out_dir / "sim_scores.csv.gz", index=False, compression="gzip")
    print(f"[eval5e5f] wrote {out_dir}/summary.json (matrix + CIs, no text) and "
         f"{out_dir}/sim_scores.csv.gz (scalars only, no text)", flush=True)
    print(f"[eval5e5f] DONE. elapsed={time.time()-t_start:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
