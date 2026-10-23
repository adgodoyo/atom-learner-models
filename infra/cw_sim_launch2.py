#!/usr/bin/env python3
"""Code-generation panel for the CodeWorkout generative simulator (Table 11).

Generates first-attempt code for two strata of validation rows -- the 1,196-row
headline-problem stratum and the 591-row held-out-problem stratum -- in two regimes,
outcome-given (`conditional_imitation`, diagnostic) and blind (`ancestral`, `argmax`),
on three arms (`personalised`, `shuffled_u`, `dw0`), always with the refreshed outcome
head (`ratchet-cwsim/refreshed-head`, loaded by `cw_sim_launch2_model.py`).
Inputs: the CodeWorkout manifest and code caches (`--code-map`, `--codealone-*`,
`--diff-*`), the base and refreshed-head checkpoints, the dense validation beliefs
(`--dense-beliefs`) and the test-vector labels.
Outputs, per stratum under `--out-dir`: `generation_numeric_rows.npz`, `summary.json`
(similarity, dist-n diversity, CodeBLEU) and `generations_raw.jsonl` (the raw generated
text, kept local and never uploaded); plus `provenance.json` and `result.json`.
Run: `python infra/cw_sim_launch2.py --code-map ... --test-vector-labels ...
[--strata headline,heldout] [--bucket NAME]`. Numeric outputs can optionally be uploaded
to the S3 bucket given with `--bucket`. Generation resumes per (arm, mode) cell; a boot
fuse shuts the machine down if the run stalls, and an overrun trigger stops a run that
exceeds its time estimate. Because the raw text is never uploaded, copy it off the
machine before shutting down; `--self-terminate` would discard it.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
import time

REPO = pathlib.Path(__file__).resolve().parents[1]
for _p in (REPO / "src", REPO / "infra", REPO / "scripts", REPO):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

import cw_sim_core as cw5  # noqa: E402
from cw_sim_launch2_model import held_out_class_posterior_batch, load_generation_model  # noqa: E402
from cw_4g_codec_rank_item_mlp import rebuild_problem_codec_at_rank  # noqa: E402

from atom.codec.transform import CodecTransform  # noqa: E402
from atom.data.cw_challenge_split import HELD_OUT_ITEMS  # noqa: E402
from atom.data.cw_held_out_generation import join_nearest_prior_belief, load_held_out_problem_attempts  # noqa: E402
from atom.data.cw_manifest import build_episode_level_table  # noqa: E402
from atom.eval.codebleu import codebleu_available, codebleu_one  # noqa: E402
from atom.eval.cw_generation import (  # noqa: E402
    dist_n, numeric_generation_row, save_numeric_generation_rows, summarise_generation_rows,
)

ARMS_LAUNCH2 = ("personalised", "shuffled_u", "dw0")
MODES_LAUNCH2 = ("conditional_imitation", "ancestral", "argmax")
REGIME_OF_MODE = {"conditional_imitation": "OUTCOME-GIVEN (diagnostic)", "ancestral": "BLIND",
                  "argmax": "BLIND"}


def _arm_fuse(minutes: float) -> None:
    subprocess.run(["sudo", "shutdown", "-c"], check=False)
    subprocess.run(["sudo", "shutdown", "-h", f"+{int(minutes)}"], check=False)


def build_headline_scores(m: dict, manifest: dict, device: str, eval_k: int,
                          eval_learner_batch_size: int, row_chunk_size: int | None
                          ) -> tuple[pd.DataFrame, dict]:
    attempts_batch = cw5.select_first_attempts_on_headline(manifest["attempts"], manifest["val_learners"])
    # Filter by episode_id membership (matching the evaluation-batch construction in
    # `cw_sim_core`), not by learner membership -- episode_id is a globally positional
    # cumulative count (`atom/data/cw_sessions.py::sessionize`), not scoped per learner, so a
    # learner-membership filter can admit episodes_batch rows whose episode_id values never
    # appear in attempts_batch (or miss rows it needs), producing a plausible-sized but
    # wrong-membership batch. A value-level join on episode_id avoids this.
    eval_episode_ids = set(attempts_batch["episode_id"])
    episodes_batch = manifest["episodes"][manifest["episodes"]["episode_id"].isin(eval_episode_ids)]
    scores = cw5.score_arm_for_batch(m, attempts_batch, episodes_batch, manifest, cw5.CW5_DIMS, device,
                                     eval_k, eval_learner_batch_size, "personalised",
                                     row_chunk_size=row_chunk_size)
    return attempts_batch, {"personalised": scores}


def build_held_out_scores(m: dict, manifest: dict, dense_beliefs_path: pathlib.Path, device: str,
                          eval_k: int, seed: int, raw_dir: str | None = None) -> tuple[pd.DataFrame, dict]:
    val_learners = set(manifest["val_learners"])
    raw = load_held_out_problem_attempts(val_learners, raw_dir=raw_dir)
    # `episode_id` is a GLOBALLY-POSITIONAL cumulative count over whatever row set
    # `sessionize()` is called on (`atom/data/cw_sessions.py::sessionize`:
    # `(new_episode.cumsum() - 1)`) -- it is NOT a portable identity across different
    # manifest-building chains that sessionize DIFFERENT row scopes. `dense_beliefs`
    # (the refreshed-head validation-prefix export) was computed from a manifest
    # sessionized via `atom.data.cw_manifest.load_attempts_with_episodes()` -- a
    # different function from this script's own `manifest` (built by
    # `cw5.load_cw5_manifest`, which additionally excludes `HELD_OUT_ITEMS` and
    # out-of-scope items before its own sessionize call, which can shift the cumulative
    # count). Building `episodes_table` from `manifest["attempts"]` here would silently
    # join beliefs by a different episode_id namespace. The same chain that produced
    # `dense_beliefs` is therefore used here, and `join_nearest_prior_belief` keeps its
    # strict assertion that every dense-belief episode_id resolves.
    from atom.data.cw_manifest import load_attempts_with_episodes
    belief_source_attempts = load_attempts_with_episodes(raw_dir=raw_dir)
    belief_source_attempts = belief_source_attempts[
        belief_source_attempts["learner"].isin(val_learners)]
    episodes_table = build_episode_level_table(belief_source_attempts)
    dense = np.load(dense_beliefs_path)
    dense_dict = {k: dense[k] for k in ("learner", "episode_id", "mu", "log_sigma")}
    joined = join_nearest_prior_belief(raw, dense_dict, episodes_table)
    print(f"[launch2] held-out stratum: {joined.attrs['n_input_rows']} candidate rows, "
         f"{joined.attrs['n_dropped_no_prior_belief']} dropped (no qualifying prior belief), "
         f"{len(joined)} kept", flush=True)

    transform = CodecTransform.load(REPO / "transforms" / "problem_mean_L15.npz")
    problem_codec, pid_to_idx = rebuild_problem_codec_at_rank(cw5.CODEC_RANK, transform=transform)
    assert set(HELD_OUT_ITEMS).issubset(pid_to_idx), "the 50-item codec must include every held-out problem"
    item_codec = torch.tensor(
        np.stack([problem_codec[pid_to_idx[int(p)]] for p in joined["problem_id"]]),
        dtype=torch.float32, device=device)
    mu = torch.tensor(np.stack(joined["mu"].to_numpy()), dtype=torch.float32, device=device)
    log_sigma = torch.tensor(np.stack(joined["log_sigma"].to_numpy()), dtype=torch.float32, device=device)
    p_class = held_out_class_posterior_batch(m["outcome_head"], item_codec, mu, log_sigma,
                                             k=eval_k, seed=seed, device=device)
    scores = {
        "kept_row_idx": np.arange(len(joined)),
        "mu": mu.cpu().numpy(), "log_sigma": log_sigma.cpu().numpy(),
        "p_class_meanK": p_class.cpu().numpy(),
    }
    attempts_batch = joined.reset_index(drop=True)
    return attempts_batch, {"personalised": scores}


def score_records(m: dict, records: list[dict], transform: CodecTransform, device: str
                  ) -> tuple[list[dict], list[dict]]:
    """Numeric rows (persisted) + CodeBLEU rows (persisted, small) -- mirrors the
    generation-scoring loop of `cw_sim_full_run.py` exactly, so the numbers are computed
    the same way as the base run's generation panel."""
    numeric_rows = []
    codebleu_rows = []
    real_cache: dict = {}
    for rec in records:
        sim = cw5.generation_codec_similarity_one(m, transform, device, rec, real_cache)
        numeric_rows.append(numeric_generation_row(rec, sim))
        if codebleu_available() and rec["generated_code"].strip():
            try:
                cb = codebleu_one(rec["real_completion"], rec["generated_code"])
                codebleu_rows.append({"row_idx": rec["row_idx"], "arm": rec["arm"],
                                      "generation_mode": rec["generation_mode"], **cb})
            except Exception as e:  # pragma: no cover -- disclosed, never silently dropped
                print(f"[launch2] codebleu failed for row {rec['row_idx']}: {e}", flush=True)
    return numeric_rows, codebleu_rows


def save_raw_generation_records(records: list[dict], path: pathlib.Path) -> pathlib.Path:
    """Local-only raw-text persistence, so that other metrics can be derived later
    without re-generating. One JSON object per line, written atomically via a `.tmp`
    file. Never uploaded: no `--bucket` upload call in this file references it, because
    the generated and real CodeWorkout code falls under the dataset licence and stays on
    the machine that produced it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w") as f:
        for rec in records:
            f.write(json.dumps({
                "row_idx": rec["row_idx"], "arm": rec["arm"], "generation_mode": rec["generation_mode"],
                "learner": rec["learner"], "problem_id": rec["problem_id"],
                "true_class_code": rec["true_class_code"], "conditioned_class_code": rec["conditioned_class_code"],
                "generated_code": rec["generated_code"], "real_completion": rec["real_completion"],
                "prior_code": rec["prior_code"], "prompt_text": rec["prompt_text"],
                "code_state_id": rec["code_state_id"],
            }) + "\n")
    tmp.replace(path)
    return path


def load_raw_generation_records(path: pathlib.Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def dist_n_by_cell(records: list[dict], arms: tuple, modes: tuple) -> dict:
    """Cell-level (arm, mode) dist-1/dist-2 diversity, computed from the raw text
    locally -- only these small scalars enter the summary."""
    out = {}
    for mode in modes:
        out[mode] = {}
        for arm in arms:
            texts = [r["generated_code"] for r in records if r["arm"] == arm and r["generation_mode"] == mode]
            out[mode][arm] = {"dist_1": dist_n(texts, 1), "dist_2": dist_n(texts, 2), "n_texts": len(texts)}
    return out


def cumulative_cell_progress(cell_index: int, done: int, total_per_cell: int) -> int:
    """Cumulative units completed across an entire stratum, given the 0-based index of
    the CURRENT (arm, mode) cell and its own per-cell `done` count. `run_batched_
    generation_pass`'s `on_progress(arm, mode, done, total, elapsed)` resets `done`/
    `total` to a fresh per-cell count at the start of every new cell, while `elapsed`
    keeps accumulating across the WHOLE stratum -- passing the raw per-cell `done`
    straight to `OverrunTrigger.check()` against a full-stratum `total_units`
    denominator under-credits every already-completed prior cell, so the ratio spikes
    at the start of every new cell even on a healthy run. This matters because
    `OverrunTrigger._fire()` self-terminates unconditionally (it is not gated by
    `--self-terminate`), so the naive per-cell count would stop a healthy run on
    entering its second cell."""
    return cell_index * total_per_cell + done


def run_stratum(name: str, m: dict, attempts_batch: pd.DataFrame, manifest: dict, scores_by_arm: dict,
                device: str, max_new_tokens: int, seed: int, batch_size: int, out_dir: pathlib.Path,
                bucket: str, out_prefix: str, estimated_minutes: float, fuse_minutes: float) -> dict:
    stratum_dir = out_dir / name
    stratum_dir.mkdir(parents=True, exist_ok=True)
    completed_path = stratum_dir / "generation_numeric_rows.npz"
    completed_row_idx = None
    if completed_path.exists():
        from atom.eval.cw_generation import completed_row_indices, load_numeric_generation_rows
        prior_rows = load_numeric_generation_rows(completed_path)
        completed_row_idx = completed_row_indices(prior_rows, arms=ARMS_LAUNCH2, modes=MODES_LAUNCH2)
        print(f"[launch2:{name}] resume: {len(completed_row_idx)} rows already complete", flush=True)

    overrun = cw5.OverrunTrigger(estimated_minutes, stratum_dir, bucket, out_prefix, f"launch2-{name}")
    t0 = time.time()
    cell_order: list[tuple[str, str]] = []  # first-seen order; cells are visited

    # contiguously (nested arm/mode loop), so .index() below is always O(1) in practice.

    def on_progress(arm, mode, done, total, elapsed):
        _arm_fuse(fuse_minutes)
        cell = (arm, mode)
        if cell not in cell_order:
            cell_order.append(cell)
        cumulative_done = cumulative_cell_progress(cell_order.index(cell), done, total)
        cumulative_total = total * len(ARMS_LAUNCH2) * len(MODES_LAUNCH2)
        print(f"[launch2:{name}] arm={arm} mode={mode} {done}/{total} rows "
             f"(cumulative {cumulative_done}/{cumulative_total}, {elapsed:.0f}s elapsed)", flush=True)
        if overrun.check(cumulative_done, cumulative_total, time.time() - t0):
            raise SystemExit(1)

    records = cw5.run_batched_generation_pass(
        m, attempts_batch, manifest, scores_by_arm, device, len(attempts_batch), max_new_tokens,
        seed, arms=ARMS_LAUNCH2, modes=MODES_LAUNCH2, batch_size=batch_size,
        on_progress=on_progress, completed_row_idx=completed_row_idx)

    # Local only, never uploaded -- retained so other metrics can be derived later
    # without re-generating. Merge with any already-persisted raw records the same way
    # the numeric path merges `prior_numeric` on resume.
    raw_path = stratum_dir / "generations_raw.jsonl"
    prior_raw = load_raw_generation_records(raw_path) if completed_row_idx else []
    save_raw_generation_records(prior_raw + records, raw_path)

    if completed_row_idx:
        from atom.eval.cw_generation import load_numeric_generation_rows
        prior_numeric = load_numeric_generation_rows(completed_path)
    else:
        prior_numeric = []

    transform = CodecTransform.load(REPO / "transforms" / "problem_mean_L15.npz")
    new_numeric, codebleu_rows = score_records(m, records, transform, device)
    all_numeric = prior_numeric + new_numeric
    save_numeric_generation_rows(all_numeric, completed_path)
    if bucket:
        import boto3
        boto3.client("s3").upload_file(str(completed_path), bucket,
                                       f"{out_prefix.rstrip('/')}/{name}_generation_numeric_rows.npz",
                                       ExtraArgs={"ServerSideEncryption": "AES256"})

    diversity = dist_n_by_cell(records, ARMS_LAUNCH2, MODES_LAUNCH2)
    summary = summarise_generation_rows(all_numeric, arms=ARMS_LAUNCH2, modes=MODES_LAUNCH2)
    summary["dist_n"] = diversity
    if codebleu_rows:
        summary["codebleu"] = {
            "n_scored": len(codebleu_rows),
            "mean_by_arm_mode": {
                f"{arm}__{mode}": float(np.mean([r["codebleu"] for r in codebleu_rows
                                                 if r["arm"] == arm and r["generation_mode"] == mode]))
                for arm in ARMS_LAUNCH2 for mode in MODES_LAUNCH2
                if any(r["arm"] == arm and r["generation_mode"] == mode for r in codebleu_rows)
            },
            "component_means": {
                comp: float(np.mean([r[comp] for r in codebleu_rows]))
                for comp in ("ngram_match", "weighted_ngram_match", "syntax_match", "dataflow_match")
            },
        }
    else:
        summary["codebleu"] = "unavailable_on_this_platform" if not codebleu_available() else "no_rows_scored"
    summary["regime_of_mode"] = REGIME_OF_MODE
    summary["stratum"] = name
    summary["n_rows_in_stratum"] = len(attempts_batch)
    (stratum_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=True))
    if bucket:
        import boto3
        boto3.client("s3").upload_file(str(stratum_dir / "summary.json"), bucket,
                                       f"{out_prefix.rstrip('/')}/{name}_summary.json")
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default=None)
    for name in ("code_map", "diff_map", "codealone_meta", "codealone_states", "diff_meta", "diff_states"):
        ap.add_argument("--" + name.replace("_", "-"), required=True)
    ap.add_argument("--base-checkpoint", default="results/cw_sim_base_run/final.pt")
    ap.add_argument("--head-checkpoint", default="results/cw_sim_head_refresh/final.pt")
    ap.add_argument("--dense-beliefs", default="artifacts/cw_sim_refreshed_head_val_prefix_beliefs.npz")
    ap.add_argument("--test-vector-labels", required=True)
    ap.add_argument("--strata", default="headline,heldout")
    ap.add_argument("--eval-k", type=int, default=8)
    ap.add_argument("--eval-learner-batch-size", type=int, default=8)
    ap.add_argument("--eval-row-chunk-size", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="results/cw_sim_launch2")
    ap.add_argument("--bucket", default="")
    ap.add_argument("--out-prefix", default="cw-sim-launch2")
    ap.add_argument("--estimated-minutes", type=float, default=180.0)
    ap.add_argument("--fuse-minutes", type=float, default=600.0)
    ap.add_argument("--self-terminate", action="store_true")
    args = ap.parse_args()

    _arm_fuse(args.fuse_minutes)
    print(f"[launch2] boot fuse armed: shutdown in {args.fuse_minutes} min unless cancelled", flush=True)

    device = cw5.default_device()
    manifest = cw5.load_cw5_manifest(args)
    tv_labels_path = pathlib.Path(args.test_vector_labels)
    print(f"[launch2] manifest: {len(manifest['attempts'])} attempts, "
         f"{len(manifest['val_learners'])} val learners, device={device}", flush=True)

    m, provenance = load_generation_model(pathlib.Path(args.base_checkpoint), pathlib.Path(args.head_checkpoint),
                                          device, allow_download=True)
    print(f"[launch2] model loaded: {json.dumps(provenance, indent=2)}", flush=True)

    out_dir = REPO / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "provenance.json").write_text(json.dumps(provenance, indent=2))
    if args.bucket:
        import boto3
        boto3.client("s3").upload_file(str(out_dir / "provenance.json"), args.bucket,
                                       f"{args.out_prefix.rstrip('/')}/provenance.json")

    strata = args.strata.split(",")
    summaries = {}

    if "headline" in strata:
        attempts_batch, scores_by_arm = build_headline_scores(
            m, manifest, device, args.eval_k, args.eval_learner_batch_size, args.eval_row_chunk_size)
        print(f"[launch2] headline stratum: {len(attempts_batch)} rows", flush=True)
        summaries["headline"] = run_stratum(
            "headline", m, attempts_batch, manifest, scores_by_arm, device, args.max_new_tokens,
            args.seed, args.batch_size, out_dir, args.bucket, args.out_prefix,
            args.estimated_minutes, args.fuse_minutes)

    if "heldout" in strata:
        attempts_batch, scores_by_arm = build_held_out_scores(
            m, manifest, REPO / args.dense_beliefs, device, args.eval_k, args.seed, raw_dir=args.raw_dir)
        print(f"[launch2] held-out stratum: {len(attempts_batch)} rows", flush=True)
        summaries["heldout"] = run_stratum(
            "heldout", m, attempts_batch, manifest, scores_by_arm, device, args.max_new_tokens,
            args.seed, args.batch_size, out_dir, args.bucket, args.out_prefix,
            args.estimated_minutes, args.fuse_minutes)

    result = {"provenance": provenance, "strata": summaries}
    result_path = out_dir / "result.json"
    result_path.write_text(json.dumps(result, indent=2, allow_nan=True))
    if args.bucket:
        import boto3
        boto3.client("s3").upload_file(str(result_path), args.bucket,
                                       f"{args.out_prefix.rstrip('/')}/result.json")
    print(json.dumps({"strata": list(summaries), "provenance": provenance}, indent=2), flush=True)

    if args.self_terminate:
        subprocess.run(["sudo", "shutdown", "-h", "now"], check=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
