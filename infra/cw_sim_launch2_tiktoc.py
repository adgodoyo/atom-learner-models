#!/usr/bin/env python3
"""TIKTOC same-data evaluation of the CodeWorkout generative simulator (Table 11).

Scores the simulator, with its refreshed outcome head (`ratchet-cwsim/refreshed-head`),
on TIKTOC's 17-problem, 246-student, 3,714-row first-submission set in two halves:
1. Test-vector AUC/F1 (not a generation task, so it has no blind/outcome-given split),
   using `test_vector_eval_predictions`/`test_vector_auc_panel` from `cw_sim_full_run.py`
   and `atom.eval.tiktoc_metrics.tiktoc_comparison_panel`, which compares against
   TIKTOC's published Table 3 (AUC 0.764 / F1 0.794 / accuracy 0.723).
2. CodeBLEU in the blind regime (`ancestral`, `argmax`); `conditional_imitation` is
   reported alongside as a diagnostic, never substituted for the blind comparison.
   Only the `personalised` arm is run, because TIKTOC's reported numbers are
   single-system comparisons (CodeBLEU 0.554 TIKTOC / 0.522 OKT-TC).
Inputs: the TIKTOC data (`--tiktoc-dir`), the TIKTOC code-state cache written by
`infra/cw_03_encode_tiktoc.py` (`--code-map`, `--codealone-*`; a separate hash-keyed
cache, since TIKTOC's code text differs from S19/F19's), the shared 50-item problem
codec and statements (coverage is asserted, no rebuild), and the base and refreshed-head
checkpoints. No diff-scope cache is needed: every TIKTOC episode has length 1.
Outputs under `--out-dir`: `auc_f1_panel.json`, `codebleu_panel.json`,
`generation_numeric_rows.npz`, `generations_raw.jsonl` (raw text, local only, never
uploaded), `provenance.json`, `result.json`.
Run: `python infra/cw_sim_launch2_tiktoc.py [--tiktoc-dir DIR] [--pilot-rows N]
[--bucket NAME]`; the panels can optionally be uploaded to the S3 bucket given with
`--bucket`. Copy the raw JSONL off the machine before shutting it down; `--self-terminate`
would discard it.
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

import cw_sim_core as cw5  # noqa: E402
from cw_sim_launch2 import (  # noqa: E402
    load_raw_generation_records, save_raw_generation_records, score_records,
)
from cw_sim_launch2_model import load_generation_model  # noqa: E402

from atom.codec.extract import text_hash  # noqa: E402
from atom.codec.transform import CodecTransform  # noqa: E402
from atom.data.cw_manifest import build_codec_vector_lookup  # noqa: E402
from atom.data.cw_tiktoc_transfer import build_tiktoc_manifest, build_tiktoc_test_vector_labels  # noqa: E402
from atom.eval.cw_generation import completed_row_indices, save_numeric_generation_rows  # noqa: E402
from atom.eval.tiktoc_metrics import tiktoc_comparison_panel  # noqa: E402

ARMS_TIKTOC = ("personalised",)
MODES_TIKTOC = ("conditional_imitation", "ancestral", "argmax")  # ancestral/argmax = the
# blind comparison; conditional_imitation is the labelled diagnostic reported
# alongside it, as in cw_sim_launch2.py.


def _arm_fuse(minutes: float) -> None:
    subprocess.run(["sudo", "shutdown", "-c"], check=False)
    subprocess.run(["sudo", "shutdown", "-h", f"+{int(minutes)}"], check=False)


def attach_tiktoc_codec(manifest: dict, code_map_path: pathlib.Path, codealone_meta: pathlib.Path,
                        codealone_states: pathlib.Path, transform: CodecTransform) -> dict:
    """Extends `build_tiktoc_manifest`'s output with the four keys `forward_outcome`/
    `build_attempt_batch` read (`code_of_hash`, `diff_of_hash`, `problem_codec`,
    `pid_to_idx`) plus `statements` -- the same shared 50-item problem codec/statements
    the CodeWorkout simulator uses (no TIKTOC-specific rebuild, coverage asserted below),
    and a TIKTOC-specific
    codealone hash lookup built by `infra/cw_03_encode_tiktoc.py`.

    Mirrors the bad-episode drop of `cw_sim_core.load_cw5_manifest` for
    rows whose code has no cached hash: drops the WHOLE episode (never just the row),
    sized in `manifest['hash_stats']`, never silently absorbed. `diff_hash` is NaN for
    every row (TIKTOC has no diff scope, per the ingest module's own docstring) and
    `diff_of_hash` is the empty dict -- `build_attempt_batch`'s own "missing diff ->
    zero vector" convention covers every TIKTOC row already."""
    attempts = manifest["attempts"]
    code_map = np.load(code_map_path, allow_pickle=True)
    code_map_df = pd.DataFrame({"code_state_id": code_map["code_state_id"],
                               "code_hash": code_map["code_hash"]})
    ev = attempts.merge(code_map_df, on="code_state_id", how="left")
    ev["diff_hash"] = np.nan

    bad_episodes = set(ev.loc[ev["code_hash"].isna(), "episode_id"].unique())
    n_bad = int(ev["code_hash"].isna().sum())
    clean = ev.loc[~ev["episode_id"].isin(bad_episodes)].reset_index(drop=True)
    hash_stats = {"n_attempts_missing_code_hash": n_bad, "n_episodes_dropped_for_missing_code": len(bad_episodes),
                 "n_attempts_before": int(len(attempts)), "n_attempts_after": int(len(clean))}

    code_hashes_needed = set(clean["code_hash"].dropna())
    code_of_hash = build_codec_vector_lookup(code_hashes_needed, codealone_meta, codealone_states,
                                             transform, k=5, rank=cw5.CODEC_RANK)

    item = np.load(cw5.PROBLEM_CODEC_DIM64_PATH, allow_pickle=True)
    problem_codec = item["codec"].astype(np.float32)
    pid_to_idx = {int(p): i for i, p in enumerate(item["problem_id"])}
    statements = cw5.load_problem_statements()

    missing_codec = set(clean["problem_id"].unique()) - set(pid_to_idx.keys())
    assert not missing_codec, (
        f"{len(missing_codec)} TIKTOC problem_id(s) absent from the shared dim64 problem "
        f"codec: {missing_codec} -- cannot score these rows at all, must not proceed silently")

    episodes = cw5.build_episode_level_table(clean)
    return {
        **manifest, "attempts": clean, "episodes": episodes,
        "code_of_hash": code_of_hash, "diff_of_hash": {},
        "problem_codec": problem_codec, "pid_to_idx": pid_to_idx, "statements": statements,
        "hash_stats": hash_stats,
    }


def build_tiktoc_tv_lookup(base_dir=None) -> dict:
    """In-memory adapter: `build_tiktoc_test_vector_labels`'s array-shaped output into
    the exact `{"lookup": {(learner,problem_id,code_state_id): {...}}, ...}` shape
    `infra.cw_sim_full_run.load_test_vector_labels` produces from disk -- same key
    construction, no disk round-trip (TIKTOC's labels are small and computed locally).

    `base_dir` must thread the same `--tiktoc-dir` `build_tiktoc_manifest` receives --
    left un-threaded, this silently falls back to `default_cw_dir()` while the manifest
    build correctly used the CLI override, so on any host where the TIKTOC granular file
    isn't at the default path (e.g. `--tiktoc-dir` pointing elsewhere) the label build
    would raise FileNotFoundError or read a different file from the manifest's."""
    d = build_tiktoc_test_vector_labels(base_dir=base_dir)
    lookup = {}
    for i in range(len(d["learner"])):
        key = (str(d["learner"][i]), int(d["problem_id"][i]), str(d["code_state_id"][i]))
        lookup[key] = {"test_vector": d["test_vector"][i], "valid_mask": d["valid_mask"][i],
                      "n_cases": int(d["n_cases"][i]), "compiled": bool(d["compiled"][i])}
    return {"lookup": lookup, "max_test_cases": int(d["max_test_cases"]),
           "headline_problem_ids": set(int(x) for x in d["headline_problem_ids"])}


def run_auc_f1_panel(m: dict, manifest: dict, tv_labels: dict, device: str,
                     micro_batch_size: int = 256) -> dict:
    """Chunked `forward_outcome` + `infra.cw_sim_full_run.test_vector_eval_predictions`
    over the full TIKTOC manifest (3,714 rows before drops -- small enough to need no
    resumability of its own, unlike the blocked evaluation in `cw_sim_full_run.py`),
    pooled into `tiktoc_comparison_panel`. `cw_sim_full_run` is imported locally and
    lazily (as in `cw_sim_launch2_batch_equality_check.py`): it is a large module of
    which this script needs only two functions.

    Learner-aligned chunking (a small pilot that fits in one chunk does not exercise
    this): `test_vector_eval_predictions` filters its internal `episodes_batch` by
    learner membership (the convention of `cw_sim_full_run.py`, whose
    `run_blocked_eval` chunks by learner blocks, never by row count). A plain
    row-count slice can split one learner's episodes across two chunks -- for TIKTOC
    (up to 17 rows/episodes per learner across the WHOLE manifest, no train/val split to
    narrow it), that breaks `_distinct_episode_positions`'s positional
    zip(learner-filtered episodes_batch, chunk-scoped `ep_batch`) assumption: episodes
    belonging to a learner whose OTHER rows fell in a different chunk go missing from
    `ep_batch`'s zip partner, surfacing as `KeyError` on that episode id. Chunking by
    whole learners (never splitting one learner's rows) restores the invariant that
    `run_blocked_eval` relies on, with no change to the shared function itself."""
    import cw_sim_full_run as cwsim

    attempts = manifest["attempts"]
    learner_order = attempts["learner"].drop_duplicates().tolist()
    rows_per_learner = attempts.groupby("learner", sort=False).size().to_dict()

    all_preds = []
    i = 0
    while i < len(learner_order):
        block_learners: list[str] = []
        block_rows = 0
        while i < len(learner_order) and (not block_learners or block_rows < micro_batch_size):
            learner = learner_order[i]
            block_learners.append(learner)
            block_rows += rows_per_learner[learner]
            i += 1
        block = attempts[attempts["learner"].isin(set(block_learners))].reset_index(drop=True)
        block_episode_ids = set(block["episode_id"])
        block_episodes = manifest["episodes"][manifest["episodes"]["episode_id"].isin(block_episode_ids)]
        out = cw5.forward_outcome(m, block, block_episodes, manifest, cw5.CW5_DIMS, device, free_bits=0.05)
        pred = cwsim.test_vector_eval_predictions(m, block, out, manifest, tv_labels, device)
        if pred is not None:
            all_preds.append(pred)

    auc_panel = cwsim.test_vector_auc_panel(all_preds)
    comparison = tiktoc_comparison_panel(all_preds, tv_labels["headline_problem_ids"])
    return {"auc_panel": auc_panel, "comparison_panel": comparison, "n_chunks": len(all_preds)}


def run_codebleu_blind_panel(m: dict, manifest: dict, device: str, max_new_tokens: int, seed: int,
                             batch_size: int, out_dir: pathlib.Path, estimated_minutes: float,
                             fuse_minutes: float) -> dict:
    """Mirrors `cw_sim_launch2.run_stratum`, restricted to `ARMS_TIKTOC`/`MODES_TIKTOC`
    (personalised arm; blind + diagnostic modes). Only rows whose problem_id has a
    statement (`manifest['statements']`, `row_generation_plan`'s own prompt-building
    dependency) are generation-eligible -- a DIFFERENT, narrower filter than the AUC/F1
    half's codec-only coverage requirement, disclosed and counted here rather than
    silently intersected upstream."""
    attempts = manifest["attempts"]
    has_statement = attempts["problem_id"].isin(manifest["statements"].keys())
    n_no_statement = int((~has_statement).sum())
    gen_attempts = attempts.loc[has_statement].reset_index(drop=True)
    gen_episode_ids = set(gen_attempts["episode_id"])
    gen_episodes = manifest["episodes"][manifest["episodes"]["episode_id"].isin(gen_episode_ids)]

    scores = cw5.score_arm_for_batch(m, gen_attempts, gen_episodes, manifest, cw5.CW5_DIMS, device,
                                     k=8, micro_batch_size=8, arm="personalised")
    scores_by_arm = {"personalised": scores}

    out_dir.mkdir(parents=True, exist_ok=True)
    completed_path = out_dir / "generation_numeric_rows.npz"
    completed_row_idx = None
    if completed_path.exists():
        from atom.eval.cw_generation import load_numeric_generation_rows
        prior_rows = load_numeric_generation_rows(completed_path)
        completed_row_idx = completed_row_indices(prior_rows, arms=ARMS_TIKTOC, modes=MODES_TIKTOC)

    overrun = cw5.OverrunTrigger(estimated_minutes, out_dir, "", "launch2-tiktoc", "launch2-tiktoc")
    t0 = time.time()
    cell_order: list[tuple[str, str]] = []

    def on_progress(arm, mode, done, total, elapsed):
        # Cumulative across cells, not per-cell `done` alone -- `elapsed` accumulates
        # across the whole run while `done`/`total` reset every (arm,mode) cell, and a
        # per-cell count would trip the overrun trigger on entering the second cell
        # (OverrunTrigger._fire() self-terminates unconditionally). See
        # `cw_sim_launch2.cumulative_cell_progress` for details.
        from cw_sim_launch2 import cumulative_cell_progress
        _arm_fuse(fuse_minutes)
        cell = (arm, mode)
        if cell not in cell_order:
            cell_order.append(cell)
        cumulative_done = cumulative_cell_progress(cell_order.index(cell), done, total)
        cumulative_total = total * len(ARMS_TIKTOC) * len(MODES_TIKTOC)
        print(f"[launch2-tiktoc] arm={arm} mode={mode} {done}/{total} rows "
             f"(cumulative {cumulative_done}/{cumulative_total}, {elapsed:.0f}s)", flush=True)
        if overrun.check(cumulative_done, cumulative_total, time.time() - t0):
            raise SystemExit(1)

    records = cw5.run_batched_generation_pass(
        m, gen_attempts, manifest, scores_by_arm, device, len(gen_attempts), max_new_tokens,
        seed, arms=ARMS_TIKTOC, modes=MODES_TIKTOC, batch_size=batch_size,
        on_progress=on_progress, completed_row_idx=completed_row_idx)

    raw_path = out_dir / "generations_raw.jsonl"  # LOCAL-ONLY, never S3 -- see module docstring
    prior_raw = load_raw_generation_records(raw_path) if completed_row_idx else []
    save_raw_generation_records(prior_raw + records, raw_path)

    from atom.eval.cw_generation import load_numeric_generation_rows
    prior_numeric = load_numeric_generation_rows(completed_path) if completed_row_idx else []
    transform = CodecTransform.load(REPO / "transforms" / "problem_mean_L15.npz")
    new_numeric, codebleu_rows = score_records(m, records, transform, device)
    all_numeric = prior_numeric + new_numeric
    save_numeric_generation_rows(all_numeric, completed_path)

    codebleu_by_mode = {}
    for mode in MODES_TIKTOC:
        vals = [r["codebleu"] for r in codebleu_rows if r["generation_mode"] == mode]
        codebleu_by_mode[mode] = float(np.mean(vals)) if vals else None

    return {
        "n_rows_total": len(attempts), "n_rows_no_statement": n_no_statement,
        "n_rows_generation_eligible": len(gen_attempts),
        "n_generations": len(all_numeric), "codebleu_by_mode": codebleu_by_mode,
        "n_codebleu_scored": len(codebleu_rows),
        "reference": {"TIKTOC": 0.554, "OKT-TC": 0.522},
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiktoc-dir", default=None)
    ap.add_argument("--code-map", default="cache/cw_03_vectors_tiktoc/cw_tiktoc_code_state_hash_map.npz")
    ap.add_argument("--codealone-meta", default="cache/cw_03_vectors_tiktoc/codealone/meta.json")
    ap.add_argument("--codealone-states", default="cache/cw_03_vectors_tiktoc/codealone/states_mean.npy")
    ap.add_argument("--base-checkpoint", default="results/cw_sim_base_run/final.pt")
    ap.add_argument("--head-checkpoint", default="results/cw_sim_head_refresh/final.pt")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="results/cw_sim_launch2_tiktoc")
    ap.add_argument("--bucket", default="")
    ap.add_argument("--out-prefix", default="cw-sim-launch2-tiktoc")
    ap.add_argument("--estimated-minutes", type=float, default=60.0)
    ap.add_argument("--fuse-minutes", type=float, default=180.0)
    ap.add_argument("--self-terminate", action="store_true")
    ap.add_argument("--pilot-rows", type=int, default=None,
                    help="Truncate manifest['attempts'] to the first N rows before codec "
                         "attach/scoring -- a cheap end-to-end smoke pass rather than the "
                         "full 3,714-row run. Not used for reported numbers.")
    args = ap.parse_args()

    _arm_fuse(args.fuse_minutes)
    print(f"[launch2-tiktoc] boot fuse armed: shutdown in {args.fuse_minutes} min unless cancelled", flush=True)

    device = cw5.default_device()
    manifest = build_tiktoc_manifest(base_dir=args.tiktoc_dir)
    print(f"[launch2-tiktoc] raw manifest: {len(manifest['attempts'])} attempts, "
         f"{len(manifest['val_learners'])} learners, "
         f"{manifest['n_dropped_invalid_outcome']} dropped (invalid outcome)", flush=True)

    if args.pilot_rows:
        manifest["attempts"] = manifest["attempts"].head(args.pilot_rows).reset_index(drop=True)
        print(f"[launch2-tiktoc] PILOT MODE: truncated to {len(manifest['attempts'])} rows", flush=True)

    transform = CodecTransform.load(REPO / "transforms" / "problem_mean_L15.npz")
    manifest = attach_tiktoc_codec(manifest, pathlib.Path(args.code_map), pathlib.Path(args.codealone_meta),
                                   pathlib.Path(args.codealone_states), transform)
    print(f"[launch2-tiktoc] codec-attached manifest: {json.dumps(manifest['hash_stats'])}", flush=True)

    m, provenance = load_generation_model(pathlib.Path(args.base_checkpoint), pathlib.Path(args.head_checkpoint),
                                          device, allow_download=True)
    print(f"[launch2-tiktoc] model loaded: {json.dumps(provenance)}", flush=True)

    out_dir = REPO / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "provenance.json").write_text(json.dumps(provenance, indent=2))

    tv_labels = build_tiktoc_tv_lookup(base_dir=args.tiktoc_dir)
    auc_f1 = run_auc_f1_panel(m, manifest, tv_labels, device)
    print(f"[launch2-tiktoc] AUC/F1 panel: pooled_auc={auc_f1['auc_panel']['pooled_auc']}, "
         f"n_pairs={auc_f1['auc_panel']['n_pairs']}", flush=True)
    (out_dir / "auc_f1_panel.json").write_text(json.dumps(auc_f1, indent=2, allow_nan=True))
    if args.bucket:
        import boto3
        boto3.client("s3").upload_file(str(out_dir / "auc_f1_panel.json"), args.bucket,
                                       f"{args.out_prefix.rstrip('/')}/auc_f1_panel.json")

    codebleu = run_codebleu_blind_panel(m, manifest, device, args.max_new_tokens, args.seed,
                                        args.batch_size, out_dir, args.estimated_minutes, args.fuse_minutes)
    print(f"[launch2-tiktoc] CodeBLEU panel: {json.dumps(codebleu, indent=2)}", flush=True)
    (out_dir / "codebleu_panel.json").write_text(json.dumps(codebleu, indent=2, allow_nan=True))
    if args.bucket:
        import boto3
        boto3.client("s3").upload_file(str(out_dir / "codebleu_panel.json"), args.bucket,
                                       f"{args.out_prefix.rstrip('/')}/codebleu_panel.json")
        boto3.client("s3").upload_file(str(out_dir / "generation_numeric_rows.npz"), args.bucket,
                                       f"{args.out_prefix.rstrip('/')}/generation_numeric_rows.npz")

    result = {"provenance": provenance, "auc_f1": auc_f1, "codebleu": codebleu}
    (out_dir / "result.json").write_text(json.dumps(result, indent=2, allow_nan=True))

    if args.self_terminate:
        subprocess.run(["sudo", "shutdown", "-h", "now"], check=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
