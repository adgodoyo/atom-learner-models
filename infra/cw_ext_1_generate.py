#!/usr/bin/env python3
"""Generates the CodeWorkout generative simulator's code for the first-submission rows on the 30 not-yet-generated CodeWorkout problems and scores it with CodeBLEU.

Inputs: the CodeWorkout manifest, the row set from `cw_ext_1_row_selection.select_ext1_rows`
(~2,806 rows), the base checkpoint and refreshed outcome head (`ratchet-cwsim` and
`ratchet-cwsim/refreshed-head`; both SHA-256-pinned, and the script refuses to run if either
hash differs), and the frozen codec transform `transforms/problem_mean_L15.npz`.
One cell only: the personalised arm with argmax decoding. Generation reuses
`cw_sim_core.run_batched_generation_pass` unchanged, called once per chunk of
`--problems-per-chunk` problems, with raw and numeric stores appended after every chunk
and chunk-level resume through `progress.json`. CodeBLEU runs in a separate Python 3.11
interpreter (`--codebleu-python`, see `cw_ext_1_codebleu_score.py`); a preflight call
checks it before any generation. `problem_id`/`learner` are joined back onto each scored
row by chunk-local `row_idx` for the per-problem spread.
Outputs (in `--out-dir`): generations_raw.jsonl (local only; never uploaded),
generation_numeric_rows.npz, codebleu_rows.json, progress.json, provenance.json and
result.json (30-problem CodeBLEU and per-problem summaries). Only the scalar files are
uploaded, when `--bucket` is given.
Run:  python infra/cw_ext_1_generate.py --pilot-rows 60 --out-dir results/cw_ext_1_pilot
      python infra/cw_ext_1_generate.py --out-dir results/cw_ext_1
Serves: thesis Table 11 (the all-50-problem CodeBLEU, comparable to KCGen-KT's published 0.580).
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
from cw_ext_1_row_selection import OTHER_30_PROBLEM_IDS, _ManifestArgs, select_ext1_rows  # noqa: E402
from cw_sim_launch2 import (  # noqa: E402
    cumulative_cell_progress, load_raw_generation_records, save_raw_generation_records,
    score_records,
)
from cw_sim_launch2_model import load_generation_model  # noqa: E402

from atom.codec.transform import CodecTransform  # noqa: E402
from atom.eval.cw_generation import (  # noqa: E402
    completed_row_indices, load_numeric_generation_rows, save_numeric_generation_rows,
)

ARMS_EXT1 = ("personalised",)
MODES_EXT1 = ("argmax",)

EXPECTED_BASE_SHA256 = "3143d05122957e855b2067f9bd9c61cd7aabf414b380ffc5fc2e46a0ccf7c3e9"
EXPECTED_HEAD_SHA256 = "5337abbbad40e0d1474ddce1e3bdc69cd1159a94e29722cedef70d8c56d7a56d"


def _arm_fuse(minutes: float) -> None:
    subprocess.run(["sudo", "shutdown", "-c"], check=False)
    subprocess.run(["sudo", "shutdown", "-h", f"+{int(minutes)}"], check=False)


def load_progress(path: pathlib.Path) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    return {"completed_problem_ids": [], "n_rows_done": 0}


def save_progress(path: pathlib.Path, progress: dict) -> None:
    path.write_text(json.dumps(progress, indent=2))


def upload_scalars(out_dir: pathlib.Path, bucket: str, out_prefix: str) -> None:
    """Uploads ONLY scalar/derived artifacts -- never `generations_raw.jsonl`."""
    if not bucket:
        return
    import boto3
    s3 = boto3.client("s3")
    for name in ("generation_numeric_rows.npz", "codebleu_rows.json", "progress.json", "result.json"):
        p = out_dir / name
        if p.exists():
            s3.upload_file(str(p), bucket, f"{out_prefix.rstrip('/')}/{name}",
                           ExtraArgs={"ServerSideEncryption": "AES256"} if name.endswith(".npz") else {})


def run_chunk(m: dict, chunk_attempts: pd.DataFrame, manifest: dict, device: str,
             max_new_tokens: int, seed: int, batch_size: int, out_dir: pathlib.Path,
             overrun, t0: float, fuse_minutes: float, rows_done_before_this_chunk: int,
             total_rows_registered: int, codebleu_python: str) -> tuple[int, int]:
    """One problem-group chunk: score -> generate -> save raw (append) -> score_records
    -> save numeric (append) -> save codebleu rows w/ problem_id (append). Returns
    (n_rows_generated_this_chunk, n_codebleu_scored_this_chunk)."""
    episode_ids = set(chunk_attempts["episode_id"])
    episodes_batch = manifest["episodes"][manifest["episodes"]["episode_id"].isin(episode_ids)]

    scores = cw5.score_arm_for_batch(m, chunk_attempts, episodes_batch, manifest, cw5.CW5_DIMS, device,
                                     k=8, micro_batch_size=8, arm="personalised")
    scores_by_arm = {"personalised": scores}
    n_dropped_budget = len(chunk_attempts) - len(scores.get("kept_row_idx", []))
    if n_dropped_budget:
        print(f"[cw-ext-1-gen] chunk: {n_dropped_budget} row(s) dropped by completion-token budget", flush=True)

    cell_order: list[tuple[str, str]] = []

    def on_progress(arm, mode, done, total, elapsed):
        _arm_fuse(fuse_minutes)
        cell = (arm, mode)
        if cell not in cell_order:
            cell_order.append(cell)
        cell_done = cumulative_cell_progress(cell_order.index(cell), done, total)
        global_done = rows_done_before_this_chunk + cell_done
        print(f"[cw-ext-1-gen] arm={arm} mode={mode} {done}/{total} in-chunk "
             f"(global {global_done}/{total_rows_registered}, {elapsed:.0f}s)", flush=True)
        if overrun.check(global_done, total_rows_registered, time.time() - t0):
            raise SystemExit(1)

    records = cw5.run_batched_generation_pass(
        m, chunk_attempts, manifest, scores_by_arm, device, len(chunk_attempts), max_new_tokens,
        seed, arms=ARMS_EXT1, modes=MODES_EXT1, batch_size=batch_size, on_progress=on_progress)

    raw_path = out_dir / "generations_raw.jsonl"  # local only, never uploaded
    save_raw_generation_records(load_raw_generation_records(raw_path) + records, raw_path)

    transform = CodecTransform.load(REPO / "transforms" / "problem_mean_L15.npz")
    # `score_records` is reused unchanged for the codec-similarity half (numeric_rows).
    # Its own `codebleu_rows` output is empty here: this process runs under Python 3.12
    # (needed for torch/CUDA), where `atom.eval.codebleu.codebleu_available()` reports
    # False because of the vendored evaluator's Python 3.11 / tree_sitter API dependency
    # (see `cw_ext_1_codebleu_score.py`). CodeBLEU is computed by the subprocess below.
    new_numeric, _empty_codebleu_rows = score_records(m, records, transform, device)

    numeric_path = out_dir / "generation_numeric_rows.npz"
    prior_numeric = load_numeric_generation_rows(numeric_path) if numeric_path.exists() else []
    save_numeric_generation_rows(prior_numeric + new_numeric, numeric_path)

    codebleu_joined = score_codebleu_via_subprocess(records, codebleu_python)
    codebleu_path = out_dir / "codebleu_rows.json"
    prior_codebleu = json.loads(codebleu_path.read_text()) if codebleu_path.exists() else []
    codebleu_path.write_text(json.dumps(prior_codebleu + codebleu_joined, indent=2))

    return len(records), len(codebleu_joined)


def score_codebleu_via_subprocess(records: list[dict], codebleu_python: str) -> list[dict]:
    """Runs `cw_ext_1_codebleu_score.py` under a SEPARATE Python 3.11 interpreter
    (`codebleu_python`) -- see that script's docstring for why. Passes
    reference/hypothesis via stdin (never argv, which would be visible in `ps`) and reads
    results via stdout; only the returned numeric fields, never the code text, are logged
    or persisted by this process. `problem_id`/`learner`/`code_state_id` are joined back
    in from `records` by `row_idx`, because `score_records`'s own `codebleu_rows` lack
    `problem_id`."""
    payload = [{"id": int(r["row_idx"]), "reference": r["real_completion"], "hypothesis": r["generated_code"]}
              for r in records]
    if not payload:
        return []
    proc = subprocess.run(
        [codebleu_python, str(REPO / "infra" / "cw_ext_1_codebleu_score.py")],
        input=json.dumps(payload), capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        print(f"[cw-ext-1-gen] codebleu subprocess failed (code {proc.returncode}): {proc.stderr[-2000:]}", flush=True)
        return []
    results = json.loads(proc.stdout)
    records_by_row_idx = {int(r["row_idx"]): r for r in records}
    joined = []
    for cb in results:
        rec = records_by_row_idx[int(cb["id"])]
        joined.append({
            "learner": rec["learner"], "problem_id": int(rec["problem_id"]),
            "code_state_id": rec["code_state_id"], "arm": rec["arm"], "generation_mode": rec["generation_mode"],
            "codebleu": cb["codebleu"], "ngram_match": cb["ngram_match"],
            "weighted_ngram_match": cb["weighted_ngram_match"], "syntax_match": cb["syntax_match"],
            "dataflow_match": cb["dataflow_match"],
        })
    return joined


def summarise(out_dir: pathlib.Path) -> dict:
    codebleu_path = out_dir / "codebleu_rows.json"
    rows = json.loads(codebleu_path.read_text()) if codebleu_path.exists() else []
    if not rows:
        return {"n_scored": 0, "all_50_codebleu": None}
    vals = np.array([r["codebleu"] for r in rows], dtype=np.float64)
    by_problem: dict[int, list[float]] = {}
    for r in rows:
        by_problem.setdefault(int(r["problem_id"]), []).append(r["codebleu"])
    per_problem = {
        str(pid): {"n": len(v), "mean": float(np.mean(v)), "min": float(np.min(v)),
                  "median": float(np.median(v)), "max": float(np.max(v))}
        for pid, v in sorted(by_problem.items())
    }
    return {
        "n_scored": len(rows), "n_problems_scored": len(by_problem),
        "thirty_problem_codebleu": float(np.mean(vals)),
        "per_problem": per_problem,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-checkpoint", default="results/cw_sim_base_run/final.pt")
    ap.add_argument("--head-checkpoint", default="results/cw_sim_head_refresh/final.pt")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--problems-per-chunk", type=int, default=5)
    ap.add_argument("--codebleu-python", default=str(pathlib.Path.home() / "codebleu-py311" / "bin" / "python3"),
                    help="Path to a Python 3.11 interpreter with tree_sitter==0.20.4/numpy/pandas "
                         "installed, used ONLY for CodeBLEU scoring (see cw_ext_1_codebleu_score.py's "
                         "docstring -- a Python 3.12 environment cannot run the vendored CodeBLEU "
                         "evaluator; the generation pass itself stays on 3.12/torch).")
    ap.add_argument("--out-dir", default="results/cw_ext_1")
    ap.add_argument("--bucket", default="")
    ap.add_argument("--out-prefix", default="cw-ext-1")
    ap.add_argument("--estimated-minutes", type=float, default=180.0)
    ap.add_argument("--fuse-minutes", type=float, default=240.0)
    ap.add_argument("--pilot-rows", type=int, default=None,
                    help="Truncate the selected row set to the first N rows (sorted by "
                         "problem, then learner) -- a real, cheap end-to-end pilot to "
                         "measure the batched rate before the full launch. Not used for "
                         "the reported numbers.")
    args = ap.parse_args()

    _arm_fuse(args.fuse_minutes)
    print(f"[cw-ext-1-gen] boot fuse armed: shutdown in {args.fuse_minutes} min unless cancelled", flush=True)

    preflight = score_codebleu_via_subprocess(
        [{"row_idx": -1, "learner": "preflight", "problem_id": -1, "code_state_id": "preflight",
         "arm": "preflight", "generation_mode": "preflight",
         "real_completion": "public int f(){return 1;}", "generated_code": "public int f(){return 2;}"}],
        args.codebleu_python)
    assert preflight and preflight[0]["codebleu"] is not None, (
        f"CodeBLEU preflight FAILED under --codebleu-python={args.codebleu_python} -- refusing to spend "
        f"GPU time on a run that cannot score its own headline metric. Fix the Python 3.11 venv first "
        f"(see cw_ext_1_codebleu_score.py's docstring: uv venv --python 3.11, then "
        f"uv pip install 'tree_sitter==0.20.4' numpy pandas).")
    print(f"[cw-ext-1-gen] CodeBLEU preflight PASS via {args.codebleu_python}: {preflight[0]}", flush=True)

    device = cw5.default_device()
    print(f"[cw-ext-1-gen] device={device}", flush=True)

    print("[cw-ext-1-gen] loading CodeWorkout manifest...", flush=True)
    manifest = cw5.load_cw5_manifest(_ManifestArgs())
    ext1_rows = select_ext1_rows(manifest)
    ext1_rows = ext1_rows.sort_values(["problem_id", "learner"], kind="mergesort").reset_index(drop=True)
    if args.pilot_rows:
        ext1_rows = ext1_rows.head(args.pilot_rows).reset_index(drop=True)
        print(f"[cw-ext-1-gen] PILOT MODE: truncated to {len(ext1_rows)} rows", flush=True)
    print(f"[cw-ext-1-gen] row set: {len(ext1_rows)} rows across "
         f"{ext1_rows['problem_id'].nunique()} problems", flush=True)

    m, provenance = load_generation_model(REPO / args.base_checkpoint, REPO / args.head_checkpoint,
                                          device, allow_download=True)
    assert provenance["base_checkpoint_sha256"] == EXPECTED_BASE_SHA256, (
        f"base checkpoint sha mismatch: got {provenance['base_checkpoint_sha256']}, "
        f"expected {EXPECTED_BASE_SHA256} -- refusing to generate under an unpinned/swapped substrate")
    assert provenance["head_checkpoint_sha256"] == EXPECTED_HEAD_SHA256, (
        f"head checkpoint sha mismatch: got {provenance['head_checkpoint_sha256']}, "
        f"expected {EXPECTED_HEAD_SHA256} -- refusing to generate under an unpinned/swapped substrate")
    print(f"[cw-ext-1-gen] model loaded, SHA-pinned and verified: {json.dumps(provenance)}", flush=True)

    out_dir = REPO / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "provenance.json").write_text(json.dumps(provenance, indent=2))

    progress_path = out_dir / "progress.json"
    progress = load_progress(progress_path)
    completed_pids = set(progress["completed_problem_ids"])

    problem_ids = sorted(ext1_rows["problem_id"].unique().tolist())
    chunks = [problem_ids[i:i + args.problems_per_chunk] for i in range(0, len(problem_ids), args.problems_per_chunk)]
    total_rows_registered = len(ext1_rows)

    overrun = cw5.OverrunTrigger(args.estimated_minutes, out_dir, args.bucket, args.out_prefix, "cw-ext-1")
    t0 = time.time()
    rows_done = progress["n_rows_done"]

    for ci, chunk_pids in enumerate(chunks):
        remaining_pids = [p for p in chunk_pids if p not in completed_pids]
        if not remaining_pids:
            print(f"[cw-ext-1-gen] chunk {ci + 1}/{len(chunks)} ({chunk_pids}) already complete, skipping", flush=True)
            continue
        chunk_attempts = ext1_rows[ext1_rows["problem_id"].isin(remaining_pids)].reset_index(drop=True)
        print(f"[cw-ext-1-gen] chunk {ci + 1}/{len(chunks)}: problems {remaining_pids}, "
             f"{len(chunk_attempts)} rows", flush=True)

        n_gen, n_scored = run_chunk(m, chunk_attempts, manifest, device, args.max_new_tokens, args.seed,
                                    args.batch_size, out_dir, overrun, t0, args.fuse_minutes,
                                    rows_done, total_rows_registered, args.codebleu_python)
        rows_done += len(chunk_attempts)
        completed_pids.update(remaining_pids)
        progress = {"completed_problem_ids": sorted(completed_pids), "n_rows_done": rows_done,
                   "n_chunks_done": ci + 1, "n_chunks_total": len(chunks)}
        save_progress(progress_path, progress)
        upload_scalars(out_dir, args.bucket, args.out_prefix)
        print(f"[cw-ext-1-gen] chunk {ci + 1}/{len(chunks)} done: {n_gen} generated, {n_scored} codebleu-scored "
             f"(cumulative rows_done={rows_done}/{total_rows_registered}), {time.time() - t0:.0f}s elapsed", flush=True)

    summary = summarise(out_dir)
    result = {
        "provenance": provenance, "n_rows_registered": total_rows_registered,
        "n_problems": len(problem_ids), "elapsed_s": time.time() - t0, **summary,
        "raw_text_uploaded": False,
    }
    (out_dir / "result.json").write_text(json.dumps(result, indent=2, allow_nan=True))
    upload_scalars(out_dir, args.bucket, args.out_prefix)
    print(f"[cw-ext-1-gen] DONE: {json.dumps(result, indent=2)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
