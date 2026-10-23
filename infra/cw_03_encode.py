#!/usr/bin/env python
"""Encodes every CodeWorkout code submission and episode-local code diff with the frozen codec.

Configuration: code-alone scope, layer 15, readouts mean + boundary_mean + last, batched
and length-bucketed on a GPU via `atom.codec.cw_batch_extract` (batching is checked to be
cosine-equivalent to the single-item path in tests/test_cw_batch_extract.py).
Inputs: the two raw CodeWorkout zips (S19_All_Release_2_10_22.zip, F19_Release_All_05_23_22.zip)
in `--raw-dir`, copied to the GPU machine over SSH. Raw code and diff text stay on that
machine's local disk; only text-free derived vectors (hash-keyed) and hash-map lookups are
uploaded to the S3 bucket given with `--bucket`.
Outputs: the hash-keyed vector caches for the `codealone` and `diff` scopes, the lookups
cw_code_state_hash_map_<mode>.npz and cw_diff_hash_map_<mode>.npz, and summary_<mode>.json.
Caches are synced to the bucket periodically and pulled back on start-up, so a relaunch
resumes rather than restarting; the machine's shutdown timer is re-armed to `--fuse-minutes`.
Run:  python infra/cw_03_encode.py --bucket <bucket> --raw-dir <dir> [--pilot-n 300]
(`--pilot-n` encodes a length-stratified sample for throughput calibration only.)
Serves: thesis Ch 5, §5.2 (the codec applied to CodeWorkout code).
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from atom.codec.backbone import DEFAULT_BACKBONE, load_backbone  # noqa: E402
from atom.codec.cw_batch_extract import (  # noqa: E402
    SingleLayerVectorCache,
    extract_many_l15_into_cache,
)
from atom.codec.extract import text_hash  # noqa: E402
from atom.data.cw_diffs import build_code_diffs  # noqa: E402
from atom.data.cw_events import build_attempts, load_raw_tables  # noqa: E402
from atom.data.cw_sessions import sessionize  # noqa: E402
from atom.utils.logging import RunLogger  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[1]
LAYER = 15  # the codec's frozen production layer for code -- not re-derived here
TAU_SESSION_MINUTES = 60.0  # the sessioniser's frozen, S19-train-fit value -- must match, never refit


def rearm_boot_fuse(minutes: float) -> None:
    """Re-arms the machine's shutdown timer: the boot script's unconditional 300-min
    shutdown is a backstop meant to be re-armed once a script knows its own wall-clock. `check=False` throughout -- a failed re-arm should not crash the
    encode, and the 300-min backstop still applies if this call itself fails."""
    subprocess.run(["sudo", "shutdown", "-c"], check=False)
    subprocess.run(["sudo", "shutdown", "-h", f"+{int(minutes)}"], check=False)


def s3_prefix(bucket: str, out_prefix: str, scope: str) -> str:
    return f"s3://{bucket}/{out_prefix}/cache/{scope}"


def try_download_cache_from_s3(bucket: str, out_prefix: str, scope: str, local_dir: pathlib.Path) -> bool:
    """Best-effort pull of a previously-synced vector cache down from S3 before this run
    starts encoding -- so a relaunch after total instance loss resumes from where the
    last one left off (extract_many_l15_into_cache's hash-based skip does the rest)."""
    src = s3_prefix(bucket, out_prefix, scope)
    r = subprocess.run(["aws", "s3", "sync", src, str(local_dir)], check=False,
                       capture_output=True, text=True)
    return r.returncode == 0 and local_dir.exists() and any(local_dir.iterdir())


def sync_cache_to_s3(bucket: str, out_prefix: str, scope: str, local_dir: pathlib.Path) -> None:
    dst = s3_prefix(bucket, out_prefix, scope)
    subprocess.run(["aws", "s3", "sync", str(local_dir), dst, "--only-show-errors"], check=False)


def build_full_corpus(raw_dir: pathlib.Path) -> pd.DataFrame:
    """S19+F19 attempts (with code text -- raw code stays on this machine's local disk
    only, as described above), sessionised at the frozen tau."""
    frames = []
    for release in ["S19", "F19"]:
        main, code_states = load_raw_tables(release, base_dir=raw_dir, include_code=True)
        a = build_attempts(main, code_states=code_states)
        a["release"] = release
        frames.append(a)
    attempts = pd.concat(frames, ignore_index=True)
    return sessionize(attempts, tau_session_minutes=TAU_SESSION_MINUTES)


def stratified_sample(hash_to_text: dict[str, str], n: int, seed: int = 0) -> dict[str, str]:
    """Length-stratified sample for PILOT throughput calibration -- covers the length
    distribution's tail (long items dominate wall-clock), not just a random head."""
    items = sorted(hash_to_text.items(), key=lambda kv: len(kv[1]))
    if len(items) <= n:
        return dict(items)
    idx = np.linspace(0, len(items) - 1, n).astype(int)
    idx = sorted(set(idx))
    return {items[i][0]: items[i][1] for i in idx}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bucket", required=True)
    ap.add_argument("--raw-dir", required=True, help="local dir holding the two rsynced "
                    "CodeWorkout zips (S19_All_Release_2_10_22.zip, F19_Release_All_05_23_22.zip)")
    ap.add_argument("--fuse-minutes", type=float, default=240.0)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--boundary-cap", type=int, default=32)
    ap.add_argument("--max-length", type=int, default=4096)
    ap.add_argument("--sync-every", type=int, default=2000)
    ap.add_argument("--out-prefix", default="cw-03-encode")
    ap.add_argument("--pilot-n", type=int, default=None,
                    help="if set: PILOT mode, encode only this many length-stratified "
                        "items per scope (code, diff) for throughput calibration -- "
                        "writes to a separate cache prefix, never the production one")
    ap.add_argument("--cache-dir", default="/tmp/cw_03_cache")
    args = ap.parse_args()

    rearm_boot_fuse(args.fuse_minutes)
    mode = "pilot" if args.pilot_n else "full"
    out_prefix = f"{args.out_prefix}-{mode}"

    logger = RunLogger(config={"task": "cw_03_encode", "mode": mode, "layer": LAYER,
                               "batch_size": args.batch_size, "pilot_n": args.pilot_n},
                       data_version="CW-S19-F19-full")

    logger.heartbeat(step=0, force=True, stage="building attempts + episodes + diffs")
    t0 = time.time()
    sess = build_full_corpus(pathlib.Path(args.raw_dir))
    diffs = build_code_diffs(sess)
    build_s = time.time() - t0

    valid_code = sess["code"].notna() & (sess["code"].astype(str).str.strip() != "")
    n_missing_code = int((~valid_code).sum())
    code_hashes: dict[str, str] = {}
    for code in sess.loc[valid_code, "code"]:
        code_hashes.setdefault(text_hash(code), code)
    diff_hashes: dict[str, str] = {}
    n_empty_diff = int((diffs["diff_text"] == "").sum())
    for d in diffs.loc[diffs["diff_text"] != "", "diff_text"]:
        diff_hashes.setdefault(text_hash(d), d)

    n_code_full, n_diff_full = len(code_hashes), len(diff_hashes)
    if args.pilot_n:
        code_hashes = stratified_sample(code_hashes, args.pilot_n)
        diff_hashes = stratified_sample(diff_hashes, max(1, args.pilot_n // 3))

    logger.heartbeat(step=0, force=True, stage="corpus built", n_attempts=len(sess),
                     n_missing_code=n_missing_code, n_diffed_attempts=len(diffs),
                     n_empty_diff=n_empty_diff, n_code_unique_full=n_code_full,
                     n_diff_unique_full=n_diff_full, n_code_this_run=len(code_hashes),
                     n_diff_this_run=len(diff_hashes), build_s=round(build_s, 1))

    logger.heartbeat(step=0, force=True, stage="loading backbone")
    # allow_download=True: google/gemma-4-E4B is apache-2.0 and not gated on HF, and a
    # fresh GPU machine starts with an empty HF cache.
    bundle = load_backbone(DEFAULT_BACKBONE, allow_download=True)

    cache_root = pathlib.Path(args.cache_dir) / mode
    scopes = {"codealone": code_hashes, "diff": diff_hashes}
    caches: dict[str, SingleLayerVectorCache] = {}

    for scope, hash_to_text in scopes.items():
        local_dir = cache_root / scope
        resumed = try_download_cache_from_s3(args.bucket, out_prefix, scope, local_dir)
        cache = SingleLayerVectorCache(local_dir, hidden_dim=bundle["hidden_dim"],
                                       capacity=max(len(hash_to_text), 1),
                                       model=bundle["hf_id"], scope=scope)
        caches[scope] = cache
        logger.heartbeat(step=0, force=True, stage=f"{scope} cache ready",
                         resumed_from_s3=resumed, n_already_cached=cache.n_cached,
                         n_todo=len(hash_to_text) - sum(1 for h in hash_to_text if cache.has(h)))

        t_scope0 = time.time()

        def _progress(done: int, total: int, _scope=scope, _t0=t_scope0) -> None:
            elapsed = time.time() - _t0
            rate = done / elapsed if elapsed > 0 else 0.0
            logger.heartbeat(step=done, total=total, stage=f"encoding {_scope}",
                             items_per_s=round(rate, 3))

        def _sync(_scope=scope, _local_dir=local_dir) -> None:
            sync_cache_to_s3(args.bucket, out_prefix, _scope, _local_dir)

        extract_many_l15_into_cache(
            bundle, hash_to_text, layer=LAYER, batch_size=args.batch_size, cache=cache,
            boundary_cap=args.boundary_cap, max_length=args.max_length,
            sync_every=args.sync_every, sync_cb=_sync, progress_cb=_progress,
        )
        scope_s = time.time() - t_scope0
        logger.heartbeat(step=0, force=True, stage=f"{scope} done", wall_s=round(scope_s, 1),
                         items_per_s=round(cache.n_cached / scope_s, 3) if scope_s > 0 else None)

    # -- lookups: CodeStateID/attempt -> hash; small and text-free
    art_dir = pathlib.Path("/tmp/cw_03_artifacts")
    art_dir.mkdir(exist_ok=True)

    code_map_path = art_dir / f"cw_code_state_hash_map_{mode}.npz"
    valid_sess = sess.loc[valid_code]
    np.savez(code_map_path,
            code_state_id=valid_sess["code_state_id"].to_numpy(),
            code_hash=np.array([text_hash(c) for c in valid_sess["code"]]),
            episode_id=valid_sess["episode_id"].to_numpy(),
            layer=LAYER, scope="codealone", model=bundle["hf_id"])

    diffed = diffs[diffs["diff_text"] != ""]
    diff_map_path = art_dir / f"cw_diff_hash_map_{mode}.npz"
    np.savez(diff_map_path,
            code_state_id=diffed["code_state_id"].to_numpy(),
            prev_code_state_id=diffed["prev_code_state_id"].to_numpy(),
            episode_id=diffed["episode_id"].to_numpy(),
            diff_hash=np.array([text_hash(d) for d in diffed["diff_text"]]),
            layer=LAYER, scope="diff", model=bundle["hf_id"])

    subprocess.run(["aws", "s3", "cp", str(code_map_path),
                    f"s3://{args.bucket}/{out_prefix}/artifacts/{code_map_path.name}"], check=False)
    subprocess.run(["aws", "s3", "cp", str(diff_map_path),
                    f"s3://{args.bucket}/{out_prefix}/artifacts/{diff_map_path.name}"], check=False)

    n_code_finite = int(np.isfinite(caches["codealone"].states("mean")).all())
    n_diff_finite = int(np.isfinite(caches["diff"].states("mean")).all()) if caches["diff"].n_cached else 1
    passed = (
        caches["codealone"].n_cached == len(code_hashes)
        and caches["diff"].n_cached == len(diff_hashes)
        and bool(n_code_finite) and bool(n_diff_finite)
    )

    summary = {
        "mode": mode, "n_attempts": len(sess), "n_diffed_attempts": len(diffs),
        "n_empty_diff": n_empty_diff,
        "n_code_unique_full_corpus": n_code_full, "n_diff_unique_full_corpus": n_diff_full,
        "n_code_this_run": len(code_hashes), "n_diff_this_run": len(diff_hashes),
        "n_code_encoded": caches["codealone"].n_cached, "n_diff_encoded": caches["diff"].n_cached,
        "all_finite": bool(n_code_finite) and bool(n_diff_finite),
        "gate_pass": passed, "layer": LAYER, "batch_size": args.batch_size,
    }
    summary_path = art_dir / f"summary_{mode}.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    subprocess.run(["aws", "s3", "cp", str(summary_path),
                    f"s3://{args.bucket}/{out_prefix}/results/summary.json"], check=False)

    logger.log_metrics(**summary)
    logger.summary(status="ok" if passed else "gate-fail", **summary)
    print(f"\ncw_03_encode ({mode}): {'PASS' if passed else 'FAIL'} -- "
         f"code {caches['codealone'].n_cached}/{len(code_hashes)}, "
         f"diff {caches['diff'].n_cached}/{len(diff_hashes)}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
