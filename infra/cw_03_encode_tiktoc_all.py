#!/usr/bin/env python
"""Encodes TIKTOC's full attempt-level file (code and episode-local diffs) with the frozen codec.

Covers `dataset_granular_all`, so the multi-attempt episodes built by
`build_tiktoc_manifest_all` have both a code cache (a superset of the first-attempt-only
cache from `cw_03_encode_tiktoc.py`, since students attempt each problem more than once)
and a diff cache. Uses the same machinery as `cw_03_encode.py` (`build_code_diffs`,
`SingleLayerVectorCache`, `extract_many_l15_into_cache`, `text_hash`) and the same frozen
layer 15. The corpus is small (13,083 rows against CodeWorkout's ~195k), so there is no
S3 resume and no pilot mode.
Inputs: the TIKTOC data directory (`--tiktoc-dir`).
Outputs (under `--out-dir`): `codealone/` and `diff/` vector caches,
cw_tiktoc_all_code_state_hash_map.npz, cw_tiktoc_all_diff_hash_map.npz and summary.json.
Run:  python infra/cw_03_encode_tiktoc_all.py --tiktoc-dir <dir> \
          --out-dir cache/cw_03_vectors_tiktoc_all --fuse-minutes 60
Serves: thesis Ch 5, §5.2 (the codec applied to the TIKTOC transfer corpus).
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

from atom.codec.backbone import DEFAULT_BACKBONE, load_backbone  # noqa: E402
from atom.codec.cw_batch_extract import (  # noqa: E402
    SingleLayerVectorCache,
    extract_many_l15_into_cache,
)
from atom.codec.extract import text_hash  # noqa: E402
from atom.data.cw_diffs import build_code_diffs  # noqa: E402
from atom.data.cw_sessions import sessionize  # noqa: E402
from atom.data.cw_tiktoc_transfer import TAU_SESSION_MINUTES, build_tiktoc_attempts_all  # noqa: E402

LAYER = 15  # frozen production layer -- matches cw_03_encode.py/cw_03_encode_tiktoc.py, never re-derived here


def rearm_boot_fuse(minutes: float) -> None:
    subprocess.run(["sudo", "shutdown", "-c"], check=False)
    subprocess.run(["sudo", "shutdown", "-h", f"+{int(minutes)}"], check=False)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiktoc-dir", default=None)
    ap.add_argument("--fuse-minutes", type=float, default=60.0)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--boundary-cap", type=int, default=32)
    ap.add_argument("--max-length", type=int, default=4096)
    ap.add_argument("--out-dir", default="cache/cw_03_vectors_tiktoc_all")
    args = ap.parse_args()

    rearm_boot_fuse(args.fuse_minutes)
    REPO = pathlib.Path(__file__).resolve().parents[1]

    t0 = time.time()
    attempts = build_tiktoc_attempts_all(base_dir=args.tiktoc_dir)
    attempts = sessionize(attempts, TAU_SESSION_MINUTES)
    diffs = build_code_diffs(attempts)
    build_s = time.time() - t0

    valid_code = attempts["code"].notna() & (attempts["code"].astype(str).str.strip() != "")
    n_missing_code = int((~valid_code).sum())
    code_hashes: dict[str, str] = {}
    for code in attempts.loc[valid_code, "code"]:
        code_hashes.setdefault(text_hash(code), code)

    n_empty_diff = int((diffs["diff_text"] == "").sum())
    diff_hashes: dict[str, str] = {}
    for d in diffs.loc[diffs["diff_text"] != "", "diff_text"]:
        diff_hashes.setdefault(text_hash(d), d)

    print(f"[cw-03-tiktoc-all] {len(attempts)} attempts, {n_missing_code} missing code, "
         f"{len(code_hashes)} unique code texts, {len(diffs)} diffed attempts "
         f"({n_empty_diff} empty), {len(diff_hashes)} unique diff texts to encode "
         f"({build_s:.1f}s)", flush=True)

    print("[cw-03-tiktoc-all] loading backbone", flush=True)
    bundle = load_backbone(DEFAULT_BACKBONE, allow_download=True)

    scopes = {"codealone": code_hashes, "diff": diff_hashes}
    caches: dict[str, SingleLayerVectorCache] = {}
    for scope, hash_to_text in scopes.items():
        out_dir = REPO / args.out_dir / scope
        cache = SingleLayerVectorCache(out_dir, hidden_dim=bundle["hidden_dim"],
                                       capacity=max(len(hash_to_text), 1),
                                       model=bundle["hf_id"], scope=scope)
        caches[scope] = cache
        t_scope0 = time.time()

        def _progress(done: int, total: int, _scope=scope, _t0=t_scope0) -> None:
            elapsed = time.time() - _t0
            rate = done / elapsed if elapsed > 0 else 0.0
            print(f"[cw-03-tiktoc-all] encoding {_scope} {done}/{total} ({rate:.2f}/s)", flush=True)

        extract_many_l15_into_cache(
            bundle, hash_to_text, layer=LAYER, batch_size=args.batch_size, cache=cache,
            boundary_cap=args.boundary_cap, max_length=args.max_length, progress_cb=_progress,
        )
        scope_s = time.time() - t_scope0
        print(f"[cw-03-tiktoc-all] {scope} done in {scope_s:.1f}s "
             f"({cache.n_cached}/{len(hash_to_text)} cached)", flush=True)

    code_map_path = REPO / args.out_dir / "cw_tiktoc_all_code_state_hash_map.npz"
    valid_attempts = attempts.loc[valid_code]
    np.savez(code_map_path,
            code_state_id=valid_attempts["code_state_id"].to_numpy(),
            code_hash=np.array([text_hash(c) for c in valid_attempts["code"]]),
            layer=LAYER, scope="codealone", model=bundle["hf_id"])

    diffed = diffs[diffs["diff_text"] != ""]
    diff_map_path = REPO / args.out_dir / "cw_tiktoc_all_diff_hash_map.npz"
    np.savez(diff_map_path,
            code_state_id=diffed["code_state_id"].to_numpy(),
            prev_code_state_id=diffed["prev_code_state_id"].to_numpy(),
            episode_id=diffed["episode_id"].to_numpy(),
            diff_hash=np.array([text_hash(d) for d in diffed["diff_text"]]),
            layer=LAYER, scope="diff", model=bundle["hf_id"])

    n_code_finite = int(np.isfinite(caches["codealone"].states("mean")).all())
    n_diff_finite = int(np.isfinite(caches["diff"].states("mean")).all()) if caches["diff"].n_cached else 1
    passed = (caches["codealone"].n_cached == len(code_hashes)
             and caches["diff"].n_cached == len(diff_hashes)
             and bool(n_code_finite) and bool(n_diff_finite))

    summary = {
        "n_attempts": len(attempts), "n_missing_code": n_missing_code,
        "n_diffed_attempts": len(diffs), "n_empty_diff": n_empty_diff,
        "n_code_unique": len(code_hashes), "n_diff_unique": len(diff_hashes),
        "n_code_encoded": caches["codealone"].n_cached, "n_diff_encoded": caches["diff"].n_cached,
        "all_finite": bool(n_code_finite) and bool(n_diff_finite),
        "build_s": round(build_s, 1), "layer": LAYER, "batch_size": args.batch_size,
        "gate_pass": passed,
    }
    summary_path = REPO / args.out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    print(f"\ncw_03_encode_tiktoc_all: {'PASS' if passed else 'FAIL'} -- "
         f"code {caches['codealone'].n_cached}/{len(code_hashes)}, "
         f"diff {caches['diff'].n_cached}/{len(diff_hashes)}, "
         f"code_map -> {code_map_path}, diff_map -> {diff_map_path}", flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
