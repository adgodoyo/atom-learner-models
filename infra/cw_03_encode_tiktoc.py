#!/usr/bin/env python
"""Encodes the code text of TIKTOC's 17-problem benchmark submissions with the frozen codec.

Produces the code-alone layer-15 vectors that `atom.data.cw_manifest.build_codec_vector_lookup`
looks up when a manifest is built with `atom.data.cw_tiktoc_transfer.build_tiktoc_manifest`.
Uses the same machinery as `cw_03_encode.py` (`SingleLayerVectorCache`,
`extract_many_l15_into_cache`, `text_hash`), the same backbone and the same frozen layer.
Two simplifications follow from TIKTOC's small size (3,714 rows): only the code-alone scope
is encoded, because every TIKTOC episode has length 1 (one row per (SubjectID, ProblemID)),
so every attempt is a first attempt and takes the zero diff vector; and there is no S3
resume, since the extraction finishes in minutes. The shutdown timer is still armed.
Inputs: the TIKTOC data directory (`--tiktoc-dir`).
Outputs (under `--out-dir`): the `codealone/` vector cache,
cw_tiktoc_code_state_hash_map.npz and summary.json.
Run:  python infra/cw_03_encode_tiktoc.py --tiktoc-dir <dir> \\
          --out-dir cache/cw_03_vectors_tiktoc --fuse-minutes 60
Serves: thesis Ch 5, §5.2, and the TIKTOC row of Table 11.
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
from atom.data.cw_tiktoc_transfer import build_tiktoc_attempts  # noqa: E402

LAYER = 15  # frozen production layer, the same as for the CodeWorkout caches; never re-derived here


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
    ap.add_argument("--out-dir", default="cache/cw_03_vectors_tiktoc")
    args = ap.parse_args()

    rearm_boot_fuse(args.fuse_minutes)
    REPO = pathlib.Path(__file__).resolve().parents[1]

    t0 = time.time()
    attempts = build_tiktoc_attempts(base_dir=args.tiktoc_dir)
    valid_code = attempts["code"].notna() & (attempts["code"].astype(str).str.strip() != "")
    n_missing_code = int((~valid_code).sum())

    code_hashes: dict[str, str] = {}
    for code in attempts.loc[valid_code, "code"]:
        code_hashes.setdefault(text_hash(code), code)
    build_s = time.time() - t0
    print(f"[cw-03-tiktoc] {len(attempts)} attempts, {n_missing_code} missing code, "
         f"{len(code_hashes)} unique code texts to encode ({build_s:.1f}s)", flush=True)

    print("[cw-03-tiktoc] loading backbone", flush=True)
    # allow_download=True: google/gemma-4-E4B is Apache-2.0 and not gated, and a fresh
    # cloud instance starts with an empty Hugging Face cache.
    bundle = load_backbone(DEFAULT_BACKBONE, allow_download=True)

    out_dir = REPO / args.out_dir / "codealone"
    cache = SingleLayerVectorCache(out_dir, hidden_dim=bundle["hidden_dim"],
                                   capacity=max(len(code_hashes), 1),
                                   model=bundle["hf_id"], scope="codealone")
    n_already = cache.n_cached
    t_extract0 = time.time()

    def _progress(done: int, total: int) -> None:
        elapsed = time.time() - t_extract0
        rate = done / elapsed if elapsed > 0 else 0.0
        print(f"[cw-03-tiktoc] encoding {done}/{total} ({rate:.2f}/s)", flush=True)

    extract_many_l15_into_cache(
        bundle, code_hashes, layer=LAYER, batch_size=args.batch_size, cache=cache,
        boundary_cap=args.boundary_cap, max_length=args.max_length, progress_cb=_progress,
    )
    extract_s = time.time() - t_extract0

    code_map_path = REPO / args.out_dir / "cw_tiktoc_code_state_hash_map.npz"
    valid_attempts = attempts.loc[valid_code]
    np.savez(code_map_path,
            code_state_id=valid_attempts["code_state_id"].to_numpy(),
            code_hash=np.array([text_hash(c) for c in valid_attempts["code"]]),
            layer=LAYER, scope="codealone", model=bundle["hf_id"])

    n_finite = int(np.isfinite(cache.states("mean")).all())
    passed = cache.n_cached == len(code_hashes) and bool(n_finite)

    summary = {
        "n_attempts": len(attempts), "n_missing_code": n_missing_code,
        "n_code_unique": len(code_hashes), "n_already_cached_on_start": n_already,
        "n_code_encoded": cache.n_cached, "all_finite": bool(n_finite),
        "build_s": round(build_s, 1), "extract_s": round(extract_s, 1),
        "layer": LAYER, "batch_size": args.batch_size, "gate_pass": passed,
    }
    summary_path = REPO / args.out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    print(f"\ncw_03_encode_tiktoc: {'PASS' if passed else 'FAIL'} -- "
         f"code {cache.n_cached}/{len(code_hashes)}, code_map -> {code_map_path}", flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
