#!/usr/bin/env python
"""Builds the raw hidden-state calibration cache for the FA codec (answer-in-context encoding).

Encodes a seeded calibration slice of FA interactions (`question ‖ answer`, pooled at answer
positions: last-token and mean-over-span) at all layers of the frozen backbone into a
resumable, hash-keyed cache, then verifies that a re-run is 100% cache hits and that the cached
states are finite and correctly shaped. Inputs: FoundationalASSIST interactions and problems,
the locally cached backbone. Outputs: `cache/fa_calib/` (derived vectors and hashed keys, no
raw text) and `reports/p2_02a_extract.md`.
Run: `python scripts/p2_02a_extract_calibration.py --n 500`.  Serves §5.2 (codec).
"""

from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from atom.codec.backbone import DEFAULT_BACKBONE, is_cached, load_backbone  # noqa: E402
from atom.codec.extract import RawStateCache, extract_calibration  # noqa: E402
from atom.data.fa import load_interactions, load_problems  # noqa: E402
from atom.eval.report import gate_report  # noqa: E402
from atom.utils.logging import RunLogger  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[1]
CACHE_DIR = REPO / "cache" / "fa_calib"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500, help="calibration interactions to sample")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    from atom.codec.backbone import MODEL_REGISTRY

    if not is_cached(MODEL_REGISTRY[DEFAULT_BACKBONE]["hf_id"])[0]:
        print("gemma4-e4b not cached; run p2_01_backbone.py with --allow-download first.")
        return 3

    logger = RunLogger(config={"task": "p2_02a_extract_calibration", "n": args.n,
                               "seed": args.seed}, data_version="FA-raw-2026-03")
    logger.heartbeat(step=0, force=True, stage="loading FA + cleaning problems")
    inter = load_interactions(parse_time=False)
    prob = load_problems(clean_body=True)
    logger.heartbeat(step=0, force=True, stage="loading backbone")
    bundle = load_backbone(DEFAULT_BACKBONE)

    logger.heartbeat(step=0, force=True, stage="encoding calibration slice")
    stats = extract_calibration(inter, prob, bundle, CACHE_DIR, n=args.n, seed=args.seed,
                                logger=logger)
    logger.heartbeat(step=stats["n_unique"], total=stats["n_unique"], force=True,
                     stage="encoded", **{k: stats[k] for k in ("n_unique", "n_encoded")})

    # Resumability: a second pass must encode nothing (100% cache hits).
    rerun = extract_calibration(inter, prob, bundle, CACHE_DIR, n=args.n, seed=args.seed)
    resumable = rerun["n_encoded"] == 0

    # Integrity: cached states finite and correctly shaped.
    cache = RawStateCache(CACHE_DIR, stats["n_layers"], stats["hidden_dim"],
                          capacity=stats["n_unique"], model=bundle["hf_id"],
                          template="")
    last, mean = cache.states("last"), cache.states("mean")
    shape_ok = (last.shape == (stats["n_unique"], stats["n_layers"], stats["hidden_dim"])
                and mean.shape == last.shape)
    finite_ok = bool(np.isfinite(last).all() and np.isfinite(mean).all())

    passed = (stats["n_encoded"] == stats["n_unique"] and resumable and shape_ok and finite_ok)
    logger.log_metrics(**stats, rerun_encoded=rerun["n_encoded"], resumable=resumable,
                       shape_ok=shape_ok, finite_ok=finite_ok)
    logger.summary(status="ok" if passed else "gate-fail", **stats, resumable=resumable)

    gate_report(
        path="reports/p2_02a_extract.md",
        title="P2-2a - Raw-state calibration cache (answer-in-context)",
        meta={"run_id": logger.run_id, "git_sha": logger.git_sha,
              "config_hash": logger.config_hash, "cache": str(CACHE_DIR.relative_to(REPO))},
        metrics={
            "interactions sampled": stats["n_interactions"],
            "unique question‖answer": stats["n_unique"],
            "encoded (1st pass)": stats["n_encoded"],
            "encoded (2nd pass)": f"{rerun['n_encoded']}  (resumable={resumable})",
            "states shape": f"{last.shape}  (unique × layers × dim)",
            "all finite": finite_ok,
            "pooled readouts": "last-token + mean-over-answer-span, all 43 layers",
        },
        passed=passed,
        gate_desc="calibration slice encoded at all layers (both poolings); re-run is 100% "
                  "cache hits; states finite and correctly shaped.",
        notes=[
            "answer-in-context: `question ‖ answer`, read at answer token positions only.",
            "cache stores derived vectors + hashed keys + interaction ids (no raw text, licence boundary).",
            f"cache dir (gitignored): {CACHE_DIR.relative_to(REPO)}",
            "layer/pooling/k selection is made by p2_02c_sweep.py.",
        ],
    )
    print(f"\nP2-2a: {'PASS' if passed else 'FAIL'} · {stats['n_unique']} unique encoded, "
          f"resumable={resumable} · report: reports/p2_02a_extract.md")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
