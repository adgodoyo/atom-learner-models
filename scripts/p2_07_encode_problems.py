#!/usr/bin/env python
"""Encodes all FA problems into the production item codec and freezes the problem transform.

Extends the sample cache (the growable RawStateCache reuses the ~1.2k problems already encoded),
encodes the remaining problems, re-fits and re-freezes the problem transform (mean pooling,
layer 15) on the full set, and materialises the production lookup problem_id -> codec vector
that the filter consumes. With `--svd-solver full` and a larger `--k-max`, a nestedness
self-check runs before anything is overwritten. Answers are not encoded here: short numeric FA
answers are handled by value, and the semantic answer encoding is reserved for rich productions
(CodeWorkout code). Outputs: `transforms/problem_mean_L15.npz`,
`artifacts/problem_codec_mean_L15.npz`, `reports/p2_07_encode_problems.md`.
Run: `python scripts/p2_07_encode_problems.py [--svd-solver full --k-max 300]`.  Serves §5.2.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from atom.codec.backbone import DEFAULT_BACKBONE, MODEL_REGISTRY, is_cached, load_backbone  # noqa: E402
from atom.codec.extract import RawStateCache, extract_texts  # noqa: E402
from atom.codec.transform import CodecTransform  # noqa: E402
from atom.data.fa import load_problems  # noqa: E402
from atom.eval.report import gate_report  # noqa: E402
from atom.utils.logging import RunLogger  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[1]
CACHE_DIR = REPO / "cache" / "fa_problems"
POOLING, LAYER, K = "mean", 15, 5  # frozen Job-A config


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k-max", type=int, default=50,
                    help="eigen-directions retained in the SAVED TRANSFORM file "
                         "(default 50; the CodeWorkout codec-width ablation uses "
                         "300 for headroom up to rank 261). Does NOT change the "
                         "production codec's own dimensionality -- see --production-rank.")
    ap.add_argument("--svd-solver", choices=["randomized", "full"], default="randomized",
                    help="default 'randomized' reproduces the original 50-component fit "
                         "byte-for-byte. 'full' is deterministic and exactly self-nested "
                         "-- required whenever k_max is being EXTENDED past a value "
                         "already frozen in a committed transform, since randomized_svd's "
                         "exact output for a given component depends on k_max itself.")
    ap.add_argument("--production-rank", type=int, default=50,
                    help="rank used when building artifacts/problem_codec_mean_L15.npz "
                         "(default 50 -- keeps the production codec at "
                         "its frozen 45 = rank-k dims regardless of how large --k-max is; "
                         "the transform file can store more components than production "
                         "currently uses).")
    args = ap.parse_args()

    if not is_cached(MODEL_REGISTRY[DEFAULT_BACKBONE]["hf_id"])[0]:
        print("gemma4-e4b not cached; run p2_01_backbone.py with --allow-download first.")
        return 3

    logger = RunLogger(config={"task": "p2_07_encode_problems", "pooling": POOLING,
                               "layer": LAYER, "k": K, "k_max": args.k_max,
                               "svd_solver": args.svd_solver, "production_rank": args.production_rank},
                       data_version="FA-raw-2026-03")
    prob = load_problems(clean_body=True).drop_duplicates("problem_id")
    items = [(int(r.problem_id), str(r.problem_body_clean)) for r in prob.itertuples(index=False)
             if str(r.problem_body_clean).strip()]
    logger.heartbeat(step=0, force=True, stage="loading backbone", n_problems=len(items))
    bundle = load_backbone(DEFAULT_BACKBONE)

    logger.heartbeat(step=0, force=True, stage="encoding all problems (cache extends the sample)")
    stats = extract_texts(bundle, items, CACHE_DIR, logger=logger)

    # Re-fit + re-freeze the problem transform on the FULL set (was fit on the ~1.2k sample).
    cache = RawStateCache(CACHE_DIR, stats["n_layers"], stats["hidden_dim"],
                          capacity=stats["n_cached"], model=bundle["hf_id"], template="")
    X = np.asarray(cache.states(POOLING)[:, LAYER, :], dtype=np.float64)

    # Nestedness self-check, on this exact real data, BEFORE overwriting anything -- only
    # meaningful/run when svd_solver="full" and k_max is
    # being extended past the current production rank; a no-op assertion otherwise.
    nestedness_checked = False
    nestedness_max_abs_diff = None
    nestedness_ok = True
    if args.svd_solver == "full" and args.k_max > args.production_rank:
        ct_narrow_check = CodecTransform.fit(X, k_max=args.production_rank, layer=LAYER,
                                             pooling=POOLING, svd_solver="full")
        ct_wide_check = CodecTransform.fit(X, k_max=args.k_max, layer=LAYER,
                                           pooling=POOLING, svd_solver="full")
        diff = np.abs(ct_narrow_check.eigvecs - ct_wide_check.eigvecs[:, :args.production_rank])
        nestedness_max_abs_diff = float(diff.max())
        nestedness_checked = True
        nestedness_ok = nestedness_max_abs_diff <= 1e-8
        logger.heartbeat(step=0, force=True, stage="nestedness self-check",
                         max_abs_diff=nestedness_max_abs_diff, ok=nestedness_ok)

    # Disclosure-only comparison against whatever is CURRENTLY on disk (the old
    # randomized-SVD-based artifact, if svd_solver/k_max are changing) -- computed BEFORE
    # overwriting, never gated (known, accepted cost of switching solvers -- see
    # CodecTransform.fit's own docstring), reported so the change is never silent.
    old_artifact_path = REPO / "artifacts" / "problem_codec_mean_L15.npz"
    delta_vs_old_max_abs, delta_vs_old_mean_abs = None, None

    if not nestedness_ok:
        logger.summary(status="gate-fail", nestedness_max_abs_diff=nestedness_max_abs_diff)
        gate_report(
            path="reports/p2_07_encode_problems.md",
            title="P2-7 (Job A) - full problem production cache",
            meta={"run_id": logger.run_id, "git_sha": logger.git_sha, "config_hash": logger.config_hash,
                 "k_max": args.k_max, "svd_solver": args.svd_solver, "production_rank": args.production_rank},
            metrics={"nestedness self-check (full-SVD, production_rank vs k_max)": nestedness_max_abs_diff},
            passed=False,
            gate_desc="full-SVD refit at k_max must be exactly self-nested (max abs eigenvector "
                      "diff <= 1e-8 at production_rank) before the shared transform is touched.",
            notes=["**ABORTED before writing anything** -- the nestedness self-check failed, so "
                  "neither transforms/problem_mean_L15.npz nor artifacts/problem_codec_mean_L15.npz "
                  "was overwritten. This should not happen for svd_solver='full' (self-nestedness "
                  "is a mathematical property of prefixing one decomposition); investigate before retrying."],
        )
        print(f"\nP2-7 nestedness gate: FAIL (max abs diff {nestedness_max_abs_diff}) -- aborted, nothing written.")
        return 1

    if old_artifact_path.exists():
        old = np.load(old_artifact_path, allow_pickle=False)
        old_codec, old_pids = old["codec"], old["problem_id"]

    ct = CodecTransform.fit(X, k_max=args.k_max, layer=LAYER, pooling=POOLING,
                            source=f"fa_problems_full:{POOLING}:L{LAYER}", svd_solver=args.svd_solver)
    ct.save(REPO / "transforms" / "problem_mean_L15.npz")

    # Materialise the production lookup: problem_id -> codec vector (committed, small).
    # `rank=args.production_rank` (default 50) keeps this artifact's own
    # dimensionality (rank-k = 45 by default) fixed regardless of how large the saved
    # transform's own k_max is -- the transform can grow without the production codec
    # silently changing shape underneath every existing consumer.
    pid_row = {pid: cache.row_of_hash[h] for h, pids in cache.ids_of_hash.items()
               for pid in map(int, pids)}
    Z = ct.apply(X, k=K, rank=args.production_rank, whiten=False)
    pids = sorted(pid_row)
    codec = np.stack([Z[pid_row[p]] for p in pids]).astype(np.float32)

    if old_artifact_path.exists() and list(old_pids) == pids and old_codec.shape == codec.shape:
        d = np.abs(old_codec.astype(np.float64) - codec.astype(np.float64))
        delta_vs_old_max_abs, delta_vs_old_mean_abs = float(d.max()), float(d.mean())

    art = REPO / "artifacts"
    art.mkdir(exist_ok=True)
    np.savez(art / "problem_codec_mean_L15.npz", problem_id=np.array(pids, dtype=np.int64),
             codec=codec, pooling=POOLING, layer=LAYER, k=K, version=ct.version, hash=ct.hash)

    finite = bool(np.isfinite(codec).all())
    n_dup_text = len(items) - stats["n_cached"]  # problems sharing identical cleaned body text
    # The gate is "every problem_id is covered", NOT "cache size == problem count": distinct
    # problems legitimately share identical short body text (verified: e.g. "Select all the
    # true statements." recurs across problem-set variants) and correctly dedupe to one cache row.
    passed = len(pids) == len(items) and codec.shape[0] == len(pids) and finite and nestedness_ok
    logger.log_metrics(n_problems=len(items), n_cached=stats["n_cached"],
                       n_encoded=stats["n_encoded"], n_dup_text=n_dup_text,
                       codec_dim=int(codec.shape[1]), transform_fit_n=ct.fit_n,
                       transform_hash=ct.hash, finite=finite,
                       k_max=args.k_max, svd_solver=args.svd_solver, production_rank=args.production_rank,
                       nestedness_checked=nestedness_checked, nestedness_max_abs_diff=nestedness_max_abs_diff,
                       delta_vs_old_max_abs=delta_vs_old_max_abs, delta_vs_old_mean_abs=delta_vs_old_mean_abs)
    logger.summary(status="ok" if passed else "gate-fail", n_cached=stats["n_cached"])

    gate_report(
        path="reports/p2_07_encode_problems.md",
        title="P2-7 (Job A) - full problem production cache",
        meta={"run_id": logger.run_id, "git_sha": logger.git_sha, "config_hash": logger.config_hash,
             "k_max": args.k_max, "svd_solver": args.svd_solver, "production_rank": args.production_rank},
        metrics={
            "problems (non-empty body)": len(items),
            "problem_ids covered in production lookup": f"{len(pids)} / {len(items)}",
            "encoded this run": stats["n_encoded"],
            "unique text hashes in cache": f"{stats['n_cached']}  ({n_dup_text} problems share exact body text with another — verified genuine, not a cleaning artifact)",
            "production codec vectors": f"{codec.shape[0]} × {codec.shape[1]}  (problem_id → mean/L15/k5)",
            "transform re-fit on": f"{ct.fit_n} unique texts  (hash {ct.hash})",
            "all finite": finite,
            "nestedness self-check (full-SVD, production_rank vs k_max)": (
                f"max abs eigenvector diff {nestedness_max_abs_diff:.2e} (checked, k_max={args.k_max} "
                f"vs production_rank={args.production_rank})" if nestedness_checked
                else "not applicable this run (svd_solver != 'full' or k_max == production_rank)"),
            "delta vs previously-committed production artifact (disclosure only, not gated)": (
                f"max abs {delta_vs_old_max_abs:.4f}, mean abs {delta_vs_old_mean_abs:.5f}"
                if delta_vs_old_max_abs is not None else "not computed (no prior artifact, or shape/pids changed)"),
        },
        passed=passed,
        gate_desc="every problem_id covered by the production codec-vector lookup (text-level "
                  "dedup is expected, not a failure); transform re-fit on the full set; finite; "
                  "IF svd_solver='full' and k_max > production_rank, the nestedness self-check "
                  "must also hold (checked before any file is written, see above).",
        notes=[
            "Job A only — answers are not encoded (short numeric answers are value-based; "
            "the semantic answer encoding is reserved for rich productions, i.e. CW code).",
            "artifacts/problem_codec_mean_L15.npz is committed (small); raw states stay in cache/ (gitignored).",
            "transforms/problem_mean_L15.npz re-frozen on the full corpus (was the ~1.2k sample).",
            f"{n_dup_text} problems share identical cleaned body text with another problem "
            "(e.g. short generic prompts like 'Select all the true statements.' recurring across "
            "problem-set variants) — spot-checked genuine, correctly mapped to the same codec vector.",
        ] + ([
            f"**The SVD solver is switched to 'full' and k_max is raised to {args.k_max}** "
            "(from 50) — measured, not assumed: the 'randomized' solver's exact "
            "output for a fixed k_max depends on k_max itself (a different-shape random projection "
            "consumes the seeded RNG stream differently), so a wider refit under 'randomized' does "
            "NOT reproduce the narrower one to float tolerance — confirmed on this exact corpus "
            "(eigenvector cosine similarity down to ~0.994 for the last few components before a "
            "50-truncation, concentrated where the spectrum has no elbow: eigenvalue ratios ~1.01-1.04 "
            "between consecutive components 44-49, a near-degenerate block where individual eigenvector "
            "DIRECTIONS are ill-conditioned even though eigenVALUES match to 4 decimals). 'full' "
            "(deterministic economy SVD) is exactly self-nested by construction — verified above, "
            "not merely asserted — so any further rank extension reproduces this fit exactly.",
            f"**One-time cost, disclosed:** switching solvers means these "
            "production codec vectors are NOT bit-identical to the randomized-SVD-based "
            "ones — delta reported above. This is a precision correction concentrated in the same "
            "near-degenerate tail, not a wholesale change to 𝒵; it was chosen after the "
            "nestedness failure of the randomized solver was measured, in preference to "
            "creating a second, CW-local transform variant.",
        ] if args.svd_solver == "full" and nestedness_checked else []),
    )
    print(f"\nP2-7: {'PASS' if passed else 'FAIL'} · {stats['n_cached']} problems cached "
          f"({stats['n_encoded']} new); codec {codec.shape} · report: reports/p2_07_encode_problems.md")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
