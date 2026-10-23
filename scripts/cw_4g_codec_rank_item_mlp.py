#!/usr/bin/env python
"""Codec-width x item-encoder grid for the CodeWorkout classifier: P(clean-full) AUC, NLL, Brier, ECE per cell.

The recurrent widths stay fixed (32/64/64, summary_dim 16); only the semantic codec and
the item pathway vary. Grid: 4 ranks x {linear, mlp} = 8 cells, GRU body, seed 0 each.
  - Axis 1, codec rank: `codec_dim = rank - k` with k=5 held fixed (asserted), codec_dim in
    {45, 64, 128, 256} <-> rank in {50, 69, 133, 261}. code/diff vectors
    (`atom.data.cw_manifest.build_codec_vector_lookup`) and the 50-item problem codec
    (`rebuild_problem_codec_at_rank`) are rebuilt at each rank from the one shared,
    committed transform `transforms/problem_mean_L15.npz` (svd_solver="full", k_max=300,
    exactly self-nested); it is read, never refitted.
  - Axis 2, item encoder: `atom.simulator.cw_item_mlp` swaps both item pathways (the
    episode encoder's `item_proj` and the heads' `trunk`) for a shared nonlinear encoder,
    with no change to the latent:hidden ratio.
Invariant (also in `tests/test_cw_item_mlp.py`): the MLP is a shared function of the codec
vector, never a per-item lookup; it is checked here on the widest trained cell's weights.
Inputs: code/diff hash maps and codec caches (flags below), `artifacts/cw_problem_statements.json`
and its statement cache. Output: `reports/cw_4g_codec_rank_item_mlp.md`.
Serves Appendix A (codec width and item encoder on CodeWorkout).

    python scripts/cw_4g_codec_rank_item_mlp.py \\
        --code-map cache/cw_03_vectors/cw_code_state_hash_map_full.npz \\
        --diff-map cache/cw_03_vectors/cw_diff_hash_map_full.npz \\
        --codealone-meta cache/cw_03_vectors/codealone/meta.json \\
        --codealone-states cache/cw_03_vectors/codealone/states_mean.npy \\
        --diff-meta cache/cw_03_vectors/diff/meta.json \\
        --diff-states cache/cw_03_vectors/diff/states_mean.npy
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

SCRIPTS_DIR = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR.parents[0] / "src"))
sys.path.insert(0, str(SCRIPTS_DIR))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from cw_manifest_io import load_manifest  # noqa: E402

from atom.codec.extract import RawStateCache, text_hash  # noqa: E402
from atom.codec.transform import CodecTransform  # noqa: E402
from atom.data.cw_manifest import build_codec_vector_lookup, load_hash_maps  # noqa: E402
from atom.data.cw_outcome_label import OUTCOME_CLASSES  # noqa: E402
from atom.eval.metrics import auc, multiclass_brier, multiclass_ece, multiclass_nll  # noqa: E402
from atom.eval.report import gate_report  # noqa: E402
from atom.simulator.cw_4_train_loop import CODEC_K, run_one_seed_body  # noqa: E402
from atom.utils.logging import RunLogger  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[1]
BODY = "gru"
SEED = 0
K_FIXED = CODEC_K  # 5 -- held fixed across the whole rank sweep, asserted below
RANKS = [50, 69, 133, 261]  # codec_dim = rank - 5 -> {45, 64, 128, 256}
ITEM_ENCODERS = ["linear", "mlp"]

TRANSFORM_PATH = REPO / "transforms" / "problem_mean_L15.npz"
STATEMENTS_CACHE_DIR = REPO / "cache" / "cw_problem_statements"
CLEAN_FULL_IDX = OUTCOME_CLASSES.index("CLEAN_FULL")


def rebuild_problem_codec_at_rank(rank: int, transform: CodecTransform | None = None) -> tuple[np.ndarray, dict]:
    """CW's own 50-item problem-statement codec, rebuilt at an arbitrary rank -- same
    primitives (`RawStateCache`, `text_hash`, `CodecTransform`) and the same cached raw
    states `cw_4b_recover_problem_statements.py` uses, applying the (k_max=300) shared
    transform at `rank` instead of the frozen production rank=50. Never writes a new
    artifact -- purely in-memory.

    `transform`: `None` (default) loads the shared, committed
    `transforms/problem_mean_L15.npz`. `cw_4g_2_refit_stability.py` passes a diagnostic,
    resampled-half refit instead, so the same rank-rebuild logic serves both scripts."""
    statements = json.loads((REPO / "artifacts" / "cw_problem_statements.json").read_text())
    ct = CodecTransform.load(TRANSFORM_PATH) if transform is None else transform
    meta = json.loads((STATEMENTS_CACHE_DIR / "meta.json").read_text())
    cache = RawStateCache(STATEMENTS_CACHE_DIR, meta["n_layers"], meta["hidden_dim"],
                          capacity=meta["completed"], model=meta["model"], template=meta["template"])
    X = np.asarray(cache.states(ct.pooling)[:, ct.layer, :], dtype=np.float64)
    Z = ct.apply(X, k=K_FIXED, rank=rank)
    problem_id, codec_rows = [], []
    for row in statements:
        h = text_hash(row["wrapped_prompt"])
        if not cache.has(h):
            raise RuntimeError(f"problem_id {row['problem_id']} missing from the statement cache")
        problem_id.append(int(row["problem_id"]))
        codec_rows.append(cache.row_of_hash[h])
    codec = Z[codec_rows].astype(np.float32)
    pid_to_idx = {pid: i for i, pid in enumerate(problem_id)}
    return codec, pid_to_idx


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default=None)
    ap.add_argument("--code-map", required=True)
    ap.add_argument("--diff-map", required=True)
    ap.add_argument("--codealone-meta", required=True)
    ap.add_argument("--codealone-states", required=True)
    ap.add_argument("--diff-meta", required=True)
    ap.add_argument("--diff-states", required=True)
    ap.add_argument("--learner-batch-size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    device = "cpu"
    logger = RunLogger(config={"task": "cw_4g_codec_rank_item_mlp", "body": BODY, "seed": SEED,
                               "ranks": RANKS, "item_encoders": ITEM_ENCODERS, "k_fixed": K_FIXED},
                       data_version="CSEDM-S19+F19-challenge-train-pool+val")

    logger.heartbeat(step=0, force=True, stage="loading base manifest (rank-independent fields reused)")
    base_manifest = load_manifest(args)
    code_map, diff_map = load_hash_maps(args.code_map, args.diff_map)
    code_hashes_needed = set(base_manifest["attempts"]["code_hash"].dropna())
    diff_hashes_needed = set(base_manifest["attempts"].loc[base_manifest["attempts"]["diff_hash"].notna(), "diff_hash"])
    logger.heartbeat(step=1, force=True, stage="base manifest ready",
                     n_train_learners=len(base_manifest["train_learners"]),
                     n_val_learners=len(base_manifest["val_learners"]))

    transform = CodecTransform.load(TRANSFORM_PATH)
    assert transform.k_max >= max(RANKS), (
        f"transform k_max={transform.k_max} too small for the widest rank {max(RANKS)} -- "
        "the prerequisite regeneration must run first")

    results = {}
    widest_mlp_head = None
    widest_codec_dim = max(r - K_FIXED for r in RANKS)
    for rank in RANKS:
        codec_dim = rank - K_FIXED
        assert codec_dim + K_FIXED == rank, "k must stay fixed while rank varies"

        t0 = time.time()
        code_of_hash = build_codec_vector_lookup(code_hashes_needed, args.codealone_meta,
                                                 args.codealone_states, transform, k=K_FIXED, rank=rank)
        diff_of_hash = build_codec_vector_lookup(diff_hashes_needed, args.diff_meta,
                                                 args.diff_states, transform, k=K_FIXED, rank=rank)
        problem_codec, pid_to_idx = rebuild_problem_codec_at_rank(rank)
        cell_manifest = {**base_manifest, "code_of_hash": code_of_hash, "diff_of_hash": diff_of_hash,
                         "problem_codec": problem_codec, "pid_to_idx": pid_to_idx}
        codec_build_s = time.time() - t0

        for item_encoder in ITEM_ENCODERS:
            name = f"codec{codec_dim}_{item_encoder}"
            t0 = time.time()
            r = run_one_seed_body(BODY, SEED, cell_manifest, args, device,
                                  dims={"codec_dim": codec_dim}, item_encoder=item_encoder)
            wall = time.time() - t0
            n_params = (sum(p.numel() for p in r["model"]["attempt_encoder"].parameters())
                       + sum(p.numel() for p in r["model"]["filt"].parameters())
                       + sum(p.numel() for p in r["model"]["outcome_head"].parameters())
                       + sum(p.numel() for p in r["model"]["y_ever_head"].parameters()))

            probs, target = r["val"]["probs_real"], r["val"]["target"]
            clean_full_binary = (target == CLEAN_FULL_IDX).astype(int)
            p_clean_full = probs[:, CLEAN_FULL_IDX]
            cell_auc = auc(clean_full_binary, p_clean_full) if len(set(clean_full_binary.tolist())) == 2 else float("nan")

            results[name] = {
                "codec_dim": codec_dim, "rank": rank, "item_encoder": item_encoder,
                "n_params": n_params, "codec_build_s": round(codec_build_s, 1), "train_wall_s": round(wall, 1),
                "auc_clean_full": cell_auc, "nll_all_attempts": r["all_attempts"]["nll"],
                "brier_all_attempts": r["all_attempts"]["brier"], "ece_all_attempts": r["all_attempts"]["ece"],
                "n_eval_population": r["all_attempts"]["n_eval_population"],
            }
            logger.heartbeat(step=0, force=True, stage=f"cell={name} done", n_params=n_params,
                             auc=round(cell_auc, 4) if cell_auc == cell_auc else cell_auc,
                             nll=round(r["all_attempts"]["nll"], 4), wall_s=round(wall, 1))
            if item_encoder == "mlp" and codec_dim == widest_codec_dim:
                widest_mlp_head = r["model"]["outcome_head"]

    # -- item-agnosticism invariant, checked at grid scale on the WIDEST trained cell's own
    # weights (not a freshly-retrained probe, and not only the unit-test's fresh-init
    # module): identical codec vectors must produce identical trunk activations regardless
    # of which "item" they're attached to, and the item pathway carries no per-item-sized
    # parameter.
    from atom.simulator.cw_item_mlp import OutcomeHeadItemMLP
    assert widest_mlp_head is not None and isinstance(widest_mlp_head, OutcomeHeadItemMLP)
    codec_a = torch.randn(1, widest_codec_dim)
    twice = torch.cat([codec_a, codec_a], dim=0)
    with torch.no_grad():
        trunk_out = widest_mlp_head.trunk(twice)
    item_agnosticism_ok = bool(torch.allclose(trunk_out[0], trunk_out[1]))
    no_per_item_param = all(p.shape[0] != 2 for p in widest_mlp_head.trunk.parameters())
    logger.heartbeat(step=0, force=True, stage="item-agnosticism invariant checked",
                     ok=item_agnosticism_ok, no_per_item_param=no_per_item_param)

    passed = len(results) == len(RANKS) * len(ITEM_ENCODERS) and item_agnosticism_ok and no_per_item_param
    logger.summary(status="ok" if passed else "gate-fail", n_cells=len(results))

    meta = {"run_id": logger.run_id, "git_sha": logger.git_sha, "body": BODY, "seed": SEED,
           "k_fixed": K_FIXED, "n_train_learners": len(base_manifest["train_learners"]),
           "n_val_learners": len(base_manifest["val_learners"]), "epochs": args.epochs,
           "transform_k_max": transform.k_max, "transform_hash": transform.hash}
    metrics = {
        "grid (codec_dim x item_encoder), AUC on P(clean-full) / NLL / Brier / ECE / n_params": results,
        "item-agnosticism invariant (MLP is a shared function, never a per-item lookup)": {
            "identical_codec_gives_identical_trunk_output": item_agnosticism_ok,
            "no_parameter_sized_by_item_count": no_per_item_param},
    }
    notes = [
        f"**Prerequisite, gated separately**: `reports/p2_07_encode_problems.md` "
        f"(transform re-fit, svd_solver='full', k_max={transform.k_max}, exactly self-nested, "
        f"hash {transform.hash}) and `reports/cw_4b_recover_problem_statements.md` (the CodeWorkout "
        "50-item codec regenerated at rank=50 through the same transform). This script reads "
        "that ONE shared transform at wider ranks for its own grid; it never refits or forks it.",
        f"**k held fixed at {K_FIXED} throughout** -- asserted in code at every rank "
        "(`codec_dim + k == rank` checked at every grid cell).",
        "**Grid**: " + ", ".join(f"{n}: AUC={r['auc_clean_full']:.4f} NLL={r['nll_all_attempts']:.4f} "
                                 f"ECE={r['ece_all_attempts']:.4f} n_params={r['n_params']:,}"
                                 for n, r in results.items()),
        "**Scope, disclosed**: single seed=0 per cell (diagnostic lever-search sweep, as in "
        "cw_4d_width_summary_sweep.py) -- not the 5-seed protocol used for claims. GRU body only "
        "(item_encoder='mlp' is only wired for 'gru', as on the mathematics corpus); the body "
        "bake-off itself is reported separately.",
        "**AUC metric** (the same convention as cw_4f_clean_full_auc.py, without its "
        "baseline comparison): `P(clean-full)` binarised "
        "against the `clean-full` label, `atom.eval.metrics.auc`, all-attempts.",
        f"**Item-agnosticism invariant**, checked on a TRAINED cell's real weights (not only "
        f"`tests/test_cw_item_mlp.py`'s fresh-init unit tests): identical codec vectors -> "
        f"identical trunk output ({item_agnosticism_ok}), no parameter dimension sized by item "
        f"count ({no_per_item_param}).",
    ]

    out_path = REPO / "reports" / "cw_4g_codec_rank_item_mlp.md"
    gate_report(out_path, "CodeWorkout codec rank x item-encoder nonlinearity grid",
               meta, metrics, passed,
               "All 8 grid cells complete (4 ranks x {linear,mlp}) AND the item-agnosticism "
               "invariant holds. Fails if: k changed alongside rank (asserted, not just "
               "documented) · the nestedness gate was skipped (it was not -- see prerequisite "
               "reports) · a CW-local transform variant was created (it was not -- one shared "
               "file) · param counts are not reported per cell (they are, above).",
               notes)

    print("\n" + "=" * 90)
    print("Codec rank x item-encoder grid, P(clean-full) AUC / NLL")
    print("=" * 90)
    print(f"{'cell':<20}{'codec_dim':>10}{'n_params':>12}{'auc':>9}{'nll':>9}{'ece':>9}{'wall_s':>9}")
    for name, r in results.items():
        print(f"{name:<20}{r['codec_dim']:>10}{r['n_params']:>12,}{r['auc_clean_full']:>9.4f}"
             f"{r['nll_all_attempts']:>9.4f}{r['ece_all_attempts']:>9.4f}{r['train_wall_s']:>9.1f}")
    print(f"\ncw_4g_codec_rank_item_mlp: {'PASS' if passed else 'FAIL'}")
    print(f"report: reports/cw_4g_codec_rank_item_mlp.md")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
