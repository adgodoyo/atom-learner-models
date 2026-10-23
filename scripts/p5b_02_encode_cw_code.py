#!/usr/bin/env python
"""Encodes all CodeWorkout code submissions (code alone) at the production configuration chosen by
the probe battery: mean pooling (geometry) and boundary-mean (evidence) at layer 15, both raw
and undeflated.

The probe battery (`p2_05b_probe_battery.py`) measured codealone/mean/raw AUC 0.995 (task probe)
and codealone/boundary_mean/raw AUC 0.874 (person probe), with the raw view beating the deflated
k5 view on both, so no transform is applied here. One causal forward pass per unique code text
(atom.codec.cw_extract.extract_scope); no prefix is ever re-encoded. The probe battery's codealone
cache (`cache/cw_p25b/codealone`, hash-keyed and resumable) is extended, not rebuilt, so the
1,020 submissions it already holds are reused; only the remaining unique codes (of 11,775 unique
texts across 13,083 submissions, 246 students, 17 problems) are encoded.

Full per-boundary trajectories (variable length, the episode encoder's input) are not
precomputed here: the episode encoder calls `boundary.extract_boundary_states` directly when it
needs the per-step sequence. Outputs: the extended cache, `artifacts/cw_code_state_hash_map.npz`
(CodeStateID -> code_hash), `reports/p5b_02_encode_cw_code.md`.
Run: `python scripts/p5b_02_encode_cw_code.py`.  Serves §5.2 (codec on CW).
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from atom.codec.backbone import (  # noqa: E402
    DEFAULT_BACKBONE,
    MODEL_REGISTRY,
    is_cached,
    load_backbone,
)
from atom.codec.cw_extract import BoundaryCache, extract_scope, probe_metadata  # noqa: E402
from atom.codec.extract import text_hash  # noqa: E402
from atom.data.cw import load_granular  # noqa: E402
from atom.eval.report import gate_report  # noqa: E402
from atom.utils.logging import RunLogger  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[1]
CACHE_DIR = REPO / "cache" / "cw_p25b" / "codealone"  # extends the probe battery's cache, same scope
LAYER = 15  # the probe battery's best layer for this readout family, fixed everywhere
FLUSH_EVERY = 100


def main() -> int:
    if not is_cached(MODEL_REGISTRY[DEFAULT_BACKBONE]["hf_id"])[0]:
        print("gemma4-e4b not cached; run p2_01_backbone.py with --allow-download first.")
        return 3

    logger = RunLogger(config={"task": "p5b_02_encode_cw_code", "layer": LAYER},
                       data_version="CW-TIKTOC-2024")

    logger.heartbeat(step=0, force=True, stage="loading CW submissions")
    df = load_granular(include_code=True)
    non_empty = df["Code"].apply(lambda c: bool(str(c).strip()))
    n_missing_code = int((~non_empty).sum())

    unique_codes: dict[str, str] = {}
    for code in df.loc[non_empty, "Code"]:
        h = text_hash(code)
        unique_codes.setdefault(h, code)

    logger.heartbeat(step=0, force=True, stage="loading backbone",
                     n_submissions=len(df), n_unique_codes=len(unique_codes),
                     n_missing_code=n_missing_code)
    bundle = load_backbone(DEFAULT_BACKBONE)
    meta = probe_metadata(bundle)

    cache = BoundaryCache(CACHE_DIR, meta["n_layers"], meta["hidden_dim"],
                         capacity=max(len(unique_codes), 1),
                         model=bundle["hf_id"], scope="codealone")

    todo = [h for h in unique_codes if not cache.has(h)]
    n_already_cached = len(unique_codes) - len(todo)
    logger.heartbeat(step=0, force=True, stage="encoding",
                     n_todo=len(todo), n_already_cached=n_already_cached)

    for i, h in enumerate(todo):
        vecs = extract_scope(bundle, unique_codes[h], prefix="")
        cache.add(h, vecs)
        if (i + 1) % FLUSH_EVERY == 0:
            cache.flush()
        logger.heartbeat(step=i + 1, total=len(todo), stage="encoding")
    cache.flush()

    # -- production artifact: a SMALL per-submission (CodeStateID -> code_hash) mapping,
    # not a copy of the vectors themselves. The full raw mean/L15 + boundary_mean/L15
    # states already live in `cache` (13,083 submissions x 2,560 dims x 2 readouts is
    # ~270MB -- cache-tier, not "small artifact" tier; committing a duplicate of it
    # would break the size convention of the other artifacts/*.npz, which are all
    # item-level and sub-2MB). Downstream consumers (the episode encoder)
    # load this mapping plus BoundaryCache(CACHE_DIR, ...).states("mean"/"boundary_mean")
    # directly -- one source of truth, no duplicated vectors to drift out of sync.
    code_state_ids, code_hashes = [], []
    for r in df.itertuples(index=False):
        if not str(r.Code).strip():
            continue
        h = text_hash(r.Code)
        assert cache.has(h)  # guaranteed by the encode loop above; defensive only
        code_state_ids.append(r.CodeStateID)
        code_hashes.append(h)

    art = REPO / "artifacts"
    art.mkdir(exist_ok=True)
    out_path = art / "cw_code_state_hash_map.npz"
    np.savez(out_path,
            code_state_id=np.array(code_state_ids), code_hash=np.array(code_hashes),
            cache_dir=str(CACHE_DIR.relative_to(REPO)), layer=LAYER, scope="codealone",
            pooling_geometry="mean", pooling_evidence="boundary_mean",
            transform="none (raw, undeflated -- the probe battery found raw beat deflated k5 for both uses)",
            model=bundle["hf_id"])

    n_covered = len(code_state_ids)
    n_expected = len(df) - n_missing_code
    mean_L15 = cache.states("mean")[:, LAYER, :]
    bmean_L15 = cache.states("boundary_mean")[:, LAYER, :]
    finite = bool(np.isfinite(mean_L15).all() and np.isfinite(bmean_L15).all())
    passed = (n_covered == n_expected) and finite

    logger.log_metrics(n_submissions=len(df), n_unique_codes=len(unique_codes),
                       n_encoded_this_run=len(todo), n_already_cached=n_already_cached,
                       n_covered=n_covered, n_expected=n_expected, finite=finite)
    logger.summary(status="ok" if passed else "gate-fail", n_covered=n_covered)

    gate_report(
        path="reports/p5b_02_encode_cw_code.md",
        title="P5b-2 - CW Job-B (code-alone) production encode",
        meta={"run_id": logger.run_id, "git_sha": logger.git_sha, "config_hash": logger.config_hash,
              "layer": LAYER, "scope": "codealone"},
        metrics={
            "CW submissions (13083 expected)": f"{n_covered} / {n_expected}",
            "unique code texts": len(unique_codes),
            "encoded this run": len(todo),
            "reused from P2-5b's own cache": n_already_cached,
            "production lookup": f"{len(code_state_ids)} CodeStateID -> code_hash rows "
                                 f"(vectors themselves: {mean_L15.shape[0]} x {mean_L15.shape[1]} "
                                 f"mean/L15/raw + boundary_mean/L15/raw, in cache/, not duplicated)",
            "all finite": finite,
        },
        passed=passed,
        gate_desc="every CW submission with non-empty code covered by the per-CodeStateID "
                 "codec-vector lookup (mean + boundary_mean, both raw/undeflated at L15, "
                 "the production config chosen by the probe battery); finite.",
        notes=[
            "Job-B (code-alone) only -- extends the probe battery's codealone cache "
            "(cache/cw_p25b/codealone) rather than rebuilding it; its 1,020 already-"
            "encoded submissions are reused as-is.",
            "Raw (undeflated) vectors only, per the probe battery's finding that raw beats the "
            "deflated k5 transform for both the task-geometry probe (0.995) and the "
            "person/style-evidence probe (0.874) -- no codec transform is fit or applied "
            "here.",
            "Full per-boundary TRAJECTORIES (variable-length) are deliberately NOT "
            "precomputed here -- boundary.py assigns that to the "
            "episode encoder (extract_boundary_states, called directly when needed), "
            "avoiding a second forward pass per code now for something not consumed "
            "until the next step.",
            "artifacts/cw_code_state_hash_map.npz is committed (small, ~13k rows of two "
            "strings): CodeStateID -> code_hash only. The actual mean/boundary_mean "
            "vectors stay in cache/cw_p25b/codealone (gitignored) -- 13,083 x 2,560 x 2 "
            "readouts is ~270MB, cache-tier not small-artifact-tier (every other "
            "artifacts/*.npz is item-level and sub-2MB); duplicating it "
            "into a committed file would break that convention and create a second copy "
            "to drift out of sync. Downstream consumers join this mapping against the "
            "cache directly.",
        ],
    )
    print(f"\nP5b-2: {'PASS' if passed else 'FAIL'} - {n_covered}/{n_expected} submissions "
         f"covered - report: reports/p5b_02_encode_cw_code.md")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
