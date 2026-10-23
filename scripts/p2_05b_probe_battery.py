#!/usr/bin/env python
"""Code-representation probe battery on CodeWorkout: which axes of code meaning the frozen codec
carries (Table 6).

A single same/different-behaviour probe on the frozen answer configuration gives a chance-level
AUC that is ambiguous without a positive control. This battery asks three separate questions:

  (a) does the pipeline capture the code at all (task probe = positive control);
  (b) which axis of code meaning (task / behaviour-gross / behaviour-near-miss / person) the
      cached states carry, and how strongly;
  (c) which view of the cache (raw vs the frozen deflated transform) serves which probe -- the
      two-view hypothesis (deflated = item geometry, raw = person-style evidence for the filter).

One shared per-problem submission pool feeds all four probes (src/atom/codec/cw_probes.py), so
every submission is encoded exactly once per scope (src/atom/codec/cw_extract.py): prompt‖code
and code-alone, three readouts each (last, mean, boundary-mean -- the last via
src/atom/codec/boundary.py, also used by the episode encoder).

Discipline: problems are split calibrate/confirm before any pair is drawn (seed 0, 30% confirm).
The full grid (probe x view x readout x scope) is computed on calibrate only; one confirmation
pass re-scores the recommended cell per consumer on held-out problems. The frozen codec
configuration is not re-selected: the same frozen answer_last_L15 transform is read, extended
across more readouts, probes and views.

Pre-registered gate: the task probe (positive control) must clear TASK_GATE_AUC in at least one
cell, or the whole battery is void (pipeline bug). Person/behaviour outcomes are reported
findings, interpreted by the pre-registered outcome map, not pass/fail.
Outputs: `cache/cw_p25b/`, `reports/p2_05b_probe_battery.md`, `reports/p2_05b_grid.csv`.
Run: `python scripts/p2_05b_probe_battery.py`.
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from atom.codec.backbone import (  # noqa: E402
    DEFAULT_BACKBONE,
    MODEL_REGISTRY,
    is_cached,
    load_backbone,
)
from atom.codec.cw_extract import BoundaryCache, extract_scope, probe_metadata  # noqa: E402
from atom.codec.cw_probes import (  # noqa: E402
    build_behaviour_gross_pairs,
    build_behaviour_nearmiss_pairs,
    build_person_pairs,
    build_submission_pool,
    build_task_pairs,
    edit_similarity,
    split_problems,
)
from atom.codec.extract import qa_prefix_full, text_hash  # noqa: E402
from atom.codec.transform import CodecTransform  # noqa: E402
from atom.data.cw import load_granular  # noqa: E402
from atom.eval.metrics import auc  # noqa: E402
from atom.eval.report import gate_report  # noqa: E402
from atom.utils.logging import RunLogger  # noqa: E402
from atom.utils.seeds import seed_everything  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[1]
CACHE_PC = REPO / "cache" / "cw_p25b" / "promptcode"
CACHE_CA = REPO / "cache" / "cw_p25b" / "codealone"
TRANSFORMS_DIR = REPO / "transforms"
SEED = 0
CONFIRM_FRAC = 0.3
POOL_PER_PROBLEM = 60
PAIRS_PER_PROBLEM = 20
BOUNDARY_CAP = 32
K_VIEWS = (5, 10)
READOUTS = ("last", "mean", "boundary_mean")
SCOPES = ("promptcode", "codealone")
TASK_GATE_AUC = 0.65
FROZEN_TRANSFORM = "answer_last_L15"  # the frozen FA answer transform; not re-fit here


def _auc(labels, scores) -> float:
    return auc(np.asarray(labels, dtype=float), np.asarray(scores, dtype=float))


def _bootstrap(labels, scores, b: int = 1000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    labels, scores = np.asarray(labels), np.asarray(scores)
    n = len(labels)
    vals = []
    for _ in range(b):
        idx = rng.integers(0, n, n)
        if len(set(labels[idx].tolist())) < 2:
            continue
        vals.append(_auc(labels[idx], scores[idx]))
    if not vals:
        return (float("nan"), float("nan"))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def _add_hash_columns(pool: pd.DataFrame) -> pd.DataFrame:
    pool = pool.copy()
    h_pc, h_ca = [], []
    for r in pool.itertuples(index=False):
        _, full = qa_prefix_full(r.prompt, r.Code)
        h_pc.append(text_hash(full))
        h_ca.append(text_hash(r.Code))
    pool["h_pc"], pool["h_ca"] = h_pc, h_ca
    return pool


def _encode_pool(pool: pd.DataFrame, logger: RunLogger) -> tuple[BoundaryCache, BoundaryCache]:
    """Encode every unique submission in `pool`, both scopes, resumable + cached."""
    hf_id = MODEL_REGISTRY[DEFAULT_BACKBONE]["hf_id"]
    unique_pc: dict[str, tuple[str, str]] = {}   # hash -> (prefix, code)
    unique_ca: dict[str, tuple[str, str]] = {}   # hash -> ("", code)
    for r in pool.itertuples(index=False):
        prefix, _ = qa_prefix_full(r.prompt, r.Code)
        unique_pc.setdefault(r.h_pc, (prefix, r.Code))
        unique_ca.setdefault(r.h_ca, ("", r.Code))

    bundle = None
    n_layers = hidden_dim = None
    caches = {}
    for scope, unique, cache_dir in (("promptcode", unique_pc, CACHE_PC),
                                     ("codealone", unique_ca, CACHE_CA)):
        meta_path = cache_dir / "meta.json"
        have_meta = meta_path.exists()
        if have_meta:
            import json
            m = json.loads(meta_path.read_text())
            cache = BoundaryCache(cache_dir, m["n_layers"], m["hidden_dim"], m["capacity"],
                                  m["model"], scope)
        else:
            if bundle is None:
                if not is_cached(hf_id)[0]:
                    raise RuntimeError(f"{hf_id} not cached; run p2_01_backbone.py with --allow-download first.")
                logger.heartbeat(step=0, force=True, stage=f"loading backbone for {scope}")
                bundle = load_backbone(DEFAULT_BACKBONE)
                meta = probe_metadata(bundle)
                n_layers, hidden_dim = meta["n_layers"], meta["hidden_dim"]
            cache = BoundaryCache(cache_dir, n_layers, hidden_dim, capacity=max(len(unique), 1),
                                  model=hf_id, scope=scope)
        n_todo = sum(1 for h in unique if not cache.has(h))
        if n_todo:
            if bundle is None:
                if not is_cached(hf_id)[0]:
                    raise RuntimeError(f"{hf_id} not cached; run p2_01_backbone.py with --allow-download first.")
                logger.heartbeat(step=0, force=True, stage=f"loading backbone for {scope}")
                bundle = load_backbone(DEFAULT_BACKBONE)
            i = 0
            for h, (prefix, code) in unique.items():
                if cache.has(h):
                    continue
                vecs = extract_scope(bundle, code, prefix=prefix, boundary_cap=BOUNDARY_CAP)
                cache.add(h, vecs)
                i += 1
                if i % 50 == 0:
                    cache.flush()
                    logger.heartbeat(step=i, total=n_todo, stage=f"encoding {scope}")
            cache.flush()
        caches[scope] = cache
    return caches["promptcode"], caches["codealone"]


def _score_view(X_layer: np.ndarray, view: str, ct: CodecTransform) -> np.ndarray:
    if view == "raw":
        Z = X_layer
    else:
        k = int(view[1:])
        Z = ct.apply(X_layer, k=k, whiten=False)  # deflated-cosine convention
    return Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12)


def _pair_scores(pairs: pd.DataFrame, pool: pd.DataFrame, hash_col: str,
                 cache: BoundaryCache, readout: str, ct: CodecTransform, view: str) -> np.ndarray:
    X = np.asarray(cache.states(readout)[:, ct.layer, :], dtype=np.float64)
    Zn = _score_view(X, view, ct)
    row = cache.row_of_hash
    a_idx = [row[pool.loc[i, hash_col]] for i in pairs["id_a"]]
    b_idx = [row[pool.loc[i, hash_col]] for i in pairs["id_b"]]
    return (Zn[a_idx] * Zn[b_idx]).sum(axis=1)


def _grid_for_split(probes: dict[str, pd.DataFrame], pool: pd.DataFrame,
                    cache_pc: BoundaryCache, cache_ca: BoundaryCache,
                    ct: CodecTransform) -> list[dict]:
    caches = {"promptcode": (cache_pc, "h_pc"), "codealone": (cache_ca, "h_ca")}
    cells = []
    for probe_name, pairs in probes.items():
        if len(pairs) == 0 or pairs["label"].nunique() < 2:
            continue
        labels = pairs["label"].to_numpy()
        for scope in SCOPES:
            cache, hash_col = caches[scope]
            for readout in READOUTS:
                for view in ("raw", "k5", "k10"):
                    scores = _pair_scores(pairs, pool, hash_col, cache, readout, ct, view)
                    a = _auc(labels, scores)
                    lo, hi = _bootstrap(labels, scores, seed=SEED)
                    cells.append({"probe": probe_name, "scope": scope, "readout": readout,
                                 "view": view, "auc": a, "ci_lo": lo, "ci_hi": hi,
                                 "n": len(pairs)})
    return cells


def _edit_baseline(probes: dict[str, pd.DataFrame], pool: pd.DataFrame) -> dict[str, float]:
    out = {}
    for name, pairs in probes.items():
        if len(pairs) == 0 or pairs["label"].nunique() < 2:
            continue
        scores = [edit_similarity(pool.loc[r.id_a, "Code"], pool.loc[r.id_b, "Code"])
                 for r in pairs.itertuples(index=False)]
        out[name] = _auc(pairs["label"].to_numpy(), np.array(scores))
    return out


def main() -> int:
    seed_everything(SEED)
    ct = CodecTransform.load(TRANSFORMS_DIR / f"{FROZEN_TRANSFORM}.npz")

    logger = RunLogger(config={"task": "p2_05b_probe_battery", "seed": SEED,
                               "confirm_frac": CONFIRM_FRAC, "pool_per_problem": POOL_PER_PROBLEM,
                               "pairs_per_problem": PAIRS_PER_PROBLEM, "boundary_cap": BOUNDARY_CAP,
                               "frozen_transform": FROZEN_TRANSFORM, "frozen_layer": ct.layer},
                       data_version="CW-TIKTOC-2024")

    logger.heartbeat(step=0, force=True, stage="loading CW data + splitting problems")
    df = load_granular(include_code=True)
    problem_ids = sorted(df["ProblemID"].unique().tolist())
    calib_ids, confirm_ids = split_problems(problem_ids, confirm_frac=CONFIRM_FRAC, seed=SEED)
    logger.heartbeat(step=0, force=True, stage="split", calib=calib_ids, confirm=confirm_ids)

    pool_calib = _add_hash_columns(build_submission_pool(df, calib_ids, POOL_PER_PROBLEM, SEED))
    pool_confirm = _add_hash_columns(build_submission_pool(df, confirm_ids, POOL_PER_PROBLEM, SEED))
    pool_all = pd.concat([pool_calib, pool_confirm], ignore_index=False)

    def _probes(pool, problem_ids):
        return {
            "task": build_task_pairs(pool, pairs_total=PAIRS_PER_PROBLEM * len(problem_ids), seed=SEED),
            "behaviour-gross": build_behaviour_gross_pairs(pool, PAIRS_PER_PROBLEM, SEED),
            "behaviour-nearmiss": build_behaviour_nearmiss_pairs(pool, PAIRS_PER_PROBLEM, SEED),
            "person": build_person_pairs(pool, PAIRS_PER_PROBLEM, SEED),
        }

    probes_calib = _probes(pool_calib, calib_ids)
    probes_confirm = _probes(pool_confirm, confirm_ids)
    counts = {k: (len(v), int(v["label"].sum()) if len(v) else 0) for k, v in probes_calib.items()}
    logger.heartbeat(step=0, force=True, stage="pairs built (calibrate)", **{
        f"{k}_n": c[0] for k, c in counts.items()})

    cache_pc, cache_ca = _encode_pool(pool_all, logger)
    logger.heartbeat(step=0, force=True, stage="encoding complete",
                     n_promptcode=cache_pc.n_cached, n_codealone=cache_ca.n_cached)

    grid = _grid_for_split(probes_calib, pool_calib, cache_pc, cache_ca, ct)
    grid_df = pd.DataFrame(grid)

    # -- pre-registered outcome map ------------------------------------------------------
    task_cells = grid_df[grid_df["probe"] == "task"]
    task_best = task_cells.loc[task_cells["auc"].idxmax()] if len(task_cells) else None
    task_passes_somewhere = bool(task_cells["auc"].max() >= TASK_GATE_AUC) if len(task_cells) else False

    def _best(probe, view_prefix=None):
        sub = grid_df[grid_df["probe"] == probe]
        if view_prefix:
            sub = sub[sub["view"] == view_prefix]
        return sub.loc[sub["auc"].idxmax()] if len(sub) else None

    person_raw_best = _best("person", "raw")
    person_defl_best = grid_df[(grid_df["probe"] == "person") & (grid_df["view"] != "raw")]
    person_defl_best = person_defl_best.loc[person_defl_best["auc"].idxmax()] if len(person_defl_best) else None
    two_view_confirmed = bool(person_raw_best is not None and person_defl_best is not None
                              and person_raw_best["auc"] > person_defl_best["ci_hi"])

    gross_best = _best("behaviour-gross")
    nearmiss_best = _best("behaviour-nearmiss")
    edit_calib = _edit_baseline(probes_calib, pool_calib)

    # -- confirmation pass: re-score ONLY the recommended cell per consumer, held-out problems
    recs = {
        "geometry (deflated, task probe)":
            {"probe": "task", "scope": task_best["scope"], "readout": task_best["readout"],
             "view": task_best["view"]} if task_best is not None else None,
        "filter-evidence (person probe, best view)":
            {"probe": "person",
             "scope": (person_raw_best if two_view_confirmed or person_raw_best is None
                      or person_defl_best is None or person_raw_best["auc"] >= person_defl_best["auc"]
                      else person_defl_best)["scope"],
             "readout": (person_raw_best if two_view_confirmed or person_raw_best is None
                        or person_defl_best is None or person_raw_best["auc"] >= person_defl_best["auc"]
                        else person_defl_best)["readout"],
             "view": (person_raw_best if two_view_confirmed or person_raw_best is None
                     or person_defl_best is None or person_raw_best["auc"] >= person_defl_best["auc"]
                     else person_defl_best)["view"]} if person_raw_best is not None else None,
        "near-miss (best cell, any view)": {"probe": "behaviour-nearmiss",
             "scope": nearmiss_best["scope"], "readout": nearmiss_best["readout"],
             "view": nearmiss_best["view"]} if nearmiss_best is not None else None,
    }
    confirm_results = {}
    caches_confirm = {"promptcode": (cache_pc, "h_pc"), "codealone": (cache_ca, "h_ca")}
    for name, cfg in recs.items():
        if cfg is None:
            continue
        pairs = probes_confirm.get(cfg["probe"])
        if pairs is None or len(pairs) == 0 or pairs["label"].nunique() < 2:
            confirm_results[name] = None
            continue
        cache, hash_col = caches_confirm[cfg["scope"]]
        scores = _pair_scores(pairs, pool_confirm, hash_col, cache, cfg["readout"], ct, cfg["view"])
        a = _auc(pairs["label"].to_numpy(), scores)
        lo, hi = _bootstrap(pairs["label"].to_numpy(), scores, seed=SEED)
        confirm_results[name] = {"cfg": cfg, "auc": a, "ci": (lo, hi), "n": len(pairs)}

    passed = task_passes_somewhere  # the ONLY hard pass/fail gate; rest are reported findings

    logger.log_metrics(passed=passed, task_best_auc=float(task_cells["auc"].max()) if len(task_cells) else float("nan"),
                       person_raw_best_auc=float(person_raw_best["auc"]) if person_raw_best is not None else float("nan"),
                       person_defl_best_auc=float(person_defl_best["auc"]) if person_defl_best is not None else float("nan"),
                       two_view_confirmed=two_view_confirmed,
                       gross_best_auc=float(gross_best["auc"]) if gross_best is not None else float("nan"),
                       nearmiss_best_auc=float(nearmiss_best["auc"]) if nearmiss_best is not None else float("nan"))
    logger.summary(status="ok" if passed else "gate-fail", task_best_auc=float(task_cells["auc"].max()) if len(task_cells) else float("nan"))

    def _fmt(row):
        if row is None:
            return "n/a"
        return f"AUC {row['auc']:.3f} [{row['ci_lo']:.3f},{row['ci_hi']:.3f}] ({row['scope']}/{row['readout']}/{row['view']}, n={row['n']})"

    def _fmt_confirm(name):
        r = confirm_results.get(name)
        if not r:
            return "n/a (too few confirm-split pairs)"
        cfg = r["cfg"]
        return (f"AUC {r['auc']:.3f} [{r['ci'][0]:.3f},{r['ci'][1]:.3f}] (n={r['n']}) "
               f"-- {cfg['scope']}/{cfg['readout']}/{cfg['view']}")

    grid_df.to_csv(REPO / "reports" / "p2_05b_grid.csv", index=False)

    gate_report(
        path="reports/p2_05b_probe_battery.md",
        title="P2-5b - code representation probe battery (CW, calibrate/confirm)",
        meta={"run_id": logger.run_id, "git_sha": logger.git_sha, "config_hash": logger.config_hash,
              "n_problems_calib": len(calib_ids), "n_problems_confirm": len(confirm_ids),
              "calib_problems": calib_ids, "confirm_problems": confirm_ids,
              "n_pool_calib": len(pool_calib), "n_pool_confirm": len(pool_confirm),
              "frozen_transform": FROZEN_TRANSFORM, "frozen_layer": ct.layer},
        metrics={
            "GATE - task probe (positive control), best cell over the full grid":
                f"{_fmt(task_best)} -- {'PASS' if task_passes_somewhere else 'FAIL'} (gate >= {TASK_GATE_AUC})",
            "person probe - best RAW-view cell": _fmt(person_raw_best),
            "person probe - best DEFLATED-view cell (k5/k10)": _fmt(person_defl_best),
            "two-view hypothesis (raw > deflated, CI-separated)": str(two_view_confirmed),
            "behaviour-gross - best cell": _fmt(gross_best),
            "behaviour-nearmiss - best cell": _fmt(nearmiss_best),
            "edit-distance baseline (calibrate, per probe)":
                ", ".join(f"{k}={v:.3f}" for k, v in edit_calib.items()),
            "CONFIRM (held-out problems) - geometry cell": _fmt_confirm("geometry (deflated, task probe)"),
            "CONFIRM (held-out problems) - filter-evidence cell": _fmt_confirm("filter-evidence (person probe, best view)"),
            "CONFIRM (held-out problems) - near-miss cell": _fmt_confirm("near-miss (best cell, any view)"),
        },
        passed=passed,
        gate_desc=f"Task probe (positive control) must clear AUC >= {TASK_GATE_AUC} in at "
                  "least one grid cell, or the battery is void (pipeline bug, stop-and-fix). "
                  "Person/behaviour outcomes are reported findings interpreted via the "
                  "pre-registered outcome map, not a second gate.",
        notes=[
            f"Problems split calibrate ({len(calib_ids)}) / confirm ({len(confirm_ids)}) "
            "BEFORE any pair was drawn (seed 0, 30% confirm) -- this CW table has only 17 "
            "problems (an exact match to TIKTOC), so the probe is widened by the SAMPLE per "
            f"problem ({POOL_PER_PROBLEM} pool submissions/problem, against about 75 in total "
            "for a single earlier probe), not by the problem count.",
            "Grid = probe(4) x scope(2: prompt‖code, code-alone) x readout(3: last, "
            "mean-over-span, boundary-mean) x view(3: raw, deflated k=5, deflated k=10) "
            "= 72 cells, computed once on calibrate; full grid in reports/p2_05b_grid.csv.",
            "The deflated view reuses the SAME frozen answer_last_L15 transform "
            "(fit on FA, layer 15) -- applying its eigenbasis to CW's mean/boundary-pooled "
            "vectors (not just the last-token pooling it was fit on) is exactly the "
            "generalisation question this battery asks, not a new fit.",
            "behaviour-gross pairs = same test-vector vs a vector differing on >= half the "
            "test suite; behaviour-nearmiss = differing on EXACTLY one test -- a coarse "
            "'any difference' pairing mixes these two regimes together, so a single AUC "
            "number cannot distinguish 'no signal' from 'signal only at gross "
            "distances'.",
            "person pairs hold problem identity FIXED (both submissions on the same "
            "problem) so task/topic geometry cannot leak into the same-author/different-"
            "author label; learner IDs are natural, free labels -- probe-only, never a "
            "training signal.",
            "Equivalence-shaping of the codec is not applied regardless of this battery's "
            "outcome -- it would be considered only if a supervised test-vector head "
            "underperforms AND an ablation shows the representation is the bottleneck.",
        ],
    )
    print(f"\nP2-5b: {'PASS' if passed else 'FAIL'} (task positive control) -- "
          f"task best {_fmt(task_best)} -- report: reports/p2_05b_probe_battery.md")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
