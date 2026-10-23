#!/usr/bin/env python
"""Builds the two text bundles FA-SIM3 trains and evaluates on, locally and deterministically.

  (a) `fa_sim3_train_corrected.tar.gz` -- the 20% training bundle
      (`staging/bundles/fa_06_stage2_20pct.tar.gz`: 239,181 rows, 3,500 learners, a nested
      superset of the 5% bundle used by FA-SIM1/2) with the corrected question text from
      `fa_stage_corrected_bundles.build_corrected_question_text` applied, so the prompt text is identical
      to the earlier simulators' over 4x the rows.
  (b) `fa_sim3_val_coldstart.tar.gz` -- an evaluation bundle that oversamples cold start. The
      earlier val-eval bundle (`fa_06_val_eval.tar.gz`, 3,398 rows) holds only about 110 rows
      with t<10, too few for a cold-start stratum; the val split (500 learners, every one with
      at least 10 events) holds 5,000. Construction (seeded):
        1. Load v2 val-split events, join question_text/answer_text (same join as
           `fa_04b_stage_bundle.py::build_bundle_rows`), drop rows with no answer_text.
        2. Compute `outcome_label` and keep `outcome_valid` rows only.
        3. Cold-start stratum: sample exactly `--n-cold` (500) rows from the valid t<10 pool,
           or all of them if fewer survive (reported, not padded).
        4. Rest stratum: sample the remainder up to `--n-total` (5,000) uniformly from the
           valid t>=10 pool, disjoint from the cold-start draw and not reweighted.
        5. Concatenate and apply the same corrected question text as (a).

Inputs: local FoundationalASSIST CSVs, `artifacts/fa_manifest_v2.parquet`, the v2 splits and the
`staging/` bundle above. Outputs: the two tarballs in `staging/bundles/` and a JSON summary at
`--report-path` (with .json suffix). Raw text stays on the local machine.
Run: `python scripts/fa_sim3_stage_bundles.py [--seed 0 --n-cold 500 --n-total 5000]`.
Staging for Tables 10 and 12 (FA-SIM3, the mathematics simulator).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
import tarfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from atom.data.fa import load_interactions, load_problems  # noqa: E402
from atom.data.fa_manifest import load_v2_events  # noqa: E402
from atom.data.fa_outcome_label import outcome_label  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[1]
MANIFEST = REPO / "artifacts" / "fa_manifest_v2.parquet"
LEARNER_SPLIT = REPO / "splits" / "fa_learner_split_v2.csv"
ITEM_SPLIT = REPO / "splits" / "fa_item_split_v2.csv"

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from fa_stage_corrected_bundles import build_corrected_question_text  # noqa: E402


def _sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_bundle(rows: pd.DataFrame, out_path: pathlib.Path) -> None:
    stage_dir = pathlib.Path(f"/tmp/fa_sim3_stage_{out_path.stem}")
    stage_dir.mkdir(exist_ok=True)
    events_path = stage_dir / "events.jsonl"
    rows.to_json(events_path, orient="records", lines=True, force_ascii=False)
    with tarfile.open(out_path, "w:gz") as tf:
        tf.add(events_path, arcname="events.jsonl")


def _verify_roundtrip(out_path: pathlib.Path, expected_rows: int) -> dict:
    extract_dir = pathlib.Path(f"/tmp/fa_sim3_verify_{out_path.stem}")
    extract_dir.mkdir(exist_ok=True)
    with tarfile.open(out_path, "r:gz") as tf:
        tf.extractall(extract_dir, filter="data")
    reloaded = pd.read_json(extract_dir / "events.jsonl", lines=True)
    return {
        "row_count_matches": len(reloaded) == expected_rows,
        "no_blank_question_text": bool((reloaded["question_text"].astype(str) != "").all()),
        "no_blank_answer_text": bool((reloaded["answer_text"].astype(str) != "").all()),
        "no_duplicate_ids": bool(reloaded["id"].is_unique),
    }


def stage_train_20pct(seed: int) -> dict:
    """Part (a): the existing 20% row selection with the corrected question text applied."""
    in_path = REPO / "staging" / "bundles" / "fa_06_stage2_20pct.tar.gz"
    out_path = REPO / "staging" / "bundles" / "fa_sim3_train_corrected.tar.gz"
    extract_dir = pathlib.Path("/tmp/fa_sim3_stage_20pct_in")
    extract_dir.mkdir(exist_ok=True)
    with tarfile.open(in_path, "r:gz") as tf:
        tf.extractall(extract_dir, filter="data")
    bundle = pd.read_json(extract_dir / "events.jsonl", lines=True)

    problems = load_problems(clean_body=True).drop_duplicates("problem_id").set_index("problem_id")
    corrected_text_by_problem = {
        pid: build_corrected_question_text(row) for pid, row in problems.iterrows()
    }
    missing = bundle.loc[~bundle["problem_id"].isin(corrected_text_by_problem), "problem_id"].unique()
    assert len(missing) == 0, f"{len(missing)} problem_id(s) missing from Problems.csv: {missing[:10]}"
    n_before = len(bundle)
    bundle["question_text"] = bundle["problem_id"].map(corrected_text_by_problem)
    assert len(bundle) == n_before

    _write_bundle(bundle, out_path)
    checks = _verify_roundtrip(out_path, n_before)
    return {
        "out_path": str(out_path), "n_rows": n_before,
        "n_unique_learners": int(bundle["learner"].nunique()),
        "n_unique_problems": int(bundle["problem_id"].nunique()),
        "archive_sha256": _sha256(out_path), "checks": checks,
    }


def stage_coldstart_eval(seed: int, n_cold: int, n_total: int) -> dict:
    """Part (b): the cold-start-oversampled eval bundle, built + corrected in one pass."""
    ev = load_v2_events(MANIFEST, LEARNER_SPLIT, ITEM_SPLIT)
    val = ev[ev["split"] == "val"].reset_index(drop=True)
    n_val_raw = len(val)
    n_val_learners = int(val["learner"].nunique())
    n_t_lt10_raw = int((val["t"] < 10).sum())

    interactions = load_interactions(parse_time=False)
    problems_raw = load_problems()
    answers = interactions.drop_duplicates(subset="id").set_index("id")["answer_text"]
    questions_raw = problems_raw.drop_duplicates("problem_id").set_index("problem_id")["problem_body"]

    val = val.copy()
    val["question_text"] = val["problem_id"].map(questions_raw)
    val["answer_text"] = val["id"].map(answers)
    val = val[val["answer_text"].notna() & (val["answer_text"].astype(str) != "")].reset_index(drop=True)
    n_after_answer_join = len(val)

    labels = outcome_label(val["support_code"], val["score"], val["score_missing"])
    val = val[labels["outcome_valid"].to_numpy()].reset_index(drop=True)
    n_valid = len(val)

    cold_pool = val[val["t"] < 10]
    rest_pool = val[val["t"] >= 10]
    n_cold_pool, n_rest_pool = len(cold_pool), len(rest_pool)

    rng = np.random.default_rng(seed)
    n_cold_take = min(n_cold, n_cold_pool)
    cold_idx = rng.choice(cold_pool.index.to_numpy(), size=n_cold_take, replace=False) if n_cold_take else np.array([], dtype=int)
    n_rest_take = min(n_total - n_cold_take, n_rest_pool)
    rest_idx = rng.choice(rest_pool.index.to_numpy(), size=n_rest_take, replace=False) if n_rest_take else np.array([], dtype=int)

    selected = val.loc[np.concatenate([cold_idx, rest_idx])].reset_index(drop=True)
    selected = selected.sort_values(["learner", "t"], kind="mergesort").reset_index(drop=True)

    problems_clean = load_problems(clean_body=True).drop_duplicates("problem_id").set_index("problem_id")
    corrected_text_by_problem = {
        pid: build_corrected_question_text(row) for pid, row in problems_clean.iterrows()
    }
    missing = selected.loc[~selected["problem_id"].isin(corrected_text_by_problem), "problem_id"].unique()
    assert len(missing) == 0, f"{len(missing)} problem_id(s) missing from Problems.csv: {missing[:10]}"
    selected["question_text"] = selected["problem_id"].map(corrected_text_by_problem)

    keep_cols = ["learner", "id", "problem_id", "t", "support_code", "score", "score_missing",
                "question_text", "answer_text"]
    out_rows = selected[keep_cols].reset_index(drop=True)

    out_path = REPO / "staging" / "bundles" / "fa_sim3_val_coldstart.tar.gz"
    _write_bundle(out_rows, out_path)
    checks = _verify_roundtrip(out_path, len(out_rows))

    return {
        "out_path": str(out_path), "seed": seed,
        "n_val_raw_events": n_val_raw, "n_val_learners": n_val_learners,
        "n_t_lt10_raw": n_t_lt10_raw,
        "n_after_answer_join": n_after_answer_join, "n_outcome_valid": n_valid,
        "n_cold_pool_valid": n_cold_pool, "n_rest_pool_valid": n_rest_pool,
        "n_cold_taken": int(n_cold_take), "n_rest_taken": int(n_rest_take),
        "n_rows_final": len(out_rows),
        "n_t_lt10_final": int((out_rows["t"] < 10).sum()),
        "n_unique_learners_final": int(out_rows["learner"].nunique()),
        "archive_sha256": _sha256(out_path), "checks": checks,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-cold", type=int, default=500)
    ap.add_argument("--n-total", type=int, default=5000)
    ap.add_argument("--report-path", default=str(REPO / "reports" / "fa_sim3_prep_bundles.md"))
    args = ap.parse_args()

    print("[fa-sim3-stage] part (a): 20% train bundle, corrected prompts...", flush=True)
    train_result = stage_train_20pct(args.seed)
    print(f"[fa-sim3-stage] train: {train_result}", flush=True)

    print("[fa-sim3-stage] part (b): cold-start-oversampled eval bundle...", flush=True)
    eval_result = stage_coldstart_eval(args.seed, args.n_cold, args.n_total)
    print(f"[fa-sim3-stage] eval: {eval_result}", flush=True)

    passed = (
        all(train_result["checks"].values())
        and all(eval_result["checks"].values())
        and eval_result["n_t_lt10_final"] >= min(args.n_cold, eval_result["n_cold_pool_valid"])
    )

    report = {"train": train_result, "eval": eval_result, "passed": passed,
              "seed": args.seed, "n_cold_requested": args.n_cold, "n_total_requested": args.n_total}
    pathlib.Path(args.report_path).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.report_path).with_suffix(".json").write_text(json.dumps(report, indent=2, default=str))
    print(f"[fa-sim3-stage] {'PASS' if passed else 'FAIL'}. JSON written to "
         f"{pathlib.Path(args.report_path).with_suffix('.json')}", flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
