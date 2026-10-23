#!/usr/bin/env python
"""Builds a local FA prompt bundle: a seeded subsample of events with their question and answer text.

Draws from the cleaned v2 FA manifest (deduplicated, missing scores masked, 4-class support
label) via `load_v2_events`, keeps one split (`--split`, default train) and only rows whose
platform score is present (a row with a missing score cannot supervise the success head),
then selects `--n-learners` learners and `--frac` of their events with `select_lm_subsample`.
Inputs: artifacts/fa_manifest_v2.parquet, the v2 learner and item splits, Interactions.csv
and Problems.csv.
Outputs (under `--out-dir`, default staging/bundles/): `<name>.tar.gz` holding events.jsonl
(learner, id, problem_id, t, support_code, score, score_missing, question_text,
answer_text), and `<name>.manifest.json` (counts and hashes, no text); plus a gate report.
The archive contains licensed FA text: it is built locally and never published.
Run:  python scripts/fa_04b_stage_bundle.py [--split val] [--name <name>]
Serves: staging for FA-SIM3 (Tables 10 and 12). `fa_sim3_stage_bundles.py` reuses
`build_bundle_rows` and corrects the prompt text (`fa_stage_corrected_bundles.py`) to build
the bundles that `infra/fa_sim3_full_run.py` loads.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import shutil
import sys
import tarfile
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from atom.data.fa import load_interactions, load_problems  # noqa: E402
from atom.data.fa_manifest import load_v2_events  # noqa: E402
from atom.data.tuples import select_lm_subsample  # noqa: E402
from atom.eval.report import gate_report  # noqa: E402
from atom.utils.logging import RunLogger, git_sha  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[1]
MANIFEST = REPO / "artifacts" / "fa_manifest_v2.parquet"
LEARNER_SPLIT = REPO / "splits" / "fa_learner_split_v2.csv"
ITEM_SPLIT = REPO / "splits" / "fa_item_split_v2.csv"


def build_bundle_rows(events_subset: pd.DataFrame, interactions: pd.DataFrame,
                      problems: pd.DataFrame) -> pd.DataFrame:
    """Attach question_text/answer_text to the selected v2 events and drop rows with no answer.

    Interactions.csv and Problems.csv both carry raw duplicate rows; dropping them before
    indexing never changes which text an id/problem_id maps to, since duplicate-ID groups
    are byte-identical (see fa_manifest.dedup_interactions)."""
    answers = interactions.drop_duplicates(subset="id").set_index("id")["answer_text"]
    questions = problems.drop_duplicates("problem_id").set_index("problem_id")["problem_body"]
    out = events_subset[["learner", "id", "problem_id", "t", "support_code", "score",
                         "score_missing"]].reset_index(drop=True).copy()
    out["question_text"] = out["problem_id"].map(questions)
    out["answer_text"] = out["id"].map(answers)

    n_before = len(out)
    out = out[out["answer_text"].notna() & (out["answer_text"] != "")]
    n_after_answer_filter = len(out)
    out.attrs["n_before"] = n_before
    out.attrs["n_dropped_missing_answer"] = n_before - n_after_answer_filter
    return out.reset_index(drop=True)


def sha256_of(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def package_bundle(rows: pd.DataFrame, out_dir: pathlib.Path, name: str, seed: int,
                   frac: float, n_learners: int | None, split: str = "train") -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    stage_dir = pathlib.Path(tempfile.mkdtemp(prefix="fa_04b_stage_"))
    try:
        jsonl_path = stage_dir / "events.jsonl"
        rows.to_json(jsonl_path, orient="records", lines=True, force_ascii=False)

        archive_path = out_dir / f"{name}.tar.gz"
        with tarfile.open(archive_path, "w:gz") as tf:
            tf.add(jsonl_path, arcname="events.jsonl")
        archive_sha256 = sha256_of(archive_path)

        manifest = {
            "name": name, "event_count": int(len(rows)), "seed": seed, "frac": frac,
            "n_learners": n_learners, "split": split, "manifest_version": "v2",
            "unique_learners": int(rows["learner"].nunique()),
            "unique_problems": int(rows["problem_id"].nunique()),
            "archive_sha256": archive_sha256, "archive_bytes": archive_path.stat().st_size,
            "git_sha": git_sha(),
        }
        (out_dir / f"{name}.manifest.json").write_text(json.dumps(manifest, indent=2))
        return manifest
    finally:
        shutil.rmtree(stage_dir, ignore_errors=True)


def verify_roundtrip(out_dir: pathlib.Path, name: str, manifest: dict) -> dict:
    archive_path = out_dir / f"{name}.tar.gz"
    rehash_ok = sha256_of(archive_path) == manifest["archive_sha256"]
    extract_dir = pathlib.Path(tempfile.mkdtemp(prefix="fa_04b_verify_"))
    try:
        with tarfile.open(archive_path, "r:gz") as tf:
            tf.extractall(extract_dir, filter="data")
        reloaded = pd.read_json(extract_dir / "events.jsonl", lines=True)
        return {
            "archive_rehash_matches_manifest": rehash_ok,
            "reloaded_row_count_matches": len(reloaded) == manifest["event_count"],
            "no_blank_question_text": bool(
                (reloaded["question_text"].notna() & (reloaded["question_text"] != "")).all()
            ),
            "no_duplicate_event_ids": bool(reloaded["id"].is_unique),
            "no_score_missing_rows": bool((~reloaded["score_missing"]).all()),
        }
    finally:
        shutil.rmtree(extract_dir, ignore_errors=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frac", type=float, default=0.05)
    ap.add_argument("--n-learners", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default=str(REPO / "staging" / "bundles"))
    ap.add_argument("--name", default="fa_04b_v2_lm_subsample")
    ap.add_argument("--split", default="train", choices=("train", "val"),
                    help="which split's events to package; split=val builds an evaluation "
                        "bundle for scoring held-out NLL. This flag only selects which "
                        "population's text is packaged; the script has no training loop")
    ap.add_argument("--report-path", default=str(REPO / "reports" / "fa_04b_stage_bundle.md"),
                    help="where the gate report is written; give each bundle its own path "
                        "so that building several bundles does not overwrite one report")
    args = ap.parse_args()
    n_learners = None if args.n_learners == 0 else args.n_learners

    logger = RunLogger(config={"task": "fa_04b_stage_bundle", "frac": args.frac,
                               "n_learners": n_learners, "seed": args.seed,
                               "split": args.split},
                       data_version="fa-v2")

    logger.heartbeat(step=0, force=True, stage="loading FA-0 v2 events (Mac local)")
    ev = load_v2_events(MANIFEST, LEARNER_SPLIT, ITEM_SPLIT)
    train_events = ev[(ev["split"] == args.split) & (~ev["score_missing"])]
    interactions = load_interactions(parse_time=False)
    problems = load_problems()
    logger.heartbeat(step=1, force=True, stage="loaded", n_events=len(ev),
                     n_eligible_train_events=len(train_events))

    subsample = select_lm_subsample(train_events, args.frac, args.seed, n_learners)
    rows = build_bundle_rows(subsample, interactions, problems)
    logger.heartbeat(step=2, force=True, stage="subsample built", n_selected=len(rows),
                     n_dropped_missing_answer=rows.attrs["n_dropped_missing_answer"])

    out_dir = pathlib.Path(args.out_dir)
    manifest = package_bundle(rows, out_dir, args.name, args.seed, args.frac, n_learners,
                          split=args.split)
    checks = verify_roundtrip(out_dir, args.name, manifest)
    passed = all(checks.values())

    metrics = {**manifest, **checks}
    logger.summary(status="ok" if passed else "gate-fail", **metrics)

    notes = [
        f"bundle: staging/bundles/{args.name}.tar.gz (local only, holds licensed FA text) and "
        f"{args.name}.manifest.json (committed, no text).",
        "sourced from the cleaned v2 manifest (deduplicated, score never imputed, 4-class "
        "support), one split only, score_missing rows excluded (nothing to stage for a "
        "row that cannot supervise the success head); the v1 event stream, which lacks "
        "the query-independent belief boundary and the missing-score mask, is not used.",
        f"selection: {n_learners or 'all'} learners (seeded draw) x {args.frac:.0%} of "
        f"their eligible events (seeded draw, distinct RNG stream) -> "
        f"{rows.attrs['n_before']} events selected, {rows.attrs['n_dropped_missing_answer']} "
        f"dropped for missing/blank answer_text -> **{manifest['event_count']} final events**.",
        "fields staged per event: learner (hashed), id, problem_id, t, support_code, "
        "score, score_missing, question_text, answer_text; support_code and score are "
        "small numeric labels (not text) needed by the simulator's support and success "
        "losses; nothing else from the raw tables.",
        f"round-trip verified fresh: rehash "
        f"{'matches' if checks['archive_rehash_matches_manifest'] else 'MISMATCH'}, "
        f"row count {'matches' if checks['reloaded_row_count_matches'] else 'MISMATCH'}, "
        f"no blank question text: {checks['no_blank_question_text']}, "
        f"no duplicate ids: {checks['no_duplicate_event_ids']}, "
        f"no score_missing rows: {checks['no_score_missing_rows']}.",
        "Licence note: the archive holds licensed FA text; it may be copied to a "
        "private training machine but is never uploaded to shared storage or published.",
    ]

    gate_report(
        path=args.report_path,
        title="FA-4b prep - v2 §2.5 staging bundle",
        meta={"run_id": logger.run_id, "git_sha": logger.git_sha,
              "config_hash": logger.config_hash, "status": "ok" if passed else "gate-fail"},
        metrics={k: v for k, v in metrics.items() if k != "archive_sha256"} | {
            "archive_sha256 (first 16 hex chars)": manifest["archive_sha256"][:16],
        },
        passed=passed,
        gate_desc="v2-manifest LM-subsample bundle built from the requested split only, "
                  "score_missing excluded, hashed learner ids; round-trip verified.",
        notes=notes,
    )
    print(f"\nfa_04b_stage_bundle: {'PASS' if passed else 'FAIL'} "
         f"· report: {args.report_path}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
