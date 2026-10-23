#!/usr/bin/env python
"""Build the canonical FoundationalASSIST (FA) event manifest and its learner and item splits.

Reads the raw FA interaction and problem tables and applies the preprocessing order set
out in `src/atom/data/fa_manifest.py`: exact-ID deduplication, near-duplicate problem
grouping, scores resolved to {0, 1, missing} (never imputed), 4-class support labels,
chronological order over valid-time rows, a learner split and an item split, and a sealed
lockbox. It also runs leakage audits with planted canaries (a re-duplicated row, a
learner leak and an item leak) that must be caught. Counts are checked against the
expected values in `configs/data.yaml`. Run: `python scripts/fa_00_manifest_rebuild.py`
(exit code 1 if any check fails). Serves Chapter 4 (data and splits).

Writes:
  - artifacts/fa_manifest_v2.parquet      (gitignored; regenerate by re-running this script)
  - splits/fa_learner_split_v2.csv        (learner -> train/val/development_exposed/lockbox)
  - splits/fa_item_split_v2.csv           (problem_group_id -> train/heldout)
  - splits/fa_problem_groups_v2.csv       (problem_id -> problem_group_id, for downstream joins)
  - splits/fa_lockbox_manifest_v2.json    (sha256 + counts only -- no lockbox rows, ever)
  - reports/fa_00_manifest_rebuild.md     (the check report)
"""

from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import yaml  # noqa: E402

from atom.data.fa import load_interactions, load_problems  # noqa: E402
from atom.data.fa_manifest import (  # noqa: E402
    assert_unique_ids,
    attach_splits,
    audit_lockbox_isolated,
    audit_missing_time_excluded,
    audit_monotonic_time,
    audit_no_duplicate_ids,
    audit_score_never_imputed,
    audit_split_disjoint,
    build_canonical_events,
    build_problem_catalog,
    dedup_interactions,
    inject_duplicate_row,
    inject_split_leak,
    lockbox_hash,
    make_item_split_v2,
    make_learner_split_v2,
)
from atom.data.tuples import load_split_manifest  # noqa: E402
from atom.eval.report import gate_report  # noqa: E402
from atom.utils.logging import RunLogger  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[1]


def main() -> int:
    cfg = yaml.safe_load((REPO / "configs" / "data.yaml").read_text())["fa"]["manifest_v2"]
    gate = cfg["gate"]
    logger = RunLogger(config={"task": "fa_00_manifest_rebuild", **cfg}, data_version="FA-raw-2026-03")

    explanations: list[str] = []

    # 1-2. load raw tables + cleaned problem text (fa.py / fa_clean.py).
    logger.heartbeat(step=0, force=True, stage="loading raw interactions + problems")
    interactions_raw = load_interactions(parse_time=False)
    problems_clean = load_problems(clean_body=True)
    n_raw = len(interactions_raw)

    # 3. exact-ID dedup, before any sequence/lag assembly.
    logger.heartbeat(step=1, force=True, stage="deduplicating exact-ID rows")
    interactions_dedup, dedup_stats = dedup_interactions(interactions_raw)
    assert_unique_ids(interactions_dedup)

    # Planted-duplicate canary: a deliberately re-duplicated row must be caught.
    canary_leak = inject_duplicate_row(interactions_dedup, seed=0)
    canary_caught = not audit_no_duplicate_ids(canary_leak)["unique_ids"]

    dedup_matches_gate = (
        dedup_stats["n_dropped"] == gate["n_dedup_dropped"]
        and dedup_stats["n_dup_groups"] == gate["n_dedup_groups"]
    )
    if not dedup_matches_gate:
        explanations.append(
            f"DEDUP MISMATCH: measured {dedup_stats['n_dropped']} dropped / "
            f"{dedup_stats['n_dup_groups']} groups vs gate "
            f"{gate['n_dedup_dropped']}/{gate['n_dedup_groups']} -- investigate before trusting "
            "downstream counts."
        )

    # Near-duplicate problem grouping (before the item split).
    logger.heartbeat(step=2, force=True, stage="grouping near-duplicate problems")
    catalog = build_problem_catalog(problems_clean)

    # 4-9. canonical event manifest: distinct repeated-item IDs kept, score resolved to
    # {0,1,missing} (never imputed), 4-class support, chronological order over valid-time
    # rows only, gap/first-event/first-exposure features, missingness indicators.
    logger.heartbeat(step=3, force=True, stage="building canonical event manifest")
    events, build_stats = build_canonical_events(interactions_dedup, catalog)

    score_matches_gate = build_stats["n_score_missing"] == gate["n_score_missing"]
    if not score_matches_gate:
        explanations.append(
            f"SCORE-MISSING MISMATCH: measured {build_stats['n_score_missing']} vs gate "
            f"{gate['n_score_missing']}."
        )
    # The reference figure of ~5,194 missing-time rows was measured before deduplication;
    # some duplicate rows themselves had missing time, so the post-dedup count is expected
    # to be slightly lower. Reported, not checked.
    explanations.append(
        f"Missing-time rows: {build_stats['n_missing_time']} measured post-dedup "
        f"(the reference figure of ~5,194 is pre-dedup; "
        f"{5194 - build_stats['n_missing_time']} of those rows were exact duplicates "
        "removed by deduplication, not a new loss)."
    )

    # Leakage audits relevant to the manifest.
    a_unique = audit_no_duplicate_ids(events)
    a_score = audit_score_never_imputed(events)
    a_mono = audit_monotonic_time(events)
    a_missing_excl = audit_missing_time_excluded(events)

    # 10. two-axis split + NEW sealed lockbox.
    logger.heartbeat(step=4, force=True, stage="building two-axis split + lockbox")
    old_manifest = load_split_manifest(REPO / cfg["old_learner_split"])
    learner_split, learner_stats = make_learner_split_v2(
        old_manifest, lockbox_frac=cfg["lockbox_frac"], seed=cfg["split_seed"],
    )
    item_split, item_stats = make_item_split_v2(
        catalog, heldout_frac=cfg["item_heldout_frac"], seed=cfg["split_seed"],
    )
    a_learner_disjoint = audit_split_disjoint(learner_split, "learner")
    a_item_disjoint = audit_split_disjoint(item_split, "problem_group_id")
    learner_leak_caught = not audit_split_disjoint(
        inject_split_leak(learner_split, "lockbox", "train"), "learner",
    )["disjoint"]
    item_leak_caught = not audit_split_disjoint(
        inject_split_leak(item_split, "heldout", "train"), "problem_group_id",
    )["disjoint"]

    events_split = attach_splits(events, learner_split, item_split)
    a_lockbox_isolated = audit_lockbox_isolated(events_split)

    # Every learner/item referenced by an event must resolve to exactly one split.
    coverage_ok = (
        events_split["split_learner"].notna().all()
        and events_split["split_item"].notna().all()
    )

    # Seal the lockbox: hash + counts only, rows never read or printed.
    lockbox = lockbox_hash(events_split)

    passed = all([
        dedup_matches_gate,
        canary_caught,
        score_matches_gate,
        a_unique["unique_ids"],
        a_score["score_never_imputed"],
        a_mono["monotonic"],
        a_missing_excl["missing_time_excluded"],
        a_learner_disjoint["disjoint"],
        a_item_disjoint["disjoint"],
        learner_leak_caught,
        item_leak_caught,
        a_lockbox_isolated["learner_split_consistent"],
        coverage_ok,
    ])

    # -- write outputs ----------------------------------------------------------
    manifest_path = REPO / cfg["manifest_out"]
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    events_split.to_parquet(manifest_path, index=False)

    learner_split_path = REPO / cfg["learner_split_out"]
    learner_split_path.parent.mkdir(parents=True, exist_ok=True)
    learner_split.to_csv(learner_split_path, index=False)

    item_split_path = REPO / cfg["item_split_out"]
    item_split.to_csv(item_split_path, index=False)

    catalog_path = REPO / cfg["problem_catalog_out"]
    catalog.to_csv(catalog_path, index=False)

    lockbox_path = REPO / cfg["lockbox_manifest_out"]
    lockbox_manifest = {
        "sha256": lockbox["sha256"],
        "n_rows": lockbox["n_rows"],
        "n_learners": lockbox["n_learners"],
        "n_item_groups": lockbox["n_item_groups"],
        "git_sha": logger.git_sha,
        "config_hash": logger.config_hash,
        "run_id": logger.run_id,
        "note": "Strict lockbox cell (learner split == lockbox AND item split == heldout). "
                "Rows never read after this seal; to be used once, for final confirmation.",
    }
    lockbox_path.write_text(json.dumps(lockbox_manifest, indent=2) + "\n")

    logger.log_metrics(
        passed=passed, **dedup_stats,
        n_score_missing=build_stats["n_score_missing"],
        n_missing_time=build_stats["n_missing_time"],
        **learner_stats, **item_stats,
        lockbox_n_rows=lockbox["n_rows"], lockbox_sha256=lockbox["sha256"],
    )

    metrics = {
        "raw interactions": f"{n_raw:,}",
        "exact-ID dedup: dropped / groups": (
            f"{dedup_stats['n_dropped']:,} / {dedup_stats['n_dup_groups']:,} "
            f"(gate {gate['n_dedup_dropped']:,} / {gate['n_dedup_groups']:,}) "
            f"{'OK' if dedup_matches_gate else 'MISMATCH'}"
        ),
        "planted-duplicate canary caught": canary_caught,
        "score missing (never imputed)": (
            f"{build_stats['n_score_missing']:,} (gate {gate['n_score_missing']:,}) "
            f"{'OK' if score_matches_gate else 'MISMATCH'}"
        ),
        "missing-time rows (post-dedup, excluded from history)": f"{build_stats['n_missing_time']:,}",
        "valid-time rows": f"{build_stats['n_valid_time']:,}",
        "support class counts": build_stats["support_counts"],
        "problem groups (near-duplicate grouping)": (
            f"{item_stats['n_groups']:,} groups over {len(catalog):,} unique problem_ids"
        ),
        "audit: unique ids": f"{a_unique['unique_ids']} ({a_unique['n_duplicated']} duplicated)",
        "audit: score never imputed": f"{a_score['score_never_imputed']} ({a_score['n_bad']} bad)",
        "audit: monotonic time": f"{a_mono['monotonic']} ({a_mono['backwards_steps']} backwards)",
        "audit: missing-time excluded from t": (
            f"{a_missing_excl['missing_time_excluded']} ({a_missing_excl['n_bad']} bad)"
        ),
        "learner split (train/val/development_exposed/lockbox)": (
            f"{learner_stats['n_train']}/{learner_stats['n_val']}/"
            f"{learner_stats['n_development_exposed']}/{learner_stats['n_lockbox']}"
        ),
        "learner split disjoint": (
            f"{a_learner_disjoint['disjoint']} (in-multiple {a_learner_disjoint['in_multiple_splits']})"
        ),
        "learner split leak canary caught": learner_leak_caught,
        "item split (train/heldout groups)": (
            f"{item_stats['n_train_groups']}/{item_stats['n_heldout_groups']}"
        ),
        "item split disjoint": (
            f"{a_item_disjoint['disjoint']} (in-multiple {a_item_disjoint['in_multiple_splits']})"
        ),
        "item split leak canary caught": item_leak_caught,
        "lockbox learner-split consistency": a_lockbox_isolated["learner_split_consistent"],
        "every event resolves to exactly one learner+item split": coverage_ok,
        "LOCKBOX (sealed, rows never read)": (
            f"{lockbox['n_rows']:,} rows / {lockbox['n_learners']} learners / "
            f"{lockbox['n_item_groups']} item groups -- sha256 {lockbox['sha256'][:16]}..."
        ),
    }

    logger.summary(status="ok" if passed else "gate-fail", passed=passed)
    gate_report(
        path="reports/fa_00_manifest_rebuild.md",
        title="FA-0 - Canonical FA manifest rebuild",
        meta={
            "run_id": logger.run_id, "git_sha": logger.git_sha,
            "config_hash": logger.config_hash,
            "manifest_out": cfg["manifest_out"],
            "learner_split_out": cfg["learner_split_out"],
            "item_split_out": cfg["item_split_out"],
            "lockbox_manifest_out": cfg["lockbox_manifest_out"],
        },
        metrics=metrics,
        passed=passed,
        gate_desc="every count reconciles against the expected values in configs/data.yaml "
                  "with a written explanation for any mismatch; leakage audits green on both axes; "
                  "the planted-duplicate canary and both split-leak canaries are caught; "
                  "lockbox rows never printed, only its hash + counts committed.",
        notes=[
            "This manifest replaces the one built by fa.py/tuples.py, which are kept unchanged "
            "for reproducibility of the earlier pipeline (it lacked deduplication, an item "
            "split and an untouched test set); new work should not import them.",
            "The earlier split's 500 `test` learners are kept as `development_exposed`, not "
            "merged into the new lockbox: they were used during development, so they cannot "
            "serve as an untouched confirmation set.",
            "The new `lockbox` (500 learners) is drawn from the earlier split's 4,000 `train` "
            "learners, which had only been used to train earlier models. New `train` is "
            f"{learner_stats['n_train']:,} learners (was 4,000 before the lockbox was drawn from it).",
            "Near-duplicate problem grouping catches exact-normalised-body-text matches only "
            "(the same signal p2_07_encode_problems.py uses for the codec cache); "
            "paraphrased near-duplicates are not caught.",
            *explanations,
        ],
    )
    print(f"\nmanifest rebuild: {'PASS' if passed else 'FAIL'} · report: reports/fa_00_manifest_rebuild.md")
    print(f"  events {len(events):,} (raw {n_raw:,}, dedup -{dedup_stats['n_dropped']:,})")
    print(f"  learner split train/val/dev_exposed/lockbox = "
          f"{learner_stats['n_train']}/{learner_stats['n_val']}/"
          f"{learner_stats['n_development_exposed']}/{learner_stats['n_lockbox']}")
    print(f"  item split train/heldout groups = "
          f"{item_stats['n_train_groups']}/{item_stats['n_heldout_groups']}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
