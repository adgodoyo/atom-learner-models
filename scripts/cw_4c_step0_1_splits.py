#!/usr/bin/env python
"""Adopt the release's official learner split and freeze the 7-item CodeWorkout holdout.

The learner split defines every training/val/lockbox population and the item holdout
defines which problems the CodeWorkout models may train on, so this runs before any of them.
Steps:
  1. Load S19 + F19 raw attempts to get the real learner population per release
     (MainTable-derived; `Subject.csv` under-covers S19: 413 vs 372 rows).
  2. Read the three Data Challenge subject-id files and gate the partition
     (`cw_challenge_split.verify_challenge_partition`): hash match, pairwise disjoint,
     union equals the real F19 population.
  3. Train pool = S19 + F19-train + 16 unassigned (796); lockbox = 123 F19-test. Carve
     val stratified by semester (target 110) and write `splits/cw_learner_split.csv`.
  4. Freeze the 7-item holdout, check it is disjoint from the TIKTOC-17 and present in the
     shared 50-problem catalogue, write `splits/cw_s19_item_split.csv`.
  5. Measure median events after the last held-out-item attempt (the items should sit at
     the tail of a learner's history, not in the middle).
Output: the two split files and `reports/cw_4c_step0_1_splits.md`.
Run: `python scripts/cw_4c_step0_1_splits.py`. Serves Chapter 4 (data and splits).
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from atom.data.cw_challenge_split import (
    HELD_OUT_ITEMS,
    read_challenge_subject_ids,
    stratified_val_carve,
    verify_challenge_partition,
    verify_item_holdout,
)
from atom.data.cw_events import build_attempts, load_raw_tables
from atom.data.cw_sessions import tiktoc_stratum_ids
from atom.eval.report import gate_report
from atom.utils.logging import RunLogger

REPO = pathlib.Path(__file__).resolve().parents[1]
RAW_DIR = REPO.parent / "Datasets" / "CodeWorkout_CMU"
VAL_FRACTION_TARGET = 110 / 796  # ~13.82%, i.e. ~110 of the 796 train-pool learners


def median_events_after_last_held_out(attempts: pd.DataFrame, held_out: set[int]) -> dict:
    """Per learner (pooled across releases): count of that learner's own events strictly
    after their last held-out-item attempt, chronologically. Learners who never touch a
    held-out item are excluded (nothing to measure). Sanity check that these items sit
    at the tail of a typical learner's history."""
    a = attempts.sort_values(["learner", "timestamp"], kind="mergesort").reset_index(drop=True)
    a["_pos"] = a.groupby("learner", sort=False).cumcount()
    a["_n_total"] = a.groupby("learner")["_pos"].transform("size")
    held_mask = a["problem_id"].isin(held_out)
    last_held_pos = a.loc[held_mask].groupby("learner")["_pos"].max()
    n_total = a.groupby("learner")["_pos"].size()
    after = (n_total.reindex(last_held_pos.index) - 1 - last_held_pos).astype(int)
    return {
        "n_learners_touching_held_out": int(len(after)),
        "median_events_after": float(after.median()) if len(after) else float("nan"),
        "mean_events_after": float(after.mean()) if len(after) else float("nan"),
        "frac_zero_events_after": float((after == 0).mean()) if len(after) else float("nan"),
    }


def main() -> int:
    logger = RunLogger(config={"task": "cw_4c_step0_1_splits"}, data_version="CSEDM-S19+F19-challenge")
    logger.heartbeat(step=0, force=True, stage="loading S19 + F19 raw attempts")

    frames = {}
    for release in ["S19", "F19"]:
        main_table, _ = load_raw_tables(release, base_dir=RAW_DIR)
        frames[release] = build_attempts(main_table)
    s19_attempts, f19_attempts = frames["S19"], frames["F19"]
    s19_learners = set(s19_attempts["learner"])
    f19_learners = set(f19_attempts["learner"])

    logger.heartbeat(step=1, force=True, stage="step 0: challenge partition gate", n_s19=len(s19_learners), n_f19=len(f19_learners))

    # -- STEP 0: adopt the release's official split --------------------------------
    subject_ids = read_challenge_subject_ids(RAW_DIR)
    gate0 = verify_challenge_partition(subject_ids, f19_learners)
    s19_f19_overlap = s19_learners & f19_learners
    gate0_pass = (
        gate0["train_matched"] == gate0["n_train"] == 367
        and gate0["test_matched"] == gate0["n_test"] == 123
        and gate0["unassigned_matched"] == gate0["n_unassigned"] == 16
        and gate0["pairwise_disjoint"]
        and gate0["union_equals_f19_learners"]
        and len(s19_f19_overlap) == 0
    )

    train_pool_s19 = s19_learners
    train_pool_f19 = gate0["train_hashes"] | gate0["unassigned_hashes"]
    lockbox = gate0["test_hashes"]
    train_pool = train_pool_s19 | train_pool_f19
    n_train_pool, n_lockbox = len(train_pool), len(lockbox)

    logger.heartbeat(step=2, force=True, stage="step 0: val carve", n_train_pool=n_train_pool, n_lockbox=n_lockbox)

    val_assignment = stratified_val_carve(
        {"S19": sorted(train_pool_s19), "F19": sorted(train_pool_f19)},
        val_fraction=VAL_FRACTION_TARGET, seed=0,
    )
    n_val = sum(1 for v in val_assignment.values() if v == "val")
    n_train = sum(1 for v in val_assignment.values() if v == "train")

    rows = []
    for learner in sorted(train_pool_s19):
        rows.append({"learner": learner, "release": "S19", "challenge_role": "s19",
                     "split": val_assignment[learner]})
    for learner in sorted(gate0["train_hashes"]):
        rows.append({"learner": learner, "release": "F19", "challenge_role": "f19_train",
                     "split": val_assignment[learner]})
    for learner in sorted(gate0["unassigned_hashes"]):
        rows.append({"learner": learner, "release": "F19", "challenge_role": "f19_unassigned",
                     "split": val_assignment[learner]})
    for learner in sorted(lockbox):
        rows.append({"learner": learner, "release": "F19", "challenge_role": "f19_test_lockbox",
                     "split": "lockbox"})
    split_df = pd.DataFrame(rows)
    split_path = REPO / "splits" / "cw_learner_split.csv"
    split_df.to_csv(split_path, index=False)
    n_total_rows = len(split_df)
    n_everyone = len(s19_learners | f19_learners)

    logger.heartbeat(step=3, force=True, stage="step 1: item holdout gate")

    # -- STEP 1: freeze the 7-item holdout -------------------------------------------
    catalog_s19 = set(s19_attempts["problem_id"].unique())
    catalog_f19 = set(f19_attempts["problem_id"].unique())
    catalog_union = catalog_s19 | catalog_f19
    tiktoc_ids = tiktoc_stratum_ids()
    gate1 = verify_item_holdout(HELD_OUT_ITEMS, tiktoc_ids, catalog_union)
    gate1_pass = gate1["disjoint_from_tiktoc_17"] and gate1["all_in_catalog"] and catalog_s19 == catalog_f19

    item_rows = [{"problem_id": pid, "item_split": "heldout" if pid in HELD_OUT_ITEMS else "train"}
                for pid in sorted(catalog_union)]
    item_split_df = pd.DataFrame(item_rows)
    item_split_path = REPO / "splits" / "cw_s19_item_split.csv"
    item_split_df.to_csv(item_split_path, index=False)

    logger.heartbeat(step=4, force=True, stage="free measurement: events after last held-out item")

    all_attempts = pd.concat([s19_attempts, f19_attempts], ignore_index=True)
    tail_stats = median_events_after_last_held_out(all_attempts, set(HELD_OUT_ITEMS))

    passed = bool(gate0_pass and gate1_pass and n_val == 110)

    meta = {
        "run_id": logger.run_id, "git_sha": logger.git_sha, "config_hash": logger.config_hash,
        "n_s19_learners": len(s19_learners), "n_f19_learners": len(f19_learners),
        "n_train_pool": n_train_pool, "n_lockbox": n_lockbox, "n_val_target": 110,
        "n_problems_catalog": len(catalog_union),
    }
    metrics = {
        "step0_challenge_partition_gate": {
            "n_train": gate0["n_train"], "n_test": gate0["n_test"], "n_unassigned": gate0["n_unassigned"],
            "train_matched": gate0["train_matched"], "test_matched": gate0["test_matched"],
            "unassigned_matched": gate0["unassigned_matched"], "pairwise_disjoint": gate0["pairwise_disjoint"],
            "union_equals_f19_learners": gate0["union_equals_f19_learners"],
            "s19_f19_learner_overlap": len(s19_f19_overlap),
        },
        "step0_final_split": {"train_pool": n_train_pool, "train": n_train, "val": n_val,
                              "lockbox": n_lockbox, "n_rows_written": n_total_rows,
                              "everyone_accounted_for": n_total_rows == n_everyone},
        "step1_item_holdout_gate": gate1,
        "held_out_items": sorted(HELD_OUT_ITEMS),
        "tail_measurement": tail_stats,
    }
    notes = [
        f"**Step 0 verified, not assumed:** `hash_learner(SubjectID)` recovers {gate0['train_matched']}/367 "
        f"train, {gate0['test_matched']}/123 test, {gate0['unassigned_matched']}/16 unassigned against the "
        "REAL F19 learner population (MainTable-derived, 506 learners) -- the three sets are pairwise "
        "disjoint and their union is exactly that population. S19/F19 learner overlap is "
        f"{len(s19_f19_overlap)} (different cohorts, as expected).",
        f"**Val carve:** target fraction {VAL_FRACTION_TARGET:.5f} (110/796), applied per-stratum "
        f"(S19 {len(train_pool_s19)}, F19 {len(train_pool_f19)}), seed=0, deterministic "
        f"`random.Random.sample`. Landed at exactly {n_val} (target 110).",
        f"**`splits/cw_s19_learner_split.csv` (old, S19-only 330/83) is SUPERSEDED, left in place** "
        "for reproducibility of numbers computed on it. Current artifact: `splits/cw_learner_split.csv`.",
        f"**Step 1:** held-out {sorted(HELD_OUT_ITEMS)}, disjoint from the TIKTOC-17 "
        f"({'confirmed' if gate1['disjoint_from_tiktoc_17'] else 'VIOLATED'}), all present in the shared "
        f"50-problem catalog (S19 catalog == F19 catalog: {catalog_s19 == catalog_f19}). "
        "**`splits/cw_item_split.csv` (older holdout {5,12,13,38,44,45,71}, written by cw_2_sessions.py) is "
        "superseded** -- a different holdout set, left in place but must not be used. Current "
        "artifact: `splits/cw_s19_item_split.csv`.",
        f"**Free tail measurement:** of {tail_stats['n_learners_touching_held_out']} learners who touch a "
        f"held-out item at all, median events after their LAST held-out attempt is "
        f"{tail_stats['median_events_after']:.1f} ({tail_stats['frac_zero_events_after']:.1%} have exactly "
        "zero) -- consistent with these items sitting at the tail, not mid-history, for "
        "the typical learner.",
    ]
    text = gate_report(
        REPO / "reports" / "cw_4c_step0_1_splits.md",
        "CodeWorkout learner split (the release's official split) and the frozen 7-item holdout",
        meta, metrics, passed,
        "Step 0: hash_learner recovers the challenge's train/test/unassigned partition exactly "
        "(367/367, 123/123, 16/16), pairwise disjoint, union == real F19 population, val lands at 110. "
        "Step 1: the 7-item holdout is disjoint from the TIKTOC-17 and present in both releases' catalog.",
        notes=notes,
    )
    print(text)
    logger.summary(status="ok" if passed else "gate-fail", n_train_pool=n_train_pool, n_lockbox=n_lockbox, n_val=n_val)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
