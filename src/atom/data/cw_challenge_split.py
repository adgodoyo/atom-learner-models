"""Adopt the release's official learner split and freeze the 7-item holdout.

The item holdout must be fixed before any CodeWorkout model is trained, because it
defines which problems training is permitted to see.

**Step 0, the split.** The CodeWorkout release ships its own train/test
division (`F19_Release_Train` = 367 subjects, `F19_Release_Test` = 123 subjects with
`Test_Solution` withheld 11 months), which is used here rather than an ad hoc
semester split. The split is: **train-pool = S19 + F19-train + 16 unassigned subjects (present in
`F19_Release_All` but in neither 2021 challenge file) = 796**, **lockbox = the 123
F19-test subjects**, val carved from the 796 stratified by semester (S19 vs F19). The
mapping is `hash_learner(SubjectID)` (`atom.data.tuples.hash_learner`), the same hash
every CW attempts table already carries as `learner` -- this module's job is verifying
that hash actually recovers the challenge's own partition before trusting it for
anything, not assuming it.

**Step 1, the item holdout.** Fixed in advance, not drawn here: problems
`{51, 56, 57, 64, 70, 112, 118}` (`HELD_OUT_ITEMS`), disjoint from the 17-problem TIKTOC
stratum by construction -- verified against `cw_sessions.tiktoc_stratum_ids()`, never
duplicated. Held out from targets AND histories (a target-only removal leaks: the
held-out outcome would still enter as history when predicting later events) -- enforcing
that removal is the training script's job, not this module's; this module only freezes
*which* items and verifies the freeze is sound (disjoint from TIKTOC-17, present in both
releases' catalog).
"""

from __future__ import annotations

import random
import zipfile
from pathlib import Path

import pandas as pd

from atom.data.tuples import hash_learner

CHALLENGE_ZIPS = {
    "f19_all": ("F19_Release_All_05_23_22.zip", "All/Data/LinkTables/Subject.csv"),
    "f19_train": ("F19_Release_Train_06-28-21.zip", "Train/Data/LinkTables/Subject.csv"),
    "f19_test": ("F19_Release_Test_06-28-21.zip", "Test/Data/LinkTables/Subject.csv"),
}

# Frozen set. Disjoint from the TIKTOC-17 by construction (verified, not assumed -- see
# `verify_item_holdout`): 51 stands in for 71, which is one of the TIKTOC 17.
HELD_OUT_ITEMS: tuple[int, ...] = (51, 56, 57, 64, 70, 112, 118)


def read_challenge_subject_ids(raw_dir) -> dict[str, set[str]]:
    """Raw (unhashed) `SubjectID` sets straight out of the three challenge zips."""
    out: dict[str, set[str]] = {}
    for key, (zip_name, path) in CHALLENGE_ZIPS.items():
        with zipfile.ZipFile(Path(raw_dir) / zip_name) as z:
            with z.open(path) as f:
                out[key] = set(pd.read_csv(f, dtype=str)["SubjectID"])
    return out


def verify_challenge_partition(subject_ids: dict[str, set[str]], f19_learners: set[str]) -> dict:
    """The step-0 gate. Hashes the three challenge subject-id sets and checks: each is a
    subset of the REAL F19 learner population (from `cw_events.build_attempts`, not
    `Subject.csv` -- `Subject.csv` can under-cover the true population, verified true for
    S19), the three are pairwise disjoint, and their union recovers the full F19 learner
    set exactly (nobody is double-counted or dropped).
    """
    train_h = {hash_learner(s) for s in subject_ids["f19_train"]}
    test_h = {hash_learner(s) for s in subject_ids["f19_test"]}
    unassigned_ids = subject_ids["f19_all"] - subject_ids["f19_train"] - subject_ids["f19_test"]
    unassigned_h = {hash_learner(s) for s in unassigned_ids}
    return {
        "n_train": len(train_h), "n_test": len(test_h), "n_unassigned": len(unassigned_h),
        "train_matched": len(train_h & f19_learners), "test_matched": len(test_h & f19_learners),
        "unassigned_matched": len(unassigned_h & f19_learners),
        "pairwise_disjoint": (len(train_h & test_h) == 0 and len(train_h & unassigned_h) == 0
                              and len(test_h & unassigned_h) == 0),
        "union_equals_f19_learners": (train_h | test_h | unassigned_h) == f19_learners,
        "train_hashes": train_h, "test_hashes": test_h, "unassigned_hashes": unassigned_h,
    }


def stratified_val_carve(train_pool_by_stratum: dict[str, list[str]], val_fraction: float,
                         seed: int = 0) -> dict[str, str]:
    """Deterministic per-stratum val carve: within each stratum, `round(n * val_fraction)`
    learners (sorted, then a seeded `random.Random` sample -- reproducible, not order-
    dependent on however the caller's set iterated) go to `"val"`, the rest to `"train"`.
    Stratifying separately (rather than pooling then sampling) is what keeps val
    proportionate to each semester instead of val drifting toward whichever stratum is
    larger. Returns `learner -> "train"|"val"` for every learner across every stratum.
    """
    out: dict[str, str] = {}
    for stratum, learners in train_pool_by_stratum.items():
        learners_sorted = sorted(set(learners))
        n_val = round(len(learners_sorted) * val_fraction)
        rng = random.Random(seed)
        val_set = set(rng.sample(learners_sorted, n_val))
        for learner in learners_sorted:
            out[learner] = "val" if learner in val_set else "train"
    return out


def verify_item_holdout(held_out_items: tuple[int, ...], tiktoc_ids: list[int],
                        catalog_ids: set[int]) -> dict:
    """The step-1 gate: the held-out set is disjoint from the TIKTOC-17 stratum (a
    standing rule -- one held-out problem cannot both support an unseen-item claim and
    anchor the TIKTOC-comparable production number) and every held-out id is a real
    problem in the shared 50-problem catalog."""
    held = set(held_out_items)
    tiktoc = set(tiktoc_ids)
    return {
        "n_held_out": len(held),
        "disjoint_from_tiktoc_17": held.isdisjoint(tiktoc),
        "overlap_with_tiktoc_17": sorted(held & tiktoc),
        "all_in_catalog": held.issubset(catalog_ids),
        "missing_from_catalog": sorted(held - catalog_ids),
    }
