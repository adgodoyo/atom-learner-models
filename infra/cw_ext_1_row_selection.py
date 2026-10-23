#!/usr/bin/env python3
"""Selects the first-submission rows on the 30 CodeWorkout problems not yet generated, and checks the selection.

The rule is `cw_sim_core.select_first_attempts_on_headline`, reused unchanged with
`headline_ids` set to the 30-problem complement instead of its 13-problem default. The
30-problem list is derived, never typed by hand: every problem with a statement (the full
50-item pool) minus the 13 `HEADLINE_PROBLEM_IDS` minus the 7 `HELD_OUT_ITEMS`; the
arithmetic and both disjointness conditions are asserted.
This module only selects rows and reports counts (no model forward pass);
`cw_ext_1_generate.py` imports `OTHER_30_PROBLEM_IDS` and `select_ext1_rows` from here so
the selection logic is defined once.
Inputs: the CodeWorkout manifest and the codec caches under `cache/cw_03_vectors/`.
Output: row_selection.json (row count, per-problem coverage, overlap checks).
Run:  python infra/cw_ext_1_row_selection.py --out results/cw_ext_1/row_selection.json
Serves: thesis Table 11 (generation).
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
for _p in (REPO / "src", REPO / "infra", REPO / "scripts", REPO):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import cw_sim_core as cw5  # noqa: E402
from atom.data.cw_challenge_split import HELD_OUT_ITEMS  # noqa: E402


class _ManifestArgs:
    """Minimal stand-in for the `args` namespace `cw_sim_core.load_cw5_manifest` expects:
    the seven path fields of the codec caches and hash maps."""
    raw_dir = None
    code_map = str(REPO / "cache" / "cw_03_vectors" / "cw_code_state_hash_map_full.npz")
    diff_map = str(REPO / "cache" / "cw_03_vectors" / "cw_diff_hash_map_full.npz")
    codealone_meta = str(REPO / "cache" / "cw_03_vectors" / "codealone" / "meta.json")
    codealone_states = str(REPO / "cache" / "cw_03_vectors" / "codealone" / "states_mean.npy")
    diff_meta = str(REPO / "cache" / "cw_03_vectors" / "diff" / "meta.json")
    diff_states = str(REPO / "cache" / "cw_03_vectors" / "diff" / "states_mean.npy")


def derive_other_30_problem_ids() -> tuple[int, ...]:
    """Every problem with a statement (the full 50-item pool) minus the 13 headline
    problems minus the 7 held-out problems. Asserts the arithmetic rather than trusting
    it: 50 - 13 - 7 = 30, and both subtracted sets must already be subsets of the full
    50 (a HEADLINE or HELD_OUT id absent from the statement pool would silently produce
    31, not 30, and pass unnoticed without this check)."""
    all_50 = set(cw5.load_problem_statements().keys())
    headline = set(cw5.HEADLINE_PROBLEM_IDS)
    held_out = set(HELD_OUT_ITEMS)
    assert len(all_50) == 50, f"expected 50 problem statements, got {len(all_50)}"
    assert headline <= all_50, f"HEADLINE_PROBLEM_IDS not a subset of the 50-item pool: {headline - all_50}"
    assert held_out <= all_50, f"HELD_OUT_ITEMS not a subset of the 50-item pool: {held_out - all_50}"
    other = all_50 - headline - held_out
    assert len(other) == 30, f"expected 30 remaining problems, got {len(other)}: {sorted(other)}"
    assert other.isdisjoint(headline), "derived 30-problem set overlaps HEADLINE_PROBLEM_IDS"
    assert other.isdisjoint(held_out), "derived 30-problem set overlaps HELD_OUT_ITEMS"
    return tuple(sorted(other))


OTHER_30_PROBLEM_IDS = derive_other_30_problem_ids()


def select_ext1_rows(manifest: dict) -> pd.DataFrame:
    """The actual row-selection call -- `select_first_attempts_on_headline`, unchanged,
    with `headline_ids=OTHER_30_PROBLEM_IDS`. Returns one row per (val learner, one of
    the 30 problems), that learner's earliest attempt by causal index `t`."""
    return cw5.select_first_attempts_on_headline(
        manifest["attempts"], manifest["val_learners"], headline_ids=OTHER_30_PROBLEM_IDS)


def assert_and_report(rows: pd.DataFrame, manifest: dict) -> dict:
    """The pre-registered checks, run BEFORE any generation: exact row count (reported,
    not asserted equal to the expected ~2,808, because `load_cw5_manifest`'s filter chain
    differs from the one that produced the 4,593/1,194/591 figures, so the true count may
    land near but not exactly on 2,808), per-problem coverage over all 30 problems, `has_prior`
    (via `prior_code` all-NaN, since `has_prior` is not a stored column -- it is
    derived at scoring time in `atom.eval.cw_generation` from exactly this condition),
    and disjointness from both frozen lists."""
    n_rows = len(rows)

    per_problem_counts = rows.groupby("problem_id").size().to_dict()
    per_problem_counts = {int(k): int(v) for k, v in per_problem_counts.items()}
    covered = set(per_problem_counts.keys())
    missing_problems = set(OTHER_30_PROBLEM_IDS) - covered
    extra_problems = covered - set(OTHER_30_PROBLEM_IDS)
    assert not extra_problems, f"rows contain problem_id(s) outside the 30-problem list: {extra_problems}"
    assert covered == set(OTHER_30_PROBLEM_IDS), (
        f"not all 30 problems have at least one row -- missing: {sorted(missing_problems)} "
        f"(covered {len(covered)}/30; a missing problem is disclosed, not silently absorbed)")

    has_prior_violations = int(rows["prior_code"].notna().sum())
    assert has_prior_violations == 0, (
        f"{has_prior_violations} row(s) have a non-null prior_code -- these are NOT "
        f"first attempts, the earliest-by-t selection is broken")

    headline_overlap = covered & set(cw5.HEADLINE_PROBLEM_IDS)
    held_out_overlap = covered & set(HELD_OUT_ITEMS)
    assert not headline_overlap, f"selected rows overlap HEADLINE_PROBLEM_IDS: {headline_overlap}"
    assert not held_out_overlap, f"selected rows overlap HELD_OUT_ITEMS: {held_out_overlap}"

    n_unique_learners = int(rows["learner"].nunique())
    n_val_learners_in_manifest = len(manifest["val_learners"])

    val_beliefs_path = REPO / "artifacts" / "cw_ratchet_val_prefix_beliefs.npz"
    npz_learners = None
    val_learner_set_matches_npz = None
    if val_beliefs_path.exists():
        d = np.load(val_beliefs_path, allow_pickle=False)
        npz_learners = set(str(x) for x in np.unique(d["learner"]))
        val_learner_set_matches_npz = (set(manifest["val_learners"]) == npz_learners)

    report = {
        "other_30_problem_ids": list(OTHER_30_PROBLEM_IDS),
        "n_rows": n_rows,
        "expected_approx": 2808,
        "delta_from_expected": n_rows - 2808,
        "n_unique_learners_in_selection": n_unique_learners,
        "n_val_learners_in_manifest": n_val_learners_in_manifest,
        "n_val_learners_in_npz": len(npz_learners) if npz_learners is not None else None,
        "val_learner_set_matches_npz": val_learner_set_matches_npz,
        "per_problem_counts": per_problem_counts,
        "n_problems_covered": len(covered),
        "has_prior_violations": has_prior_violations,
        "headline_overlap": sorted(headline_overlap),
        "held_out_overlap": sorted(held_out_overlap),
        "all_assertions_passed": True,
    }
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "results" / "cw_ext_1" / "row_selection.json"))
    args = ap.parse_args()

    print(f"[cw-ext-1-rows] derived 30-problem list: {OTHER_30_PROBLEM_IDS}", flush=True)
    print("[cw-ext-1-rows] loading CodeWorkout manifest (S19+F19 train+val, HELD_OUT_ITEMS excluded)...", flush=True)
    manifest = cw5.load_cw5_manifest(_ManifestArgs())
    print(f"[cw-ext-1-rows] manifest: {len(manifest['attempts'])} attempts, "
         f"{len(manifest['val_learners'])} val learners, "
         f"{manifest['n_problems_after_holdout']} problems after holdout", flush=True)

    rows = select_ext1_rows(manifest)
    report = assert_and_report(rows, manifest)
    print(f"[cw-ext-1-rows] ALL ASSERTIONS PASSED: {json.dumps(report, indent=2)}", flush=True)

    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    print(f"[cw-ext-1-rows] report written -> {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
