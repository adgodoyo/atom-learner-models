#!/usr/bin/env python3
"""Per-attempt test-vector labels for the CodeWorkout generative simulator.

For every individual attempt (S19 and F19 pooled) on the 13 join-verified TIKTOC
problems (`cw_5_functional_equivalence.HEADLINE_PROBLEM_IDS`; the other 4 problems of the
nominal 17-problem set have known join issues and are excluded), this script runs
that attempt's own code through TIKTOC's compile-and-run harness
(`cw_5_functional_equivalence.score_submission`, with a 30 s timeout) to obtain the
per-test-case pass/fail vector. Each attempt is labelled independently: an early attempt
rarely passes the same tests as the learner's later corrected one, so one label per
learner and problem would supervise early attempts against the wrong target. Rows are
keyed `(learner, problem_id, code_state_id)`, because a bare `code_state_id` is not
guaranteed unique across the pooled releases.

Attempts that fail to compile are not executed (every test case is False). If more
compile-success attempts exist than `--max-executed` (default 4000), a seeded random
sample is executed; the output records `n_compile_success_total` and `n_executed` so the
cap is visible. The 30 s timeout (longer than the harness's 15 s default) avoids biasing
failures towards early attempts, which more often contain infinite loops or blocking
input reads.

Output: `artifacts/cw_sim_test_vector_labels.npz` (learner, problem_id, code_state_id,
release, compiled, test_vector padded to MAX_TEST_CASES=32 with a validity mask,
n_cases), read by the pilot and full simulator training runs.
Run: `python scripts/cw_sim_test_vector_labels.py [--workers N] [--max-executed N]`.
Serves Table 11 (test-vector head of the CodeWorkout generative simulator).
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import pathlib
import sys
import time

import numpy as np
import pandas as pd

REPO = pathlib.Path(__file__).resolve().parents[1]
for p in (REPO / "src", REPO):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from atom.data.cw_events import load_raw_tables, build_attempts  # noqa: E402
from cw_5_functional_equivalence import (  # noqa: E402
    HEADLINE_PROBLEM_IDS, load_test_cases, score_submission,
)

MAX_TEST_CASES = 32


def build_all_attempts_pool() -> pd.DataFrame:
    """Every attempt (not just each learner's last) on the 13 headline problems, S19+F19
    pooled: the per-attempt label needs every attempt's own code, not one attempt's
    label copied across the rest."""
    frames = []
    for release in ("S19", "F19"):
        main, code = load_raw_tables(release, include_code=True)
        attempts = build_attempts(main, code)
        sub = attempts[attempts["problem_id"].astype(int).isin(HEADLINE_PROBLEM_IDS)].copy()
        sub["release"] = release
        frames.append(sub)
    pooled = pd.concat(frames, ignore_index=True)
    return pooled.reset_index(drop=True)


def _score_one(args):
    idx, code_text, pid, test_cases = args
    r = score_submission(code_text, test_cases, timeout=30.0)
    return idx, r


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-executed", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=2)  # laptop default; raise on a larger machine
    ap.add_argument("--out", default=str(REPO / "artifacts" / "cw_sim_test_vector_labels.npz"))
    ns = ap.parse_args()

    pool = build_all_attempts_pool()
    print(f"[tv-labels] all-attempts pool (per-attempt): {len(pool)} rows "
         f"({(pool['compile_result'] == 'Success').sum()} compile-success)")

    compiled_mask = pool["compile_result"] == "Success"
    n_compile_success_total = int(compiled_mask.sum())
    to_execute = pool[compiled_mask].reset_index(drop=True)
    if len(to_execute) > ns.max_executed:
        rng = np.random.default_rng(ns.seed)
        keep = rng.choice(len(to_execute), size=ns.max_executed, replace=False)
        to_execute = to_execute.iloc[sorted(keep)].reset_index(drop=True)
    n_executed = len(to_execute)
    print(f"[tv-labels] executing {n_executed}/{n_compile_success_total} compile-success rows "
         f"(cap={ns.max_executed}, seed={ns.seed})")

    test_cases_cache = {pid: load_test_cases(pid) for pid in HEADLINE_PROBLEM_IDS}

    jobs = [(i, row["code"], int(row["problem_id"]), test_cases_cache[int(row["problem_id"])])
           for i, row in to_execute.iterrows()]

    results = {}
    t0 = time.time()
    with cf.ProcessPoolExecutor(max_workers=ns.workers) as ex:
        for n_done, (idx, r) in enumerate(ex.map(_score_one, jobs), start=1):
            results[idx] = r
            if n_done % 200 == 0:
                elapsed = time.time() - t0
                print(f"[tv-labels] {n_done}/{n_executed} scored, {elapsed:.0f}s elapsed, "
                     f"{elapsed / n_done:.3f} s/row")

    learners, problem_ids, code_state_ids, releases, compiled, test_vectors, valid_masks, n_cases_arr = (
        [], [], [], [], [], [], [], [])

    for i, row in to_execute.iterrows():
        pid = int(row["problem_id"])
        r = results[i]
        n_cases = len(test_cases_cache[pid])
        tv = np.zeros(MAX_TEST_CASES, dtype=bool)
        vm = np.zeros(MAX_TEST_CASES, dtype=bool)
        vm[:n_cases] = True
        tv[:n_cases] = r["test_vector"]
        learners.append(str(row["learner"])); problem_ids.append(pid)
        code_state_ids.append(str(row["code_state_id"]))
        releases.append(str(row["release"])); compiled.append(bool(r["compiled"]))
        test_vectors.append(tv); valid_masks.append(vm); n_cases_arr.append(n_cases)

    uncompiled = pool[~compiled_mask]
    for _, row in uncompiled.iterrows():
        pid = int(row["problem_id"])
        n_cases = len(test_cases_cache[pid])
        tv = np.zeros(MAX_TEST_CASES, dtype=bool)
        vm = np.zeros(MAX_TEST_CASES, dtype=bool)
        vm[:n_cases] = True
        learners.append(str(row["learner"])); problem_ids.append(pid)
        code_state_ids.append(str(row["code_state_id"]))
        releases.append(str(row["release"])); compiled.append(False)
        test_vectors.append(tv); valid_masks.append(vm); n_cases_arr.append(n_cases)

    out_path = pathlib.Path(ns.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        learner=np.asarray(learners), problem_id=np.asarray(problem_ids, dtype=np.int64),
        code_state_id=np.asarray(code_state_ids),
        release=np.asarray(releases), compiled=np.asarray(compiled, dtype=bool),
        test_vector=np.stack(test_vectors), valid_mask=np.stack(valid_masks),
        n_cases=np.asarray(n_cases_arr, dtype=np.int64),
        max_test_cases=np.asarray(MAX_TEST_CASES),
        headline_problem_ids=np.asarray(HEADLINE_PROBLEM_IDS, dtype=np.int64),
        n_compile_success_total=np.asarray(n_compile_success_total), n_executed=np.asarray(n_executed),
        n_uncompiled=np.asarray(len(uncompiled)),
        label_construct=np.asarray("per_attempt"),  # one label per attempt
    )
    print(f"[tv-labels] wrote {out_path}: {len(learners)} rows "
         f"({n_executed} executed, {len(uncompiled)} compile failures labelled without execution, "
         f"cap disclosed: {n_executed}/{n_compile_success_total} of compile-success executed)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
