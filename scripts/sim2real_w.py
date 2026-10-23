#!/usr/bin/env python3
"""Builds and checks the learner-specific menus for the policy trained only inside the simulator
(§8.3, Figure 11), as a generation-free, fail-closed smoke check.

The pre-registered design gives each of the 543 fit learners of `rl_2.py` a learner-specific,
without-replacement menu of real candidate episodes drawn from the learner's own history, and a
warm start (a real chronological prefix), rather than an empty prefix and one global menu. This
module builds that partition before any generation model is loaded and proves that the
registered corpus can exist; if the data cannot satisfy the design it exits before any licensed
text is generated or any GPU is used. Menu size is a cap, not a requirement: a learner receives
``min(16, |candidate_pool|)`` distinct own items. Starts are sampled uniformly from the feasible
members of ``{3,10}``; start 3 is the only feasible choice for learners with no start-10
candidate. This keeps all 543 fit learners without fabricating duplicate or cross-learner actions.
Output: ``--out`` (default results/sim2real_w_smoke/structural_audit.json).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pathlib
import sys
from collections import Counter
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
for p in (REPO / "src", REPO / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from atom.data.cw_manifest import load_attempts_with_episodes  # noqa: E402

W_SEED = 20260904
FIT_LEARNERS = 543
STARTS = (3, 10)
MENU_SIZE = 16
TARGET_N = 5
N_CLASSES = 4


def stable_seed(*parts: Any, seed: int = W_SEED) -> int:
    """Derive a reproducible per-learner seed without depending on row order."""

    payload = "|".join(map(str, (*parts, seed))).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def read_split(path: pathlib.Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = csv.DictReader(handle)
        out = {str(row["learner"]): str(row["split"]) for row in rows}
    if not out:
        raise ValueError(f"empty learner split: {path}")
    return out


def real_event_rows(raw_dir: pathlib.Path) -> dict[str, list[dict[str, Any]]]:
    """Return the same chronological episode identity used by the CW scenario code.

    Only identifiers are loaded for the structural gate; no source code text is
    retained.  A later generation stage may attach codecs and summaries to these
    rows, but it must not alter the partition.
    """

    attempts = load_attempts_with_episodes(raw_dir)
    episodes = (
        attempts.sort_values(["learner", "timestamp"], kind="mergesort")
        .groupby("episode_id", sort=False)
        .agg(
            learner=("learner", "first"),
            problem_id=("problem_id", "first"),
            start_timestamp=("timestamp", "min"),
        )
        .reset_index()
    )
    out: dict[str, list[dict[str, Any]]] = {}
    for _, row in episodes.sort_values(
        ["learner", "start_timestamp"], kind="mergesort"
    ).iterrows():
        learner = str(row["learner"])
        out.setdefault(learner, []).append(
            {
                "order": len(out.get(learner, [])),
                "episode_id": str(row["episode_id"]),
                "problem_id": int(row["problem_id"]),
            }
        )
    return out


def rl2_fit_ids(
    events: Mapping[str, Sequence[Mapping[str, Any]]],
    split: Mapping[str, str],
    expected: int = FIT_LEARNERS,
) -> list[str]:
    """Reproduce the sorted train-start-0 80/20 fit partition of rl_2.py by value."""

    train = sorted(
        learner
        for learner, rows in events.items()
        if split.get(str(learner)) == "train" and len(rows) >= 2
    )
    if len(train) < expected:
        raise AssertionError(f"rl_2 fit partition has {len(train)} eligible learners, expected >= {expected}")
    return train[:expected]


def candidate_pool(
    all_events: Sequence[Mapping[str, Any]], start: int, target_n: int = TARGET_N
) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    """Apply the real harness partition exactly: prefix / candidates / targets."""

    if len(all_events) < start + 2:
        raise ValueError(f"history has {len(all_events)} events; start={start} is unavailable")
    n_target = min(target_n, len(all_events) - start - 1)
    targets = list(all_events[-n_target:])
    candidates = list(all_events[start:-n_target])
    prefix = list(all_events[:start])
    return prefix, candidates, targets


def draw_episode(
    learner: str,
    events: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    seed: int = W_SEED,
    starts: Sequence[int] = STARTS,
    menu_size: int = MENU_SIZE,
) -> dict[str, Any]:
    """Draw one warm start and a distinct up-to-16 own-candidate menu.

    Sampling is intentionally without replacement.  Duplicate menu items would
    make the ``(learner,item,class)`` corpus key non-unique.  If start 10 has no
    usable candidates, start 3 is selected from the feasible-start set; this is
    the registered all-learner amendment, not a post-hoc data-dependent retry.
    """

    rows = list(events[str(learner)])
    feasible: list[int] = []
    for candidate_start in starts:
        try:
            _, candidate_pool_rows, _ = candidate_pool(rows, int(candidate_start))
        except ValueError:
            continue
        if candidate_pool_rows:
            feasible.append(int(candidate_start))
    if not feasible:
        raise ValueError(f"no feasible warm start for learner={learner}")
    start_rng = np.random.default_rng(stable_seed("start", learner, seed=seed))
    start = int(start_rng.choice(np.asarray(feasible, dtype=int)))
    prefix, pool, targets = candidate_pool(rows, start)
    n_menu = min(menu_size, len(pool))
    if n_menu < 1:
        raise ValueError(f"empty candidate pool for learner={learner}, start={start}")
    menu_rng = np.random.default_rng(stable_seed("menu", learner, start, seed=seed))
    idx = menu_rng.choice(len(pool), size=n_menu, replace=False)
    candidates = [dict(pool[int(i)]) for i in idx]
    prefix_ids = {str(x["episode_id"]) for x in prefix}
    candidate_ids = {str(x["episode_id"]) for x in candidates}
    target_ids = {str(x["episode_id"]) for x in targets}
    if prefix_ids & candidate_ids or prefix_ids & target_ids or candidate_ids & target_ids:
        raise AssertionError(f"prefix ∩ candidates ∩ targets != ∅ for learner={learner}")
    own_problem_ids = {int(x["problem_id"]) for x in pool}
    if not all(int(x["problem_id"]) in own_problem_ids for x in candidates):
        raise AssertionError(f"non-own candidate item for learner={learner}")
    return {
        "learner": str(learner),
        "start": start,
        "prefix": prefix,
        "candidates": candidates,
        "targets": targets,
        "candidate_pool_size": len(pool),
        "menu_size": len(candidates),
        "feasible_starts": feasible,
    }


def structural_audit(
    events: Mapping[str, Sequence[Mapping[str, Any]]],
    split: Mapping[str, str],
    *,
    smoke_learners: int = 20,
    seed: int = W_SEED,
) -> dict[str, Any]:
    """Run the pre-generation smoke and return a machine-readable audit."""

    fit = rl2_fit_ids(events, split)
    smoke = fit[:smoke_learners]
    if len(smoke) != smoke_learners:
        raise AssertionError(f"smoke requested {smoke_learners}, got {len(smoke)}")
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for learner in smoke:
        try:
            rows.append(draw_episode(learner, events, seed=seed))
        except (AssertionError, ValueError) as exc:
            # Keep the full diagnostic so the report can show the exact binding
            # failure.  Do not proceed to generation after the first failure.
            failures.append({"learner": learner, "error": str(exc)})

    # Also count the complete registered fit set without sampling text.  This
    # distinguishes a smoke-only problem from a full-design impossibility.
    availability: Counter[str] = Counter()
    short_examples: list[dict[str, Any]] = []
    full_draws: list[dict[str, Any]] = []
    for learner in fit:
        full_draws.append(draw_episode(learner, events, seed=seed))
        for start in STARTS:
            try:
                _, pool, _ = candidate_pool(events[learner], start)
                n = len(pool)
            except ValueError:
                n = 0
            availability[f"start{start}_n_ge_16"] += int(n >= MENU_SIZE)
            availability[f"start{start}_n_lt_16"] += int(n < MENU_SIZE)
            if n < MENU_SIZE and len(short_examples) < 20:
                short_examples.append({"learner": learner, "start": start, "candidate_pool": n})

    result = {
        "status": "pass" if not failures else "failed",
        "seed": seed,
        "fit_learners": len(fit),
        "smoke_learners": smoke_learners,
        "starts": list(STARTS),
        "menu_size": MENU_SIZE,
        "target_n": TARGET_N,
        "expected_corpus_rows": sum(len(x["candidates"]) for x in full_draws) * N_CLASSES,
        "full_menu_size_total": sum(len(x["candidates"]) for x in full_draws),
        "full_start_counts": dict(Counter(str(x["start"]) for x in full_draws)),
        "smoke_rows_if_pass": sum(len(x["candidates"]) for x in rows) * N_CLASSES,
        "smoke_menu_sizes": [len(x["candidates"]) for x in rows],
        "smoke_start_counts": dict(Counter(str(x["start"]) for x in rows)),
        "menu_rule": "min(16, candidate_pool_size), without replacement",
        "start_rule": "uniform over feasible starts in {3,10}; start3 fallback when start10 unavailable",
        "smoke_failures": failures,
        "fit_availability": dict(availability),
        "short_examples": short_examples,
        "generated_text": False,
        "gpu_launched": False,
        "real_outcome_rows_in_training": 0,
    }
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw-dir", default=str(REPO.parent / "Datasets" / "CodeWorkout_CMU"))
    ap.add_argument("--split", default="splits/cw_learner_split.csv")
    ap.add_argument("--smoke-learners", type=int, default=20)
    ap.add_argument("--out", default="results/sim2real_w_smoke/structural_audit.json")
    args = ap.parse_args()
    raw = pathlib.Path(args.raw_dir).resolve()
    split = read_split((REPO / args.split).resolve())
    events = real_event_rows(raw)
    result = structural_audit(events, split, smoke_learners=args.smoke_learners)
    out = (REPO / args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
