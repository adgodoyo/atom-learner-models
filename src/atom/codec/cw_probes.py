"""Code representation probe battery -- pair construction.

Four probes share ONE per-problem submission pool (deduped by code text) so every
submission is encoded exactly once regardless of how many probes reference it:

  - **task**: cross-problem vs within-problem pairs (positive control -- must separate
    somewhere or the whole battery is void, per the pre-registered outcome map).
  - **behaviour-gross**: same test-vector vs a DIFFERENT vector that disagrees on many
    test cases (>= half the problem's test suite).
  - **behaviour-near-miss**: same test-vector vs a DIFFERENT vector that disagrees on
    EXACTLY one test case -- the hard case a coarse "any difference" pairing cannot
    isolate.
  - **person**: same-author vs different-author pairs, problem identity held fixed
    (both submissions on the SAME problem) so problem geometry can't leak into the
    label -- learner IDs are natural, free labels, never a supervision signal.

Problems are split calibrate/confirm BEFORE any pair is drawn (`split_problems`): all
exploration happens on calibrate; confirm is touched once.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def split_problems(problem_ids: list[int], confirm_frac: float, seed: int
                    ) -> tuple[list[int], list[int]]:
    """Deterministic calibrate/confirm split of problem ids, fixed before any pair is drawn."""
    ids = sorted(set(int(p) for p in problem_ids))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(ids))
    n_confirm = max(1, round(len(ids) * confirm_frac))
    confirm_idx = set(perm[:n_confirm].tolist())
    confirm = sorted(ids[i] for i in confirm_idx)
    calib = sorted(ids[i] for i in range(len(ids)) if i not in confirm_idx)
    return calib, confirm


def build_submission_pool(df: pd.DataFrame, problem_ids: list[int], pool_per_problem: int,
                           seed: int) -> pd.DataFrame:
    """One row per sampled, code-text-deduped submission, restricted to `problem_ids`.

    This is the SHARED pool every probe below draws pairs from -- each submission is
    encoded once no matter how many probes reference it.
    """
    rng = np.random.default_rng(seed)
    frames = []
    for pid in problem_ids:
        g = df[df["ProblemID"] == pid].drop_duplicates(subset=["Code"])
        if len(g) == 0:
            continue
        n = min(pool_per_problem, len(g))
        idx = rng.choice(g.index.to_numpy(), size=n, replace=False)
        frames.append(g.loc[idx])
    if not frames:
        return df.iloc[0:0]
    return pd.concat(frames, ignore_index=False)


def _sample_pairs(pool: list, n: int, rng: np.random.Generator) -> list[tuple]:
    if len(pool) <= n:
        return pool
    idx = rng.choice(len(pool), n, replace=False)
    return [pool[i] for i in idx]


def build_task_pairs(pool: pd.DataFrame, pairs_total: int, seed: int) -> pd.DataFrame:
    """same_problem: True (two submissions, same problem) vs False (different problems)."""
    rng = np.random.default_rng(seed)
    by_problem = {pid: g.index.tolist() for pid, g in pool.groupby("ProblemID")}
    problems = sorted(by_problem)
    same_pool, diff_pool = [], []
    for pid, idxs in by_problem.items():
        for a in range(len(idxs)):
            for b in range(a + 1, len(idxs)):
                same_pool.append((idxs[a], idxs[b]))
    for a in range(len(problems)):
        for b in range(a + 1, len(problems)):
            for ia in by_problem[problems[a]]:
                for ib in by_problem[problems[b]]:
                    diff_pool.append((ia, ib))
    n = pairs_total // 2
    same_sel = _sample_pairs(same_pool, n, rng)
    diff_sel = _sample_pairs(diff_pool, n, rng)
    rows = []
    for (ia, ib), same in [(p, True) for p in same_sel] + [(p, False) for p in diff_sel]:
        rows.append({"probe": "task", "id_a": ia, "id_b": ib, "label": same})
    return pd.DataFrame(rows)


def _hamming(v1: tuple, v2: tuple) -> int:
    return sum(1 for x, y in zip(v1, v2) if x != y)


def _behaviour_pairs(pool: pd.DataFrame, pairs_per_problem: int, seed: int,
                     mode: str) -> pd.DataFrame:
    """mode='gross' (disagree on >= half the test suite) or 'nearmiss' (disagree on exactly 1)."""
    rng = np.random.default_rng(seed)
    rows = []
    for pid, g in pool.groupby("ProblemID"):
        by_vec: dict[tuple, list] = {}
        for idx, vec in zip(g.index, g["binary_correctness"].map(tuple)):
            by_vec.setdefault(vec, []).append(idx)
        same_pool = []
        for ids in by_vec.values():
            for a in range(len(ids)):
                for b in range(a + 1, len(ids)):
                    same_pool.append((ids[a], ids[b]))
        vec_groups = list(by_vec.items())
        diff_pool = []
        for a in range(len(vec_groups)):
            va, ids_a = vec_groups[a]
            for b in range(a + 1, len(vec_groups)):
                vb, ids_b = vec_groups[b]
                d = _hamming(va, vb)
                is_gross = d >= max(1, len(va) // 2)
                is_nearmiss = d == 1
                if (mode == "gross" and is_gross) or (mode == "nearmiss" and is_nearmiss):
                    for ia in ids_a:
                        for ib in ids_b:
                            diff_pool.append((ia, ib))
        n = min(pairs_per_problem, len(same_pool), len(diff_pool)) if diff_pool else 0
        if n == 0:
            continue
        same_sel = _sample_pairs(same_pool, n, rng)
        diff_sel = _sample_pairs(diff_pool, n, rng)
        for (ia, ib), same in [(p, True) for p in same_sel] + [(p, False) for p in diff_sel]:
            rows.append({"probe": f"behaviour-{mode}", "id_a": ia, "id_b": ib, "label": same})
    return pd.DataFrame(rows)


def build_behaviour_gross_pairs(pool: pd.DataFrame, pairs_per_problem: int, seed: int) -> pd.DataFrame:
    return _behaviour_pairs(pool, pairs_per_problem, seed, mode="gross")


def build_behaviour_nearmiss_pairs(pool: pd.DataFrame, pairs_per_problem: int, seed: int) -> pd.DataFrame:
    return _behaviour_pairs(pool, pairs_per_problem, seed, mode="nearmiss")


def build_person_pairs(pool: pd.DataFrame, pairs_per_problem: int, seed: int) -> pd.DataFrame:
    """Label `same_author` is True for the same student, same problem, different code, and
    False for different students on the same problem. Problem identity is FIXED within every pair by construction."""
    rng = np.random.default_rng(seed)
    rows = []
    for pid, g in pool.groupby("ProblemID"):
        by_student: dict = {}
        for idx, sid in zip(g.index, g["SubjectID"]):
            by_student.setdefault(sid, []).append(idx)
        same_pool = []
        for idxs in by_student.values():
            for a in range(len(idxs)):
                for b in range(a + 1, len(idxs)):
                    same_pool.append((idxs[a], idxs[b]))
        students = list(by_student)
        diff_pool = []
        for a in range(len(students)):
            for b in range(a + 1, len(students)):
                for ia in by_student[students[a]]:
                    for ib in by_student[students[b]]:
                        diff_pool.append((ia, ib))
        n = min(pairs_per_problem, len(same_pool), len(diff_pool)) if same_pool else 0
        if n == 0:
            continue
        same_sel = _sample_pairs(same_pool, n, rng)
        diff_sel = _sample_pairs(diff_pool, n, rng)
        for (ia, ib), same in [(p, True) for p in same_sel] + [(p, False) for p in diff_sel]:
            rows.append({"probe": "person", "id_a": ia, "id_b": ib, "label": same})
    return pd.DataFrame(rows)


def token_edit_distance(a: str, b: str) -> int:
    """Levenshtein distance over whitespace-split tokens (pure python, small DP rows)."""
    ta, tb = a.split(), b.split()
    if len(ta) < len(tb):
        ta, tb = tb, ta
    prev = list(range(len(tb) + 1))
    for i, x in enumerate(ta, 1):
        cur = [i] + [0] * len(tb)
        for j, y in enumerate(tb, 1):
            cost = 0 if x == y else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[-1]


def edit_similarity(a: str, b: str) -> float:
    """1 - normalised token edit distance (higher = more similar); the baseline for every probe."""
    d = token_edit_distance(a, b)
    denom = max(len(a.split()), len(b.split()), 1)
    return 1.0 - d / denom
