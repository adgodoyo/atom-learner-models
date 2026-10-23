"""CodeWorkout same-behaviour/different-style pair construction for the code-semantics gate.

Test vectors (per-submission binary pass/fail across a problem's test cases) are only
meaningfully comparable WITHIN a problem -- different problems have different test suites
of different lengths. So pairs are built per-problem:

  - "same behaviour" = two DIFFERENT code submissions (different CodeStateID, and
    different code TEXT -- otherwise it's not actually a different-style example) whose
    binary_correctness vectors are IDENTICAL.
  - "different behaviour" = two submissions for the same problem whose binary_correctness
    vectors DIFFER.

Read as an ordering test only: same-behaviour pairs are never
required (nor trained) to coincide -- two correct programs can differ in algorithm, and
that difference is meaning the simulator must predict, not style. The gate asks whether
codec similarity trends higher for same-behaviour than different-behaviour pairs on
average (AUC), not whether same-behaviour pairs collapse together.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_behaviour_pairs(df: pd.DataFrame, pairs_per_problem: int, seed: int) -> pd.DataFrame:
    """df: TIKTOC granular dataframe WITH the Code column
    (atom.data.cw.load_granular(..., include_code=True)).

    Returns one row per sampled pair: problem_id, prompt, id_a/id_b (CodeStateID),
    code_a/code_b (raw text), same_behaviour (bool). Deterministic given `seed`.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for pid, g in df.groupby("ProblemID"):
        g = g.drop_duplicates(subset=["Code"])  # "different style" needs different code text
        if len(g) < 2:
            continue
        prompt = g["prompt"].iloc[0]
        by_vec: dict[tuple, list] = {}
        for idx, vec in zip(g.index, g["binary_correctness"].map(tuple)):
            by_vec.setdefault(vec, []).append(idx)

        same_pool = []
        for ids in by_vec.values():
            for a in range(len(ids)):
                for b in range(a + 1, len(ids)):
                    same_pool.append((ids[a], ids[b]))

        vec_groups = list(by_vec.values())
        diff_pool = []
        for a in range(len(vec_groups)):
            for b in range(a + 1, len(vec_groups)):
                for ia in vec_groups[a]:
                    for ib in vec_groups[b]:
                        diff_pool.append((ia, ib))

        n_same = min(pairs_per_problem, len(same_pool))
        n_diff = min(pairs_per_problem, len(diff_pool))
        if n_same == 0 or n_diff == 0:
            continue

        same_sel = [same_pool[i] for i in rng.choice(len(same_pool), n_same, replace=False)]
        diff_sel = [diff_pool[i] for i in rng.choice(len(diff_pool), n_diff, replace=False)]

        for (ia, ib), same in [(p, True) for p in same_sel] + [(p, False) for p in diff_sel]:
            rows.append({
                "problem_id": int(pid), "prompt": prompt,
                "id_a": g.loc[ia, "CodeStateID"], "id_b": g.loc[ib, "CodeStateID"],
                "code_a": g.loc[ia, "Code"], "code_b": g.loc[ib, "Code"],
                "same_behaviour": same,
            })
    return pd.DataFrame(rows)
