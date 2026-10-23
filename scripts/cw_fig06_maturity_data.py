#!/usr/bin/env python
"""Data for thesis Figure 6: ATOM's CodeWorkout first-submission AUC by belief maturity.
Buckets first submissions by how many problems the learner had already worked through
(0, 1-8, 9-16, 17-36, 37+) and scores P(passes all tests) with learner-clustered 95% intervals.
Inputs: ATOM's saved validation predictions, splits/, ../Datasets/ (for item difficulty).
Output: results/cw_first_submission_baselines/fig06_cw_maturity.csv (aggregates only).
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import pandas as pd

SCRIPTS_DIR = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))

from cw_first_submission_baselines import (  # noqa: E402
    DEFAULT_REPO, auc, clustered_auc_ci, first_rows, load_atom_rows, load_corpus_attempts,
    two_scalar_predictions,
)

BUCKETS = [("0", 0, 0), ("1-8", 1, 8), ("9-16", 9, 16), ("17-36", 17, 36), ("37+", 37, None)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--repo-root", default=str(DEFAULT_REPO))
    ap.add_argument("--raw-dir", default=None)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()
    repo = pathlib.Path(args.repo_root).resolve()
    out_dir = pathlib.Path(args.out_dir) if args.out_dir else repo / "results" / "cw_first_submission_baselines"

    first = first_rows(load_atom_rows(repo), "learner", "problem_id", "t")
    corpus_first = first_rows(load_corpus_attempts(repo, args.raw_dir), "learner", "problem_id", "t")
    first["p_difficulty"] = two_scalar_predictions(first, corpus_first[corpus_first["split"] != "val"])["p_difficulty"]

    rows = []
    for name, lo, hi in BUCKETS + [("all", 0, None)]:
        m = (first["prior_episodes"] >= lo) & (first["prior_episodes"] <= (hi if hi is not None else np.inf))
        s = first[m]
        ci = clustered_auc_ci(s["y"], s["p_atom"], s["learner"])
        rows.append({"bucket": name, "n_rows": len(s), "n_learners": s["learner"].nunique(),
                     "base_rate": round(s["y"].mean(), 4), "auc": round(auc(s["y"], s["p_atom"]), 4),
                     "lo": round(ci[0], 4), "hi": round(ci[1], 4),
                     "difficulty_only_auc": round(auc(s["y"], s["p_difficulty"]), 4)})
    table = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / "fig06_cw_maturity.csv", index=False)
    print(table.to_string(index=False))
    print("reference line (difficulty only, all first submissions):", table.iloc[-1]["difficulty_only_auc"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
