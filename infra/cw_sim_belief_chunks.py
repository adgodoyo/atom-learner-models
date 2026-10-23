"""Shared helpers for `cw_sim_tv_head_refresh`: learner-aligned chunking of an attempts frame."""
from __future__ import annotations
import pathlib
import sys
REPO = pathlib.Path(__file__).resolve().parents[1]
for _p in (REPO / "src", REPO / "infra", REPO / "scripts", REPO):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import pandas as pd  # noqa: E402


def learner_aligned_chunks(attempts: pd.DataFrame, target_rows_per_chunk: int) -> list[list]:
    """Groups whole learners into chunks whose total row count is close to
    `target_rows_per_chunk`, never splitting one learner across two chunks. This matters
    because `test_vector_eval_predictions` filters its internal episodes batch by learner
    membership, so a learner split across chunks would be evaluated on partial history."""
    learner_order = attempts["learner"].drop_duplicates().tolist()
    rows_per_learner = attempts.groupby("learner", sort=False).size().to_dict()
    chunks, i = [], 0
    while i < len(learner_order):
        block, n = [], 0
        while i < len(learner_order) and (not block or n < target_rows_per_chunk):
            learner = learner_order[i]
            block.append(learner)
            n += rows_per_learner[learner]
            i += 1
        chunks.append(block)
    return chunks
