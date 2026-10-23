"""`infra.cw_sim_launch2_tiktoc.build_tiktoc_tv_lookup` -- the one pure, Mac-testable
function in item (D)'s orchestrator (everything else needs a real/stub GPU model + a
real TIKTOC codec cache, exercised by the actual launch and its own pilot, not here).
Requires the real licensed TIKTOC granular file locally -- skipped otherwise, same
pattern as `test_cw_tiktoc_transfer.py`'s own real-data tests."""
from __future__ import annotations

import importlib.util
import pathlib

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("cw_sim_launch2_tiktoc",
                                               REPO / "infra" / "cw_sim_launch2_tiktoc.py")
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)

from atom.data.cw import default_cw_dir  # noqa: E402

pytestmark_real = pytest.mark.skipif(
    not (default_cw_dir() / "dataset_granular_1st.zip").exists()
    and not (default_cw_dir() / "dataset_granular_1st.pkl").exists(),
    reason="TIKTOC granular dataset (1st variant) not present locally",
)


@pytestmark_real
def test_build_tiktoc_tv_lookup_matches_load_test_vector_labels_key_shape():
    tv = mod.build_tiktoc_tv_lookup()
    assert len(tv["lookup"]) == 3714
    assert tv["max_test_cases"] == 32
    assert tv["headline_problem_ids"] == {1, 3, 5, 12, 13, 17, 20, 21, 22, 24, 25, 46, 71}

    key, rec = next(iter(tv["lookup"].items()))
    learner, problem_id, code_state_id = key
    assert isinstance(learner, str) and learner.startswith("tiktoc:")
    assert isinstance(problem_id, int)
    assert isinstance(code_state_id, str)
    assert set(rec) == {"test_vector", "valid_mask", "n_cases", "compiled"}
    assert rec["test_vector"].shape == (32,)
    assert rec["valid_mask"].shape == (32,)
    assert rec["valid_mask"][: rec["n_cases"]].all()
    assert not rec["valid_mask"][rec["n_cases"]:].any()


@pytestmark_real
def test_build_tiktoc_tv_lookup_keys_are_unique_per_row():
    tv = mod.build_tiktoc_tv_lookup()
    # One row per (learner, problem_id) in TIKTOC's first-submission file (confirmed in
    # test_cw_tiktoc_transfer.py) -- the dict must not have silently collapsed any two
    # distinct rows onto the same (learner, problem_id, code_state_id) key.
    assert len(tv["lookup"]) == 3714
