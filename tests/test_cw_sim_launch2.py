"""`infra.cw_sim_launch2.dist_n_by_cell`/`save_raw_generation_records`/
`load_raw_generation_records` -- the pure functions in the launch2 orchestration
script (everything else needs a real/stub GPU model + real manifest, exercised by the
actual launch and its own pilot, not here)."""
from __future__ import annotations

import importlib.util
import json
import pathlib

REPO = pathlib.Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("cw_sim_launch2", REPO / "infra" / "cw_sim_launch2.py")
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)


def _rec(row_idx, arm, mode, code):
    return {"row_idx": row_idx, "arm": arm, "generation_mode": mode, "generated_code": code}


def test_dist_n_by_cell_groups_by_arm_and_mode_only():
    records = [
        _rec(0, "personalised", "conditional_imitation", "a b c"),
        _rec(1, "personalised", "conditional_imitation", "a b d"),
        _rec(0, "dw0", "conditional_imitation", "x y z"),
        _rec(0, "personalised", "ancestral", "p q r"),
    ]
    out = mod.dist_n_by_cell(records, ("personalised", "dw0"), ("conditional_imitation", "ancestral"))
    assert set(out) == {"conditional_imitation", "ancestral"}
    assert set(out["conditional_imitation"]) == {"personalised", "dw0"}
    ci_pers = out["conditional_imitation"]["personalised"]
    assert ci_pers["n_texts"] == 2
    assert ci_pers["dist_1"] is not None
    ci_dw0 = out["conditional_imitation"]["dw0"]
    assert ci_dw0["n_texts"] == 1
    anc_pers = out["ancestral"]["personalised"]
    assert anc_pers["n_texts"] == 1
    anc_dw0 = out["ancestral"]["dw0"]
    assert anc_dw0["n_texts"] == 0
    assert anc_dw0["dist_1"] is None  # no texts in this cell -- None, not a fake 0/1


def test_dist_n_by_cell_empty_records():
    out = mod.dist_n_by_cell([], ("personalised",), ("conditional_imitation",))
    assert out["conditional_imitation"]["personalised"] == {"dist_1": None, "dist_2": None, "n_texts": 0}


def _raw_rec(row_idx, generated_code="a=1"):
    return {"row_idx": row_idx, "arm": "personalised", "generation_mode": "conditional_imitation",
            "learner": "L1", "problem_id": 5, "true_class_code": 0, "conditioned_class_code": 0,
            "generated_code": generated_code, "real_completion": "a=1", "prior_code": None,
            "prompt_text": "prompt", "code_state_id": "cs1"}


def test_raw_generation_records_round_trip_preserves_text(tmp_path):
    path = tmp_path / "generations_raw.jsonl"
    records = [_raw_rec(0, "print(1)"), _raw_rec(1, "print(2)")]
    mod.save_raw_generation_records(records, path)
    loaded = mod.load_raw_generation_records(path)
    assert [r["generated_code"] for r in loaded] == ["print(1)", "print(2)"]
    assert [r["row_idx"] for r in loaded] == [0, 1]


def test_raw_generation_records_load_missing_file_returns_empty(tmp_path):
    assert mod.load_raw_generation_records(tmp_path / "nope.jsonl") == []


def test_raw_generation_records_one_json_object_per_line(tmp_path):
    path = tmp_path / "generations_raw.jsonl"
    mod.save_raw_generation_records([_raw_rec(0), _raw_rec(1)], path)
    lines = path.read_text().strip().split("\n")
    assert len(lines) == 2
    assert all(json.loads(line)["row_idx"] in (0, 1) for line in lines)


def test_cumulative_cell_progress_credits_completed_prior_cells():
    """Cell 0 (index 0) at done=total must equal exactly `total_per_cell`, and cell 1's
    FIRST call (done=1) must be `total_per_cell + 1` -- not reset back down to 1. A reset
    makes OverrunTrigger read a false overrun (~19x entering cell 2 of 9) after a correctly
    finished cell, and it then shuts the instance down unconditionally."""
    total_per_cell = 1194
    assert mod.cumulative_cell_progress(0, 0, total_per_cell) == 0
    assert mod.cumulative_cell_progress(0, total_per_cell, total_per_cell) == total_per_cell
    # Moving into cell index 1 (the second cell) must CREDIT cell 0's full completion,
    # not reset to a small number while elapsed time keeps climbing.
    assert mod.cumulative_cell_progress(1, 1, total_per_cell) == total_per_cell + 1
    assert mod.cumulative_cell_progress(1, total_per_cell, total_per_cell) == 2 * total_per_cell
    # Cell index 8 (the last of 9 cells) at done=total_per_cell must equal the full
    # 9-cell total exactly -- the invariant `cumulative_total` in `run_stratum` assumes.
    assert mod.cumulative_cell_progress(8, total_per_cell, total_per_cell) == 9 * total_per_cell


def test_run_stratum_on_progress_never_regresses_cumulative_done_across_cells():
    """A closer-to-real regression: simulate the exact `on_progress` closure pattern
    `run_stratum`/`run_codebleu_blind_panel` use (first-seen `cell_order` list,
    `cell_order.index(cell)`) across a full 3-cell sequence and assert the cumulative
    count is monotonically non-decreasing throughout (a silent drop is exactly the
    failure that triggers a false overrun)."""
    arms, modes = ("personalised", "shuffled_u"), ("conditional_imitation",)
    total_per_cell = 50
    cell_order: list[tuple[str, str]] = []
    seen = []
    for arm in arms:
        for mode in modes:
            for done in range(1, total_per_cell + 1, 7):
                cell = (arm, mode)
                if cell not in cell_order:
                    cell_order.append(cell)
                cumulative = mod.cumulative_cell_progress(cell_order.index(cell), done, total_per_cell)
                seen.append(cumulative)
    assert seen == sorted(seen)  # never regresses, even across a cell boundary
    assert seen[-1] <= total_per_cell * len(arms) * len(modes)
