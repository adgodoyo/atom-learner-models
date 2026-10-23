from __future__ import annotations

import numpy as np
import pytest

from atom.eval.cw_generation import (
    ARM_TO_CODE,
    MODE_TO_CODE,
    completed_row_indices,
    dist_n,
    load_numeric_generation_rows,
    normalised_edit_distance,
    numeric_generation_row,
    save_numeric_generation_rows,
    summarise_generation_rows,
)


def test_dist_n_all_unique_bigrams_is_one():
    assert dist_n(["a b c d"], n=2) == 1.0  # bigrams: (a,b) (b,c) (c,d), all distinct


def test_dist_n_repeated_bigram_is_below_one():
    assert dist_n(["a b a b"], n=2) == pytest.approx(2 / 3)  # (a,b) (b,a) (a,b) -> 2 unique / 3


def test_dist_n_pools_across_multiple_texts():
    assert dist_n(["a b", "a b"], n=1) == pytest.approx(0.5)  # tokens a,b,a,b -> 2 unique / 4


def test_dist_n_returns_none_when_no_ngrams_exist():
    assert dist_n([], n=2) is None
    assert dist_n(["a"], n=2) is None  # single token, no bigram
    assert dist_n(["", "  "], n=1) is None


def test_dist_n_rejects_n_below_one():
    with pytest.raises(ValueError):
        dist_n(["a b"], n=0)


def test_normalised_edit_distance_is_character_level_and_strips_edges():
    assert normalised_edit_distance(" abc\n", "abc") == 0.0
    assert normalised_edit_distance("abc", "axc") == pytest.approx(1 / 3)
    assert normalised_edit_distance("", "") == 0.0


def test_numeric_row_removes_text_and_direct_learner_identifier():
    raw = {
        "row_idx": 7,
        "learner": "vendor-learner-42",
        "arm": "personalised",
        "generation_mode": "conditional_imitation",
        "true_class_code": 3,
        "conditioned_class_code": 3,
        "generated_code": "return 2;",
        "real_completion": "return 1;",
        "prior_code": "return 0;",
    }
    row = numeric_generation_row(raw, codec_similarity=0.75)
    assert not any(isinstance(v, str) for v in row.values())
    assert "vendor-learner-42" not in str(row)
    assert row["normalised_edit_distance"] > 0
    assert row["copy_normalised_edit_distance"] > 0


def _complete_rows(n_rows: int = 3) -> list[dict]:
    rows = []
    for row_idx in range(n_rows):
        for mode in range(2):
            for arm in range(4):
                # Conditional row 0 is missed by every arm. Conditional row 1 has only
                # personalised exact, proving one arm cannot select the missed court.
                exact = float(mode == 0 and row_idx == 1 and arm == 0)
                rows.append({
                    "row_idx": row_idx,
                    "learner_hash": 100 + row_idx,
                    "arm_code": arm,
                    "mode_code": mode,
                    "true_class": 3,
                    "conditioned_class": 3 if mode == 0 else 1,
                    "exact_match": exact,
                    "normalised_edit_distance": 0.2 + 0.1 * arm,
                    "codec_similarity": 0.9 - 0.1 * arm,
                    "generated_chars": 10 + arm,
                    "has_prior": int(row_idx != 2),
                    "copy_normalised_edit_distance": 0.4 if row_idx != 2 else np.nan,
                    "copy_exact_match": 0.5 if row_idx != 2 else np.nan,
                })
    return rows


def test_numeric_npz_round_trip_has_no_object_arrays(tmp_path):
    rows = _complete_rows(2)
    path = save_numeric_generation_rows(rows, tmp_path / "scores.npz")
    with np.load(path, allow_pickle=False) as data:
        assert all(data[name].dtype != object for name in data.files)
    loaded = load_numeric_generation_rows(path)
    assert loaded == rows


# `_complete_rows` deliberately builds only the ORIGINAL 2-mode cell set
# (conditional_imitation/ancestral, no argmax) -- item 8+9(v)'s additive "argmax" mode
# changed `completed_row_indices`/`summarise_generation_rows`'s own DEFAULT `modes` to
# 3, so every call against this fixture must say explicitly which 2 it means (matching
# those functions' own documented "fails safe on a mismatch" contract, not a regression
# to route around).
TWO_MODES = ("conditional_imitation", "ancestral")


def test_completed_rows_requires_exactly_all_eight_cells():
    rows = _complete_rows(2)
    assert completed_row_indices(rows, modes=TWO_MODES) == {0, 1}
    assert completed_row_indices(rows[:-1], modes=TWO_MODES) == {0}
    assert completed_row_indices(rows + [rows[0]], modes=TWO_MODES) == {1}


def test_completed_rows_defaults_to_the_full_mode_set_and_fails_safe_on_2mode_data():
    # The DEFAULT `modes` now includes "argmax"; 2-mode-only data can never satisfy it,
    # and must report zero completed rows rather than a silently wrong subset.
    rows = _complete_rows(2)
    assert completed_row_indices(rows) == set()


def test_summary_missed_cell_is_arm_symmetric_and_copy_baseline_is_row_unique():
    summary = summarise_generation_rows(_complete_rows(3), n_boot=20, seed=0, modes=TWO_MODES)
    missed = summary["all_arms_missed_conditional"]
    assert missed["arm_symmetric"] is True
    assert missed["n_rows"] == 2  # rows 0 and 2; row 1 is excluded by personalised exact
    assert {cell["n"] for cell in missed["cells"].values()} == {2}
    copy = summary["copy_baseline"]
    assert copy["n_selected_rows"] == 3
    assert copy["n_prior_eligible"] == 2
    assert copy["n_first_attempt_excluded"] == 1
    assert copy["mean_prior_to_current_normalised_edit_distance"] == pytest.approx(0.4)
    assert copy["prior_equals_current_rate"] == pytest.approx(0.5)


def test_summary_uses_named_arm_and_mode_cells():
    summary = summarise_generation_rows(_complete_rows(2), n_boot=10, seed=1, modes=TWO_MODES)
    assert set(summary["cells"]) == set(TWO_MODES)
    assert set(summary["cells"]["ancestral"]) == set(ARM_TO_CODE)
    assert summary["scope"] == "partial_generic_student_no_personalisation"
    assert summary["codebleu"] == "deferred_to_B4"


def _complete_rows_3modes_3arms(n_rows: int = 2) -> list[dict]:
    """item 8+9(v)'s own restricted cell set: 3 arms (never u_mean), 3 modes
    (argmax included) -- launch2's actual shape, distinct from `_complete_rows`'s
    legacy 4-arm/2-mode fixture."""
    arms = ("personalised", "shuffled_u", "dw0")
    modes = ("conditional_imitation", "ancestral", "argmax")
    rows = []
    for row_idx in range(n_rows):
        for mode in modes:
            for arm in arms:
                rows.append({
                    "row_idx": row_idx,
                    "learner_hash": 100 + row_idx,
                    "arm_code": ARM_TO_CODE[arm],
                    "mode_code": MODE_TO_CODE[mode],
                    "true_class": 3,
                    "conditioned_class": 3,
                    "exact_match": 0.0,
                    "normalised_edit_distance": 0.3,
                    "codec_similarity": 0.8,
                    "generated_chars": 10,
                    "has_prior": 0,
                    "copy_normalised_edit_distance": np.nan,
                    "copy_exact_match": np.nan,
                })
    return rows, arms, modes


def test_argmax_mode_and_restricted_3arm_set_round_trip():
    rows, arms, modes = _complete_rows_3modes_3arms(2)
    completed = completed_row_indices(rows, arms=arms, modes=modes)
    assert completed == {0, 1}
    summary = summarise_generation_rows(rows, n_boot=10, seed=2, arms=arms, modes=modes)
    assert set(summary["cells"]) == set(modes)
    assert set(summary["cells"]["argmax"]) == set(ARM_TO_CODE)  # per-arm keys unrestricted
    assert summary["cells"]["argmax"]["personalised"]["n"] == 2
    assert summary["cells"]["argmax"]["u_mean"]["n"] == 0  # never generated for this run
    assert summary["n_expected_records_per_completed_row"] == len(arms) * len(modes)
    # paired contrasts only against the two REQUESTED other arms, never u_mean
    assert set(summary["paired_ci_learner_clustered_personalised_minus_other"]["argmax"]["exact_match"]) == {
        "shuffled_u", "dw0",
    }
