"""Tests for src/atom/filter/batch.py: padded-batch construction and the DataFrame-rows
<-> padded-tensor bridge (mask_from_rows) that P3-3 relies on to score IRT-comparison cells."""

import numpy as np
import pandas as pd
import pytest
import torch

from atom.filter.batch import (
    build_padded_batch,
    build_padded_batch_v2,
    check_coverage,
    load_production_codec,
    mask_from_rows,
)


def _toy_ev():
    # learner "a": 3 events (t=0,1,2); learner "b": 1 event (t=0). Deliberately uneven so
    # padding is exercised.
    rows = [
        {"learner": "a", "t": 0, "problem_id": 10, "correct": 1, "hint_count": 0, "saw_answer": 0},
        {"learner": "a", "t": 1, "problem_id": 20, "correct": 0, "hint_count": 2, "saw_answer": 1},
        {"learner": "a", "t": 2, "problem_id": 10, "correct": 1, "hint_count": 0, "saw_answer": 0},
        {"learner": "b", "t": 0, "problem_id": 20, "correct": 0, "hint_count": 0, "saw_answer": 0},
    ]
    return pd.DataFrame(rows)


def _toy_codec():
    codec_matrix = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)  # 2 problems, dim 2
    pid_to_idx = {10: 0, 20: 1}
    return codec_matrix, pid_to_idx


def _toy_ev_with_timestamps():
    # learner "a": t=0 at day 0, t=1 at day 2 (gap 2d), t=2 at day 2.5 (gap 12h = 0.5d).
    # learner "b": single event (t=0) -> no previous event, delta_days must be 0.
    base = pd.Timestamp("2026-01-01", tz="UTC")
    rows = [
        {"learner": "a", "t": 0, "problem_id": 10, "correct": 1, "hint_count": 0, "saw_answer": 0,
         "timestamp": base},
        {"learner": "a", "t": 1, "problem_id": 20, "correct": 0, "hint_count": 0, "saw_answer": 0,
         "timestamp": base + pd.Timedelta(days=2)},
        {"learner": "a", "t": 2, "problem_id": 10, "correct": 1, "hint_count": 0, "saw_answer": 0,
         "timestamp": base + pd.Timedelta(days=2.5)},
        {"learner": "b", "t": 0, "problem_id": 20, "correct": 0, "hint_count": 0, "saw_answer": 0,
         "timestamp": base},
    ]
    return pd.DataFrame(rows)


def test_build_padded_batch_shapes_and_padding():
    codec_matrix, pid_to_idx = _toy_codec()
    batch = build_padded_batch(_toy_ev(), codec_matrix, pid_to_idx, device="cpu")
    assert batch["learners"] == ["a", "b"]  # sorted
    assert batch["valid_mask"].shape == (2, 3)  # N=2, T_max=3 (learner a's length)
    assert torch.equal(batch["lengths"], torch.tensor([3, 1]))
    assert torch.equal(batch["valid_mask"], torch.tensor([[True, True, True], [True, False, False]]))
    # Padded region is exactly zero.
    assert torch.equal(batch["correct"][1, 1:], torch.zeros(2))
    assert torch.equal(batch["item_codec"][1, 1:], torch.zeros(2, 2))


def test_build_padded_batch_looks_up_correct_codec_rows():
    codec_matrix, pid_to_idx = _toy_codec()
    batch = build_padded_batch(_toy_ev(), codec_matrix, pid_to_idx, device="cpu")
    # learner a: problems [10, 20, 10] -> codec rows [0, 1, 0]
    expected = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    assert torch.allclose(batch["item_codec"][0], expected)


def test_build_padded_batch_hint_binarised_and_saw_answer_passthrough():
    codec_matrix, pid_to_idx = _toy_codec()
    batch = build_padded_batch(_toy_ev(), codec_matrix, pid_to_idx, device="cpu")
    # learner a's t=1 has hint_count=2 -> binarised to 1.0; saw_answer passes through as 1.0.
    assert batch["hint_bin"][0, 1].item() == 1.0
    assert batch["saw_answer"][0, 1].item() == 1.0


def test_check_coverage_raises_on_missing_problem_id():
    _codec_matrix, pid_to_idx = _toy_codec()
    try:
        check_coverage([10, 20, 99], pid_to_idx)
        raised = False
    except RuntimeError:
        raised = True
    assert raised


def test_check_coverage_passes_when_all_covered():
    _codec_matrix, pid_to_idx = _toy_codec()
    check_coverage([10, 20], pid_to_idx)  # must not raise


def test_load_production_codec_dim_mismatch_raises(tmp_path):
    path = tmp_path / "codec.npz"
    np.savez(path, problem_id=np.array([1, 2]), codec=np.zeros((2, 5), dtype=np.float32))
    try:
        load_production_codec(path, expected_dim=8)
        raised = False
    except AssertionError:
        raised = True
    assert raised


def test_load_production_codec_round_trip(tmp_path):
    path = tmp_path / "codec.npz"
    np.savez(path, problem_id=np.array([7, 3]), codec=np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
    codec_matrix, pid_to_idx = load_production_codec(path, expected_dim=2)
    assert pid_to_idx == {7: 0, 3: 1}
    assert np.allclose(codec_matrix[pid_to_idx[3]], [3.0, 4.0])


def test_mask_from_rows_marks_exactly_the_given_cells():
    rows = pd.DataFrame({"learner": ["a", "a", "b"], "t": [0, 2, 1]})
    mask = mask_from_rows(rows, learners=["a", "b"], t_max=3, device="cpu")
    expected = torch.tensor([[True, False, True], [False, True, False]])
    assert torch.equal(mask, expected)


def test_mask_from_rows_disjoint_from_complement():
    """A calib/eval-style split (P3-1's protocol) should produce two masks that never overlap."""
    calib_rows = pd.DataFrame({"learner": ["a", "b"], "t": [0, 0]})
    eval_rows = pd.DataFrame({"learner": ["a", "b"], "t": [1, 1]})
    learners = ["a", "b"]
    calib_mask = mask_from_rows(calib_rows, learners, t_max=2, device="cpu")
    eval_mask = mask_from_rows(eval_rows, learners, t_max=2, device="cpu")
    assert not torch.any(calib_mask & eval_mask)


# -- build_padded_batch_v2 (FA-1): score_missing kept as a sequence position, flagged -------

def _toy_ev_v2():
    # learner "a": t=0,1,2, t=1's score is MISSING (not imputed to 0). learner "b": t=0 only.
    rows = [
        {"learner": "a", "t": 0, "problem_id": 10, "score": 1.0, "score_missing": False,
         "hint_count": 0, "saw_answer": 0},
        {"learner": "a", "t": 1, "problem_id": 20, "score": np.nan, "score_missing": True,
         "hint_count": 2, "saw_answer": 1},
        {"learner": "a", "t": 2, "problem_id": 10, "score": 1.0, "score_missing": False,
         "hint_count": 0, "saw_answer": 0},
        {"learner": "b", "t": 0, "problem_id": 20, "score": 0.0, "score_missing": False,
         "hint_count": 0, "saw_answer": 0},
    ]
    return pd.DataFrame(rows)


def test_build_padded_batch_v2_keeps_missing_score_as_a_position_but_masks_it():
    codec_matrix, pid_to_idx = _toy_codec()
    batch = build_padded_batch_v2(_toy_ev_v2(), codec_matrix, pid_to_idx, device="cpu")
    assert batch["learners"] == ["a", "b"]
    # t=1 (learner a) is a REAL sequence position -- valid_mask True -- even though its
    # score is missing (label_valid False there): the event still happened.
    assert batch["valid_mask"][0, 1].item() is True
    assert batch["score_missing"][0, 1].item() is True
    assert batch["label_valid"][0, 1].item() is False
    # every other real position is label-valid.
    assert batch["label_valid"][0, 0].item() is True
    assert batch["label_valid"][0, 2].item() is True
    assert batch["label_valid"][1, 0].item() is True
    # correct at the missing-score cell is a harmless 0.0 placeholder, not read unmasked.
    assert batch["correct"][0, 1].item() == 0.0
    # item_codec is still populated at the missing-score position (it's real context).
    assert torch.equal(batch["item_codec"][0, 1], torch.tensor([0.0, 1.0]))


def test_build_padded_batch_v2_label_valid_equals_valid_and_not_missing():
    codec_matrix, pid_to_idx = _toy_codec()
    batch = build_padded_batch_v2(_toy_ev_v2(), codec_matrix, pid_to_idx, device="cpu")
    expected = batch["valid_mask"] & ~batch["score_missing"]
    assert torch.equal(batch["label_valid"], expected)


# -- delta_days (P3-4) ----------------------------------------------------------------------

def test_delta_days_omitted_by_default():
    codec_matrix, pid_to_idx = _toy_codec()
    batch = build_padded_batch(_toy_ev_with_timestamps(), codec_matrix, pid_to_idx, device="cpu")
    assert "delta_days" not in batch


def test_delta_days_matches_expected_gaps_and_zero_at_first_event():
    codec_matrix, pid_to_idx = _toy_codec()
    batch = build_padded_batch(_toy_ev_with_timestamps(), codec_matrix, pid_to_idx, device="cpu",
                               include_delta_days=True)
    assert batch["learners"] == ["a", "b"]
    # learner a: t=0 first event -> 0; t=1 gap 2d; t=2 gap 0.5d.
    assert torch.allclose(batch["delta_days"][0], torch.tensor([0.0, 2.0, 0.5]), atol=1e-4)
    # learner b: single event, t=0 -> 0 (padded positions also 0, harmless since masked out).
    assert batch["delta_days"][1, 0].item() == pytest.approx(0.0, abs=1e-6)


def test_delta_days_aligned_with_valid_mask():
    """Padded (invalid) cells must not carry spurious nonzero delta_days."""
    codec_matrix, pid_to_idx = _toy_codec()
    batch = build_padded_batch(_toy_ev_with_timestamps(), codec_matrix, pid_to_idx, device="cpu",
                               include_delta_days=True)
    padded = ~batch["valid_mask"]
    assert torch.all(batch["delta_days"][padded] == 0.0)
