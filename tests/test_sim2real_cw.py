"""Tests for the train-only belief handoff loaded by the simulator-trained CodeWorkout policy."""
from __future__ import annotations

import numpy as np

from scripts.imagination_cw import load_train_handoff


def test_sim2real_train_handoff_uses_train_only_value_join(tmp_path):
    learners = np.asarray([f"L{i:03d}" for i in range(686)])
    path = tmp_path / "train.npz"
    np.savez_compressed(
        path,
        learner=learners,
        episode_id=np.asarray([f"E{i:03d}" for i in range(686)]),
        mu=np.zeros((686, 2), dtype=np.float32),
        log_sigma=np.zeros((686, 2), dtype=np.float32),
        release=np.asarray(["S19"] * 686),
        split=np.asarray(["train"] * 686),
        checkpoint_sha256=np.asarray(
            ["5337abbbad40e0d1474ddce1e3bdc69cd1159a94e29722cedef70d8c56d7a56d"]
        ),
    )
    loaded = load_train_handoff(path, expected_sha="")
    assert loaded["sha256"]
    assert len(loaded["last_index"]) == 686
    assert set(loaded["split"]) == {"train"}


def test_sim2real_train_handoff_rejects_non_train_rows(tmp_path):
    learners = np.asarray([f"L{i:03d}" for i in range(686)])
    splits = np.asarray(["train"] * 685 + ["val"])
    path = tmp_path / "bad.npz"
    np.savez_compressed(
        path,
        learner=learners,
        episode_id=np.asarray([f"E{i:03d}" for i in range(686)]),
        mu=np.zeros((686, 2), dtype=np.float32),
        log_sigma=np.zeros((686, 2), dtype=np.float32),
        release=np.asarray(["S19"] * 686),
        split=splits,
        checkpoint_sha256=np.asarray(
            ["5337abbbad40e0d1474ddce1e3bdc69cd1159a94e29722cedef70d8c56d7a56d"]
        ),
    )
    try:
        load_train_handoff(path, expected_sha="")
    except AssertionError as exc:
        assert "non-train" in str(exc)
    else:
        raise AssertionError("non-train handoff row was accepted")
