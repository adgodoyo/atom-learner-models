import pathlib
import sys

import numpy as np
import pandas as pd
import pytest
import torch

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from cw_sigma import categorical_jensen_gap, fixed_history_pairs, paired_fall_ci  # noqa: E402


def test_categorical_jensen_gap_is_zero_when_samples_are_identical():
    probs = torch.tensor([[[0.2, 0.8], [0.2, 0.8], [0.2, 0.8]]], dtype=torch.float64)
    gap = categorical_jensen_gap(probs)
    assert torch.allclose(gap, torch.zeros_like(gap), atol=1e-12)


def test_categorical_jensen_gap_is_positive_when_samples_disagree():
    probs = torch.tensor([[[0.99, 0.01], [0.01, 0.99]]], dtype=torch.float64)
    assert categorical_jensen_gap(probs).item() > 0.5


def _pair_fixture():
    attempts = pd.DataFrame(
        {
            "code_state_id": ["a0", "a1", "a2", "b0", "b1", "b2"],
            "learner": ["A", "A", "A", "B", "B", "B"],
            "episode_id": ["A10", "A10", "A100", "B10", "B100", "B100"],
            "problem_id": [1, 1, 2, 1, 2, 2],
            "t": [10, 11, 100, 10, 100, 101],
        }
    )
    episodes = pd.DataFrame(
        {
            "learner": ["A", "A", "B", "B"],
            "episode_id": ["A10", "A100", "B10", "B100"],
            "problem_id": [1, 2, 1, 2],
            "start_timestamp": [1, 2, 1, 2],
        }
    )
    # Use 10/100 as source-order positions by inserting harmless earlier episodes.
    padding = []
    for learner in ("A", "B"):
        for t in range(101):
            if t in (10, 100):
                continue
            padding.append(
                {"learner": learner, "episode_id": f"{learner}{t}", "problem_id": 9,
                 "start_timestamp": t + 1}
            )
    episodes.loc[episodes["episode_id"].str.endswith("10"), "start_timestamp"] = 11
    episodes.loc[episodes["episode_id"].str.endswith("100"), "start_timestamp"] = 101
    episodes = pd.concat([episodes, pd.DataFrame(padding)], ignore_index=True)
    val = {
        "attempt_id": attempts["code_state_id"].to_numpy(),
        "learner": attempts["learner"].to_numpy(),
        "episode_id": attempts["episode_id"].to_numpy(),
        "problem_id": attempts["problem_id"].to_numpy(),
        "prior_episode_count": np.array([10, 10, 100, 10, 100, 100]),
    }
    mu = np.array([[1, 2], [1, 2], [3, 4], [5, 6], [7, 8], [7, 8]], dtype=np.float32)
    log_sigma = mu / 100.0
    return val, attempts, episodes, mu, log_sigma


def test_fixed_history_pairs_joins_by_value_and_keeps_common_cohort():
    val, attempts, episodes, mu, log_sigma = _pair_fixture()
    pairs = fixed_history_pairs(val, attempts, episodes, mu, log_sigma)
    assert pairs["learner"].tolist() == ["A", "B"]
    assert pairs["row_low"].tolist() == [0, 3]
    assert pairs["row_high"].tolist() == [2, 4]
    assert pairs["episode_history_low"].tolist() == [10, 10]
    assert pairs["episode_history_high"].tolist() == [100, 100]


def test_fixed_history_pairs_catches_corrupt_problem_membership():
    val, attempts, episodes, mu, log_sigma = _pair_fixture()
    val["problem_id"] = val["problem_id"].copy()
    val["problem_id"][2] = 999
    with pytest.raises(AssertionError, match="problem_id"):
        fixed_history_pairs(val, attempts, episodes, mu, log_sigma)


def test_paired_fall_ci_requires_clear_positive_lower_bound():
    low = np.arange(1.0, 21.0)
    high = low - 1.0
    result = paired_fall_ci(low, high, seed=0)
    assert result["mean_fall_t10_minus_t100"] == pytest.approx(1.0)
    assert result["ci_clear_fall"] is True
