from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from im_s3_dream import (  # noqa: E402
    DreamConfig,
    DreamCorpusGenerator,
    ImaginedRolloutEnv,
    OneToOneMixer,
    RealOnlyCheckpointSelector,
    index_dream_corpus,
    leakage_canary,
    marginal_guard,
    random_policy_floor_guard,
    rollout_marginal_guard,
    run_smoke,
    validate_dream_coverage,
)


def test_complete_smoke_path():
    result = run_smoke()
    assert result["leakage"]["pass"]
    assert result["marginal"]["pass"]
    assert result["selected_epoch"] == 1
    assert result["env_done"]


def test_corpus_is_batched_and_local(tmp_path):
    calls = []
    generator = DreamCorpusGenerator(lambda batch: (calls.append(len(batch)) or ["x"] * len(batch)),
                                     DreamConfig(batch_size=2, classes_per_row=1))
    rows = [{"learner": "L", "problem_id": i, "source_split": "train", "class_probs": [0.5, 0.5]}
            for i in range(5)]
    manifest = generator.generate(rows, tmp_path / "generations.jsonl", "sha")
    assert calls == [2, 2, 1]
    assert manifest["raw_text_uploaded"] is False
    with pytest.raises(ValueError):
        generator.generate(rows, "s3://bucket/generations.jsonl", "sha")


def test_leakage_guard_and_marginal_guard():
    records = [{"source_split": "train", "conditioned_class": c} for c in [0, 1, 0, 1]]
    assert leakage_canary(records)["pass"]
    assert marginal_guard(records, [0.5, 0.5])["pass"]
    with pytest.raises(AssertionError):
        leakage_canary([{**records[0], "source_split": "val"}])


def test_rollout_guard_uses_sampled_outcomes_not_conditioned_corpus_classes():
    records = [
        {"learner": "L", "problem_id": 1, "conditioned_class": 0,
         "class_probs": [0.5, 0.5], "source_split": "train"},
        {"learner": "L", "problem_id": 1, "conditioned_class": 1,
         "class_probs": [0.5, 0.5], "source_split": "train"},
    ]
    # Top-K corpus rows are not themselves sampled outcomes; the cache must still index
    # exactly once per (learner, problem, class).
    assert len(index_dream_corpus(records)) == 2
    events = [
        {"outcome_code": 0, "class_probs": [0.5, 0.5]},
        {"outcome_code": 1, "class_probs": [0.5, 0.5]},
    ] * 20
    result = rollout_marginal_guard(events)
    assert result["pass"] and result["observed_field"] == "outcome_code"
    with pytest.raises(AssertionError, match="classes_per_row=None"):
        validate_dream_coverage(records, [{"learner": "L", "problem_id": 1}], n_classes=4)


def test_imagined_environment_never_exposes_reference():
    env = ImaginedRolloutEnv(
        learner="L", candidates=[{"problem_id": 1}], initial_belief=np.array([0.0]),
        b_full=np.array([1.0]), predict_class=lambda _b, _c: [1.0],
        update_belief=lambda _b, _e: np.array([1.0]), distance=lambda b, f: float(abs(b[0] - f[0])),
        budget=1, seed=1,
    )
    assert "b_full" not in env.observation()
    obs, reward, done, info = env.step(0)
    assert done and reward == 1.0 and info["event"]["source"] == "imagined"
    assert env.outcome_history[0]["outcome_code"] == 0


def test_one_to_one_and_real_only_selector():
    mixed = OneToOneMixer().mix([{"id": 1}], [{"id": 2}])
    assert [x["source"] for x in mixed] == ["real_replay", "imagined"]
    selector = RealOnlyCheckpointSelector()
    assert selector.consider(1, 2.0, {"epoch": 1}, [{"split": "val", "source": "real_replay"}])
    with pytest.raises(AssertionError):
        selector.consider(2, 1.0, {}, [{"split": "val", "source": "imagined"}])


def test_random_floor_guard():
    result = random_policy_floor_guard([0.2, 0.3], [0.1, 0.4, 0.2, 0.3])
    assert result["pass"]
