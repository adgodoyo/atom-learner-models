import json

import numpy as np
import pandas as pd
import pytest

from atom.data.cw_manifest import (
    attach_code_diff_hashes,
    attach_prior_attempt_code,
    build_codec_vector_lookup,
    build_episode_level_table,
    fit_global_class_freq,
    fit_per_problem_base_rate,
    load_hash_maps,
    per_problem_base_rate_probs,
    strict_cell_learners,
    strict_cell_row_mask,
    strip_held_out_items,
    volume_matched_removal,
)


def _attempts():
    # episode 0: 2 attempts (code_state_id a, b); episode 1: 1 attempt (c) with NO
    # code_hash match (simulates the disclosed missing-code gap in the CodeWorkout encoding).
    return pd.DataFrame({
        "episode_id": [0, 0, 1],
        "code_state_id": ["a", "b", "c"],
        "timestamp": pd.to_datetime(["2024-01-01T00:00", "2024-01-01T01:00", "2024-01-02T00:00"]),
        "learner": ["L1", "L1", "L2"],
        "problem_id": [10, 10, 11],
        "correct": [0, 1, 0],  # episode 0: attempt 1 succeeds -> y_ever=True; episode 1: never -> False
    })


def test_attach_code_diff_hashes_joins_and_drops_episode_with_missing_code():
    attempts = _attempts()
    code_map = pd.DataFrame({"code_state_id": ["a", "b"], "code_hash": ["ha", "hb"]})
    diff_map = pd.DataFrame({"code_state_id": ["b"], "diff_hash": ["db"]})
    out, stats = attach_code_diff_hashes(attempts, code_map, diff_map)

    # episode 1 (code_state_id "c") has no code_hash match -> the WHOLE episode drops
    assert set(out["episode_id"]) == {0}
    assert len(out) == 2
    assert stats["n_episodes_dropped_for_missing_code"] == 1
    assert stats["n_attempts_missing_code_hash"] == 1
    assert stats["n_attempts_before"] == 3
    assert stats["n_attempts_after"] == 2

    row_a = out[out["code_state_id"] == "a"].iloc[0]
    row_b = out[out["code_state_id"] == "b"].iloc[0]
    assert row_a["code_hash"] == "ha"
    assert pd.isna(row_a["diff_hash"])  # episode-first attempt, no previous code
    assert row_b["code_hash"] == "hb"
    assert row_b["diff_hash"] == "db"


def test_attach_code_diff_hashes_no_drop_when_all_code_present():
    attempts = _attempts().iloc[:2]  # only episode 0, both rows have code
    code_map = pd.DataFrame({"code_state_id": ["a", "b"], "code_hash": ["ha", "hb"]})
    diff_map = pd.DataFrame({"code_state_id": [], "diff_hash": []})
    out, stats = attach_code_diff_hashes(attempts, code_map, diff_map)
    assert len(out) == 2
    assert stats["n_episodes_dropped_for_missing_code"] == 0


def test_load_hash_maps_reads_npz(tmp_path):
    code_path = tmp_path / "code_map.npz"
    diff_path = tmp_path / "diff_map.npz"
    np.savez(code_path, code_state_id=np.array(["a", "b"]), code_hash=np.array(["ha", "hb"]))
    np.savez(diff_path, code_state_id=np.array(["b"]), diff_hash=np.array(["db"]))
    code_map, diff_map = load_hash_maps(code_path, diff_path)
    assert list(code_map["code_hash"]) == ["ha", "hb"]
    assert list(diff_map["diff_hash"]) == ["db"]


class _IdentityTransform:
    """Stand-in for CodecTransform: apply(X, k=5) just returns X[:, :dim-k] cast to
    float32, avoiding any dependency on a real fitted transform in this unit test."""

    def apply(self, X, k=5, rank=None):
        rank = X.shape[1] if rank is None else rank
        return np.asarray(X, dtype=np.float64)[:, :rank - k].astype(np.float32)


def test_build_codec_vector_lookup_gathers_and_transforms_only_requested_hashes(tmp_path):
    hidden_dim = 8
    meta = {"row_of_hash": {"h0": 0, "h1": 1, "h2": 2}, "hidden_dim": hidden_dim}
    meta_path = tmp_path / "meta.json"
    meta_path.write_text(json.dumps(meta))

    states_path = tmp_path / "states_mean.npy"
    raw = np.arange(3 * hidden_dim, dtype=np.float16).reshape(3, hidden_dim)
    np.lib.format.open_memmap(states_path, mode="w+", dtype="float16", shape=raw.shape)[:] = raw

    transform = _IdentityTransform()
    out = build_codec_vector_lookup({"h0", "h2"}, meta_path, states_path, transform, k=3)

    assert set(out.keys()) == {"h0", "h2"}
    assert out["h0"].shape == (hidden_dim - 3,)
    assert np.allclose(out["h0"], raw[0, :hidden_dim - 3])
    assert np.allclose(out["h2"], raw[2, :hidden_dim - 3])


def test_build_codec_vector_lookup_raises_on_unknown_hash(tmp_path):
    meta_path = tmp_path / "meta.json"
    meta_path.write_text(json.dumps({"row_of_hash": {"h0": 0}, "hidden_dim": 4}))
    states_path = tmp_path / "states_mean.npy"
    np.lib.format.open_memmap(states_path, mode="w+", dtype="float16", shape=(1, 4))[:] = 0.0
    with pytest.raises(KeyError):
        build_codec_vector_lookup({"h0", "does_not_exist"}, meta_path, states_path, _IdentityTransform())


def test_build_episode_level_table_one_row_per_episode_first_timestamp():
    attempts = _attempts()
    out = build_episode_level_table(attempts)
    assert len(out) == 2
    ep0 = out[out["episode_id"] == 0].iloc[0]
    assert ep0["learner"] == "L1"
    assert ep0["problem_id"] == 10
    assert ep0["start_timestamp"] == pd.Timestamp("2024-01-01")
    assert bool(ep0["y_ever"]) is True  # episode 0's 2nd attempt succeeds
    ep1 = out[out["episode_id"] == 1].iloc[0]
    assert bool(ep1["y_ever"]) is False  # episode 1 never succeeds


def test_fit_global_class_freq_sums_to_one():
    df = pd.DataFrame({"outcome_code": [0, 0, 1, 2, 3]})
    freq = fit_global_class_freq(df, "outcome_code", n_classes=4)
    assert freq.shape == (4,)
    assert np.isclose(freq.sum(), 1.0)
    assert np.isclose(freq[0], 0.4)


def test_fit_global_class_freq_raises_on_empty():
    df = pd.DataFrame({"outcome_code": []})
    with pytest.raises(ValueError):
        fit_global_class_freq(df, "outcome_code", n_classes=4)


def test_per_problem_base_rate_shrinks_toward_global_and_falls_back_for_unseen_problem():
    global_freq = np.array([0.25, 0.25, 0.25, 0.25])
    # problem 10: all class 0 -> with shrinkage_k=5 the estimate is pulled toward uniform
    train_df = pd.DataFrame({"problem_id": [10, 10, 10], "outcome_code": [0, 0, 0]})
    table = fit_per_problem_base_rate(train_df, "outcome_code", "problem_id", 4, global_freq,
                                      shrinkage_k=5.0)
    probs10 = table[10]
    assert np.isclose(probs10.sum(), 1.0)
    assert 0.25 < probs10[0] < 1.0  # pulled toward but not equal to uniform

    val_df = pd.DataFrame({"problem_id": [10, 999]})  # 999 never seen in train
    out = per_problem_base_rate_probs(val_df, table, global_freq, "problem_id")
    assert np.allclose(out[0], probs10)
    assert np.allclose(out[1], global_freq)  # unseen problem falls back to global


def _episoded_attempts():
    # 4 episodes: L1's episode 0 (problem 51, held-out) and episode 1 (problem 20, kept);
    # L2's episode 2 (problem 51, held-out) and episode 3 (problem 21, kept). L3 is not in
    # `learners` at all -- must be untouched by either stripping function.
    return pd.DataFrame({
        "episode_id": [0, 1, 2, 3, 4],
        "learner": ["L1", "L1", "L2", "L2", "L3"],
        "problem_id": [51, 20, 51, 21, 51],
        "timestamp": pd.to_datetime(["2024-01-01"] * 5),
    })


def test_strip_held_out_items_drops_only_held_items_for_target_learners():
    attempts = _episoded_attempts()
    out = strip_held_out_items(attempts, held_out_items={51}, learners={"L1", "L2"})
    # episodes 0 and 2 (problem 51, L1/L2) dropped; episode 4 (problem 51, L3) kept
    # because L3 is not a target learner.
    assert set(out["episode_id"]) == {1, 3, 4}
    assert 51 not in out.loc[out["learner"].isin(["L1", "L2"]), "problem_id"].to_numpy()


def test_strip_held_out_items_never_touches_non_target_learners():
    attempts = _episoded_attempts()
    out = strip_held_out_items(attempts, held_out_items={51}, learners={"L1"})
    # L2's own problem-51 episode (id 2) survives; only L1's is stripped.
    assert 2 in set(out["episode_id"])
    assert 0 not in set(out["episode_id"])


def test_strip_held_out_items_no_op_when_no_held_items_present():
    attempts = _episoded_attempts()
    out = strip_held_out_items(attempts, held_out_items={9999}, learners={"L1", "L2"})
    assert len(out) == len(attempts)


def test_volume_matched_removal_removes_exact_episode_count_deterministically():
    attempts = _episoded_attempts()
    out1 = volume_matched_removal(attempts, learners={"L1", "L2"}, n_episodes_to_remove=2, seed=0)
    # L3's episode (4) is untouched regardless of the random draw.
    assert 4 in set(out1["episode_id"])
    n_removed_from_targets = len({0, 1, 2, 3} - set(out1["episode_id"]))
    assert n_removed_from_targets == 2

    out2 = volume_matched_removal(attempts, learners={"L1", "L2"}, n_episodes_to_remove=2, seed=0)
    assert set(out1["episode_id"]) == set(out2["episode_id"])  # deterministic, same seed


def test_volume_matched_removal_raises_when_asked_to_remove_more_than_available():
    attempts = _episoded_attempts()
    with pytest.raises(ValueError):
        volume_matched_removal(attempts, learners={"L1", "L2"}, n_episodes_to_remove=99, seed=0)


# ---------------------------------------------------------------------------------------
# The strict cell is defined at ROW level. Restricting LEARNERS and then scoring ALL of
# their attempts gives arithmetically impossible row counts for a 7-item cell -- these
# tests exercise the row-level mask directly, and the attempt-1-vs-first-exposure
# DIVERGENCE that a learner-level definition cannot express.
# ---------------------------------------------------------------------------------------

def test_strict_cell_learners_qualifies_a_learner_whose_last_event_is_the_held_touch():
    attempts = pd.DataFrame({
        "learner": ["A", "A"],
        "problem_id": [10, 20],  # 20 held, and it is A's LAST event -> qualifies
        "timestamp": pd.to_datetime(["2024-01-01", "2024-01-02"]),
    })
    qualifying = strict_cell_learners(attempts, held_out_items={20, 30}, candidate_learners={"A"})
    assert qualifying == {"A"}


def test_strict_cell_learners_disqualifies_a_learner_with_a_real_non_held_tail():
    attempts = pd.DataFrame({
        "learner": ["A", "A", "B", "B"],
        "problem_id": [20, 99, 20, 30],  # A's last event (99) is non-held -> disqualified
        "timestamp": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-01", "2024-01-02"]),
    })
    qualifying = strict_cell_learners(attempts, held_out_items={20, 30}, candidate_learners={"A", "B"})
    assert qualifying == {"B"}


def test_strict_cell_row_mask_restricts_to_held_item_rows_of_qualifying_learners_only():
    """The defect-3 regression test directly: a qualifying learner's NON-held rows must
    NOT enter the strict cell (the old code's `evaluate_on_learners`-on-all-rows bug)."""
    problem_id = np.array([20, 99, 30, 20])       # learner A: held, non-held; learner B: held, held
    learner = np.array(["A", "A", "B", "B"])
    prior_same_problem_count = np.array([0, 0, 0, 0])
    attempt_within_idx = np.array([0, 0, 0, 0])
    masks = strict_cell_row_mask(problem_id, learner, prior_same_problem_count, attempt_within_idx,
                                 held_out_items={20, 30}, qualifying_learners={"A", "B"})
    # row 1 (A, problem 99) is non-held -> excluded even though A qualifies as a learner.
    assert masks["base_mask"].tolist() == [True, False, True, True]


def test_strict_cell_row_mask_excludes_non_qualifying_learners_even_on_held_items():
    problem_id = np.array([20, 20])
    learner = np.array(["A", "C"])  # C never qualified as a learner
    prior_same_problem_count = np.array([0, 0])
    attempt_within_idx = np.array([0, 0])
    masks = strict_cell_row_mask(problem_id, learner, prior_same_problem_count, attempt_within_idx,
                                 held_out_items={20}, qualifying_learners={"A"})
    assert masks["base_mask"].tolist() == [True, False]


def test_strict_cell_row_mask_attempt1_and_first_exposure_diverge_on_a_revisit():
    """The property the old learner-level design could not even express: a learner who
    revisits a held item in a LATER, separate episode has attempt_within_idx==0 there
    (episode-local first attempt) but prior_same_problem_count > 0 (NOT their first-ever
    exposure to the problem) -- attempt-1 and first-exposure must give DIFFERENT masks."""
    problem_id = np.array([20, 20])         # same learner, same held problem, two episodes
    learner = np.array(["A", "A"])
    prior_same_problem_count = np.array([0, 1])  # second row is a revisit
    attempt_within_idx = np.array([0, 0])         # but BOTH are attempt-1 of their own episode
    masks = strict_cell_row_mask(problem_id, learner, prior_same_problem_count, attempt_within_idx,
                                 held_out_items={20}, qualifying_learners={"A"})
    assert masks["attempt1_mask"].tolist() == [True, True]
    assert masks["first_exposure_mask"].tolist() == [True, False]


def test_attach_prior_attempt_code_first_attempt_gets_none():
    """An episode's first attempt (attempt_within_idx==0)
    must get `prior_code=None`/NaN, never an accidentally-empty string or a leak from a
    DIFFERENT episode's last attempt."""
    attempts = pd.DataFrame({
        "episode_id": [10, 10, 10, 20],
        "code": ["int x=0;", "int x=1;", "int x=2;", "int y=9;"],
    })
    out = attach_prior_attempt_code(attempts)
    assert out["attempt_within_idx"].tolist() == [0, 1, 2, 0]
    assert pd.isna(out.loc[out["episode_id"].eq(10) & out["attempt_within_idx"].eq(0), "prior_code"].iloc[0])
    assert pd.isna(out.loc[3, "prior_code"])  # episode 20's own first attempt


def test_attach_prior_attempt_code_reads_the_immediately_preceding_attempt_in_the_same_episode():
    attempts = pd.DataFrame({
        "episode_id": [10, 10, 10],
        "code": ["v0", "v1", "v2"],
    })
    out = attach_prior_attempt_code(attempts)
    assert out["prior_code"].tolist()[1:] == ["v0", "v1"]


def test_attach_prior_attempt_code_never_leaks_across_episode_boundaries():
    """The regression this exists to prevent: a naive `shift(1)` over the WHOLE frame
    (no groupby) would hand episode B's first attempt episode A's LAST code -- exactly
    the shape of leak `attach_prior_attempt_code` must not have."""
    attempts = pd.DataFrame({
        "episode_id": [10, 10, 20, 20],
        "code": ["a0", "a1", "b0", "b1"],
    })
    out = attach_prior_attempt_code(attempts)
    assert pd.isna(out.loc[2, "prior_code"])  # episode 20's first attempt, NOT "a1"
    assert out.loc[3, "prior_code"] == "b0"
