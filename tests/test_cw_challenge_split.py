from atom.data.cw_challenge_split import (
    stratified_val_carve,
    verify_challenge_partition,
    verify_item_holdout,
)
from atom.data.tuples import hash_learner


def test_verify_challenge_partition_clean_case_passes_every_check():
    raw = {"a1": "t1", "a2": "t2", "a3": "e1", "a4": "e2", "a5": "u1"}
    subject_ids = {
        "f19_all": set(raw.values()),
        "f19_train": {"t1", "t2"},
        "f19_test": {"e1", "e2"},
    }
    f19_learners = {hash_learner(v) for v in raw.values()}
    out = verify_challenge_partition(subject_ids, f19_learners)
    assert out["n_train"] == 2 and out["n_test"] == 2 and out["n_unassigned"] == 1
    assert out["train_matched"] == 2 and out["test_matched"] == 2 and out["unassigned_matched"] == 1
    assert out["pairwise_disjoint"] is True
    assert out["union_equals_f19_learners"] is True


def test_verify_challenge_partition_catches_a_learner_dropped_from_f19_learners():
    subject_ids = {
        "f19_all": {"t1", "t2", "e1"},
        "f19_train": {"t1", "t2"},
        "f19_test": {"e1"},
    }
    # f19_learners is missing t2's hash -- simulates a real population that doesn't
    # actually cover the challenge file's claimed subjects.
    f19_learners = {hash_learner("t1"), hash_learner("e1")}
    out = verify_challenge_partition(subject_ids, f19_learners)
    assert out["train_matched"] == 1  # only t1 matched, not 2
    assert out["union_equals_f19_learners"] is False


def test_verify_challenge_partition_catches_train_test_overlap():
    subject_ids = {
        "f19_all": {"t1", "t2"},
        "f19_train": {"t1", "t2"},
        "f19_test": {"t2"},  # t2 wrongly appears in both
    }
    f19_learners = {hash_learner("t1"), hash_learner("t2")}
    out = verify_challenge_partition(subject_ids, f19_learners)
    assert out["pairwise_disjoint"] is False


def test_stratified_val_carve_respects_per_stratum_fraction_and_is_deterministic():
    strata = {
        "S19": [f"s{i}" for i in range(100)],
        "F19": [f"f{i}" for i in range(50)],
    }
    out1 = stratified_val_carve(strata, val_fraction=0.2, seed=0)
    n_val_s19 = sum(1 for k, v in out1.items() if k.startswith("s") and v == "val")
    n_val_f19 = sum(1 for k, v in out1.items() if k.startswith("f") and v == "val")
    assert n_val_s19 == 20  # round(100 * 0.2)
    assert n_val_f19 == 10  # round(50 * 0.2)
    assert len(out1) == 150

    out2 = stratified_val_carve(strata, val_fraction=0.2, seed=0)
    assert out1 == out2  # deterministic, same seed


def test_stratified_val_carve_different_seed_gives_different_val_set():
    strata = {"S19": [f"s{i}" for i in range(50)]}
    out_a = stratified_val_carve(strata, val_fraction=0.3, seed=0)
    out_b = stratified_val_carve(strata, val_fraction=0.3, seed=1)
    assert out_a != out_b  # not asserting exact disagreement count, just that seed matters


def test_stratified_val_carve_every_train_pool_learner_appears_exactly_once():
    strata = {"S19": ["a", "b", "c"], "F19": ["x", "y"]}
    out = stratified_val_carve(strata, val_fraction=0.4, seed=0)
    assert set(out) == {"a", "b", "c", "x", "y"}
    assert all(v in ("train", "val") for v in out.values())


def test_verify_item_holdout_passes_on_a_clean_disjoint_set():
    out = verify_item_holdout((51, 56, 57), tiktoc_ids=[1, 3, 5], catalog_ids=set(range(1, 60)))
    assert out["disjoint_from_tiktoc_17"] is True
    assert out["overlap_with_tiktoc_17"] == []
    assert out["all_in_catalog"] is True
    assert out["missing_from_catalog"] == []


def test_verify_item_holdout_catches_tiktoc_overlap():
    out = verify_item_holdout((51, 56, 71), tiktoc_ids=[71, 3, 5], catalog_ids=set(range(1, 60)))
    assert out["disjoint_from_tiktoc_17"] is False
    assert out["overlap_with_tiktoc_17"] == [71]


def test_verify_item_holdout_catches_an_id_missing_from_the_catalog():
    out = verify_item_holdout((51, 999), tiktoc_ids=[], catalog_ids=set(range(1, 60)))
    assert out["all_in_catalog"] is False
    assert out["missing_from_catalog"] == [999]
