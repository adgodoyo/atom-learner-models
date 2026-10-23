"""FA-0: canonical FA manifest rebuild -- synthetic unit tests (fast, no dataset required).

Real-data reconciliation (exact dedup/missing counts against the frozen gate targets) is
`scripts/fa_00_manifest_rebuild.py`'s job; these tests exercise each pipeline step and every
mandatory peek canary against small, hand-constructed frames.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from atom.data.fa_manifest import (
    assert_unique_ids,
    attach_splits,
    audit_lockbox_isolated,
    audit_missing_time_excluded,
    audit_monotonic_time,
    audit_no_duplicate_ids,
    audit_score_never_imputed,
    audit_split_disjoint,
    build_canonical_events,
    build_problem_catalog,
    dedup_interactions,
    inject_duplicate_row,
    inject_split_leak,
    load_v2_events,
    lockbox_hash,
    make_item_split_v2,
    make_learner_split_v2,
    resolve_score,
    support_label,
)


# -- fixtures ------------------------------------------------------------------

def _raw_interactions() -> pd.DataFrame:
    """8 rows: a clean stream + one exact-duplicate-id pair + one missing-time row."""
    return pd.DataFrame({
        "id": [1, 2, 3, 4, 4, 5, 6, 7],  # id=4 duplicated (fully identical row)
        "problem_id": [10, 10, 11, 12, 12, 10, 11, 10],
        "hint_count": [0, 1, 0, 2, 2, 0, 0, 0],
        "answer_text": ["a", "b", "c", "d", "d", "e", "f", "g"],
        "saw_answer": [False, False, True, False, False, False, True, False],
        "discrete_score": [1.0, 0.0, np.nan, 0.0, 0.0, 1.0, np.nan, 1.0],
        "end_time": [
            "2026-01-01T00:00:00Z", "2026-01-01T00:05:00Z", "2026-01-01T00:10:00Z",
            "2026-01-01T00:15:00Z", "2026-01-01T00:15:00Z", None,
            "2026-01-01T00:20:00Z", "2026-01-02T00:00:00Z",
        ],
        "user_id": ["u1", "u1", "u1", "u1", "u1", "u1", "u2", "u2"],
    })


def _problem_catalog() -> pd.DataFrame:
    problems = pd.DataFrame({
        "problem_id": [10, 11, 12],
        "problem_body_clean": ["what is 2+2?", "what is 2+2?", "name a prime"],
        "problem_type": ["Fill-in-the-blank(s)", "Fill-in-the-blank(s)", "Fill-in-the-blank(s)"],
    })
    return build_problem_catalog(problems)


# -- dedup -----------------------------------------------------------------

def test_dedup_drops_exact_duplicate_id_only():
    raw = _raw_interactions()
    out, stats = dedup_interactions(raw)
    assert stats["n_before"] == 8
    assert stats["n_dropped"] == 1
    assert stats["n_dup_groups"] == 1
    assert stats["n_rows_in_dup_groups"] == 2
    assert out["id"].tolist() == [1, 2, 3, 4, 5, 6, 7]


def test_assert_unique_ids_passes_after_dedup():
    out, _ = dedup_interactions(_raw_interactions())
    assert_unique_ids(out)  # must not raise


def test_planted_duplicate_canary_is_caught():
    """FA-0 gate: a deliberately re-duplicated row must be caught by the dedup assertion."""
    out, _ = dedup_interactions(_raw_interactions())
    leaked = inject_duplicate_row(out, seed=0)
    with pytest.raises(AssertionError):
        assert_unique_ids(leaked)
    audit = audit_no_duplicate_ids(leaked)
    assert audit["unique_ids"] is False
    assert audit["n_duplicated"] == 2


# -- score resolution --------------------------------------------------------

def test_resolve_score_never_imputes_missing_to_zero():
    s = resolve_score(pd.Series([1.0, 0.0, np.nan, 1.0]))
    assert s.tolist() == [1, 0, pd.NA, 1]
    assert s.isna().sum() == 1


# -- support label ------------------------------------------------------------

def test_support_label_four_classes():
    out = support_label(
        pd.Series([0, 1, 0, 2]), pd.Series([False, False, True, True]),
    )
    assert out["support"].tolist() == ["NONE", "HINT_ONLY", "REVEAL_ONLY", "HINT_AND_REVEAL"]
    assert out["support_code"].tolist() == [0, 1, 2, 3]


# -- problem catalog / near-duplicate grouping --------------------------------

def test_near_duplicate_problems_share_a_group():
    cat = _problem_catalog()
    g10 = cat.loc[cat["problem_id"] == 10, "problem_group_id"].iloc[0]
    g11 = cat.loc[cat["problem_id"] == 11, "problem_group_id"].iloc[0]
    g12 = cat.loc[cat["problem_id"] == 12, "problem_group_id"].iloc[0]
    assert g10 == g11  # identical cleaned body text
    assert g10 != g12


# -- canonical event manifest --------------------------------------------------

def _build():
    raw, _ = dedup_interactions(_raw_interactions())
    cat = _problem_catalog()
    return build_canonical_events(raw, cat)


def test_canonical_events_row_count_and_missing_time_excluded():
    ev, stats = _build()
    assert stats["n_events"] == 7  # 8 raw - 1 exact duplicate
    assert stats["n_missing_time"] == 1
    assert stats["n_valid_time"] == 6
    missing_row = ev[~ev["has_valid_time"]]
    assert len(missing_row) == 1
    assert missing_row["t"].isna().all()
    assert audit_missing_time_excluded(ev)["missing_time_excluded"] is True


def test_score_missing_never_imputed_in_manifest():
    ev, _ = _build()
    assert audit_score_never_imputed(ev)["score_never_imputed"] is True
    assert ev.loc[ev["id"] == 3, "score"].isna().all()  # NaN discrete_score, saw_answer=True


def test_t_and_gaps_computed_only_over_valid_time_rows_per_learner():
    ev, _ = _build()
    u1_learner = ev.loc[ev["id"] == 1, "learner"].iloc[0]
    u1 = ev[(ev["learner"] == u1_learner) & ev["has_valid_time"]].sort_values("t")
    assert u1["id"].tolist() == [1, 2, 3, 4]  # id=5 (same learner) has missing time, excluded
    assert u1["t"].tolist() == [0, 1, 2, 3]
    assert u1["first_event"].tolist() == [True, False, False, False]
    assert u1["missing_gap"].tolist() == [True, False, False, False]
    # 5-minute gap between the first two valid events -> log1p(5)
    assert u1["gap_log1p_min"].iloc[1] == pytest.approx(np.log1p(5.0))


def test_first_exposure_flags_repeated_problem_within_learner():
    ev, _ = _build()
    same_problem = ev[(ev["problem_id"] == 10) & ev["has_valid_time"]].sort_values("t")
    # learner u1 sees problem 10 at t=0 (first) then again at t=... (not first)
    assert same_problem["first_exposure"].tolist()[0] == True  # noqa: E712
    assert same_problem["first_exposure"].tolist().count(False) >= 1


def test_monotonic_time_audit_passes_on_clean_stream():
    ev, _ = _build()
    assert audit_monotonic_time(ev)["monotonic"] is True


# -- mandatory peek canaries (design doc §9.5) ---------------------------------

def test_duplicated_id_cannot_appear_twice_in_a_learner_sequence():
    ev, _ = _build()
    assert audit_no_duplicate_ids(ev)["unique_ids"] is True


def test_missing_success_is_missing_not_negative():
    ev, _ = _build()
    row = ev.loc[ev["id"] == 3]
    assert row["score_missing"].iloc[0]
    assert pd.isna(row["score"].iloc[0])


def test_removing_current_event_does_not_change_history_representation():
    """Dropping the last (chronologically final) event must not change earlier rows' t/gaps."""
    raw, _ = dedup_interactions(_raw_interactions())
    cat = _problem_catalog()
    ev_full, _ = build_canonical_events(raw, cat)

    raw_trunc = raw[raw["id"] != 7].copy()  # id=7 is the globally last valid-time event
    ev_trunc, _ = build_canonical_events(raw_trunc, cat)

    key_cols = ["t", "gap_log1p_min", "first_event", "first_exposure"]
    common_ids = set(ev_full["id"]) & set(ev_trunc["id"])
    a = ev_full[ev_full["id"].isin(common_ids)].sort_values("id")[key_cols].reset_index(drop=True)
    b = ev_trunc[ev_trunc["id"].isin(common_ids)].sort_values("id")[key_cols].reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)


# -- two-axis split + lockbox ---------------------------------------------------

def _old_manifest():
    return pd.DataFrame({
        "learner": [f"L{i}" for i in range(10)],
        "split": ["train"] * 6 + ["val"] * 2 + ["test"] * 2,
    })


def test_learner_split_v2_preserves_val_and_renames_test():
    manifest, stats = make_learner_split_v2(_old_manifest(), lockbox_frac=1 / 3, seed=0)
    assert set(manifest.loc[manifest["split"] == "val", "learner"]) == {"L6", "L7"}
    assert set(manifest.loc[manifest["split"] == "development_exposed", "learner"]) == {"L8", "L9"}
    assert stats["n_lockbox"] == 2  # round(6 * 1/3)
    assert stats["n_train"] == 4
    # lockbox learners come only from the old train pool
    lockbox = set(manifest.loc[manifest["split"] == "lockbox", "learner"])
    assert lockbox <= {f"L{i}" for i in range(6)}
    assert audit_split_disjoint(manifest, "learner")["disjoint"] is True


def test_learner_split_v2_leak_canary_caught():
    manifest, _ = make_learner_split_v2(_old_manifest(), lockbox_frac=1 / 3, seed=0)
    leaked = inject_split_leak(manifest, from_split="lockbox", to_split="train")
    assert audit_split_disjoint(leaked, "learner")["disjoint"] is False


def test_item_split_v2_groups_never_split():
    catalog = pd.DataFrame({
        "problem_group_id": [f"g{i}" for i in range(10)],
        "problem_id": list(range(10)),
    })
    manifest, stats = make_item_split_v2(catalog, heldout_frac=0.3, seed=0)
    assert stats["n_heldout_groups"] == 3
    assert audit_split_disjoint(manifest, "problem_group_id")["disjoint"] is True


def test_lockbox_hash_never_exposes_rows_and_is_seed_stable():
    ev, _ = _build()
    learner_split = pd.DataFrame({
        "learner": ev["learner"].unique(),
        "split": ["lockbox"] * len(ev["learner"].unique()),
    })
    item_split = pd.DataFrame({
        "problem_group_id": ev["problem_group_id"].unique(),
        "split": ["heldout"] * ev["problem_group_id"].nunique(),
    })
    item_split = item_split.rename(columns={"split": "split"})
    combined = attach_splits(
        ev, learner_split.rename(columns={"split": "split"}), item_split,
    )
    result = lockbox_hash(combined)
    assert isinstance(result["sha256"], str) and len(result["sha256"]) == 64
    assert result["n_rows"] == combined.shape[0]
    # deterministic: same input -> same hash
    result2 = lockbox_hash(combined)
    assert result["sha256"] == result2["sha256"]


def test_load_v2_events_restricts_to_valid_time_and_aliases_split(tmp_path):
    ev, _ = _build()
    ev.to_parquet(tmp_path / "manifest.parquet", index=False)
    learners = ev["learner"].unique()
    pd.DataFrame({"learner": learners, "split": ["train"] * len(learners)}).to_csv(
        tmp_path / "learner_split.csv", index=False)
    groups = ev["problem_group_id"].unique()
    pd.DataFrame({"problem_group_id": groups, "split": ["train"] * len(groups)}).to_csv(
        tmp_path / "item_split.csv", index=False)

    out = load_v2_events(tmp_path / "manifest.parquet", tmp_path / "learner_split.csv",
                         tmp_path / "item_split.csv")
    assert out["has_valid_time"].all()
    assert (out["split"] == "train").all()
    assert len(out) == int(ev["has_valid_time"].sum())


def test_audit_lockbox_isolated_flags_inconsistent_learner_split():
    ev, _ = _build()
    bad_split = pd.DataFrame({"learner": ev["learner"].unique(), "split": "lockbox"})
    # deliberately give one learner two different split rows
    extra = bad_split.iloc[[0]].copy()
    extra["split"] = "train"
    bad_split = pd.concat([bad_split, extra], ignore_index=True)
    item_split = pd.DataFrame({
        "problem_group_id": ev["problem_group_id"].unique(),
        "split": "train",
    })
    combined = attach_splits(ev, bad_split, item_split)
    assert audit_lockbox_isolated(combined)["learner_split_consistent"] is False
