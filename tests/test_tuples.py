"""Unit tests for event tuples + learner splits + leakage audits (synthetic, fast)."""

import pandas as pd

from atom.data.tuples import (
    EVENT_COLUMNS,
    apply_split,
    audit_disjoint,
    audit_monotonic_time,
    audit_split_coverage,
    build_events,
    hash_learner,
    inject_leak,
    make_learner_split,
    select_lm_subsample,
)


def _toy() -> pd.DataFrame:
    # u1: three events out of chronological order; u2: two events with tied timestamp
    # (id breaks the tie); u3: one valid + one NaT-timestamp (dropped); u4: one event.
    return pd.DataFrame({
        "user_id": ["u1", "u1", "u1", "u2", "u2", "u3", "u3", "u4"],
        "id":      [12, 10, 11, 21, 20, 30, 31, 40],
        "problem_id": [1, 2, 3, 4, 5, 6, 7, 8],
        "end_time": [
            "2020-01-03", "2020-01-01", "2020-01-02",  # u1 out of order
            "2020-02-01", "2020-02-01",                # u2 tie
            "2020-03-01", "",                          # u3 second is NaT
            "2020-04-01",
        ],
        "discrete_score": [1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        "hint_count": [0, 1, 0, 0, 0, 2, 0, 0],
        "saw_answer": [False, True, False, False, False, False, True, False],
    })


def test_hash_learner_stable_and_distinct():
    assert hash_learner("u1") == hash_learner("u1")
    assert len(hash_learner("u1")) == 16
    assert hash_learner("u1") != hash_learner("u2")


def test_build_events_schema_and_nat_drop():
    ev = build_events(_toy())
    assert list(ev.columns) == EVENT_COLUMNS
    assert len(ev) == 7  # one NaT-timestamp row dropped (u3 id31)
    # discrete_score==1 rows surviving the drop: ids 12,11,21,30,40 -> 5 correct
    assert ev["correct"].tolist().count(1) == 5


def test_build_events_orders_by_time_then_id():
    ev = build_events(_toy())
    u1 = hash_learner("u1")
    seq = ev[ev["learner"] == u1].sort_values("t")
    # times: id10=Jan-01, id11=Jan-02, id12=Jan-03 -> chronological ids [10, 11, 12]
    assert seq["id"].tolist() == [10, 11, 12]
    assert seq["t"].tolist() == [0, 1, 2]

    u2 = hash_learner("u2")
    tied = ev[ev["learner"] == u2].sort_values("t")
    assert tied["id"].tolist() == [20, 21]  # tie broken by id


def test_build_events_time_monotonic():
    ev = build_events(_toy())
    assert audit_monotonic_time(ev)["monotonic"]


def test_make_learner_split_deterministic_and_disjoint():
    learners = [hash_learner(f"u{i}") for i in range(100)]
    a = make_learner_split(learners, fracs=(0.8, 0.1, 0.1), seed=0)
    b = make_learner_split(learners, fracs=(0.8, 0.1, 0.1), seed=0)
    assert a.equals(b)  # reproducible
    assert audit_disjoint(a)["disjoint"]
    counts = a["split"].value_counts().to_dict()
    assert counts["train"] == 80 and counts["val"] == 10 and counts["test"] == 10


def test_make_learner_split_rejects_bad_fracs():
    import pytest

    with pytest.raises(ValueError):
        make_learner_split(["a", "b"], fracs=(0.5, 0.4, 0.2))


def test_apply_split_covers_all_learners():
    ev = build_events(_toy())
    manifest = make_learner_split(ev["learner"], fracs=(0.5, 0.25, 0.25), seed=0)
    assert audit_split_coverage(ev, manifest)["all_covered"]
    joined = apply_split(ev, manifest)
    assert joined["split"].notna().all()


def test_leak_canary_is_caught():
    learners = [hash_learner(f"u{i}") for i in range(20)]
    manifest = make_learner_split(learners, fracs=(0.5, 0.25, 0.25), seed=1)
    assert audit_disjoint(manifest)["disjoint"]
    leaked = inject_leak(manifest)
    assert not audit_disjoint(leaked)["disjoint"]  # the audit must catch the injected leak


def _events_pool(n_learners: int = 50, events_per_learner: int = 5) -> pd.DataFrame:
    rows = []
    next_id = 0
    for i in range(n_learners):
        for _ in range(events_per_learner):
            rows.append({"learner": f"L{i}", "id": next_id})
            next_id += 1
    return pd.DataFrame(rows)


def test_select_lm_subsample_deterministic_given_seed():
    pool = _events_pool()
    a = select_lm_subsample(pool, frac=0.2, seed=0, n_learners=10)
    b = select_lm_subsample(pool, frac=0.2, seed=0, n_learners=10)
    assert sorted(a["id"]) == sorted(b["id"])


def test_select_lm_subsample_different_seeds_differ():
    pool = _events_pool()
    a = select_lm_subsample(pool, frac=0.2, seed=0, n_learners=10)
    b = select_lm_subsample(pool, frac=0.2, seed=1, n_learners=10)
    assert sorted(a["id"]) != sorted(b["id"])


def test_select_lm_subsample_n_learners_caps_distinct_learners():
    pool = _events_pool(n_learners=50, events_per_learner=5)
    out = select_lm_subsample(pool, frac=1.0, seed=0, n_learners=10)
    assert out["learner"].nunique() == 10


def test_select_lm_subsample_frac_controls_event_count():
    pool = _events_pool(n_learners=1, events_per_learner=100)
    out = select_lm_subsample(pool, frac=0.3, seed=0, n_learners=None)
    assert len(out) == 30


def test_select_lm_subsample_frac_one_takes_everything():
    pool = _events_pool(n_learners=5, events_per_learner=4)
    out = select_lm_subsample(pool, frac=1.0, seed=0, n_learners=None)
    assert sorted(out["id"]) == sorted(pool["id"])


def test_select_lm_subsample_frac_zero_takes_nothing():
    pool = _events_pool()
    out = select_lm_subsample(pool, frac=0.0, seed=0, n_learners=None)
    assert len(out) == 0


def test_select_lm_subsample_no_n_learners_draws_from_full_pool():
    pool = _events_pool(n_learners=50, events_per_learner=5)
    out = select_lm_subsample(pool, frac=0.1, seed=0, n_learners=None)
    assert out["learner"].nunique() > 10  # not capped, spread across the whole pool
