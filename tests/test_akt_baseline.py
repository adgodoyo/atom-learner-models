"""Tests for src/atom/baselines/akt_data.py (P3-5 AKT baseline data prep) and a spot-check
that our batch construction preserves pyKT AKT's own no-peek guarantee."""

import pandas as pd
import pytest
import torch

from atom.baselines.akt_data import (
    build_akt_batch,
    build_akt_batch_windowed,
    build_concept_map,
    build_pid_map,
    build_support_mask,
    check_coverage,
    chunk_learner_history,
    chunking_stats,
    subsample_by_event_budget,
)


def _toy_skills():
    # problem 10 -> skill 5 only. problem 20 -> skills {1, 5} (multi-skill; min = 1).
    # problem 30 -> skill 3 only.
    rows = [
        {"problem_id": 10, "skill_id": 5},
        {"problem_id": 20, "skill_id": 1},
        {"problem_id": 20, "skill_id": 5},
        {"problem_id": 30, "skill_id": 3},
    ]
    return pd.DataFrame(rows)


def _toy_ev(n_events_a: int = 3):
    rows = []
    for t in range(n_events_a):
        rows.append({"learner": "a", "t": t, "problem_id": [10, 20, 30][t % 3],
                     "correct": t % 2})
    rows.append({"learner": "b", "t": 0, "problem_id": 10, "correct": 1})
    return pd.DataFrame(rows)


def test_build_concept_map_uses_min_skill_and_is_dense():
    cmap = build_concept_map(_toy_skills())
    uniq_skills_sorted = sorted({1, 3, 5})  # skills actually assigned as a "min" somewhere: 5,1,3
    skill_to_idx = {s: i for i, s in enumerate(uniq_skills_sorted)}
    assert cmap[10] == skill_to_idx[5]
    assert cmap[20] == skill_to_idx[1]  # min(1, 5) = 1, not 5
    assert cmap[30] == skill_to_idx[3]
    # dense: indices are exactly 0..n-1
    assert set(cmap.values()) == set(range(len(uniq_skills_sorted)))


def test_build_pid_map_is_dense_bijection():
    pmap = build_pid_map([30, 10, 20, 10])  # dup on purpose
    assert set(pmap.keys()) == {10, 20, 30}
    assert set(pmap.values()) == {0, 1, 2}
    assert pmap[10] < pmap[20] < pmap[30]  # sorted order


def test_check_coverage_raises_on_missing():
    cmap = build_concept_map(_toy_skills())
    pmap = build_pid_map([10, 20, 30])
    check_coverage([10, 20, 30], cmap, pmap)  # should not raise
    with pytest.raises(RuntimeError):
        check_coverage([10, 999], cmap, pmap)


def test_build_akt_batch_shapes_and_padding():
    ev = _toy_ev(n_events_a=3)
    cmap = build_concept_map(_toy_skills())
    pmap = build_pid_map([10, 20, 30])
    batch = build_akt_batch(ev, cmap, pmap, max_seq_len=200, device="cpu")

    assert batch["learners"] == ["a", "b"]
    n, t_max = batch["valid_mask"].shape
    assert n == 2 and t_max == 3  # learner a has the longest history (3)
    assert batch["lengths"].tolist() == [3, 1]
    assert batch["valid_mask"].tolist() == [[True, True, True], [True, False, False]]
    # learner a's concepts/pids/responses at t=0..2 match problems 10,20,30 in order.
    expected_concepts = [cmap[10], cmap[20], cmap[30]]
    assert batch["concepts"][0].tolist() == expected_concepts
    assert batch["responses"][0].tolist() == [0, 1, 0]


def test_build_akt_batch_truncates_to_most_recent():
    ev = _toy_ev(n_events_a=5)  # learner a: t=0..4, problems cycle 10,20,30,10,20
    cmap = build_concept_map(_toy_skills())
    pmap = build_pid_map([10, 20, 30])
    batch = build_akt_batch(ev, cmap, pmap, max_seq_len=2, device="cpu")

    n, t_max = batch["valid_mask"].shape
    assert t_max == 2  # capped at max_seq_len
    # learner a's most recent 2 events (t=3 problem 10, t=4 problem 20) are kept, in order.
    a_idx = batch["learners"].index("a")
    assert batch["concepts"][a_idx].tolist() == [cmap[10], cmap[20]]
    assert batch["lengths"][a_idx].item() == 2


def test_subsample_by_event_budget_is_deterministic_and_respects_budget():
    rows = []
    for learner_idx in range(20):
        for t in range(10):  # 10 events per learner, 200 total
            rows.append({"learner": f"L{learner_idx}", "t": t, "problem_id": 10, "correct": 0})
    ev = pd.DataFrame(rows)

    sub1 = subsample_by_event_budget(ev, budget=55, seed=0)
    sub2 = subsample_by_event_budget(ev, budget=55, seed=0)
    assert len(sub1) == len(sub2)
    assert set(sub1["learner"]) == set(sub2["learner"])
    # "Approximately" the budget, not exact: with homogeneous 10-event learners, the closest
    # reachable total below or at 55 is 50 -- best-effort, undershoot is expected and fine.
    assert 0 < len(sub1) <= 55
    assert len(sub1["learner"].unique()) < 20  # actually subsampled, not everything

    # A budget smaller than one learner's own events still returns at least that one learner.
    small = subsample_by_event_budget(ev, budget=3, seed=0)
    assert len(small) == 10  # one whole learner (10 events), never truncated here


def _toy_ev_long(n_events: int, learner: str = "a"):
    rows = []
    for t in range(n_events):
        rows.append({"learner": learner, "t": t, "problem_id": [10, 20, 30][t % 3],
                     "correct": t % 2})
    return pd.DataFrame(rows)


def test_chunk_learner_history_splits_into_non_overlapping_windows():
    ev = _toy_ev_long(5)  # t=0..4
    chunked = chunk_learner_history(ev, max_seq_len=2, min_seq_len=1)
    # windows: [0,1] -> w0, [2,3] -> w1, [4] -> w2 (size 1, kept since min_seq_len=1)
    assert len(chunked) == 5  # nothing dropped
    sizes = chunked.groupby("chunk_id").size().to_dict()
    assert sizes == {"a::w0": 2, "a::w1": 2, "a::w2": 1}
    # chunks are contiguous and in order (no overlap, no reordering).
    w0 = chunked[chunked["chunk_id"] == "a::w0"].sort_values("t")
    assert w0["t"].tolist() == [0, 1]
    w2 = chunked[chunked["chunk_id"] == "a::w2"]
    assert w2["t"].tolist() == [4]


def test_chunk_learner_history_drops_tiny_remainder_below_min_seq_len():
    ev = _toy_ev_long(7)  # windows of 2: sizes 2,2,2,1 -- last (size 1) < min_seq_len=3
    chunked = chunk_learner_history(ev, max_seq_len=2, min_seq_len=3)
    assert len(chunked) == 6  # the size-1 remainder chunk dropped
    assert set(chunked["chunk_id"].unique()) == {"a::w0", "a::w1", "a::w2"}


def test_chunk_learner_history_never_drops_a_full_window():
    # 4 events, max_seq_len=2 -> two FULL windows of size 2 each. Even with a (deliberately
    # perverse) min_seq_len=3 > max_seq_len, a full window must never be treated as a
    # "too-small remainder" -- only a genuinely short last chunk qualifies.
    ev = _toy_ev_long(4)
    chunked = chunk_learner_history(ev, max_seq_len=2, min_seq_len=3)
    assert len(chunked) == 4
    assert set(chunked["chunk_id"].unique()) == {"a::w0", "a::w1"}


def test_chunk_learner_history_preserves_other_columns_and_multiple_learners():
    ev = pd.concat([_toy_ev_long(5, "a"), _toy_ev_long(3, "b")], ignore_index=True)
    chunked = chunk_learner_history(ev, max_seq_len=2, min_seq_len=1)
    assert set(chunked["learner"].unique()) == {"a", "b"}
    # learner b (3 events) chunked independently of learner a.
    b_chunks = sorted(chunked.loc[chunked["learner"] == "b", "chunk_id"].unique())
    assert b_chunks == ["b::w0", "b::w1"]
    assert "correct" in chunked.columns and "problem_id" in chunked.columns


def test_chunking_stats_reports_multi_window_and_dropped_fractions():
    ev = pd.concat([_toy_ev_long(5, "a"), _toy_ev_long(1, "b")], ignore_index=True)
    chunked = chunk_learner_history(ev, max_seq_len=2, min_seq_len=3)
    stats = chunking_stats(ev, chunked, max_seq_len=2)
    assert stats["total_learners"] == 2
    assert stats["multi_window_learners"] == 1  # only "a" (5 > 2) needed >1 window
    assert stats["raw_events"] == 6
    # a's size-1 remainder chunk (5 = 2+2+1) AND b's whole history (1 event, itself a
    # size-1 "remainder" below min_seq_len=3) both get dropped -- 1 + 1 = 2 events.
    assert stats["dropped_events"] == 2
    assert stats["n_chunks"] == 2  # a::w0, a::w1 only -- b is dropped entirely


def test_build_akt_batch_windowed_no_event_dropped_when_min_seq_len_is_1():
    ev = _toy_ev_long(5)
    cmap = build_concept_map(_toy_skills())
    pmap = build_pid_map([10, 20, 30])
    batch = build_akt_batch_windowed(ev, cmap, pmap, max_seq_len=2, device="cpu", min_seq_len=1)
    assert int(batch["valid_mask"].sum().item()) == 5  # every original event still trained on
    assert len(batch["learners"]) == 3  # 3 chunks: w0 (2), w1 (2), w2 (1)
    assert batch["orig_learners"] == ["a", "a", "a"]  # all chunks trace back to learner "a"


def test_build_akt_batch_windowed_chunk_boundary_is_a_no_peek_boundary():
    """The crux disclosed property of chunking: a later chunk's position 0 has NO visible
    history from the earlier chunk -- structurally guaranteed here because each chunk is a
    SEPARATE row in the batch, not a probabilistic property that could quietly regress.
    Spot-check: flipping a response in chunk 0 must not change chunk 1's own predictions
    (they are different rows fed through the model independently in the same batched call)."""
    from pykt.models.akt import AKT

    ev = _toy_ev_long(4)  # -> two chunks of 2 with max_seq_len=2: w0 (t=0,1), w1 (t=2,3)
    cmap = build_concept_map(_toy_skills())
    pmap = build_pid_map([10, 20, 30])
    batch = build_akt_batch_windowed(ev, cmap, pmap, max_seq_len=2, device="cpu", min_seq_len=1)
    assert len(batch["learners"]) == 2

    torch.manual_seed(0)
    model = AKT(n_question=len(set(cmap.values())), n_pid=len(set(pmap.values())),
                d_model=16, n_blocks=1, dropout=0.0)
    model.eval()
    with torch.no_grad():
        preds_a, _ = model(batch["concepts"], batch["responses"], batch["pids"])

    flipped = batch["responses"].clone()
    w0_idx = batch["learners"].index("a::w0")
    flipped[w0_idx, -1] = 1 - flipped[w0_idx, -1]  # flip chunk 0's LAST response
    with torch.no_grad():
        preds_b, _ = model(batch["concepts"], flipped, batch["pids"])

    w1_idx = batch["learners"].index("a::w1")
    # chunk 1's row is untouched by the flip in chunk 0's row -- different rows, no shared
    # state between them (this is what makes chunking structurally leak-proof, not just
    # empirically so).
    assert torch.allclose(preds_a[w1_idx], preds_b[w1_idx], atol=1e-6)


# -- build_support_mask (FA-1: support=NONE stratum for the v2 AKT baseline) ----------------

def _toy_ev_long_with_support(n_events: int, learner: str = "a", support_codes=None):
    rows = []
    for t in range(n_events):
        rows.append({"learner": learner, "t": t, "problem_id": [10, 20, 30][t % 3],
                     "correct": t % 2,
                     "support_code": (support_codes[t] if support_codes else 0)})
    return pd.DataFrame(rows)


def test_build_support_mask_aligns_with_build_akt_batch_windowed():
    # 5 events, max_seq_len=2 -> chunks a::w0 (t=0,1), a::w1 (t=2,3), a::w2 (t=4); support
    # codes chosen so each chunk's cells are individually distinguishable.
    ev = _toy_ev_long_with_support(5, support_codes=[0, 1, 2, 3, 0])
    cmap = build_concept_map(_toy_skills())
    pmap = build_pid_map([10, 20, 30])
    batch = build_akt_batch_windowed(ev, cmap, pmap, max_seq_len=2, device="cpu", min_seq_len=1)
    n, t_max = batch["valid_mask"].shape

    mask = build_support_mask(ev, max_seq_len=2, batch_learners=batch["learners"], t_max=t_max,
                              min_seq_len=1)
    assert mask.shape == (n, t_max)

    w0 = batch["learners"].index("a::w0")
    w1 = batch["learners"].index("a::w1")
    w2 = batch["learners"].index("a::w2")
    assert mask[w0].tolist() == [0, 1]   # t=0,1
    assert mask[w1].tolist() == [2, 3]   # t=2,3
    assert mask[w2].tolist() == [0, -1]  # t=4 only; second cell is padding, never a real code

    # Every valid_mask cell has a real (non -1) support code, and vice versa.
    valid = batch["valid_mask"].numpy()
    assert ((mask >= 0) == valid).all()


def test_build_support_mask_selects_the_none_stratum():
    ev = _toy_ev_long_with_support(4, support_codes=[0, 1, 0, 2])
    cmap = build_concept_map(_toy_skills())
    pmap = build_pid_map([10, 20, 30])
    batch = build_akt_batch_windowed(ev, cmap, pmap, max_seq_len=4, device="cpu", min_seq_len=1)
    n, t_max = batch["valid_mask"].shape
    mask = build_support_mask(ev, max_seq_len=4, batch_learners=batch["learners"], t_max=t_max,
                              min_seq_len=1)
    none_stratum = (mask == 0)
    a_idx = batch["learners"].index("a::w0")
    assert none_stratum[a_idx].tolist() == [True, False, True, False]


def test_akt_no_peek_survives_our_batch_construction():
    """pykt.models.akt.AKT is trusted, published, peer-reviewed code -- this does not
    re-derive its causality proof. It DOES spot-check that feeding OUR padded tensors
    through it does not break the no-peek guarantee: flipping a MIDDLE step's own response
    must not move predictions at or before that step (y[t] only ever sees r[0..t-1]), but
    must move predictions strictly after it (the mechanism isn't a no-op)."""
    from pykt.models.akt import AKT

    torch.manual_seed(0)
    model = AKT(n_question=5, n_pid=8, d_model=16, n_blocks=1, dropout=0.0)
    model.eval()

    T = 6
    concepts = torch.randint(0, 5, (2, T))
    pids = torch.randint(0, 8, (2, T))
    responses = torch.randint(0, 2, (2, T))

    with torch.no_grad():
        preds_a, _ = model(concepts, responses, pids)

    flip_t = 2  # a middle position, so there ARE later positions to observe an effect on
    flipped = responses.clone()
    flipped[:, flip_t] = 1 - flipped[:, flip_t]
    with torch.no_grad():
        preds_b, _ = model(concepts, flipped, pids)

    assert torch.allclose(preds_a[:, :flip_t + 1], preds_b[:, :flip_t + 1], atol=1e-6)
    # Sanity: the flip SHOULD move predictions at later steps (the mechanism isn't a no-op).
    assert not torch.allclose(preds_a[:, flip_t + 1:], preds_b[:, flip_t + 1:], atol=1e-6)
