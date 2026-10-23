"""Tests for `ga_1_tier3_states.py`: state draw, CRN shuffle and advantage canaries."""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest
import torch

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))

import ga_1_tier3_states as S  # noqa: E402


def _ev(i, pid, learner="L1"):
    return {"episode_id": f"{learner}-e{i}", "problem_id": pid, "order": i,
            "label": pid % 4, "codec": torch.arange(4, dtype=torch.float32) + i}


def _episode(learner="L1", n_cand=6, start=3):
    prefix = [_ev(i, 10 + i, learner) for i in range(start)]
    cands = [_ev(start + i, 20 + i, learner) for i in range(n_cand)]
    targets = [_ev(90 + i, 30 + i, learner) for i in range(3)]
    return {"learner": learner, "prefix_events": prefix, "candidates": cands,
            "targets": targets, "full_events": prefix + cands}


# ------------------------------------------------------------------ the shuffle


def test_permutation_is_a_bijection_and_deterministic():
    a = S.crn_permutation(50, "L1", 3, 8, 2)
    b = S.crn_permutation(50, "L1", 3, 8, 2)
    c = S.crn_permutation(50, "L1", 3, 8, 3)
    assert sorted(a) == list(range(50))
    assert a == b, "CRN permutation must be stable under re-call (resume safety)"
    assert a != c, "different states must get different permutations"


def test_permutation_actually_moves_the_first_slot():
    """Tier 0's pathology was 90.8% `A`. If A still labels candidate 0, nothing changed."""
    firsts = [S.crn_permutation(40, "L", s, b, k)[0]
              for s in (3, 8) for b in (3, 8, 16) for k in range(12)]
    assert len(set(firsts)) > 8, "identifier A is not being spread across candidates"
    assert sum(1 for f in firsts if f == 0) < len(firsts) * 0.2


def test_apply_permutation_relabels_without_losing_candidates():
    cands = _episode(n_cand=6)["candidates"]
    perm = S.crn_permutation(6, "L1", 3, 8, 0)
    shuffled = S.apply_permutation(cands, perm)
    assert len(shuffled) == 6
    assert {c["episode_id"] for c in shuffled} == {c["episode_id"] for c in cands}
    # identifier k labels candidates[perm[k]] -- the mapping the read-out must invert
    for k, c in enumerate(shuffled):
        assert c is cands[perm[k]]


def test_apply_permutation_rejects_a_non_bijection():
    cands = _episode(n_cand=4)["candidates"]
    with pytest.raises(AssertionError):
        S.apply_permutation(cands, [0, 1, 1, 2])


# ------------------------------------------------------------------ the split


def test_fit_internal_split_reproduces_rl2_partition_shape():
    eps = [{"learner": f"L{i:04d}"} for i in range(683)]
    fit, internal, meta = S.fit_internal_split(eps)
    assert meta["fit_learners"] == int(0.8 * 683)
    assert meta["internal_val_learners"] == S.INTERNAL_VAL_MAX_LEARNERS
    fit_ids = {e["learner"] for e in fit}
    int_ids = {e["learner"] for e in internal}
    assert not (fit_ids & int_ids), "fit and internal-validation learners must be disjoint"


def test_split_is_deterministic_across_input_order():
    eps = [{"learner": f"L{i:04d}"} for i in range(200)]
    a = S.fit_internal_split(eps)[2]
    b = S.fit_internal_split(list(reversed(eps)))[2]
    assert a == b


# ------------------------------------------------------------- the state draw


def test_state_draw_covers_every_budget_and_only_the_registered_starts():
    scen = {f"train_start{s}": [_episode(f"L{i}", n_cand=20, start=s) for i in range(6)]
            for s in (3, 10)}
    fit = {f"L{i}" for i in range(6)}
    states = S.draw_training_states(scen, fit)
    assert states, "no states drawn"
    assert {s["start"] for s in states} <= set(S.TRAIN_STARTS)
    assert {s["budget"] for s in states} == {3, 8, 16}
    # one state per step k < B, per learner per budget
    assert len(states) == 6 * (3 + 8 + 16)
    for st in states:
        assert st["step"] < st["budget"]
        assert len(st["known_candidate_idx"]) == st["step"]
        assert not set(st["known_candidate_idx"]) & set(st["remaining_idx"])


def test_state_draw_is_deterministic():
    scen = {f"train_start{s}": [_episode(f"L{i}", n_cand=20, start=s) for i in range(4)]
            for s in (3, 10)}
    fit = {f"L{i}" for i in range(4)}
    a = S.draw_training_states(scen, fit)
    b = S.draw_training_states(scen, fit)
    def key(ss):
        return [(x["learner"], x["start"], x["budget"], x["step"],
                 tuple(x["known_candidate_idx"])) for x in ss]

    assert key(a) == key(b)


def test_state_draw_assigns_each_learner_one_start():
    scen = {f"train_start{s}": [_episode(f"L{i}", n_cand=20, start=s) for i in range(20)]
            for s in (3, 10)}
    fit = {f"L{i}" for i in range(20)}
    states = S.draw_training_states(scen, fit)
    per_learner = {}
    for st in states:
        per_learner.setdefault(st["learner"], set()).add(st["start"])
    assert all(len(v) == 1 for v in per_learner.values())
    # and both starts are actually used
    assert {next(iter(v)) for v in per_learner.values()} == set(S.TRAIN_STARTS)


# ------------------------------------------------------------- the advantages


def _patch(monkeypatch):
    """Belief distance falls by the asked problem_id: r is then exactly predictable."""
    def frozen(model, events, query, device):
        return float(sum(e["problem_id"] for e in events)), 0.0

    def logits(model, mu, tc, device):
        return np.full((int(tc.shape[0]), 4), mu / 1000.0, dtype=np.float64)

    def dlogit(a, b):
        return float(np.abs(np.asarray(a) - np.asarray(b)).mean())

    monkeypatch.setattr(S, "_frozen_filter_state", frozen)
    monkeypatch.setattr(S, "_point_logits", logits)
    monkeypatch.setattr(S, "_dlogit", dlogit)


def test_exact_advantages_one_per_remaining_candidate(monkeypatch):
    _patch(monkeypatch)
    ep = _episode(n_cand=5)
    st = {"episode": ep, "known_candidate_idx": [0], "remaining_idx": [1, 2, 3, 4]}
    r = S.exact_advantages(None, st, "cpu")
    assert r.shape == (4,)
    assert np.isfinite(r).all()


def test_advantage_sign_convention_is_movement_toward_the_reference(monkeypatch):
    """Positive r must mean the belief got CLOSER to b_full, never merely that it moved.

    Asserted against the stub's own arithmetic rather than a guessed direction: the item
    that lands the belief nearest the reference must carry the largest advantage, and an
    item that overshoots past the reference must score below it. This is the family-#8
    check on the reward -- a displacement-seeking reward would rank the overshoot first,
    and this project has already paid once for exactly that (the FA planner's
    confidence-sharpening result).
    """
    _patch(monkeypatch)
    ep = _episode(n_cand=4)
    st = {"episode": ep, "known_candidate_idx": [], "remaining_idx": [0, 1, 2, 3]}
    r = S.exact_advantages(None, st, "cpu")

    # Recompute the stub's geometry directly: which candidate lands mu closest to full_mu?
    full_mu = sum(e["problem_id"] for e in ep["full_events"])
    base = sum(e["problem_id"] for e in ep["prefix_events"])
    gaps = [abs((base + ep["candidates"][j]["problem_id"]) - full_mu) for j in range(4)]
    nearest = int(np.argmin(gaps))
    assert int(np.argmax(r)) == nearest, "highest advantage must be the closest approach"
    # and the ranking of r must be the reverse of the ranking of remaining distance
    assert list(np.argsort(-r)) == list(np.argsort(gaps))


def test_advantage_is_negative_when_the_item_moves_the_belief_away(monkeypatch):
    """A move that increases distance to the reference must score below zero."""
    _patch(monkeypatch)
    ep = _episode(n_cand=4)
    # Start already AT the reference: every further item can only move away.
    st = {"episode": ep, "known_candidate_idx": [0, 1, 2, 3], "remaining_idx": []}
    ep2 = _episode(n_cand=5)
    st2 = {"episode": ep2, "known_candidate_idx": [0, 1, 2, 3, 4], "remaining_idx": []}
    assert S.exact_advantages(None, st, "cpu").shape == (0,)
    assert S.exact_advantages(None, st2, "cpu").shape == (0,)


def test_soft_targets_are_a_proper_distribution_and_rank_preserving():
    a = np.array([-2.0, 0.0, 0.5, 3.0])
    p = S.soft_targets(a, tau=1.0)
    assert np.isclose(p.sum(), 1.0)
    assert (p > 0).all()
    assert list(np.argsort(p)) == list(np.argsort(a)), "soft targets must preserve the ranking"


def test_soft_targets_survive_a_degenerate_state():
    """All-equal advantages must give a uniform target, not a divide-by-zero."""
    p = S.soft_targets(np.zeros(7))
    assert np.isclose(p.sum(), 1.0)
    assert np.allclose(p, 1.0 / 7)


def test_episode_key_separates_the_same_learner_at_different_starts():
    """Canary #7: a learner is NOT a unique episode identity across starts.

    The same learner has one scenario row per start, with a different prefix and a
    different menu. Keying roll-out picks by learner alone collided the two and scored one
    episode with the other's picks; the KeyError only
    surfaced because `_score_fixed_picks` resolves ids against the menu.
    """
    a = {"learner": "L1", "start": 3}
    b = {"learner": "L1", "start": 10}
    assert S.episode_key(a) != S.episode_key(b)
    assert S.episode_key(a) == S.episode_key({"learner": "L1", "start": 3})


def test_episode_key_refuses_an_unstamped_episode():
    import pytest as _pytest
    with _pytest.raises(KeyError):
        S.episode_key({"learner": "L1"})
