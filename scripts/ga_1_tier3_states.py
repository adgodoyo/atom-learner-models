#!/usr/bin/env python3
"""Shared helpers for the fine-tuned LLM selector (Figure 12): training-state draw, CRN identifier
shuffle and exact one-step advantages.

`scripts/rl_2.py`, `scripts/cw_planner_suite.py`, `src/atom/simulator/cw_substrate.py` and the
zero-shot selector modules are imported unmodified. Three things live here:

1. Scenarios at starts {3, 10}. Training and evaluation avoid start 0, where the 2x2
   degenerates (with- and without-history prompts coincide). Both starts are in
   `STARTS = (0,3,5,10)` of the stock harness, so no start has to be synthesised:
   `make_scenarios_for_starts` narrows the tuple to a subset rather than extending it, and
   `assert_patch_is_inert` verifies that every evaluated start comes out bit-identical to the
   stock build. Consequently the `rl_2.py` actor can be quoted at both evaluation starts, and
   the blind comparators can be checked against its committed cells for exact agreement.

2. The fit / internal-validation split, reproducing `rl_2._fit_actor`'s partition exactly
   (sorted learners, first 80% fit, then at most 40 held out, picked by `linspace`): 543 fit +
   40 internal-val learners. `SPLIT_EXPECTED` asserts the counts.

3. Exact one-step advantages `r(s,j) = d(b_s, ref) - d(b_{s.j}, ref)` for every menu item at
   every drawn state, computed with the frozen primitives of `rl_2.py`.

The CRN identifier shuffle is a permutation of the candidate list before it reaches the zero-shot
renderer. Identifier `A` then labels a pseudorandom candidate, so a policy cannot score by
learning "output A" -- the zero-shot policy's measured behaviour (90.8% `A`). The renderer,
read-out and alphabet are used unchanged.
"""

from __future__ import annotations

import hashlib
import pathlib
import sys
from typing import Any, Sequence

import numpy as np
import torch

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import cw_planner_suite as _cps  # noqa: E402
from ga_1_common import stable_hash  # noqa: E402
from rl_2 import _dlogit, _frozen_filter_state, _make_scenarios, _point_logits  # noqa: E402

TRAIN_STARTS = (3, 10)
EVAL_STARTS = (3, 10)
BUDGETS = (3, 8, 16)
INTERNAL_VAL_MAX_LEARNERS = 40          # the cap used by rl_2.py, reused
SPLIT_EXPECTED = {"fit": 543, "internal": 40}
STATE_DRAW_SEED = 20260903


# ----------------------------------------------------------------- scenarios


def make_scenarios_for_starts(manifest, summaries, device, starts, max_train=0, max_val=0,
                              release_by_episode=None):
    """Call `rl_2._make_scenarios` with a different start list, then restore."""
    original = _cps.STARTS
    try:
        _cps.STARTS = tuple(int(s) for s in starts)
        out = _make_scenarios(manifest, summaries, device, max_train, max_val,
                              release_by_episode=release_by_episode)
    finally:
        _cps.STARTS = original
    # A learner appears once PER START, with a different prefix and a different menu.
    # Stamp the start on every episode so downstream code can key on (learner, start):
    # keying on the learner alone silently collides the two and scores one episode with
    # the other's picks.
    for key, rows in out.items():
        start = int(key.rsplit("start", 1)[-1])
        for ep in rows:
            ep["start"] = start
    return out


def episode_key(ep) -> str:
    """Unique identity of a scenario row: a learner alone is NOT unique across starts."""
    if "start" not in ep:
        raise KeyError("episode is missing its `start` stamp; build it through "
                       "make_scenarios_for_starts")
    return f"{ep['learner']}|{int(ep['start'])}"


def scenario_digest(rows: Sequence[dict[str, Any]]) -> str:
    """Value digest of one start's scenarios (identity of the rows, never just counts)."""
    payload = []
    for ep in rows:
        payload.append("|".join([
            str(ep["learner"]),
            ",".join(str(x["episode_id"]) for x in ep["prefix_events"]),
            ",".join(f"{x['episode_id']}:{x['problem_id']}" for x in ep["candidates"]),
            ",".join(f"{x['episode_id']}:{x['label']}" for x in ep["targets"]),
        ]))
    return stable_hash("\n".join(payload), 64)


def assert_patch_is_inert(manifest, summaries, device, max_train=0, max_val=0) -> dict[str, str]:
    """Prove the STARTS patch adds a start without perturbing the shared ones.

    Start 3 exists in both the stock tuple and ours, so it must come out bit-identical.
    A test that the patched call *ran* would be worthless; this tests that it MEANS the
    same thing.
    """
    stock = _make_scenarios(manifest, summaries, device, max_train, max_val,
                            release_by_episode=None)
    patched = make_scenarios_for_starts(manifest, summaries, device, TRAIN_STARTS,
                                        max_train, max_val)
    out = {}
    for split in ("train", "val"):
        for start in TRAIN_STARTS:
            key = f"{split}_start{start}"
            if key in stock and key in patched:
                a, b = scenario_digest(stock[key]), scenario_digest(patched[key])
                if a != b:
                    raise AssertionError(f"STARTS narrowing perturbed {key}: {a} != {b}")
                out[key] = a
    if len(out) < len(TRAIN_STARTS):
        raise AssertionError(
            f"expected every evaluated start to be verifiable against the stock build, "
            f"got {sorted(out)}")
    return out


# --------------------------------------------------------------------- split


def fit_internal_split(train_eps: Sequence[dict[str, Any]]) -> tuple[list[dict], list[dict], dict]:
    """The deterministic partition of `rl_2._fit_actor`, reproduced exactly."""
    learners = sorted({str(e["learner"]) for e in train_eps})
    cut = max(1, int(0.8 * len(learners)))
    fit_ids = set(learners[:cut])
    internal_all = [x for x in learners if x not in fit_ids]
    if len(internal_all) > INTERNAL_VAL_MAX_LEARNERS:
        picks = np.linspace(0, len(internal_all) - 1, INTERNAL_VAL_MAX_LEARNERS, dtype=int)
        internal_ids = {internal_all[int(i)] for i in picks}
    else:
        internal_ids = set(internal_all)
    fit = [e for e in train_eps if str(e["learner"]) in fit_ids]
    internal = [e for e in train_eps if str(e["learner"]) in internal_ids]
    meta = {"fit_learners": len(fit_ids), "internal_val_learners": len(internal_ids),
            "internal_val_cap": INTERNAL_VAL_MAX_LEARNERS}
    return fit, internal, meta


# ----------------------------------------------------------- the CRN shuffle


def crn_permutation(n: int, *parts: Any, seed: int = STATE_DRAW_SEED) -> list[int]:
    """Deterministic per-state permutation of candidate positions.

    Keyed by the state's identity, not by a call counter, so it is stable under resume,
    reordering and re-runs, and identical for every policy scored on that state (CRN).
    """
    key = "|".join(str(p) for p in parts) + f"|{seed}"
    digest = hashlib.sha256(key.encode()).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
    return [int(i) for i in rng.permutation(n)]


def apply_permutation(candidates: Sequence[dict[str, Any]], perm: Sequence[int]) -> list[dict]:
    """Reorder candidates so identifier k labels `candidates[perm[k]]`."""
    if sorted(perm) != list(range(len(candidates))):
        raise AssertionError("permutation is not a bijection over the candidate list")
    return [candidates[i] for i in perm]


# ------------------------------------------------------------- state drawing


def draw_training_states(scenarios: dict[str, list[dict]], fit_learners: set[str],
                         starts: Sequence[int] = TRAIN_STARTS,
                         budgets: Sequence[int] = BUDGETS,
                         seed: int = STATE_DRAW_SEED) -> list[dict[str, Any]]:
    """Uniform-random-policy trajectories; one state per step k < B.

    Each fit learner is assigned ONE start uniformly from `starts` (CRN-keyed on the
    learner, so the assignment is stable), then contributes one random-policy trajectory
    per budget, emitting the state at every step before the budget is spent.  Random
    rather than actor trajectories, by design: an actor-visited state distribution would
    distil the actor through the back door.
    """
    states: list[dict[str, Any]] = []
    for learner in sorted(fit_learners):
        pick = crn_permutation(len(starts), "start-assign", learner, seed=seed)[0]
        start = int(starts[pick])
        rows = [ep for ep in scenarios[f"train_start{start}"] if str(ep["learner"]) == learner]
        if not rows:
            continue
        ep = rows[0]
        n_cand = len(ep["candidates"])
        for budget in budgets:
            rng = np.random.default_rng(
                int.from_bytes(hashlib.sha256(
                    f"{learner}|{start}|{budget}|{seed}".encode()).digest()[:8], "big"))
            remaining = list(range(n_cand))
            known_idx: list[int] = []
            for step in range(min(int(budget), n_cand)):
                states.append({
                    "learner": learner, "start": start, "budget": int(budget), "step": step,
                    "known_candidate_idx": list(known_idx),
                    "remaining_idx": list(remaining),
                    "episode": ep,
                })
                pick = int(rng.integers(len(remaining)))
                known_idx.append(remaining.pop(pick))
    return states


# --------------------------------------------------------- exact advantages


def exact_advantages(model: dict, state: dict[str, Any], device: str) -> np.ndarray:
    """`r(s,j) = d(b_s, ref) - d(b_{s.j}, ref)` for every remaining candidate.

    Positive r means asking j moves the belief TOWARD the learner's own full-history
    reference.  Movement toward `b_full`, never movement per se -- the FA planner's
    displacement-seeking policy was measured to be pure confidence-sharpening.
    """
    ep = state["episode"]
    targets = ep["targets"]
    query = targets[0]["codec"]
    target_codecs = torch.stack([t["codec"] for t in targets])
    full_mu, _ = _frozen_filter_state(model, ep["full_events"], query, device)
    full_logits = _point_logits(model, full_mu, target_codecs, device)

    known = list(ep["prefix_events"]) + [ep["candidates"][i] for i in state["known_candidate_idx"]]
    mu, _ = _frozen_filter_state(model, sorted(known, key=lambda x: x["order"]), query, device)
    d_now = _dlogit(_point_logits(model, mu, target_codecs, device), full_logits)

    out = np.empty(len(state["remaining_idx"]), dtype=np.float64)
    for pos, j in enumerate(state["remaining_idx"]):
        nxt = sorted(known + [ep["candidates"][j]], key=lambda x: x["order"])
        mu2, _ = _frozen_filter_state(model, nxt, query, device)
        d_next = _dlogit(_point_logits(model, mu2, target_codecs, device), full_logits)
        out[pos] = d_now - d_next
    return out


def soft_targets(advantages: np.ndarray, tau: float = 1.0) -> np.ndarray:
    """`p*(j|s) ∝ exp(A/tau)` with A z-scored within the state, std floored at 1e-6."""
    a = np.asarray(advantages, dtype=np.float64)
    std = a.std()
    z = (a - a.mean()) / (std if std > 1e-6 else 1e-6)
    z = z / float(tau)
    z -= z.max()
    p = np.exp(z)
    return p / p.sum()
