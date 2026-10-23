#!/usr/bin/env python3
"""Shared helpers for the utility model and question-writing experiments of §8.4
(`ga_1_tier4_*`): the state draw and the exact-advantage table.

Every belief, logit, distance, predictive feature and EPIG score comes from ``scripts/rl_2.py`` /
``scripts/cw_planner_suite.py`` imported unchanged; ``scripts/ga_1_common.py`` supplies hashing
and the numeric-upload guard. This module adds the pre-registered state draws (a cold draw at
start 0 and a warm-started draw), the learner and item splits, the exhaustive advantage table,
and their digests, published so that byte-identity of the draw is checkable. The draw follows
the same random-policy rule as the fine-tuned selector (`ga_1_tier3_states.py`).
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any, Sequence

import numpy as np
import torch

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from cw_planner_suite import K_EVAL, _candidate_scores  # noqa: E402
from ga_1_common import stable_hash, stable_uint64  # noqa: E402
from rl_2 import (  # noqa: E402
    _build_manifest,
    _candidate_features,
    _dlogit,
    _episode_summaries,
    _frozen_filter_state,
    _make_scenarios,
    _point_logits,
    _state,
)

# --------------------------------------------------------------------- registered constants

SUBSTRATE = "ratchet-cwsim/refreshed-head"
DRAW_SEED_TAG = "ga1-tier4-draw"
WARM_DRAW_SEED_TAG = "ga1-tier4-warm-draw"
EPS_SEED_TAG = "ga1-tier4-eps"
BUDGETS = (3, 8, 16)

# ``DRAW_START = 0`` is retained only so the utility-model (Û) validation stays
# byte-reproducible.  It must not be used for any rollout/imagination training.
# At an empty observer belief ``_frozen_filter_state(model, [], query)`` returns the prior,
# which is identical for every learner -- so the state vector and the candidate features are
# bit-identical across students and the first decision is person-blind BY CONSTRUCTION.
# Supervised fine-tuning of the question writer therefore uses ``WARM_STARTS`` via
# ``draw_warm_states``, never this.
DRAW_START = 0
WARM_STARTS = (3, 5, 10)
TRAIN_LEARNERS = 686           # the cap used by rl_2.py; the realised count is 679 learners
VAL_LEARNERS = 110
FIT_FRACTION = 0.8             # the deterministic learner partition of rl_2 ``_fit_actor``
ITEM_SPLIT_SEED = 20260903
HELDOUT_ITEM_FRACTION = 0.2
UTILITY_SEEDS = (20260902, 20260903, 20260904)
N_PROBLEMS = 50
STATE_DIM = 8
FEAT_DIM = 6
CODEC_DIM = 64


def manifest_args(raw_dir: str, **overrides: Any) -> argparse.Namespace:
    """The frozen CW artifact set every LLM-selector runner uses, as one namespace."""
    base = dict(
        raw_dir=raw_dir,
        code_map=str(REPO / "cache/cw_03_vectors/cw_code_state_hash_map_full.npz"),
        diff_map=str(REPO / "cache/cw_03_vectors/cw_diff_hash_map_full.npz"),
        codealone_meta=str(REPO / "cache/cw_03_vectors/codealone/meta.json"),
        codealone_states=str(REPO / "cache/cw_03_vectors/codealone/states_mean.npy"),
        diff_meta=str(REPO / "cache/cw_03_vectors/diff/meta.json"),
        diff_states=str(REPO / "cache/cw_03_vectors/diff/states_mean.npy"),
        derived_val_predictions="",
    )
    base.update(overrides)
    return argparse.Namespace(**base)


# ------------------------------------------------------------------------------ splits


def fit_learners(scenarios: dict[str, list[dict]]) -> list[str]:
    """The byte-identical 80/20 partition of ``train_start0`` (the ``rl_2._fit_actor`` rule)."""
    learners = sorted({str(e["learner"]) for e in scenarios["train_start0"]})
    cut = max(1, int(FIT_FRACTION * len(learners)))
    return learners[:cut]


def utility_learner_split(fit_ids: Sequence[str]) -> tuple[list[str], list[str]]:
    """Deterministic 80/20 of the fit learners into U-fit and U-internal-validation."""
    ordered = sorted(str(x) for x in fit_ids)
    cut = max(1, int(FIT_FRACTION * len(ordered)))
    return ordered[:cut], ordered[cut:]


def item_split(problem_ids: Sequence[int]) -> tuple[list[int], list[int]]:
    """Seed-fixed item-disjoint split: 20% of the problem universe is held out.

    The permutation is taken once, before any fit, from a sorted universe, so the split is
    a function of the corpus alone and cannot drift with iteration order.
    """
    ordered = sorted(int(p) for p in problem_ids)
    if len(ordered) != N_PROBLEMS:
        raise ValueError(f"expected the frozen {N_PROBLEMS}-item universe, got {len(ordered)}")
    rng = np.random.default_rng(ITEM_SPLIT_SEED)
    perm = rng.permutation(len(ordered))
    n_held = int(round(HELDOUT_ITEM_FRACTION * len(ordered)))
    held = sorted(ordered[int(i)] for i in perm[:n_held])
    train = sorted(ordered[int(i)] for i in perm[n_held:])
    if set(held) & set(train) or len(held) + len(train) != len(ordered):
        raise RuntimeError("item split is not a partition")
    return train, held


# -------------------------------------------------------------------------- the state draw


def _draw_rng(learner: str, budget: int) -> np.random.Generator:
    """Content-addressed, order-independent: the draw cannot depend on iteration order."""
    return np.random.default_rng(stable_uint64(f"{DRAW_SEED_TAG}|{learner}|B{budget}") % (2 ** 63))


def state_eps(learner: str, budget: int, step: int, latent_dim: int, device: str) -> torch.Tensor:
    """The K-sample epistemic draw for this state, shared (CRN) by every instrument."""
    seed = stable_uint64(f"{EPS_SEED_TAG}|{learner}|B{budget}|k{step}") % (2 ** 31)
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    return torch.randn(K_EVAL, latent_dim, generator=gen).to(device)


def draw_states(episodes: Sequence[dict], budgets: Sequence[int] = BUDGETS) -> list[dict]:
    """Uniform-random-policy rollouts; the state is recorded at every step before the pick.

    Random-policy states, never actor states: an actor-visited state distribution would
    distil the actor into the model trained on these states.  Returned rows carry only indices into the episode's own candidate list, so the
    draw is serialisable and hashable independently of any model.
    """
    out: list[dict] = []
    for ep_index, ep in enumerate(episodes):
        n = len(ep["candidates"])
        if n == 0:
            continue
        for budget in budgets:
            rng = _draw_rng(str(ep["learner"]), int(budget))
            remaining = list(range(n))
            picks: list[int] = []
            for step in range(min(int(budget), n)):
                out.append({
                    "episode_index": int(ep_index),
                    "learner": str(ep["learner"]),
                    "budget": int(budget),
                    "step": int(step),
                    "picks": list(picks),
                    "remaining": list(remaining),
                })
                j = int(rng.integers(len(remaining)))
                picks.append(remaining.pop(j))
    return out


def warm_start_for(learner: str, budget: int) -> int:
    """Uniform draw from {3, 5, 10}, content-addressed per (learner, budget)."""
    rng = np.random.default_rng(
        stable_uint64(f"{WARM_DRAW_SEED_TAG}|{learner}|B{budget}") % (2 ** 63))
    return int(WARM_STARTS[int(rng.integers(len(WARM_STARTS)))])


def draw_warm_states(scenarios: dict[str, list[dict]], split: str, learner_ids: set[str],
                     budgets: Sequence[int] = BUDGETS) -> tuple[list[dict], list[dict]]:
    """WARM-STARTED draw — every trajectory begins from a REAL chronological prefix.

    An observer belief with no evidence is the prior, identical for every learner, so a trajectory started
    there is person-blind by construction and spends gradient in the one regime where the
    policy cannot tell students apart.  Here each (learner, budget) draws its start
    uniformly from **{3, 5, 10} real revealed episodes** — the same cells the before/after
    panel evaluates on — and the uniform-random rollout continues from that prefix for
    state diversity.

    Returns ``(states, episodes)`` together: a state's ``episode_index`` points into the
    returned episode list, because the episode object itself differs per start (its prefix,
    menu and targets are all start-dependent in ``_make_scenarios``).
    """
    by_start: dict[int, dict[str, dict]] = {}
    for s in WARM_STARTS:
        key = f"{split}_start{s}"
        if key not in scenarios:
            raise KeyError(f"warm-start draw needs scenario cell {key}")
        by_start[s] = {str(e["learner"]): e for e in scenarios[key]}

    episodes: list[dict] = []
    index_of: dict[tuple[str, int], int] = {}
    out: list[dict] = []
    for learner in sorted(str(x) for x in learner_ids):
        for budget in budgets:
            start = warm_start_for(learner, int(budget))
            ep = by_start[start].get(learner)
            if ep is None:
                # This learner has too short a history for this start; fall back to the
                # longest available warm start, and NEVER to 0.
                for s in sorted(WARM_STARTS, reverse=True):
                    ep = by_start[s].get(learner)
                    if ep is not None:
                        start = s
                        break
            if ep is None or not ep["candidates"]:
                continue
            key = (learner, start)
            if key not in index_of:
                index_of[key] = len(episodes)
                episodes.append(ep)
            ep_index = index_of[key]
            n = len(ep["candidates"])
            remaining = list(range(n))
            picks: list[int] = []
            rng = np.random.default_rng(
                stable_uint64(f"{WARM_DRAW_SEED_TAG}|roll|{learner}|B{budget}|s{start}")
                % (2 ** 63))
            for step in range(min(int(budget), n)):
                out.append({
                    "episode_index": ep_index,
                    "learner": learner,
                    "budget": int(budget),
                    "start": int(start),
                    "step": int(step),
                    "picks": list(picks),
                    "remaining": list(remaining),
                })
                picks.append(remaining.pop(int(rng.integers(len(remaining)))))
    if any(s["start"] == 0 for s in out):
        raise RuntimeError("warm-start check: a trajectory began from an empty belief")
    return out, episodes


def draw_digest(states: Sequence[dict], episodes: Sequence[dict]) -> str:
    """Value-digest of the realised draw (identity of the rows, not just counts)."""
    rows = []
    for s in states:
        ep = episodes[s["episode_index"]]
        rows.append({
            "learner": s["learner"], "budget": s["budget"], "step": s["step"],
            "picks": [str(ep["candidates"][i]["episode_id"]) for i in s["picks"]],
            "remaining": [str(ep["candidates"][i]["episode_id"]) for i in s["remaining"]],
        })
    return stable_hash(json.dumps(rows, sort_keys=True, separators=(",", ":")), 64)


# --------------------------------------------------------------------- the advantage table


def zscore_advantages(r: np.ndarray) -> np.ndarray:
    """The z-scoring of `ga_1_tier3_states.soft_targets`, std floored at 1e-6."""
    r = np.asarray(r, dtype=np.float64)
    return ((r - r.mean()) / max(float(r.std()), 1e-6)).astype(np.float32)


def state_rows(model: dict, ep: dict, state: dict, device: str,
               with_epig: bool = True) -> dict[str, Any]:
    """Exact advantages plus deployable features for every remaining candidate at one state.

    ``r(s, j) = d(b_s, b_full) - d(b_{s.j}, b_full)`` with ``d`` the registered centred-logit
    distance and ``b_full`` this learner's own end-state belief -- the reward of the
    fine-tuned selector verbatim, computed with the frozen primitives of rl_2.py.
    """
    query = ep["targets"][0]["codec"]
    target_codecs = torch.stack([t["codec"] for t in ep["targets"]])
    full_mu, _ = _frozen_filter_state(model, ep["full_events"], query, device)
    full_logits = _point_logits(model, full_mu, target_codecs, device)

    known = list(ep["prefix_events"]) + [ep["candidates"][i] for i in state["picks"]]
    known = sorted(known, key=lambda x: x["order"])
    mu, log_sigma = _frozen_filter_state(model, known, query, device)
    before = _dlogit(_point_logits(model, mu, target_codecs, device), full_logits)

    cands = [ep["candidates"][i] for i in state["remaining"]]
    r = np.empty(len(cands), dtype=np.float64)
    for i, cand in enumerate(cands):
        after_events = sorted(known + [cand], key=lambda z: z["order"])
        mu2, _ = _frozen_filter_state(model, after_events, query, device)
        r[i] = before - _dlogit(_point_logits(model, mu2, target_codecs, device), full_logits)

    eps = state_eps(state["learner"], state["budget"], state["step"], mu.numel(), device)
    feats, _ = _candidate_features(model, mu, log_sigma, cands, ep["targets"], eps, device)
    sfeat = _state(mu, log_sigma, state["step"], state["budget"], len(cands)).detach().cpu().numpy()
    epig = np.full(len(cands), np.nan, dtype=np.float32)
    if with_epig:
        scored = _candidate_scores(model, mu, cands, ep["targets"], eps, device)
        epig = np.asarray([scored[i]["epig"] for i in range(len(cands))], dtype=np.float32)
    return {
        "r": r,
        "advantage": zscore_advantages(r),
        "features": np.asarray(feats, dtype=np.float32),
        "state_features": np.asarray(sfeat, dtype=np.float32),
        "epig": epig,
        "problem_ids": np.asarray([int(c["problem_id"]) for c in cands], dtype=np.int64),
        "before": float(before),
    }


def build_table(model: dict, episodes: Sequence[dict], states: Sequence[dict], device: str,
                on_state: Any | None = None, with_epig: bool = True) -> dict[str, np.ndarray]:
    """Flatten the per-state advantage rows into one ragged NPZ-friendly table."""
    offsets, counts = [], []
    r_all, a_all, f_all, e_all, p_all = [], [], [], [], []
    s_feat, s_learner, s_budget, s_step, s_epi = [], [], [], [], []
    cursor = 0
    for n_done, st in enumerate(states, start=1):
        row = state_rows(model, episodes[st["episode_index"]], st, device, with_epig=with_epig)
        k = len(row["r"])
        offsets.append(cursor)
        counts.append(k)
        cursor += k
        r_all.append(row["r"])
        a_all.append(row["advantage"])
        f_all.append(row["features"])
        e_all.append(row["epig"])
        p_all.append(row["problem_ids"])
        s_feat.append(row["state_features"])
        s_learner.append(st["learner"])
        s_budget.append(st["budget"])
        s_step.append(st["step"])
        s_epi.append(st["episode_index"])
        if on_state is not None:
            on_state(n_done, len(states))
    return {
        "state_offset": np.asarray(offsets, dtype=np.int64),
        "state_count": np.asarray(counts, dtype=np.int64),
        "state_features": np.stack(s_feat).astype(np.float32),
        "state_learner": np.asarray(s_learner, dtype=object).astype(str),
        "state_budget": np.asarray(s_budget, dtype=np.int64),
        "state_step": np.asarray(s_step, dtype=np.int64),
        "state_episode_index": np.asarray(s_epi, dtype=np.int64),
        "cand_r": np.concatenate(r_all).astype(np.float64),
        "cand_advantage": np.concatenate(a_all).astype(np.float32),
        "cand_features": np.concatenate(f_all).astype(np.float32),
        "cand_epig": np.concatenate(e_all).astype(np.float32),
        "cand_problem_id": np.concatenate(p_all).astype(np.int64),
    }


def codec_matrix(manifest: dict) -> tuple[np.ndarray, dict[int, int]]:
    """The frozen 64-dim problem codec, keyed by problem_id.  Never a learned embedding."""
    codec = np.asarray(manifest["problem_codec"], dtype=np.float32)
    pid_to_idx = {int(k): int(v) for k, v in manifest["pid_to_idx"].items()}
    if codec.shape[1] != CODEC_DIM:
        raise ValueError(f"expected a {CODEC_DIM}-dim frozen codec, got {codec.shape}")
    return codec, pid_to_idx


def load_field(substrate: str, raw_dir: str, device: str) -> dict[str, Any]:
    """Load the substrate, manifest, summaries and scenarios exactly as rl_2.py does."""
    from rl_2 import _load_bundle

    model, _, meta = _load_bundle(substrate, device)
    manifest = _build_manifest(manifest_args(raw_dir))
    summaries = _episode_summaries(model, manifest, device)
    scenarios = _make_scenarios(manifest, summaries, device, TRAIN_LEARNERS, VAL_LEARNERS,
                                release_by_episode=None)
    return {"model": model, "meta": meta, "manifest": manifest, "scenarios": scenarios}
