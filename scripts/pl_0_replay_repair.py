#!/usr/bin/env python
"""Leak check for the planner's replay rules (§5.7): perturbing a candidate's logged future channels
must leave every score and trace bit-identical.

Two leaks in a replay harness, and the rules that close them:

  (i)  `src/atom/planner/evsi.py::_materialize` must not feed a candidate's REAL logged
       `hint_bin`/`saw_answer` into the model inside `candidate_evsi_scores`'s hypothetical
       y-branches; otherwise a candidate's own future channels leak into every "what if this
       candidate's outcome were y" branch, whichever y is hypothesised. Rule: the acquired
       observation is CORRECTNESS-ONLY -- revealing a candidate, real or hypothetical, conveys
       `correct` alone; `hint_bin`/`saw_answer` are always neutralised to 0 for the revealed
       portion of the window.
  (ii) the candidate pool must not exclude give-ups (`saw_answer==1`): `saw_answer` is unknowable
       at selection time, and letting it decide which candidates are offered makes the pool's
       composition depend on the future. Rule: `src/atom/planner/candidates.py
       ::recency_bounded_pool` never inspects any outcome channel.

The check: perturbing a candidate's logged `hint_bin`/`saw_answer` must leave its EVSI score,
and every policy's full replay trace, bit-identical. The same invariant is pinned in
tests/test_planner_evsi.py and tests/test_planner_candidates.py; this script re-derives it
independently on fresh synthetic tensors and writes `reports/pl_0_replay_repair.md`. It builds
and checks the machinery only; it makes no planner claim.
Run: `python scripts/pl_0_replay_repair.py` (CPU, seconds).
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from atom.eval.report import gate_report  # noqa: E402
from atom.filter.s1 import S1Filter  # noqa: E402
from atom.heads.s1_heads import FiLMHeads  # noqa: E402
from atom.planner.candidates import recency_bounded_pool  # noqa: E402
from atom.planner.evsi import POLICIES, candidate_evsi_scores, run_episode  # noqa: E402
from atom.utils.logging import RunLogger  # noqa: E402
from atom.utils.seeds import seed_everything  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[1]


def _tiny_model(seed: int) -> tuple[S1Filter, FiLMHeads, int]:
    torch.manual_seed(seed)
    codec_dim, latent_dim, hidden = 6, 4, 8
    filter_model = S1Filter(codec_dim, hidden, hidden, latent_dim)
    heads = FiLMHeads(codec_dim, latent_dim, hidden)
    filter_model.eval()
    heads.eval()
    return filter_model, heads, codec_dim


def _windows(codec_dim: int, m_prefix: int, w_cand: int, n_target: int, seed: int) -> tuple[dict, dict, dict]:
    g = torch.Generator().manual_seed(seed)

    def window(n: int) -> dict:
        return {
            "item_codec": torch.randn(n, codec_dim, generator=g),
            "correct": torch.randint(0, 2, (n,), generator=g).float(),
            "hint_bin": torch.zeros(n),
            "saw_answer": torch.zeros(n),
        }
    return window(m_prefix), window(w_cand), window(n_target)


def _canary_candidate_scores(seed: int) -> bool:
    """Perturbing every remaining candidate's hint_bin/saw_answer must leave every EVSI
    score bit-identical."""
    filter_model, heads, codec_dim = _tiny_model(seed)
    prefix, cand, target = _windows(codec_dim, m_prefix=3, w_cand=6, n_target=3, seed=seed)
    latent_dim = filter_model.knowledge_filter.mu_head.out_features
    noise = torch.randn(8, latent_dim)
    remaining = list(range(6))

    before = candidate_evsi_scores(filter_model, heads, prefix, cand, [], remaining,
                                   target["item_codec"], 8, "cpu", 0.0, noise=noise)
    cand_perturbed = dict(cand)
    cand_perturbed["hint_bin"] = torch.ones_like(cand["hint_bin"])
    cand_perturbed["saw_answer"] = torch.ones_like(cand["saw_answer"])
    after = candidate_evsi_scores(filter_model, heads, prefix, cand_perturbed, [], remaining,
                                  target["item_codec"], 8, "cpu", 0.0, noise=noise)
    return before.keys() == after.keys() and all(before[c] == after[c] for c in before)


def _canary_full_episode(seed: int) -> bool:
    """Same perturbation, checked end-to-end: every policy's full trace must be unchanged."""
    filter_model, heads, codec_dim = _tiny_model(seed)
    prefix, cand, target = _windows(codec_dim, m_prefix=3, w_cand=6, n_target=3, seed=seed)
    latent_dim = filter_model.knowledge_filter.mu_head.out_features
    noise = torch.randn(8, latent_dim)

    cand_perturbed = dict(cand)
    cand_perturbed["hint_bin"] = torch.ones_like(cand["hint_bin"])
    cand_perturbed["saw_answer"] = torch.ones_like(cand["saw_answer"])

    for policy in POLICIES:
        rng_a = np.random.default_rng(0) if policy in ("random", "top5_softened") else None
        rng_b = np.random.default_rng(0) if policy in ("random", "top5_softened") else None
        trace_a, evsi_a = run_episode(filter_model, heads, prefix, cand, target["item_codec"],
                                      target["correct"], k=8, device="cpu", policy=policy,
                                      budget=6, rng=rng_a, noise=noise)
        trace_b, evsi_b = run_episode(filter_model, heads, prefix, cand_perturbed, target["item_codec"],
                                      target["correct"], k=8, device="cpu", policy=policy,
                                      budget=6, rng=rng_b, noise=noise)
        if trace_a != trace_b or evsi_a != evsi_b:
            return False
    return True


def _pool_never_filters_on_outcome() -> bool:
    """recency_bounded_pool's signature has no outcome channel to filter on, and its
    output is exactly the lookback window -- give-ups included by construction."""
    import inspect
    params = inspect.signature(recency_bounded_pool).parameters
    if "saw_answer" in params or "correct" in params:
        return False
    pool = recency_bounded_pool(target_start=200, pool_lookback=40)
    return list(pool) == list(range(160, 200))


def main() -> int:
    seed_everything(0)
    logger = RunLogger(config={"task": "pl_0_replay_repair"}, data_version="none", seed=0)

    seeds = [11, 12, 13, 21, 22]
    candidate_canary = all(_canary_candidate_scores(s) for s in seeds)
    episode_canary = all(_canary_full_episode(s) for s in seeds)
    pool_canary = _pool_never_filters_on_outcome()
    passed = candidate_canary and episode_canary and pool_canary

    logger.log_metrics(candidate_canary=candidate_canary, episode_canary=episode_canary,
                       pool_canary=pool_canary, n_seeds=len(seeds), passed=passed)
    logger.summary(status="ok" if passed else "fail", passed=passed)

    gate_report(
        path="reports/pl_0_replay_repair.md",
        title="PL-0 - Replay repair (leak canaries)",
        meta={"run_id": logger.run_id, "git_sha": logger.git_sha, "config_hash": logger.config_hash,
             "n_canary_seeds": len(seeds), "device": "cpu"},
        metrics={
            "(i) candidate_evsi_scores invariant to hint_bin/saw_answer perturbation":
                "PASS" if candidate_canary else "FAIL",
            "(i) run_episode full trace (all 5 policies) invariant to the same perturbation":
                "PASS" if episode_canary else "FAIL",
            "(ii) recency_bounded_pool has no outcome channel to filter on; pool = exact lookback window":
                "PASS" if pool_canary else "FAIL",
        },
        passed=passed,
        gate_desc="Replay leak canary -- perturbing a candidate's logged "
                 "future channels must leave its score bit-identical. Fails if: the canary moves.",
        notes=[
            "(i) fixed in src/atom/planner/evsi.py::_materialize -- the acquired observation from "
            "revealing a candidate (real or hypothetical) is redefined as correctness-only; "
            "hint_bin/saw_answer for the revealed portion of the window are always neutralised to 0, "
            "never read from the candidate's logged values. Enforced in exactly one place, so every "
            "caller (run_episode's real reveals, candidate_evsi_scores' hypothetical y-branches) "
            "inherits it automatically.",
            "(ii) fixed via src/atom/planner/candidates.py::recency_bounded_pool -- the repaired "
            "replacement for a give-up-exclusion pool construction, "
            "which filtered on saw_answer and was itself a leak.",
            "(iii) prefix beliefs are not involved: this script builds and checks the machinery "
            "only -- no planner result is claimed here.",
            "Permanent regression guards for both canaries live in tests/test_planner_evsi.py "
            "(test_materialize_neutralises_hint_and_saw_answer_for_revealed_candidates, "
            "test_candidate_evsi_scores_invariant_to_hint_and_saw_answer_perturbation, "
            "test_run_episode_invariant_to_hint_and_saw_answer_perturbation) and "
            "tests/test_planner_candidates.py (test_recency_bounded_pool_does_not_exclude_give_ups).",
        ],
    )
    print(f"replay leak check: {'PASS' if passed else 'FAIL'} -- report: reports/pl_0_replay_repair.md")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
