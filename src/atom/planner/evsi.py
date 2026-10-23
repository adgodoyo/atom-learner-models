"""Ask policies for the planner replay harness (thesis Section 5.7). Each policy consumes
one learner's prefix/candidate/target split (src/atom/planner/replay.py) and returns a
per-budget-step trace of target-set V(b), log-loss and accuracy, so the info-efficiency
curve can be read off directly.

V(b) is selectable (`v_flavor`). The default, "jensen", is minus the mean K-sample Jensen
gap over the fixed target set (epistemic-only); "total" is minus the mean total predictive
entropy. Greedy EVSI scores each remaining candidate by the refilter-free EPIG estimator of
atom.planner.epig -- the mutual information between that candidate's outcome and the
target-set outcomes -- and asks the best. The two "max-uncertainty" baselines pick by the
candidate's OWN predicted uncertainty (as if it were a one-item target) instead of by its
effect on the target set: total predictive entropy, and the epistemic-only K-sample spread
(the stronger comparator, giving the EPIG-vs-BALD contrast). `top5_softened` uses the SAME
EVSI scores as greedy but draws uniformly from the top 5 instead of taking the single
argmax -- the standard ranking-and-selection mitigation, used as an optimiser's-curse test.

**The acquired observation is correctness-only.** Revealing a candidate, real or
hypothetical, conveys `correct` and nothing else; `hint_bin`/`saw_answer` are always
neutralised to 0 for the revealed portion of the window, never read from the candidate's
logged values. Otherwise a candidate's own logged future channels (whether it was actually
hinted or given up on) would leak into every hypothetical outcome the greedy policy scores,
regardless of which y was being hypothesised (thesis Appendix A.4). This is enforced in exactly one place (`_materialize` below), so every
caller -- real reveals in `run_episode`, hypothetical y-branches in
`candidate_evsi_scores` -- inherits it automatically. The permanent regression guard is a
leak canary (tests/test_planner_evsi.py): perturbing a candidate's logged `hint_bin` or
`saw_answer` must leave every EVSI score, and every policy's full trace, bit-identical.

"Reveal e" replays e's REAL observed outcome from the held-out log (this is a replay of
real history, not a simulation) -- selection never uses target labels, only each
candidate's own predicted p(e) (itself derived from prefix+already-revealed evidence).

**Chronological materialisation.** A policy REVEALS candidates in whatever order it
selects them (selection order), but the belief the filter computes must always be fed
events in their own TRUE chronological order -- the argument in replay.py's module
docstring licenses skipping gaps between revealed events, never feeding them out of order. `_materialize` below is the
single place this is enforced: given a SET of revealed candidate indices (regardless of
the order they were added to that set), it always re-sorts them chronologically before
building the sequence fed to the filter. One consequence, now a permanent regression guard
(tests/test_planner_evsi.py): at full budget every policy has revealed the same SET (all
candidates), so `_materialize` returns a BIT-IDENTICAL window for every policy -- if that
ever stops being true, the code is broken.
"""

from __future__ import annotations

import numpy as np
import torch

from atom.planner.epig import (
    information_bound_violations,
    joint_jensen_gap,
    joint_jensen_gap_binary_categorical,
    sample_based_epig,
    sample_based_epig_binary_categorical,
    total_entropy,
)
from atom.planner.replay import predict_on_targets

Window = dict[str, torch.Tensor]  # keys: item_codec, correct, hint_bin, saw_answer


V_FLAVORS = ("total", "jensen")


def _value(target_out: dict[str, torch.Tensor], v_flavor: str = "jensen") -> float:
    """V(b), selectable as a configuration flag so that both forms stay reachable.
    `v_flavor="jensen"` (default): -mean K-sample Jensen gap over the target set --
    epistemic-only BY CONSTRUCTION (entropy is concave, so the gap is non-negative for any
    finite K, not merely in expectation). `v_flavor="total"`: -mean TOTAL predictive
    entropy -- the original form, kept reachable on purpose so its failure stays
    reproducible (thesis Appendix A.4): its E_y[V(b')]-V(b) = mutual-information
    justification ("the aleatoric floor cancels") holds only for a Bayes-consistent
    posterior, which an amortised filter trained by a prequential ELBO is not. Checking the
    EVSI<=H(p_e) invariant against that arm on real data tests whether its estimate violates
    the bound it claims to compute."""
    if v_flavor not in V_FLAVORS:
        raise ValueError(f"unknown v_flavor {v_flavor!r}, expected one of {V_FLAVORS}")
    key = "total_entropy" if v_flavor == "total" else "jensen_gap"
    # A predictor may supply a SEPARATE target-side uncertainty under `value_<key>`.
    # This exists because the 5-class arm is deliberately ASYMMETRIC: targets are PREDICTED
    # so their uncertainty is categorical, while a candidate is REVEALED so what it conveys
    # stays correctness-only (three of the five classes ARE the help-seeking channel).
    # Keeping the two under different keys is what stops the candidate-side baselines
    # (`_own_uncertainty_pick`) and `p_e` from silently switching answer space with V. When
    # absent (binary predictors) this line is a no-op and `_value` is the form the
    # synthetic positive control validated.
    value_key = f"value_{key}"
    quantity = target_out[value_key] if value_key in target_out else target_out[key]
    return float(-quantity.mean().item())


def _score_step(target_out: dict[str, torch.Tensor], target_correct: torch.Tensor,
                n_asked: int, v_flavor: str = "jensen", scorer=None) -> dict:
    """`scorer=None` (default) scores the Bernoulli target. The 5-class arm passes one that
    reads the categorical predictive and scores `-log p(true class)`, because that arm's
    external score must live in the same answer space as its targets. Injected rather than
    branched on so the default path stays exactly the code the binary runs and the
    synthetic positive control were measured on."""
    if scorer is not None:
        return scorer(target_out, target_correct, n_asked, v_flavor)
    p = target_out["mean_prob"].clamp(1e-6, 1 - 1e-6)
    logloss = float(-(target_correct * p.log() + (1 - target_correct) * (1 - p).log()).mean().item())
    accuracy = float(((p >= 0.5).float() == target_correct).float().mean().item())
    return {"n_asked": n_asked, "V": _value(target_out, v_flavor), "logloss": logloss, "accuracy": accuracy}


def _predict(filter_model, heads, window: Window, target_codec: torch.Tensor, k: int,
             device: str, noise: torch.Tensor | None, predictor=None) -> dict[str, torch.Tensor]:
    """The ONE place a belief is turned into target predictions, so the policy layer is
    belief-implementation-agnostic.

    `predictor=None` (default) is v1's `replay.predict_on_targets`. Other beliefs pass their
    own predictor (e.g. `atom.planner.replay_v2.predict_on_targets_v2`, which needs the
    extra `score_missing` channel and `build_prev_feat_v2`'s five inputs, or
    `atom.planner.replay_sim3.predict_on_targets_sim3`) so that a query-independent belief
    can be scored by the SAME policy code.

    Injecting here rather than forking the module is deliberate: `candidate_evsi_scores`,
    `_greedy_pick`, `_materialize`'s chronological ordering, the correctness-only
    neutralisation and the `v_flavor` flag are the pieces the synthetic positive control
    validated. A parallel policy module would run code the control never checked.
    """
    if predictor is None:
        return predict_on_targets(filter_model, heads, window["item_codec"], window["correct"],
                                  window["hint_bin"], window["saw_answer"], target_codec, k,
                                  device, noise=noise)
    return predictor(filter_model, heads, window, target_codec, k, device, noise)


def _materialize(prefix: Window, cand: Window, revealed_idx: list[int],
                 correct_override: dict[int, float] | None = None) -> Window:
    """The one place chronological order is enforced: `revealed_idx` is a SET (any
    insertion order), always re-sorted ascending before being fed to the filter -- so two
    policies that reveal the same set in different orders get the IDENTICAL window.
    `correct_override` (used only for greedy's hypothetical y-branches) substitutes a
    hypothetical outcome for one candidate without mutating `cand` itself.

    The acquired observation is correctness-only (module docstring). The revealed
    portion's `hint_bin`/`saw_answer` are therefore always 0 here -- NEVER read from
    `cand["hint_bin"]`/`cand["saw_answer"]` -- regardless of what those candidates' real
    logged values are. This is what makes the leak canary hold: a candidate's real future
    channels have no path into any score, hypothetical or real.
    """
    if not revealed_idx:
        return {key: prefix[key] for key in prefix}
    sorted_idx = sorted(revealed_idx)
    idx_t = torch.as_tensor(sorted_idx, dtype=torch.long, device=cand["item_codec"].device)
    cand_correct = cand["correct"]
    if correct_override:
        cand_correct = cand_correct.clone()
        for i, y in correct_override.items():
            cand_correct[i] = y
    neutral = torch.zeros(idx_t.shape[0], dtype=prefix["hint_bin"].dtype, device=cand["item_codec"].device)
    out = {
        "item_codec": torch.cat([prefix["item_codec"], cand["item_codec"].index_select(0, idx_t)], dim=0),
        "correct": torch.cat([prefix["correct"], cand_correct.index_select(0, idx_t)], dim=0),
        "hint_bin": torch.cat([prefix["hint_bin"], neutral], dim=0),
        "saw_answer": torch.cat([prefix["saw_answer"], neutral], dim=0),
    }
    if "score_missing" in prefix:
        # The missing-score flag (v2 beliefs only; missing scores are flagged, never
        # imputed). The PREFIX carries its real values -- a placeholder score there is a
        # fact about the history and must stay visible to the filter. Revealed candidates
        # carry 0 because the FA replay restricts the candidate pool to label-valid rows: a row the platform never graded has no outcome
        # for the EVSI hypothesis branches to enumerate, so it is not an askable action.
        out["score_missing"] = torch.cat([prefix["score_missing"], neutral], dim=0)
    return out


def candidate_evsi_scores(filter_model, heads, prefix: Window, cand: Window, revealed_idx: list[int],
                          remaining: list[int], target_codec: torch.Tensor, k: int, device: str,
                          v_b: float, noise: torch.Tensor | None = None,
                          v_flavor: str = "jensen", predictor=None) -> dict[int, float]:
    """Refilter-free acquisition score for every candidate in ``remaining``.

    Candidate and target likelihoods are computed under the SAME K committee; no answer is
    fabricated and the recurrent filter is not re-run. ``jensen`` is the corrected EPIG
    ``I(Y_e;Y_t)`` (averaged over targets). ``total`` retains the one-term joint-Jensen
    quantity as a diagnostic arm; it is not called EPIG. The ``v_b`` argument is retained
    for API compatibility with the earlier refiltered scorer but is deliberately unused by
    the K-sample estimator.

    If the predictor supplies ``value_probs_k`` the targets are categorical (the 5-class
    arm) while the candidate remains binary/correctness-only. Otherwise both are binary.
    """
    del v_b
    revealed = _materialize(prefix, cand, revealed_idx)
    scores = {}
    for c in remaining:
        probes = torch.cat((cand["item_codec"][c:c + 1], target_codec), dim=0)
        out = _predict(filter_model, heads, revealed, probes, k, device, noise, predictor)
        probs_e_k = out["probs_k"][:, 0]
        if "value_probs_k" in out:
            probs_t_k = out["value_probs_k"][:, 1:, :]
            per_target = (
                sample_based_epig_binary_categorical(probs_e_k, probs_t_k)
                if v_flavor == "jensen"
                else joint_jensen_gap_binary_categorical(probs_e_k, probs_t_k)
            )
        else:
            probs_t_k = out["probs_k"][:, 1:]
            per_target = (
                sample_based_epig(probs_e_k, probs_t_k)
                if v_flavor == "jensen"
                else joint_jensen_gap(probs_e_k, probs_t_k)
            )
        score = per_target.mean()
        if v_flavor == "jensen":
            h_e = total_entropy(probs_e_k)
            if bool(information_bound_violations(score.view(1), h_e.view(1)).any()):
                raise RuntimeError(
                    f"EPIG information bound violated for candidate {c}: "
                    f"score={float(score):.8f} > H(Y_e)={float(h_e):.8f}"
                )
        scores[c] = float(score.item())
    return scores


def _greedy_pick(filter_model, heads, prefix: Window, cand: Window, revealed_idx: list[int],
                 remaining: list[int], target_codec: torch.Tensor, k: int, device: str, v_b: float,
                 noise: torch.Tensor | None = None, v_flavor: str = "jensen",
                 predictor=None) -> tuple[int, float]:
    scores = candidate_evsi_scores(filter_model, heads, prefix, cand, revealed_idx, remaining,
                                   target_codec, k, device, v_b, noise=noise, v_flavor=v_flavor,
                                   predictor=predictor)
    best_idx = max(scores, key=scores.get)
    return best_idx, scores[best_idx]


def _top_n_softened_pick(filter_model, heads, prefix: Window, cand: Window, revealed_idx: list[int],
                         remaining: list[int], target_codec: torch.Tensor, k: int, device: str, v_b: float,
                         rng: np.random.Generator, top_n: int,
                         noise: torch.Tensor | None = None, v_flavor: str = "jensen",
                         predictor=None) -> tuple[int, float]:
    """The pre-registered optimiser's-curse test: the standard ranking-and-selection
    mitigation for an argmax over noisy estimates -- draw uniformly from the top `top_n`
    scoring candidates instead of taking the single argmax. Same EVSI scores as greedy
    (same candidate_evsi_scores call, same CRN noise), only the final selection rule
    differs. If softening the argmax materially closes the gap greedy shows against the
    baselines, that is evidence the loss is winner's-curse-shaped (an inflated top score
    from noise, not a genuinely bad criterion) rather than a criterion that is simply wrong.
    """
    scores = candidate_evsi_scores(filter_model, heads, prefix, cand, revealed_idx, remaining,
                                   target_codec, k, device, v_b, noise=noise, v_flavor=v_flavor,
                                   predictor=predictor)
    ranked = sorted(scores, key=scores.get, reverse=True)
    top = ranked[:min(top_n, len(ranked))]
    idx = int(rng.choice(top))
    return idx, scores[idx]


def _own_uncertainty_pick(filter_model, heads, revealed: Window, cand: Window,
                          remaining: list[int], k: int, device: str, flavor: str,
                          noise: torch.Tensor | None = None, predictor=None) -> int:
    """flavor: "total" (predictive entropy) or "epistemic" (K-sample spread) -- the two
    max-uncertainty baseline flavours (the EPIG-vs-BALD contrast)."""
    best_idx, best_score = remaining[0], -float("inf")
    for c in remaining:
        own = _predict(filter_model, heads, revealed, cand["item_codec"][c:c + 1], k, device,
                       noise, predictor)
        key = "total_entropy" if flavor == "total" else "epistemic_std"
        score = float(own[key].item())
        if score > best_score:
            best_idx, best_score = c, score
    return best_idx


POLICIES = ("greedy_evsi", "random", "max_entropy", "max_epistemic", "top5_softened")


def run_episode(filter_model, heads, prefix: Window, cand: Window, target_codec: torch.Tensor,
                target_correct: torch.Tensor, k: int, device: str, policy: str, budget: int,
                rng: np.random.Generator | None = None, noise: torch.Tensor | None = None,
                selection_budget: int | None = None,
                pick_log: list[int] | None = None, softened_top_n: int = 5,
                v_flavor: str = "jensen", predictor=None, scorer=None) -> tuple[list[dict], float | None]:
    """One learner, one policy, budget candidates revealed one at a time in the policy's
    chosen SELECTION order -- but always fed to the filter in chronological order
    (`_materialize`), never selection order. Every candidate is drawn from this learner's
    own candidate window, every target strictly later (split_prefix_candidate_target).

    `v_flavor`: "jensen" (default) or "total" (the original form, kept reachable on
    purpose -- see `_value`). Affects every policy's
    "V" field and greedy_evsi's/top5_softened's own EVSI(c) scores identically; does not
    affect logloss/accuracy (those score `mean_prob` against the true label, unaffected by
    which V flavor picked the candidates) or max_entropy/max_epistemic's own-uncertainty
    picks (those already select by `total_entropy`/`epistemic_std` directly, never via V).

    `noise`: optional common-random-numbers draw (see replay.py), reused across every
    forward pass in this call -- pass the SAME tensor across all four policies for one
    learner to make their comparison a paired one (common-random-numbers variance control).

    `selection_budget` (default = budget): steps below this use the policy's real
    (possibly expensive, e.g. greedy's hypothesis testing) selection logic; steps from
    `selection_budget` up to `budget` reveal whatever remains in a fixed, CHEAP order (no
    selection computation) -- since `_materialize` re-sorts chronologically regardless,
    this cannot change any full-budget result, it only lets a full-budget run (needed for
    the bit-identity check) stay cheap without re-running expensive selection past the
    point that's actually being compared.

    `pick_log` (optional): if given an empty list, this call APPENDS the candidate index
    picked at each REAL-selection step (step < selection_budget) to it, in order -- an
    output parameter, not a return value, so it changes nothing for a caller that omits
    it. Used to report pick composition: WHICH candidates a policy actually chose, e.g.
    whether they are enriched for give-up-flagged events.

    `softened_top_n`: for policy="top5_softened" only, the number of top-scoring
    candidates to draw uniformly from (requires `rng`, same as "random"). Ignored by every
    other policy.

    Returns (trace, first_pick_evsi): trace is a list of per-step score dicts for
    n_asked=0..budget; first_pick_evsi is greedy_evsi's own EVSI score for its FIRST pick
    only (None for other policies) -- a per-learner value-of-asking signal that can be
    correlated against other per-learner quantities.
    """
    if policy not in POLICIES:
        raise ValueError(f"unknown policy {policy!r}, expected one of {POLICIES}")
    if v_flavor not in V_FLAVORS:
        raise ValueError(f"unknown v_flavor {v_flavor!r}, expected one of {V_FLAVORS}")
    n_cand = cand["item_codec"].shape[0]
    budget = min(budget, n_cand)
    selection_budget = budget if selection_budget is None else min(selection_budget, budget)
    revealed_idx: list[int] = []
    remaining = list(range(n_cand))

    revealed = _materialize(prefix, cand, revealed_idx)
    out0 = _predict(filter_model, heads, revealed, target_codec, k, device, noise, predictor)
    trace = [_score_step(out0, target_correct, 0, v_flavor, scorer)]
    first_pick_evsi = None

    for step in range(budget):
        if step < selection_budget:
            v_b = trace[-1]["V"]
            if policy == "greedy_evsi":
                idx, evsi = _greedy_pick(filter_model, heads, prefix, cand, revealed_idx, remaining,
                                         target_codec, k, device, v_b, noise=noise, v_flavor=v_flavor,
                                         predictor=predictor)
                if step == 0:
                    first_pick_evsi = evsi
            elif policy == "random":
                idx = int(rng.choice(remaining))
            elif policy == "top5_softened":
                idx, _ = _top_n_softened_pick(filter_model, heads, prefix, cand, revealed_idx, remaining,
                                              target_codec, k, device, v_b, rng, softened_top_n, noise=noise,
                                              v_flavor=v_flavor, predictor=predictor)
            else:
                flavor = "total" if policy == "max_entropy" else "epistemic"
                revealed = _materialize(prefix, cand, revealed_idx)
                idx = _own_uncertainty_pick(filter_model, heads, revealed, cand, remaining, k, device,
                                            flavor, noise=noise, predictor=predictor)
            if pick_log is not None:
                pick_log.append(idx)
        else:
            idx = remaining[0]  # cheap fixed order -- _materialize re-sorts regardless

        revealed_idx.append(idx)
        remaining.remove(idx)

        revealed = _materialize(prefix, cand, revealed_idx)
        out = _predict(filter_model, heads, revealed, target_codec, k, device, noise, predictor)
        trace.append(_score_step(out, target_correct, step + 1, v_flavor, scorer))

    return trace, first_pick_evsi
