"""Derangements for the shuffled-`u` control arm: each learner is given another learner's belief.

No learner may keep their own belief, so the control needs a derangement, not merely a
random permutation. `torch.randperm` is not enough: for n=2 it returns the identity
permutation half the time (only two permutations exist, one of them the identity), and a
"shuffle" that silently leaves rows unshuffled is not a control at all.
"""

from __future__ import annotations

import numpy as np
import torch


def derangement_permutation(n: int, generator: torch.Generator | None = None) -> torch.Tensor:
    """A permutation of `range(n)` with NO fixed points: `perm[i] != i` for every `i`.

    Rejection-sampled -- draw a fresh permutation, retry if any index maps to itself.
    The acceptance rate approaches `1/e` (~36.8%) for large `n` and is exactly 50% at
    n=2, so this converges in a handful of draws at any evaluation population size; it is not a search, it is drawing until a cheap check passes.

    Raises `ValueError` for `n <= 1` (no derangement exists) -- callers must guard
    against a batch/population of size 1 before calling this.
    """
    if n <= 1:
        raise ValueError(f"no derangement exists for n={n}")
    identity = torch.arange(n)
    for _ in range(10_000):
        perm = torch.randperm(n, generator=generator)
        if not torch.any(perm == identity):
            return perm
    raise RuntimeError(f"failed to find a derangement for n={n} after 10,000 attempts "
                       "-- this should be essentially impossible for any n > 1")


def assert_derangement(perm: torch.Tensor) -> None:
    """The permanent canary this module exists to make possible: assert a candidate
    permutation has no fixed points. Raises with the exact offending indices on failure,
    not just a bare AssertionError, since "which rows kept their own belief" is the
    actionable information when this fires."""
    n = perm.shape[0]
    fixed = torch.nonzero(perm == torch.arange(n, device=perm.device), as_tuple=True)[0]
    assert fixed.numel() == 0, (
        f"NOT a derangement: {fixed.numel()} of {n} indices map to themselves "
        f"(e.g. {fixed[:10].tolist()}) -- a vacuous shuffle "
        "(torch.randperm has no derangement guarantee).")


def learner_history_matched_derangement(learner_ids: np.ndarray, bin_ids: np.ndarray,
                                        rng: np.random.Generator | None = None,
                                        max_attempts: int = 10_000
                                        ) -> tuple[np.ndarray, np.ndarray]:
    """LEARNER-LEVEL, HISTORY-MATCHED derangement for FA-SIM3's shuffled-`u` arm. `derangement_permutation` above is ROW-level: it guarantees no row
    keeps its OWN belief, but says nothing about identity or history maturity, so it can
    (and, over enough rows, will) hand a cold-start row a mature learner's belief or --
    worse -- hand one row of a multi-row learner ANOTHER row belonging to the SAME
    learner, which is not a control at all (same person, not "another learner's belief").

    Deranges WITHIN each `bin_ids` group (callers bin by history length `t`, matching
    `atom.eval.evals_5g.history_curve`'s own bins, so the shuffled belief is never more
    or less mature than the row's own) -- and, within a bin, rejects any candidate
    permutation where a row's `learner_ids` matches the learner it would receive, not
    merely its own row index (`sub_learners[local_perm] == sub_learners`, not
    `local_perm == arange`) -- a stricter condition than `derangement_permutation`'s,
    since one learner can own several rows in the same bin.

    A bin with fewer than 2 DISTINCT learners cannot be deranged at all (there is no
    "another learner" to assign) -- those rows are NOT silently left self-mapped and
    passed off as a control; `deranged_mask[i]=False` flags them, and callers must
    exclude flagged rows from the shuffled-`u` arm's comparisons (never trust `perm[i]`
    for a flagged row -- it is `perm[i] == i` there, i.e. IDENTITY, which is the exact
    failure mode this whole function exists to prevent elsewhere).

    Returns `(perm, deranged_mask)`: `perm[i]` is the row index whose belief row `i`
    receives; `deranged_mask[i]` is `True` iff row `i` was genuinely deranged (a
    different learner's belief)."""
    rng = rng if rng is not None else np.random.default_rng()
    n = len(learner_ids)
    perm = np.arange(n)
    deranged_mask = np.ones(n, dtype=bool)
    for b in np.unique(bin_ids):
        idx = np.flatnonzero(bin_ids == b)
        sub_learners = np.asarray(learner_ids)[idx]
        local_n = len(idx)
        _, counts = np.unique(sub_learners, return_counts=True)
        if counts.size < 2:
            deranged_mask[idx] = False
            continue
        # Pigeonhole feasibility check, not merely an attempt-budget heuristic: a
        # learner-matched derangement (a bijection where no position maps to its own
        # group) exists iff the largest group has at most floor(n/2) members -- if one
        # learner owns a majority of the bin, every one of their own rows would need a
        # DISTINCT non-self-group target, but there are not enough such targets to go
        # around. Checked upfront so an infeasible bin fails FAST and with a clear
        # reason, not after burning `max_attempts` rejection-sampling draws that could
        # never have succeeded.
        if counts.max() > local_n // 2:
            raise RuntimeError(
                f"bin {b!r} cannot be learner-matched deranged: the largest learner "
                f"group has {counts.max()} of {local_n} rows, exceeding floor(n/2)="
                f"{local_n // 2} -- no bijection avoiding same-learner pairs can exist "
                "for this composition (pigeonhole); this bin's history-bin boundaries "
                "or minimum-bin-size should be reconsidered, not retried")
        found = False
        for _ in range(max_attempts):
            local_perm = rng.permutation(local_n)
            if not np.any(sub_learners[local_perm] == sub_learners):
                perm[idx] = idx[local_perm]
                found = True
                break
        if not found:
            raise RuntimeError(
                f"failed to find a learner-matched derangement for bin {b!r} "
                f"(n={local_n}, n_distinct_learners={counts.size}) "
                f"after {max_attempts} attempts despite passing the feasibility check -- "
                "investigate before retrying")
    return perm, deranged_mask


def assert_learner_matched_derangement(perm: np.ndarray, learner_ids: np.ndarray,
                                       deranged_mask: np.ndarray) -> None:
    """The permanent canary for the learner-matched derangement: every row flagged
    `deranged_mask=True` must map to a DIFFERENT learner's row
    (`learner_id[perm[i]] != learner_id[i]`). Rows with `deranged_mask=False`
    are exempt (they are documented as never deranged, not silently passed as if they
    were) but must still map to themselves (`perm[i] == i`), never to an unverified
    row -- catches a caller that mixed up which rows were flagged."""
    learner_ids = np.asarray(learner_ids)
    same_learner = (learner_ids[perm] == learner_ids) & deranged_mask
    bad = np.flatnonzero(same_learner)
    assert bad.size == 0, (
        f"NOT a learner-matched derangement: {bad.size} rows flagged deranged=True map "
        f"to their OWN learner's belief (e.g. rows {bad[:10].tolist()}) -- "
        "learner_id[perm[i]] != learner_id[i] must hold for every deranged row.")
    not_self = np.flatnonzero((perm == np.arange(len(perm))) != (~deranged_mask))
    assert not_self.size == 0, (
        f"{not_self.size} rows disagree between `perm[i]==i` and `deranged_mask[i]==False` "
        f"(e.g. rows {not_self[:10].tolist()}) -- an underanged row must map to itself, "
        "and a deranged row must never map to itself.")
