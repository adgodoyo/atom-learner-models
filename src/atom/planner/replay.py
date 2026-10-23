"""Held-out-pool replay harness for the planner (thesis Section 5.7). Shared
infrastructure every ask policy (greedy EVSI, random, max-uncertainty) replays through: a
windowed prefix/candidate/target split per learner, and a belief-update primitive that
re-runs the FROZEN GRU belief filter (`S1Filter`) on a
*constructed* sub-sequence -- prefix events plus whichever candidates have been
"revealed" so far, in their own chronological order.

**Windowed causality (binding protocol constraint).** Each learner's sequence splits
prefix ‖ candidate window ‖ target window, with every target strictly LATER than every
candidate -- revealed evidence is always causally upstream of what it is scored on,
so the quasi-static-latent assumption only has to span the (bounded) candidate window,
not the whole hidden tail.

Why reconstruction, not masking-in-place: the GRU filter (src/atom/filter/gru.py) is causal
and carries state strictly forward through whatever sequence it is fed; it has no
mechanism to skip over an unrevealed item while still advancing past it. Reconstructing
a shorter sequence out of exactly the events that were actually observed (prefix +
revealed) is therefore the correct way to ask "what would the belief be if this
learner's real history were only these events" -- not an approximation of convenience.

This reconstruction is architecturally sound for this filter specifically because it has
no forgetting kernel (the OU kernel was ablated and dropped; thesis Appendix A) and no
elapsed-time input at all (build_prev_feat carries only [prev_correct, prev_hint_bin,
prev_saw_answer, prev_valid] -- no gap/duration field), so the model already treats any
two consecutive fed events identically regardless of how much real time separated them.
Skipping the events between "revealed" items introduces no hidden time-elapsed artifact
that isn't already absent from the model's inputs. **But "any two consecutive fed events
identically" means the events fed to the GRU must still be in their own true chronological
ORDER** -- the absence of a time input licenses skipping gaps, it does NOT license
feeding events out of order (that would be a different learner's history, not a gapped
slice of this one). `_materialize` below (used by every caller, policy or sanity check)
enforces this by re-sorting whichever candidate indices are "revealed" into chronological
order before every forward pass, regardless of what order they were selected/revealed in.

`sanity_check_reconstruction` is the DETERMINISTIC validity check that this
reconstruction path is bug-free: it compares the belief (mu, log_sigma) directly, no
K-sampling and no heads, so its floor is true float32 noise (~1e-5), not the K-sample
sampling-noise floor (~0.010 at k=64) that a probability-based comparison would measure
instead of a real bug.

`predict_on_targets` batches every target item as a SEPARATE row (not appended in one
causal chain), each sharing the identical revealed-so-far prefix: this is what stops a
later target from "seeing" an earlier target's true label through prev_feat's shift --
each target's prediction is conditioned only on prefix+revealed, never on other targets.
It returns the K-sample decomposition V(b) needs: the mean predicted
probability, its total predictive (Bernoulli) entropy, the K-sample Jensen gap
(the epistemic component of that entropy, used by the default V(b) in evsi.py), the
epistemic-only K-spread (the stronger max-uncertainty baseline, giving the EPIG-vs-BALD
contrast), and the raw K-sample probabilities consumed by the EPIG estimator.
Its optional `noise` argument implements the common-random-numbers variance control
(see src/atom/filter/gru.py::sample_u) -- reusing one frozen K-draw across every call in
a learner's episode cancels independent-sampling noise from the comparisons BETWEEN those
calls (candidates, y-branches, policies) without changing what any single call computes.
"""

from __future__ import annotations

import numpy as np
import torch

from atom.filter.episode import build_prev_feat
from atom.filter.gru import sample_u


def split_prefix_candidate_target(t_max: int, prefix_k: int, candidate_w: int,
                                  n_target: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One learner's positions 0..t_max-1 -> (prefix_idx, candidate_idx, target_idx),
    three CONTIGUOUS, ORDERED windows (not sampled) -- prefix ‖ candidates ‖ targets, so
    every target index is strictly greater than every candidate index (windowed
    causality). Returns empty candidate/target arrays if `t_max` is too short to fit
    the requested window sizes after the prefix (disclosed by the caller, not silently
    truncated to something else).
    """
    prefix_idx = np.arange(min(prefix_k, t_max))
    cand_start = prefix_k
    cand_end = min(prefix_k + candidate_w, t_max)
    candidate_idx = np.arange(cand_start, cand_end)
    target_start = cand_end
    target_end = min(target_start + n_target, t_max)
    target_idx = np.arange(target_start, target_end)
    return prefix_idx, candidate_idx, target_idx


def _binary_entropy(p: torch.Tensor) -> torch.Tensor:
    p = p.clamp(1e-6, 1 - 1e-6)
    return -(p * p.log() + (1 - p) * (1 - p).log())


def _filter_belief(filter_model, revealed_item_codec: torch.Tensor, revealed_correct: torch.Tensor,
                   revealed_hint_bin: torch.Tensor, revealed_saw_answer: torch.Tensor,
                   probe_item_codec: torch.Tensor, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """The deterministic half of predict_on_targets, factored out so the sanity check can
    compare (mu, log_sigma) directly without ever touching heads or K-sampling.
    revealed_*: [m] (m may be 0). probe_item_codec: [n_probe, codec_dim]. Returns
    (mu, log_sigma), each [n_probe, latent_dim] -- the PRE-OBSERVATION belief at each
    probe row (one probe per row, none sees another's outcome), exactly the belief
    predict_on_targets samples `u` from.
    """
    m = revealed_item_codec.shape[0]
    n_probe = probe_item_codec.shape[0]
    codec_dim = probe_item_codec.shape[1]
    t_total = m + 1

    item_codec = torch.zeros(n_probe, t_total, codec_dim, device=device)
    correct = torch.zeros(n_probe, t_total, device=device)
    hint_bin = torch.zeros(n_probe, t_total, device=device)
    saw_answer = torch.zeros(n_probe, t_total, device=device)
    if m > 0:
        item_codec[:, :m] = revealed_item_codec.unsqueeze(0)
        correct[:, :m] = revealed_correct.unsqueeze(0)
        hint_bin[:, :m] = revealed_hint_bin.unsqueeze(0)
        saw_answer[:, :m] = revealed_saw_answer.unsqueeze(0)
    item_codec[:, m] = probe_item_codec
    # correct/hint_bin/saw_answer at position m are never read: build_prev_feat only
    # shifts positions 0..m-1 into prev_feat[:, m]; there is no position m+1 here for a
    # shifted prev_feat to read the probe's own (unknown-until-scored) label from.

    with torch.no_grad():
        prev_feat = build_prev_feat(correct, hint_bin, saw_answer)
        mu, log_sigma = filter_model(item_codec, prev_feat)
    return mu[:, m], log_sigma[:, m]  # [n_probe, latent_dim] each


def predict_on_targets(filter_model, heads, revealed_item_codec: torch.Tensor,
                       revealed_correct: torch.Tensor, revealed_hint_bin: torch.Tensor,
                       revealed_saw_answer: torch.Tensor, target_item_codec: torch.Tensor,
                       k: int, device: str, noise: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
    """revealed_*: [m] (m may be 0). target_item_codec: [n_targets, codec_dim]. `noise`:
    optional [K, latent_dim] common-random-numbers draw (see module docstring) -- reused
    as-is instead of fresh sampling when given.

    Returns a dict, each value [n_targets]:
      mean_prob        - K-sample-averaged predicted correctness probability.
      total_entropy     - H(mean_prob): total predictive uncertainty (V(b) for
                          v_flavor="total").
      jensen_gap        - H(mean_prob) - mean_k[H(p_k)]: the epistemic component of
                          total_entropy (>=0 by Jensen's inequality); V(b) for the
                          default v_flavor="jensen".
      epistemic_std     - std_k(p_k): K-sample spread of the probability itself, the
                          stronger max-uncertainty baseline.
    plus `probs_k` [K, n_targets], the per-sample probabilities.
    """
    mu, log_sigma = _filter_belief(filter_model, revealed_item_codec, revealed_correct,
                                   revealed_hint_bin, revealed_saw_answer, target_item_codec, device)
    with torch.no_grad():
        u = sample_u(mu.unsqueeze(1), log_sigma.unsqueeze(1), k=k, noise=noise)
        logits, _ = heads(target_item_codec.unsqueeze(1), u)
        probs_k = torch.sigmoid(logits).squeeze(-1)  # [K, n_targets]

    mean_prob = probs_k.mean(dim=0)
    total_entropy = _binary_entropy(mean_prob)
    mean_entropy_k = _binary_entropy(probs_k).mean(dim=0)
    jensen_gap = (total_entropy - mean_entropy_k).clamp(min=0.0)
    epistemic_std = probs_k.std(dim=0)

    return {"mean_prob": mean_prob, "total_entropy": total_entropy,
            "jensen_gap": jensen_gap, "epistemic_std": epistemic_std,
            # The refilter-free EPIG scorer consumes the SAME K committee values
            # directly. Keeping them here avoids a second forward and makes sample identity
            # explicit rather than inferred from two aggregate-only calls.
            "probs_k": probs_k}


def sanity_check_reconstruction(filter_model, item_codec_full: torch.Tensor, correct_full: torch.Tensor,
                                hint_bin_full: torch.Tensor, saw_answer_full: torch.Tensor,
                                prefix_idx: np.ndarray, candidate_idx: np.ndarray, device: str) -> float:
    """Deterministic validity check: compares the belief (mu, log_sigma) directly via
    `_filter_belief` -- no K-sampling, no heads -- so any divergence reflects true float32
    noise, not the K-sample noise floor a probability-based comparison would measure (about
    0.010 at k=64, which such a check reproduces almost exactly -- it measures itself, not
    a bug). It does not take `heads`/`k`: this checks the FILTER's belief only, which is what both the
    reconstruction argument (module docstring) and predict_on_targets's own K-sampling
    are built on top of.

    With EVERY candidate revealed in chronological order (no skipping), the reconstructed
    belief at each candidate position must reproduce the belief from directly slicing the
    natural, uninterrupted prefix+candidate-window sequence -- any divergence beyond float
    noise is a real implementation bug. Note that this checks the reconstruction path only;
    reveal-order correctness on the policy path is guarded separately by the full-budget
    bit-identity test of src/atom/planner/evsi.py.

    item_codec_full/correct_full/hint_bin_full/saw_answer_full: this learner's full,
    natural-order sequence, [T, ...]. Returns max(mean |mu diff|, mean |log_sigma diff|)
    over the candidate window's own positions.
    """
    all_idx = np.concatenate([prefix_idx, candidate_idx])
    natural_item_codec = item_codec_full[all_idx]
    natural_correct = correct_full[all_idx]
    natural_hint = hint_bin_full[all_idx]
    natural_saw = saw_answer_full[all_idx]

    with torch.no_grad():
        prev_feat = build_prev_feat(natural_correct.unsqueeze(0), natural_hint.unsqueeze(0),
                                    natural_saw.unsqueeze(0))
        natural_mu, natural_log_sigma = filter_model(natural_item_codec.unsqueeze(0), prev_feat)
    # natural_mu/log_sigma[0, i] is the belief-derived pre-observation state FOR position
    # all_idx[i] itself, using history strictly before it -- the candidate-window slice of
    # this lines up exactly with "prefix + all-candidates-so-far" at each successive candidate.
    n_prefix = len(prefix_idx)
    natural_cand_mu = natural_mu[0, n_prefix:]
    natural_cand_log_sigma = natural_log_sigma[0, n_prefix:]

    recon_mu, recon_log_sigma = [], []
    revealed_item_codec = item_codec_full[prefix_idx]
    revealed_correct = correct_full[prefix_idx]
    revealed_hint = hint_bin_full[prefix_idx]
    revealed_saw = saw_answer_full[prefix_idx]
    for pos in candidate_idx:
        mu, log_sigma = _filter_belief(filter_model, revealed_item_codec, revealed_correct,
                                       revealed_hint, revealed_saw, item_codec_full[pos:pos + 1],
                                       device=item_codec_full.device)
        recon_mu.append(mu[0])
        recon_log_sigma.append(log_sigma[0])
        revealed_item_codec = torch.cat([revealed_item_codec, item_codec_full[pos:pos + 1]])
        revealed_correct = torch.cat([revealed_correct, correct_full[pos:pos + 1]])
        revealed_hint = torch.cat([revealed_hint, hint_bin_full[pos:pos + 1]])
        revealed_saw = torch.cat([revealed_saw, saw_answer_full[pos:pos + 1]])

    recon_mu_t = torch.stack(recon_mu)
    recon_log_sigma_t = torch.stack(recon_log_sigma)
    mu_diff = float((recon_mu_t - natural_cand_mu).abs().mean().item())
    log_sigma_diff = float((recon_log_sigma_t - natural_cand_log_sigma).abs().mean().item())
    return max(mu_diff, log_sigma_diff)
