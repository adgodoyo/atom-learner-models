"""Belief/prediction path for running the planner on the query-independent `S1FilterV2`
belief (thesis Section 5.7).

**Why a new module rather than an edit to `replay.py`.** `replay.py` serves the v1 filter
and is left unchanged so results computed with it stay reproducible. `S1FilterV2` also needs
a genuinely different recurrent input — `build_prev_feat_v2`'s **5** channels, with the
`prev_score_missing` flag required because missing scores are flagged, never imputed — so a
v2 path is not a parameter tweak. `replay.py` is untouched and its tests pass unedited.

**The three differences from v1, all of them load-bearing:**

1. **`prev_feat` has five channels, not four.** The fifth is `prev_score_missing`. The FA
   manifest never imputes a missing platform score to 0, so `prev_correct` at such a
   position is a placeholder rather than a fact; dropping the flag would hand the filter's
   recurrent input placeholder scores as if they were observations. The planner must therefore carry
   `score_missing` through its windows — which is why every function here takes it explicitly
   instead of defaulting it to zeros.
2. **The one-position shift happens INSIDE `S1FilterV2.forward`.** Callers hand it the
   CURRENT item's codec, unshifted, exactly as `FiLMHeads` also consumes it. Pre-shifting
   here would double-shift and silently mis-align the belief by one event.
3. **The belief is query-independent by construction** (`b_t = f(H_<t)`), which v1's filter
   was not. `assert_query_independent_v2` is the peek canary for that property, checked
   through THIS path rather than assumed from the filter's own training harness.

Everything else — the reconstruction argument (skip gaps, never reorder), one probe per row
so no target sees another's label, the K-sample decomposition, the common-random-numbers
`noise` argument — is v1's semantics unchanged, because none of it was what changed.
"""

from __future__ import annotations

import torch

from atom.filter.episode_v2 import build_prev_feat_v2
from atom.filter.gru import sample_u
from atom.planner.replay import _binary_entropy


def filter_belief_v2(filter_model, revealed_item_codec: torch.Tensor,
                     revealed_correct: torch.Tensor, revealed_hint_bin: torch.Tensor,
                     revealed_saw_answer: torch.Tensor, revealed_score_missing: torch.Tensor,
                     probe_item_codec: torch.Tensor, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """v2 counterpart of `replay._filter_belief`. revealed_*: [m] (m may be 0).
    probe_item_codec: [n_probe, codec_dim]. Returns (mu, log_sigma), each
    [n_probe, latent_dim] — the belief at each probe row.

    Under `S1FilterV2(query_independent=True)` the probe's own codec cannot reach the belief
    (the recurrent input at position m is event m-1's codec), so all n_probe rows share one
    belief by construction. That is the property, not an implementation accident, and
    `assert_query_independent_v2` checks it rather than trusting it.
    """
    m = revealed_item_codec.shape[0]
    n_probe = probe_item_codec.shape[0]
    codec_dim = probe_item_codec.shape[1]
    t_total = m + 1

    item_codec = torch.zeros(n_probe, t_total, codec_dim, device=device)
    correct = torch.zeros(n_probe, t_total, device=device)
    hint_bin = torch.zeros(n_probe, t_total, device=device)
    saw_answer = torch.zeros(n_probe, t_total, device=device)
    score_missing = torch.zeros(n_probe, t_total, device=device)
    if m > 0:
        item_codec[:, :m] = revealed_item_codec.unsqueeze(0)
        correct[:, :m] = revealed_correct.unsqueeze(0)
        hint_bin[:, :m] = revealed_hint_bin.unsqueeze(0)
        saw_answer[:, :m] = revealed_saw_answer.unsqueeze(0)
        score_missing[:, :m] = revealed_score_missing.unsqueeze(0)
    item_codec[:, m] = probe_item_codec
    # Position m's own correct/hint/saw/score_missing are never read: build_prev_feat_v2
    # shifts positions 0..m-1 into prev_feat[:, m], and there is no position m+1 here.

    with torch.no_grad():
        prev_feat = build_prev_feat_v2(correct, hint_bin, saw_answer, score_missing)
        mu, log_sigma = filter_model(item_codec, prev_feat)
    return mu[:, m], log_sigma[:, m]


def predict_on_targets_v2(filter_model, heads, revealed_item_codec: torch.Tensor,
                          revealed_correct: torch.Tensor, revealed_hint_bin: torch.Tensor,
                          revealed_saw_answer: torch.Tensor, revealed_score_missing: torch.Tensor,
                          target_item_codec: torch.Tensor, k: int, device: str,
                          noise: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
    """v2 counterpart of `replay.predict_on_targets`; identical return contract
    (`mean_prob`, `total_entropy`, `jensen_gap`, `epistemic_std`, each [n_targets]) plus
    `mu`/`log_sigma` so callers can report Σ-trace without a second forward pass."""
    mu, log_sigma = filter_belief_v2(filter_model, revealed_item_codec, revealed_correct,
                                     revealed_hint_bin, revealed_saw_answer,
                                     revealed_score_missing, target_item_codec, device)
    with torch.no_grad():
        u = sample_u(mu.unsqueeze(1), log_sigma.unsqueeze(1), k=k, noise=noise)
        logits, _ = heads(target_item_codec.unsqueeze(1), u)
        probs_k = torch.sigmoid(logits).squeeze(-1)  # [K, n_targets]

    mean_prob = probs_k.mean(dim=0)
    total_entropy = _binary_entropy(mean_prob)
    jensen_gap = (total_entropy - _binary_entropy(probs_k).mean(dim=0)).clamp(min=0.0)
    return {"mean_prob": mean_prob, "total_entropy": total_entropy, "jensen_gap": jensen_gap,
            "epistemic_std": probs_k.std(dim=0), "mu": mu, "log_sigma": log_sigma,
            "probs_k": probs_k}


def sigma_trace(log_sigma: torch.Tensor) -> float:
    """tr(Σ) = Σ_l σ_l² for the diagonal Gaussian belief — the scalar 'how wide is the
    belief' the headroom curve plots beside log-loss. Averaged over probe rows, which are
    identical under a query-independent filter."""
    return float((2.0 * log_sigma).exp().sum(dim=-1).mean().item())


def assert_query_independent_v2(filter_model, revealed_item_codec: torch.Tensor,
                                revealed_correct: torch.Tensor, revealed_hint_bin: torch.Tensor,
                                revealed_saw_answer: torch.Tensor,
                                revealed_score_missing: torch.Tensor,
                                probe_a: torch.Tensor, probe_b: torch.Tensor,
                                device: str) -> float:
    """The peek canary, fired through the PLANNER's own path: same history, two different
    candidate questions must give a bit-identical belief. Returns max(|Δmu|, |Δlog_sigma|),
    which must be EXACTLY 0.0.

    The property was established for the filter in its own training harness, which is not
    the same as proving this module wires it up correctly. A double-shift or a mis-built `prev_feat` here would reintroduce
    query dependence while every shape still matched.
    """
    args = (revealed_item_codec, revealed_correct, revealed_hint_bin, revealed_saw_answer,
            revealed_score_missing)
    mu_a, ls_a = filter_belief_v2(filter_model, *args, probe_a, device)
    mu_b, ls_b = filter_belief_v2(filter_model, *args, probe_b, device)
    return max((mu_a - mu_b).abs().max().item(), (ls_a - ls_b).abs().max().item())
