"""Belief/prediction path for running the planner on FA-SIM3, the mathematics simulator
(thesis Section 8.2).

**What this stack is.** FA-SIM3's own hierarchical FIRST STAGE: `S1FilterV2ItemMLP` (the
query-independent filter with an MLP item encoder) plus `OutcomeHead` — P(class | u, s(q))
over the 5-class FA outcome label. Both come out of `artifacts/checkpoints/fa_sim3_final.pt`. The checkpoint's
`hyper`/`basis` entries are the hypernetwork and the LoRA basis, i.e. the GENERATION path
through the frozen backbone: they are deliberately **not loaded here**. The emission the
planner prices is a small head with no language-model call, and that is what makes running
this belief inside the planner's per-candidate loop cheap.

**Why a separate module.** `replay.py` and `replay_v2.py` serve the earlier beliefs and are
left unchanged so results computed with them stay reproducible. This module is new because
the head genuinely changed shape — 5 categorical logits from a [N,D]×[N,L] call, not one Bernoulli
logit from the `FiLMHeads` [N,T,·] convention — and because the answer space is now a *choice* that has
to be made explicitly rather than inherited.

**The asymmetry, and why it is not a fudge** (pre-registered before any measurement).
Three of the five classes — HINT_ONLY, REVEAL_ONLY, HINT_AND_REVEAL — *are* the help-seeking
channel. The acquired observation is correctness-only precisely because a candidate's logged
help channels are future outcomes unknowable at selection time, and a give-up is not an
askable action. So:

  * **Targets are PREDICTED, never revealed** ⇒ their uncertainty may be the native 5-class
    categorical. Nothing about a target is ever fed back into the belief, so no future
    information can leak. These land under `value_total_entropy` / `value_jensen_gap`, which
    `evsi._value` reads for V(b).
  * **Candidates are REVEALED** ⇒ what asking one conveys stays binary (unaided-correct vs
    not). These land under the ordinary `mean_prob` / `total_entropy` / `epistemic_std` keys,
    which is what `candidate_evsi_scores` uses for `p_e` and what the two max-uncertainty
    baselines pick by — so the baselines price exactly the quantity that is actually
    acquirable, and the EPIG-vs-BALD contrast stays in one answer space.

Net (arm A, `answer_space="categorical"`): `EVSI(e) = I(Y_e^binary ; Y_T^5-class)`. Arm B (`answer_space="binary"`) collapses both
sides to the Bernoulli marginal P(UNAIDED_CORRECT) and is the within-substrate control that
isolates the answer space from the belief.
"""

from __future__ import annotations

import pathlib

import torch

from atom.filter.episode_v2 import build_prev_feat_v2
from atom.filter.gru import sample_u
from atom.filter.item_encoder_mlp import S1FilterV2ItemMLP
from atom.heads.fa_hierarchical_heads import OutcomeHead
from atom.planner.replay import _binary_entropy

ANSWER_SPACES = ("categorical", "binary")
UNAIDED_CORRECT = 0  # atom.data.fa_outcome_label.OUTCOME_CLASSES[0]

# Sampler-semantics tag. A checkpoint without this tag may exponentiate `log_sigma`
# inconsistently, so its beliefs are unusable; the planner must never silently consume one.
GOOD_SAMPLER_SEMANTICS = "reparameterize_from_log_sigma_v1_2026-08-15"


def load_sim3_stack(ckpt_path: pathlib.Path, device: str = "cpu",
                    head_hidden: int | None = None) -> tuple[S1FilterV2ItemMLP, OutcomeHead, dict]:
    """Load FA-SIM3's filter + OutcomeHead, frozen and in eval mode, asserting the three
    things that would otherwise fail silently: the sampler semantics tag is the expected one, the
    architecture is the one recorded in the checkpoint (never a hardcoded guess), and both
    state dicts load STRICTLY. `hyper`/`basis` are ignored by design (see module docstring).
    """
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    got = ck.get("sampler_semantics")
    if got != GOOD_SAMPLER_SEMANTICS:
        raise ValueError(
            f"{ckpt_path.name} carries sampler_semantics={got!r}, not {GOOD_SAMPLER_SEMANTICS!r}. "
            "A checkpoint with other sampler semantics has an unreliable log_sigma; refusing to build beliefs from it.")
    arch = dict(ck["architecture"])
    # The width-parameterisation ablation's checkpoint (infra/fa_sigma_a3) uses the
    # precision-accumulating sigma head and records it here. Checkpoints without this
    # metadata use the direct parameterisation, so the default preserves their loading path
    # while allowing the ablation's checkpoint to be consumed exactly as saved.
    sigma_parameterization = arch.get("sigma_parameterization", "direct")
    filt = S1FilterV2ItemMLP(codec_dim=arch["codec_dim"], episode_hidden=arch["episode_hidden"],
                             filter_hidden=arch["filter_hidden"], latent_dim=arch["latent_dim"],
                             query_independent=True,
                             sigma_parameterization=sigma_parameterization).to(device)
    filt.load_state_dict(ck["filter"], strict=True)
    # The head's hidden width is not in `architecture`; read it BY VALUE off the weight it
    # determines rather than hardcoding 2560 (metadata carried by value, never re-derived).
    hidden = head_hidden if head_hidden is not None else int(ck["outcome_head"]["outcome.weight"].shape[1])
    n_classes = int(ck["outcome_head"]["outcome.weight"].shape[0])
    head = OutcomeHead(codec_dim=arch["codec_dim"], latent_dim=arch["latent_dim"],
                       hidden=hidden, n_classes=n_classes).to(device)
    head.load_state_dict(ck["outcome_head"], strict=True)
    filt.eval()
    head.eval()
    for p in list(filt.parameters()) + list(head.parameters()):
        p.requires_grad_(False)   # the planner measures the belief, it never trains it.
    meta = {"architecture": arch, "head_hidden": hidden, "n_classes": n_classes,
            "sampler_semantics": got, "git_sha": ck.get("git_sha"), "seed": ck.get("seed"),
            "steps_personalised": ck.get("steps_personalised"),
            "codec_path": ck.get("codec_path"), "codec_sha256": ck.get("codec_sha256")}
    return filt, head, meta


def filter_belief_sim3(filter_model, revealed_item_codec: torch.Tensor,
                       revealed_correct: torch.Tensor, revealed_hint_bin: torch.Tensor,
                       revealed_saw_answer: torch.Tensor, revealed_score_missing: torch.Tensor,
                       n_probe: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """b_t = f(H_<t), returned expanded to [n_probe, latent_dim].

    **Why this takes `n_probe` instead of the probe codecs themselves.** Under
    `query_independent=True` the probe's own codec provably cannot reach the belief, so all
    n_probe rows are the same vector — `replay_v2.filter_belief_v2` builds an [n_probe, T, D]
    tensor and runs the GRU n_probe times to produce n_probe copies of one answer. At
    `S1FilterV2`'s width (hidden 64, latent 32) that waste is affordable; FA-SIM3 is hidden
    256 / latent 256 with a 2-layer MLP item encoder, and the planner evaluates this inside a
    per-candidate loop. So the belief is computed ONCE and expanded.

    This is a shortcut that is only valid because of a property, so the property is CHECKED,
    never assumed: `assert_query_independent_sim3` is the peek canary, and
    `test_shared_belief_fast_path_is_bit_identical_to_per_probe` pins this function
    bit-for-bit against the honest per-probe construction.
    """
    m = revealed_item_codec.shape[0]
    codec_dim = revealed_item_codec.shape[1] if m else filter_model.episode_encoder.item_proj.net[0].in_features
    t_total = m + 1

    item_codec = torch.zeros(1, t_total, codec_dim, device=device)
    correct = torch.zeros(1, t_total, device=device)
    hint_bin = torch.zeros(1, t_total, device=device)
    saw_answer = torch.zeros(1, t_total, device=device)
    score_missing = torch.zeros(1, t_total, device=device)
    if m > 0:
        item_codec[0, :m] = revealed_item_codec
        correct[0, :m] = revealed_correct
        hint_bin[0, :m] = revealed_hint_bin
        saw_answer[0, :m] = revealed_saw_answer
        score_missing[0, :m] = revealed_score_missing
    # Position m's item codec stays ZERO: it is the probe slot, and under a query-independent
    # filter the recurrent input at position m is event m-1's codec, so whatever sits at m is
    # never read into the belief at m. Leaving it zero rather than filling in a probe makes
    # that independence structural here instead of merely true.

    with torch.no_grad():
        prev_feat = build_prev_feat_v2(correct, hint_bin, saw_answer, score_missing)
        mu, log_sigma = filter_model(item_codec, prev_feat)
    return mu[0, m].expand(n_probe, -1), log_sigma[0, m].expand(n_probe, -1)


def _class_probs(head, item_codec: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """Softmax class probabilities [K, T, C] from the SHIPPED `OutcomeHead`, called once on a
    flattened [K*T, ·] batch rather than reimplemented in a vectorised form -- a hand-rolled
    copy of the FiLM arithmetic is exactly how a control stops testing the shipped code.
    item_codec: [T, D]. u: [K, T, L]."""
    k, t = u.shape[0], u.shape[1]
    flat_codec = item_codec.unsqueeze(0).expand(k, -1, -1).reshape(k * t, -1)
    with torch.no_grad():
        logits = head(flat_codec, u.reshape(k * t, -1))
    return torch.softmax(logits.view(k, t, -1), dim=-1)


def _categorical_entropy(p: torch.Tensor) -> torch.Tensor:
    """H over the last axis, in nats. Clamped like `_binary_entropy` so a saturated class
    cannot produce a NaN that would then propagate silently into V(b)."""
    p = p.clamp_min(1e-12)
    return -(p * p.log()).sum(dim=-1)


def predict_on_targets_sim3(filter_model, head, revealed_item_codec: torch.Tensor,
                            revealed_correct: torch.Tensor, revealed_hint_bin: torch.Tensor,
                            revealed_saw_answer: torch.Tensor, revealed_score_missing: torch.Tensor,
                            target_item_codec: torch.Tensor, k: int, device: str,
                            noise: torch.Tensor | None = None,
                            answer_space: str = "categorical") -> dict[str, torch.Tensor]:
    """Return contract is a SUPERSET of `replay_v2.predict_on_targets_v2`'s, so the shipped
    policy code consumes it unchanged.

    Always present (the CANDIDATE-side, correctness-only quantities):
      `mean_prob` = P(UNAIDED_CORRECT), `total_entropy`, `jensen_gap`, `epistemic_std`, all [T].
    Present iff `answer_space="categorical"` (the TARGET-side, native quantities):
      `mean_probs` [T,C], `value_total_entropy` [T], `value_jensen_gap` [T].

    `evsi._value` prefers `value_*` when present, so arm A prices targets categorically while
    `p_e` and the max-uncertainty baselines stay binary. Under `answer_space="binary"` the
    `value_*` keys are absent and every quantity is the Bernoulli marginal -- arm B.
    """
    if answer_space not in ANSWER_SPACES:
        raise ValueError(f"unknown answer_space {answer_space!r}, expected one of {ANSWER_SPACES}")
    n_t = target_item_codec.shape[0]
    mu, log_sigma = filter_belief_sim3(filter_model, revealed_item_codec, revealed_correct,
                                       revealed_hint_bin, revealed_saw_answer,
                                       revealed_score_missing, n_t, device)
    with torch.no_grad():
        u = sample_u(mu.unsqueeze(1), log_sigma.unsqueeze(1), k=k, noise=noise)  # [K, T, 1, L]
        probs_kc = _class_probs(head, target_item_codec, u.squeeze(2))            # [K, T, C]

    # The binary marginal: "the learner solves it unaided". Its complement pools the four
    # other classes, which is exactly the coarsening a revealed candidate conveys.
    p0_k = probs_kc[..., UNAIDED_CORRECT]                                          # [K, T]
    mean_prob = p0_k.mean(dim=0)
    total_entropy = _binary_entropy(mean_prob)
    out = {
        "mean_prob": mean_prob,
        "total_entropy": total_entropy,
        "jensen_gap": (total_entropy - _binary_entropy(p0_k).mean(dim=0)).clamp(min=0.0),
        "epistemic_std": p0_k.std(dim=0),
        "probs_k": p0_k,
        "mu": mu, "log_sigma": log_sigma,
    }
    if answer_space == "categorical":
        mean_probs = probs_kc.mean(dim=0)                                          # [T, C]
        cat_total = _categorical_entropy(mean_probs)
        out["mean_probs"] = mean_probs
        out["value_probs_k"] = probs_kc
        out["value_total_entropy"] = cat_total
        out["value_jensen_gap"] = (cat_total - _categorical_entropy(probs_kc).mean(dim=0)).clamp(min=0.0)
    return out


def categorical_score_step(target_out: dict[str, torch.Tensor], target_class: torch.Tensor,
                           n_asked: int, v_flavor: str = "jensen") -> dict:
    """`evsi._score_step`'s categorical counterpart, injected via its `scorer` hook. Scores
    `-log p̄(true class)` and top-1 accuracy over the 5-class predictive -- arm A's external
    score has to live in the same answer space as arm A's targets, or V(b) and the number it
    is judged by would be measuring different things.

    `target_class`: [T] int64 outcome codes. Note this is the ONLY place the true 5-class
    label is read; selection never sees it."""
    from atom.planner.evsi import _value
    p = target_out["mean_probs"].clamp_min(1e-12)
    idx = target_class.long().view(-1, 1)
    logloss = float(-p.gather(1, idx).squeeze(1).log().mean().item())
    accuracy = float((p.argmax(dim=-1) == target_class.long()).float().mean().item())
    return {"n_asked": n_asked, "V": _value(target_out, v_flavor), "logloss": logloss,
            "accuracy": accuracy}


def sigma_trace(log_sigma: torch.Tensor) -> float:
    """tr(Σ) = Σ_l σ_l². Reported per belief; identical across probe rows by construction."""
    return float((2.0 * log_sigma).exp().sum(dim=-1).mean().item())


def assert_query_independent_sim3(filter_model, revealed_item_codec: torch.Tensor,
                                  revealed_correct: torch.Tensor, revealed_hint_bin: torch.Tensor,
                                  revealed_saw_answer: torch.Tensor,
                                  revealed_score_missing: torch.Tensor,
                                  probe_a: torch.Tensor, probe_b: torch.Tensor,
                                  device: str) -> float:
    """The peek canary, fired through THIS module's own wiring: same history, two different
    candidate questions must give a bit-identical belief. Returns max(|Δmu|, |Δlog σ|), which
    must be EXACTLY 0.0. Built to go through the honest per-probe construction (probe codecs
    actually placed at position m), so it can still fail -- a canary computed through the
    shared-belief fast path would be true by construction and would test nothing."""
    args = (revealed_item_codec, revealed_correct, revealed_hint_bin, revealed_saw_answer,
            revealed_score_missing)
    mu_a, ls_a = _belief_with_probe(filter_model, *args, probe_a, device)
    mu_b, ls_b = _belief_with_probe(filter_model, *args, probe_b, device)
    return max((mu_a - mu_b).abs().max().item(), (ls_a - ls_b).abs().max().item())


def _belief_with_probe(filter_model, revealed_item_codec: torch.Tensor,
                       revealed_correct: torch.Tensor, revealed_hint_bin: torch.Tensor,
                       revealed_saw_answer: torch.Tensor, revealed_score_missing: torch.Tensor,
                       probe_item_codec: torch.Tensor, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """The honest per-probe belief: probe codecs really are written into position m. Used by
    the peek canary and by the fast path's bit-identity test -- never on the hot path."""
    m = revealed_item_codec.shape[0]
    n_probe, codec_dim = probe_item_codec.shape
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
    with torch.no_grad():
        prev_feat = build_prev_feat_v2(correct, hint_bin, saw_answer, score_missing)
        mu, log_sigma = filter_model(item_codec, prev_feat)
    return mu[:, m], log_sigma[:, m]
