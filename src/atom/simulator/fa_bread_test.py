"""The bread test: does personalisation help on a task only the personalised path can do?
It scores the model on something the controls cannot do by construction. Primary metric: single-select MC
option-normalised NLL + top-1, scored on `atom.data.fa_mc_options.extract_single_select_mc`'s
population, personalised model vs three controls -- frozen backbone, global-LoRA
(`atom.binding.GlobalMixingVector`), and ΔW=0 (same LoRA-hooked path, mixing vector forced
to exactly zero). The decisive comparison is the ΔW=0 control: same backbone, same
data, personalisation off vs on.

Two layers, same split as `atom.eval.spread_error` vs `atom.simulator.fa_train_step`: the
per-option LM scoring needs the real model (`mc_option_log_likelihoods`, below); turning
those log-likelihoods into NLL/top-1/paired-CI is pure and lives in
`mc_option_metrics`/the metrics helpers here, so it is testable without a model too.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from atom.data.fa_outcome_label import N_OUTCOME
from atom.eval.metrics import bootstrap_mean_ci
from atom.simulator.fa_alt2 import (_prompt_prefix_suffix, collate, encode_example,
                                    forward_personalised, joint_kc_mixture_nll,
                                    production_log_likelihood_sum_per_row, production_nll_per_row)


def _assert_class_phrase_survives(example: dict, tok, outcome_code: int) -> None:
    """Class-phrase canary on the SCORING path: every scoring path must build the prompt
    shape the model was trained on, so the training-side check is applied here to every
    prompt that is scored, not only to the training encoder. Structural check, not a substring search --
    mirrors `tests/test_fa_sim2_class_phrase_canary.py`'s own assertion, applied here at
    call time on every prompt actually scored rather than only in a test."""
    _, suffix = _prompt_prefix_suffix(outcome_code)
    suffix_ids = tok(suffix, add_special_tokens=False)["input_ids"]
    prompt_ids = example["input_ids"][: example["prompt_end_index"] + 1]
    if len(prompt_ids) < len(suffix_ids) or prompt_ids[-len(suffix_ids):] != suffix_ids:
        raise AssertionError(
            f"class-phrase canary failed (outcome_code={outcome_code}): the scored prompt's "
            "tail does not match the class-conditioned suffix -- refusing to score a prompt "
            "shape no FA-SIM checkpoint was ever trained on")


@torch.no_grad()
def mc_option_log_likelihoods(model, hyper, context, u: torch.Tensor, tok, question: str,
                              options: list[str], max_len: int, device: str,
                              outcome_code: int) -> torch.Tensor:
    """`(len(options),)` length-normalised log-likelihood of the model producing each
    option string as the completion to `question`, all under the SAME personalisation `u`
    (one forward pass covering every option at once -- `u` is broadcast, not re-sampled per
    option, since the question being asked about is one fixed context). Higher = the model
    finds that option more likely, matching `-production_nll_per_row`'s own sign
    convention (a loss becomes a log-likelihood by negation, nothing else changes).

    `u` must be a single `(u_dim,)` vector (one example, not a batch) -- this function
    handles the per-option broadcast internally so callers never have to repeat `u` by
    hand and risk mismatching its count against `len(options)`.

    `outcome_code`: REQUIRED, no default. Falling back to `encode_example`'s own
    `outcome_code=None` default would build the UNCONDITIONED template, a prompt shape
    absent from every FA-SIM checkpoint's training data, so the default is removed here
    rather than corrected at one call site. There is no correct default -- the caller states
    which class to condition on. Every built prompt is checked by
    `_assert_class_phrase_survives` before scoring.
    """
    if u.dim() != 1:
        raise ValueError(f"mc_option_log_likelihoods expects a single (u_dim,) vector, got shape {tuple(u.shape)}")
    examples = [encode_example(question, opt, tok, max_len, outcome_code=outcome_code) for opt in options]
    for ex in examples:
        _assert_class_phrase_survives(ex, tok, outcome_code)
    tok_batch = collate(examples, pad_id=tok.pad_token_id, device=device)
    u_rep = u.unsqueeze(0).expand(len(options), -1)
    out = forward_personalised(model, hyper, context, u_rep, tok_batch["input_ids"],
                               tok_batch["attention_mask"], tok_batch["prompt_end_index"])
    per_row_nll, row_has_any = production_nll_per_row(
        out["lm_logits"], tok_batch["labels"], tok_batch["production_valid"])
    if not bool(row_has_any.all()):
        raise ValueError("mc_option_log_likelihoods: at least one option produced zero valid "
                         "completion tokens after max_len truncation -- an empty/degenerate "
                         "option string, or max_len too small for this question+option pair")
    return -per_row_nll


@torch.no_grad()
def mc_option_joint_log_likelihoods(model, hyper, context, outcome_head, u_k: torch.Tensor,
                                    item_codec: torch.Tensor, tok, question: str,
                                    options: list[str], max_len: int, device: str,
                                    max_batch_size: int = 16) -> dict:
    """The joint (k, c) MARGINALISED option scorer, implementing per option: `p(opt) = sum_c sum_k p(c|u_k)
    p(opt|c,u_k) / K`. Reuses `atom.simulator.fa_alt2.joint_kc_mixture_nll` (the
    brute-force-tested (k, c) estimator, `tests/
    test_fa_sim3_joint_kc_estimator.py`) UNCHANGED -- only the per-(class, K-sample)
    forward orchestration below is new, mirroring `infra/fa_sim3_full_run.py::
    k_marginalised_eval`'s own per-class loop at option-set granularity.

    `u_k`: `[K, u_dim]`, K already-drawn samples of this row's belief (reparameterise
    upstream, matching every other K-sample call site in this codebase -- this function
    never samples). `item_codec`: `[codec_dim]`, this row's own item vector --
    `outcome_head(item_codec, u_k)` gives `p(c|u_k)`, exactly as at training/eval time.

    For each class `c`, EVERY option is encoded under the class-conditioned template
    (`outcome_code=c`, as in `mc_option_log_likelihoods`) and checked by
    `_assert_class_phrase_survives`, applied to every prompt
    actually scored. RAW SEQUENCE log-likelihoods (`production_log_likelihood_sum_per_row`
    -- never a per-token mean; `joint_kc_mixture_nll` only accepts a true sequence
    probability; a per-token mean inside the mixture would be wrong) are collected into `[C,
    K, n_options]` and combined with `[C, K, n_options]` class log-probabilities
    (broadcast across options -- `p(c|u_k)` is a property of the ROW, never of which
    option is being scored) via one `logsumexp` over the joint `(c, k)` grid, length-
    normalised ONCE at the end. The resulting per-option score is then consumed by
    `mc_option_metrics` exactly as the single-sample scores are; only how the
    per-option log-likelihood is BUILT differs.

    Common-completion-budget check, applied PER OPTION across its 5
    class-encodings (an option's own completion text never changes with class -- only the
    PROMPT does, via the class-token suffix's length -- so a token-count mismatch across
    classes signals a censoring interaction at option granularity): an
    option failing it makes the WHOLE ROW unscoreable (dropping one option from a closed
    set changes what `correct_option_index`/`response_option_index` mean, so a row is
    scored in full or not at all -- `budget_consistent=False` tells the caller to exclude
    and count it, the eval bundle's own "exclude-and-report" convention, never a partial
    reconstruction).

    `max_batch_size` (default 16, chosen after an out-of-memory failure on a 24GB-class GPU
    -- see the batching comment inline): bounds how many `(k, option)` pairs enter any ONE forward call, since
    `logits_to_keep` restricts only the LM head's vocab projection, never the full
    transformer body's own batch x seq_len activation cost. Lower it (e.g. 8) on a smaller
    card; the mathematical result is identical at any value, only wall-clock/peak-memory
    change (chunking is exact concatenation, not an approximation).

    Returns `{"log_likelihoods": [n_options] length-normalised joint log p(opt) or None,
    "budget_consistent": bool, "n_valid_tok": [n_options] float or None}`. Raises
    `ValueError` on a degenerate option set (mirrors `mc_option_log_likelihoods`) or
    `AssertionError` if the class-phrase canary fails.
    """
    if u_k.dim() != 2:
        raise ValueError(f"mc_option_joint_log_likelihoods expects u_k as [K, u_dim], got {tuple(u_k.shape)}")
    if item_codec.dim() != 1:
        raise ValueError("mc_option_joint_log_likelihoods expects a single (codec_dim,) "
                         f"item_codec, got shape {tuple(item_codec.shape)}")
    if len(options) < 2 or any(not isinstance(o, str) or not o.strip() for o in options):
        raise ValueError("mc_option_joint_log_likelihoods: degenerate option set "
                         "(need >=2 non-empty options)")

    k_eff = u_k.shape[0]
    n_opt = len(options)

    examples_by_class: dict[int, list[dict]] = {}
    for c in range(N_OUTCOME):
        exs = [encode_example(question, opt, tok, max_len, outcome_code=c) for opt in options]
        for ex in exs:
            _assert_class_phrase_survives(ex, tok, c)
        examples_by_class[c] = exs

    counts = np.array([
        [sum(1 for t in examples_by_class[c][j]["labels"] if t != -100) for c in range(N_OUTCOME)]
        for j in range(n_opt)
    ])  # [n_opt, C]
    budget_consistent = bool((counts == counts[:, :1]).all())
    if not budget_consistent:
        return {"log_likelihoods": None, "budget_consistent": False, "n_valid_tok": None}

    n_valid_tok = torch.from_numpy(counts[:, 0]).float()  # [n_opt] -- identical across classes

    # One forward call per CLASS, K samples tiled into CHUNKED batches (was K separate
    # calls, one per k) -- on MPS/CPU backends the dominant cost is per-call dispatch
    # overhead, not FLOPs (measured: a 4-row/K=8 smoke pass at one-call-per-(c,k) took >25
    # minutes and was still not done), so batching K into fewer, larger calls is a real
    # wall-clock fix. `max_batch_size` BOUNDS how many (k, option) pairs go into one
    # forward call at once: tiling ALL of K into a single call runs out of memory on a
    # 24GB-class GPU at the widest option count (7), because the full
    # transformer body (all layers, full seq_len) sees every tiled row, not just the
    # `logits_to_keep`-restricted LM head -- `logits_to_keep` bounds the vocab-projection
    # term, never the batch-times-seq_len activation term that dominates memory.
    # Chunk size adapts to `n_opt` (`max(1, max_batch_size // n_opt)`) so a row with more
    # options gets proportionally fewer K samples per call, keeping every call's total
    # batch (`k_chunk * n_opt`) under the same bound regardless of the row's own shape.
    # Row order after tiling is BLOCKED by k (`[k=0: opt_0..opt_{n-1}, k=1: opt_0..opt_{n-1},
    # ...]`, via `.repeat(k_chunk, 1)`), matching `u_rep`'s own block order exactly, so
    # `.view(k_chunk, n_opt)` recovers the (k, option) grid correctly per chunk.
    k_chunk_size = max(1, max_batch_size // n_opt)
    log_p_seq_given_kc = torch.empty(N_OUTCOME, k_eff, n_opt)
    for c in range(N_OUTCOME):
        tok_batch = collate(examples_by_class[c], pad_id=tok.pad_token_id, device=device)
        for k_start in range(0, k_eff, k_chunk_size):
            k_end = min(k_start + k_chunk_size, k_eff)
            k_chunk = k_end - k_start
            input_ids_tiled = tok_batch["input_ids"].repeat(k_chunk, 1)
            attention_mask_tiled = tok_batch["attention_mask"].repeat(k_chunk, 1)
            prompt_end_index_tiled = tok_batch["prompt_end_index"].repeat(k_chunk)
            labels_tiled = tok_batch["labels"].repeat(k_chunk, 1)
            production_valid_tiled = tok_batch["production_valid"].repeat(k_chunk)
            u_rep = u_k[k_start:k_end].unsqueeze(1).expand(-1, n_opt, -1).reshape(
                k_chunk * n_opt, -1)  # [k_chunk*n_opt, u_dim]
            # Completion-span restriction (the same pattern as the CodeWorkout
            # generative simulator's scorer): every row's prompt
            # tokens carry label=-100 (masked out of the loss already), and here EVERY
            # option under class `c` shares the identical prompt (`build_reserved_prompt_ids`
            # depends only on question+class, never the answer/option text) -- so
            # restricting the LM-head's vocab projection to `[offset:]`, `offset =
            # prompt_end_index.min()`, is an IDENTITY for the loss, never an approximation.
            offset = int(prompt_end_index_tiled.min().item())
            seq_len = input_ids_tiled.shape[1]
            out = forward_personalised(model, hyper, context, u_rep, input_ids_tiled,
                                       attention_mask_tiled, prompt_end_index_tiled,
                                       logits_to_keep=seq_len - offset)
            log_p_seq, _n_tok, row_has_any = production_log_likelihood_sum_per_row(
                out["lm_logits"], labels_tiled[:, offset:], production_valid_tiled)
            if not bool(row_has_any.all()):
                raise ValueError("mc_option_joint_log_likelihoods: an option produced zero "
                                 f"valid completion tokens under class conditioning c={c}")
            log_p_seq_given_kc[c, k_start:k_end] = log_p_seq.detach().cpu().view(k_chunk, n_opt)
            del out, log_p_seq, row_has_any
            if device == "cuda":
                # PyTorch's allocator
                # can hold freed activations in its own cache across many small chunked
                # calls rather than releasing them, and this loop runs C*ceil(K/k_chunk)
                # times per row across 983 rows -- return memory between chunks rather
                # than let cache slack accumulate over the whole run.
                torch.cuda.empty_cache()

    logits_k = outcome_head(item_codec.unsqueeze(0).expand(k_eff, -1), u_k)  # [K, C] -- one call
    log_p_class_k = torch.log_softmax(logits_k, dim=-1).detach().cpu()  # [K, C]
    log_p_class_kc = log_p_class_k.transpose(0, 1).unsqueeze(-1).expand(-1, -1, n_opt)  # [C, K, n_opt]

    _mixture_nll_raw, mixture_nll_norm = joint_kc_mixture_nll(log_p_class_kc, log_p_seq_given_kc, n_valid_tok)
    return {
        "log_likelihoods": -mixture_nll_norm,  # [n_opt] -- mc_option_log_likelihoods's own sign convention
        "budget_consistent": True,
        "n_valid_tok": n_valid_tok.numpy(),
    }


def uniform_option_floor(n_options: Any) -> dict:
    """Canary: a metric with a mechanical null must never land BELOW that null. The null for
    closed-set option scoring is a UNIFORM random guesser: `mean_nll_floor = mean(log(n_i))`,
    `mean_top1_floor = mean(1/n_i)`, over the ACTUAL per-row option-count distribution --
    never a single fixed n, since FA's MC items range 2-7 options and a fixed-n floor would
    misstate the null for a mixed population."""
    n = np.asarray(n_options, dtype=float)
    if (n < 2).any():
        raise ValueError("uniform_option_floor: every row needs >=2 options")
    return {"mean_nll_floor": float(np.mean(np.log(n))), "mean_top1_floor": float(np.mean(1.0 / n))}


def assert_above_uniform_option_floor(top1_rate: float, n_options: Any) -> dict:
    """Halts (raises) rather than silently reporting a metric that scored WORSE than
    random guessing (an unconditioned-template scorer once landed at top-1 0.1251, below
    the floor for its own option-count distribution). Returns the floor dict on success so
    callers log it alongside the passing metric."""
    floor = uniform_option_floor(n_options)
    if top1_rate < floor["mean_top1_floor"]:
        raise AssertionError(
            f"CANARY FAILED (uniform floor): top1_rate={top1_rate:.4f} is BELOW the uniform-"
            f"option floor={floor['mean_top1_floor']:.4f} for this population's option-count "
            "distribution -- halting rather than reporting; audit the instrument before the "
            "model")
    return floor


def mc_option_metrics(log_likelihoods: Any, correct_index: int) -> dict[str, Any]:
    """Pure: turn one item's per-option log-likelihoods into (a) the option-normalised NLL
    of the CORRECT option under a softmax over all options (the primary metric -- "option-
    normalised" means the model's mass is compared only against the other options actually
    offered, not the full vocabulary) and (b) whether the argmax (top-1 prediction) is the
    correct option."""
    ll = np.asarray(log_likelihoods, dtype=float) if not isinstance(log_likelihoods, torch.Tensor) \
        else log_likelihoods.detach().cpu().numpy()
    log_probs = ll - _logsumexp(ll)
    nll = float(-log_probs[correct_index])
    top1 = bool(int(np.argmax(ll)) == correct_index)
    return {"nll": nll, "top1_correct": top1, "n_options": int(len(ll))}


def _logsumexp(x: np.ndarray) -> float:
    m = x.max()
    return float(m + np.log(np.exp(x - m).sum()))


def paired_diff_ci(metric_a: Any, metric_b: Any, n_boot: int = 2000, alpha: float = 0.05,
                   seed: int = 0) -> dict[str, float]:
    """The bread test's gate requirement: a paired CI excluding zero. `metric_a`/`metric_b` are PER-EXAMPLE arrays
    (e.g. personalised NLL vs a control's NLL, same items, same order) -- this is a thin,
    documented wrapper around `atom.eval.metrics.bootstrap_mean_ci` applied to the
    per-example DIFFERENCE array, because a CI on each arm's own mean separately is not
    the same statement as a CI on their paired difference (variance of a difference of
    correlated arms is not the sum of each arm's own variance), and it is easy to
    mis-apply the existing primitive to two separate arrays by mistake."""
    a = np.asarray(metric_a, dtype=float)
    b = np.asarray(metric_b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"paired_diff_ci requires matched-shape arrays; got {a.shape} vs {b.shape}")
    diff = a - b  # positive => a (typically the CONTROL, e.g. frozen backbone) scores
                  # worse than b (the personalised model) when lower-is-better metrics
                  # (NLL) are passed as (control, personalised) -- caller states the sign
                  # convention explicitly when reading the result, this function is symmetric.
    mean, lo, hi = bootstrap_mean_ci(diff, n_boot=n_boot, alpha=alpha, seed=seed)
    return {"mean_diff": mean, "ci_lo": lo, "ci_hi": hi, "excludes_zero": bool(lo > 0 or hi < 0)}
