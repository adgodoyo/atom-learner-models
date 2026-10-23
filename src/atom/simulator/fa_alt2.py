"""Forward orchestration of the FA simulator (thesis Sections 5.4-5.6), plus its prompt
encoding and losses.

Wires independent pieces -- `S1FilterV2`'s query-independent belief, `atom.binding`'s
hypernetwork/LoRA-basis/K-sample machinery, the frozen backbone's own LM head -- into one
path:

    filter(H_<t) -> (mu_t, log_sigma_t) -> reparameterise -> u
    -> hypernetwork(u) -> c -> personalised backbone forward on q_t
    -> h_q (prompt-boundary hidden state) -> {support head, success head}
    -> backbone's own vocabulary head -> production distribution

None of the wired pieces change; this module only connects them and adds the prompt/
label bookkeeping the three losses need. Deliberately backbone-agnostic: every
function here relies only on the `model(input_ids=, attention_mask=,
output_hidden_states=True) -> (.logits, .hidden_states)` contract every causal LM (HF or
the test-only `CausalStubLM`) satisfies -- the fast pytest canaries run against the stub;
the same functions are run once against the real gemma4-e4b to check structure.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from atom.binding import reparameterize_from_log_sigma
from atom.data.fa_outcome_label import OUTCOME_CLASS_TOKENS

PROMPT_TEMPLATE = "Question:\n{question}\n\nThis learner's response:"

# Conditioning route: the class enters as a FROZEN TEXT TOKEN in the
# prompt, so it is masked from the loss by the exact same mechanism that already masks
# every other prompt token -- no new masking logic, no new machinery to re-verify.
CLASS_CONDITIONED_PROMPT_TEMPLATE = (
    "Question:\n{question}\n\nThis learner's response ({class_token}):"
)


def build_prompt_completion(question: str, answer: str,
                            outcome_code: int | None = None) -> tuple[str, str]:
    """One frozen, disclosed prompt template. No learner ID,
    skill tags, reference answer, observed platform success, or future events; the
    wording does not instruct the model to solve correctly.

    `outcome_code`: production conditioning. `None` (default) keeps the unconditioned
    template, used by callers that predate class conditioning (the dual-forward test,
    the pilot/stage-2 trainers). An int in
    `range(N_OUTCOME)` switches to `CLASS_CONDITIONED_PROMPT_TEMPLATE`, which names the
    OBSERVED support/outcome class -- teacher-forced at train (the true class),
    ancestrally sampled at simulation, and swept over every class for the
    class-marginalised eval mixture (never the true class alone, which would leak the
    label into the score)."""
    if outcome_code is None:
        return PROMPT_TEMPLATE.format(question=question), f" {answer}"
    prompt = CLASS_CONDITIONED_PROMPT_TEMPLATE.format(
        question=question, class_token=OUTCOME_CLASS_TOKENS[outcome_code],
    )
    return prompt, f" {answer}"


def _prompt_prefix_suffix(outcome_code: int | None) -> tuple[str, str]:
    """Splits the (possibly class-conditioned) prompt template around `{question}` into
    its immutable prefix and suffix. The suffix carries the class phrase, the response
    delimiter -- everything after the question -- and it is exactly the piece
    `build_reserved_prompt_ids` guarantees survives truncation."""
    template = PROMPT_TEMPLATE if outcome_code is None else CLASS_CONDITIONED_PROMPT_TEMPLATE
    prefix, _, suffix_template = template.partition("{question}")
    suffix = suffix_template.format(class_token=(
        OUTCOME_CLASS_TOKENS[outcome_code] if outcome_code is not None else None))
    return prefix, suffix


def build_reserved_prompt_ids(question: str, tok, max_len: int,
                              min_completion_tokens: int = 8,
                              outcome_code: int | None = None) -> tuple[list[int], bool]:
    """Class-phrase-preserving prompt tokenisation. Tokenising once and slicing the tail
    truncates the prompt's TAIL, where the class phrase lives -- measured exactly, all rows,
    real tokenizer, max_len=256: 204/3,398 val (6.00%) and 3,851/59,798 train (6.44%) would
    lose it. Instead: tokenise the immutable
    prefix/suffix and the question as SEPARATE segments and concatenate, so only the
    question body can be truncated -- the class phrase, delimiter and prefix are always
    present in full, by construction, not by a length check after the fact.

    Returns `(prompt_ids, question_truncated)`. `question_truncated` is True exactly when
    this row's question would have overrun the reserved budget -- i.e. exactly the rows
    a tail-slicing encoder would have clipped the class phrase on. It is a property of the
    ROW (question length vs budget), not of what actually got truncated (which is now
    always the question, never the class phrase) -- kept under this name because it is
    the stratification variable for truncated rows, not a claim that the class
    phrase was ever cut.
    """
    prefix, suffix = _prompt_prefix_suffix(outcome_code)
    prefix_ids = tok(prefix, add_special_tokens=True)["input_ids"]
    suffix_ids = tok(suffix, add_special_tokens=False)["input_ids"]
    question_ids = tok(question, add_special_tokens=False)["input_ids"]

    max_prompt_len = max(1, max_len - min_completion_tokens)
    reserved = len(prefix_ids) + len(suffix_ids)
    question_budget = max(0, max_prompt_len - reserved)
    question_truncated = len(question_ids) > question_budget
    question_ids = question_ids[:question_budget]

    return prefix_ids + question_ids + suffix_ids, question_truncated


def encode_example(question: str, answer: str, tok, max_len: int,
                    min_completion_tokens: int = 8,
                    outcome_code: int | None = None) -> dict:
    """Tokenise one (question, answer) pair.

    Returns `input_ids`/`labels` (prompt tokens masked -100) and
    `prompt_end_index` -- prompt-boundary pooling: the LAST PROMPT token's
    position, where `h_q` is gathered for the support/success heads. Never the final
    non-padding token, which would be an observed answer token during training.

    `outcome_code`: forwarded to `build_reserved_prompt_ids` unchanged. Because
    the class token lives INSIDE the prompt string, it is masked from `L_production`
    by the `labels = [-100]*len(p_ids) + c_ids` line below -- no separate
    masking path for it exists, so there is nothing new to peek-canary on the loss
    side; the canary that matters is that `OutcomeHead` (which reads only
    `(item_codec, u)`, never `input_ids`) cannot see this token either
    (`test_outcome_head_is_blind_to_the_production_class_token`).

    Truncation (per-type/per-outcome truncation-rate reporting is not done here): only
    the QUESTION may be truncated (`build_reserved_prompt_ids`) -- the class phrase and delimiter are reserved in full, and
    `min_completion_tokens` still reserves room so a long question cannot silently
    swallow the entire completion. `production_valid=False` when truncation leaves zero
    *content* tokens (see the EOS note below for what "content" excludes);
    support/success remain trainable regardless (the masks are independent).

    EOS and censoring: `tok.eos_token_id` is appended **only
    when the full completion fits** in the available room (`len(full_content_ids) + 1
    <= room`). A row whose completion does NOT fit at that length is `completion_censored
    =True`: its `c_ids` is the truncated content with **no** EOS appended -- appending EOS to a
    truncated completion (`content_ids[:room-1] + [eos]`) would teach the model to stop at
    an arbitrary length cutoff rather than at the answer's actual end. The canary is
    two-directional: *every uncensored row's labels end in EOS, and no censored row's labels
    do*. Measured on FA-SIM3 data: 4.01% train / 4.30% eval rows censor (answer p95 = 23
    tokens against a ~7-8 token minimum content room); the median alone would suggest
    censoring is immaterial on FA, but the tail is not. `production_valid` is driven by
    *content* tokens actually used regardless of branch (a genuinely empty answer is
    still `production_valid=False`, uncensored, EOS-only).
    """
    p_ids, class_token_truncated = build_reserved_prompt_ids(
        question, tok, max_len, min_completion_tokens, outcome_code=outcome_code)
    prompt_end_index = len(p_ids) - 1

    completion = f" {answer}"
    full_content_ids = tok(completion, add_special_tokens=False)["input_ids"]
    room = max(0, max_len - len(p_ids))
    completion_censored = (len(full_content_ids) + 1) > room
    if completion_censored:
        c_ids = full_content_ids[:room]
        n_content_used = len(c_ids)
    else:
        c_ids = full_content_ids + [tok.eos_token_id]
        n_content_used = len(full_content_ids)

    ids = p_ids + c_ids
    labels = [-100] * len(p_ids) + c_ids
    return {
        "input_ids": ids,
        "labels": labels,
        "completion_censored": completion_censored,
        "prompt_end_index": prompt_end_index,
        "production_valid": n_content_used > 0,
        "class_token_truncated": class_token_truncated,
    }


def collate(examples: list[dict], pad_id: int, device: str = "cpu") -> dict:
    """Right-pad a list of `encode_example` outputs to the batch's own max length.
    `prompt_end_index` needs no adjustment under right-padding (it indexes from the
    start, unaffected by trailing pad tokens) -- matches the other padded-
    batch convention (`src/atom/filter/batch.py`).

    `completion_censored` rides along per row so a caller
    can exclude censored rows from the production loss/metrics or apply a data-derived
    completion budget -- `production_valid` alone does not carry this
    distinction (a censored row can still be `production_valid=True`, since truncated
    content is still content)."""
    seq_len = max(len(e["input_ids"]) for e in examples)
    n = len(examples)
    input_ids = torch.full((n, seq_len), pad_id, dtype=torch.long)
    labels = torch.full((n, seq_len), -100, dtype=torch.long)
    attention_mask = torch.zeros((n, seq_len), dtype=torch.long)
    prompt_end_index = torch.zeros(n, dtype=torch.long)
    production_valid = torch.zeros(n, dtype=torch.bool)
    completion_censored = torch.zeros(n, dtype=torch.bool)
    for i, e in enumerate(examples):
        length = len(e["input_ids"])
        input_ids[i, :length] = torch.tensor(e["input_ids"], dtype=torch.long)
        labels[i, :length] = torch.tensor(e["labels"], dtype=torch.long)
        attention_mask[i, :length] = 1
        prompt_end_index[i] = e["prompt_end_index"]
        production_valid[i] = e["production_valid"]
        completion_censored[i] = e["completion_censored"]
    return {
        "input_ids": input_ids.to(device),
        "attention_mask": attention_mask.to(device),
        "labels": labels.to(device),
        "prompt_end_index": prompt_end_index.to(device),
        "production_valid": production_valid.to(device),
        "completion_censored": completion_censored.to(device),
    }


def gather_prompt_hidden(hidden_states: torch.Tensor, prompt_end_index: torch.Tensor) -> torch.Tensor:
    """hidden_states: [N, T, H] (the model's FINAL layer hidden state -- the h_q
    source). prompt_end_index: [N]. Returns [N, H]."""
    n = hidden_states.shape[0]
    return hidden_states[torch.arange(n, device=hidden_states.device), prompt_end_index]


def gather_belief(mu_all: torch.Tensor, log_sigma_all: torch.Tensor,
                   learner_idx: torch.Tensor, t_idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Differentiable gather of the query-independent belief at each selected (learner, t)
    query row out of a full padded-history filter forward -- `mu_all`/`log_sigma_all`:
    [N_learners, T, latent_dim]. Unlike `scripts/p5_01_prepare_beliefs.py` (which read
    this out under `torch.no_grad()` from a FROZEN checkpoint), this keeps the filter in
    the live autograd graph: the filter and hypernetwork are trained jointly, as one
    continuing checkpoint, not a filter frozen upstream of the simulator."""
    return mu_all[learner_idx, t_idx], log_sigma_all[learner_idx, t_idx]


def forward_personalised(model, hyper, context, u: torch.Tensor,
                          input_ids: torch.Tensor, attention_mask: torch.Tensor,
                          prompt_end_index: torch.Tensor, logits_to_keep: int = 0) -> dict:
    """One personalised forward pass for a batch that already shares ONE mixing vector
    per row (`u`: [N, u_dim] -- a single K-sample; callers loop over K themselves,
    matching every other K-sample site in this codebase, e.g.
    `infra/p5_01_pilot_train.py`).

    Returns `lm_logits` [N,T,V] (the production head -- the base LLM's own
    vocabulary projection, unmodified) and `h_q` [N,H] (prompt-boundary
    pooling). Does NOT run the support/success heads -- callers own that, since some
    canaries need `h_q` reused across multiple head calls (e.g. peek canary 1's "same
    prompt, different response tokens").

    `logits_to_keep`: 0 (default) computes logits for the FULL sequence -- HF's own
    convention (`slice(-0, None) == slice(0, None)`), used by every FA site. A caller that
    only needs the completion span (the CodeWorkout production forward, whose long code/problem-statement prompts made gemma4-e4b's
    full-sequence, full-vocab logits tensor the dominant OOM term, not parameter count)
    passes `seq_len - offset` -- the trunk (`self.model`, inside the HF forward) still
    computes hidden states for EVERY position regardless of this argument, so `h_q`'s
    prompt-boundary pooling (which can land anywhere before the completion starts) is
    unaffected; only the vocab-sized `lm_head` projection is restricted, verified
    against `transformers.models.gemma4.modeling_gemma4.Gemma4ForCausalLM.forward`'s own
    source (`self.lm_head(hidden_states[:, slice_indices, :])`) and mirrored in the
    smoke-test stub (`atom.binding.stub_causal_lm.CausalStubLM`) so the identity check
    runs locally without the real backbone.
    """
    c = hyper(u)
    context.set(c)
    out = model(input_ids=input_ids, attention_mask=attention_mask,
               output_hidden_states=True, logits_to_keep=logits_to_keep)
    h_q = gather_prompt_hidden(out.hidden_states[-1], prompt_end_index)
    # support_head/success_head are fp32 (trained, not frozen); a bf16 backbone (e.g. the
    # real gemma4-e4b) would otherwise hand them a bf16 h_q and crash inside LayerNorm.
    return {"lm_logits": out.logits, "h_q": h_q.float()}


def forward_global_control(model, global_mixing, context, n: int,
                           input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """The global-LoRA control arm (the bread test's "personalised vs global vs ΔW=0"
    comparison, `atom.binding.
    GlobalMixingVector` + `atom.binding.build_second_basis`/`set_active_basis`): the SAME
    hooked backbone, but `c` comes from ONE shared learned vector broadcast to every row
    -- never from `hyper(u)`.

    Returns only `lm_logits` (production), never `h_q`: the global arm has no `u` to feed
    an outcome head with, and `OutcomeHead` reads `(item_codec, u)` directly regardless of
    which LoRA basis is currently active on the backbone (`set_active_basis` only changes
    what the backbone's own forward does) -- so the global-vs-personalised comparison this
    arm exists for is a PRODUCTION-only comparison by construction, matching the bread
    test's own framing. Caller is responsible for having already called `set_active_basis`
    to point the hooked layers at the global-control basis before this runs (this function
    only sets `context.c`, never which basis the hooked layers read from -- same division
    of labour as `forward_personalised`, which sets `c` but never touches the basis
    dict either).
    """
    context.set(global_mixing(n))
    out = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=False)
    return out.logits


def support_loss(support_logits: torch.Tensor, support_target: torch.Tensor,
                  support_valid: torch.Tensor) -> torch.Tensor:
    """L_S -- cross-entropy over `support_valid` rows only. Exact zero
    (not a 0/0 NaN) when no row in the batch is valid, matching this codebase's other
    all-masked-batch guards (e.g. `infra/p5_01_pilot_train.py::nll_loss`)."""
    if not bool(support_valid.any()):
        return support_logits.sum() * 0.0
    per = F.cross_entropy(support_logits, support_target.long(), reduction="none")
    valid = support_valid.float()
    return (per * valid).sum() / valid.sum()


def outcome_loss(outcome_logits: torch.Tensor, outcome_target: torch.Tensor,
                 outcome_valid: torch.Tensor) -> torch.Tensor:
    """L_outcome -- masked 5-class cross-entropy, replacing BOTH `support_loss` and
    `success_loss` in the hierarchical training path (they stay defined, for
    reproducibility, and are not called by the hierarchical trainer). Same masked-CE
    pattern as `support_loss` (exact zero, not 0/0 NaN, on an all-invalid batch) --
    kept as its own named function rather than a call-site alias so the loss
    combination reads as `L_outcome + lambda_A * L_production + beta * KL`."""
    if not bool(outcome_valid.any()):
        return outcome_logits.sum() * 0.0
    per = F.cross_entropy(outcome_logits, outcome_target.long(), reduction="none")
    valid = outcome_valid.float()
    return (per * valid).sum() / valid.sum()


def outcome_kmixture_nll_per_row(outcome_head, item_codec: torch.Tensor, u_k: torch.Tensor,
                                 outcome_target: torch.Tensor,
                                 outcome_head_kwargs: dict | None = None) -> torch.Tensor:
    """The K-MIXTURE outcome
    likelihood, `-log mean_k p(y | mu + sigma*eps_k)`, a PROBABILITY-space average over K
    samples of the SAME continuous belief for an OBSERVED class (never `joint_kc_mixture_nll`'s
    logsumexp-over-log-probs, which marginalises a discrete class variable jointly with K --
    here the class is the label, not a hypothesis being summed over). Why: the single-sample loss's gradient on sigma is (head curvature x
    noise), tiny once the head is insensitive to the noise, and Jensen-biased toward
    shrinking sigma regardless of how much evidence exists. Scoring the K-sample MIXTURE
    instead rewards a wide sigma exactly where the K draws disagree on the outcome (little
    evidence) and lets it shrink where they agree (much evidence) -- evidence-tracking width
    by construction of the objective, not by a separate regulariser.

    `item_codec`: `[N, codec_dim]`. `u_k`: `[N, K, latent_dim]` -- K SAMPLES for the SAME
    row; sampling is the CALLER's job (`reparameterize_from_log_sigma(mu, log_sigma, k=K)`),
    matching this module's backbone-agnostic, already-computed-tensors convention (this
    function never samples). `outcome_target`: `[N]` int64. Returns per-row NLL `[N]`, never
    batch-reduced -- caller masks/means, matching `outcome_loss`'s own convention.

    `outcome_head_kwargs`: passthrough for heads that need more than
    `(item_codec, u)` -- CW's `OutcomeHeadItemMLP` raises if constructed with `prefix_dim`
    set and called without `prefix=`. Default `None` -> `{}`, as FA callers use."""
    k_eff = u_k.shape[1]
    head_kwargs = outcome_head_kwargs or {}
    probs_sum = None
    for j in range(k_eff):
        logits = outcome_head(item_codec, u_k[:, j, :], **head_kwargs)
        p = torch.softmax(logits, dim=-1)
        probs_sum = p if probs_sum is None else probs_sum + p
    probs_mean = probs_sum / k_eff
    p_true = probs_mean.gather(1, outcome_target.long().unsqueeze(1)).squeeze(1).clamp(min=1e-8)
    return -torch.log(p_true)


def outcome_loss_likelihood(outcome_head, item_codec: torch.Tensor, mu: torch.Tensor,
                            log_sigma: torch.Tensor, outcome_target: torch.Tensor,
                            outcome_valid: torch.Tensor, likelihood: str = "single",
                            k: int = 1, generator: torch.Generator | None = None,
                            eps: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """The outcome-likelihood switch every outcome-head trainer (FA or CW) shares.
    `likelihood='single'` (default) is the single-sample-averaged loss (`u =
    reparameterize_from_log_sigma(mu, log_sigma, k=1)[:, 0, :]` then masked cross-entropy --
    exactly `outcome_loss`'s own computation, routed through this one call site so both
    branches share one masking convention rather than drifting apart). `likelihood='kmixture'`
    scores the K-sample mixture instead (`outcome_kmixture_nll_per_row`; the pre-registered
    spec's `K_train >= 8` is the CALLER's contract, not enforced here -- this stays a general
    K>=1 utility, matching `reparameterize_from_log_sigma`'s own convention of never
    special-casing K). Returns `(masked_mean_loss, u)`: `u` is `[N, u_dim]`, ONE representative
    sample (`u_k[:, 0, :]` under kmixture) for a caller that also needs a single `u` downstream
    (e.g. the production head in a WITH-backbone trainer) -- the K-mixture changes only the
    outcome likelihood, never the production path, so production always sees one sample either
    way."""
    if likelihood not in ("single", "kmixture"):
        raise ValueError(f"outcome_loss_likelihood: unknown likelihood {likelihood!r}, "
                         "expected 'single' or 'kmixture'")
    k_eff = 1 if likelihood == "single" else k
    u_k = reparameterize_from_log_sigma(mu, log_sigma, k=k_eff, generator=generator, eps=eps)
    u = u_k[:, 0, :]
    if likelihood == "single":
        logits = outcome_head(item_codec, u)
        per_row = F.cross_entropy(logits, outcome_target.long(), reduction="none")
    else:
        per_row = outcome_kmixture_nll_per_row(outcome_head, item_codec, u_k, outcome_target)
    if not bool(outcome_valid.any()):
        return per_row.sum() * 0.0, u
    valid = outcome_valid.float()
    return (per_row * valid).sum() / valid.sum(), u


def success_loss(success_logits: torch.Tensor, support_target: torch.Tensor,
                  success_target: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """L_Y -- BCE against the logit SELECTED by the observed support class.
    `support_target` is used only to select which of the 4 logits is supervised; it is
    never fed forward into the model (peek canary 2). `valid` must already be the
    caller's `success_valid AND support_valid` (a
    success label conditioned on an unknown support class is not a well-posed target).

    Missing success contributes exactly zero via EXCLUSION
    from the masked mean -- never by defaulting the target to a value, which is what
    "not a negative label" rules out.
    """
    if not bool(valid.any()):
        return success_logits.sum() * 0.0
    selected = success_logits.gather(1, support_target.long().unsqueeze(1)).squeeze(1)
    per = F.binary_cross_entropy_with_logits(selected, success_target.float(), reduction="none")
    v = valid.float()
    return (per * v).sum() / v.sum()


def pick_lambda_a(grad_norm_a: float, grad_norm_other: float,
                   grid: tuple[float, ...] = (0.1, 0.3, 1.0)) -> float:
    """The pre-registered rule: select `lambda_A` from a fixed grid by
    GRADIENT-NORM BALANCE at the shared LoRA basis, never by validation performance.
    `grad_norm_a`/`grad_norm_other` are the L2 norms of dL_A/d(basis) and
    d(L_S+L_Y)/d(basis) respectively, measured separately (two backward passes on the SAME
    fixed batch) so this function only does the selection, not the measurement -- picks
    whichever grid value makes `lambda_A * grad_norm_a` closest to `grad_norm_other`
    (production dominating neither vanishes into nor overwhelms the other two losses'
    combined pull on the basis)."""
    if grad_norm_a <= 0.0:
        return grid[0]
    return min(grid, key=lambda lam: abs(lam * grad_norm_a - grad_norm_other))


def production_nll_per_row(lm_logits: torch.Tensor, labels: torch.Tensor,
                           production_valid: torch.Tensor,
                           row_chunk_size: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """The row-wise term `production_loss` reduces over -- factored out (the bread
    test's per-option scoring needs a length-normalised NLL PER example/
    option, never batch-reduced) so both callers share one masking implementation instead
    of two that could silently drift apart. Returns `(per_row_nll, row_has_any)`:
    `per_row_nll[i]` is response i's average token NLL over its own valid completion
    tokens (0.0 where the row has none -- caller decides whether/how to use
    `row_has_any` to exclude it, matching `production_loss`'s own `row_has_any` masking).

    `row_chunk_size` (the training-path counterpart of
    `production_log_likelihood_sum_per_row`'s eval-path option): when set
    and `< batch`, `F.cross_entropy` runs on row-sub-chunks instead of the whole batch at
    once. `F.cross_entropy(..., reduction="none")` is an elementwise-per-position op, so
    chunking the FORWARD computation this way is a pure peak-memory optimisation with no
    effect on the result or on gradient flow -- the caller's single later `.sum()` /
    `.backward()` over the concatenated output sees the identical autograd graph it would
    have seen unchunked, just built from smaller intermediate tensors (bit-identical,
    verified by `tests/test_cw_sim_chunked_nll.py`'s established pattern for the eval-path
    version; `gemma4-e4b`'s huge vocab makes full-sequence logits the dominant memory
    term). Default `None` runs one unchunked call."""
    shift_logits = lm_logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous().clone()
    shift_labels[~production_valid] = -100

    valid_token = shift_labels != -100
    b = shift_logits.size(0)
    if row_chunk_size is not None and row_chunk_size < b:
        per_token_parts = []
        for start in range(0, b, row_chunk_size):
            end = min(start + row_chunk_size, b)
            chunk_logits = shift_logits[start:end].contiguous()
            chunk_labels = shift_labels[start:end].contiguous()
            per_token_parts.append(F.cross_entropy(
                chunk_logits.reshape(-1, chunk_logits.size(-1)), chunk_labels.reshape(-1),
                ignore_index=-100, reduction="none",
            ).view(chunk_labels.shape))
        per_token = torch.cat(per_token_parts, dim=0)
    else:
        per_token = F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1),
            ignore_index=-100, reduction="none",
        ).view(shift_labels.shape)
    n_valid_tok = valid_token.sum(dim=1).clamp(min=1)
    # .float(): cross_entropy on a bf16 backbone's logits (e.g. the real gemma4-e4b)
    # returns a bf16 per_row tensor, which torch.Tensor.numpy() cannot convert (numpy
    # has no native bf16) -- such a crash would surface only after a full GPU pass.
    # float32 is also
    # simply the right precision for a value downstream callers feed into means/bootstrap
    # CIs, independent of the numpy issue.
    per_row = ((per_token * valid_token).sum(dim=1) / n_valid_tok).float()
    row_has_any = valid_token.any(dim=1).float()
    return per_row, row_has_any


def production_topk_per_row(lm_logits: torch.Tensor, labels: torch.Tensor,
                            production_valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Teacher-forced per-row top-1 accuracy and exact-match, same shift/mask convention
    as `production_nll_per_row` (NLL alone can be moved by formatting, so
    the wrong-answer production test reports these alongside it). TEACHER-FORCED, not
    free-running generation -- greedy argmax under the true prefix at every position, the
    same convention `production_loss` itself scores under. `exact_match[i]=1` iff every
    valid completion token's argmax equals the true token; `top1_frac[i]` is the fraction
    that do. Returns `(top1_frac, exact_match, row_has_any)`, `row_has_any` shared with
    `production_nll_per_row` so a caller can build one mask for both."""
    shift_logits = lm_logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous().clone()
    shift_labels[~production_valid] = -100

    valid_token = shift_labels != -100
    pred = shift_logits.argmax(dim=-1)
    correct = (pred == shift_labels) & valid_token
    n_valid_tok = valid_token.sum(dim=1).clamp(min=1)
    top1_frac = correct.sum(dim=1).float() / n_valid_tok.float()
    row_has_any = valid_token.any(dim=1)
    exact_match = ((correct.sum(dim=1) == valid_token.sum(dim=1)) & row_has_any).float()
    return top1_frac, exact_match, row_has_any.float()


def production_log_likelihood_sum_per_row(lm_logits: torch.Tensor, labels: torch.Tensor,
                                          production_valid: torch.Tensor,
                                          row_chunk_size: int | None = None,
                                          ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sibling of `production_nll_per_row`, returning the RAW SUM sequence log-likelihood
    -- never divided by token count -- plus the token count itself. A mixture over hypotheses (classes, or any other discrete latent) combines actual
    SEQUENCE probabilities, `log p(y) = sum_t log p(y_t | y_<t)`; exponentiating
    `production_nll_per_row`'s MEAN-token NLL as though it were that quantity silently
    discards the length term. Returns `(log_p_seq, n_valid_tok,
    row_has_any)`, all `[B]`.

    `row_chunk_size` (memory headroom): `F.cross_entropy` internally upcasts a bf16/fp16 `lm_logits` to fp32 for
    the softmax normalisation and `shift_logits.contiguous()` above it makes its own
    same-sized copy -- at gemma4-e4b's vocab (262,144) and eval batch sizes of 8-16 this
    pipeline alone can reach tens of GB (logits tensors of 13 GiB and 43.5 GB have run out
    of memory on CodeWorkout prompts). Cross-entropy with `reduction="none"` is computed
    INDEPENDENTLY per flattened (row, token) position -- a row's own loss never depends on
    any other row in the batch -- so chunking along the ROW axis and concatenating is
    mathematically a no-op: this only bounds PEAK memory to `row_chunk_size` rows' worth of
    the logits/upcast tensors at a time, it never changes a single number. Default `None`
    runs a single call; a caller
    scoring the fixed-cost eval path passes a small chunk (e.g. 4) to unlock a larger outer
    batch size without growing peak memory."""
    shift_logits = lm_logits[:, :-1, :]
    shift_labels = labels[:, 1:].contiguous().clone()
    shift_labels[~production_valid] = -100

    valid_token = shift_labels != -100
    b = shift_logits.shape[0]
    if row_chunk_size is None or row_chunk_size >= b:
        per_token = F.cross_entropy(
            shift_logits.contiguous().reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1),
            ignore_index=-100, reduction="none",
        ).view(shift_labels.shape)
    else:
        per_token_chunks = []
        for start in range(0, b, row_chunk_size):
            end = min(start + row_chunk_size, b)
            logits_c = shift_logits[start:end].contiguous()
            labels_c = shift_labels[start:end]
            per_token_c = F.cross_entropy(
                logits_c.reshape(-1, logits_c.size(-1)), labels_c.reshape(-1),
                ignore_index=-100, reduction="none",
            ).view(labels_c.shape)
            per_token_chunks.append(per_token_c)
            del logits_c
        per_token = torch.cat(per_token_chunks, dim=0)
    n_valid_tok = valid_token.sum(dim=1)
    log_p_seq = -(per_token * valid_token).sum(dim=1).float()
    row_has_any = valid_token.any(dim=1).float()
    return log_p_seq, n_valid_tok, row_has_any


def joint_kc_mixture_nll(log_p_class_kc: torch.Tensor, log_p_seq_given_kc: torch.Tensor,
                         n_valid_tok: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """THE (k, c) MIXTURE ESTIMATOR. It replaces a K-factorised formula (the one in
    `infra/fa_sim2_full_run.py`) that had two faults: (a) class
    probability and token likelihood were each averaged over K SEPARATELY
    (`logsumexp(log_p_tok_given_c_k, dim=1)` then, independently, `p_class.mean(dim=1)`),
    discarding the pairing between a specific sampled `u_k` and its own production
    forward -- a proper Monte-Carlo mixture must combine both factors WITHIN the same k
    before summing; (b) the token-likelihood term was `production_nll_per_row`'s MEAN
    per-token NLL negated and exponentiated, not a true sequence log-likelihood (see
    `production_log_likelihood_sum_per_row`'s docstring).

    `log_p_class_kc` / `log_p_seq_given_kc`: `[C, K, B]`, `log p(c|u_k,q)` and the RAW
    SUM `log p(y|c,u_k,q)` (never a per-token mean) respectively, for the SAME `u_k` at
    matching `(c, k)` indices. `n_valid_tok`: `[B]`, the row's completion token count --
    assumed identical across every class for a row passed in here (the
    common-completion-budget filter is what guarantees this upstream; this function
    does not itself check it, and a caller with a row that fails that check must exclude
    it before calling, not pass mismatched-length classes through).

    One `logsumexp` over the JOINT `(c, k)` grid: `p(y) = (1/K) sum_k sum_c p(c|u_k)
    p(y|c,u_k)`, i.e. `log p(y) = logsumexp_{c,k}[log p(c|u_k) + log p(y|c,u_k)] - log K`.
    The class dimension is a genuine probability SUM (no `-log C` correction, `c` is a
    real marginalised random variable); only the `k` dimension is a Monte-Carlo average
    (the `- log K` term) over samples of the SAME continuous `u`. `mixture_nll = -log
    p(y)` is thus the raw SEQUENCE nll; `mixture_nll / n_valid_tok` is the LENGTH-
    NORMALISED version, computed ONCE, at the end, from the already-combined joint
    quantity -- never per-(c,k) before combining.

    Returns `(mixture_nll, mixture_nll_length_normalized)`, both `[B]`."""
    c_dim, k_dim, b_dim = log_p_class_kc.shape
    joint = log_p_class_kc + log_p_seq_given_kc  # [C, K, B]
    mixture_log_p = torch.logsumexp(joint.reshape(c_dim * k_dim, b_dim), dim=0) - float(np.log(k_dim))
    mixture_nll = -mixture_log_p
    mixture_nll_length_normalized = mixture_nll / n_valid_tok.clamp(min=1).float()
    return mixture_nll, mixture_nll_length_normalized


def production_loss(lm_logits: torch.Tensor, labels: torch.Tensor,
                     production_valid: torch.Tensor) -> torch.Tensor:
    """L_A -- average token loss WITHIN each response first, then average
    across responses (so a long answer does not count as many independent examples),
    `production_valid` rows only.

    Masking is enforced explicitly here (`encode_example` can leave an EOS label on an
    invalid, empty-answer row, so the labels alone are not enough: `production_valid` is
    authoritative regardless of what a caller's labels tensor happens to contain, matching this codebase's other "flag, don't trust silent
    structure" discipline, e.g. `filter/batch.py`'s `label_valid`)."""
    per_row, row_has_any = production_nll_per_row(lm_logits, labels, production_valid)
    if not bool(row_has_any.any()):
        return lm_logits.sum() * 0.0
    return (per_row * row_has_any).sum() / row_has_any.sum()
