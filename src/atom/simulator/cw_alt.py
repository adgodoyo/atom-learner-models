"""Segment-reserved prompt encoder for the CodeWorkout code-production head.

Ports the prompt construction of `fa_alt2.py` (segment-reserved tokenisation, so that
truncation can never touch a reserved segment; EOS appended only to uncensored
completions) onto the CodeWorkout prompt shape.

**CW's segments differ from FA's in one structural way FA never had to solve: CW has
TWO variable-length (truncatable) segments, not one** -- the problem STATEMENT and the
episode's PRIOR-ATTEMPT CODE (the immediately preceding attempt in the same episode, or
an explicit START segment at an episode's first attempt). A measurement on the real
corpus (`cw_5_b1_prompt_length_measurement.py`) found the prior-attempt code to be the
bigger segment at every candidate `max_len`, and set the priority this module encodes
as fixed policy, not a runtime choice: **the STATEMENT is truncated
first** (down to zero if it must); the prior-attempt code is preserved whole whenever the
budget allows, and only shrinks once the statement is already exhausted. Whichever
segment IS truncated keeps its HEAD (matches FA's own `question_ids[:budget]` convention,
and is the more information-dense end for both a problem statement's opening sentence and
a submission's method signature / opening logic).

**ONE TARGET ROW PER ATTEMPT**: the completion is THIS attempt's own submitted code,
never the episode's eventual terminal code. This matches the per-attempt training unit
of the hierarchical attempt-level model (`cw_4_train_loop.py`); code is the behavioural
target, at attempt grain rather than episode grain.

Every downstream numeric primitive this needs (prompt-boundary gather, the personalised/
global-control forward passes, the production NLL/top-1/exact-match scorers, the
corrected joint (k,c) mixture estimator, gradient-norm-balance arm selection) is already
backbone- and domain-agnostic in `fa_alt2.py` and is imported here unchanged rather
than reimplemented. Only the prompt construction itself is CodeWorkout-specific.
"""

from __future__ import annotations

import torch

from atom.data.cw_outcome_label import CW_OUTCOME_CLASS_TOKENS

# Re-exported, backbone/domain-agnostic primitives -- CW's forward orchestration reuses
# these exactly as FA does; no CW-specific reimplementation exists or should be written.
from atom.simulator.fa_alt2 import (  # noqa: F401
    forward_global_control,
    forward_personalised,
    gather_belief,
    gather_prompt_hidden,
    joint_kc_mixture_nll,
    outcome_loss,
    pick_lambda_a,
    production_log_likelihood_sum_per_row,
    production_loss,
    production_nll_per_row,
    production_topk_per_row,
)

PROMPT_PREFIX = "Problem:\n"
STATEMENT_TO_PRIOR = "\n\nLearner's previous attempt on this problem:\n"
PRIOR_TO_SUFFIX = "\n\n"
CLASS_SUFFIX_TEMPLATE = "This attempt ({class_token}):\n"
# Fixed, short, never truncated -- an episode's first attempt has no prior code by
# construction, and this is what stands in its place. Kept as a
# constant string (not e.g. an empty segment) so the model sees an explicit, always-
# present token for "no prior attempt exists" rather than inferring it from absence.
PRIOR_CODE_START_TEXT = "(none -- this is the learner's first attempt on this problem)"


def _reserved_segment_ids(tok, outcome_code: int) -> dict:
    """Tokenise the four IMMUTABLE pieces once per call (they depend only on
    `outcome_code`, never on the row's statement/prior-code text)."""
    return {
        "prefix_ids": tok(PROMPT_PREFIX, add_special_tokens=True)["input_ids"],
        "mid_ids": tok(STATEMENT_TO_PRIOR, add_special_tokens=False)["input_ids"],
        "tail_ids": tok(PRIOR_TO_SUFFIX, add_special_tokens=False)["input_ids"],
        "suffix_ids": tok(
            CLASS_SUFFIX_TEMPLATE.format(class_token=CW_OUTCOME_CLASS_TOKENS[outcome_code]),
            add_special_tokens=False,
        )["input_ids"],
    }


def build_reserved_prompt_ids(statement: str, prior_code: str | None, tok, max_len: int,
                              min_completion_tokens: int, outcome_code: int) -> tuple[list[int], bool, bool]:
    """Segment-reserved tokenisation. `prior_code=None` marks an episode's
    first attempt -- the fixed `PRIOR_CODE_START_TEXT` is used in
    its place and is NEVER truncated regardless of budget (it is a handful of tokens by
    construction).

    Priority (measured, `cw_5_b1_prompt_length_measurement.py`, `max_len=512`: prior-
    attempt code is the bigger segment at every candidate length once truncation is
    needed at all -- 640/162,741 rows, 0.39%): the STATEMENT shrinks first, down to zero
    if necessary; PRIOR-ATTEMPT CODE is preserved whole up to whatever budget remains
    after the statement's allocation, and only shrinks once the statement already has
    none. Both truncations, when they happen, keep the segment's HEAD.

    Returns `(prompt_ids, statement_truncated, prior_code_truncated)`. Like FA's
    `question_truncated`, these flag the ROW'S property (would this row's segment have
    overrun its share of the budget), not "something got cut" in some looser sense.
    """
    seg = _reserved_segment_ids(tok, outcome_code)
    reserved = len(seg["prefix_ids"]) + len(seg["mid_ids"]) + len(seg["tail_ids"]) + len(seg["suffix_ids"])
    max_prompt_len = max(1, max_len - min_completion_tokens)
    variable_budget = max(0, max_prompt_len - reserved)

    statement_ids_full = tok(statement, add_special_tokens=False)["input_ids"]
    is_first_attempt = prior_code is None
    prior_ids_full = tok(
        PRIOR_CODE_START_TEXT if is_first_attempt else prior_code, add_special_tokens=False,
    )["input_ids"]

    if is_first_attempt:
        # Fixed, tiny, immutable in practice -- reserved off the top, never counted
        # against either truncation flag.
        prior_budget = len(prior_ids_full)
    else:
        prior_budget = min(len(prior_ids_full), variable_budget)
    remaining_for_statement = max(0, variable_budget - prior_budget)

    statement_truncated = len(statement_ids_full) > remaining_for_statement
    statement_ids_used = statement_ids_full[:remaining_for_statement]

    prior_code_truncated = (not is_first_attempt) and (len(prior_ids_full) > prior_budget)
    prior_ids_used = prior_ids_full[:prior_budget]

    ids = (seg["prefix_ids"] + statement_ids_used + seg["mid_ids"] + prior_ids_used
           + seg["tail_ids"] + seg["suffix_ids"])
    return ids, statement_truncated, prior_code_truncated


def encode_example(statement: str, prior_code: str | None, code: str, tok, max_len: int,
                    min_completion_tokens: int = 8, outcome_code: int = 0) -> dict:
    """One attempt row -> one training example. `outcome_code`: the OBSERVED 4-class
    outcome for THIS attempt (`cw_outcome_label.outcome_label`) -- teacher-forced at
    train, ancestrally sampled at generation (mirroring FA-SIM3's conditional-imitation/
    ancestral split), swept over all 4 classes for the class-marginalised eval mixture
    (never the true class alone -- that would leak the label into the score, the same
    rule as FA-SIM3's evaluation).

    Censored-EOS rule, applied exactly as in FA-SIM3:
    `tok.eos_token_id` is appended ONLY when the full completion fits in the room left
    after the prompt; a completion that doesn't fit is `completion_censored=True` with NO
    EOS appended (EOS at an arbitrary truncation cutoff would teach premature
    termination, not the code's actual end)."""
    p_ids, statement_truncated, prior_code_truncated = build_reserved_prompt_ids(
        statement, prior_code, tok, max_len, min_completion_tokens, outcome_code)
    prompt_end_index = len(p_ids) - 1

    completion = f" {code}"
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
        "statement_truncated": statement_truncated,
        "prior_code_truncated": prior_code_truncated,
        "is_first_attempt": prior_code is None,
    }


def collate(examples: list[dict], pad_id: int, device: str = "cpu") -> dict:
    """Right-pad a batch of `encode_example` outputs. Carries all three per-row
    truncation/provenance flags -- `statement_truncated`, `prior_code_truncated`, `is_first_attempt` -- alongside
    `completion_censored`, matching `fa_alt2.collate`'s pattern exactly (prompt_end_index
    needs no padding adjustment; it indexes from the start)."""
    seq_len = max(len(e["input_ids"]) for e in examples)
    n = len(examples)
    input_ids = torch.full((n, seq_len), pad_id, dtype=torch.long)
    labels = torch.full((n, seq_len), -100, dtype=torch.long)
    attention_mask = torch.zeros((n, seq_len), dtype=torch.long)
    prompt_end_index = torch.zeros(n, dtype=torch.long)
    production_valid = torch.zeros(n, dtype=torch.bool)
    completion_censored = torch.zeros(n, dtype=torch.bool)
    statement_truncated = torch.zeros(n, dtype=torch.bool)
    prior_code_truncated = torch.zeros(n, dtype=torch.bool)
    is_first_attempt = torch.zeros(n, dtype=torch.bool)
    for i, e in enumerate(examples):
        length = len(e["input_ids"])
        input_ids[i, :length] = torch.tensor(e["input_ids"], dtype=torch.long)
        labels[i, :length] = torch.tensor(e["labels"], dtype=torch.long)
        attention_mask[i, :length] = 1
        prompt_end_index[i] = e["prompt_end_index"]
        production_valid[i] = e["production_valid"]
        completion_censored[i] = e["completion_censored"]
        statement_truncated[i] = e["statement_truncated"]
        prior_code_truncated[i] = e["prior_code_truncated"]
        is_first_attempt[i] = e["is_first_attempt"]
    return {
        "input_ids": input_ids.to(device),
        "attention_mask": attention_mask.to(device),
        "labels": labels.to(device),
        "prompt_end_index": prompt_end_index.to(device),
        "production_valid": production_valid.to(device),
        "completion_censored": completion_censored.to(device),
        "statement_truncated": statement_truncated.to(device),
        "prior_code_truncated": prior_code_truncated.to(device),
        "is_first_attempt": is_first_attempt.to(device),
    }
