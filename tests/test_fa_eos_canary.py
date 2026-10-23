"""Semantic canary for the stop symbol: "every UNCENSORED training label ends in a stop
symbol, and no CENSORED label does."

If `labels` end at the last answer token, the position that would emit EOS receives zero
gradient, so content is learned but termination never is (real `4` -> generated
`4500000...`; an earlier FA simulator was trained this way). Appending EOS to every
completion with any room at all, INCLUDING ones truncated to fit, teaches the model to stop
at an arbitrary length cutoff rather than at the answer's actual end -- a subtler version of
the same defect. The rule: `tok.eos_token_id` is appended only when the FULL completion
fits; a row whose answer is too long for the remaining room is `completion_censored=True`
and carries NO EOS.

As a semantic canary -- assert the quantity IS what it claims to be, not that the code
ran -- this checks the STRUCTURAL invariant directly on `labels`, in BOTH
directions, keyed off `completion_censored` rather than assumed from context.
"""

from __future__ import annotations

import pytest

from atom.data.fa_outcome_label import N_OUTCOME
from atom.simulator.fa_alt2 import build_reserved_prompt_ids, encode_example


@pytest.fixture(scope="module")
def real_tokenizer():
    """The real gemma4-e4b tokenizer, if cached locally (~3 s load)."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained("google/gemma-4-E4B", local_files_only=True)


def _last_completion_token(ex: dict) -> int | None:
    """The structural invariant itself: the last entry of `labels` that survived the
    prompt mask. `None` iff the row has zero completion tokens at all (only possible
    when the prompt alone consumes the entire `max_len` budget -- `room == 0`, which is
    always `completion_censored=True` under the refined rule)."""
    for tok_id in reversed(ex["labels"]):
        if tok_id != -100:
            return tok_id
    return None


def _assert_two_directional_canary(ex: dict, eos_id: int, context: str = "") -> None:
    """The canary itself, in the exact two-directional form the spec names: uncensored
    ⇒ ends in EOS; censored ⇒ does NOT end in EOS (either a real content token, or
    `None` in the room==0 boundary case)."""
    last = _last_completion_token(ex)
    if ex["completion_censored"]:
        assert last != eos_id, f"censored row ends in EOS {context}"
    else:
        assert last == eos_id, f"uncensored row does not end in EOS {context}"


QUESTION_LENGTHS_CHARS = [1, 20, 60, 150, 400, 1000, 3000, 8000]
ANSWER_LENGTHS_CHARS = [1, 5, 20, 80, 300]


@pytest.mark.parametrize("n_chars", QUESTION_LENGTHS_CHARS)
@pytest.mark.parametrize("outcome_code", list(range(N_OUTCOME)) + [None])
def test_canary_holds_at_every_question_length_and_every_class(real_tokenizer, n_chars, outcome_code):
    """Question length only ever eats into the QUESTION's own truncation budget
    (`build_reserved_prompt_ids`), never the completion's -- so this short, fixed answer
    ("42") should stay uncensored (and thus EOS-terminated) at every question length."""
    question = ("A rectangle has width 4.5 and length 6.2. " * (n_chars // 40 + 1))[:n_chars]
    ex = encode_example(question, "42", real_tokenizer, max_len=256, outcome_code=outcome_code)
    assert ex["completion_censored"] is False
    _assert_two_directional_canary(ex, real_tokenizer.eos_token_id, context=f"n_chars={n_chars}")


@pytest.mark.parametrize("n_chars", ANSWER_LENGTHS_CHARS)
def test_canary_holds_at_every_answer_length_up_to_the_uncensored_range(real_tokenizer, n_chars):
    answer = ("seven point two five " * (n_chars // 20 + 1))[:n_chars]
    ex = encode_example("What is the answer?", answer, real_tokenizer, max_len=256)
    _assert_two_directional_canary(ex, real_tokenizer.eos_token_id, context=f"n_chars={n_chars}")


def test_canary_holds_for_every_row_of_a_synthetic_bundle(real_tokenizer):
    """The bundle-level form of the canary: one assertion per row, not a
    single hand-picked example. Mixes short/medium/long questions, empty through long
    answers (including some deliberately long enough to censor), and every outcome class
    including the unconditioned template -- checked in BOTH directions per row."""
    q_lengths = [3, 30, 90, 250, 600, 1500, 4000]
    a_lengths = [1, 4, 15, 50, 150, 2000, 5000]  # last two are long enough to censor
    bundle = [
        (
            (f"Question body number {i} " * (qn // 25 + 1))[:qn],
            (f"answer text {i} " * (an // 12 + 1))[:an],
            i % N_OUTCOME if i % 3 else None,
        )
        for i, (qn, an) in enumerate(zip((q_lengths * 2)[: len(a_lengths)], a_lengths))
    ]
    saw_censored = False
    for question, answer, outcome_code in bundle:
        ex = encode_example(question, answer, real_tokenizer, max_len=256, outcome_code=outcome_code)
        saw_censored = saw_censored or ex["completion_censored"]
        _assert_two_directional_canary(
            ex, real_tokenizer.eos_token_id,
            context=f"question={question!r} answer={answer!r} outcome_code={outcome_code}")
    assert saw_censored, (
        "this bundle was meant to include at least one censored row (the long answers) -- "
        "if none censored, the test has no power on that direction; widen a_lengths")


def test_canary_holds_on_a_deliberately_long_answer_that_must_censor(real_tokenizer):
    """A direct, unambiguous censored-row test: an answer far longer than any possible
    completion room at `max_len=256` must censor, and must NOT end in EOS."""
    long_answer = "seven point two five repeating and also negative three halves " * 50
    ex = encode_example("What is the answer?", long_answer, real_tokenizer, max_len=256)
    assert ex["completion_censored"] is True
    assert ex["production_valid"] is True  # truncated content is still content
    _assert_two_directional_canary(ex, real_tokenizer.eos_token_id, context="deliberately long answer")


class _FakeTokWordSplit:
    """Word-splitting stub (matches every `_FakeTok` in this suite's other FA test
    files): unlike the real tokenizer, a whitespace-only completion tokenises to
    genuinely zero ids under this scheme, which is what makes it useful here -- the real
    gemma4-e4b tokenizer emits at least one id for `" "` itself, so it cannot exercise
    this specific boundary (a truly empty content stream) directly."""

    pad_token_id = 0
    eos_token_id = 63

    def __call__(self, text: str, add_special_tokens: bool = True) -> dict:
        ids = [(sum(ord(c) for c in w) % 40) + 2 for w in text.split(" ") if w != ""]
        if add_special_tokens:
            ids = [1, *ids]
        return {"input_ids": ids}


def test_canary_holds_when_the_answer_is_genuinely_empty():
    """An empty answer has zero content tokens, which fits trivially (0+1<=room) --
    UNCENSORED, EOS-only completion, `production_valid=False` (content-only signal,
    unaffected by EOS's presence)."""
    tok = _FakeTokWordSplit()
    ex = encode_example("What is 2+2?", "", tok, max_len=32)
    assert ex["completion_censored"] is False
    assert ex["production_valid"] is False
    assert _last_completion_token(ex) == tok.eos_token_id


def test_room_zero_is_censored_by_construction_and_carries_no_eos(real_tokenizer):
    """If the prompt alone consumes the entire `max_len` budget, there is no room left
    for even one completion token, EOS included -- `completion_censored=True` (0
    content-plus-EOS tokens can never fit non-positive room), `production_valid=False`,
    and no EOS is possible."""
    very_long_question = "A rectangle has width 4.5 and length 6.2. " * 200
    p_ids, _ = build_reserved_prompt_ids(very_long_question, real_tokenizer, max_len=8,
                                         min_completion_tokens=8)
    assert len(p_ids) >= 8  # the prompt alone already meets/exceeds max_len
    ex = encode_example(very_long_question, "42", real_tokenizer, max_len=8)
    assert ex["completion_censored"] is True
    assert ex["production_valid"] is False
    assert _last_completion_token(ex) is None


def test_would_have_caught_the_original_eos_free_defect_reconstructing_old_targets(real_tokenizer):
    """Power check #1 (matching `test_fa_sim2_class_phrase_canary.py`): reconstruct EXACTLY
    what an EOS-free `encode_example` produces (completion ids with no EOS appended at all,
    ever) and confirm the canary correctly FAILS the "uncensored ⇒ ends in EOS" direction
    on it -- proving this test class has power to detect that defect."""
    p_ids, _ = build_reserved_prompt_ids("What is 2+2?", real_tokenizer, max_len=256,
                                         min_completion_tokens=8)
    completion = " 4"
    old_c_ids = real_tokenizer(completion, add_special_tokens=False)["input_ids"]
    room = 256 - len(p_ids)
    old_c_ids = old_c_ids[:room]  # the original-defect truncation, with no EOS ever appended
    old_labels = [-100] * len(p_ids) + old_c_ids

    old_ex = {"labels": old_labels, "completion_censored": False}  # this row easily fits -> should be uncensored
    assert _last_completion_token(old_ex) != real_tokenizer.eos_token_id, (
        "the EOS-free reconstruction already ends in EOS on this input by coincidence -- "
        "the regression check has no power here; pick a different answer")
    with pytest.raises(AssertionError):
        _assert_two_directional_canary(old_ex, real_tokenizer.eos_token_id)


def test_would_have_caught_the_superseded_censored_eos_defect(real_tokenizer):
    """Power check #2: reconstruct the unconditional-EOS logic (`content_ids[:room-1] +
    [eos]` UNCONDITIONALLY, no censoring distinction) on a deliberately long answer that
    the censoring rule marks `completion_censored=True`. Under that logic this row still
    ends in EOS -- teaching the model to stop at the truncation cutoff rather than the
    answer's real end. Confirm the two-directional canary correctly FAILS on this
    reconstruction, proving it has power to catch that defect, not just the EOS-free one."""
    long_answer = "seven point two five repeating and also negative three halves " * 50
    p_ids, _ = build_reserved_prompt_ids("What is the answer?", real_tokenizer, max_len=256,
                                         min_completion_tokens=8)
    room = 256 - len(p_ids)
    full_content_ids = real_tokenizer(f" {long_answer}", add_special_tokens=False)["input_ids"]
    assert len(full_content_ids) + 1 > room, (
        "this answer must actually be long enough to censor under the refined rule -- "
        "the regression check has no power here; make the answer longer")

    # the superseded landed logic: always reserve the last slot for EOS, regardless of
    # whether the full completion fit.
    superseded_c_ids = full_content_ids[: room - 1] + [real_tokenizer.eos_token_id]
    superseded_labels = [-100] * len(p_ids) + superseded_c_ids
    superseded_ex = {"labels": superseded_labels, "completion_censored": True}  # refined rule says True

    assert _last_completion_token(superseded_ex) == real_tokenizer.eos_token_id, (
        "the superseded reconstruction must end in EOS for this test to have power")
    with pytest.raises(AssertionError):
        _assert_two_directional_canary(superseded_ex, real_tokenizer.eos_token_id)
