"""Semantic canaries for the CodeWorkout segment-reserved prompt encoder (translated onto
its two-truncatable-segment shape): the class phrase survives encoding for every row, and
every UNCENSORED
target ends in EOS while no CENSORED target does. Mirrors
`test_fa_sim2_class_phrase_canary.py` / `test_fa_eos_canary.py`'s exact structure --
STRUCTURAL assertions on the token ids themselves, never "no exception was raised."

Also covers CW's genuinely new piece FA never needed: the two-way truncation PRIORITY
(statement truncated first, prior-attempt code preserved preferentially -- decided by
`cw_5_b1_prompt_length_measurement.py`'s real-corpus measurement) and the empty/START
prior-code segment for an episode's first attempt.
"""

from __future__ import annotations

import pytest

from atom.data.cw_outcome_label import CW_OUTCOME_CLASS_TOKENS, N_OUTCOME
from atom.simulator.cw_alt import (
    CLASS_SUFFIX_TEMPLATE,
    PRIOR_CODE_START_TEXT,
    _reserved_segment_ids,
    build_reserved_prompt_ids,
    collate,
    encode_example,
)


class _FakeTok:
    """Deterministic, dependency-free tokenizer stand-in (matches every other `_FakeTok`
    in this suite, e.g. `test_fa_alt2.py`) -- avoids Python's per-process string-hash
    randomisation so token ids are stable across runs."""

    pad_token_id = 0
    eos_token_id = 63

    def __call__(self, text: str, add_special_tokens: bool = True) -> dict:
        words = [w for w in text.replace("\n", " \n ").split(" ") if w != ""]
        ids = [self._word_id(w) for w in words]
        if add_special_tokens:
            ids = [1, *ids]
        return {"input_ids": ids}

    @staticmethod
    def _word_id(w: str) -> int:
        return (sum(ord(c) for c in w) % 40) + 2


@pytest.fixture(scope="module")
def real_tokenizer():
    """The real gemma4-e4b tokenizer, if cached locally (~3 s load)."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained("google/gemma-4-E4B", local_files_only=True)


def _assert_class_phrase_survives(prompt_ids, tok, outcome_code):
    """The structural invariant: the LAST len(suffix_ids) tokens of the prompt must equal
    the class-suffix's own token ids exactly -- not a substring check after decoding."""
    suffix_ids = tok(CLASS_SUFFIX_TEMPLATE.format(class_token=CW_OUTCOME_CLASS_TOKENS[outcome_code]),
                     add_special_tokens=False)["input_ids"]
    assert len(prompt_ids) >= len(suffix_ids), (
        f"prompt shorter than the reserved suffix alone (outcome_code={outcome_code}) -- "
        "the reservation budget itself is broken")
    tail = prompt_ids[-len(suffix_ids):]
    assert tail == suffix_ids, (
        f"class-phrase suffix did not survive intact (outcome_code={outcome_code}): "
        f"expected {suffix_ids}, got {tail}")


STATEMENT_LENGTHS_CHARS = [1, 50, 200, 500, 2000]
PRIOR_CODE_LENGTHS_CHARS = [0, 50, 300, 1500, 6000]  # 0 == first-attempt (prior_code=None)


@pytest.mark.parametrize("prior_len", PRIOR_CODE_LENGTHS_CHARS)
@pytest.mark.parametrize("stmt_len", STATEMENT_LENGTHS_CHARS)
@pytest.mark.parametrize("outcome_code", list(range(N_OUTCOME)))
def test_class_phrase_survives_at_every_length_and_every_class(real_tokenizer, stmt_len, prior_len, outcome_code):
    statement = ("Given an int n, return whether it is prime. " * (stmt_len // 45 + 1))[:stmt_len]
    prior_code = None if prior_len == 0 else ("public int f(int n) { return n; } " * (prior_len // 35 + 1))[:prior_len]
    prompt_ids, _, _ = build_reserved_prompt_ids(
        statement, prior_code, real_tokenizer, max_len=512, min_completion_tokens=8, outcome_code=outcome_code)
    _assert_class_phrase_survives(prompt_ids, real_tokenizer, outcome_code)


def test_class_phrase_survives_for_every_row_of_a_synthetic_bundle(real_tokenizer):
    lengths = [3, 20, 60, 150, 400, 1200, 3000]
    bundle = [
        ((f"Problem statement number {i} " * (n // 30 + 1))[:n],
         None if i % 2 == 0 else (f"public int m{i}() {{ return {i}; }} " * (n // 25 + 1))[:n],
         i % N_OUTCOME)
        for i, n in enumerate(lengths)
    ]
    for statement, prior_code, outcome_code in bundle:
        prompt_ids, _, _ = build_reserved_prompt_ids(
            statement, prior_code, real_tokenizer, max_len=512, min_completion_tokens=8, outcome_code=outcome_code)
        _assert_class_phrase_survives(prompt_ids, real_tokenizer, outcome_code)


def test_statement_truncated_first_prior_code_preserved_when_pair_overflows(real_tokenizer):
    """The measured priority: when statement+prior-code together exceed the
    variable budget, the STATEMENT shrinks (down to zero if needed); the prior-attempt
    code is kept whole as long as it alone fits the budget."""
    long_statement = "A rectangle has width 4.5 and length 6.2, compute its area. " * 30
    modest_prior_code = "public double area(double w, double h) { return w * h; }"
    prompt_ids, statement_truncated, prior_code_truncated = build_reserved_prompt_ids(
        long_statement, modest_prior_code, real_tokenizer, max_len=64, min_completion_tokens=8, outcome_code=0)
    assert statement_truncated is True
    assert prior_code_truncated is False
    # the prior code's own tokens must appear intact, in full, inside the prompt
    prior_ids = real_tokenizer(modest_prior_code, add_special_tokens=False)["input_ids"]
    assert any(prompt_ids[i:i + len(prior_ids)] == prior_ids for i in range(len(prompt_ids) - len(prior_ids) + 1))


def test_prior_code_also_truncates_once_statement_is_already_exhausted(real_tokenizer):
    """If the prior-attempt code ALONE overflows the entire variable budget (statement
    given nothing, 0 tokens), the prior code must itself truncate -- keeping its HEAD."""
    tiny_statement = "x"
    huge_prior_code = "public int f(int n) { return n + 1; } " * 200
    prompt_ids, statement_truncated, prior_code_truncated = build_reserved_prompt_ids(
        tiny_statement, huge_prior_code, real_tokenizer, max_len=48, min_completion_tokens=8, outcome_code=0)
    assert prior_code_truncated is True
    _assert_class_phrase_survives(prompt_ids, real_tokenizer, outcome_code=0)
    # HEAD preserved: the prompt must contain the prior code's OWN opening tokens, not its tail.
    prior_head_ids = real_tokenizer(huge_prior_code, add_special_tokens=False)["input_ids"][:5]
    assert any(prompt_ids[i:i + 5] == prior_head_ids for i in range(len(prompt_ids) - 4))


def test_first_attempt_uses_the_fixed_start_text_and_is_never_flagged_truncated(real_tokenizer):
    """`prior_code=None` marks an episode's first attempt -- encodes
    to the fixed `PRIOR_CODE_START_TEXT`, and `prior_code_truncated` is always False for
    it (it is a handful of fixed tokens, never subject to the truncation budget)."""
    very_long_statement = "Given 2 ints a and b, return their sum. " * 50
    prompt_ids, statement_truncated, prior_code_truncated = build_reserved_prompt_ids(
        very_long_statement, None, real_tokenizer, max_len=64, min_completion_tokens=8, outcome_code=1)
    assert prior_code_truncated is False
    start_ids = real_tokenizer(PRIOR_CODE_START_TEXT, add_special_tokens=False)["input_ids"]
    assert any(prompt_ids[i:i + len(start_ids)] == start_ids for i in range(len(prompt_ids) - len(start_ids) + 1))


def test_encode_example_is_first_attempt_flag_matches_prior_code_argument():
    tok = _FakeTok()
    first = encode_example("statement", None, "code", tok, max_len=64, outcome_code=0)
    revision = encode_example("statement", "prior code", "code", tok, max_len=64, outcome_code=0)
    assert first["is_first_attempt"] is True
    assert revision["is_first_attempt"] is False


# --- FA-EOS censored-refinement pattern, translated onto CW's own completion (code) ---


def _last_completion_token(ex: dict):
    for tok_id in reversed(ex["labels"]):
        if tok_id != -100:
            return tok_id
    return None


def _assert_two_directional_canary(ex: dict, eos_id: int, context: str = "") -> None:
    last = _last_completion_token(ex)
    if ex["completion_censored"]:
        assert last != eos_id, f"censored row ends in EOS {context}"
    else:
        assert last == eos_id, f"uncensored row does not end in EOS {context}"


def test_eos_canary_holds_for_every_row_of_a_synthetic_bundle(real_tokenizer):
    stmt_lengths = [3, 30, 90, 250, 600]
    code_lengths = [1, 4, 15, 50, 150, 2000, 5000]  # last two long enough to censor at max_len=256
    bundle = [
        ((f"Problem {i} statement text " * (sn // 25 + 1))[:sn],
         None if i % 3 == 0 else (f"public void m{i}() {{ }} " * (sn // 20 + 1))[:sn],
         (f"public int solve{i}() {{ return {i}; }} " * (cn // 20 + 1))[:cn],
         i % N_OUTCOME)
        for i, (sn, cn) in enumerate(zip((stmt_lengths * 2)[:len(code_lengths)], code_lengths))
    ]
    saw_censored = False
    for statement, prior_code, code, outcome_code in bundle:
        ex = encode_example(statement, prior_code, code, real_tokenizer, max_len=256, outcome_code=outcome_code)
        saw_censored = saw_censored or ex["completion_censored"]
        _assert_two_directional_canary(ex, real_tokenizer.eos_token_id, context=f"code={code!r}")
    assert saw_censored, "bundle meant to include at least one censored row -- widen code_lengths"


def test_eos_canary_holds_on_a_deliberately_long_code_completion_that_must_censor(real_tokenizer):
    long_code = "public int helper() { int x = 0; x = x + 1; return x; } " * 50
    ex = encode_example("statement", None, long_code, real_tokenizer, max_len=256, outcome_code=0)
    assert ex["completion_censored"] is True
    assert ex["production_valid"] is True
    _assert_two_directional_canary(ex, real_tokenizer.eos_token_id, context="deliberately long code")


def test_room_zero_is_censored_by_construction_and_carries_no_eos(real_tokenizer):
    very_long_statement = "A rectangle has width 4.5 and length 6.2. " * 200
    ex = encode_example(very_long_statement, None, "public int f() { return 1; }", real_tokenizer,
                        max_len=8, outcome_code=0)
    assert ex["completion_censored"] is True
    assert ex["production_valid"] is False
    assert _last_completion_token(ex) is None


def test_would_have_caught_the_no_reservation_defect_reconstructing_old_style_truncation(real_tokenizer):
    """Power check: reconstruct
    what a naive "tokenise the whole prompt string, slice the tail" encoder would have
    done on a known over-length statement, and confirm it clips into the class-phrase
    suffix -- proving this test class has power to detect the original class of bug
    (the same shape as FA's canary #4, applied to CW's own segments)."""
    very_long_statement = "A rectangle has width 4.5 and length 6.2. " * 60
    prior_code = "public double area() { return 0.0; }"
    full_prompt_text = (
        f"Problem:\n{very_long_statement}\n\nLearner's previous attempt on this problem:\n"
        f"{prior_code}\n\n{CLASS_SUFFIX_TEMPLATE.format(class_token=CW_OUTCOME_CLASS_TOKENS[0])}"
    )
    old_p_ids = real_tokenizer(full_prompt_text, add_special_tokens=True)["input_ids"]
    max_prompt_len = 96 - 8
    old_p_ids_truncated = old_p_ids[:max_prompt_len]

    suffix_ids = real_tokenizer(CLASS_SUFFIX_TEMPLATE.format(class_token=CW_OUTCOME_CLASS_TOKENS[0]),
                                add_special_tokens=False)["input_ids"]
    old_tail = old_p_ids_truncated[-len(suffix_ids):]
    assert old_tail != suffix_ids, (
        "the naive reconstruction did not actually clip the class phrase on this input -- "
        "no power; pick a longer statement")


def test_would_have_caught_the_eos_always_appended_defect(real_tokenizer):
    """Reconstruct the FA-EOS-superseded logic (EOS appended even to a truncated
    completion) on a deliberately long code completion, and confirm the two-directional
    canary correctly FAILS on it."""
    long_code = "public int helper() { int x = 0; x = x + 1; return x; } " * 50
    prompt_ids, _, _ = build_reserved_prompt_ids("statement", None, real_tokenizer, max_len=256,
                                                  min_completion_tokens=8, outcome_code=0)
    room = 256 - len(prompt_ids)
    full_content_ids = real_tokenizer(f" {long_code}", add_special_tokens=False)["input_ids"]
    assert len(full_content_ids) + 1 > room, "code not long enough to censor -- widen it"

    superseded_c_ids = full_content_ids[:room - 1] + [real_tokenizer.eos_token_id]
    superseded_labels = [-100] * len(prompt_ids) + superseded_c_ids
    superseded_ex = {"labels": superseded_labels, "completion_censored": True}

    assert _last_completion_token(superseded_ex) == real_tokenizer.eos_token_id
    with pytest.raises(AssertionError):
        _assert_two_directional_canary(superseded_ex, real_tokenizer.eos_token_id)


def test_collate_carries_all_three_truncation_flags_and_is_first_attempt():
    tok = _FakeTok()
    examples = [
        encode_example("short stmt", None, "short code", tok, max_len=64, outcome_code=0),
        encode_example("a much longer statement " * 20, "some prior code", "code", tok, max_len=32, outcome_code=2),
    ]
    batch = collate(examples, pad_id=tok.pad_token_id)
    assert batch["is_first_attempt"].tolist() == [True, False]
    assert "statement_truncated" in batch and "prior_code_truncated" in batch
    assert batch["input_ids"].shape[0] == 2
    assert batch["prompt_end_index"].dtype.is_floating_point is False


def test_reserved_segment_ids_depends_only_on_outcome_code(real_tokenizer):
    a = _reserved_segment_ids(real_tokenizer, outcome_code=0)
    b = _reserved_segment_ids(real_tokenizer, outcome_code=0)
    assert a == b
    c = _reserved_segment_ids(real_tokenizer, outcome_code=3)
    assert a["suffix_ids"] != c["suffix_ids"]
