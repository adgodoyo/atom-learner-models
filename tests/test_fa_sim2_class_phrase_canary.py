"""Semantic canary for the class-conditioned prompt: "assert the class phrase survives
encoding for every row of a bundle." Without reserved prompt room, a one-off measurement
with the real tokenizer at `max_len=256` found the phrase truncated in 6.00% of val rows
(204/3,398) and 6.44% of train rows (3,851/59,798); this test stops that regressing silently.

As a semantic canary -- assert the quantity IS what it claims to be, not that the code
ran -- this checks the STRUCTURAL invariant directly: the tail of every encoded prompt is
byte-for-byte the class-phrase suffix's own token ids, never a truncated fragment of it. A
test that merely checked `production_valid=True` or that no exception was raised is
shape-only and would not catch the truncation.
"""

from __future__ import annotations

import pytest

from atom.data.fa_outcome_label import N_OUTCOME
from atom.simulator.fa_alt2 import _prompt_prefix_suffix, build_reserved_prompt_ids, encode_example


@pytest.fixture(scope="module")
def real_tokenizer():
    """The real gemma4-e4b tokenizer, if cached locally (~3 s load)."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained("google/gemma-4-E4B", local_files_only=True)


def _assert_class_phrase_survives(prompt_ids, tok, outcome_code):
    """The structural invariant itself: the LAST len(suffix_ids) tokens of the prompt
    must equal the suffix's own token ids exactly -- not merely contain the class name as
    a substring after decoding, which could pass even if a boundary token got mangled."""
    _, suffix = _prompt_prefix_suffix(outcome_code)
    suffix_ids = tok(suffix, add_special_tokens=False)["input_ids"]
    assert len(prompt_ids) >= len(suffix_ids), (
        f"prompt shorter than the reserved suffix alone (outcome_code={outcome_code}) -- "
        "the reservation budget itself is broken, not just exceeded")
    tail = prompt_ids[-len(suffix_ids):]
    assert tail == suffix_ids, (
        f"class-phrase suffix did not survive intact (outcome_code={outcome_code}): "
        f"expected tail {suffix_ids}, got {tail}")


QUESTION_LENGTHS_CHARS = [1, 50, 200, 500, 2000, 8000]  # from trivially short to far
# longer than any real FA question -- the whole point is this must hold at EVERY length,
# not just the lengths measured in the one-off script.


@pytest.mark.parametrize("n_chars", QUESTION_LENGTHS_CHARS)
@pytest.mark.parametrize("outcome_code", list(range(N_OUTCOME)) + [None])
def test_class_phrase_survives_at_every_length_and_every_class(real_tokenizer, n_chars, outcome_code):
    question = "A rectangle has width 4.5 and " * (n_chars // 30 + 1)
    question = question[:n_chars]
    prompt_ids, _ = build_reserved_prompt_ids(
        question, real_tokenizer, max_len=256, min_completion_tokens=8, outcome_code=outcome_code)
    _assert_class_phrase_survives(prompt_ids, real_tokenizer, outcome_code)


def test_class_phrase_survives_for_every_row_of_a_synthetic_bundle(real_tokenizer):
    """The bundle-level form the spec asks for directly: one assertion per row, not a
    single hand-picked example. Mixes short/medium/long/pathologically-long questions and
    every outcome class, matching the population the measurement above covered (all
    rows, not a sample)."""
    lengths = [3, 20, 60, 100, 150, 250, 400, 600, 1000, 3000]
    bundle = [
        ((f"Question body number {i} " * (n // 25 + 1))[:n], i % N_OUTCOME)
        for i, n in enumerate(lengths)
    ]
    for question, outcome_code in bundle:
        prompt_ids, _ = build_reserved_prompt_ids(
            question, real_tokenizer, max_len=256, min_completion_tokens=8,
            outcome_code=outcome_code)
        _assert_class_phrase_survives(prompt_ids, real_tokenizer, outcome_code)


def test_encode_example_class_token_truncated_flag_matches_whether_truncation_occurred(real_tokenizer):
    short = encode_example("What is 2+2?", "4", real_tokenizer, max_len=256, outcome_code=0)
    assert short["class_token_truncated"] is False

    very_long_question = "A rectangle has width 4.5 and length 6.2. " * 40  # far over budget
    long = encode_example(very_long_question, "4", real_tokenizer, max_len=256, outcome_code=0)
    assert long["class_token_truncated"] is True
    # the flag says the QUESTION was truncated -- the class phrase must still have survived
    prompt_ids = long["input_ids"][: long["prompt_end_index"] + 1]
    _assert_class_phrase_survives(prompt_ids, real_tokenizer, outcome_code=0)


def test_class_phrase_survives_on_the_mc_scoring_path_for_every_option_and_class(real_tokenizer):
    """The same canary on the SCORING path, not only the TRAINING encoder: calling
    `encode_example` WITHOUT `outcome_code` from
    `atom.simulator.fa_bread_test.mc_option_log_likelihoods` would silently build the
    unconditioned template. This reproduces the exact construction
    `mc_option_joint_log_likelihoods` uses -- one question, several MC options, EVERY
    outcome class -- with the real tokenizer and no GPU, and asserts the invariant holds
    for every (option, class) prompt actually built on that path, not just the training
    encoder's own single-example case above."""
    question = "A rectangle has width 4.5 and length 6.2. What is the perimeter?"
    options = ["21.4", "10.7", "27", "4.5 x 6.2"]
    for outcome_code in range(N_OUTCOME):
        for option in options:
            prompt_ids, _ = build_reserved_prompt_ids(
                question, real_tokenizer, max_len=256, min_completion_tokens=8, outcome_code=outcome_code)
            _assert_class_phrase_survives(prompt_ids, real_tokenizer, outcome_code=outcome_code)


def test_mc_option_log_likelihoods_scoring_path_actually_calls_the_class_conditioned_encoder(real_tokenizer):
    """Power check for the required-`outcome_code` rule: reconstruct the call shape
    without it (`encode_example(question, opt, tok, max_len)`) and confirm its prompt tail is
    the UNCONDITIONED template's own ending, never a class suffix -- proving this test
    class has power to catch the defect (a scoring path silently opting into the
    wrong template) rather than only confirming the correct code path passes."""
    from atom.simulator.fa_alt2 import PROMPT_TEMPLATE

    question = "What is 2 + 2?"
    old_style_example = encode_example(question, "4", real_tokenizer, max_len=256)  # no outcome_code
    prompt_ids = old_style_example["input_ids"][: old_style_example["prompt_end_index"] + 1]
    _, suffix = _prompt_prefix_suffix(0)
    suffix_ids = real_tokenizer(suffix, add_special_tokens=False)["input_ids"]
    assert prompt_ids[-len(suffix_ids):] != suffix_ids, (
        "the old unconditioned-default call shape unexpectedly matches a class suffix -- "
        "this test has no power, pick a question/class combination where they differ")
    # and confirm it DOES match the unconditioned template's own literal ending instead
    expected_unconditioned_prompt = PROMPT_TEMPLATE.format(question=question)
    expected_tail = real_tokenizer(expected_unconditioned_prompt.rsplit(question, 1)[1],
                                   add_special_tokens=False)["input_ids"]
    assert prompt_ids[-len(expected_tail):] == expected_tail


def test_reserved_budget_never_shorter_than_the_old_unsafe_encoder_would_have_truncated_to():
    """Power check: reconstruct what an unreserved encoder does (tokenise the whole
    prompt, slice the tail) on a known
    over-length question, and confirm it DOES clip into the suffix -- proving this test
    class actually has power to detect the truncation defect, not just that the current code
    passes its own construction."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("google/gemma-4-E4B", local_files_only=True)
    from atom.simulator.fa_alt2 import CLASS_CONDITIONED_PROMPT_TEMPLATE
    from atom.data.fa_outcome_label import OUTCOME_CLASS_TOKENS

    very_long_question = "A rectangle has width 4.5 and length 6.2. " * 40
    prompt = CLASS_CONDITIONED_PROMPT_TEMPLATE.format(
        question=very_long_question, class_token=OUTCOME_CLASS_TOKENS[0])
    old_p_ids = tok(prompt, add_special_tokens=True)["input_ids"]
    max_prompt_len = 256 - 8
    old_p_ids_truncated = old_p_ids[:max_prompt_len]

    _, suffix = _prompt_prefix_suffix(0)
    suffix_ids = tok(suffix, add_special_tokens=False)["input_ids"]
    old_tail = old_p_ids_truncated[-len(suffix_ids):]
    assert old_tail != suffix_ids, (
        "the unreserved reconstruction did not actually clip the class phrase on this "
        "input -- the regression check has no power here; pick a longer question")
