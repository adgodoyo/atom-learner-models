"""Lane T0 -- GA-1 tier 0 prompt/identifier canaries.

These assert what the design depends on (the model can actually see and address every
menu item; the read-out position is where tier 3 expects it), not merely that the
renderer ran.
"""

from __future__ import annotations

import pathlib
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))

import ga_1_tier0_prompt as P  # noqa: E402


class FakeTokenizer:
    """Whitespace/character tokenizer with gemma's relevant property: single-char = 1 token."""

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [0] * max(1, len(str(text).split()))}


def _statements(n: int) -> dict[int, dict[str, str]]:
    return {
        p: {"method_name": f"method{p}", "statement": f"statement number {p} " + "word " * 5}
        for p in range(n)
    }


def _cands(n: int) -> list[dict[str, int]]:
    return [{"problem_id": p} for p in range(n)]


def test_alphabet_covers_the_menu_cap():
    assert P.MAX_MENU <= len(P.IDENTIFIERS)
    assert len(set(P.IDENTIFIERS)) == len(P.IDENTIFIERS)
    assert P.identifiers_for(50)[:3] == ("A", "B", "C")


def test_menu_larger_than_alphabet_fails_closed():
    with pytest.raises(P.MenuTooLarge):
        P.identifiers_for(len(P.IDENTIFIERS) + 1)
    with pytest.raises(P.MenuTooLarge):
        P.identifiers_for(0)


def test_every_identifier_is_addressable_in_the_rendered_prompt():
    tok = FakeTokenizer()
    for n in (1, 5, 26, 27, 50):
        prompt = P.build_prompt([], _cands(n), 8, 0, {}, _statements(n), tok,
                                max_prompt_tokens=10**6, with_history=False)
        P.assert_prompt_wellformed(prompt, _cands(n), tok, 10**6)
        for ident in P.identifiers_for(n):
            assert f"\n{ident}. " in prompt


def test_prompt_ends_on_the_tier3_readout_position():
    tok = FakeTokenizer()
    prompt = P.build_prompt([], _cands(6), 8, 2, {}, _statements(6), tok,
                            max_prompt_tokens=10**6, with_history=False)
    assert prompt.endswith("CHOICE=")
    assert "SELECTIONS REMAINING INCLUDING THIS ONE: 6" in prompt


def test_over_long_fixed_part_raises_instead_of_truncating():
    """The generator left-truncates; an over-long menu must never reach it."""
    tok = FakeTokenizer()
    with pytest.raises(P.PromptTooLong):
        P.build_prompt([], _cands(50), 8, 0, {}, _statements(50), tok,
                       max_prompt_tokens=32, with_history=False)


def test_wellformed_canary_catches_a_dropped_menu_line():
    tok = FakeTokenizer()
    cands = _cands(10)
    prompt = P.build_prompt([], cands, 8, 0, {}, _statements(10), tok,
                            max_prompt_tokens=10**6, with_history=False)
    mutilated = prompt.replace("\nC. ", "\n?. ", 1)
    with pytest.raises(AssertionError):
        P.assert_prompt_wellformed(mutilated, cands, tok, 10**6)


def test_wellformed_canary_catches_an_oversized_prompt():
    tok = FakeTokenizer()
    cands = _cands(10)
    prompt = P.build_prompt([], cands, 8, 0, {}, _statements(10), tok,
                            max_prompt_tokens=10**6, with_history=False)
    with pytest.raises(AssertionError):
        P.assert_prompt_wellformed(prompt, cands, tok, 4)


def test_no_history_prompt_differs_only_in_the_history_block():
    tok = FakeTokenizer()
    cands = _cands(4)
    text = {
        "e1": {"episode_id": "e1", "method_name": "m", "statement": "s", "n_attempts": 2,
               "outcome_trace": ["CLEAN_ZERO", "CLEAN_FULL"], "final_compile_result": "ok",
               "final_score": 1.0, "final_code": "class A {}"}
    }
    events = [{"episode_id": "e1", "order": 0}]
    with_h = P.build_prompt(events, cands, 8, 0, text, _statements(4), tok,
                            max_prompt_tokens=10**6, with_history=True)
    without = P.build_prompt(events, cands, 8, 0, text, _statements(4), tok,
                             max_prompt_tokens=10**6, with_history=False)
    assert "class A {}" in with_h and "class A {}" not in without
    assert "No student history has been revealed." in without
    head_w, head_n = with_h.split("REVEALED HISTORY")[0], without.split("REVEALED HISTORY")[0]
    tail_w, tail_n = with_h.split("CURRENT MENU")[1], without.split("CURRENT MENU")[1]
    assert head_w == head_n and tail_w == tail_n


@pytest.mark.parametrize("output,expected,status", [
    ("A", 0, "first_token"),
    (" B", 1, "first_token"),
    ("C\n", 2, "first_token"),
    ("Z", 25, "first_token"),
    ("a", 26, "first_token"),
    ("CHOICE=D", 3, "later_token"),
    ("pick E.", 4, "later_token"),
    # A bare leading identifier wins even inside prose: on a 50-item menu "I"
    # IS identifier 9.  Documented, not accidental -- the prompt ends on
    # "CHOICE=", so a prose continuation is already off-distribution.
    ("I would pick E.", 8, "first_token"),
    ("", 0, "fallback_first"),
    ("42", 0, "fallback_first"),
    ("<unk>", 0, "fallback_first"),
])
def test_parser_maps_identifiers_to_slots(output, expected, status):
    slot, got = P.parse_choice(output, 50)
    assert (slot, got) == (expected, status)


def test_parser_never_returns_an_out_of_menu_slot():
    for output in ("Z", "q", "hello world", "", "A"):
        for n in (1, 3, 26, 50):
            slot, _ = P.parse_choice(output, n)
            assert 0 <= slot < n


def test_parser_ignores_identifiers_inside_words():
    """'the' must not resolve to identifier 't' on a 50-item menu."""
    slot, status = P.parse_choice("the", 50)
    assert status == "fallback_first" and slot == 0


def test_single_token_canary_on_the_real_tokenizer():
    transformers = pytest.importorskip("transformers")
    try:
        tok = transformers.AutoTokenizer.from_pretrained(
            "google/gemma-4-E4B", local_files_only=True)
    except Exception as exc:  # pragma: no cover - cache-dependent
        pytest.skip(f"gemma4-e4b tokenizer not cached: {exc}")
    ids = P.assert_single_token(tok)
    assert len(ids) == len(P.IDENTIFIERS)
    assert all(len(v) == 1 for v in ids.values())
    # The convention this replaces: numerals past 9 are NOT single tokens.
    multi = [str(i) for i in range(1, 51)
             if len(tok(str(i), add_special_tokens=False)["input_ids"]) != 1]
    assert len(multi) >= 40, "numeric menu identifiers unexpectedly single-token"


def test_instruction_after_the_menu_contains_no_menu_identifier():
    """The pilot's phantom slot-23 choices: the model copied the instruction's own `X`."""
    tok = FakeTokenizer()
    cands = _cands(30)
    prompt = P.build_prompt([], cands, 8, 0, {}, _statements(30), tok,
                            max_prompt_tokens=10**6, with_history=False)
    P.assert_prompt_wellformed(prompt, cands, tok, 10**6)
    # Reintroducing a bare identifier into the trailing instruction must be caught.
    bad = prompt[: -len(P.CHOICE_TAIL)] + "where X is one of them.\n" + P.CHOICE_TAIL
    with pytest.raises(AssertionError, match="instruction after the menu"):
        P.assert_prompt_wellformed(bad, cands, tok, 10**6)
