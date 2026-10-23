"""FA-6 gate (iii): single-select MC extraction. Synthetic problems/interactions only --
this project's licence rule keeps real FA question/option/response text off any committed
artifact, tests included.
"""

from __future__ import annotations

import pandas as pd

from atom.data.fa_mc_options import (_match_text_to_option, _normalize_mc_text,
                                     extract_single_select_mc)


def _problems():
    return pd.DataFrame({
        "problem_id": [1, 2, 3, 4],
        "mc_options": [
            "Alpha || Beta || Gamma",       # single-select
            "One || Two",                    # single-select
            "X || Y || Z",                   # multi-select (2 correct) -- excluded
            None,                             # not MC at all -- excluded
        ],
        "mc_answers": ["Beta", "One", "X || Z", None],
    })


def _interactions():
    return pd.DataFrame({
        "id": [10, 11, 12, 13, 14],
        "problem_id": [1, 1, 2, 3, 1],
        "answer_text": ["Beta", "b", "Two", "X", "Something else entirely"],
    })


def test_normalize_mc_text_strips_html_and_trailing_period():
    assert _normalize_mc_text("<p>Beta.</p>") == "beta"
    assert _normalize_mc_text("  Two  ") == "two"
    assert _normalize_mc_text(None) == ""


def test_match_text_to_option_exact_then_letter_then_digit_fallback():
    options = ["alpha", "beta", "gamma"]
    assert _match_text_to_option("beta", options) == 1
    assert _match_text_to_option("b", options) == 1       # letter shorthand
    assert _match_text_to_option("3", options) == 2        # 1-indexed digit shorthand
    assert _match_text_to_option("nonsense", options) == -1


def test_extract_single_select_mc_excludes_multi_select_and_non_mc_problems():
    out = extract_single_select_mc(_problems(), _interactions())
    # problem 3 (multi-select) and problem 4 (no mc_options) never appear -- only
    # problems 1/2's interactions (ids 10, 11, 14 for problem 1; id 12 for problem 2).
    assert set(out["problem_id"]) == {1, 2}
    assert set(out["id"]) == {10, 11, 12, 14}


def test_extract_single_select_mc_correct_option_index_matches_the_answer():
    out = extract_single_select_mc(_problems(), _interactions())
    row1 = out[out["problem_id"] == 1].iloc[0]
    assert row1["options"] == ["Alpha", "Beta", "Gamma"]
    assert row1["correct_option_index"] == 1  # "Beta"
    row2 = out[out["problem_id"] == 2].iloc[0]
    assert row2["correct_option_index"] == 0  # "One"


def test_extract_single_select_mc_response_matching_exact_and_fallback_and_miss():
    out = extract_single_select_mc(_problems(), _interactions()).set_index("id")
    assert out.loc[10, "response_option_index"] == 1       # "Beta" -> exact
    assert out.loc[10, "response_matched"]
    assert out.loc[11, "response_option_index"] == 1       # "b" -> letter fallback
    assert out.loc[11, "response_matched"]
    assert out.loc[14, "response_option_index"] == -1      # "Something else entirely" -> no match
    assert not out.loc[14, "response_matched"]


def test_extract_single_select_mc_returns_raw_not_normalised_option_text():
    """Callers score the RAW option text under the LM -- normalisation is match-only,
    never a substitute for the real text a personalised model would actually see."""
    out = extract_single_select_mc(_problems(), _interactions())
    row1 = out[out["problem_id"] == 1].iloc[0]
    assert row1["options"] == ["Alpha", "Beta", "Gamma"]  # original case preserved
