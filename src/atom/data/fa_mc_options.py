"""Single-select multiple-choice items: the primary population for evaluating which
distractor a learner picks. `Problems.csv`'s `mc_options`/`mc_answers` columns (`` || ``-
delimited) are joined against each interaction's own `answer_text` (the learner's actual
recorded response) to identify: which option was CORRECT, and which option the learner
ACTUALLY SELECTED -- the second is what makes the wrong-answer stratum measurable at all
(predicting not only failure but its shape).

Matching a free-text `answer_text` back to one of the option strings is imperfect (HTML
entities, trailing punctuation, single-letter/digit UI shorthand for "option A"/"option
1"). Measured on the real corpus: normalise (unescape HTML, strip tags, collapse
whitespace, lowercase, drop a trailing period) plus a positional letter/digit fallback
gets **81.4%** of the 336,705 single-select-MC interactions matched to a definite option
index; the remainder is EXCLUDED, not guessed -- `response_option_index = -1` /
`response_matched = False` disclosed on every row, never silently dropped from the count.

Per FA's licence rule: raw question/option/response TEXT stays on the local machine and
is never written into a committed artifact -- this module returns
in-memory DataFrames only, exactly like `atom.data.fa`/`atom.data.fa_manifest` already do;
callers may compute and persist AGGREGATE statistics (counts, match rates) but not the
text itself.
"""

from __future__ import annotations

import html
import re

import pandas as pd

MC_OPTION_DELIM = " || "


def _normalize_mc_text(s: object) -> str:
    """HTML-unescape, strip tags, collapse whitespace, lowercase, drop a trailing period
    -- enough to close most of the gap between a learner's typed response and the option
    string it corresponds to, without attempting anything more aggressive (e.g. fuzzy
    matching) that could silently pair a response with the WRONG option."""
    if not isinstance(s, str):
        return ""
    s = html.unescape(s)
    s = re.sub(r"<[^>]+>", "", s)
    s = " ".join(s.split()).strip().lower()
    return s.rstrip(".")


def _match_text_to_option(text_norm: str, options_norm: list[str]) -> int:
    """Index into `options_norm`, or -1 if no match. Exact match first; falls back to a
    single letter (a/b/c/... -> option 0/1/2/...) or a 1-indexed digit ("2" -> option 1)
    ONLY when that fallback lands in range -- covers UI shorthand where a learner's raw
    response is just the option's letter/number label rather than its text."""
    try:
        return options_norm.index(text_norm)
    except ValueError:
        pass
    if len(text_norm) == 1 and text_norm.isalpha():
        idx = ord(text_norm) - ord("a")
        if 0 <= idx < len(options_norm):
            return idx
    if text_norm.isdigit():
        idx = int(text_norm) - 1
        if 0 <= idx < len(options_norm):
            return idx
    return -1


def extract_single_select_mc(problems: pd.DataFrame, interactions: pd.DataFrame) -> pd.DataFrame:
    """Join `interactions` against `problems`' single-select MC items only (`mc_answers`
    names exactly one option -- multi-select items, ~380 of the corpus's 1,175 MC
    problems, are out of scope for this primary population).

    Returns one row per matching interaction, with `interactions`' own columns plus:
      - `options`: list[str], the RAW (un-normalised) option strings, in their original
        order -- this is what a caller scores under the LM, never the normalised text.
      - `correct_option_index`: index into `options` of the item's correct answer.
      - `response_option_index`: index into `options` of the learner's recorded response,
        or -1 if `_match_text_to_option` could not resolve it.
      - `response_matched`: `response_option_index >= 0`, so callers can filter/report the
        exclusion rate explicitly rather than it being implicit in a row count.
    """
    mc = problems[problems["mc_options"].notna()].copy()
    n_answers = mc["mc_answers"].str.split(MC_OPTION_DELIM, regex=False).str.len()
    single = mc[n_answers == 1].copy()
    single["options"] = single["mc_options"].str.split(MC_OPTION_DELIM, regex=False)
    single["_correct_text"] = single["mc_answers"]

    joined = interactions.merge(
        single[["problem_id", "options", "_correct_text"]], on="problem_id", how="inner")
    options_norm = joined["options"].apply(lambda opts: [_normalize_mc_text(o) for o in opts])
    joined["correct_option_index"] = [
        _match_text_to_option(_normalize_mc_text(t), o)
        for t, o in zip(joined["_correct_text"], options_norm)
    ]
    joined["response_option_index"] = [
        _match_text_to_option(_normalize_mc_text(t), o)
        for t, o in zip(joined["answer_text"], options_norm)
    ]
    joined["response_matched"] = joined["response_option_index"] >= 0
    return joined.drop(columns=["_correct_text"])
