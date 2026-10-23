from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from ga_1_common import (  # noqa: E402
    LocalGenerationStore,
    NumericProgress,
    assert_numeric_upload_path,
    build_tier0_prompt,
    build_tier1_prompt,
    codec_neighbour_metrics,
    generated_manifold_metrics,
    parse_menu_choice,
    spearman_rank,
    truncate_code,
    validate_no_external_api_surface,
)


class WordTokenizer:
    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": str(text).split()}


def _history_fixture():
    event = {"episode_id": "7", "problem_id": 11}
    text = {"7": {"method_name": "foo", "statement": "Return x.", "n_attempts": 2,
                  "outcome_trace": ["COMPILE_ERROR", "CLEAN_FULL"],
                  "final_score": 1.0, "final_compile_result": "Success",
                  "final_code": "return x;"}}
    return event, text


def test_tier1_prompt_contains_revealed_history_but_no_menu_or_budget():
    event, episode_text = _history_fixture()
    prompt = build_tier1_prompt([event], episode_text, WordTokenizer())
    assert "return x;" in prompt
    assert "CURRENT MENU" not in prompt
    assert "TOTAL DIAGNOSTIC BUDGET" not in prompt


def test_tier0_prompt_contains_only_supplied_history_menu_and_budget():
    event, episode_text = _history_fixture()
    statements = {11: {"method_name": "foo", "statement": "Return x."}}
    prompt = build_tier0_prompt([event], [event], 8, 2, episode_text, statements,
                                WordTokenizer(), max_prompt_tokens=1000)
    assert "TOTAL DIAGNOSTIC BUDGET: 8" in prompt
    assert "CURRENT MENU" in prompt and "[1] foo" in prompt
    assert "return x;" in prompt


def test_menu_choice_parser_and_deterministic_fallback():
    assert parse_menu_choice("CHOICE=3", 5) == (2, "parsed")
    assert parse_menu_choice("[2]", 5) == (1, "parsed")
    assert parse_menu_choice("I refuse", 5) == (0, "fallback_first")
    assert parse_menu_choice("CHOICE=99", 5) == (0, "fallback_first")


def test_codec_neighbour_agreement_and_rank_correlation():
    menu = np.asarray([[0.0, 0.0], [1.0, 0.0], [3.0, 0.0]])
    got = codec_neighbour_metrics(np.asarray([0.9, 0.0]), menu, np.asarray([0.0, 3.0, 1.0]))
    assert got["actor_pick"] == 1
    assert got["nearest_pick"] == 1
    assert got["top1"] == got["top3"] == got["top5"] == 1
    assert got["rank_spearman"] > 0.4


def test_spearman_uses_average_tie_ranks():
    assert spearman_rank([1, 2, 3], [10, 20, 30]) == pytest.approx(1.0)
    assert np.isfinite(spearman_rank([1, 1, 2], [3, 2, 1]))


def test_manifold_support_uses_real_leave_one_out_p95():
    real = np.asarray([[0.0], [1.0], [2.0], [3.0]])
    generated = np.asarray([[0.1], [1.9], [50.0]])
    got = generated_manifold_metrics(generated, real)
    assert got["support_fraction_at_real_p95"] == pytest.approx(2 / 3)
    assert got["nearest_problem_index"].tolist() == [0, 2, 3]


def test_generated_text_store_is_append_only_and_resume_keyed(tmp_path):
    store = LocalGenerationStore(tmp_path / "generations.jsonl")
    store.append({"mode": "history", "learner": "a", "start": 3, "generated_text": "Q1"})
    store.append({"mode": "history", "learner": "b", "start": 5, "generated_text": "Q2"})
    loaded = store.load(("mode", "learner", "start"))
    assert loaded[("history", "a", "3")]["generated_text"] == "Q1"
    assert len((tmp_path / "generations.jsonl").read_text().splitlines()) == 2


def test_s3_upload_guard_rejects_generation_corpora():
    assert_numeric_upload_path("summary.json")
    assert_numeric_upload_path("agreement_rows.npz")
    with pytest.raises(ValueError, match="forbids"):
        assert_numeric_upload_path("generations.jsonl")
    with pytest.raises(ValueError, match="forbids"):
        assert_numeric_upload_path("raw_text.json")


def test_long_code_keeps_head_and_tail_with_explicit_marker():
    value = "a" * 100 + "b" * 100
    got = truncate_code(value, max_chars=60)
    assert got.startswith("a") and got.endswith("b")
    assert "truncated" in got


def test_numeric_progress_enforces_preregistered_stage_overrun(tmp_path, monkeypatch):
    clock = [1000.0]
    monkeypatch.setattr("ga_1_common.time.time", lambda: clock[0])
    progress = NumericProgress(tmp_path, overrun_minutes=5)
    progress.heartbeat("generate", 0, 100)
    clock[0] += 61.0
    with pytest.raises(RuntimeError, match="preregistered 5-minute"):
        progress.heartbeat("generate", 8, 100, enforce_overrun=True)


def test_external_api_canary_scans_itself_without_self_reference(tmp_path):
    validate_no_external_api_surface([REPO / "scripts" / "ga_1_common.py"])
    bad = tmp_path / "bad.py"
    bad.write_text("import openai\n")
    with pytest.raises(RuntimeError, match="external-API surface"):
        validate_no_external_api_surface([bad])
