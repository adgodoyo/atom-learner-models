"""Canaries for the zero-shot LLM selector runner (no GPU, no backbone, no substrate)."""

from __future__ import annotations

import pathlib
import re
import sys

import numpy as np
import pytest
import torch

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))

import ga_1_tier0_run as R  # noqa: E402


def test_comparator_panel_is_the_registered_one():
    """difficulty_matched is a registered blind comparator, not an optional extra."""
    assert R.COMPARATOR_POLICIES == ("rl_actor", "random", "max_entropy",
                                     "difficulty_matched", "privileged_oracle_upper_bound")
    assert set(R.BLIND_POLICIES) == {"random", "max_entropy", "difficulty_matched"}
    assert "privileged_oracle_upper_bound" not in R.BLIND_POLICIES
    assert R.BUDGETS == (3, 8, 16)


def test_every_comparator_is_implemented_by_lane_p():
    """Each policy name must be a branch of RL-2's unedited _trace_policy_v2."""
    import inspect

    from rl_2 import _trace_policy_v2
    source = inspect.getsource(_trace_policy_v2)
    for policy in R.COMPARATOR_POLICIES:
        assert f'"{policy}"' in source, policy


def test_substrate_mapping_rejects_unknown_and_duplicate_names():
    with pytest.raises(ValueError):
        R._mapping(["not-a-substrate=/tmp/x"], "--rl2-dir")
    with pytest.raises(ValueError):
        R._mapping(["ratchet-original=/tmp/a", "ratchet-original=/tmp/b"], "--rl2-dir")
    with pytest.raises(ValueError):
        R._mapping(["ratchet-original"], "--rl2-dir")
    got = R._mapping(["ratchet-original=results/x"], "--rl2-dir")
    assert got["ratchet-original"].is_absolute()


def _fake_episode(learner: str, n_cand: int = 4, n_target: int = 3):
    def ev(i, pid):
        return {"episode_id": f"{learner}-e{i}", "problem_id": pid, "order": i,
                "label": pid % 4, "codec": torch.arange(4, dtype=torch.float32) + i}
    prefix = [ev(0, 10), ev(1, 11)]
    cands = [ev(2 + i, 20 + i) for i in range(n_cand)]
    targets = [ev(50 + i, 30 + i) for i in range(n_target)]
    return {"learner": learner, "prefix_events": prefix, "candidates": cands,
            "targets": targets, "full_events": prefix + cands}


def test_geometry_digest_is_a_value_join_not_a_count(monkeypatch):
    a = {"val_start0": [_fake_episode("L1"), _fake_episode("L2")]}
    b = {"val_start0": [_fake_episode("L1"), _fake_episode("L3")]}
    assert R._geometry_digest(a, [0]) == R._geometry_digest(a, [0])
    # Same shape, same counts, different identities: the digest must separate them.
    assert R._geometry_digest(a, [0]) != R._geometry_digest(b, [0])


def _patch_primitives(monkeypatch):
    """A deterministic stand-in belief: state = sum of revealed problem ids."""
    def frozen_filter_state(model, events, query, device):
        return float(sum(e["problem_id"] for e in events)), 0.0

    def point_logits(model, mu, target_codecs, device):
        n = int(target_codecs.shape[0])
        return np.asarray([[mu / 100.0, -mu / 100.0, 0.0, 0.5]] * n, dtype=np.float64)

    monkeypatch.setattr(R, "_frozen_filter_state", frozen_filter_state)
    monkeypatch.setattr(R, "_point_logits", point_logits)


def test_score_fixed_picks_walks_the_chosen_sequence(monkeypatch):
    _patch_primitives(monkeypatch)
    ep = _fake_episode("L1", n_cand=4)
    picks = [ep["candidates"][2]["episode_id"], ep["candidates"][0]["episode_id"]]
    traces = R._score_fixed_picks(None, ep, picks, "cpu")
    assert traces["picks"] == picks
    assert traces["pick_problems"] == [22, 20]
    # One belief evaluation per step plus the terminal state.
    assert len(traces["dlogit"]) == len(picks) + 1
    assert all(np.isfinite(traces[m]).all() for m in R.METRICS)


def test_score_fixed_picks_rejects_a_repeated_or_off_menu_pick(monkeypatch):
    _patch_primitives(monkeypatch)
    ep = _fake_episode("L1", n_cand=3)
    first = ep["candidates"][0]["episode_id"]
    with pytest.raises(KeyError):
        R._score_fixed_picks(None, ep, [first, first], "cpu")
    with pytest.raises(KeyError):
        R._score_fixed_picks(None, ep, ["not-on-the-menu"], "cpu")


def test_partial_roundtrip_preserves_every_policy_and_metric(tmp_path):
    arrays = {p: {m: np.arange(5, dtype=np.float64) + i + j
                  for j, m in enumerate(R.METRICS)}
              for i, p in enumerate(R.ALL_POLICIES)}
    path = R._partial_path(tmp_path, "ratchet-cwsim/cold-head", 1, 3, 8)
    assert "/" not in path.name
    R._save_partial(path, arrays)
    back = R._load_partial(path)
    assert set(back) == set(R.ALL_POLICIES)
    for p in R.ALL_POLICIES:
        for m in R.METRICS:
            assert np.array_equal(back[p][m], arrays[p][m])


def test_numeric_rows_carry_their_own_name_tables(tmp_path):
    rows = [{"substrate": "ratchet-original", "seed": 1, "start": 0, "budget": 3,
             "policy": p, "learner_hash": 7, **{m: 0.5 for m in R.METRICS}}
            for p in R.ALL_POLICIES]
    path = tmp_path / "rows.npz"
    R._write_numeric_rows(path, rows)
    saved = np.load(path, allow_pickle=False)
    assert list(saved["policy_names"]) == list(R.ALL_POLICIES)
    assert list(saved["substrate_names"]) == ["ratchet-original"]
    assert len(saved["dlogit"]) == len(R.ALL_POLICIES)


def test_no_external_api_surface_in_this_lanes_sources():
    from ga_1_common import validate_no_external_api_surface
    validate_no_external_api_surface([
        REPO / "scripts/ga_1_tier0_run.py",
        REPO / "scripts/ga_1_tier0_prompt.py",
        REPO / "scripts/ga_1_common.py",
    ])


def test_rl2_reference_loader_rejects_an_incomplete_summary(tmp_path):
    import json
    d = tmp_path / "ratchet-original"
    d.mkdir()
    (d / "summary.json").write_text(json.dumps({"status": "running",
                                                "substrate": "ratchet-original", "cells": {}}))
    with pytest.raises(ValueError):
        R._load_rl2_reference({"ratchet-original": d}, [0], [3])


@pytest.mark.parametrize("substrate", R.SUBSTRATES)
def test_real_rl2_reference_has_every_cell_this_run_needs(substrate):
    """The stored RL-2 reference panel must already contain all five comparators."""
    d = REPO / "results" / "rl_2" / substrate.replace("/", "_")
    if not (d / "summary.json").exists():
        pytest.skip(f"RL-2 result not present for {substrate}")
    summaries, manifest = R._load_rl2_reference(
        {substrate: d}, [0, 3, 5, 10], list(R.BUDGETS))
    assert summaries[substrate]["status"] == "complete"
    assert len(manifest[substrate]["actor_sha256"]) == 3


def test_readout_is_constrained_to_the_live_menu(monkeypatch):
    """The read-out can never return an off-menu slot, whatever the vocabulary prefers."""
    import ga_1_tier0_readout as RO

    class FakeTok:
        padding_side = "right"
        truncation_side = "right"

        def __call__(self, batch, **kw):
            if isinstance(batch, str):
                return {"input_ids": [ord(batch[0])]}
            n = len(batch)
            return _FakeEnc({"input_ids": torch.zeros(n, 3, dtype=torch.long),
                             "attention_mask": torch.ones(n, 3, dtype=torch.long)})

    class _FakeEnc(dict):
        def to(self, device):
            return self

    vocab = 200
    # The single highest-logit token in the vocabulary is NOT an identifier at all.
    logits = torch.zeros(2, 3, vocab)
    logits[:, -1, 199] = 100.0
    logits[0, -1, ord("C")] = 5.0   # best identifier for row 0
    logits[1, -1, ord("B")] = 5.0   # best identifier for row 1

    class FakeModel:
        def __call__(self, **kw):
            class Out:
                pass
            out = Out()
            out.logits = logits
            return out

    bundle = {"model": FakeModel(), "tokenizer": FakeTok(), "device": "cpu"}
    ids = [ord(c) for c in "ABCDEF"]
    # Row 0 sees a 3-item menu (A,B,C); row 1 sees a 2-item menu (A,B).
    slots, diags = RO.read_out_choices(bundle, ["p0", "p1"], [3, 2], ids,
                                       batch_size=2, max_prompt_tokens=16)
    assert slots == [2, 1]
    for slot, n in zip(slots, (3, 2)):
        assert 0 <= slot < n
    assert all(0.0 <= d["menu_mass_vs_vocab"] <= 1.0 for d in diags)
    assert all(d["menu_entropy_nats"] >= 0.0 for d in diags)


def test_readout_rejects_a_multi_token_identifier():
    import ga_1_tier0_readout as RO

    class Tok:
        def __call__(self, s, add_special_tokens=False):
            return {"input_ids": [1, 2] if s == "B" else [1]}

    with pytest.raises(AssertionError, match="not single-token"):
        RO.identifier_token_ids(Tok(), ["A", "B"])


def test_only_the_oracle_is_cached_across_seeds():
    """Regression: caching max_entropy/difficulty_matched produced wrong traces.

    The harness seeds `eps`, the K-sample epistemic draws, from the CRN seed
    (`torch.Generator().manual_seed(seed)`), and both of those policies rank candidates on
    scores derived from `eps` -- so they are seed-dependent. An earlier version of this
    test counted `rng.integers` occurrences and passed while the premise was false; it was
    testing which rng is consumed, not whether the trace is the same. Assert the
    *dependency*, not the spelling.
    """
    import inspect

    from rl_2 import _trace_policy_v2
    source = inspect.getsource(_trace_policy_v2)
    # `eps` is seeded from the CRN seed ...
    assert re.search(r"eps\s*=\s*torch\.randn\(.*manual_seed\(seed\)", source, re.S)
    # ... and is fed to the scorer that max_entropy and difficulty_matched rank on.
    assert "_candidate_scores(model, mu, cands, targets, eps, device)" in source
    for seed_dependent in ("max_entropy", "difficulty_matched", "random", "rl_actor"):
        assert seed_dependent not in R.SEED_INVARIANT_POLICIES, seed_dependent
    # The oracle's branch reads neither eps nor rng -- it is the one safe entry.
    oracle = source[source.index('elif policy == "privileged_oracle_upper_bound"'):
                    source.index('elif policy == "rl_actor"')]
    assert "eps" not in oracle and "rng" not in oracle
    assert R.SEED_INVARIANT_POLICIES == ("privileged_oracle_upper_bound",)
