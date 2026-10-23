"""FA-6 gate (ii): the AKT-matched arm. `build_matched_arm_filter_batch`/
`remap_t_idx_to_matched_arm` are pure data-prep (no model, tested directly against a
synthetic manifest); `success_eval`/`matched_arm_success_eval` need a forward pass, tested
against the same CausalStubLM + attach_lora_basis pipeline `test_fa_06_stage2_train.py`
and `test_fa_train_step.py` already use.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
import torch

from atom.binding import Hypernetwork, attach_lora_basis
from atom.binding.stub_causal_lm import CausalStubLM
from atom.filter.batch import build_padded_batch_v2, load_production_codec  # noqa: F401 (load_production_codec unused, kept for parity with prod imports)
from atom.filter.factory import build_filter
from atom.heads.fa_alt2_heads import SuccessHead
from atom.simulator.fa_akt_matched_arm import (build_matched_arm_filter_batch,
                                               matched_arm_success_eval,
                                               remap_t_idx_to_matched_arm, success_eval)
from atom.simulator.fa_alt2 import encode_example
from atom.simulator.fa_train_step import build_targets

CODEC_DIM = 5
LATENT_DIM = 8
HIDDEN_DIM = 32
FA_MATCHED_ARM_BODIES = ("gru", "c_distance_time")


def _toy_manifest() -> pd.DataFrame:
    """Two learners, each with 4 chronological events: score-missing at t=0,2 and a real
    score at t=1,3 -- so exclude_score_missing_from_history should leave exactly the
    t=1/t=3 events per learner, resequenced to t'=0/1. Learner "a" scores 1.0 both times,
    learner "b" scores 0.0 both times -- a MIXED target (not single-class), so AUC is a
    meaningful (non-NaN) number, not just a metric that happens not to crash."""
    base = pd.Timestamp("2026-01-01T00:00:00Z")
    rows = []
    for learner, learner_score in (("a", 1.0), ("b", 0.0)):
        for t in range(4):
            missing = t % 2 == 0
            rows.append({
                "learner": learner, "id": f"{learner}-{t}", "t": t,
                "timestamp": base + pd.Timedelta(minutes=10 * t),
                "problem_id": 100 + (t % 2), "score": np.nan if missing else learner_score,
                "score_missing": missing, "hint_count": 0, "saw_answer": False,
                "has_valid_time": True, "support_code": 0,
            })
    return pd.DataFrame(rows)


def _codec():
    codec_matrix = np.random.default_rng(0).normal(size=(2, CODEC_DIM)).astype(np.float32)
    pid_to_idx = {100: 0, 101: 1}
    return codec_matrix, pid_to_idx


def test_build_matched_arm_filter_batch_drops_score_missing_and_resequences_t():
    ev = _toy_manifest()
    codec_matrix, pid_to_idx = _codec()

    filter_batch, matched_ev = build_matched_arm_filter_batch(ev, ["a", "b"], codec_matrix, pid_to_idx)

    # 2 kept events per learner (t=1,3 originally), each learner's own new t = 0, 1.
    assert sorted(matched_ev["learner"].unique()) == ["a", "b"]
    for learner in ("a", "b"):
        sub = matched_ev[matched_ev["learner"] == learner].sort_values("t")
        assert sub["t"].tolist() == [0, 1]
        assert sub["id"].tolist() == [f"{learner}-1", f"{learner}-3"]
    assert filter_batch["learners"] == ["a", "b"]
    assert filter_batch["valid_mask"].sum().item() == 4  # 2 learners x 2 kept events


def test_remap_t_idx_to_matched_arm_maps_kept_rows_and_flags_dropped_ones():
    ev = _toy_manifest()
    codec_matrix, pid_to_idx = _codec()
    _filter_batch, matched_ev = build_matched_arm_filter_batch(ev, ["a", "b"], codec_matrix, pid_to_idx)

    # examples_df mimics a val bundle row set: ALL 8 original events (score-missing
    # included), same convention infra/fa_06_stage2_train.py's own examples_df uses.
    row_has_match, new_t = remap_t_idx_to_matched_arm(ev, matched_ev)

    for i, row in ev.reset_index(drop=True).iterrows():
        if row["score_missing"]:
            assert row_has_match[i] == False  # noqa: E712 -- explicit bool, not just falsy
        else:
            assert row_has_match[i] == True  # noqa: E712
            # t=1 -> new t'=0, t=3 -> new t'=1 (per learner)
            assert new_t[i] == (0 if row["t"] == 1 else 1)


def _build_matched_arm_pipeline(body: str, seed: int = 0):
    tok = _FakeTok()
    model = CausalStubLM(vocab_size=64, dim=HIDDEN_DIM, n_layers=2, n_heads=4,
                         pad_token_id=tok.pad_token_id, seed=seed)
    bases, context = attach_lora_basis(model, m=8, r=4, seed=seed + 1)
    hyper = Hypernetwork(u_dim=LATENT_DIM, m=8, kind="mlp", seed=seed + 2)
    filt, filt_forward = build_filter(body, codec_dim=CODEC_DIM, episode_hidden=8,
                                      filter_hidden=8, latent_dim=LATENT_DIM,
                                      n_heads=2, d_ff=8, n_rbf=4)
    success_head = SuccessHead(HIDDEN_DIM)
    return tok, model, bases, context, hyper, filt, filt_forward, success_head


class _FakeTok:
    pad_token_id = 0
    eos_token_id = 63

    def __call__(self, text: str, add_special_tokens: bool = True) -> dict:
        words = [w for w in text.replace("\n", " \n ").split(" ") if w != ""]
        ids = [(sum(ord(c) for c in w) % 40) + 2 for w in words]
        if add_special_tokens:
            ids = [1, *ids]
        return {"input_ids": ids}


def _scored_rows_examples(ev: pd.DataFrame, tok, max_len: int = 16):
    """Build examples/examples_df/targets for the NON-score-missing rows only (the ones
    with a defined success target) -- matches how `build_examples` + `build_targets`
    already work in infra/fa_06_stage2_train.py, just inlined for a tiny synthetic set."""
    scored = ev[~ev["score_missing"]].reset_index(drop=True)
    examples = [encode_example(f"Question {row.id}", "answer", tok, max_len=max_len)
               for row in scored.itertuples()]
    targets = build_targets(scored, device="cpu")
    return examples, scored, targets


@pytest.mark.parametrize("body", FA_MATCHED_ARM_BODIES)
def test_success_eval_produces_finite_metrics(body):
    ev = _toy_manifest()
    tok, model, bases, context, hyper, filt, filt_forward, success_head = \
        _build_matched_arm_pipeline(body)
    codec_matrix, pid_to_idx = _codec()
    native_filter_batch = build_padded_batch_v2(ev, codec_matrix, pid_to_idx, device="cpu", include_time_min=True)
    from atom.filter.episode_v2 import build_prev_feat_v2
    native_filter_batch["prev_feat"] = build_prev_feat_v2(
        native_filter_batch["correct"], native_filter_batch["hint_bin"],
        native_filter_batch["saw_answer"], native_filter_batch["score_missing"])

    examples, scored, targets = _scored_rows_examples(ev, tok)
    learner_to_idx = {u: i for i, u in enumerate(native_filter_batch["learners"])}
    learner_idx = torch.tensor([learner_to_idx[u] for u in scored["learner"]])
    t_idx = torch.tensor(scored["t"].to_numpy().astype("int64"))

    metrics = success_eval(filt_forward, hyper, model, context, success_head,
                           native_filter_batch, learner_idx, t_idx, examples, targets,
                           pad_id=tok.pad_token_id, device="cpu", batch_size=2)

    assert math.isfinite(metrics["success_logloss"]["value"])
    assert metrics["success_logloss"]["n_eval_population"] == 4  # 2 learners x 2 scored rows
    # AUC/Brier exist specifically so this is comparable to AKT's own reported metric set
    # (AUC/acc/ECE/Brier, reports/fa_01_akt_v2.md) -- not just log-loss/ECE, which AKT's
    # report never states.
    assert math.isfinite(metrics["success_auc"]["value"])
    assert 0.0 <= metrics["success_auc"]["value"] <= 1.0
    assert math.isfinite(metrics["success_brier"]["value"])


@pytest.mark.parametrize("body", FA_MATCHED_ARM_BODIES)
def test_matched_arm_success_eval_reports_both_arms_and_akt_numbers(body):
    ev = _toy_manifest()
    tok, model, bases, context, hyper, filt, filt_forward, success_head = \
        _build_matched_arm_pipeline(body)
    codec_matrix, pid_to_idx = _codec()
    native_filter_batch = build_padded_batch_v2(ev, codec_matrix, pid_to_idx, device="cpu", include_time_min=True)
    from atom.filter.episode_v2 import build_prev_feat_v2
    native_filter_batch["prev_feat"] = build_prev_feat_v2(
        native_filter_batch["correct"], native_filter_batch["hint_bin"],
        native_filter_batch["saw_answer"], native_filter_batch["score_missing"])

    examples, scored, targets = _scored_rows_examples(ev, tok)

    out = matched_arm_success_eval(filt_forward, hyper, model, context, success_head,
                                   native_filter_batch, ev, examples, scored, targets,
                                   codec_matrix, pid_to_idx, pad_id=tok.pad_token_id,
                                   device="cpu", batch_size=2)

    for arm in ("native_arm", "matched_arm"):
        assert math.isfinite(out[arm]["success_logloss"]["value"]), arm
        assert out[arm]["success_logloss"]["n_eval_population"] == 4, arm
        assert math.isfinite(out[arm]["success_auc"]["value"]), arm
        assert math.isfinite(out[arm]["success_brier"]["value"]), arm
    assert out["akt_v2_marginal_val_auc"] == 0.8016
    assert out["akt_v2_n_excluded_score_missing"] == 6980
