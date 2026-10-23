"""Gradient-reach and row-identity canaries for the CodeWorkout training loop.

A `.detach().cpu().numpy()` of the episode summary before it is scattered into the
cross-episode filter's `prev_feat` tensor would sever the graph between
`AttemptEncoder.summary_head` and every downstream loss: forwards run, losses fall, and the
module never leaves its random initialisation (defect 1). These tests therefore assert
finite non-zero gradients AND that parameters change after an optimiser step, per named
module -- not merely that `loss.backward()` ran or that some gradient exists somewhere.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from atom.filter.cw_attempt_encoder import AttemptEncoder
from atom.filter.factory import build_filter
from atom.heads.fa_hierarchical_heads import OutcomeHead
from atom.simulator.cw_4_train_loop import (
    assert_row_identity_join_canary,
    evaluate_val,
    forward_batch,
    row_identity,
    train_one_body,
)
from atom.simulator.cw_train_step import losses_for_cw_hierarchical_batch, y_ever_loss_for_batch

CODEC_DIM = 4
LATENT_DIM = 3
EPISODE_HIDDEN = 5
FILTER_HIDDEN = 5
ATTEMPT_PROJ_DIM = 3
ATTEMPT_HIDDEN = 4
SUMMARY_DIM = 2
N_CLASSES = 4


def _two_learners_two_episodes_each():
    """Two learners, two episodes each, 1-2 attempts per episode -- enough for the
    cross-episode filter to actually read a PREVIOUS episode's summary (the exact
    tensor the severed graph made unreachable)."""
    attempts = pd.DataFrame({
        "episode_id": [0, 0, 1, 2, 3, 3],
        "learner": ["A", "A", "A", "B", "B", "B"],
        "problem_id": [10, 10, 11, 10, 11, 11],
        "code_hash": ["h0", "h1", "h2", "h3", "h4", "h5"],
        "diff_hash": [np.nan, "d1", np.nan, np.nan, np.nan, "d5"],
        "compile_result": ["Error", "Success", "Success", "Success", "Error", "Success"],
        "score_frac": [0.0, 1.0, 0.5, 1.0, 0.0, 1.0],
        "n_compile_errors": [1, 0, 0, 0, 1, 0],
        "outcome_code": [0, 3, 2, 3, 0, 3],
        "t": [0, 1, 2, 0, 1, 2],
        "release": ["S19", "S19", "S19", "F19", "F19", "F19"],
        "code_state_id": ["cs0", "cs1", "cs2", "cs3", "cs4", "cs5"],
    })
    episodes = pd.DataFrame({
        "episode_id": [0, 1, 2, 3],
        "learner": ["A", "A", "B", "B"],
        "problem_id": [10, 11, 10, 11],
        "start_timestamp": pd.to_datetime(["2024-01-01T00:00", "2024-01-01T01:00",
                                           "2024-01-01T00:00", "2024-01-01T02:00"]),
        "y_ever": [True, True, True, True],
    })
    code_of_hash = {f"h{i}": np.random.RandomState(i).randn(CODEC_DIM).astype(np.float32) for i in range(6)}
    diff_of_hash = {"d1": np.random.RandomState(99).randn(CODEC_DIM).astype(np.float32),
                    "d5": np.random.RandomState(98).randn(CODEC_DIM).astype(np.float32)}
    problem_codec = np.random.RandomState(7).randn(2, CODEC_DIM).astype(np.float32)
    pid_to_idx = {10: 0, 11: 1}
    return attempts, episodes, code_of_hash, diff_of_hash, problem_codec, pid_to_idx


def _build_model(device="cpu"):
    torch.manual_seed(0)
    attempt_encoder = AttemptEncoder(CODEC_DIM, CODEC_DIM, ATTEMPT_PROJ_DIM, ATTEMPT_HIDDEN, SUMMARY_DIM).to(device)
    filt, filt_forward = build_filter("gru", codec_dim=CODEC_DIM, episode_hidden=EPISODE_HIDDEN,
                                      filter_hidden=FILTER_HIDDEN, latent_dim=LATENT_DIM,
                                      prev_feat_dim=SUMMARY_DIM)
    filt.to(device)
    outcome_head = OutcomeHead(codec_dim=CODEC_DIM, latent_dim=LATENT_DIM, hidden=ATTEMPT_HIDDEN,
                               n_classes=N_CLASSES, prefix_dim=ATTEMPT_HIDDEN).to(device)
    y_ever_head = OutcomeHead(codec_dim=CODEC_DIM, latent_dim=LATENT_DIM, hidden=ATTEMPT_HIDDEN,
                              n_classes=1).to(device)
    return attempt_encoder, filt, filt_forward, outcome_head, y_ever_head


def test_summary_head_receives_finite_nonzero_gradient_and_its_parameters_move():
    """The canary itself: one full `forward_batch`
    backward, then one optimizer step, checked on `AttemptEncoder.summary_head`'s OWN
    named parameters -- not the module's total loss, not any other module's gradient."""
    device = "cpu"
    attempts, episodes, code_of_hash, diff_of_hash, problem_codec, pid_to_idx = _two_learners_two_episodes_each()
    attempt_encoder, filt, filt_forward, outcome_head, y_ever_head = _build_model(device)

    optimizer = torch.optim.Adam(
        list(attempt_encoder.parameters()) + list(filt.parameters())
        + list(outcome_head.parameters()) + list(y_ever_head.parameters()), lr=1e-2)

    before = {name: p.detach().clone() for name, p in attempt_encoder.summary_head.named_parameters()}
    assert len(before) > 0, "test fixture bug: summary_head has no parameters to check"

    optimizer.zero_grad()
    out = forward_batch(attempt_encoder, filt_forward, outcome_head, y_ever_head, attempts, episodes,
                        code_of_hash, diff_of_hash, problem_codec, pid_to_idx, device,
                        latent_dim=LATENT_DIM, codec_dim=CODEC_DIM)
    total = out["l_outcome"] + out["l_y_ever"]
    assert torch.isfinite(total)
    total.backward()

    for name, p in attempt_encoder.summary_head.named_parameters():
        assert p.grad is not None, f"summary_head.{name} received NO gradient at all (the exact defect 1 signature)"
        assert torch.isfinite(p.grad).all(), f"summary_head.{name} gradient has non-finite entries"
        assert p.grad.abs().sum().item() > 0.0, f"summary_head.{name} gradient is all-zero"

    optimizer.step()
    for name, p in attempt_encoder.summary_head.named_parameters():
        assert not torch.allclose(before[name], p.detach()), (
            f"summary_head.{name} did not change after an optimizer step -- gradient existed "
            "but did not translate into a real parameter update")


def test_every_trainable_module_receives_gradient_not_just_summary_head():
    """The general form of the canary: every module that should be trainable IS,
    checked per named module."""
    device = "cpu"
    attempts, episodes, code_of_hash, diff_of_hash, problem_codec, pid_to_idx = _two_learners_two_episodes_each()
    attempt_encoder, filt, filt_forward, outcome_head, y_ever_head = _build_model(device)

    modules = {"attempt_encoder": attempt_encoder, "filt": filt,
              "outcome_head": outcome_head, "y_ever_head": y_ever_head}
    optimizer = torch.optim.Adam(
        [p for m in modules.values() for p in m.parameters()], lr=1e-2)

    optimizer.zero_grad()
    out = forward_batch(attempt_encoder, filt_forward, outcome_head, y_ever_head, attempts, episodes,
                        code_of_hash, diff_of_hash, problem_codec, pid_to_idx, device,
                        latent_dim=LATENT_DIM, codec_dim=CODEC_DIM)
    total = out["l_outcome"] + out["l_y_ever"]
    total.backward()

    for mod_name, module in modules.items():
        for name, p in module.named_parameters():
            assert p.grad is not None, f"{mod_name}.{name} received no gradient"
            assert torch.isfinite(p.grad).all(), f"{mod_name}.{name} gradient has non-finite entries"


def test_train_one_body_emits_progress_and_full_resume_state_additively():
    attempts, episodes, code_of_hash, diff_of_hash, problem_codec, pid_to_idx = _two_learners_two_episodes_each()
    manifest = {
        "attempts": attempts,
        "episodes": episodes,
        "code_of_hash": code_of_hash,
        "diff_of_hash": diff_of_hash,
        "problem_codec": problem_codec,
        "pid_to_idx": pid_to_idx,
        "train_learners": {"A", "B"},
    }
    seen = []
    model = train_one_body(
        "gru",
        manifest,
        SimpleNamespace(lr=1e-3, learner_batch_size=1, epochs=1),
        "cpu",
        seed=0,
        dims={
            "codec_dim": CODEC_DIM,
            "latent_dim": LATENT_DIM,
            "episode_hidden": EPISODE_HIDDEN,
            "filter_hidden": FILTER_HIDDEN,
            "attempt_proj_dim": ATTEMPT_PROJ_DIM,
            "attempt_hidden": ATTEMPT_HIDDEN,
            "summary_dim": SUMMARY_DIM,
        },
        progress_callback=lambda *args: seen.append(args),
    )
    assert [entry[0] for entry in seen] == [1, 2]
    assert all(entry[1] == 2 and entry[2] == 1 for entry in seen)
    for key in (
        "optimizer_state_dict", "torch_rng_state", "torch_cuda_rng_state_all",
        "numpy_rng_state", "python_rng_state",
    ):
        assert key in model


# ---------------------------------------------------------------------------------------
# Row-identity persistence (defect 2). Using an EPISODE index (0..n_episodes-1) as an
# attempts-table ROW LABEL (`val_attempts.loc[attempt_batch["attempt_episode_row_idx"],
# "problem_id"]`) is the error guarded against here -- exactly the
# shape these tests exercise: 6 attempts, only 4 episodes, so episode index 4 or 5 would
# either KeyError or (worse, on a differently-shaped real corpus) silently hit an
# unrelated row. The fixture below is chosen so n_attempts != n_episodes for this reason.
# ---------------------------------------------------------------------------------------

def _six_attempts_four_episodes_identity_fixture():
    """Learner A revisits problem 10 in a LATER, separate episode (episode 2) -- the
    case `prior_same_problem_count` must count ACROSS episode boundaries, not just
    within one episode's own attempts."""
    return pd.DataFrame({
        "episode_id": [0, 0, 1, 2, 3, 3],
        "learner": ["A", "A", "A", "A", "B", "B"],
        "problem_id": [10, 10, 11, 10, 20, 20],
        "t": [0, 1, 2, 3, 0, 1],
        "release": ["S19", "S19", "S19", "S19", "F19", "F19"],
        "code_state_id": ["cs0", "cs1", "cs2", "cs3", "cs4", "cs5"],
    })


def test_row_identity_reads_source_columns_directly_not_by_episode_index():
    """The structural regression test for defect 2: with n_episodes (4) != n_attempts
    (6), an episode-index-as-row-label bug would produce either an out-of-range error or
    silently wrong values at the rows beyond the episode count. `row_identity` must
    reproduce the source columns EXACTLY, in the source's own row order."""
    df = _six_attempts_four_episodes_identity_fixture()
    identity = row_identity(df, split_flag="val")
    assert identity["learner"].tolist() == df["learner"].tolist()
    assert identity["episode_id"].tolist() == df["episode_id"].tolist()
    assert identity["problem_id"].tolist() == df["problem_id"].tolist()
    assert identity["release"].tolist() == df["release"].tolist()
    assert identity["attempt_id"].tolist() == df["code_state_id"].tolist()
    assert (identity["split_flag"] == "val").all()


def test_prior_same_problem_count_counts_across_episode_boundaries():
    df = _six_attempts_four_episodes_identity_fixture()
    identity = row_identity(df, split_flag="val")
    # learner A, problem 10: attempts at t=0 (1st, count=0), t=1 (2nd, count=1),
    # t=3 in a LATER episode (3rd, count=2) -- t=2 is a DIFFERENT problem (11, count=0).
    counts = dict(zip(zip(df["learner"], df["t"]), identity["prior_same_problem_count"]))
    assert counts[("A", 0)] == 0
    assert counts[("A", 1)] == 1
    assert counts[("A", 2)] == 0  # problem 11, first (and only) attempt
    assert counts[("A", 3)] == 2  # problem 10 again, third attempt overall
    assert counts[("B", 0)] == 0
    assert counts[("B", 1)] == 1


def test_join_canary_passes_on_correct_identity():
    df = _six_attempts_four_episodes_identity_fixture()
    identity = row_identity(df, split_flag="val")
    assert_row_identity_join_canary(df, identity, n_sample=6)  # sample every row


def test_join_canary_fails_on_corrupted_identity():
    """Directly reproduces the historical bug's SIGNATURE (right size, wrong membership)
    -- the canary must catch a shifted/misaligned identity array even though every field
    still has the correct length and dtype."""
    df = _six_attempts_four_episodes_identity_fixture()
    identity = row_identity(df, split_flag="val")
    corrupted = dict(identity)
    corrupted["problem_id"] = np.roll(identity["problem_id"], 1)  # same size, wrong membership
    with pytest.raises(AssertionError):
        assert_row_identity_join_canary(df, corrupted, n_sample=6)


def test_evaluate_val_end_to_end_problem_id_matches_source_not_episode_index():
    """Integration regression test: runs the FULL `evaluate_val` path (the actual
    production call site defect 2 lived in) on a fixture where n_episodes != n_attempts,
    and asserts the returned `problem_id`/`learner`/`episode_id` arrays match the val
    attempts table's OWN columns exactly, in order -- not shifted, not truncated, not
    resolved through an unrelated index."""
    device = "cpu"
    attempts, episodes, code_of_hash, diff_of_hash, problem_codec, pid_to_idx = _two_learners_two_episodes_each()
    attempt_encoder, filt, filt_forward, outcome_head, y_ever_head = _build_model(device)
    model = {"attempt_encoder": attempt_encoder, "filt": filt, "filt_forward": filt_forward,
            "outcome_head": outcome_head, "y_ever_head": y_ever_head}
    manifest = {"attempts": attempts, "episodes": episodes, "code_of_hash": code_of_hash,
               "diff_of_hash": diff_of_hash, "problem_codec": problem_codec, "pid_to_idx": pid_to_idx,
               "val_learners": {"A", "B"}}

    val = evaluate_val(model, manifest, device, latent_dim=LATENT_DIM, codec_dim=CODEC_DIM)

    assert val["problem_id"].tolist() == attempts["problem_id"].tolist()
    assert val["learner"].tolist() == attempts["learner"].tolist()
    assert val["episode_id"].tolist() == attempts["episode_id"].tolist()
    assert val["attempt_id"].tolist() == attempts["code_state_id"].tolist()
