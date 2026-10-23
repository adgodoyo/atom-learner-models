import numpy as np
import pandas as pd
import torch

from atom.filter.cw_filter_batch import build_attempt_batch, build_episode_batch

CODE_DIM = 3
DIFF_DIM = 3


def _attempts_two_episodes():
    """Episode 0 (learner A, problem 10): 2 attempts, error then clean-full.
    Episode 1 (learner A, problem 11): 1 attempt, clean-partial."""
    return pd.DataFrame({
        "episode_id": [0, 0, 1],
        "code_hash": ["h0", "h1", "h2"],
        "diff_hash": [np.nan, "d1", np.nan],
        "compile_result": ["Error", "Success", "Success"],
        "score_frac": [0.0, 1.0, 0.5],
        "n_compile_errors": [1, 0, 0],
    })


def _code_diff_lookups():
    code_of_hash = {"h0": np.array([1.0, 0.0, 0.0]), "h1": np.array([0.0, 1.0, 0.0]),
                    "h2": np.array([0.0, 0.0, 1.0])}
    diff_of_hash = {"d1": np.array([9.0, 9.0, 9.0])}
    return code_of_hash, diff_of_hash


def test_build_attempt_batch_shapes_and_lengths():
    attempts = _attempts_two_episodes()
    code_of_hash, diff_of_hash = _code_diff_lookups()
    out = build_attempt_batch(attempts, code_of_hash, diff_of_hash, CODE_DIM, DIFF_DIM, device="cpu")
    assert out["episode_ids"] == [0, 1]
    assert out["code_vec"].shape == (2, 2, CODE_DIM)  # T_max=2 (episode 0 has 2 attempts)
    assert out["lengths"].tolist() == [2, 1]


def test_build_attempt_batch_code_vectors_land_at_correct_positions():
    attempts = _attempts_two_episodes()
    code_of_hash, diff_of_hash = _code_diff_lookups()
    out = build_attempt_batch(attempts, code_of_hash, diff_of_hash, CODE_DIM, DIFF_DIM, device="cpu")
    # episode 0 (row 0): attempt 0 -> h0, attempt 1 -> h1
    assert torch.allclose(out["code_vec"][0, 0], torch.tensor([1.0, 0.0, 0.0]))
    assert torch.allclose(out["code_vec"][0, 1], torch.tensor([0.0, 1.0, 0.0]))
    # episode 1 (row 1): attempt 0 -> h2
    assert torch.allclose(out["code_vec"][1, 0], torch.tensor([0.0, 0.0, 1.0]))


def test_build_attempt_batch_missing_diff_hash_is_zero_vector():
    attempts = _attempts_two_episodes()
    code_of_hash, diff_of_hash = _code_diff_lookups()
    out = build_attempt_batch(attempts, code_of_hash, diff_of_hash, CODE_DIM, DIFF_DIM, device="cpu")
    # episode 0, attempt 0 has diff_hash=NaN (episode-first attempt) -> zero diff vector
    assert torch.allclose(out["diff_vec"][0, 0], torch.zeros(DIFF_DIM))
    # episode 0, attempt 1 has diff_hash="d1" -> the real diff vector
    assert torch.allclose(out["diff_vec"][0, 1], torch.tensor([9.0, 9.0, 9.0]))


def test_build_attempt_batch_scalar_feats_reflect_compile_and_score():
    attempts = _attempts_two_episodes()
    code_of_hash, diff_of_hash = _code_diff_lookups()
    out = build_attempt_batch(attempts, code_of_hash, diff_of_hash, CODE_DIM, DIFF_DIM, device="cpu")
    # scalar_feats: [compile_ok, score_frac, n_compile_errors(log1p,clip), attempt_idx(log1p,clip)]
    ep0 = out["scalar_feats"][0]
    assert ep0[0, 0] == 0.0  # attempt 0: compile Error -> compile_ok=0
    assert ep0[1, 0] == 1.0  # attempt 1: compile Success -> compile_ok=1
    assert ep0[1, 1] == 1.0  # attempt 1: score_frac=1.0


def test_build_attempt_batch_attempt_idx_is_causal_not_episode_length_normalised():
    """A feature `attempt_frac = row_t / (episode_length - 1)` would make the
    episode's total length exactly recoverable from attempt 1 onward -- a real leak, not
    a style issue (base rates make it a ~3.4x lift on the primary label). Two episodes whose attempt 0/1 are otherwise identical but whose
    TOTAL length differs must produce IDENTICAL attempt_idx features at the shared
    positions -- with a length-normalised feature this would fail (the denominator differs)."""
    short_ep = pd.DataFrame({
        "episode_id": [0, 0], "code_hash": ["a", "b"], "diff_hash": [np.nan, "d"],
        "compile_result": ["Error", "Success"], "score_frac": [0.0, 1.0],
        "n_compile_errors": [1, 0],
    })
    long_ep = pd.DataFrame({
        "episode_id": [0, 0, 0, 0], "code_hash": ["a", "b", "c", "d"],
        "diff_hash": [np.nan, "d", "d", "d"],
        "compile_result": ["Error", "Success", "Success", "Success"],
        "score_frac": [0.0, 1.0, 1.0, 1.0], "n_compile_errors": [1, 0, 0, 0],
    })
    code_of_hash = {"a": np.zeros(CODE_DIM), "b": np.zeros(CODE_DIM),
                    "c": np.zeros(CODE_DIM), "d": np.zeros(CODE_DIM)}
    diff_of_hash = {"d": np.zeros(DIFF_DIM)}

    out_short = build_attempt_batch(short_ep, code_of_hash, diff_of_hash, CODE_DIM, DIFF_DIM, device="cpu")
    out_long = build_attempt_batch(long_ep, code_of_hash, diff_of_hash, CODE_DIM, DIFF_DIM, device="cpu")
    # attempt_idx (column 3) at positions 0 and 1 must be IDENTICAL regardless of each
    # episode's own total length (2 vs 4) -- this is the property the old ratio violated.
    assert torch.allclose(out_short["scalar_feats"][0, 0, 3], out_long["scalar_feats"][0, 0, 3])
    assert torch.allclose(out_short["scalar_feats"][0, 1, 3], out_long["scalar_feats"][0, 1, 3])


def _one_episode_four_attempts():
    """One episode, 4 attempts -- long enough to truncate after attempt k=0,1,2 and
    still have a real 'rest of the episode' being deleted each time."""
    return pd.DataFrame({
        "episode_id": [0, 0, 0, 0],
        "code_hash": ["h0", "h1", "h2", "h3"],
        "diff_hash": [np.nan, "d1", "d2", "d3"],
        "compile_result": ["Error", "Error", "Success", "Success"],
        "score_frac": [0.0, 0.0, 0.5, 1.0],
        "n_compile_errors": [1, 2, 0, 0],
    })


def test_build_attempt_batch_scalar_feats_truncation_invariance():
    """A future-information leak (`attempt_frac` normalised by episode length) can live in
    `build_attempt_batch`, not only in `AttemptEncoder` -- the other truncation canaries
    only drive `AttemptEncoder` with hand-built tensors, so they would not catch an
    episode-level normaliser reintroduced at THIS layer. Build the attempts DataFrame, call
    `build_attempt_batch` on the full
    episode and on the same episode truncated after attempt k, and assert the returned
    `scalar_feats[:, :k+1]` are bit-identical for every k."""
    full = _one_episode_four_attempts()
    code_of_hash = {"h0": np.zeros(CODE_DIM), "h1": np.zeros(CODE_DIM),
                    "h2": np.zeros(CODE_DIM), "h3": np.zeros(CODE_DIM)}
    diff_of_hash = {"d1": np.zeros(DIFF_DIM), "d2": np.zeros(DIFF_DIM), "d3": np.zeros(DIFF_DIM)}
    out_full = build_attempt_batch(full, code_of_hash, diff_of_hash, CODE_DIM, DIFF_DIM, device="cpu")

    for k in range(len(full) - 1):  # truncate after attempt index k (keep attempts 0..k)
        truncated = full.iloc[:k + 1]
        out_trunc = build_attempt_batch(truncated, code_of_hash, diff_of_hash, CODE_DIM, DIFF_DIM, device="cpu")
        max_abs_delta = (out_full["scalar_feats"][:, :k + 1] - out_trunc["scalar_feats"]).abs().max().item()
        assert max_abs_delta == 0.0, f"truncation at k={k}: expected exact 0.0, got {max_abs_delta:.2e}"


def test_build_attempt_batch_attempt_episode_row_and_within_idx_match_the_scatter():
    """`attempt_episode_row_idx`/`attempt_within_idx` must be the SAME (row_n, row_t)
    pair the function used to scatter code_vec/scalar_feats -- gathering code_vec at
    those indices must reproduce the exact per-attempt vectors."""
    attempts = _attempts_two_episodes()
    code_of_hash, diff_of_hash = _code_diff_lookups()
    out = build_attempt_batch(attempts, code_of_hash, diff_of_hash, CODE_DIM, DIFF_DIM, device="cpu")
    ep_idx = out["attempt_episode_row_idx"]
    within_idx = out["attempt_within_idx"]
    assert ep_idx.tolist() == [0, 0, 1]  # episode 0 (2 attempts), episode 1 (1 attempt)
    assert within_idx.tolist() == [0, 1, 0]
    gathered = out["code_vec"][ep_idx, within_idx]
    expected = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    assert torch.allclose(gathered, expected)


def _episode_table_two_episodes():
    return pd.DataFrame({
        "episode_id": [0, 1],
        "learner": ["A", "A"],
        "problem_id": [10, 11],
        "start_timestamp": pd.to_datetime(["2024-01-01T00:00:00", "2024-01-01T01:00:00"]),
    })


def test_build_episode_batch_item_codec_looks_up_correct_problem():
    episodes = _episode_table_two_episodes()
    problem_codec = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)  # problem 10, 11
    pid_to_idx = {10: 0, 11: 1}
    episode_summary = torch.tensor([[100.0], [200.0]])  # dim=1, one per episode_id
    out = build_episode_batch(episodes, problem_codec, pid_to_idx, episode_summary,
                              episode_ids_of_summary=[0, 1], device="cpu")
    assert out["item_codec"].shape == (1, 2, 2)  # 1 learner, T_max=2 episodes, codec_dim=2
    assert torch.allclose(out["item_codec"][0, 0], torch.tensor([1.0, 2.0]))  # episode 0 -> problem 10
    assert torch.allclose(out["item_codec"][0, 1], torch.tensor([3.0, 4.0]))  # episode 1 -> problem 11


def test_build_episode_batch_prev_feat_is_shifted_previous_episode_summary():
    episodes = _episode_table_two_episodes()
    problem_codec = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    pid_to_idx = {10: 0, 11: 1}
    episode_summary = torch.tensor([[100.0], [200.0]])
    out = build_episode_batch(episodes, problem_codec, pid_to_idx, episode_summary,
                              episode_ids_of_summary=[0, 1], device="cpu")
    # prev_feat at t=0 (episode 0) must be ZERO (no previous episode)
    assert torch.allclose(out["prev_feat"][0, 0], torch.zeros(1))
    # prev_feat at t=1 (episode 1) must be episode 0's OWN summary (100.0), not episode 1's own
    assert torch.allclose(out["prev_feat"][0, 1], torch.tensor([100.0]))


def test_build_episode_batch_learner_idx_and_t_idx_align_with_input_row_order():
    episodes = _episode_table_two_episodes()
    problem_codec = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    pid_to_idx = {10: 0, 11: 1}
    episode_summary = torch.tensor([[100.0], [200.0]])
    out = build_episode_batch(episodes, problem_codec, pid_to_idx, episode_summary,
                              episode_ids_of_summary=[0, 1], device="cpu")
    # both episodes belong to the single learner "A" -> learner_idx all 0
    assert out["learner_idx"].tolist() == [0, 0]
    assert out["t_idx"].tolist() == [0, 1]
    assert out["learners"] == ["A"]
