import torch

from atom.filter.cw_attempt_encoder import AttemptEncoder, build_scalar_feats

CODE_DIM = 6
DIFF_DIM = 6
PROJ_DIM = 4
HIDDEN = 8
SUMMARY_DIM = 5


def _encoder():
    torch.manual_seed(0)
    return AttemptEncoder(CODE_DIM, DIFF_DIM, PROJ_DIM, HIDDEN, SUMMARY_DIM)


def _episode(n_ep, t_max, seed=0):
    g = torch.Generator().manual_seed(seed)
    code_vec = torch.randn(n_ep, t_max, CODE_DIM, generator=g)
    diff_vec = torch.randn(n_ep, t_max, DIFF_DIM, generator=g)
    scalar = torch.randn(n_ep, t_max, 4, generator=g)
    return code_vec, diff_vec, scalar


def test_output_shapes_and_variable_length_episodes():
    enc = _encoder()
    n_ep, t_max = 3, 5
    code_vec, diff_vec, scalar = _episode(n_ep, t_max)
    lengths = torch.tensor([1, 3, 5])
    prefix_states, summary = enc(code_vec, diff_vec, scalar, lengths)
    assert prefix_states.shape == (n_ep, t_max, HIDDEN)
    assert summary.shape == (n_ep, SUMMARY_DIM)
    assert torch.isfinite(prefix_states).all()
    assert torch.isfinite(summary).all()


def test_padding_beyond_length_does_not_change_summary():
    """Regression: the pooled summary must be pad-invariant -- perturbing values past
    an episode's own true length must not move its summary at all."""
    enc = _encoder()
    n_ep, t_max = 2, 4
    lengths = torch.tensor([2, 4])
    code_vec, diff_vec, scalar = _episode(n_ep, t_max)
    _, summary_a = enc(code_vec, diff_vec, scalar, lengths)

    code_vec2, diff_vec2, scalar2 = code_vec.clone(), diff_vec.clone(), scalar.clone()
    code_vec2[0, 2:] = 999.0
    diff_vec2[0, 2:] = 999.0
    scalar2[0, 2:] = 999.0
    _, summary_b = enc(code_vec2, diff_vec2, scalar2, lengths)

    assert torch.allclose(summary_a[0], summary_b[0], atol=1e-6)
    assert torch.allclose(summary_a[1], summary_b[1], atol=1e-6)  # untouched episode, sanity


def test_flip_attempt_k_causality_canary_exact_zero():
    """CLAUSE 1's own required standard: 'the flip-k canary asserts max|Delta| =
    0.00e+00 exactly, like FA's peek canary, not small'. Flip attempt index K's own
    code/diff/scalar inputs entirely (not just perturb) and confirm every prefix state
    at position <= K is BIT-IDENTICAL -- prefix_states[:, k] must depend only on
    attempts 0..k-1, never on attempt k itself or anything later."""
    enc = _encoder()
    n_ep, t_max = 1, 6
    lengths = torch.tensor([6])
    code_vec, diff_vec, scalar = _episode(n_ep, t_max)
    prefix_a, summary_a = enc(code_vec, diff_vec, scalar, lengths)

    K = 3
    code_vec2, diff_vec2, scalar2 = code_vec.clone(), diff_vec.clone(), scalar.clone()
    code_vec2[0, K] = torch.randn(CODE_DIM) * 1000.0  # flip attempt K entirely
    diff_vec2[0, K] = torch.randn(DIFF_DIM) * 1000.0
    scalar2[0, K] = torch.randn(4) * 1000.0
    prefix_b, summary_b = enc(code_vec2, diff_vec2, scalar2, lengths)

    # prefix states at positions 0..K (inclusive) must be EXACTLY unchanged: prefix_states[:,k]
    # is built from attempts 0..k-1, so flipping attempt K can only reach prefix_states[:,K+1:].
    max_abs_delta = (prefix_a[:, :K + 1] - prefix_b[:, :K + 1]).abs().max().item()
    assert max_abs_delta == 0.0, f"expected exact 0.0, got {max_abs_delta:.2e}"

    # prefix states AFTER the flipped attempt (and the pooled summary) MUST move --
    # otherwise the flip test would be vacuous (the encoder ignoring its own inputs).
    assert (prefix_a[:, K + 1:] - prefix_b[:, K + 1:]).abs().max().item() > 0.0
    assert not torch.allclose(summary_a, summary_b)


def test_prefix_truncation_invariance_canary_exact_zero():
    """PRE-REGISTERED (13 Aug review, after the attempt_frac length-leak): stronger than
    flip-k and subsumes it. Flip-k perturbs one attempt's OWN outcome/content -- it
    cannot detect a leak that arrives through a FEATURE'S NORMALISER rather than through
    attention/recurrence to a later position (the episode-length leak sat entirely
    inside `attempt_frac`'s denominator and flip-k's max|Delta| was correctly, uselessly,
    0.0 the whole time it was live). This canary deletes the entire future instead of
    perturbing one attempt: for every truncation point k, the prefix states for attempts
    0..k computed from the episode CUT OFF after attempt k must be bit-identical to those
    computed from the full episode. Any dependence on anything past position k -- whether
    via attention, recurrence, OR a feature computed from the eventual total length --
    fails this test."""
    enc = _encoder()
    t_full = 6
    code_vec, diff_vec, scalar = _episode(1, t_full)
    prefix_full, _ = enc(code_vec, diff_vec, scalar, torch.tensor([t_full]))

    for k in range(1, t_full):  # truncate after attempt index k-1 (i.e. keep attempts 0..k-1)
        code_trunc, diff_trunc, scalar_trunc = code_vec[:, :k], diff_vec[:, :k], scalar[:, :k]
        prefix_trunc, _ = enc(code_trunc, diff_trunc, scalar_trunc, torch.tensor([k]))
        max_abs_delta = (prefix_full[:, :k] - prefix_trunc[:, :k]).abs().max().item()
        assert max_abs_delta == 0.0, f"truncation at k={k}: expected exact 0.0, got {max_abs_delta:.2e}"


def test_prefix_truncation_invariance_would_have_caught_the_length_leak():
    """Direct demonstration that this canary is the one that fires: reproduces the OLD
    (buggy) attempt_frac = position / (episode_length - 1) feature by hand, feeds it as
    the 4th scalar channel (bypassing build_scalar_feats' own log1p+clamp, which is
    orthogonal to this leak), and confirms truncation invariance FAILS -- the causal-mask
    machinery itself was always correct; only a length-dependent feature could break this."""
    enc = _encoder()
    t_full = 4
    code_vec, diff_vec, _ = _episode(1, t_full)
    leaky_frac = torch.tensor([[0.0, 1.0, 2.0, 3.0]]) / (t_full - 1)  # OLD, buggy feature
    scalar_leaky = torch.stack([torch.zeros(1, t_full), torch.zeros(1, t_full),
                                torch.zeros(1, t_full), leaky_frac], dim=-1)
    prefix_full, _ = enc(code_vec, diff_vec, scalar_leaky, torch.tensor([t_full]))

    # t_trunc=3 (not 2): prefix_states[:,1] depends ONLY on attempt 0, whose leaky value is
    # 0/(L-1)=0 regardless of L, so a 2-attempt truncation is accidentally invariant at
    # position 1 -- not because the leak is gone, but because the check needs a position
    # whose OWN attempt has a nonzero leaky value (attempt 1, feeding prefix_states[:,2]).
    t_trunc = 3
    leaky_frac_trunc = torch.tensor([[0.0, 1.0, 2.0]]) / (t_trunc - 1)  # SAME positions, shorter episode
    scalar_leaky_trunc = torch.stack(
        [torch.zeros(1, t_trunc), torch.zeros(1, t_trunc), torch.zeros(1, t_trunc), leaky_frac_trunc], dim=-1)
    prefix_trunc, _ = enc(code_vec[:, :t_trunc], diff_vec[:, :t_trunc], scalar_leaky_trunc,
                          torch.tensor([t_trunc]))

    max_abs_delta = (prefix_full[:, :t_trunc] - prefix_trunc[:, :t_trunc]).abs().max().item()
    assert max_abs_delta > 0.0, "the leaky feature should have broken truncation invariance"


def test_degenerate_length_one_episode_prefix_is_the_fixed_init_state():
    """CLAUSE 5: at episode length 1, the within-episode prefix is just the init state
    and collapses exactly onto FA's two-input (item_codec, u) shape -- i.e. prefix_states
    at attempt 1 is IDENTICAL across every episode regardless of that episode's own
    content, so it contributes nothing episode-specific (a fixed constant, not a
    per-episode signal)."""
    enc = _encoder()
    code_a, diff_a, scalar_a = _episode(1, 1, seed=1)
    code_b, diff_b, scalar_b = _episode(1, 1, seed=2)
    assert not torch.allclose(code_a, code_b)  # the two episodes really do differ

    prefix_a, _ = enc(code_a, diff_a, scalar_a, torch.tensor([1]))
    prefix_b, _ = enc(code_b, diff_b, scalar_b, torch.tensor([1]))
    assert torch.equal(prefix_a[:, 0], prefix_b[:, 0])
    assert torch.equal(prefix_a[:, 0], torch.zeros(1, HIDDEN))


def test_first_attempt_diff_vector_can_be_all_zero():
    """Episode-first attempts get no diff row (cw_diffs' own convention) -- confirms an
    all-zero diff channel at t=0 does not crash or produce NaN."""
    enc = _encoder()
    n_ep, t_max = 1, 1
    code_vec, _, scalar = _episode(n_ep, t_max)
    diff_vec = torch.zeros(n_ep, t_max, DIFF_DIM)
    lengths = torch.tensor([1])
    prefix_states, summary = enc(code_vec, diff_vec, scalar, lengths)
    assert torch.isfinite(prefix_states).all()
    assert torch.isfinite(summary).all()


def test_rejects_zero_length_episode():
    enc = _encoder()
    code_vec, diff_vec, scalar = _episode(1, 3)
    lengths = torch.tensor([0])
    try:
        enc(code_vec, diff_vec, scalar, lengths)
    except ValueError:
        return
    raise AssertionError("expected ValueError for a zero-length episode")


def test_build_scalar_feats_order_and_compression():
    compile_ok = torch.tensor([1.0, 0.0])
    score_frac = torch.tensor([1.0, 0.0])
    n_compile_errors = torch.tensor([0.0, 100.0])
    attempt_idx = torch.tensor([0.0, 1.0])
    feats = build_scalar_feats(compile_ok, score_frac, n_compile_errors, attempt_idx)
    assert feats.shape == (2, 4)
    assert feats[0, 0] == 1.0 and feats[1, 0] == 0.0  # compile_ok passthrough
    assert feats[0, 2] == 0.0  # log1p(0) == 0
    assert feats[1, 2] <= 5.0  # clipped
    assert feats[0, 3] == 0.0  # log1p(0) == 0
    assert abs(feats[1, 3].item() - torch.log1p(torch.tensor(1.0)).item()) < 1e-6  # log1p, not raw


def test_gradient_flows_to_all_projections_from_both_outputs():
    enc = _encoder()
    n_ep, t_max = 2, 3
    code_vec, diff_vec, scalar = _episode(n_ep, t_max)
    code_vec.requires_grad_(True)
    diff_vec.requires_grad_(True)
    scalar.requires_grad_(True)
    lengths = torch.tensor([2, 3])
    prefix_states, summary = enc(code_vec, diff_vec, scalar, lengths)
    (prefix_states.sum() + summary.sum()).backward()
    for p in enc.parameters():
        assert p.grad is not None
        assert torch.isfinite(p.grad).all()
