"""Tests for the bread test (multiple-choice option scoring). `mc_option_metrics`/
`paired_diff_ci`/`uniform_option_floor`/`assert_above_uniform_option_floor` are pure (no
model); `mc_option_log_likelihoods`/`mc_option_joint_log_likelihoods` need a forward pass,
tested against the CausalStubLM + attach_lora_basis pipeline.

`outcome_code` is a REQUIRED parameter of `mc_option_log_likelihoods`: leaving
`encode_example`'s own `outcome_code=None` default in place would silently score the
unconditioned template, a prompt shape the class-conditioned simulators are never trained
on. Every call site below passes it explicitly, and a test proves the primitive enforces it.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from atom.binding import Hypernetwork, attach_lora_basis
from atom.binding.stub_causal_lm import CausalStubLM
from atom.data.fa_outcome_label import N_OUTCOME
from atom.heads.fa_hierarchical_heads import OutcomeHead
from atom.simulator.fa_bread_test import (_assert_class_phrase_survives, assert_above_uniform_option_floor,
                                          mc_option_joint_log_likelihoods, mc_option_log_likelihoods,
                                          mc_option_metrics, paired_diff_ci, uniform_option_floor)

HIDDEN_DIM = 32
LATENT_DIM = 8
CODEC_DIM = 6


class _FakeTok:
    pad_token_id = 0
    eos_token_id = 63

    def __call__(self, text: str, add_special_tokens: bool = True) -> dict:
        words = [w for w in text.replace("\n", " \n ").split(" ") if w != ""]
        ids = [(sum(ord(c) for c in w) % 40) + 2 for w in words]
        if add_special_tokens:
            ids = [1, *ids]
        return {"input_ids": ids}


def _build_pipeline(seed: int = 0):
    tok = _FakeTok()
    model = CausalStubLM(vocab_size=64, dim=HIDDEN_DIM, n_layers=2, n_heads=4,
                         pad_token_id=tok.pad_token_id, seed=seed)
    bases, context = attach_lora_basis(model, m=8, r=4, seed=seed + 1)
    hyper = Hypernetwork(u_dim=LATENT_DIM, m=8, kind="mlp", seed=seed + 2)
    return tok, model, bases, context, hyper


def _build_outcome_head(seed: int = 3) -> OutcomeHead:
    torch.manual_seed(seed)
    return OutcomeHead(codec_dim=CODEC_DIM, latent_dim=LATENT_DIM, hidden=16)


# -- mc_option_log_likelihoods (model-coupled) -----------------------------------------

def test_mc_option_log_likelihoods_shape_and_finite():
    tok, model, bases, context, hyper = _build_pipeline()
    u = torch.randn(LATENT_DIM)
    ll = mc_option_log_likelihoods(model, hyper, context, u, tok, "What is 2+2?",
                                   ["four", "five", "six"], max_len=16, device="cpu",
                                   outcome_code=0)
    assert ll.shape == (3,)
    assert torch.isfinite(ll).all()


def test_mc_option_log_likelihoods_rejects_a_batched_u():
    tok, model, bases, context, hyper = _build_pipeline()
    u_batched = torch.randn(2, LATENT_DIM)  # wrong: this function wants ONE example's u
    with pytest.raises(ValueError):
        mc_option_log_likelihoods(model, hyper, context, u_batched, tok, "Q?",
                                  ["a", "b"], max_len=16, device="cpu", outcome_code=0)


def test_mc_option_log_likelihoods_raises_on_an_empty_option_string():
    """An empty completion has zero valid tokens after encode_example's own truncation --
    must fail loudly (a silent zero would corrupt the softmax over options), not average
    away as if it were a normal option."""
    tok, model, bases, context, hyper = _build_pipeline()
    u = torch.randn(LATENT_DIM)
    with pytest.raises(ValueError):
        mc_option_log_likelihoods(model, hyper, context, u, tok, "Q?",
                                  ["a real option", ""], max_len=16, device="cpu", outcome_code=0)


def test_mc_option_log_likelihoods_deterministic_given_fixed_model_and_u():
    tok, model, bases, context, hyper = _build_pipeline()
    model.eval()
    u = torch.randn(LATENT_DIM, generator=torch.Generator().manual_seed(0))
    ll1 = mc_option_log_likelihoods(model, hyper, context, u, tok, "What is 2+2?",
                                    ["four", "five"], max_len=16, device="cpu", outcome_code=1)
    ll2 = mc_option_log_likelihoods(model, hyper, context, u, tok, "What is 2+2?",
                                    ["four", "five"], max_len=16, device="cpu", outcome_code=1)
    assert torch.equal(ll1, ll2)


def test_mc_option_log_likelihoods_requires_outcome_code():
    """The call shape without `outcome_code` must fail loudly at the call boundary -- a
    regression check against a silently-applied default choosing the wrong prompt
    template."""
    tok, model, bases, context, hyper = _build_pipeline()
    u = torch.randn(LATENT_DIM)
    with pytest.raises(TypeError):
        mc_option_log_likelihoods(model, hyper, context, u, tok, "What is 2+2?",  # type: ignore[call-arg]
                                  ["four", "five"], max_len=16, device="cpu")


def test_mc_option_log_likelihoods_scores_different_classes_differently():
    """Sanity that `outcome_code` is actually threaded through to the prompt (not just
    accepted and ignored) -- two different classes must, in general, produce different
    log-likelihoods for the SAME options under the SAME u."""
    tok, model, bases, context, hyper = _build_pipeline()
    u = torch.randn(LATENT_DIM, generator=torch.Generator().manual_seed(2))
    ll_class0 = mc_option_log_likelihoods(model, hyper, context, u, tok, "What is 2+2?",
                                          ["four", "five"], max_len=32, device="cpu", outcome_code=0)
    ll_class4 = mc_option_log_likelihoods(model, hyper, context, u, tok, "What is 2+2?",
                                          ["four", "five"], max_len=32, device="cpu", outcome_code=4)
    assert not torch.allclose(ll_class0, ll_class4)


# -- _assert_class_phrase_survives (pure) ----------------------------------------------

def test_class_phrase_canary_passes_on_a_correctly_built_prompt():
    tok = _FakeTok()
    from atom.simulator.fa_alt2 import encode_example

    ex = encode_example("What is 2+2?", "four", tok, max_len=32, outcome_code=2)
    _assert_class_phrase_survives(ex, tok, outcome_code=2)  # must not raise


def test_would_have_caught_class_phrase_canary_catches_a_wrong_class_label():
    """Would-have-caught regression check: an example built for one class, checked
    against a DIFFERENT class's suffix, must be caught -- proving the canary has power to
    detect a mismatched prompt shape, not just that it passes on correct input (exactly
    the shape of defect this canary exists to catch: a scorer silently building the wrong
    template)."""
    tok = _FakeTok()
    from atom.simulator.fa_alt2 import encode_example

    ex = encode_example("What is 2+2?", "four", tok, max_len=32, outcome_code=2)
    with pytest.raises(AssertionError):
        _assert_class_phrase_survives(ex, tok, outcome_code=3)


def test_would_have_caught_class_phrase_canary_catches_the_unconditioned_template():
    """The exact regression this whole task exists to prevent: an example built under
    `mc_option_log_likelihoods`'s OLD behaviour (the unconditioned template,
    `outcome_code=None`) checked against ANY real class's suffix must fail."""
    tok = _FakeTok()
    from atom.simulator.fa_alt2 import encode_example

    ex_unconditioned = encode_example("What is 2+2?", "four", tok, max_len=32, outcome_code=None)
    with pytest.raises(AssertionError):
        _assert_class_phrase_survives(ex_unconditioned, tok, outcome_code=0)


# -- mc_option_joint_log_likelihoods (model-coupled) ------------------------------------

def test_mc_option_joint_log_likelihoods_shape_and_finite():
    tok, model, bases, context, hyper = _build_pipeline()
    outcome_head = _build_outcome_head()
    u_k = torch.randn(4, LATENT_DIM, generator=torch.Generator().manual_seed(5))
    item_codec = torch.randn(CODEC_DIM, generator=torch.Generator().manual_seed(6))
    out = mc_option_joint_log_likelihoods(model, hyper, context, outcome_head, u_k, item_codec,
                                          tok, "What is 2+2?", ["four", "five", "six"],
                                          max_len=32, device="cpu")
    assert out["budget_consistent"] is True
    assert out["log_likelihoods"].shape == (3,)
    assert torch.isfinite(out["log_likelihoods"]).all()
    assert out["n_valid_tok"].shape == (3,)


def test_mc_option_joint_log_likelihoods_rejects_a_1d_u_k():
    tok, model, bases, context, hyper = _build_pipeline()
    outcome_head = _build_outcome_head()
    u_k_1d = torch.randn(LATENT_DIM)  # wrong: this function wants [K, u_dim], K explicit
    item_codec = torch.randn(CODEC_DIM)
    with pytest.raises(ValueError):
        mc_option_joint_log_likelihoods(model, hyper, context, outcome_head, u_k_1d, item_codec,
                                        tok, "Q?", ["a", "b"], max_len=32, device="cpu")


def test_mc_option_joint_log_likelihoods_rejects_a_degenerate_option_set():
    tok, model, bases, context, hyper = _build_pipeline()
    outcome_head = _build_outcome_head()
    u_k = torch.randn(2, LATENT_DIM)
    item_codec = torch.randn(CODEC_DIM)
    with pytest.raises(ValueError):
        mc_option_joint_log_likelihoods(model, hyper, context, outcome_head, u_k, item_codec,
                                        tok, "Q?", ["only one option"], max_len=32, device="cpu")


def test_mc_option_joint_log_likelihoods_flags_budget_inconsistency_instead_of_crashing():
    """Requirement 12's own shape, applied at option granularity: a tight `max_len` where
    the class-token suffix's varying length pushes SOME classes' completion past the
    budget and others not -- measured directly against `_FakeTok` (16 tokens gives counts
    [7,7,6,6,4] across the 5 classes for this question/option pair). Must be reported via
    `budget_consistent=False`, never silently scored on a truncated subset."""
    tok, model, bases, context, hyper = _build_pipeline()
    outcome_head = _build_outcome_head()
    u_k = torch.randn(2, LATENT_DIM)
    item_codec = torch.randn(CODEC_DIM)
    out = mc_option_joint_log_likelihoods(
        model, hyper, context, outcome_head, u_k, item_codec, tok, "What is 2+2?",
        ["a moderately long option text here", "four"], max_len=16, device="cpu")
    assert out["budget_consistent"] is False
    assert out["log_likelihoods"] is None


def test_mc_option_joint_log_likelihoods_reduces_to_single_class_scoring_when_one_class_dominates():
    """Consistency check between the two public scorers: when the outcome head puts
    essentially ALL its mass on one class (a huge bias on that class's logit) and K=1, the
    joint marginalised score must closely match `mc_option_log_likelihoods` called
    directly under that SAME class and u -- the joint estimator degenerates to the
    single-class case exactly as `joint_kc_mixture_nll`'s own K=1 test (`tests/
    test_fa_sim3_joint_kc_estimator.py`) proves for the raw mixture math; this is the same
    property one layer up, through the option-scoring orchestration."""
    tok, model, bases, context, hyper = _build_pipeline()
    dominant_class = 2
    outcome_head = _build_outcome_head()
    with torch.no_grad():
        outcome_head.outcome.bias.zero_()
        outcome_head.outcome.bias[dominant_class] = 50.0  # softmax ~= one-hot
        outcome_head.outcome.weight.zero_()  # kill any u/item_codec-dependence of the logits
    u_k = torch.randn(1, LATENT_DIM, generator=torch.Generator().manual_seed(7))
    item_codec = torch.randn(CODEC_DIM, generator=torch.Generator().manual_seed(8))
    options = ["four", "five", "six"]

    joint_out = mc_option_joint_log_likelihoods(model, hyper, context, outcome_head, u_k, item_codec,
                                                tok, "What is 2+2?", options, max_len=32, device="cpu")
    direct_ll = mc_option_log_likelihoods(model, hyper, context, u_k[0], tok, "What is 2+2?",
                                          options, max_len=32, device="cpu", outcome_code=dominant_class)
    assert torch.allclose(joint_out["log_likelihoods"], direct_ll, atol=1e-4)


def test_mc_option_joint_log_likelihoods_is_invariant_to_the_chunk_size():
    """Tiling ALL of K into one forward call can run out of GPU memory at the widest option
    count (observed on a 24 GB A10G), so `max_batch_size` chunks K into smaller forward
    calls, and chunking must be EXACT concatenation, never an approximation. Runs the SAME
    (u_k, item_codec, options) through a small `max_batch_size` (forces multiple chunks --
    K=6, n_opt=4 -> chunk size 2, three chunks) and a large one (K=6 fits in one chunk) and
    asserts the results are identical, not merely close."""
    tok, model, bases, context, hyper = _build_pipeline()
    outcome_head = _build_outcome_head()
    u_k = torch.randn(6, LATENT_DIM, generator=torch.Generator().manual_seed(11))
    item_codec = torch.randn(CODEC_DIM, generator=torch.Generator().manual_seed(12))
    options = ["four", "five", "six", "seven"]

    chunked = mc_option_joint_log_likelihoods(model, hyper, context, outcome_head, u_k, item_codec,
                                              tok, "What is 2+2?", options, max_len=32, device="cpu",
                                              max_batch_size=8)  # -> chunk size 2, three chunks
    unchunked = mc_option_joint_log_likelihoods(model, hyper, context, outcome_head, u_k, item_codec,
                                                tok, "What is 2+2?", options, max_len=32, device="cpu",
                                                max_batch_size=1000)  # -> one chunk, all K at once
    assert torch.allclose(chunked["log_likelihoods"], unchunked["log_likelihoods"], atol=1e-5)


# -- uniform_option_floor / assert_above_uniform_option_floor (pure, canary family #8) --

def test_uniform_option_floor_matches_hand_computed_values():
    floor = uniform_option_floor([2, 4])
    assert math.isclose(floor["mean_nll_floor"], (math.log(2) + math.log(4)) / 2, rel_tol=1e-9)
    assert math.isclose(floor["mean_top1_floor"], (0.5 + 0.25) / 2, rel_tol=1e-9)


def test_uniform_option_floor_rejects_a_row_with_fewer_than_two_options():
    with pytest.raises(ValueError):
        uniform_option_floor([2, 1])


def test_assert_above_uniform_option_floor_passes_when_metric_beats_chance():
    floor = assert_above_uniform_option_floor(top1_rate=0.6, n_options=[2, 3, 4])
    assert floor["mean_top1_floor"] < 0.6


def test_would_have_caught_assert_above_uniform_option_floor_halts_on_the_exact_fa_sim3_shape():
    """Regression check on a known failure shape: a metric scoring BELOW random guessing
    (an unconditioned-template scorer on FA-SIM3 measured top1=0.1251 against a higher
    floor) must halt (raise), not report."""
    with pytest.raises(AssertionError):
        assert_above_uniform_option_floor(top1_rate=0.1251, n_options=[2, 2, 3, 3, 3, 4, 5])


# -- mc_option_metrics (pure) -----------------------------------------------------------

def test_mc_option_metrics_confident_correct_prediction_has_low_nll_and_top1():
    ll = np.array([5.0, 0.0, -1.0])  # option 0 is by far the most likely
    out = mc_option_metrics(ll, correct_index=0)
    assert out["top1_correct"] is True
    assert out["nll"] < 0.01
    assert out["n_options"] == 3


def test_mc_option_metrics_confident_wrong_prediction_has_high_nll_and_no_top1():
    ll = np.array([5.0, 0.0, -1.0])  # option 0 is favoured, but the TRUE answer is option 2
    out = mc_option_metrics(ll, correct_index=2)
    assert out["top1_correct"] is False
    assert out["nll"] > 5.0


def test_mc_option_metrics_uniform_ll_gives_log_n_options_nll():
    ll = np.zeros(4)
    out = mc_option_metrics(ll, correct_index=1)
    assert math.isclose(out["nll"], math.log(4), rel_tol=1e-9)


def test_mc_option_metrics_accepts_torch_tensor_too():
    ll = torch.tensor([1.0, 2.0, 0.5])
    out = mc_option_metrics(ll, correct_index=1)
    assert out["top1_correct"] is True


# -- paired_diff_ci (pure) ---------------------------------------------------------------

def test_paired_diff_ci_rejects_mismatched_shapes():
    with pytest.raises(ValueError):
        paired_diff_ci([1.0, 2.0], [1.0, 2.0, 3.0])


def test_paired_diff_ci_excludes_zero_for_a_clear_paired_advantage():
    rng = np.random.default_rng(0)
    control_nll = rng.normal(loc=2.0, scale=0.1, size=100)     # control: worse (higher NLL)
    personalised_nll = rng.normal(loc=1.0, scale=0.1, size=100)  # personalised: better
    out = paired_diff_ci(control_nll, personalised_nll, n_boot=2000, seed=1)
    assert out["excludes_zero"] is True
    assert out["mean_diff"] > 0  # control - personalised > 0, i.e. control scores worse


def test_paired_diff_ci_straddles_zero_when_arms_are_indistinguishable():
    rng = np.random.default_rng(0)
    a = rng.normal(loc=1.0, scale=1.0, size=30)
    b = a + rng.normal(scale=0.01, size=30)  # nearly identical to a, tiny independent noise
    out = paired_diff_ci(a, b, n_boot=2000, seed=1)
    assert out["excludes_zero"] is False
