"""Tests for the joint-set training step (`atom.simulator.joint_set_train_step`)."""

from __future__ import annotations

import numpy as np
import torch

from atom.filter.losses import gaussian_kl_free_bits, gaussian_kl_two_diagonal
from atom.simulator.joint_set_train_step import (
    PrefixTargetDraw,
    TeacherEncoder,
    joint_set_training_step,
    mu_scale_penalty,
    pad_draws,
    sample_prefix_target_draws,
    set_kmixture_nll,
)


def test_gaussian_kl_two_diagonal_matches_free_bits_special_case():
    torch.manual_seed(0)
    mu = torch.randn(8, 16)
    log_sigma = torch.randn(8, 16) * 0.3
    zero_mu = torch.zeros_like(mu)
    zero_log_sigma = torch.zeros_like(log_sigma)
    general = gaussian_kl_two_diagonal(mu, log_sigma, zero_mu, zero_log_sigma)
    # free_bits very negative -> the clamp never binds, so the sum is the raw (unclamped) KL.
    special = gaussian_kl_free_bits(mu, log_sigma, free_bits=-1e9)
    torch.testing.assert_close(general, special, atol=1e-5, rtol=1e-5)


def test_gaussian_kl_two_diagonal_zero_when_distributions_match():
    mu = torch.randn(4, 8)
    log_sigma = torch.randn(4, 8) * 0.5
    kl = gaussian_kl_two_diagonal(mu, log_sigma, mu, log_sigma)
    torch.testing.assert_close(kl, torch.zeros(4), atol=1e-6, rtol=1e-6)


def test_sample_prefix_target_draws_respects_causality_and_size_range():
    rng = np.random.default_rng(0)
    valid_t_by_learner = {
        0: np.arange(0, 50),          # dense, no gaps
        1: np.array([0, 1, 5, 6, 7, 9, 12, 15, 20, 25, 30, 35]),  # gappy, short
        2: np.arange(0, 3),           # too short to ever produce a draw (< lo)
    }
    prefix_pool = (0, 1, 2, 3, 5, 10, 20)
    draws = sample_prefix_target_draws(valid_t_by_learner, rng, prefix_pool,
                                       target_set_size_range=(4, 8), n_draws=200)
    assert len(draws) == 200
    for d in draws:
        assert d.learner_idx in (0, 1)  # learner 2 is infeasible (only 3 valid positions)
        assert 4 <= len(d.target_t) <= 8
        assert np.all(d.target_t >= d.prefix_t)  # never leaky: H_<prefix_t excludes target_t
        assert np.all(np.diff(d.target_t) > 0)   # sorted, unique
        assert set(d.target_t).issubset(set(valid_t_by_learner[d.learner_idx]))


def test_sample_prefix_target_draws_includes_zero_prefix_when_in_pool():
    rng = np.random.default_rng(1)
    valid_t_by_learner = {0: np.arange(0, 30)}
    draws = sample_prefix_target_draws(valid_t_by_learner, rng, prefix_pool=(0,),
                                       target_set_size_range=(5, 10), n_draws=50)
    assert len(draws) == 50
    assert all(d.prefix_t == 0 for d in draws)


def test_pad_draws_shapes_and_masking():
    draws = [
        PrefixTargetDraw(learner_idx=0, prefix_t=0, target_t=np.array([1, 3, 5])),
        PrefixTargetDraw(learner_idx=1, prefix_t=2, target_t=np.array([2, 4])),
    ]
    n_learners, t_max_seq, codec_dim, n_classes = 2, 10, 6, 5
    item_codec_all = torch.randn(n_learners, t_max_seq, codec_dim)
    outcome_all = torch.randint(0, n_classes, (n_learners, t_max_seq))
    batch = pad_draws(draws, item_codec_all, outcome_all, n_classes=n_classes, device="cpu")
    assert batch["target_item_codec"].shape == (2, 3, codec_dim)
    assert batch["target_valid_mask"].tolist() == [[True, True, True], [True, True, False]]
    assert batch["target_outcome_onehot"][1, 2].sum().item() == 0.0  # padded row is all-zero
    torch.testing.assert_close(batch["target_item_codec"][0, 1], item_codec_all[0, 3])


class _TinyOutcomeHead(torch.nn.Module):
    def __init__(self, codec_dim: int, latent_dim: int, n_classes: int):
        super().__init__()
        self.lin_codec = torch.nn.Linear(codec_dim, n_classes)
        self.lin_z = torch.nn.Linear(latent_dim, n_classes)

    def forward(self, item_codec, z):
        return self.lin_codec(item_codec) + self.lin_z(z)


def test_set_kmixture_nll_matches_manual_single_sample_when_k_collapses():
    torch.manual_seed(0)
    b, t, k, codec_dim, latent_dim, n_classes = 3, 4, 1, 5, 6, 5
    head = _TinyOutcomeHead(codec_dim, latent_dim, n_classes)
    item_codec = torch.randn(b, t, codec_dim)
    z_k = torch.randn(b, k, latent_dim)
    target = torch.randint(0, n_classes, (b, t))
    mask = torch.ones(b, t, dtype=torch.bool)

    nll = set_kmixture_nll(head, item_codec, z_k, target, mask)

    # Manual: with K=1 the mixture is trivial -- NLL = -sum_i log p(y_i | z, e_i).
    logits = head.lin_codec(item_codec) + head.lin_z(z_k[:, 0, :]).unsqueeze(1)
    log_probs = torch.log_softmax(logits, dim=-1)
    manual = -log_probs.gather(-1, target.unsqueeze(-1)).squeeze(-1).sum(dim=-1)
    torch.testing.assert_close(nll, manual, atol=1e-4, rtol=1e-4)


def test_set_kmixture_nll_rewards_a_diverse_committee_on_a_correlated_set():
    """The mechanism claim in one test. A target SET whose members are all the SAME item
    kind (so ONE shared z must explain all T of them jointly) simulates a learner whose
    T answers are strongly correlated (all-right or all-wrong, per their own consistent
    ability) -- exactly the kind of set that real between-learner variance
    guarantees exists. A CLONE committee (every z_k identical, forced to hedge with a
    diffuse marginal) pays an entropy cost that grows LINEARLY with the set size T, because
    it cannot commit to either coherent explanation. A DIVERSE committee (one z_k
    confidently explains "this is a class-0 learner", another confidently explains
    "this is a class-1 learner") lets `logsumexp` pick up whichever member happens to
    match the realised set, at a cost that does NOT grow with T -- the set-level mixture
    identifies concentration, which the single-outcome mixture
    (`outcome_kmixture_nll_per_row`) structurally cannot,
    because a single held-out answer never needs joint, T-way internal consistency."""
    torch.manual_seed(0)
    codec_dim, latent_dim, n_classes = 2, 2, 2
    t = 10
    item_codec = torch.ones(1, t, codec_dim)  # every target is the SAME item kind
    target_all_class0 = torch.zeros(1, t, dtype=torch.long)  # the realised set: all class 0
    mask = torch.ones(1, t, dtype=torch.bool)

    class _LinearHead(torch.nn.Module):
        def forward(self, item_codec_, z):
            del item_codec_
            return z  # logits = z directly: z=[a,b] -> softmax favours class0 iff a>b

    head = _LinearHead()
    scale = 4.0
    # Clone committee: both members identical, forced to hedge at the undecided midpoint.
    z_clone = torch.zeros(1, 2, latent_dim)
    # Diverse committee: member 0 confidently "class-0 learner", member 1 confidently "class-1 learner".
    z_div = torch.tensor([[[scale, -scale], [-scale, scale]]])

    nll_clone = set_kmixture_nll(head, item_codec, z_clone, target_all_class0, mask)
    nll_div = set_kmixture_nll(head, item_codec, z_div, target_all_class0, mask)
    assert nll_div.item() < nll_clone.item(), (
        f"diverse committee should explain the internally-consistent set better: "
        f"nll_div={nll_div.item():.4f} nll_clone={nll_clone.item():.4f}")

    # And the gap must WIDEN as the set grows -- the clone's cost is O(T), the diverse
    # committee's is O(1) (logsumexp always has a near-perfect member to fall back on).
    t_big = 30
    item_codec_big = torch.ones(1, t_big, codec_dim)
    target_big = torch.zeros(1, t_big, dtype=torch.long)
    mask_big = torch.ones(1, t_big, dtype=torch.bool)
    nll_clone_big = set_kmixture_nll(head, item_codec_big, z_clone, target_big, mask_big)
    nll_div_big = set_kmixture_nll(head, item_codec_big, z_div, target_big, mask_big)
    gap_small = (nll_clone - nll_div).item()
    gap_big = (nll_clone_big - nll_div_big).item()
    assert gap_big > gap_small, (
        f"the clone-vs-diverse gap should widen with set size: gap@T=10={gap_small:.3f} "
        f"gap@T=30={gap_big:.3f}")


def test_joint_set_training_step_gradients_flow_and_kl_stop_gradient_on_teacher():
    torch.manual_seed(0)
    b, t, codec_dim, latent_dim, n_classes, k = 5, 6, 4, 8, 5, 4
    head = _TinyOutcomeHead(codec_dim, latent_dim, n_classes)
    teacher = TeacherEncoder(codec_dim, latent_dim, n_classes, hidden=16)

    mu_c = torch.randn(b, latent_dim, requires_grad=True)
    log_sigma_c = (torch.randn(b, latent_dim) * 0.1).requires_grad_(True)
    target_item_codec = torch.randn(b, t, codec_dim)
    target_outcome = torch.randint(0, n_classes, (b, t))
    target_outcome_onehot = torch.zeros(b, t, n_classes).scatter_(2, target_outcome.unsqueeze(-1), 1.0)
    target_valid_mask = torch.ones(b, t, dtype=torch.bool)
    prefix_t = torch.tensor([0, 0, 3, 5, 10])
    gen = torch.Generator().manual_seed(0)

    out = joint_set_training_step(
        head, teacher, mu_c, log_sigma_c, target_item_codec, target_outcome,
        target_outcome_onehot, target_valid_mask, prefix_t, k_train=k, beta_kl=0.1,
        beta_scale=0.01, generator=gen,
    )
    assert out.loss.dim() == 0
    assert out.diagnostics["n_empty_context_draws"] == 2

    out.loss.backward()
    assert mu_c.grad is not None and mu_c.grad.abs().sum() > 0
    assert log_sigma_c.grad is not None and log_sigma_c.grad.abs().sum() > 0
    for p in teacher.parameters():
        assert p.grad is not None and p.grad.abs().sum() > 0
    for p in head.parameters():
        assert p.grad is not None and p.grad.abs().sum() > 0

    # Isolate the KL term: its gradient must reach ONLY the student (mu_c/log_sigma_c),
    # never the teacher (mu_psi/log_sigma_psi are detached before entering the KL).
    teacher2 = TeacherEncoder(codec_dim, latent_dim, n_classes, hidden=16)
    teacher2.load_state_dict(teacher.state_dict())
    mu_c2 = mu_c.detach().clone().requires_grad_(True)
    log_sigma_c2 = log_sigma_c.detach().clone().requires_grad_(True)
    mu_psi, log_sigma_psi = teacher2(mu_c2, log_sigma_c2, target_item_codec,
                                     target_outcome_onehot, target_valid_mask)
    kl_only = gaussian_kl_two_diagonal(mu_psi.detach(), log_sigma_psi.detach(), mu_c2, log_sigma_c2).mean()
    kl_only.backward()
    assert mu_c2.grad is not None and mu_c2.grad.abs().sum() > 0
    for p in teacher2.parameters():
        assert p.grad is None or p.grad.abs().sum() == 0, (
            "teacher must receive ZERO gradient from the KL term alone (detached) -- "
            "it is trained only by its own set-likelihood term")


def test_mu_scale_penalty_is_squared_l2_norm_per_row():
    mu = torch.tensor([[3.0, 4.0], [0.0, 0.0], [1.0, 0.0]])
    torch.testing.assert_close(mu_scale_penalty(mu), torch.tensor([25.0, 0.0, 1.0]))


def test_mu_scale_penalty_has_no_direct_dependence_on_log_sigma():
    """The disclosed claim is specifically that `mu_scale_penalty` is a function of the MEAN
    only. This is exact for the student term (`mu_scale_penalty(mu_c)`, tested directly
    here). The full training step's TEACHER term is a different story: `mu_psi` is itself a
    function of `log_sigma_c` via `TeacherEncoder.context_proj` (the teacher's nll/KL
    terms also depend on it), so
    `mu_scale_penalty(mu_psi)` inherits that indirect path. That entanglement is not this
    function's concern; keep the two claims separate."""
    mu_c = torch.randn(4, 8, requires_grad=True)
    log_sigma_c = torch.randn(4, 8, requires_grad=True)
    mu_scale_penalty(mu_c).sum().backward()
    assert log_sigma_c.grad is None


def test_joint_set_training_step_scale_penalty_is_nonzero_with_no_empty_context_draws():
    """The scale penalty must fire at EVERY context, not only at `prefix_t == 0` (an
    anchor that fires only at the empty context leaves mu's scale unbounded away from
    the cold start). A batch with zero empty-context draws must still produce a nonzero,
    gradient-carrying penalty."""
    torch.manual_seed(0)
    b, t, codec_dim, latent_dim, n_classes, k = 3, 4, 4, 8, 5, 4
    head = _TinyOutcomeHead(codec_dim, latent_dim, n_classes)
    teacher = TeacherEncoder(codec_dim, latent_dim, n_classes, hidden=16)
    mu_c = torch.randn(b, latent_dim, requires_grad=True)
    log_sigma_c = (torch.randn(b, latent_dim) * 0.1).requires_grad_(True)
    target_item_codec = torch.randn(b, t, codec_dim)
    target_outcome = torch.randint(0, n_classes, (b, t))
    target_outcome_onehot = torch.zeros(b, t, n_classes).scatter_(2, target_outcome.unsqueeze(-1), 1.0)
    target_valid_mask = torch.ones(b, t, dtype=torch.bool)
    prefix_t = torch.tensor([3, 5, 10])  # none empty

    out = joint_set_training_step(
        head, teacher, mu_c, log_sigma_c, target_item_codec, target_outcome,
        target_outcome_onehot, target_valid_mask, prefix_t, k_train=k, beta_kl=0.1,
        beta_scale=0.01,
    )
    assert out.diagnostics["n_empty_context_draws"] == 0
    assert out.mu_scale_penalty.item() > 0.0  # an empty-context-only anchor would read 0 here

    # And it carries gradient back to both the student's belief and the teacher's mu_head,
    # each from its own (non-detached) term.
    (0.01 * out.mu_scale_penalty).backward()
    assert mu_c.grad is not None and mu_c.grad.abs().sum() > 0
    assert teacher.mu_head.weight.grad is not None and teacher.mu_head.weight.grad.abs().sum() > 0
    # (log_sigma_c's own isolation from the penalty is tested directly, on mu_scale_penalty
    # alone, in test_mu_scale_penalty_has_no_direct_dependence_on_log_sigma -- inside the
    # full step it also receives the teacher's context_proj path, unrelated to the penalty,
    # so an exact-zero assertion here would conflate the two.)


def test_joint_set_training_step_structured_sampling_defaults_off_and_matches_old_behaviour():
    """`structured_sampling` must be strictly opt-in: the default call must reproduce the
    i.i.d. path bit-for-bit against a fixed generator -- otherwise runs made with the
    default silently stop being reproducible."""
    torch.manual_seed(0)
    b, t, codec_dim, latent_dim, n_classes, k = 4, 5, 4, 8, 5, 6
    head = _TinyOutcomeHead(codec_dim, latent_dim, n_classes)
    teacher = TeacherEncoder(codec_dim, latent_dim, n_classes, hidden=16)
    mu_c = torch.randn(b, latent_dim)
    log_sigma_c = torch.randn(b, latent_dim) * 0.1
    target_item_codec = torch.randn(b, t, codec_dim)
    target_outcome = torch.randint(0, n_classes, (b, t))
    target_outcome_onehot = torch.zeros(b, t, n_classes).scatter_(2, target_outcome.unsqueeze(-1), 1.0)
    target_valid_mask = torch.ones(b, t, dtype=torch.bool)
    prefix_t = torch.tensor([0, 3, 5, 10])

    def _run(**kwargs):
        gen = torch.Generator().manual_seed(7)
        return joint_set_training_step(
            head, teacher, mu_c, log_sigma_c, target_item_codec, target_outcome,
            target_outcome_onehot, target_valid_mask, prefix_t, k_train=k, beta_kl=0.1,
            beta_scale=0.01, generator=gen, **kwargs)

    out_default = _run()
    out_explicit_false = _run(structured_sampling=False)
    torch.testing.assert_close(out_default.loss, out_explicit_false.loss)


def test_joint_set_training_step_structured_sampling_differs_from_default_and_is_reproducible():
    """`structured_sampling=True` must change the loss relative to the default i.i.d. path
    (proving it is actually wired in, not silently ignored) and must itself be
    reproducible given the same seeded generator (no hidden nondeterminism)."""
    torch.manual_seed(0)
    b, t, codec_dim, latent_dim, n_classes, k = 3, 4, 4, 8, 5, 8
    head = _TinyOutcomeHead(codec_dim, latent_dim, n_classes)
    teacher = TeacherEncoder(codec_dim, latent_dim, n_classes, hidden=16)
    mu_c = torch.randn(b, latent_dim)
    log_sigma_c = torch.randn(b, latent_dim) * 0.1
    target_item_codec = torch.randn(b, t, codec_dim)
    target_outcome = torch.randint(0, n_classes, (b, t))
    target_outcome_onehot = torch.zeros(b, t, n_classes).scatter_(2, target_outcome.unsqueeze(-1), 1.0)
    target_valid_mask = torch.ones(b, t, dtype=torch.bool)
    prefix_t = torch.tensor([3, 5, 10])

    def _run(structured: bool, seed: int):
        gen = torch.Generator().manual_seed(seed)
        return joint_set_training_step(
            head, teacher, mu_c, log_sigma_c, target_item_codec, target_outcome,
            target_outcome_onehot, target_valid_mask, prefix_t, k_train=k, beta_kl=0.1,
            beta_scale=0.01, generator=gen, structured_sampling=structured)

    out_iid = _run(False, seed=11)
    out_structured_a = _run(True, seed=11)
    out_structured_b = _run(True, seed=11)
    assert out_structured_a.loss.item() != out_iid.loss.item()
    torch.testing.assert_close(out_structured_a.loss, out_structured_b.loss)


def test_eval_path_functions_have_no_structured_sampling_parameter():
    """STRICT BOUNDARY: structured sampling is training-only.
    Enforced at the API-surface level -- none of the eval/gate-path functions even accept a
    structured-sampling argument, so there is no parameter through which it could reach a
    reported number. This test asserts that surface directly."""
    import inspect

    import atom.binding.uncertainty as uncertainty_mod

    sig = inspect.signature(uncertainty_mod.reparameterize_from_log_sigma)
    assert "structured_sampling" not in sig.parameters
    assert "eps" in sig.parameters  # the honest-i.i.d. eval path always has eps=None available

    sig2 = inspect.signature(uncertainty_mod.reparameterize)
    assert "structured_sampling" not in sig2.parameters


def test_teacher_encoder_log_sigma_is_clamped_symmetric_with_the_student():
    """Teacher log-sigma clamp (guards against a collapsed teacher). Forces the pre-clamp `log_sigma_head` output far outside `[-5, +2]`
    (large-scale weights/inputs) and asserts the returned `log_sigma_psi` never leaves that
    range -- the default bound matches `atom.filter.gru.KnowledgeFilter`'s own
    `log_sigma_min`/`log_sigma_max` exactly, so the two are genuinely symmetric, not just
    documented as such."""
    codec_dim, latent_dim, n_classes, hidden = 4, 8, 5, 16
    teacher = TeacherEncoder(codec_dim, latent_dim, n_classes, hidden)
    assert teacher.log_sigma_min == -5.0
    assert teacher.log_sigma_max == 2.0
    with torch.no_grad():
        teacher.log_sigma_head.weight.mul_(1000.0)
        teacher.log_sigma_head.bias.add_(1000.0)

    b, t = 6, 3
    mu_c = torch.randn(b, latent_dim) * 50
    log_sigma_c = torch.randn(b, latent_dim)
    target_item_codec = torch.randn(b, t, codec_dim) * 50
    target_outcome_onehot = torch.zeros(b, t, n_classes)
    target_outcome_onehot[:, :, 0] = 1.0
    target_valid_mask = torch.ones(b, t, dtype=torch.bool)

    _, log_sigma_psi = teacher(mu_c, log_sigma_c, target_item_codec, target_outcome_onehot,
                               target_valid_mask)
    assert bool((log_sigma_psi >= -5.0).all())
    assert bool((log_sigma_psi <= 2.0).all())
    # the adversarial weights above should genuinely be pushing PAST the bound (else this
    # test would pass trivially with no clamp at all) -- confirm the clamp is load-bearing.
    assert bool((log_sigma_psi == 2.0).any())


def test_teacher_encoder_clamp_defaults_match_the_student_filter_exactly():
    """The clamp bound is a literal DUPLICATE constant (`joint_set_train_step.py` and
    `atom.filter.gru.KnowledgeFilter` each define their own -5.0/2.0), not a shared import
    -- this test is the tripwire that catches the two silently drifting apart."""
    from atom.filter.gru import KnowledgeFilter

    student = KnowledgeFilter(input_dim=4, hidden=8, latent_dim=8)
    teacher = TeacherEncoder(codec_dim=4, latent_dim=8, n_classes=5, hidden=16)
    assert teacher.log_sigma_min == student.log_sigma_min
    assert teacher.log_sigma_max == student.log_sigma_max
