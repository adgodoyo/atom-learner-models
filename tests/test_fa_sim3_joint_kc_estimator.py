"""FA-SIM3 requirement 7 -- the corrected joint (k, c) mixture estimator
(`atom.simulator.fa_alt2.joint_kc_mixture_nll`), replacing SIM2's K-factorised formula
(`infra/fa_sim2_full_run.py:297-306`). Pure-math unit tests, no model/filter -- extends
`test_fa_06c_hierarchical_train.py::test_mixture_nll_formula_matches_a_brute_force_
probability_space_reference` (FA-6c's K=1 case) to K>1, exercising exactly the coupling
FA-6c never had to handle. Per the SEMANTIC CANARIES rule, this asserts the ESTIMATOR IS
what it claims to be (a correct probability-space mixture) via independent brute-force
computation, not merely that the function runs and returns a finite number.
"""

from __future__ import annotations

import torch

from atom.simulator.fa_alt2 import joint_kc_mixture_nll


def _brute_force_joint_mixture_nll(log_p_class_kc: torch.Tensor, log_p_seq_given_kc: torch.Tensor,
                                   n_valid_tok: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The textbook-literal reading, exponentiated directly into raw probability space:
    p(y) = (1/K) * sum_k sum_c p(c|u_k) * p(y|c,u_k), computed with a plain Python loop
    over (c, k) -- deliberately not vectorised the same way as the function under test,
    so an indexing bug in one would not be mirrored in the other."""
    c_dim, k_dim, b_dim = log_p_class_kc.shape
    p_class = log_p_class_kc.exp()
    p_seq = log_p_seq_given_kc.exp()
    total = torch.zeros(b_dim, dtype=torch.float64)
    for c in range(c_dim):
        for k in range(k_dim):
            total += (p_class[c, k].double() * p_seq[c, k].double())
    total /= k_dim
    mixture_nll = -torch.log(total).float()
    return mixture_nll, mixture_nll / n_valid_tok.clamp(min=1).float()


def test_joint_estimator_matches_brute_force_probability_space_k1_c5():
    """K=1 sanity check: with only one sample, the joint estimator must degenerate to
    exactly FA-6c's own K=1 formula (`log p(y) = logsumexp_c[log p(c|u) + log p(y|c,u)]`,
    no `-log K` correction needed at K=1)."""
    log_p_class = torch.log(torch.tensor([[0.7, 0.1, 0.1, 0.05, 0.05]])).T.unsqueeze(1)  # [5,1,1]
    sum_nll = torch.tensor([[1.0], [3.0], [4.0], [5.0], [6.0]]).unsqueeze(1)  # [5,1,1]
    log_p_seq = -sum_nll
    n_valid_tok = torch.tensor([4.0])

    mixture_nll, mixture_nll_norm = joint_kc_mixture_nll(log_p_class, log_p_seq, n_valid_tok)
    brute_nll, brute_norm = _brute_force_joint_mixture_nll(log_p_class, log_p_seq, n_valid_tok)
    assert torch.allclose(mixture_nll, brute_nll, atol=1e-5)
    assert torch.allclose(mixture_nll_norm, brute_norm, atol=1e-5)


def test_joint_estimator_matches_brute_force_probability_space_k4_c5_multirow():
    """The real case: K>1, C>1 (N_OUTCOME=5), multiple rows with distinct token counts --
    exact enumeration in probability space (float64 for headroom), independent of the
    function under test's own vectorisation."""
    torch.manual_seed(0)
    c_dim, k_dim, b_dim = 5, 4, 3
    log_p_class_kc = torch.log_softmax(torch.randn(c_dim, k_dim, b_dim) * 2.0, dim=0)
    sum_nll = torch.rand(c_dim, k_dim, b_dim) * 8.0 + 0.5  # positive NLLs, varied by (c,k,b)
    log_p_seq_given_kc = -sum_nll
    n_valid_tok = torch.tensor([3.0, 7.0, 12.0])

    mixture_nll, mixture_nll_norm = joint_kc_mixture_nll(log_p_class_kc, log_p_seq_given_kc, n_valid_tok)
    brute_nll, brute_norm = _brute_force_joint_mixture_nll(log_p_class_kc, log_p_seq_given_kc, n_valid_tok)
    assert torch.allclose(mixture_nll, brute_nll, atol=1e-4)
    assert torch.allclose(mixture_nll_norm, brute_norm, atol=1e-4)


def test_joint_estimator_length_normalisation_happens_once_at_the_end():
    """The length-normalised output must equal the RAW mixture_nll divided by
    n_valid_tok, computed AFTER combining (c, k) -- not derived from any per-(c,k)
    length-normalised intermediate. A regression that reintroduces per-cell
    normalisation before the logsumexp would still produce SOME number here; this
    isolates that the two returned tensors are related by exactly one division."""
    torch.manual_seed(1)
    c_dim, k_dim, b_dim = 5, 8, 4
    log_p_class_kc = torch.log_softmax(torch.randn(c_dim, k_dim, b_dim), dim=0)
    log_p_seq_given_kc = -(torch.rand(c_dim, k_dim, b_dim) * 10.0)
    n_valid_tok = torch.tensor([2.0, 5.0, 9.0, 21.0])

    mixture_nll, mixture_nll_norm = joint_kc_mixture_nll(log_p_class_kc, log_p_seq_given_kc, n_valid_tok)
    assert torch.allclose(mixture_nll_norm, mixture_nll / n_valid_tok, atol=1e-6)


def test_would_have_caught_sim2s_k_decoupled_formula():
    """Would-have-caught regression check (SEMANTIC CANARIES convention): reconstruct
    SIM2's ORIGINAL formula -- average class-prob over K, average token-likelihood over
    K, SEPARATELY, then combine -- on inputs where class probability and token
    likelihood are ANTI-correlated across k (the class the model favours at k=0 is the
    class the answer least resembles, and vice versa at k=1). Under genuine (k, c)
    coupling this anti-correlation should visibly worsen (raise) the joint mixture NLL
    relative to the K-factorised formula, which cannot see the anti-correlation at all
    because it never evaluates a (c, k) pair jointly -- proving the two formulas
    disagree, and disagree in the specific direction the requirement describes as lost
    coupling, not merely differ by floating-point noise."""
    # 2 classes, 2 samples, 1 row. k=0 strongly favours class 0 AND fits class-0 tokens
    # well; k=1 strongly favours class 1 AND fits class-1 tokens well -- i.e. the joint
    # (c=argmax class, k) pairs are the GOOD-fitting ones. This is the case where
    # coupling matters most: a decoupled formula averages away the alignment.
    log_p_class_kc = torch.log(torch.tensor([
        [[0.9], [0.1]],   # class 0: p=0.9 at k=0, p=0.1 at k=1
        [[0.1], [0.9]],   # class 1: p=0.1 at k=0, p=0.9 at k=1
    ]))  # [C=2, K=2, B=1]
    sum_nll = torch.tensor([
        [[0.5], [5.0]],   # class 0: cheap (well-fit) at k=0, expensive at k=1
        [[5.0], [0.5]],   # class 1: expensive at k=0, cheap (well-fit) at k=1
    ])  # [C=2, K=2, B=1]
    log_p_seq_given_kc = -sum_nll
    n_valid_tok = torch.tensor([4.0])

    joint_nll, _ = joint_kc_mixture_nll(log_p_class_kc, log_p_seq_given_kc, n_valid_tok)

    # SIM2's own formula, reconstructed exactly (fa_sim2_full_run.py:297-306): average
    # token-likelihood over K first (per class), average class-prob over K separately,
    # THEN combine.
    log_p_tok_given_c = torch.logsumexp(-sum_nll, dim=1) - float(torch.log(torch.tensor(2.0)))  # [C,B]
    p_class_meanK = log_p_class_kc.exp().mean(dim=1)  # [C,B]
    sim2_mixture_log_prob = torch.logsumexp(torch.log(p_class_meanK.clamp_min(1e-12)) + log_p_tok_given_c, dim=0)
    sim2_nll = -sim2_mixture_log_prob  # raw scale, matching joint_nll's own raw scale

    # The two formulas must disagree materially on this deliberately anti-correlation-
    # sensitive input -- proving this test has power to distinguish them.
    assert not torch.allclose(joint_nll, sim2_nll, atol=0.05), (
        "the joint and K-factorised formulas agree on an input specifically constructed "
        "to expose their coupling difference -- this test has no power; the (c,k) "
        "alignment is too weak on this input, sharpen it")
    # The coupled formula must find the sequence MORE likely (lower NLL) than the
    # decoupled one, because it can actually see that the good class fit and the good
    # token fit occur at the SAME k -- exactly the information SIM2's formula discards.
    assert joint_nll.item() < sim2_nll.item()
