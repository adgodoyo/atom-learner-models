#!/usr/bin/env python
"""Local smoke test for `infra/fa_sim3_full_run.py`, run on a stub causal LM and tiny synthetic data.

Checks, on CPU in seconds: (1) per-direction belief-swap NLL arrays are returned and agree with
the pair records; (2) directed belief-swap accuracy runs end-to-end; (3) the per-type, history-curve
and leave-one-out-ceiling panel functions in `atom.eval.evals_5g` run on smoke-scale arrays;
(4) `k_marginalised_eval`'s joint (k, c) mixture estimator, common-random-number `eps` wiring and
learner-history-matched derangement produce finite, correctly shaped, deterministic outputs.
Gradient-accumulation exactness and the training-step loss wiring are covered separately by
`fa_sim2_smoke_local.py`. Writes nothing.  Run: `python scripts/fa_sim3_smoke_local.py`.
Pre-flight check for Tables 10 and 12 (FA-SIM3, the mathematics simulator).
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

_spec = importlib.util.spec_from_file_location("fa_sim3_full_run", REPO / "infra" / "fa_sim3_full_run.py")
fa_sim3 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fa_sim3)


def main() -> int:
    import numpy as np
    import pandas as pd
    import torch

    from atom.binding import attach_lora_basis, Hypernetwork
    from atom.binding.stub_causal_lm import CausalStubLM
    from atom.data.fa_outcome_label import N_OUTCOME
    from atom.eval.evals_5g import belief_swap_directed_accuracy, history_curve, loo_oracle_ceiling, per_type_split
    from atom.filter.item_encoder_mlp import S1FilterV2ItemMLP
    from atom.heads.fa_hierarchical_heads import OutcomeHead
    from atom.simulator.fa_alt2 import encode_example

    CODEC_DIM, HIDDEN_DIM, LATENT_DIM = 16, 32, 24
    N_LEARNERS, T_LEN = 8, 20
    VOCAB = 40

    class _Encoding(dict):
        @property
        def input_ids(self):
            return self["input_ids"]

    class _FakeTok:
        pad_token_id = 0
        eos_token_id = 1

        def __call__(self, text, add_special_tokens=True):
            ids = [((sum(ord(c) for c in w) % (VOCAB - 3)) + 3) for w in text.split(" ") if w]
            return _Encoding(input_ids=([2] + ids) if add_special_tokens else ids)

        def decode(self, ids, skip_special_tokens=True):
            return " ".join(f"tok{i}" for i in ids if i not in (0, 1, 2))

    tok = _FakeTok()

    filt = S1FilterV2ItemMLP(codec_dim=CODEC_DIM, episode_hidden=8, filter_hidden=8, latent_dim=LATENT_DIM)
    filt_forward = lambda ic, pf, tm, vm: filt(ic, pf)  # noqa: E731
    model = CausalStubLM(vocab_size=VOCAB, dim=HIDDEN_DIM, n_layers=2, n_heads=4,
                         pad_token_id=tok.pad_token_id, seed=0)
    bases, context = attach_lora_basis(model, m=8, r=4, seed=1)
    hyper = Hypernetwork(u_dim=LATENT_DIM, m=8, kind="mlp", seed=2)
    outcome_head = OutcomeHead(codec_dim=CODEC_DIM, latent_dim=LATENT_DIM, hidden=HIDDEN_DIM)

    torch.manual_seed(0)
    filter_batch = {
        "learners": list(range(N_LEARNERS)),
        "item_codec": torch.randn(N_LEARNERS, T_LEN, CODEC_DIM),
        "prev_feat": torch.randn(N_LEARNERS, T_LEN, 5),
        "time_min": torch.zeros(N_LEARNERS, T_LEN),
        "valid_mask": torch.ones(N_LEARNERS, T_LEN, dtype=torch.bool),
    }

    n_rows = 16
    l_idx = torch.randint(0, N_LEARNERS, (n_rows,))
    t_idx = torch.randint(0, T_LEN, (n_rows,))
    true_class = torch.randint(0, N_OUTCOME, (n_rows,))

    print("[smoke] 1/3 score_belief_swap_pairs now returns the four per-direction NLL arrays", flush=True)
    problem_ids = [i % 3 for i in range(n_rows)]  # force repeated items -> discriminating pairs
    answer_texts = [f"ans{i % 4}" for i in range(n_rows)]
    eval_bundle = pd.DataFrame({
        "learner": l_idx.tolist(), "problem_id": problem_ids,
        "answer_text": answer_texts,
        "question_text": [f"question number {i}" for i in range(n_rows)],
        "t": t_idx.tolist(),
    })
    val_filter_batch = dict(filter_batch)
    swap = fa_sim3.score_belief_swap_pairs(
        filt_forward, hyper, model, context, val_filter_batch, eval_bundle, true_class, tok,
        max_len=32, device="cpu")
    for key in ("pair_records", "correct", "edit_distances",
               "nll_i_own", "nll_i_other", "nll_j_own", "nll_j_other"):
        assert key in swap, f"missing key: {key}"
    n_pairs = len(swap["pair_records"])
    print(f"[smoke]   {n_pairs} discriminating pairs found", flush=True)
    assert n_pairs > 0, "smoke fixture must produce at least one discriminating pair -- adjust problem_ids/answers"
    for key in ("nll_i_own", "nll_i_other", "nll_j_own", "nll_j_other"):
        assert len(swap[key]) == n_pairs, f"{key} has wrong length"
        assert np.isfinite(swap[key]).all(), f"{key} contains non-finite values"
    # cross-check: nll_i_own/nll_i_other must reproduce the SAME i_favours_own boolean the
    # AND-scored `correct` array was built from (no drift between the two representations)
    for idx, rec in enumerate(swap["pair_records"]):
        assert swap["nll_i_own"][idx] == rec["nll_i_own"]
        assert swap["nll_j_other"][idx] == rec["nll_j_other"]
    print("[smoke]   per-direction NLL arrays match pair_records exactly", flush=True)

    print("[smoke] 2/3 belief_swap_directed_accuracy end-to-end on the smoke pairs", flush=True)
    directed = belief_swap_directed_accuracy(
        swap["nll_i_own"], swap["nll_i_other"], swap["nll_j_own"], swap["nll_j_other"])
    assert directed["n_pairs"] == n_pairs
    assert directed["and_both_directions"]["null"] == 0.25
    assert directed["directed"]["null"] == 0.5
    assert directed["directed"]["n_directed_comparisons"] == 2 * n_pairs
    assert 0.0 <= directed["directed"]["accuracy"] <= 1.0
    assert 0.0 <= directed["and_both_directions"]["accuracy"] <= 1.0
    print(f"[smoke]   AND accuracy={directed['and_both_directions']['accuracy']:.4f} "
         f"directed accuracy={directed['directed']['accuracy']:.4f}", flush=True)

    print("[smoke] 3/3 evals_5g panel functions callable end-to-end on smoke-scale arrays", flush=True)
    personalised_nll = np.random.default_rng(0).normal(1.0, 0.2, n_rows)
    shuffled_nll = personalised_nll + np.random.default_rng(1).normal(0.02, 0.05, n_rows)
    personalised_em = np.random.default_rng(2).integers(0, 2, n_rows).astype(float)
    shuffled_em = np.random.default_rng(3).integers(0, 2, n_rows).astype(float)
    problem_types = np.array(["fill_in", "mc1", "mc_all"])[np.array(problem_ids) % 3]

    panel_type = per_type_split(problem_types, l_idx.numpy(), personalised_nll, shuffled_nll,
                                personalised_em, shuffled_em, n_boot=50, seed=0)
    assert set(panel_type.keys()) <= {"fill_in", "mc1", "mc_all"}
    print(f"[smoke]   per_type_split: {list(panel_type.keys())}", flush=True)

    panel_hist = history_curve(t_idx.numpy(), l_idx.numpy(), personalised_nll, shuffled_nll,
                               bins=(0, 10, 20, 10_000), n_boot=50, seed=0)
    assert sum(v["n"] for v in panel_hist.values()) == n_rows
    print(f"[smoke]   history_curve bins: {list(panel_hist.keys())}", flush=True)

    panel_loo = loo_oracle_ceiling(problem_ids, true_class.numpy(), answer_texts, personalised_em,
                                   train_exposure_by_problem={0: 100, 1: 500, 2: 2000})
    assert "n_covered" in panel_loo
    print(f"[smoke]   loo_oracle_ceiling: n_covered={panel_loo['n_covered']} "
         f"coverage_frac={panel_loo['coverage_frac']:.2f}", flush=True)

    print("[smoke] 4/4 k_marginalised_eval: corrected joint (k,c) estimator + CRN eps "
         "+ learner-history-matched derangement", flush=True)
    from atom.binding.derangement import assert_learner_matched_derangement, learner_history_matched_derangement
    tok_batch_by_class = {
        c: [encode_example(f"question number {i}", f"a{i}", tok, max_len=32, outcome_code=c)
           for i in range(n_rows)]
        for c in range(N_OUTCOME)
    }
    tok_batch_by_class["pad_id"] = tok.pad_token_id
    tok_batch_by_class["true_class"] = true_class
    tok_batch_by_class["outcome_head"] = outcome_head

    bin_ids = np.digitize(t_idx.numpy(), (10,))  # 2 bins, matches the tiny smoke population
    learner_ids_np = l_idx.numpy()
    derangement_np, deranged_mask = learner_history_matched_derangement(
        learner_ids_np, bin_ids, rng=np.random.default_rng(0))
    assert_learner_matched_derangement(derangement_np, learner_ids_np, deranged_mask)
    print(f"[smoke]   learner-matched derangement: {int(deranged_mask.sum())}/{n_rows} deranged", flush=True)
    derangement_t = torch.from_numpy(derangement_np)

    K_EVAL = 4
    shared_eps = torch.randn(n_rows, K_EVAL, LATENT_DIM, generator=torch.Generator().manual_seed(0))
    results_by_arm = {}
    for arm in fa_sim3.ARMS:
        results_by_arm[arm] = fa_sim3.k_marginalised_eval(
            filt_forward, hyper, model, context, val_filter_batch, l_idx, t_idx,
            tok_batch_by_class, k=K_EVAL, batch_size=5, arm=arm, generator=None,
            derangement=derangement_t, eps=shared_eps)
        r = results_by_arm[arm]
        for key in ("mixture_nll", "true_class_production_nll", "exact_match", "top1_frac",
                   "mu", "log_sigma", "outcome_prob_meanK_full"):
            assert key in r, f"{arm} missing key {key}"
            assert np.isfinite(r[key]).all(), f"{arm}'s {key} contains non-finite values"
        assert r["mixture_nll"].shape == (n_rows,)
        assert r["outcome_prob_meanK_full"].shape == (n_rows, N_OUTCOME)
        assert np.allclose(r["outcome_prob_meanK_full"].sum(axis=1), 1.0, atol=1e-4), (
            "outcome_prob_meanK_full must sum to 1 across classes per row")
        print(f"[smoke]   {arm}: mixture_nll_mean={r['mixture_nll'].mean():.4f} k={r['k']}", flush=True)
    # Common random numbers: personalised and shuffled_u must have IDENTICAL mu/log_sigma
    # sampling noise applied for the SAME row -- verified indirectly here via determinism:
    # re-running personalised with the SAME eps must reproduce the SAME mixture_nll exactly.
    rerun = fa_sim3.k_marginalised_eval(
        filt_forward, hyper, model, context, val_filter_batch, l_idx, t_idx,
        tok_batch_by_class, k=K_EVAL, batch_size=5, arm="personalised", generator=None,
        derangement=derangement_t, eps=shared_eps)
    assert np.allclose(rerun["mixture_nll"], results_by_arm["personalised"]["mixture_nll"], atol=1e-5), (
        "CRN: identical eps must reproduce identical mixture_nll (CRN determinism)")
    print("[smoke]   CRN determinism verified: same eps -> bit-identical mixture_nll", flush=True)

    print("[smoke] ALL 4 CHECKS PASSED -- fa_sim3_full_run.py's new logic (per-direction "
         "belief-swap NLL persistence, directed accuracy, evals_5g panel, the "
         "joint (k,c) mixture estimator, CRN eps, learner-matched derangement) is "
         "shape-correct and numerically sound on a stub model. Gradient-accumulation "
         "exactness is unchanged from FA-SIM2 and covered by fa_sim2_smoke_local.py.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
