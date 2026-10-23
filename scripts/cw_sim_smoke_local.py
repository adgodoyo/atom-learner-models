#!/usr/bin/env python
"""Local smoke test for the CodeWorkout generative simulator's training step.

Runs one or two optimiser steps of `infra/cw_sim_full_run.py`'s training loss on a real
local CodeWorkout manifest, with a small stub backbone (`CausalStubLM`) and the real
tokenizer, and checks:

1. `load_and_freeze_ratchet` with the CodeWorkout classifier checkpoint in both modes:
   FROZEN (attempt_encoder/filt receive no gradient; hyper, LoRA basis and
   test_vector_head do) and JOINT (attempt_encoder/filt also receive gradient).
2. `test_vector_batch_loss` against a small synthetic label lookup built from the batch's
   own (learner, problem_id, code_state_id) triples, so the test does not depend on the
   output of `cw_sim_test_vector_labels.py`.
3. The combined loss (token, outcome and test-vector weights) for each filter mode, with
   the same non-finite checks that `main()` of the training script uses.

Inputs: the local CodeWorkout manifest and code-vector caches (see the CLI flags).
Outputs: none; prints progress and exits non-zero on the first failed check.
Run: `python scripts/cw_sim_smoke_local.py`. Serves Table 11 (run before training).
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import time

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "infra"))
sys.path.insert(0, str(REPO / "scripts"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

import cw_sim_core as cw5  # noqa: E402
import cw_sim_full_run as sim  # noqa: E402
from atom.binding.stub_causal_lm import CausalStubLM  # noqa: E402
from atom.simulator.cw_item_mlp import OutcomeHeadItemMLP  # noqa: E402
from atom.utils.logging import default_device  # noqa: E402
from cw_sim_smoke_common import _pick_smoke_batch  # noqa: E402


def _build_synthetic_tv_labels(attempts_batch, max_test_cases=8):
    """A tiny label lookup keyed EXACTLY like the real offline pass's output, but built
    directly from this smoke batch's own (learner, problem_id, code_state_id) triples --
    never depends on `artifacts/cw_sim_test_vector_labels.npz` existing. Keyed per
    attempt (code_state_id included), matching the real labelling script, and not
    deduplicated to one row per (learner, problem_id): different attempts on the same
    problem each get their own, independently drawn label."""
    rng = np.random.default_rng(0)
    lookup = {}
    for _, row in attempts_batch.drop_duplicates(["learner", "problem_id", "code_state_id"]).iterrows():
        n_cases = int(rng.integers(2, max_test_cases))
        tv = np.zeros(max_test_cases, dtype=bool)
        vm = np.zeros(max_test_cases, dtype=bool)
        vm[:n_cases] = True
        tv[:n_cases] = rng.random(n_cases) > 0.5
        key = (str(row["learner"]), int(row["problem_id"]), str(row["code_state_id"]))
        lookup[key] = {
            "test_vector": tv, "valid_mask": vm, "n_cases": n_cases, "compiled": True,
        }
    problems = set(int(p) for p in attempts_batch["problem_id"].unique())
    return {"lookup": lookup, "max_test_cases": max_test_cases, "headline_problem_ids": problems}


def _grad_is_nonzero(module) -> bool:
    return any(p.grad is not None and bool((p.grad != 0).any()) for p in module.parameters())


def _run_one_step(m, attempts_batch, episodes_batch, manifest, device, tv_labels, filter_mode,
                  token_loss_weight, outcome_loss_weight, test_vector_loss_weight, optimizer,
                  micro_batch_size):
    optimizer.zero_grad()
    out = cw5.forward_outcome(m, attempts_batch, episodes_batch, manifest, cw5.CW5_DIMS, device, free_bits=0.05)
    if filter_mode == "joint":
        outcome_total = outcome_loss_weight * out["l_outcome"] + 0.01 * out["kl"]
        assert torch.isfinite(outcome_total), "non-finite outcome loss"
        outcome_total.backward(retain_graph=True)

    tv_loss, tv_stats = sim.test_vector_batch_loss(m, attempts_batch, out, manifest, tv_labels, device)
    assert tv_stats["n_matched"] > 0, "smoke batch's synthetic labels must match at least one row"
    assert torch.isfinite(tv_loss), "non-finite test-vector loss"
    (test_vector_loss_weight * tv_loss).backward(retain_graph=True)

    ep_learner_idx, ep_t_idx = cw5._distinct_episode_positions(
        episodes_batch, out["ep_batch"], out["attempt_batch"]["episode_ids"], device=device)
    l_prod = cw5.production_backward_microbatched(
        m, attempts_batch, manifest, out["ep_batch"], ep_learner_idx, ep_t_idx,
        out["attempt_batch"]["attempt_episode_row_idx"], device, k=1,
        micro_batch_size=micro_batch_size, scale=token_loss_weight)
    assert l_prod == l_prod, "l_prod is NaN"
    return out, l_prod, tv_loss.item()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default=None)
    ap.add_argument("--code-map", default="cache/cw_03_vectors/cw_code_state_hash_map_full.npz")
    ap.add_argument("--diff-map", default="cache/cw_03_vectors/cw_diff_hash_map_full.npz")
    ap.add_argument("--codealone-meta", default="cache/cw_03_vectors/codealone/meta.json")
    ap.add_argument("--codealone-states", default="cache/cw_03_vectors/codealone/states_mean.npy")
    ap.add_argument("--diff-meta", default="cache/cw_03_vectors/diff/meta.json")
    ap.add_argument("--diff-states", default="cache/cw_03_vectors/diff/states_mean.npy")
    ap.add_argument("--micro-batch-size", type=int, default=4)
    args = ap.parse_args()

    device = default_device()
    print(f"[cw-sim-smoke] device={device}", flush=True)

    t0 = time.time()
    manifest = cw5.load_cw5_manifest(args)
    print(f"[cw-sim-smoke] manifest loaded in {time.time()-t0:.1f}s: "
         f"{len(manifest['attempts'])} attempts, {len(manifest['train_learners'])} train / "
         f"{len(manifest['val_learners'])} val learners", flush=True)
    learner_split_sha256 = cw5._sha256(REPO / "splits" / "cw_learner_split.csv")

    # local_files_only=False: the tokenizer alone is a small, ungated download (unlike
    # the ~16GB gated full-weights snapshot that `load_backbone` refuses to fetch without
    # `allow_download=True`), so an empty Hugging Face cache does not block this test.
    tok = AutoTokenizer.from_pretrained("google/gemma-4-E4B", local_files_only=False)
    model = CausalStubLM(vocab_size=len(tok), dim=32, n_layers=2, n_heads=4,
                         pad_token_id=tok.pad_token_id, seed=0).to(device)
    m = cw5.build_model(cw5.CW5_DIMS, {"model": model, "tokenizer": tok}, seed=0, device=device,
                        item_encoder="mlp", sigma_parameterization="precision_accumulating")

    ratchet_path = pathlib.Path(sim.RATCHET_CHECKPOINT_DEFAULT)
    ratchet_sha256 = cw5._sha256(ratchet_path)
    assert ratchet_sha256 == sim.RATCHET_CHECKPOINT_SHA256, "classifier checkpoint sha256 does not match the expected value"
    sim.load_and_freeze_ratchet(m, ratchet_path, learner_split_sha256, freeze=True, device=device)
    print(f"[cw-sim-smoke] classifier checkpoint loaded and frozen, sha256={ratchet_sha256[:12]}...", flush=True)

    m["test_vector_head"] = OutcomeHeadItemMLP(
        codec_dim=cw5.CW5_DIMS["codec_dim"], latent_dim=cw5.CW5_DIMS["latent_dim"],
        hidden=cw5.CW5_DIMS["attempt_hidden"], n_classes=sim.MAX_TEST_CASES,
        prefix_dim=cw5.CW5_DIMS["attempt_hidden"]).to(device)

    attempts_batch, episodes_batch, batch_learners = _pick_smoke_batch(manifest)
    print(f"[cw-sim-smoke] batch: {len(attempts_batch)} attempts, learners={batch_learners}", flush=True)
    tv_labels = _build_synthetic_tv_labels(attempts_batch, max_test_cases=sim.MAX_TEST_CASES)
    print(f"[cw-sim-smoke] synthetic tv labels: {len(tv_labels['lookup'])} (learner,problem,attempt) rows", flush=True)

    # -- FROZEN mode: attempt_encoder/filt must get NO gradient; hyper/basis/test_vector_head must. --
    trainable_frozen = (list(m["hyper"].parameters())
                        + [p for b in m["bases"].values() for p in b.parameters()]
                        + list(m["test_vector_head"].parameters()))
    opt_frozen = torch.optim.Adam(trainable_frozen, lr=1e-2)
    out, l_prod, l_tv = _run_one_step(m, attempts_batch, episodes_batch, manifest, device, tv_labels,
                                      "frozen", 1.0, 0.3, 0.1, opt_frozen, args.micro_batch_size)
    print(f"[cw-sim-smoke] FROZEN step 1: l_prod={l_prod:.4f} l_tv={l_tv:.4f} "
         f"l_outcome={out['l_outcome'].item():.4f} (unused, diagnostic only)", flush=True)
    assert not any(p.grad is not None and bool((p.grad != 0).any()) for p in m["attempt_encoder"].parameters()), (
        "FROZEN mode: attempt_encoder received a nonzero gradient -- freeze is not actually frozen")
    assert not any(p.grad is not None and bool((p.grad != 0).any()) for p in m["filt"].parameters()), (
        "FROZEN mode: filt received a nonzero gradient -- freeze is not actually frozen")
    assert _grad_is_nonzero(m["test_vector_head"]), "FROZEN mode: test_vector_head got no gradient at step 1"
    opt_frozen.step()  # LoRA basis B moves off its zero-init, so step 2 must reach hyper too

    opt_frozen.zero_grad()
    out, l_prod, l_tv = _run_one_step(m, attempts_batch, episodes_batch, manifest, device, tv_labels,
                                      "frozen", 1.0, 0.3, 0.1, opt_frozen, args.micro_batch_size)
    print(f"[cw-sim-smoke] FROZEN step 2: l_prod={l_prod:.4f} l_tv={l_tv:.4f}", flush=True)
    assert not any(p.grad is not None and bool((p.grad != 0).any()) for p in m["attempt_encoder"].parameters()), (
        "FROZEN mode: attempt_encoder received a nonzero gradient at step 2 -- freeze is not actually frozen")
    assert not any(p.grad is not None and bool((p.grad != 0).any()) for p in m["filt"].parameters()), (
        "FROZEN mode: filt received a nonzero gradient at step 2 -- freeze is not actually frozen")
    assert _grad_is_nonzero(m["hyper"]), "FROZEN mode: hyper got no gradient at step 2 -- production loss is not training it"
    assert _grad_is_nonzero(m["test_vector_head"]), "FROZEN mode: test_vector_head got no gradient at step 2"
    print("[cw-sim-smoke] FROZEN mode gradient isolation verified: filter frozen, hyper+test_vector_head trained", flush=True)
    opt_frozen.step()

    # -- JOINT mode: fresh model instance, filter warm-started but NOT frozen. --
    model2 = CausalStubLM(vocab_size=len(tok), dim=32, n_layers=2, n_heads=4,
                          pad_token_id=tok.pad_token_id, seed=1).to(device)
    m2 = cw5.build_model(cw5.CW5_DIMS, {"model": model2, "tokenizer": tok}, seed=1, device=device,
                         item_encoder="mlp", sigma_parameterization="precision_accumulating")
    sim.load_and_freeze_ratchet(m2, ratchet_path, learner_split_sha256, freeze=False, device=device)
    m2["test_vector_head"] = OutcomeHeadItemMLP(
        codec_dim=cw5.CW5_DIMS["codec_dim"], latent_dim=cw5.CW5_DIMS["latent_dim"],
        hidden=cw5.CW5_DIMS["attempt_hidden"], n_classes=sim.MAX_TEST_CASES,
        prefix_dim=cw5.CW5_DIMS["attempt_hidden"]).to(device)
    trainable_joint = (list(m2["hyper"].parameters())
                       + [p for b in m2["bases"].values() for p in b.parameters()]
                       + list(m2["test_vector_head"].parameters())
                       + list(m2["attempt_encoder"].parameters()) + list(m2["filt"].parameters())
                       + list(m2["outcome_head"].parameters()))
    opt_joint = torch.optim.Adam(trainable_joint, lr=1e-2)
    out2, l_prod2, l_tv2 = _run_one_step(m2, attempts_batch, episodes_batch, manifest, device, tv_labels,
                                         "joint", 1.0, 0.3, 0.1, opt_joint, args.micro_batch_size)
    print(f"[cw-sim-smoke] JOINT step 1: l_prod={l_prod2:.4f} l_tv={l_tv2:.4f} "
         f"l_outcome={out2['l_outcome'].item():.4f}", flush=True)
    assert _grad_is_nonzero(m2["attempt_encoder"]), "JOINT mode: attempt_encoder got no gradient at step 1"
    assert _grad_is_nonzero(m2["filt"]), "JOINT mode: filt got no gradient at step 1"
    assert _grad_is_nonzero(m2["test_vector_head"]), "JOINT mode: test_vector_head got no gradient at step 1"
    opt_joint.step()  # LoRA basis B moves off its zero-init, so step 2 must reach hyper too

    opt_joint.zero_grad()
    out2, l_prod2, l_tv2 = _run_one_step(m2, attempts_batch, episodes_batch, manifest, device, tv_labels,
                                         "joint", 1.0, 0.3, 0.1, opt_joint, args.micro_batch_size)
    print(f"[cw-sim-smoke] JOINT step 2: l_prod={l_prod2:.4f} l_tv={l_tv2:.4f}", flush=True)
    assert _grad_is_nonzero(m2["attempt_encoder"]), "JOINT mode: attempt_encoder got no gradient at step 2"
    assert _grad_is_nonzero(m2["filt"]), "JOINT mode: filt got no gradient at step 2"
    assert _grad_is_nonzero(m2["hyper"]), "JOINT mode: hyper got no gradient at step 2"
    assert _grad_is_nonzero(m2["test_vector_head"]), "JOINT mode: test_vector_head got no gradient at step 2"
    print("[cw-sim-smoke] JOINT mode gradient flow verified: filter, hyper, test_vector_head all trained", flush=True)
    opt_joint.step()

    print("[cw-sim-smoke] ALL CHECKS PASSED", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
