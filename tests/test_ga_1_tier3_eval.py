"""Tests for the 2x2 / DiD evaluation contract of the LLM selector fine-tuned on the simulator's pricing (`ga_1_tier3_eval.py`)."""

from __future__ import annotations

import pathlib
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))

import ga_1_tier3_eval as E  # noqa: E402
import ga_1_tier3_states as S  # noqa: E402


def test_registered_arms_and_starts():
    assert E.LLM_ARMS == ("zeroshot_history", "zeroshot_no_history",
                          "trained_history", "trained_no_history")
    assert set(E.BLIND) == {"random", "max_entropy", "difficulty_matched",
                            "privileged_oracle_upper_bound"}
    assert S.EVAL_STARTS == (3, 10), "second evaluation start is 10 (not 8)"
    assert S.TRAIN_STARTS == (3, 10), "training draw moves with evaluation"
    assert S.BUDGETS == (3, 8, 16)


def test_rl2_actor_is_reference_only_and_never_a_blind_comparator():
    """The actor is quoted, never re-run and never paired."""
    assert "rl_actor" not in E.BLIND
    assert "rl_actor" not in E.LLM_ARMS


def test_both_evaluation_starts_are_banked_by_rl2():
    """The second start is 10 so that every evaluated start is checkable against the actor run (`rl_2.py`)."""
    import cw_planner_suite as cps
    assert set(S.EVAL_STARTS) <= set(cps.STARTS), (
        "an evaluated start absent from the actor run's STARTS would have no banked cell to "
        "check against and no actor reference")


def test_only_the_oracle_is_treated_as_seed_invariant():
    """Regression (seen in the zero-shot selector): max_entropy/difficulty_matched rank on seed-drawn eps."""
    assert E.SEED_INVARIANT == ("privileged_oracle_upper_bound",)
    for p in ("max_entropy", "difficulty_matched", "random"):
        assert p not in E.SEED_INVARIANT


def test_blind_reference_tolerance_matches_tier_0():
    assert E.RL2_TOLERANCE == 5e-5


def test_did_is_the_difference_of_the_two_history_gaps():
    """The primary mechanism read, computed on arrays whose answer is known."""
    rng = np.random.default_rng(0)
    n = 40
    zs_h = rng.normal(0.50, 0.05, n)
    zs_n = zs_h + 0.02                      # zero-shot: history helps a little
    tr_h = rng.normal(0.30, 0.05, n)
    tr_n = tr_h + 0.10                      # trained: history helps a lot
    did = (tr_h - tr_n) - (zs_h - zs_n)
    assert np.isclose(did.mean(), -0.08, atol=1e-9)
    # lower dlogit is better, so a NEGATIVE DiD means training widened the gap the good way
    assert did.mean() < 0


def test_did_is_zero_when_training_does_not_change_the_history_gap():
    rng = np.random.default_rng(1)
    n = 30
    zs_h = rng.normal(0.5, 0.05, n)
    zs_n = zs_h + 0.05
    tr_h = zs_h - 0.2                        # uniformly better, same gap
    tr_n = tr_h + 0.05
    did = (tr_h - tr_n) - (zs_h - zs_n)
    assert np.allclose(did, 0.0), "a uniform improvement must not register as a mechanism"


def test_bootstrap_returns_mean_and_interval_and_is_deterministic():
    v = np.random.default_rng(3).normal(0.1, 0.05, 60)
    a = E._boot(v, 11)
    b = E._boot(v, 11)
    assert a == b, "CI must be reproducible under a fixed seed"
    assert len(a) == 3 and a[1] <= a[0] <= a[2]


def test_cell_extraction_takes_the_final_step_of_each_trace():
    rows = [{m: [9.0, 5.0, float(i)] for m in E.METRICS} for i in range(4)]
    cell = E._cell(rows)
    for m in E.METRICS:
        assert list(cell[m]) == [0.0, 1.0, 2.0, 3.0]


# ---------------------------------------------------------------- adapter-effect canary
#
# A silent load failure, or an adapter whose B never left zero, makes every trained arm
# bit-identical to its zero-shot twin.  The DiD then reads exactly 0.0 and would be
# written up as "no mechanism" -- a claim about the model that is really a fact about
# the loader.


def _toy_adapters(n_layers=3, seed=0):
    import torch.nn as nn

    import ga_1_tier3_lora as L

    return {f"layer.{i}": L.PlainLoRALinear(nn.Linear(8, 8), r=2, alpha=4, seed=seed + i)
            for i in range(n_layers)}


def test_adapter_canary_rejects_an_untrained_checkpoint():
    """B at its zero initialisation means training never moved the adapter."""
    import ga_1_tier3_eval as E
    import ga_1_tier3_lora as L

    adapters = _toy_adapters()
    ckpt = {"adapter": L.adapter_state_dict(adapters)}      # B still zero-init
    try:
        E.adapter_effect_canary(adapters, ckpt)
    except RuntimeError as exc:
        assert "zero initialisation" in str(exc)
    else:
        raise AssertionError("canary accepted an adapter that is an exact no-op")


def test_adapter_canary_accepts_a_trained_checkpoint_and_loads_it():
    import torch

    import ga_1_tier3_eval as E
    import ga_1_tier3_lora as L

    adapters = _toy_adapters()
    with torch.no_grad():
        for mod in adapters.values():
            mod.lora_b.add_(torch.randn_like(mod.lora_b) * 0.05)
    ckpt = {"adapter": L.adapter_state_dict(adapters)}

    with torch.no_grad():                                    # wipe, then reload
        for mod in adapters.values():
            mod.lora_b.zero_()
    out = E.adapter_effect_canary(adapters, ckpt)

    assert out["lora_b_nonzero_layers"] == len(adapters)
    assert out["lora_b_norm_mean"] > 0
    assert out["load_worst_relative_error"] <= 0.02
    for name, mod in adapters.items():
        assert torch.allclose(mod.lora_b, ckpt["adapter"][f"{name}.lora_b"]), name


def test_adapter_canary_catches_a_partially_zero_load():
    """One layer silently failing to load must not pass as a healthy adapter."""
    import torch

    import ga_1_tier3_eval as E
    import ga_1_tier3_lora as L

    adapters = _toy_adapters()
    with torch.no_grad():
        for mod in adapters.values():
            mod.lora_b.add_(torch.randn_like(mod.lora_b) * 0.05)
    ckpt = {"adapter": L.adapter_state_dict(adapters)}

    original = L.load_adapter_state

    def _lossy(ad, state):                                   # drop one layer's B
        original(ad, state)
        with torch.no_grad():
            next(iter(ad.values())).lora_b.zero_()

    E.load_adapter_state = _lossy
    try:
        try:
            E.adapter_effect_canary(adapters, ckpt)
        except RuntimeError as exc:
            assert "B norms" in str(exc)
        else:
            raise AssertionError("canary missed a layer that failed to load")
    finally:
        E.load_adapter_state = original


# --------------------------------------------- agreement with the actor run, disclosed
#
# A fatal version of this check would discard every completed rollout (~1 GPU-hour of
# evaluation) when it fails.  A standalone CPU-only recomputation
# (scripts/ga_1_tier3_diag_rl2_drift.py) showed the drift is narrow -- concentrated in
# `privileged_oracle_upper_bound` at the longest budget, in `logloss` only -- consistent
# with an ordinary compute-path difference, not a construction bug. The check must
# always run and always be recorded, but never abort the run.


def test_source_no_longer_raises_on_rl2_disagreement():
    src = (REPO / "scripts" / "ga_1_tier3_eval.py").read_text()
    assert 'raise RuntimeError(f"blind comparators drifted' not in src, (
        "the agreement check against the actor run must be disclosed, not fatal"
    )


def test_source_saves_cells_before_the_rl2_check():
    """Retention must not depend on the check that follows it passing."""
    src = (REPO / "scripts" / "ga_1_tier3_eval.py").read_text()
    save_pos = src.index("cells_and_comparisons_partial.json")
    check_pos = src.index("# ---- instrument check")
    assert save_pos < check_pos, "the partial save must happen BEFORE the agreement check"


def test_source_records_pass_and_worst_cells():
    src = (REPO / "scripts" / "ga_1_tier3_eval.py").read_text()
    assert '"pass": bool(checked and max_ref_err <= RL2_TOLERANCE)' in src
    assert '"worst_cells": worst_cells[:10]' in src
