"""Lane T0 -- GA-1 tier 3 rung A: EAFT loss and guard canaries."""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import torch

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))

import ga_1_tier3_train as T  # noqa: E402


def test_registered_hyperparameters_are_the_spec_values():
    assert (T.TAU, T.BETA, T.LR, T.GRAD_CLIP) == (1.0, 0.1, 1e-4, 1.0)
    assert (T.EPOCH_CAP, T.PATIENCE) == (4, 2), "§9c: cap 4, patience 2"
    assert T.SUBSTRATE == "ratchet-cwsim/refreshed-head"


def test_loss_is_zero_when_policy_equals_target_and_base():
    """The minimum must sit where the policy already matches both terms."""
    n = 6
    p = np.full(n, 1.0 / n)
    logits = torch.zeros(n)                      # uniform policy
    loss, kt, ka = T.eaft_loss(logits, torch.tensor(p, dtype=torch.float32),
                               torch.tensor(p, dtype=torch.float32))
    assert abs(kt) < 1e-6 and abs(ka) < 1e-6
    assert abs(float(loss)) < 1e-6


def test_loss_falls_as_the_policy_moves_toward_the_target():
    n = 4
    p_star = torch.tensor([0.7, 0.1, 0.1, 0.1])
    p_base = torch.full((n,), 0.25)
    far = torch.tensor([0.0, 0.0, 0.0, 3.0])     # mass on the wrong item
    near = torch.tensor([3.0, 0.0, 0.0, 0.0])    # mass on the target's item
    l_far, _, _ = T.eaft_loss(far, p_star, p_base)
    l_near, _, _ = T.eaft_loss(near, p_star, p_base)
    assert float(l_near) < float(l_far)


def test_anchor_penalises_drift_from_the_base_policy():
    """beta*KL(pi||pi_base) must grow as the policy departs from the base."""
    n = 4
    p_star = torch.tensor([0.25, 0.25, 0.25, 0.25])
    p_base = torch.full((n,), 0.25)
    _, _, a_flat = T.eaft_loss(torch.zeros(n), p_star, p_base)
    _, _, a_sharp = T.eaft_loss(torch.tensor([5.0, 0.0, 0.0, 0.0]), p_star, p_base)
    assert a_sharp > a_flat >= -1e-6


def test_loss_is_finite_on_a_degenerate_target():
    n = 5
    p_star = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0])
    p_base = torch.full((n,), 1.0 / n)
    loss, kt, ka = T.eaft_loss(torch.zeros(n), p_star, p_base)
    assert np.isfinite(float(loss)) and np.isfinite(kt) and np.isfinite(ka)


def test_leakage_canary_catches_a_reference_bearing_prompt():
    clean = [{"prompt": "CURRENT MENU\nA. foo\nCHOICE="} for _ in range(20)]
    assert T.leakage_canary(clean, n_probe=20)["pass"]
    dirty = list(clean)
    dirty[3] = {"prompt": "the b_full reference distance is 0.2\nCHOICE="}
    res = T.leakage_canary(dirty, n_probe=20)
    assert not res["pass"] and res["violations"]


def test_reward_polarity_canary_fails_on_an_inverted_reward(monkeypatch):
    """If the sign were flipped, the best-advantage action would land FURTHER away."""
    def fake_score(model, ep, picks, device):
        # distance decreases with the asked problem_id -> higher pid is genuinely better
        pid = int(picks[-1].split("-p")[-1])
        return {"dlogit": [1.0, 1.0 - pid / 100.0]}

    monkeypatch.setattr(T, "_score_fixed_picks", fake_score)
    cands = [{"episode_id": f"L1-p{p}", "problem_id": p} for p in (10, 20, 30, 40)]
    ep = {"prefix_events": [], "candidates": cands, "targets": [], "full_events": []}
    states = [{"episode": ep, "known_candidate_idx": [], "remaining_idx": [0, 1, 2, 3]}] * 20
    good = [np.array([0.1, 0.2, 0.3, 0.4])] * 20      # ranks agree with the truth
    bad = [np.array([0.4, 0.3, 0.2, 0.1])] * 20       # inverted
    assert T.reward_polarity_canary(None, states, good, "cpu", n_probe=20)["pass"]
    assert not T.reward_polarity_canary(None, states, bad, "cpu", n_probe=20)["pass"]


def test_polarity_canary_passes_only_candidate_picks_not_prefix_ids():
    """Regression: `_score_fixed_picks` resolves ids against the MENU, never the prefix.

    The first pilot died here with KeyError on a prefix episode id. Assert the contract
    by capturing what the canary actually hands downstream.
    """
    seen = []

    def spy(model, ep, picks, device):
        seen.append(list(picks))
        return {"dlogit": [1.0, 0.5 - 0.01 * len(picks)]}

    import ga_1_tier3_train as TT
    old = TT._score_fixed_picks
    TT._score_fixed_picks = spy
    try:
        prefix = [{"episode_id": f"pre{i}", "problem_id": i} for i in range(3)]
        cands = [{"episode_id": f"c{p}", "problem_id": p} for p in (10, 20, 30)]
        ep = {"prefix_events": prefix, "candidates": cands, "targets": [], "full_events": []}
        states = [{"episode": ep, "known_candidate_idx": [0], "remaining_idx": [1, 2]}] * 4
        TT.reward_polarity_canary(None, states, [np.array([0.2, 0.5])] * 4, "cpu", n_probe=4)
    finally:
        TT._score_fixed_picks = old
    assert seen, "canary never scored anything"
    ids = {i for picks in seen for i in picks}
    assert not any(i.startswith("pre") for i in ids), f"prefix ids leaked into picks: {ids}"
    assert ids <= {"c10", "c20", "c30"}


# ------------------------------------------------------------------ artifact retention
#
# Two defects found on 4 Sep, both of which destroy a completed epoch's work rather
# than corrupt it, and neither of which any green test could see.


def test_training_heartbeat_stage_name_is_per_epoch():
    """`stage_started` is set once per stage NAME and never reset.

    `NumericProgress.heartbeat` does `stage_started.setdefault(stage, now)`, so a stage
    name reused across epochs keeps epoch 1's start time while `done` resets to zero.
    At the top of epoch 2 the guard then reads "a handful of rows in ten hours",
    projects a runtime near infinity, and raises -- killing the run precisely when the
    first trained adapter exists.  The stage name must therefore carry the epoch.
    """
    src = (REPO / "scripts" / "ga_1_tier3_train.py").read_text()
    assert 'progress.maybe_heartbeat(f"tier3_train_e{epoch}"' in src, (
        "the per-epoch training heartbeat must use a per-epoch stage name"
    )
    assert 'progress.maybe_heartbeat("tier3_train"' not in src


def test_best_adapter_is_written_inside_the_epoch_loop():
    """The checkpoint must reach disk as each epoch improves, not after the loop.

    Holding the only copy in memory until all four epochs finish means any mid-loop
    failure -- overrun guard, OOM, spot reclaim, fuse -- discards every epoch already
    paid for.  Retention is a separate obligation from resumability.
    """
    src = (REPO / "scripts" / "ga_1_tier3_train.py").read_text()
    loop = src.split("for epoch in range(1, args.epoch_cap + 1):", 1)[1]
    body = loop.split("\n    if best[", 1)[0]
    assert "_write_ckpt(best)" in body, "no checkpoint write inside the epoch loop"


def test_numeric_upload_helper_rejects_a_torch_checkpoint():
    """Why `_upload_adapter` exists rather than `progress.upload`.

    The shared helper's allowlist is numeric-suffix only; handing it the `.pt` adapter
    raises, and inside the epoch loop that raise would kill the run at the moment of
    its first success.
    """
    import ga_1_common

    with np.testing.assert_raises(ValueError):
        ga_1_common.assert_numeric_upload_path(pathlib.Path("adapter_best.pt"))


def test_adapter_upload_never_raises(tmp_path, monkeypatch):
    """Telemetry must not be able to kill training that already produced the artifact."""
    import argparse

    ckpt = tmp_path / "adapter_best.pt"
    ckpt.write_bytes(b"weights")

    T._upload_adapter(ckpt, argparse.Namespace(bucket="", out_prefix=""))  # no-op

    class _Boom:
        def client(self, *a, **k):
            raise RuntimeError("no credentials")

    monkeypatch.setitem(sys.modules, "boto3", _Boom())
    T._upload_adapter(ckpt, argparse.Namespace(bucket="b", out_prefix="p"))


def test_epochs_partial_reports_best_after_this_epochs_comparison():
    """`epochs_partial.json`'s best fields must reflect the epoch just scored.

    An earlier version wrote the file BEFORE comparing the new score against `best`,
    so a reader mid-run (or after a crash) saw the PREVIOUS epoch's verdict while
    `adapter_best.pt` itself already carried the true one -- correct code, wrong
    moment, and exactly the kind of thing this file's other retention tests exist to
    catch. The write must come after the best-vs-current comparison, not before it.
    """
    src = (REPO / "scripts" / "ga_1_tier3_train.py").read_text()
    loop = src.split("for epoch in range(1, args.epoch_cap + 1):", 1)[1]
    body = loop.split("\n    if best[", 1)[0]
    write_pos = body.index('epochs_partial.json").write_text(')
    compare_pos = body.index('best["score"] - score > MIN_IMPROVEMENT')
    assert compare_pos < write_pos, (
        "epochs_partial.json is written before the best-score comparison"
    )
