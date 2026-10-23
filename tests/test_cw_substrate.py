"""`atom.simulator.cw_substrate.load_substrate`, the planner-facing adapter over the
CodeWorkout belief substrates (the `"ratchet-cwsim/refreshed-head"` entry points at the
checkpoint written by `scripts/cw_sim_head_refresh.py`).

Dispatch/error-path tests run everywhere. The end-to-end tests are skipped where the
real local checkpoints/beliefs are absent (a fresh clone) -- these are real ~400MB-scale
gitignored artifacts, not fixtures -- mirroring `tests/test_cw_fc_grid_eval.py`'s own
`RATCHET_CKPT.exists()` skip pattern."""
from __future__ import annotations

import pathlib

import numpy as np
import pytest

from atom.simulator.cw_substrate import _BUNDLE_META, load_substrate

REPO = pathlib.Path(__file__).resolve().parents[1]


def test_bundle_registry_has_the_three_named_substrates():
    assert set(_BUNDLE_META) == {
        "ratchet-original", "ratchet-cwsim/cold-head", "ratchet-cwsim/refreshed-head",
    }
    for spec in _BUNDLE_META.values():
        assert spec["checkpoint_sha256"], "every substrate entry must carry a real sha256"
        assert spec["beliefs_sha256"]


def test_unknown_substrate_name_raises_key_error():
    with pytest.raises(KeyError, match="ratchet-original"):
        load_substrate("not-a-real-substrate")


def test_missing_checkpoint_raises_file_not_found(monkeypatch):
    import atom.simulator.cw_substrate as mod

    fake = {"fake": {"checkpoint": REPO / "results/does_not_exist/final.pt",
                     "beliefs": REPO / "artifacts/does_not_exist.npz",
                     "source": "test", "checkpoint_sha256": "x", "beliefs_sha256": "y"}}
    monkeypatch.setattr(mod, "_BUNDLE_META", fake)
    with pytest.raises(FileNotFoundError, match="fake"):
        load_substrate("fake")


def _artifacts_present(name: str) -> bool:
    spec = _BUNDLE_META[name]
    return pathlib.Path(spec["checkpoint"]).exists() and pathlib.Path(spec["beliefs"]).exists()


@pytest.mark.parametrize("name", sorted(_BUNDLE_META))
def test_load_substrate_end_to_end(name):
    if not _artifacts_present(name):
        pytest.skip(f"{name}: real checkpoint/beliefs not present locally")
    bundle = load_substrate(name)
    model = bundle["model"]
    assert {"attempt_encoder", "filt", "outcome_head"}.issubset(model)
    beliefs_path = pathlib.Path(bundle["beliefs_path"])
    assert beliefs_path.exists()
    b = np.load(beliefs_path)
    assert b["mu"].shape[1] == 32
    assert bundle["checkpoint_sha256"] == _BUNDLE_META[name]["checkpoint_sha256"]


def test_cold_and_refreshed_head_share_the_same_frozen_filter():
    """The frozen-filter claim of the head refresh, re-checked from OUTSIDE the training script:
    cold-head and refreshed-head differ only in outcome_head, so their published belief
    handoffs (independently exported by two separate script runs) must be identical --
    the filter that produced them was never retrained."""
    names = ("ratchet-cwsim/cold-head", "ratchet-cwsim/refreshed-head")
    if not all(_artifacts_present(n) for n in names):
        pytest.skip("cold-head / refreshed-head artifacts not both present locally")
    cold = np.load(pathlib.Path(_BUNDLE_META[names[0]]["beliefs"]))
    refreshed = np.load(pathlib.Path(_BUNDLE_META[names[1]]["beliefs"]))
    np.testing.assert_array_equal(cold["learner"], refreshed["learner"])
    np.testing.assert_array_equal(cold["episode_id"], refreshed["episode_id"])
    np.testing.assert_allclose(cold["mu"], refreshed["mu"], atol=1e-6)
    np.testing.assert_allclose(cold["log_sigma"], refreshed["log_sigma"], atol=1e-6)
