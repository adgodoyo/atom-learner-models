import json

import pytest

from atom.utils.logging import (
    LossSpikeTripwire,
    RunLogger,
    TripwireError,
    config_hash,
    git_sha,
    rearm_shutdown_fuse,
)


def test_manifest_metrics_heartbeat_written(tmp_path):
    logger = RunLogger(config={"a": 1}, out_root=tmp_path, seed=42, echo=False)
    logger.heartbeat(step=1, force=True, loss=0.5)
    logger.log_metrics(auc=0.71, step=1)

    assert (logger.run_dir / "manifest.yaml").exists()
    hb = (logger.run_dir / "heartbeat.log").read_text()
    assert "step=1" in hb and "loss=0.5" in hb

    lines = (logger.run_dir / "metrics.jsonl").read_text().strip().splitlines()
    rec = json.loads(lines[0])
    assert rec["auc"] == 0.71
    assert rec["seed"] == 42
    assert rec["config_hash"] == logger.config_hash
    assert "git_sha" in rec and "wall_s" in rec


def test_tripwire_finite_raises_on_nan_and_inf(tmp_path):
    logger = RunLogger(config={}, out_root=tmp_path, echo=False)
    assert logger.tripwire_finite("loss", 0.3) == 0.3
    with pytest.raises(TripwireError):
        logger.tripwire_finite("loss", float("nan"))
    with pytest.raises(TripwireError):
        logger.tripwire_finite("loss", float("inf"))


def test_tripwire_finite_accepts_torch_scalar(tmp_path):
    import torch

    logger = RunLogger(config={}, out_root=tmp_path, echo=False)
    with pytest.raises(TripwireError):
        logger.tripwire_finite("loss", torch.tensor(float("nan")))
    assert logger.tripwire_finite("loss", torch.tensor(1.5)) == 1.5


def test_loss_spike_tripwire_fires():
    tw = LossSpikeTripwire(factor=3.0, window=50, patience=5, min_history=10)
    for _ in range(20):
        tw.update(1.0)
    with pytest.raises(TripwireError):
        for _ in range(10):
            tw.update(100.0)


def test_loss_spike_tripwire_survives_single_blip():
    tw = LossSpikeTripwire(factor=3.0, window=50, patience=5, min_history=10)
    for _ in range(20):
        tw.update(1.0)
    tw.update(100.0)  # one blip, below patience
    for _ in range(5):
        tw.update(1.0)  # recovers, streak resets -> no raise


def test_config_hash_is_stable_and_sensitive():
    assert config_hash({"a": 1, "b": 2}) == config_hash({"b": 2, "a": 1})
    assert config_hash({"a": 1}) != config_hash({"a": 2})


def test_git_sha_returns_string():
    assert isinstance(git_sha(), str)


def test_rearm_shutdown_fuse_no_op_before_any_rate_is_known(monkeypatch):
    calls = []
    monkeypatch.setattr("subprocess.run", lambda *a, **k: calls.append(a))
    minutes = rearm_shutdown_fuse(elapsed_s=5.0, current_step=0, total_steps=1000)
    assert minutes == 0
    assert calls == []


def test_rearm_shutdown_fuse_projects_from_measured_rate(monkeypatch):
    calls = []
    monkeypatch.setattr("subprocess.run", lambda *a, **k: calls.append(a))
    # 10s for 100 steps -> 0.1 s/step; 900 steps remain -> 90s = 1.5min projected;
    # 2x margin = 3min, below the 60min floor -> floor wins.
    minutes = rearm_shutdown_fuse(elapsed_s=10.0, current_step=100, total_steps=1000,
                                  min_fuse_minutes=60)
    assert minutes == 60
    assert len(calls) == 2  # shutdown -c, then shutdown -h +N
    assert calls[0][0] == ["sudo", "shutdown", "-c"]
    assert calls[1][0] == ["sudo", "shutdown", "-h", "+60"]


def test_rearm_shutdown_fuse_exceeds_floor_when_projected_remaining_is_large(monkeypatch):
    calls = []
    monkeypatch.setattr("subprocess.run", lambda *a, **k: calls.append(a))
    # 3600s (1h) for 1000 steps -> 3.6 s/step; 9000 steps remain -> 32400s = 540min
    # projected; 2x margin = 1080min, above the 60min floor -> that value wins.
    minutes = rearm_shutdown_fuse(elapsed_s=3600.0, current_step=1000, total_steps=10000,
                                  min_fuse_minutes=60)
    assert minutes == 1080
    assert calls[1][0] == ["sudo", "shutdown", "-h", "+1080"]
