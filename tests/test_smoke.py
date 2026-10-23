"""The whole miniature pipeline must run and pass its gate (fast, CPU, tiny config)."""

import json

from atom.smoke.run_smoke import run_smoke


def test_smoke_runs_and_passes(tmp_path):
    cfg = {"n_learners": 16, "seq_len": 40, "epochs": 20, "seed": 0}
    result = run_smoke(
        config=cfg,
        out_root=tmp_path / "runs",
        report_path=tmp_path / "reports" / "p0_03_smoke.md",
        device="cpu",
        echo=False,
    )
    assert result["passed"]
    assert result["val_auc"] >= 0.55
    assert result["wall_s"] < 300.0

    # Report and run artifacts exist.
    assert (tmp_path / "reports" / "p0_03_smoke.md").exists()
    run_dir = tmp_path / "runs"
    manifests = list(run_dir.glob("*/manifest.yaml"))
    metrics = list(run_dir.glob("*/metrics.jsonl"))
    assert manifests and metrics
    rec = json.loads(metrics[0].read_text().strip().splitlines()[-1])
    assert "val_auc" in rec and "wall_s" in rec


def test_smoke_is_reproducible(tmp_path):
    cfg = {"n_learners": 16, "seq_len": 40, "epochs": 15, "seed": 1}
    a = run_smoke(config=cfg, out_root=tmp_path / "a",
                  report_path=tmp_path / "a.md", device="cpu", echo=False)
    b = run_smoke(config=cfg, out_root=tmp_path / "b",
                  report_path=tmp_path / "b.md", device="cpu", echo=False)
    assert abs(a["val_auc"] - b["val_auc"]) < 1e-6
