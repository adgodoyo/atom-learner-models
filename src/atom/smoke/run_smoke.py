"""Fake-data end-to-end smoke test: data -> filter -> head -> train -> eval -> report.

The whole ATOM loop runs here in miniature, on synthetic learners, exercising the same
logging, metrics and report machinery the real pipeline uses. Gate: finishes in under
5 minutes on a laptop CPU/GPU, reaches validation AUC >= 0.55, and writes
reports/p0_03_smoke.md. Used by `make smoke`.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch
from torch import nn

from atom.eval.metrics import accuracy, auc, brier, ece
from atom.eval.report import gate_report
from atom.smoke.fake_data import make_fake_data, temporal_split
from atom.smoke.model import TinyFilter
from atom.utils.logging import RunLogger, default_device
from atom.utils.seeds import seed_everything

DEFAULTS: dict[str, Any] = {
    "n_learners": 20,
    "n_items": 50,
    "seq_len": 60,
    "codec_dim": 16,
    "hidden": 32,
    "epochs": 40,
    "lr": 1e-2,
    "train_frac": 0.7,
    "ability_sd": 1.0,
    "difficulty_sd": 1.0,
    "seed": 0,
    # Gate thresholds:
    "max_wall_s": 300.0,   # < 5 min (the smoke gate)
    "min_val_auc": 0.55,   # sanity: the miniature must learn *something* from history
}


def run_smoke(
    config: dict[str, Any] | None = None,
    out_root: str | Path = "runs",
    report_path: str | Path = "reports/p0_03_smoke.md",
    device: str | None = None,
    echo: bool = True,
) -> dict[str, Any]:
    cfg = {**DEFAULTS, **(config or {})}
    seed_everything(cfg["seed"])
    device = device or default_device()

    logger = RunLogger(
        config=cfg, out_root=out_root, data_version="fake-v1",
        seed=cfg["seed"], heartbeat_every_s=2.0, echo=echo,
    )
    t0 = time.time()

    data = make_fake_data(
        n_learners=cfg["n_learners"], n_items=cfg["n_items"], seq_len=cfg["seq_len"],
        codec_dim=cfg["codec_dim"], ability_sd=cfg["ability_sd"],
        difficulty_sd=cfg["difficulty_sd"], seed=cfg["seed"],
    )
    codec = data.codec.to(device)
    y = data.y.to(device)
    train_mask, val_mask = temporal_split(cfg["seq_len"], cfg["train_frac"])
    train_mask = train_mask.to(device)
    val_mask = val_mask.to(device)

    model = TinyFilter(cfg["codec_dim"], cfg["hidden"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    loss_fn = nn.BCEWithLogitsLoss(reduction="none")

    def masked_bce(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        per = loss_fn(logits, y)                       # [N, T]
        m = mask[None, :].expand_as(per)
        return (per * m).sum() / m.sum()

    for epoch in range(cfg["epochs"]):
        model.train()
        opt.zero_grad()
        logits = model(codec, y)
        loss = masked_bce(logits, train_mask)
        logger.tripwire_finite("train_loss", loss)
        loss.backward()
        opt.step()
        logger.heartbeat(step=epoch + 1, total=cfg["epochs"], loss=loss.item())

    # Final evaluation on the held-out (later) steps.
    model.eval()
    with torch.no_grad():
        logits = model(codec, y)
        probs = torch.sigmoid(logits)
    vm = val_mask[None, :].expand_as(probs)
    yv = y[vm].detach().cpu()
    pv = probs[vm].detach().cpu()

    metrics = {
        "val_auc": auc(yv, pv),
        "val_acc": accuracy(yv, pv),
        "val_ece": ece(yv, pv),
        "val_brier": brier(yv, pv),
        "n_val_preds": int(vm.sum().item()),
        "device": device,
    }
    wall_s = time.time() - t0
    metrics["wall_s"] = round(wall_s, 2)
    logger.log_metrics(**metrics)

    passed = (wall_s < cfg["max_wall_s"]) and (metrics["val_auc"] >= cfg["min_val_auc"])
    logger.summary(status="ok" if passed else "gate-fail", **metrics)

    gate_report(
        path=report_path,
        title="P0-3 - Fake-data end-to-end smoke",
        meta={
            "run_id": logger.run_id,
            "git_sha": logger.git_sha,
            "config_hash": logger.config_hash,
            "seed": cfg["seed"],
            "device": device,
        },
        metrics=metrics,
        passed=passed,
        gate_desc=(
            f"wall-clock < {cfg['max_wall_s']:.0f}s AND val AUC >= {cfg['min_val_auc']} "
            "(pipeline learns from history); report produced."
        ),
        notes=[
            "Synthetic knowledge-tracing world; 'codec' vectors are random stand-ins.",
            "No real data or real model is touched: this is the miniature of the full loop.",
            f"metrics.jsonl and manifest under {logger.run_dir}",
        ],
    )

    result = {"passed": passed, "run_dir": str(logger.run_dir), **metrics}
    return result
