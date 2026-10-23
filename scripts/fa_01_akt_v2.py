#!/usr/bin/env python
"""Train and evaluate the AKT baseline on the rebuilt FA manifest (v2), full corpus.

Uses pyKT's AKT with the skill tags as concepts, trained with minibatches
(`full_run_batch_size` in `configs/akt_baseline.yaml`), which keeps memory low enough to
run on a laptop GPU (MPS) as well as CUDA. `atom.baselines.akt_data` turns event rows into
pyKT's tensor format. pyKT's AKT feeds each response into the joint (concept, pid,
response) input embedding, not only into the loss target, so a maskable placeholder for a
missing response is not well defined: score-missing events are excluded from the AKT
stream before chunking (the excluded count is reported), whereas the GRU filter masks them.

Two scores are reported: the marginal validation AUC and the AUC on events whose support
class is NONE. `build_support_mask` repeats the same `chunk_learner_history` call
the batch builder uses, so each event's `support_code` lands on the same [N, T] grid as
the predictions.

Inputs: the v2 manifest and splits (`fa_00_manifest_rebuild.py`), FA skills and problems.
Outputs: a checkpoint in the run directory and `reports/fa_01_akt_v2.md`.
Run: `python scripts/fa_01_akt_v2.py`. Serves §6.2 (worth of the curriculum tags).
"""

from __future__ import annotations

import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

from atom.baselines.akt_data import (  # noqa: E402
    build_akt_batch_windowed,
    build_concept_map,
    build_pid_map,
    build_support_mask,
    check_coverage,
    chunk_learner_history,
    chunking_stats,
)
from atom.data.fa import load_problems, load_skills  # noqa: E402
from atom.data.fa_manifest import load_v2_events  # noqa: E402
from atom.eval.metrics import accuracy, auc, brier, ece  # noqa: E402
from atom.eval.report import gate_report  # noqa: E402
from atom.utils.checkpoints import save_checkpoint  # noqa: E402
from atom.utils.logging import RunLogger, default_device  # noqa: E402
from atom.utils.seeds import seed_everything  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[1]
V1_VAL_AUC = 0.8116  # AKT validation AUC on the earlier (v1) manifest, from the full-corpus run of
                     # infra/p3_05b_akt_full_train.py; hard-coded for the v1 -> v2 comparison.


def _masked_bce_from_probs(probs, targets, mask):
    probs = probs.clamp(1e-6, 1 - 1e-6)
    bce = -(targets * probs.log() + (1 - targets) * (1 - probs).log())
    return (bce * mask.float()).sum() / mask.float().sum().clamp(min=1)


def _score(probs, correct, mask) -> dict:
    m = mask.cpu()
    y = correct.cpu()[m].numpy()
    p = probs.detach().cpu()[m].numpy()
    return {"auc": auc(y, p), "accuracy": accuracy(y, p), "brier": brier(y, p), "ece": ece(y, p)}


def main() -> int:
    akt_cfg = yaml.safe_load((REPO / "configs" / "akt_baseline.yaml").read_text())
    v2cfg = yaml.safe_load((REPO / "configs" / "data.yaml").read_text())["fa"]["manifest_v2"]
    seed_everything(akt_cfg["seed"])
    device = default_device()
    logger = RunLogger(config={"task": "fa_01_akt_v2", **akt_cfg}, data_version="FA-v2-2026-08-09",
                       seed=akt_cfg["seed"], heartbeat_every_s=15.0)

    import pykt.models.akt as _akt_module
    from pykt.models.akt import AKT

    logger.heartbeat(step=0, force=True, stage="loading v2 manifest + splits + skills/problems",
                     device=device)
    ev = load_v2_events(REPO / v2cfg["manifest_out"], REPO / v2cfg["learner_split_out"],
                        REPO / v2cfg["item_split_out"])
    ev = ev[ev["split"].isin(["train", "val"])].copy()
    n_before = len(ev)
    ev = ev[~ev["score_missing"]].copy()
    n_excluded = n_before - len(ev)
    # build_akt_batch/chunk_learner_history expect a `correct` column (v1's name); v2's
    # `score` is nullable but every row here has already had score_missing excluded, so
    # this is a safe, total rename, not an imputation.
    ev["correct"] = ev["score"].astype("int64")
    train_ev, val_ev = ev[ev["split"] == "train"], ev[ev["split"] == "val"]

    skills = load_skills()
    problems = load_problems()
    concept_map = build_concept_map(skills)
    pid_map = build_pid_map(problems["problem_id"])
    check_coverage(ev["problem_id"].unique(), concept_map, pid_map)
    n_question, n_pid = len(set(concept_map.values())), len(set(pid_map.values()))

    max_seq_len = akt_cfg["max_seq_len"]
    t0 = time.time()
    train_batch = build_akt_batch_windowed(train_ev, concept_map, pid_map, max_seq_len, device="cpu")
    val_batch = build_akt_batch_windowed(val_ev, concept_map, pid_map, max_seq_len, device="cpu")
    chunk_s = time.time() - t0
    train_stats = chunking_stats(train_ev, chunk_learner_history(train_ev, max_seq_len), max_seq_len)
    val_stats = chunking_stats(val_ev, chunk_learner_history(val_ev, max_seq_len), max_seq_len)

    n_train_chunks, t_max = train_batch["concepts"].shape
    n_val_chunks, val_t_max = val_batch["concepts"].shape
    logger.heartbeat(step=0, force=True, stage="chunked", chunk_s=round(chunk_s, 1),
                     n_train_chunks=n_train_chunks, n_val_chunks=n_val_chunks,
                     n_excluded_score_missing=n_excluded)

    val_support = build_support_mask(val_ev, max_seq_len, val_batch["learners"], val_t_max)
    none_stratum_np = (val_support == 0)  # support_code 0 == NONE

    # pyKT 0.0.38's device global -- patch only if it doesn't already match ours (the same
    # check infra/p3_05b_akt_full_train.py uses on CUDA).
    if str(_akt_module.device) != str(torch.device(device)):
        _akt_module.device = torch.device(device)

    torch.manual_seed(akt_cfg["seed"])
    model = AKT(n_question=n_question, n_pid=n_pid, **akt_cfg["model"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=akt_cfg["lr"], weight_decay=akt_cfg["weight_decay"])

    val_concepts = val_batch["concepts"].to(device)
    val_pids = val_batch["pids"].to(device)
    val_responses = val_batch["responses"].to(device)
    val_mask = val_batch["valid_mask"].clone()
    val_mask[:, 0] = False  # pyKT's own convention: never score position 0 (zero in-chunk history)
    val_mask_eval = val_mask.to(device)
    val_mask_none = (val_mask.cpu().numpy() & none_stratum_np)
    val_mask_none_t = torch.tensor(val_mask_none, dtype=torch.bool, device=device)

    n_train = train_batch["concepts"].shape[0]
    batch_size = akt_cfg["full_run_batch_size"]
    epochs = akt_cfg["epochs"]

    rng = torch.Generator().manual_seed(akt_cfg["seed"])
    loss_history: list[float] = []
    t0 = time.time()
    step = 0
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n_train, generator=rng)
        epoch_loss_sum, epoch_loss_n = 0.0, 0
        for bstart in range(0, n_train, batch_size):
            idx = perm[bstart:bstart + batch_size]
            c = train_batch["concepts"][idx].to(device)
            p = train_batch["pids"][idx].to(device)
            r = train_batch["responses"][idx].to(device)
            m = train_batch["valid_mask"][idx].clone()
            m[:, 0] = False
            m = m.to(device)

            opt.zero_grad()
            preds, reg_loss = model(c, r, p)
            loss = _masked_bce_from_probs(preds, r.float(), m) + reg_loss
            logger.tripwire_finite("train_loss", loss)
            loss.backward()
            opt.step()
            step += 1
            epoch_loss_sum += loss.item() * m.sum().item()
            epoch_loss_n += m.sum().item()

        epoch_loss = epoch_loss_sum / max(epoch_loss_n, 1)
        loss_history.append(epoch_loss)
        logger.heartbeat(step=epoch, total=epochs, stage="train", mean_loss=round(epoch_loss, 4))

    train_wall_s = time.time() - t0

    model.eval()
    with torch.no_grad():
        val_preds, _ = model(val_concepts, val_responses, val_pids)
    marginal = _score(val_preds, val_responses, val_mask_eval)
    none_stratum = _score(val_preds, val_responses, val_mask_none_t)

    save_checkpoint({"model": model.state_dict(), "opt": opt.state_dict()},
                    logger.run_dir, step=epochs, metric=marginal["auc"], is_best=True)

    delta_v1_v2 = marginal["auc"] - V1_VAL_AUC
    loss_decreased = bool(loss_history[-1] < loss_history[0])
    lo, hi = akt_cfg["full_run_gate_auc_lo"], akt_cfg["full_run_gate_auc_hi"]
    band_status = ("below_band" if marginal["auc"] < lo
                   else "above_band" if marginal["auc"] > hi else "within_band")
    passed = bool(loss_decreased and band_status != "below_band")

    logger.log_metrics(
        passed=passed, val_auc=marginal["auc"], none_auc=none_stratum["auc"],
        delta_v1_v2=delta_v1_v2, loss_first=loss_history[0], loss_last=loss_history[-1],
        n_excluded_score_missing=n_excluded, n_train_chunks=n_train_chunks,
        n_val_chunks=n_val_chunks, train_wall_s=train_wall_s, band_status=band_status,
    )
    logger.summary(status="ok" if passed else "gate-fail", val_auc=marginal["auc"], band_status=band_status)

    def row(d):
        return f"AUC {d['auc']:.4f} · acc {d['accuracy']:.3f} · ECE {d['ece']:.3f} · Brier {d['brier']:.3f}"

    gate_report(
        path="reports/fa_01_akt_v2.md",
        title="AKT baseline on the rebuilt FA manifest (v2), full corpus",
        meta={"run_id": logger.run_id, "git_sha": logger.git_sha, "config_hash": logger.config_hash,
              "device": device, "epochs": epochs, "batch_size": batch_size,
              "n_train_chunks": n_train_chunks, "n_val_chunks": n_val_chunks},
        metrics={
            "AKT marginal VAL": row(marginal),
            "AKT support=NONE stratum": row(none_stratum),
            "v1 -> v2 delta (VAL AUC)": f"{V1_VAL_AUC:.4f} -> {marginal['auc']:.4f}  (Δ{delta_v1_v2:+.4f})",
            "loss first -> last": f"{loss_history[0]:.4f} -> {loss_history[-1]:.4f}",
            "n excluded (score_missing, not maskable for AKT's coupled input)": f"{n_excluded:,}",
            "train chunking (multi-window learners, dropped-remainder events)":
                f"{train_stats['multi_window_learners']}/{train_stats['total_learners']} learners, "
                f"{train_stats['dropped_events']:,} dropped ({train_stats['dropped_frac']:.2%})",
            "val chunking (multi-window learners, dropped-remainder events)":
                f"{val_stats['multi_window_learners']}/{val_stats['total_learners']} learners, "
                f"{val_stats['dropped_events']:,} dropped ({val_stats['dropped_frac']:.2%})",
            "train wall (s)": round(train_wall_s, 1),
            "gate band": [lo, hi],
        },
        passed=passed,
        gate_desc="loss must decrease over training; VAL AUC not below the full-run band "
                  f"[{lo}, {hi}] (a result above the band is recorded as is, and the band is "
                  "not moved; the v1 result of 0.8116 was also above it).",
        notes=[
            f"v1 reference: {V1_VAL_AUC:.4f} (the full-corpus AKT run on the earlier v1 manifest, "
            "infra/p3_05b_akt_full_train.py; hard-coded, not re-run).",
            "Minibatching (batch_size from akt_baseline.yaml, the same value as the v1 run) "
            "avoids the ~9GB attention-score tensor of a full-batch pass, so the run needs no "
            "CUDA GPU; on MPS the `_akt_module.device` global is patched to match.",
            f"{n_excluded:,} score-missing events excluded from the AKT stream entirely "
            "(disclosed, not masked): pyKT's AKT couples a step's response into the joint "
            "input embedding, not only the loss target, so a maskable missing-response "
            "placeholder is not well-defined for this architecture -- unlike the GRU filter, "
            "which cleanly separates recurrent input from the loss mask (fa_01_s1v2.py).",
            "support=NONE stratum has no v1 analogue (reported on its own, not as a "
            "delta), as in the IRT-v2 and filter scripts. Computed by re-running "
            "chunk_learner_history on the identical (ev, max_seq_len) the batch builder used "
            "internally, so its chunk/position grid is guaranteed aligned, not assumed.",
        ],
    )
    print(f"\nFA-1 (AKT v2): {'PASS' if passed else 'FAIL'} · report: reports/fa_01_akt_v2.md")
    print(f"  marginal AUC {marginal['auc']:.4f} (v1 {V1_VAL_AUC:.4f}, Δ{delta_v1_v2:+.4f}) · "
          f"support=NONE {none_stratum['auc']:.4f}")
    print(f"  train wall {train_wall_s:.0f}s, {n_train_chunks} train chunks, {n_val_chunks} val chunks")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
