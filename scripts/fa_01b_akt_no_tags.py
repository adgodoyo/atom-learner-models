#!/usr/bin/env python
"""AKT without curriculum tags: the same pyKT AKT and config, with concepts = problem IDs.

Every problem becomes its own concept, so curriculum (skill-tag) information is removed;
the Rasch `pid` term is unchanged. The run measures how much of AKT's 0.016 AUC edge over
the GRU belief filter of `fa_01_s1v2.py` (0.8016 vs 0.786, query-independent arm) comes
from the curriculum tags rather than from attention. Decision rule, fixed before the run:
AKT-no-tags within +/-0.005 AUC of the filter => the edge was curriculum and the belief
bottleneck is not binding (no query-conditioned readout is needed); clearly above =>
query-conditioned readout is a live mechanism.

The change: `atom.baselines.akt_data.build_concept_map` derives a problem's concept from
its skill tag via `Skills.csv`; this script never calls it. `concept_map` is instead
`build_pid_map(problems["problem_id"])`, the function `fa_01_akt_v2.py` uses for
`pid_map`, so concept and pid become the same dense per-problem mapping
(`n_question == n_pid`). In `pykt.models.akt.AKT.forward`, concept feeds the
knowledge-component embeddings (`q_embed`/`q_embed_diff`/`qa_embed_diff`) and pid feeds
only the scalar Rasch item-difficulty term (`difficult_param`), so the swap removes
curriculum-derived cross-item transfer while keeping per-item difficulty. Everything else
(config, seed, data, splits, architecture, batch size, epochs, checks) is copied
unchanged from `fa_01_akt_v2.py`.

Inputs: the v2 manifest and splits, FA problems. Outputs: a checkpoint in the run
directory and `reports/fa_01b_akt_no_tags.md`. Run: `python scripts/fa_01b_akt_no_tags.py`.
Serves §6.2 (worth of the curriculum tags).
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
    build_pid_map,
    build_support_mask,
    check_coverage,
    chunk_learner_history,
    chunking_stats,
)
from atom.data.fa import load_problems  # noqa: E402
from atom.data.fa_manifest import load_v2_events  # noqa: E402
from atom.eval.metrics import accuracy, auc, brier, ece  # noqa: E402
from atom.eval.report import gate_report  # noqa: E402
from atom.utils.checkpoints import save_checkpoint  # noqa: E402
from atom.utils.logging import RunLogger, default_device  # noqa: E402
from atom.utils.seeds import seed_everything  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[1]
# Comparators measured by fa_01_akt_v2.py and fa_01_s1v2.py; not re-run here.
AKT_V2_MARGINAL_AUC = 0.8016   # reports/fa_01_akt_v2.md
AKT_V2_NONE_AUC = 0.6982       # reports/fa_01_akt_v2.md
S1V2_MARGINAL_AUC = 0.786      # reports/fa_01_s1v2.md, FIXED (query-independent) arm
S1V2_NONE_AUC = 0.652          # reports/fa_01_s1v2.md, FIXED (query-independent) arm


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
    logger = RunLogger(config={"task": "fa_01b_akt_no_tags", **akt_cfg}, data_version="FA-v2-2026-08-09",
                       seed=akt_cfg["seed"], heartbeat_every_s=15.0)

    import pykt.models.akt as _akt_module
    from pykt.models.akt import AKT

    logger.heartbeat(step=0, force=True, stage="loading the v2 manifest, splits and problems",
                     device=device)
    ev = load_v2_events(REPO / v2cfg["manifest_out"], REPO / v2cfg["learner_split_out"],
                        REPO / v2cfg["item_split_out"])
    ev = ev[ev["split"].isin(["train", "val"])].copy()
    n_before = len(ev)
    ev = ev[~ev["score_missing"]].copy()
    n_excluded = n_before - len(ev)
    ev["correct"] = ev["score"].astype("int64")
    train_ev, val_ev = ev[ev["split"] == "train"], ev[ev["split"] == "val"]

    problems = load_problems()
    pid_map = build_pid_map(problems["problem_id"])
    # THE ONE CHANGE: concept = problem id, not the curriculum skill tag. No load_skills(),
    # no build_concept_map() call anywhere in this script.
    concept_map = pid_map
    check_coverage(ev["problem_id"].unique(), concept_map, pid_map)
    n_question, n_pid = len(set(concept_map.values())), len(set(pid_map.values()))
    assert n_question == n_pid, "concept=pid aliasing must yield identical cardinality"

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
    none_stratum_np = (val_support == 0)

    if str(_akt_module.device) != str(torch.device(device)):
        _akt_module.device = torch.device(device)

    torch.manual_seed(akt_cfg["seed"])
    model = AKT(n_question=n_question, n_pid=n_pid, **akt_cfg["model"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=akt_cfg["lr"], weight_decay=akt_cfg["weight_decay"])

    val_concepts = val_batch["concepts"].to(device)
    val_pids = val_batch["pids"].to(device)
    val_responses = val_batch["responses"].to(device)
    val_mask = val_batch["valid_mask"].clone()
    val_mask[:, 0] = False
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

    loss_decreased = bool(loss_history[-1] < loss_history[0])
    lo, hi = akt_cfg["full_run_gate_auc_lo"], akt_cfg["full_run_gate_auc_hi"]
    band_status = ("below_band" if marginal["auc"] < lo
                   else "above_band" if marginal["auc"] > hi else "within_band")

    delta_vs_akt_v2 = marginal["auc"] - AKT_V2_MARGINAL_AUC
    delta_vs_s1v2 = marginal["auc"] - S1V2_MARGINAL_AUC
    # Decision rule: AKT-no-tags ~= filter (within a +/-0.005 AUC band, the size of the
    # filter's own v1->v2 delta, used as the "no material difference" band) => no
    # query-conditioned readout needed ("cancel"); clearly above => it is ("license").
    NEAR_BAND = 0.005
    fa6d_decision = ("cancel" if abs(delta_vs_s1v2) <= NEAR_BAND
                     else "license" if delta_vs_s1v2 > NEAR_BAND else "inconclusive_below_s1v2")
    passed = bool(loss_decreased and band_status != "below_band")

    logger.log_metrics(
        passed=passed, val_auc=marginal["auc"], none_auc=none_stratum["auc"],
        delta_vs_akt_v2=delta_vs_akt_v2, delta_vs_s1v2=delta_vs_s1v2, fa6d_decision=fa6d_decision,
        loss_first=loss_history[0], loss_last=loss_history[-1],
        n_excluded_score_missing=n_excluded, n_train_chunks=n_train_chunks,
        n_val_chunks=n_val_chunks, train_wall_s=train_wall_s, band_status=band_status,
    )
    logger.summary(status="ok" if passed else "gate-fail", val_auc=marginal["auc"], band_status=band_status)

    def row(d):
        return f"AUC {d['auc']:.4f} · acc {d['accuracy']:.3f} · ECE {d['ece']:.3f} · Brier {d['brier']:.3f}"

    gate_report(
        path="reports/fa_01b_akt_no_tags.md",
        title="AKT without curriculum tags (concepts = problem IDs, curriculum removed)",
        meta={"run_id": logger.run_id, "git_sha": logger.git_sha, "config_hash": logger.config_hash,
              "device": device, "epochs": epochs, "batch_size": batch_size,
              "n_question": n_question, "n_pid": n_pid,
              "n_train_chunks": n_train_chunks, "n_val_chunks": n_val_chunks},
        metrics={
            "AKT-no-tags marginal VAL": row(marginal),
            "AKT-no-tags support=NONE stratum": row(none_stratum),
            "AKT v2 (WITH curriculum tags) marginal VAL": f"AUC {AKT_V2_MARGINAL_AUC:.4f}",
            "AKT v2 (WITH curriculum tags) support=NONE": f"AUC {AKT_V2_NONE_AUC:.4f}",
            "S1v2 (FIXED, no curriculum ever) marginal VAL": f"AUC {S1V2_MARGINAL_AUC:.4f}",
            "S1v2 (FIXED, no curriculum ever) support=NONE": f"AUC {S1V2_NONE_AUC:.4f}",
            "delta: AKT-no-tags - AKT v2 (cost of removing curriculum)": f"{delta_vs_akt_v2:+.4f}",
            "delta: AKT-no-tags - S1v2 (curriculum-free apples-to-apples)": f"{delta_vs_s1v2:+.4f}",
            "FA-6d decision (near-band +/-0.005, S1v2's own v1->v2 delta magnitude)": fa6d_decision,
            "loss first -> last": f"{loss_history[0]:.4f} -> {loss_history[-1]:.4f}",
            "n excluded (score_missing, not maskable for AKT's coupled input)": f"{n_excluded:,}",
            "train wall (s)": round(train_wall_s, 1),
            "gate band (mechanism check, not the FA-6d decision)": [lo, hi],
        },
        passed=passed,
        gate_desc="MECHANISM CHECK only (loss decreases, VAL AUC not below the full-run "
                  f"band [{lo}, {hi}], the same band `fa_01_akt_v2.py` uses; it shows the model "
                  "trained sanely with the swapped concept map and says nothing about the "
                  "readout question). The substantive result is the delta-vs-filter near-band "
                  "comparison reported above, which is read, not auto-decided by this script.",
        notes=[
            "THE ONE CHANGE from fa_01_akt_v2.py: concept_map = build_pid_map(...) instead "
            "of build_concept_map(skills) -- every problem is its own concept, curriculum "
            "(skill-tag) information is fully removed. n_question == n_pid by construction "
            "(asserted). Every other line -- config, seed, data, splits, architecture, "
            "batch size, epochs -- is unchanged from fa_01_akt_v2.py, so the comparison "
            "isolates the concept map.",
            "Rasch difficulty term (`difficult_param`, driven by `pid`) is COMPLETELY "
            "unaffected by this change -- pid and concept are architecturally independent "
            "inputs to pyKT's AKT (concept feeds q_embed/q_embed_diff/qa_embed_diff; pid "
            "feeds only the scalar difficulty term), as in pykt.models.akt.",
            f"{n_excluded:,} score-missing events excluded from the AKT stream entirely, "
            "same reason and same count as fa_01_akt_v2.py (pyKT couples response into the "
            "joint input embedding, not maskable).",
            "Decision rule (fa6d_decision): near-band +/-0.005 AUC (the size of the GRU "
            "filter's own v1->v2 delta, used as the 'no material difference' band) around "
            "the AKT-no-tags-minus-filter delta. Within band => the belief bottleneck is "
            "not binding and no query-conditioned readout is needed ('cancel'): AKT's edge "
            "over the filter was curriculum, not attention/compression. Clearly above => "
            "query-conditioned readout is a live mechanism ('license').",
            "Read this report's fa6d_decision field, not just the pass/fail check above -- "
            "the check is a mechanism sanity check (did training work at all with the "
            "swapped map), not the substantive finding of this run.",
        ],
    )
    print(f"\nFA-1b (AKT-no-tags): {'PASS' if passed else 'FAIL'} (mechanism check) · "
         f"report: reports/fa_01b_akt_no_tags.md")
    print(f"  marginal AUC {marginal['auc']:.4f} (AKT v2 {AKT_V2_MARGINAL_AUC:.4f}, "
         f"S1v2 {S1V2_MARGINAL_AUC:.4f}) · delta-vs-S1v2 {delta_vs_s1v2:+.4f}")
    print(f"  FA-6d decision: {fa6d_decision}")
    print(f"  train wall {train_wall_s:.0f}s, {n_train_chunks} train chunks, {n_val_chunks} val chunks")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
