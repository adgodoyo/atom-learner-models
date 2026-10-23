#!/usr/bin/env python3
"""Utility model (§8.4): trains a counterfactual utility model Û(s,q) on exact one-step replay
advantages and validates it on held-out questions against greedy-EPIG and random.

The question enters only through its frozen 64-dim codec vector and the outcome head's
predictive features -- never an item identity -- and Û is evaluated on questions it has never
scored (a seed-fixed 20% item-disjoint hold-out). Pre-registered prediction: Û ties greedy-EPIG.
A tie answers "does diagnostic utility generalise across the semantic question space?" with yes
and certifies EPIG as the scoring instrument for the question writer; a win upgrades the
instrument. Either is reported.

``rl_2.py``, ``cw_planner_suite.py``, ``ga_1_common.py`` and ``cw_substrate.py`` are imported
unchanged; ``rl_2.PointerActorV2`` is used as the Û architecture as is, which makes "Û is the
amortised actor" a fact about the code rather than a claim in prose. Checks: item disjointness
on the realised training rows, and reward reconstruction through an independent code path.

Inputs: the CodeWorkout generative simulator with its refreshed outcome head (substrate), the
frozen problem codec. Outputs in the output directory: utility_seed*.pt, manifest.json,
summary.json, test_table.npz.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from ga_1_common import NumericProgress, stable_uint64  # noqa: E402
from ga_1_tier4_common import (  # noqa: E402
    BUDGETS,
    CODEC_DIM,
    FEAT_DIM,
    STATE_DIM,
    SUBSTRATE,
    UTILITY_SEEDS,
    build_table,
    codec_matrix,
    draw_digest,
    draw_states,
    fit_learners,
    item_split,
    load_field,
    state_rows,
    utility_learner_split,
)
from rl_2 import PointerActorV2, _bootstrap_seed_learner  # noqa: E402

MAX_EPOCHS = 60
PATIENCE = 5
LR = 1e-3
LAMBDA_REG = 0.1
GRAD_CLIP = 1.0
BATCH_STATES = 64
MIN_HELDOUT_CANDIDATES = 3
N_BOOT = 2000
PROBE_STATES = 100
RANDOM_BAND_DRAWS = 200


# ------------------------------------------------------------------------------ tensors


def pack(table: dict[str, np.ndarray], codec: np.ndarray, pid_to_idx: dict[int, int],
         keep: np.ndarray | None = None) -> dict[str, torch.Tensor]:
    """Ragged advantage table -> padded [S, M, ...] tensors with a validity mask.

    ``keep`` is a per-candidate boolean selecting which candidates are eligible in this
    view (the item-disjoint restriction).  Padding is masked everywhere it is consumed.
    """
    off, cnt = table["state_offset"], table["state_count"]
    if keep is None:
        keep = np.ones(len(table["cand_r"]), dtype=bool)
    per_state = [np.arange(o, o + c)[keep[o:o + c]] for o, c in zip(off, cnt)]
    sizes = np.asarray([len(x) for x in per_state], dtype=np.int64)
    m = int(sizes.max()) if len(sizes) and sizes.max() > 0 else 1
    s = len(per_state)
    idx_codec = np.asarray([pid_to_idx[int(p)] for p in table["cand_problem_id"]], dtype=np.int64)
    q = np.zeros((s, m, CODEC_DIM + FEAT_DIM), dtype=np.float32)
    r = np.zeros((s, m), dtype=np.float64)
    a = np.zeros((s, m), dtype=np.float32)
    e = np.full((s, m), -np.inf, dtype=np.float32)
    mask = np.zeros((s, m), dtype=bool)
    for i, rows in enumerate(per_state):
        n = len(rows)
        if n == 0:
            continue
        q[i, :n] = np.concatenate((codec[idx_codec[rows]], table["cand_features"][rows]), axis=1)
        r[i, :n] = table["cand_r"][rows]
        a[i, :n] = table["cand_advantage"][rows]
        e[i, :n] = table["cand_epig"][rows]
        mask[i, :n] = True
    return {
        "state": torch.from_numpy(table["state_features"]),
        "q": torch.from_numpy(q),
        "r": torch.from_numpy(r),
        "advantage": torch.from_numpy(a),
        "epig": torch.from_numpy(e),
        "mask": torch.from_numpy(mask),
        "sizes": torch.from_numpy(sizes),
    }


def batched_scores(actor: PointerActorV2, state: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Û over a padded batch, using the imported actor's own submodules unchanged.

    The per-state path in ``PointerActorV2.forward`` takes one state at a time; this is the
    same arithmetic with the state broadcast over the menu axis.  ``test_batched_equals_
    per_state`` asserts the two agree, because a batching shortcut can silently permute
    results.
    """
    s = actor.state(state).unsqueeze(1).expand(-1, q.shape[1], -1)
    e = actor.enc(q)
    return actor.out(torch.cat((s, e), dim=2)).squeeze(2)


def pair_terms(scores: torch.Tensor, r: torch.Tensor, mask: torch.Tensor):
    """Within-state ordered pairs (j,k) with r_j > r_k, as a boolean [S,M,M] selector."""
    valid = mask.unsqueeze(2) & mask.unsqueeze(1)
    better = (r.unsqueeze(2) > r.unsqueeze(1)) & valid
    diff = scores.unsqueeze(2) - scores.unsqueeze(1)
    return diff, better


def ranking_loss(scores: torch.Tensor, r: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    diff, better = pair_terms(scores, r, mask)
    n = better.sum()
    if n == 0:
        return scores.sum() * 0.0
    return (F.softplus(-diff) * better).sum() / n


def ranking_accuracy(scores: torch.Tensor, r: torch.Tensor, mask: torch.Tensor) -> float:
    diff, better = pair_terms(scores, r, mask)
    n = int(better.sum())
    if n == 0:
        return float("nan")
    return float(((diff > 0) & better).sum().item() / n)


# ----------------------------------------------------------------------------- training


def train_utility(packed: dict[str, torch.Tensor], internal: dict[str, torch.Tensor],
                  seed: int, device: str, max_epochs: int,
                  on_epoch: Any | None = None) -> tuple[PointerActorV2, dict[str, Any]]:
    torch.manual_seed(seed)
    actor = PointerActorV2(CODEC_DIM).to(device)
    opt = torch.optim.Adam(actor.parameters(), lr=LR)
    rng = np.random.default_rng(seed)
    n = len(packed["state"])
    best_acc, best_epoch, best_state, stale = -np.inf, 0, None, 0
    rows: list[dict[str, Any]] = []
    for epoch in range(1, max_epochs + 1):
        actor.train()
        order = rng.permutation(n)
        losses = []
        for start in range(0, n, BATCH_STATES):
            sel = torch.from_numpy(order[start:start + BATCH_STATES].copy())
            st, q = packed["state"][sel].to(device), packed["q"][sel].to(device)
            r, a, m = (packed["r"][sel].to(device), packed["advantage"][sel].to(device),
                       packed["mask"][sel].to(device))
            scores = batched_scores(actor, st, q)
            loss = ranking_loss(scores, r, m)
            if m.any():
                loss = loss + LAMBDA_REG * (((scores - a) ** 2) * m).sum() / m.sum()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), GRAD_CLIP)
            opt.step()
            losses.append(float(loss.item()))
        actor.eval()
        with torch.no_grad():
            acc = ranking_accuracy(
                batched_scores(actor, internal["state"].to(device), internal["q"].to(device)),
                internal["r"].to(device), internal["mask"].to(device))
        rows.append({"epoch": epoch, "loss_mean": float(np.mean(losses)), "internal_rank_acc": acc})
        if np.isfinite(acc) and acc > best_acc + 1e-4:
            best_acc, best_epoch, stale = acc, epoch, 0
            best_state = {k: v.detach().clone() for k, v in actor.state_dict().items()}
        else:
            stale += 1
        if on_epoch is not None:
            on_epoch(epoch, max_epochs, rows[-1])
        if stale >= PATIENCE:
            break
    if best_state is None:
        raise RuntimeError(f"Û seed {seed} produced no finite internal-validation checkpoint")
    actor.load_state_dict(best_state)
    actor.eval()
    return actor, {"seed": seed, "selected_epoch": best_epoch, "best_internal_rank_acc": best_acc,
                   "epochs_run": len(rows), "epoch_rows": rows,
                   "stopping_reason": "patience" if stale >= PATIENCE else "cap"}


# --------------------------------------------------------------------------- evaluation


def _spearman_rows(scores: np.ndarray, r: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Per-state Spearman between an instrument's scores and the true realised advantage."""
    from ga_1_common import spearman_rank

    out = np.full(len(scores), np.nan)
    for i in range(len(scores)):
        k = mask[i]
        if k.sum() < MIN_HELDOUT_CANDIDATES:
            continue
        out[i] = spearman_rank(scores[i][k], r[i][k])
    return out


def _pick_values(scores: np.ndarray, r: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = np.full(len(scores), np.nan)
    for i in range(len(scores)):
        k = np.flatnonzero(mask[i])
        if len(k) < MIN_HELDOUT_CANDIDATES:
            continue
        out[i] = r[i][k[int(np.argmax(scores[i][k]))]]
    return out


def _random_scores(learners: Sequence[str], budgets: np.ndarray, steps: np.ndarray,
                   shape: tuple[int, int], tag: str) -> np.ndarray:
    """CRN random comparator: content-addressed per state, identical across instruments."""
    out = np.zeros(shape, dtype=np.float64)
    for i, (ln, b, k) in enumerate(zip(learners, budgets, steps)):
        rng = np.random.default_rng(stable_uint64(f"{tag}|{ln}|B{int(b)}|k{int(k)}") % (2 ** 63))
        out[i] = rng.random(shape[1])
    return out


def _per_learner(values: np.ndarray, learners: np.ndarray) -> np.ndarray:
    keep = np.isfinite(values)
    ids = sorted(set(learners[keep].tolist()))
    return np.asarray([values[keep & (learners == x)].mean() for x in ids], dtype=float)


def evaluate(actors: Sequence[PointerActorV2], view: dict[str, torch.Tensor],
             learners: np.ndarray, budgets: np.ndarray, steps: np.ndarray,
             device: str, label: str, boot_seed: int) -> dict[str, Any]:
    r = view["r"].numpy()
    mask = view["mask"].numpy()
    epig = view["epig"].numpy()
    rnd = _random_scores(learners, budgets, steps, r.shape, f"ga1-tier4-random|{label}")
    eligible = mask.sum(axis=1) >= MIN_HELDOUT_CANDIDATES

    u_scores = []
    for actor in actors:
        with torch.no_grad():
            u_scores.append(batched_scores(actor, view["state"].to(device),
                                           view["q"].to(device)).cpu().numpy())
    shuffled = []
    for si, u in enumerate(u_scores):
        rng = np.random.default_rng(20260905 + si)
        z = u.copy()
        for i in range(len(z)):
            k = np.flatnonzero(mask[i])
            z[i, k] = z[i, rng.permutation(k)]
        shuffled.append(z)

    def cells(score_sets: Sequence[np.ndarray], fn) -> list[np.ndarray]:
        return [_per_learner(fn(s, r, mask), learners) for s in score_sets]

    seed_sets = {
        "utility": u_scores,
        "utility_shuffled_null": shuffled,
        "greedy_epig": [epig] * len(u_scores),
        "random": [rnd] * len(u_scores),
    }
    out: dict[str, Any] = {
        "label": label,
        "n_states_total": int(len(r)),
        "n_states_eligible": int(eligible.sum()),
        "n_learners": int(len(set(learners[eligible].tolist()))),
        "mean_eligible_menu_size": float(mask.sum(axis=1)[eligible].mean()) if eligible.any() else float("nan"),
        "spearman": {}, "picked_r": {}, "paired_vs_epig": {}, "paired_vs_random": {},
    }
    ref_best = _per_learner(np.where(eligible, np.where(mask, r, -np.inf).max(axis=1), np.nan), learners)
    ref_mean = _per_learner(np.where(eligible, (r * mask).sum(axis=1) / np.maximum(mask.sum(axis=1), 1), np.nan), learners)
    out["reference_oracle_best_r"] = _bootstrap_seed_learner([ref_best], boot_seed, N_BOOT)
    out["reference_menu_mean_r"] = _bootstrap_seed_learner([ref_mean], boot_seed + 1, N_BOOT)

    picked = {}
    for name, sets in seed_sets.items():
        out["spearman"][name] = _bootstrap_seed_learner(cells(sets, _spearman_rows), boot_seed + 2, N_BOOT)
        picked[name] = cells(sets, _pick_values)
        out["picked_r"][name] = _bootstrap_seed_learner(picked[name], boot_seed + 3, N_BOOT)
    out["paired_vs_shuffled_null"] = {}
    for name in ("utility", "greedy_epig", "random", "utility_shuffled_null"):
        out["paired_vs_epig"][name] = _bootstrap_seed_learner(
            [a - b for a, b in zip(picked[name], picked["greedy_epig"])], boot_seed + 4, N_BOOT)
        out["paired_vs_random"][name] = _bootstrap_seed_learner(
            [a - b for a, b in zip(picked[name], picked["random"])], boot_seed + 5, N_BOOT)
        out["paired_vs_shuffled_null"][name] = _bootstrap_seed_learner(
            [a - b for a, b in zip(picked[name], picked["utility_shuffled_null"])],
            boot_seed + 6, N_BOOT)
    # Instrument check: ONE CRN random draw is itself a random
    # variable, and the learner-clustered bootstrap holds that draw fixed -- so a lucky or
    # unlucky draw shifts every "vs random" margin without moving any interval.  The
    # analytic chance line is the menu's own mean r; this band prices how far a single draw
    # can sit from it, and is reported beside the CRN row rather than replacing it.
    band = []
    for d in range(RANDOM_BAND_DRAWS):
        rng = np.random.default_rng(boot_seed + 90000 + d)
        v = np.full(len(r), np.nan)
        for i in range(len(r)):
            k = np.flatnonzero(mask[i])
            if len(k) >= MIN_HELDOUT_CANDIDATES:
                v[i] = r[i][k[int(rng.integers(len(k)))]]
        band.append(float(_per_learner(v, learners).mean()))
    out["random_draw_band"] = {"n_draws": RANDOM_BAND_DRAWS, "mean": float(np.mean(band)),
                               "sd": float(np.std(band)),
                               "q025": float(np.quantile(band, 0.025)),
                               "q975": float(np.quantile(band, 0.975)),
                               "crn_draw_z": float((out["picked_r"]["random"][0] - np.mean(band))
                                                   / max(np.std(band), 1e-12))}
    return out


# --------------------------------------------------------------------------------- main


def _table_for(field: dict[str, Any], episodes: list[dict], learner_ids: set[str], device: str,
               progress: NumericProgress, stage: str, budgets=BUDGETS) -> tuple[dict, list, list]:
    subset = [e for e in episodes if str(e["learner"]) in learner_ids]
    states = draw_states(subset, budgets)
    digest = draw_digest(states, subset)
    table = build_table(field["model"], subset, states, device,
                        on_state=lambda d, t: progress.maybe_heartbeat(stage, d, t))
    table["draw_digest"] = np.asarray([digest])
    return table, states, subset


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--substrate", default=SUBSTRATE)
    ap.add_argument("--out-dir", default="results/ga_1_tier4_utility")
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--pilot-learners", type=int, default=20)
    ap.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    ap.add_argument("--n-seeds", type=int, default=len(UTILITY_SEEDS))
    args = ap.parse_args()

    out = REPO / args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    progress = NumericProgress(out_dir=out)
    t0 = time.time()

    field = load_field(args.substrate, args.raw_dir, args.device)
    codec, pid_to_idx = codec_matrix(field["manifest"])
    train_eps = field["scenarios"]["train_start0"]
    val_eps = field["scenarios"]["val_start0"]
    fit_ids = fit_learners(field["scenarios"])
    u_fit, u_int = utility_learner_split(fit_ids)
    train_items, held_items = item_split(sorted(pid_to_idx))
    if args.pilot:
        u_fit, u_int = u_fit[:args.pilot_learners], u_int[:max(4, args.pilot_learners // 4)]
        val_eps = val_eps[:args.pilot_learners]
    progress.heartbeat("field", 1, 1, n_fit=len(fit_ids), n_u_fit=len(u_fit), n_u_int=len(u_int))

    fit_table, fit_states, fit_subset = _table_for(field, train_eps, set(u_fit), args.device,
                                                   progress, "advantages_fit")
    int_table, int_states, _ = _table_for(field, train_eps, set(u_int), args.device,
                                          progress, "advantages_internal")
    test_table, test_states, test_subset = _table_for(field, val_eps,
                                                      {str(e["learner"]) for e in val_eps},
                                                      args.device, progress, "advantages_test")

    held = set(held_items)
    is_held = {k: np.isin(t["cand_problem_id"], list(held))
               for k, t in (("fit", fit_table), ("int", int_table), ("test", test_table))}
    # Check 1: item disjointness, asserted on the realised training rows by value.
    fit_view = pack(fit_table, codec, pid_to_idx, keep=~is_held["fit"])
    int_view = pack(int_table, codec, pid_to_idx, keep=~is_held["int"])
    for name, view, table, mask_held in (("fit", fit_view, fit_table, is_held["fit"]),
                                         ("int", int_view, int_table, is_held["int"])):
        off, cnt = table["state_offset"], table["state_count"]
        for i, (o, c) in enumerate(zip(off, cnt)):
            n_kept = int(view["sizes"][i])
            if n_kept != int((~mask_held[o:o + c]).sum()):
                raise RuntimeError(f"{name}: item-disjoint view lost rows at state {i}")
        if np.isin(table["cand_problem_id"][mask_held], train_items).any():
            raise RuntimeError(f"{name}: held-out problem id appears in the training item set")

    test_held_view = pack(test_table, codec, pid_to_idx, keep=is_held["test"])
    test_train_view = pack(test_table, codec, pid_to_idx, keep=~is_held["test"])

    # Check 2: reward reconstruction through an independent code path.
    probe_errors = []
    rng = np.random.default_rng(20260903)
    probe_idx = rng.choice(len(test_states), size=min(PROBE_STATES, len(test_states)), replace=False)
    for i in probe_idx:
        st = test_states[int(i)]
        row = state_rows(field["model"], test_subset[st["episode_index"]], st, args.device,
                         with_epig=False)
        o, c = int(test_table["state_offset"][i]), int(test_table["state_count"][i])
        probe_errors.append(float(np.abs(row["r"] - test_table["cand_r"][o:o + c]).max()))
    max_probe_error = float(np.max(probe_errors))
    if max_probe_error > 1e-9:
        raise RuntimeError(f"reward reconstruction check failed: {max_probe_error}")

    actors, train_meta = [], []
    for seed in UTILITY_SEEDS[:args.n_seeds]:
        actor, meta = train_utility(
            fit_view, int_view, seed, args.device, args.max_epochs,
            on_epoch=lambda e, t, row, s=seed: progress.heartbeat(
                f"train_seed{s}", e, t, **{k: v for k, v in row.items() if k != "epoch"}))
        actors.append(actor)
        train_meta.append(meta)
        torch.save({"utility": actor.state_dict(), **{k: v for k, v in meta.items()
                                                      if k != "epoch_rows"}},
                   out / f"utility_seed{seed}.pt")

    learners = test_table["state_learner"]
    budgets, steps = test_table["state_budget"], test_table["state_step"]
    results = {
        "heldout_items": evaluate(actors, test_held_view, learners, budgets, steps,
                                  args.device, "heldout_items", 20260910),
        "train_items_positive_control": evaluate(actors, test_train_view, learners, budgets, steps,
                                                 args.device, "train_items", 20260920),
    }

    manifest = {
        "lane": "T4", "rung": "A", "substrate": args.substrate,
        "bundle": {k: v for k, v in field["meta"].items() if k != "handoff"},
        "handoff": field["meta"].get("handoff"),
        "pilot": bool(args.pilot),
        "n_train_start0_learners": len({str(e["learner"]) for e in train_eps}),
        "n_fit_learners": len(fit_ids),
        "n_utility_fit_learners": len(u_fit),
        "n_utility_internal_learners": len(u_int),
        "n_val_learners": len({str(e["learner"]) for e in val_eps}),
        "budgets": list(BUDGETS), "start": 0,
        "n_states": {"fit": len(fit_states), "internal": len(int_states), "test": len(test_states)},
        "draw_digest": {"fit": str(fit_table["draw_digest"][0]),
                        "internal": str(int_table["draw_digest"][0]),
                        "test": str(test_table["draw_digest"][0])},
        "item_split": {"seed": 20260903, "train_items": train_items, "heldout_items": held_items},
        "utility": {"input_dim": STATE_DIM + CODEC_DIM + FEAT_DIM, "architecture": "rl_2.PointerActorV2",
                    "lr": LR, "lambda_regression": LAMBDA_REG, "grad_clip": GRAD_CLIP,
                    "max_epochs": args.max_epochs, "patience": PATIENCE,
                    "batch_states": BATCH_STATES, "seeds": list(UTILITY_SEEDS[:args.n_seeds])},
        "canaries": {"item_disjoint_by_value": "pass",
                     "reward_reconstruction_max_abs_error": max_probe_error,
                     "reward_reconstruction_probe_states": int(len(probe_idx))},
        "train_meta": train_meta,
        "elapsed_s": time.time() - t0,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    (out / "summary.json").write_text(json.dumps(
        {"status": "complete", "manifest": manifest, "results": results}, indent=2, default=str) + "\n")
    np.savez_compressed(out / "test_table.npz",
                        **{k: v for k, v in test_table.items() if k != "draw_digest"})
    print(json.dumps(results, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
