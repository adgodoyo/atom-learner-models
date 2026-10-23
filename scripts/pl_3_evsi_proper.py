#!/usr/bin/env python
"""Pre-registered replay of the EPIG planner against blind baselines on FoundationalASSIST
validation learners, on the repaired estimator, harness and belief (context for §8.2).

All three repairs hold simultaneously here:
  * the estimator -- the co-information form of `pl_1_epig_estimator.py` (the literal one-term
    formula ranks by `I(Y_e;S) - I(Y_e;Y_t)`: candidate uncertainty minus the true objective),
    verified to 3.3e-16 against analytic MI and certified end-to-end by the planted-world
    positive control (`pl_2_positive_control.py`, 750/750 diagnostic picks);
  * the harness -- the two leak repairs of `pl_0_replay_repair.py` (correctness-only
    `_materialize`; a pool with no outcome channel to filter on);
  * the belief -- the query-independent `b_t = f(H_<t)` filter (S1v2) on the rebuilt v2
    manifest, joined by value.

Both `v_flavor` arms (old and new definitions of V) run on the identical pool under CRN, so the
estimator's contribution is separable from everything else. The margin is pre-registered as a
fraction of measured headroom, taken from the headroom curve on this exact belief and substrate
(`pl_2b_headroom_reference.py`), never as an absolute number of nats: an absolute bar can sit
above what any policy could achieve. A prediction is registered in advance: on this belief the
K-sample Jensen gap -- the quantity V(b) is built from -- grows ~8.5% under 10x more evidence
instead of shrinking, while target log-loss improves; with the estimator and harness checked, a
failure here points at the belief. V(b) is logged per budget step.
Outputs: `reports/pl_3_evsi_proper.md`, `artifacts/pl_3_evsi_proper_curves.npz`, and per-learner
partial results keyed by config hash (resumable).
Run: `python scripts/pl_3_evsi_proper.py [--pilot]` (CPU).
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import argparse  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
import resource  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

from atom.data.fa_manifest import load_v2_events  # noqa: E402
from atom.eval.metrics import bootstrap_mean_ci  # noqa: E402
from atom.eval.report import gate_report  # noqa: E402
from atom.filter.batch import (  # noqa: E402
    build_padded_batch_v2,
    check_coverage,
    load_production_codec,
)
from atom.filter.s1v2 import S1FilterV2  # noqa: E402
from atom.heads.s1_heads import FiLMHeads  # noqa: E402
from atom.planner.candidates import (  # noqa: E402
    fit_pid_clusters,
    recency_bounded_pool,
    stratified_candidates,
)
from atom.planner.evsi import run_episode  # noqa: E402
from atom.planner.replay_v2 import predict_on_targets_v2  # noqa: E402
from atom.utils.checkpoints import load_checkpoint  # noqa: E402
from atom.utils.logging import RunLogger  # noqa: E402
from atom.utils.seeds import seed_everything  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[1]
DEVICE = "cpu"


def _host_pressure() -> str:
    """RSS + system swap, logged at every heartbeat.

    Not decoration: throughput can degrade from ~4s/learner to ~190s/learner at the same
    ELAPSED point (not the same learner -- those learners take ~5s each in isolation), which
    points at host memory pressure rather than anything in the data. Without this line in the
    log a resumed run cannot tell "the machine was swapping" from "the code is slow", and the
    two have opposite fixes.
    """
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 3)  # macOS: bytes
    try:
        sw = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout
        used = re.search(r"used = ([\d.]+)M", sw)
        swap = f"{float(used.group(1)) / 1024:.2f}GB" if used else "?"
    except Exception:
        swap = "?"
    return f"rss={rss:.2f}GB swap_used={swap}"


def v2_predictor(filter_model, heads, window, target_codec, k, device, noise):
    """The injected belief path (see `evsi._predict`). The POLICY code is unchanged and is
    the same code the positive control certified; only the belief underneath it is v2."""
    return predict_on_targets_v2(filter_model, heads, window["item_codec"], window["correct"],
                                 window["hint_bin"], window["saw_answer"],
                                 window["score_missing"], target_codec, k, device, noise=noise)


def build_episode(batch, n: int, length: int, cfg: dict, pid_clusters, pids_row):
    """Replay geometry: targets are the last `n_target` LABEL-VALID events; the pool is the
    `pool_lookback` events immediately preceding the target window (no outcome filtering of
    any kind -- the repaired rule of pl_0_replay_repair.py); candidates are cluster-stratified WITHIN
    that window for diversity; the prefix is everything before the pool."""
    g = cfg["geometry"]
    label_valid = batch["label_valid"][n].numpy()
    valid_idx = np.where(label_valid[:length])[0]
    if len(valid_idx) < g["n_target"] + 5:
        return None
    target_idx = valid_idx[-g["n_target"]:]
    target_start = int(target_idx[0])

    pool_idx = recency_bounded_pool(target_start, g["pool_lookback"])
    # An askable action must have an outcome the EVSI branches can enumerate. A row the
    # platform never graded has none, so it is not an action -- the SAME scoring-necessity
    # restriction applied to targets, and NOT outcome-based selection (it is a
    # property of the data collection, not of the learner's future behaviour). Its size is
    # reported so the restriction is visible rather than silent.
    pool_idx = pool_idx[label_valid[pool_idx]]
    if len(pool_idx) < g["candidate_w"]:
        return None
    clusters = np.array([pid_clusters.get(int(pids_row[i]), -1) for i in pool_idx])
    cand_idx, n_clusters = stratified_candidates(pool_idx, clusters, g["candidate_w"])
    if len(cand_idx) < g["candidate_w"]:
        return None
    prefix_idx = np.arange(0, int(pool_idx[0]))
    return {"prefix_idx": prefix_idx, "cand_idx": cand_idx, "target_idx": target_idx,
            "n_clusters": n_clusters, "target_start": target_start, "length": length}


def windows(batch, n: int, ep: dict):
    def sl(idx):
        t = torch.as_tensor(np.asarray(idx, dtype=np.int64))
        return {"item_codec": batch["item_codec"][n].index_select(0, t),
                "correct": batch["correct"][n].index_select(0, t),
                "hint_bin": batch["hint_bin"][n].index_select(0, t),
                "saw_answer": batch["saw_answer"][n].index_select(0, t),
                "score_missing": batch["score_missing"][n].float().index_select(0, t)}
    prefix, cand = sl(ep["prefix_idx"]), sl(ep["cand_idx"])
    tt = torch.as_tensor(np.asarray(ep["target_idx"], dtype=np.int64))
    return prefix, cand, batch["item_codec"][n].index_select(0, tt), batch["correct"][n].index_select(0, tt)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--max-learners", type=int, default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load((REPO / "configs" / "pl_3_evsi_proper.yaml").read_text())
    if args.max_learners:
        cfg["max_learners"] = args.max_learners
    seed_everything(cfg["seed"])
    logger = RunLogger(config={**cfg, "pilot": bool(args.pilot)}, seed=cfg["seed"],
                       data_version="FA-v2-2026-08-09")
    g, s = cfg["geometry"], cfg["source"]

    v2 = yaml.safe_load((REPO / "configs" / "data.yaml").read_text())["fa"]["manifest_v2"]
    logger.heartbeat(step=0, force=True, stage="loading v2 manifest + splits + codec")
    ev = load_v2_events(REPO / v2["manifest_out"], REPO / v2["learner_split_out"],
                        REPO / v2["item_split_out"])
    ev = ev[ev["split_learner"] == s["split"]].copy()
    ev["learner"] = ev["learner"].astype(str)
    codec_matrix, pid_to_idx = load_production_codec(REPO / s["codec"], s["codec_dim"])
    check_coverage(ev["problem_id"].unique(), pid_to_idx)
    batch = build_padded_batch_v2(ev, codec_matrix, pid_to_idx, DEVICE)

    fm = S1FilterV2(s["codec_dim"], s["episode_hidden"], s["filter_hidden"], s["latent_dim"],
                    query_independent=True).to(DEVICE)
    hd = FiLMHeads(s["codec_dim"], s["latent_dim"], s["filter_hidden"]).to(DEVICE)
    ck = load_checkpoint(REPO / "runs" / s["run_id"] / s["variant"], which="best", map_location=DEVICE)
    fm.load_state_dict(ck["filter_model"])
    hd.load_state_dict(ck["heads"])
    fm.eval()
    hd.eval()

    pid_clusters = fit_pid_clusters(codec_matrix, pid_to_idx, g["n_codec_clusters"], cfg["seed"])
    pids = ev.sort_values(["learner", "t"], kind="mergesort")
    learners = batch["learners"]
    lidx = {u: i for i, u in enumerate(learners)}
    t_max = batch["item_codec"].shape[1]
    pid_grid = np.zeros((len(learners), t_max), dtype=np.int64)
    pid_grid[pids["learner"].map(lidx).to_numpy(), pids["t"].to_numpy().astype(np.int64)] = \
        pids["problem_id"].to_numpy()

    lengths = batch["lengths"].numpy()
    episodes = {}
    for n in range(len(learners)):
        if lengths[n] < g["min_length"]:
            continue
        e = build_episode(batch, n, int(lengths[n]), cfg, pid_clusters, pid_grid[n])
        if e is not None:
            episodes[n] = e
    order = sorted(episodes)
    if args.pilot:
        order = order[:cfg["pilot_n"]]
    elif cfg["max_learners"]:
        order = order[:cfg["max_learners"]]
    logger.heartbeat(step=0, force=True,
                     stage=f"{len(episodes)} eligible of {len(learners)} val learners; running {len(order)}")

    # --- geometry: reported for every planner result, never assumed ------------------------
    pos_frac, dists, overlaps = [], [], []
    for n in order:
        e = episodes[n]
        pos_frac.append(float(np.median(e["cand_idx"]) / e["length"]))
        dists.append(float(e["target_idx"][0] - np.median(e["cand_idx"])))
        cp = set(pid_grid[n][e["cand_idx"]].tolist())
        tp = set(pid_grid[n][e["target_idx"]].tolist())
        overlaps.append(len(cp & tp) / max(1, len(tp)))

    # --- the arms -------------------------------------------------------------------------
    policies, flavors = cfg["arms"]["policies"], cfg["arms"]["v_flavors"]
    ll = {(p, f): np.full(len(order), np.nan) for p in policies for f in flavors}
    acc = {(p, f): np.full(len(order), np.nan) for p in policies for f in flavors}
    picks = {(p, f): [] for p in policies for f in flavors}
    # V(b) itself is logged per budget step alongside log-loss and accuracy, since only the
    # entropy trace makes this visible. It is what separates "greedy loses" from "greedy
    # loses BECAUSE it sharpens V without buying any directional information" -- the
    # signature measured with the old estimator, which this run must be able to confirm or
    # refute on the repaired stack.
    vtrace = {(p, f): np.full((len(order), g["budget"] + 1), np.nan) for p in policies for f in flavors}
    lltrace = {(p, f): np.full((len(order), g["budget"] + 1), np.nan) for p in policies for f in flavors}

    # RESUMABLE EVAL: a run must never be able to lose more than one chunk of eval work.
    # Results are flushed per learner to a path keyed by the CONFIG HASH, so a resumed run can
    # only ever reuse rows computed under an identical configuration. A machine in heavy swap
    # can kill the run part-way, and without partial persistence all of it would be lost.
    part_path = REPO / "artifacts" / f"pl_3_partial_{logger.config_hash}.jsonl"
    done: dict[int, dict] = {}
    if part_path.exists() and not args.pilot:
        for line in part_path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done[int(r["n"])] = r
        logger.heartbeat(step=0, force=True, stage=f"RESUMING: {len(done)} learners already scored")
    part_fh = part_path.open("a") if not args.pilot else None

    t0 = time.time()
    n_computed = 0
    recent: list[float] = []
    # LIVE-OVERRUN TRIGGER: project the total from this run's own measured rate and
    # self-terminate if it exceeds 2x the pre-registered estimate. It is a BROKEN-RUN
    # detector, not a budget -- a drift from 4.1s to 30.3s per learner under system memory
    # pressure is exactly the signal this catches.
    est_hours, overrun_factor = 1.0, 2.0

    for row, n in enumerate(order):
        if n in done:
            r = done[n]
            for f in flavors:
                for pol in policies:
                    ll[(pol, f)][row] = r["ll"][f"{pol}|{f}"]
                    acc[(pol, f)][row] = r["acc"][f"{pol}|{f}"]
                    picks[(pol, f)].append(r["picks"][f"{pol}|{f}"])
                    vtrace[(pol, f)][row] = r["v"][f"{pol}|{f}"]
                    lltrace[(pol, f)][row] = r["llt"][f"{pol}|{f}"]
            continue

        t_learner = time.time()
        e = episodes[n]
        prefix, cand, tgt_codec, tgt_y = windows(batch, n, e)
        gen = torch.Generator().manual_seed(1000 + n)
        noise = torch.randn(g["k"], s["latent_dim"], generator=gen)   # CRN, shared by ALL arms
        for f in flavors:
            for p in policies:
                # max_entropy / max_epistemic select by the candidate's OWN uncertainty and
                # never consult V, so their trace is identical across flavors by construction
                # -- run them once and reuse, rather than paying twice for the same numbers.
                if p in ("max_entropy", "max_epistemic") and f != flavors[0]:
                    ll[(p, f)][row], acc[(p, f)][row] = ll[(p, flavors[0])][row], acc[(p, flavors[0])][row]
                    picks[(p, f)].append(picks[(p, flavors[0])][row])
                    vtrace[(p, f)][row] = vtrace[(p, flavors[0])][row]
                    lltrace[(p, f)][row] = lltrace[(p, flavors[0])][row]
                    continue
                pick_log: list[int] = []
                trace, _ = run_episode(fm, hd, prefix, cand, tgt_codec, tgt_y, g["k"], DEVICE, p,
                                       g["budget"], rng=np.random.default_rng(5000 + n),
                                       noise=noise, pick_log=pick_log, v_flavor=f,
                                       predictor=v2_predictor)
                ll[(p, f)][row], acc[(p, f)][row] = trace[-1]["logloss"], trace[-1]["accuracy"]
                vtrace[(p, f)][row] = [st["V"] for st in trace]
                lltrace[(p, f)][row] = [st["logloss"] for st in trace]
                picks[(p, f)].append(pick_log)
        if part_fh is not None:
            part_fh.write(json.dumps({
                "n": int(n),
                "ll": {f"{pol}|{f}": float(ll[(pol, f)][row]) for pol in policies for f in flavors},
                "acc": {f"{pol}|{f}": float(acc[(pol, f)][row]) for pol in policies for f in flavors},
                "picks": {f"{pol}|{f}": [int(i) for i in picks[(pol, f)][-1]]
                          for pol in policies for f in flavors},
                "v": {f"{pol}|{f}": [float(x) for x in vtrace[(pol, f)][row]]
                      for pol in policies for f in flavors},
                "llt": {f"{pol}|{f}": [float(x) for x in lltrace[(pol, f)][row]]
                        for pol in policies for f in flavors},
            }) + "\n")
            part_fh.flush()
        n_computed += 1
        recent.append(time.time() - t_learner)
        if n_computed % 5 == 0 or n_computed == 1:
            # TRAILING-WINDOW rate, not the cumulative average. The trigger is a BROKEN-RUN
            # detector; a transient host stall is not a broken run, and a cumulative mean is
            # permanently poisoned by one bad patch (five slow learners can drag the average
            # from 4.1s to 30.3s for good, which would self-terminate a healthy run on the
            # strength of a stall that had already passed).
            window = recent[-20:]
            rate = sum(window) / len(window)
            remaining = len(order) - len(done) - n_computed
            proj_h = (rate * (n_computed + remaining)) / 3600.0
            logger.heartbeat(step=row + 1, force=True,
                             stage=f"{row+1}/{len(order)} learners ({n_computed} computed), "
                                   f"{rate:.1f}s each (last {len(window)}), proj {proj_h:.2f}h, "
                                   f"{_host_pressure()}")
            if n_computed >= 20 and proj_h > overrun_factor * est_hours:
                if part_fh is not None:
                    part_fh.close()
                logger.heartbeat(step=row + 1, force=True,
                                 stage=f"LIVE-OVERRUN TRIGGER: projected {proj_h:.2f}h > "
                                       f"{overrun_factor}x{est_hours}h estimate -- self-terminating. "
                                       f"{len(done) + n_computed} learners persisted; re-launch to resume.")
                print(f"LIVE-OVERRUN TRIGGER fired: projected {proj_h:.2f}h. "
                      f"Partial results in {part_path}. Re-launch to resume.")
                return 2

    if part_fh is not None:
        part_fh.close()

    # --- results ---------------------------------------------------------------------------
    ref = cfg["headroom"]["reference_nats"]
    bar = ref * cfg["headroom"]["min_fraction_of_headroom"]
    metrics = {
        "eligible / val learners": f"{len(episodes)} / {len(learners)}",
        "learners scored": len(order),
        "GEOMETRY median candidate position (frac of history)": f"{np.median(pos_frac):.3f}",
        "GEOMETRY median candidate->target distance (events)": f"{np.median(dists):.1f}",
        "GEOMETRY candidate/target item overlap": f"{np.mean(overlaps):.3f}",
        "HEADROOM reference (PL-2b), and the I-11 bar": f"{ref:.4f} nats; bar = {bar:.4f} "
                                                        f"({cfg['headroom']['min_fraction_of_headroom']:.0%} of it)",
    }
    verdicts = {}
    for f in flavors:
        for p in policies:
            metrics[f"[{f}] {p} log-loss"] = f"{np.nanmean(ll[(p, f)]):.4f}"
        for base in ("random", "max_entropy", "max_epistemic"):
            d, lo, hi = bootstrap_mean_ci(ll[(base, f)] - ll[("greedy_evsi", f)],
                                          cfg["bootstrap_n"], cfg["ci_alpha"], 7)
            metrics[f"[{f}] greedy vs {base} (+ = greedy better)"] = f"{d:+.4f} [{lo:+.4f}, {hi:+.4f}]"
            metrics[f"[{f}] greedy vs {base}, as fraction of headroom"] = f"{d / ref:+.2f}x"
            if base == "random":
                verdicts[f] = bool(d >= bar and lo > 0)
        da, la, ha = bootstrap_mean_ci(acc[("greedy_evsi", f)] - acc[("random", f)],
                                       cfg["bootstrap_n"], cfg["ci_alpha"], 8)
        metrics[f"[{f}] greedy vs random, ACCURACY"] = f"{da:+.4f} [{la:+.4f}, {ha:+.4f}]"
        flat = [i for pl in picks[("greedy_evsi", f)] for i in pl]
        obs = [float(batch["correct"][n][episodes[n]["cand_idx"][i]]) for n, pl in
               zip(order, picks[("greedy_evsi", f)]) for i in pl]
        pool_rate = float(np.mean([float(batch["correct"][n][episodes[n]["cand_idx"]].mean())
                                   for n in order]))
        metrics[f"[{f}] PICK COMPOSITION greedy: observed-correct rate vs pool"] = \
            f"{np.mean(obs):.3f} vs {pool_rate:.3f} (n={len(flat)})"
        for pol in ("greedy_evsi", "random"):
            metrics[f"[{f}] V(b) per budget step, {pol}"] = " -> ".join(
                f"{v:.5f}" for v in np.nanmean(vtrace[(pol, f)], axis=0))
            metrics[f"[{f}] log-loss per budget step, {pol}"] = " -> ".join(
                f"{v:.4f}" for v in np.nanmean(lltrace[(pol, f)], axis=0))
        dv = float(np.nanmean(vtrace[("greedy_evsi", f)][:, -1] - vtrace[("greedy_evsi", f)][:, 0]))
        dvr = float(np.nanmean(vtrace[("random", f)][:, -1] - vtrace[("random", f)][:, 0]))
        dll = float(np.nanmean(lltrace[("greedy_evsi", f)][:, -1] - lltrace[("greedy_evsi", f)][:, 0]))
        dllr = float(np.nanmean(lltrace[("random", f)][:, -1] - lltrace[("random", f)][:, 0]))
        metrics[f"[{f}] SHARPENING vs DISCRIMINATION: dV greedy/random, dlog-loss greedy/random"] = \
            f"dV {dv:+.5f}/{dvr:+.5f} | dLL {dll:+.4f}/{dllr:+.4f}"
    for f in flavors:
        metrics[f"[{f}] I-11 VERDICT (greedy beats random by >= {bar:.4f})"] = \
            "PASS" if verdicts[f] else "FAIL"
    metrics["wall (s)"] = f"{time.time() - t0:.0f}"

    notes = [
        "EITHER OUTCOME IS THE CITABLE NUMBER. Earlier runs with the old estimator and harness "
        "give 'the floor of the OLD estimator under the OLD harness', not a standing floor.",
        "The bar is a FRACTION OF MEASURED HEADROOM taken from the headroom curve on this exact "
        f"belief and substrate ({ref:.4f} nats for a budget of questions at large t), not an "
        "absolute number of nats fixed before anyone knew what was achievable. An absolute bar "
        "can sit above the world's ceiling, so that no policy could clear it.",
        "BOTH v_flavor arms ran on the IDENTICAL pool under the SAME common-random-numbers draw "
        "per learner, so the old-vs-new estimator contrast is attributable to V's "
        "definition and nothing else. max_entropy/max_epistemic select by the candidate's own "
        "uncertainty and never consult V, so their traces are identical across flavors by "
        "construction and are computed once.",
        "GEOMETRY IS REPORTED, NOT ASSUMED -- a comparison can be flawlessly "
        "fair and still answer the wrong question. The pool is recency-bounded to the events "
        "immediately preceding the target window, and the median candidate position, "
        "candidate->target distance and item overlap are in the table above.",
        "GIVE-UPS: the pool is never filtered on any outcome channel (excluding give-ups would "
        "use the logged future `saw_answer`, itself a leak). What revealing a "
        "candidate CONVEYS is neutralised instead: correctness-only, enforced in `_materialize` "
        "and pinned by permanent leak canaries.",
        "The candidate pool is restricted to LABEL-VALID rows. An askable action must have an "
        "outcome the EVSI branches can enumerate, and a row the platform never graded has none. "
        "This is a property of the data collection, not of the learner's future behaviour -- the "
        "same scoring-necessity restriction applied to targets, not outcome-based selection.",
        "BELIEF: the query-independent S1v2 filter on the v2 manifest, reached through the "
        "`predictor` injection in `evsi.py` -- the POLICY code is byte-identical to what the "
        "positive control certified, and a test pins that injecting a v1-equivalent predictor "
        "reproduces the default path bit-for-bit.",
        "PREDICTION REGISTERED BEFORE THIS RUN: on this belief the K-sample Jensen gap -- "
        "the quantity V(b) is built from -- GROWS ~8.5% under 10x more evidence instead of "
        "shrinking. With the estimator and the harness checked separately, a failure here "
        "points at the belief rather than at decision-aware acquisition as such.",
        "Cost: $0 (Mac, CPU, single-threaded).",
    ]
    meta = {"run_id": logger.run_id, "git_sha": logger.git_sha, "config_hash": logger.config_hash,
            "device": DEVICE, "seed": cfg["seed"], "data_version": "FA-v2-2026-08-09",
            "belief": f"FA-1 FIXED S1FilterV2 runs/{s['run_id']}/{s['variant']}",
            "geometry": f"pool_lookback {g['pool_lookback']} | candidate_w {g['candidate_w']} | "
                        f"n_target {g['n_target']} | budget {g['budget']} | K {g['k']}"}
    name = "pl_3_evsi_proper" + ("_pilot" if args.pilot else "")
    passed = any(verdicts.values())
    text = gate_report(REPO / "reports" / f"{name}.md",
                       "PL-3 - EPIG vs baselines on the fully repaired stack (FA)",
                       meta, metrics, passed,
                       f"I-11: greedy-EPIG beats random by >= {bar:.4f} nats "
                       f"({cfg['headroom']['min_fraction_of_headroom']:.0%} of the measured "
                       "headroom) with the 95% paired CI clear of zero, in at least one v_flavor arm. "
                       "EITHER OUTCOME IS PUBLISHABLE -- this is a measurement, not a target.",
                       notes)
    np.savez(REPO / "artifacts" / f"{name}_curves.npz",
             **{f"{p}_{f}_logloss": ll[(p, f)] for p in policies for f in flavors},
             **{f"{p}_{f}_accuracy": acc[(p, f)] for p in policies for f in flavors},
             **{f"{p}_{f}_vtrace": vtrace[(p, f)] for p in policies for f in flavors},
             **{f"{p}_{f}_lltrace": lltrace[(p, f)] for p in policies for f in flavors},
             learners=np.array([learners[n] for n in order], dtype=str),
             pos_frac=np.array(pos_frac), dists=np.array(dists), overlaps=np.array(overlaps))
    print(text)
    print(f"\nVERDICT: {'PASS' if passed else 'FAIL'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
