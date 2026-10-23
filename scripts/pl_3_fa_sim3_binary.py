#!/usr/bin/env python
"""The EPIG planner on the belief of FA-SIM3, the mathematics simulator (§8.2): a four-policy
replay comparison on FoundationalASSIST validation learners.

The FA-SIM3 filter and its OutcomeHead stay frozen. In the default binary arm (B), target
outcomes are coarsened to P(UNAIDED_CORRECT); `--answer-space categorical` runs the native
five-class arm (A), where acquired observations stay binary. The comparison runs under both the
Jensen/EPIG value and the total-Jensen diagnostic. The policy harness is `atom.planner.evsi`
unchanged; only the belief/predictor and the data substrate differ. Random's own prefix-to-budget
movement on this pool is measured first and fixes the 25%-of-headroom bar before any other
policy is inspected.

The all-event belief table (b_t = f(H_<t)) is written once and checked row by row against an
independent manifest read (learner/id/problem_id/t). Outputs: the belief table, resumable
scalar partials, a report and summaries; with ``--bucket`` the large belief artifact and
partials are also uploaded to that S3 bucket.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import resource
import sys
import time
from typing import Any

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

from atom.data.fa_manifest import load_v2_events  # noqa: E402
from atom.data.fa_outcome_label import outcome_label  # noqa: E402
from atom.eval.metrics import bootstrap_mean_ci  # noqa: E402
from atom.eval.report import gate_report  # noqa: E402
from atom.filter.episode_v2 import build_prev_feat_v2  # noqa: E402
from atom.filter.item_encoder_mlp import S1FilterV2ItemMLP  # noqa: E402
from atom.filter.batch import load_production_codec  # noqa: E402
from atom.planner.candidates import fit_pid_clusters, recency_bounded_pool, stratified_candidates  # noqa: E402
from atom.planner.evsi import run_episode  # noqa: E402
from atom.planner.replay_sim3 import (  # noqa: E402
    categorical_score_step,
    load_sim3_stack,
    predict_on_targets_sim3,
    sigma_trace,
)
from atom.utils.logging import RunLogger  # noqa: E402
from atom.utils.seeds import seed_everything  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[1]
POLICIES = ("greedy_evsi", "random", "max_entropy", "max_epistemic")
FLAVORS = ("jensen", "total")


def _rss_gb() -> float:
    # macOS reports bytes; Linux reports KiB.
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return float(value / (1024 ** 3) if sys.platform == "darwin" else value / (1024 ** 2))


def _maybe_upload(path: pathlib.Path, bucket: str, key: str) -> None:
    if not bucket:
        return
    import boto3

    boto3.client("s3").upload_file(str(path), bucket, key)


def _sim3_predictor(filter_model, heads, window, target_codec, k, device, noise,
                    answer_space="binary"):
    if noise is not None:
        noise = noise.to(device)
    return predict_on_targets_sim3(
        filter_model, heads, window["item_codec"], window["correct"], window["hint_bin"],
        window["saw_answer"], window["score_missing"], target_codec, k, device,
        noise=noise, answer_space=answer_space,
    )


def _window(rows: pd.DataFrame, positions: np.ndarray, codec: np.ndarray,
            pid_to_idx: dict[int, int]) -> dict[str, torch.Tensor]:
    sub = rows.iloc[np.asarray(positions, dtype=np.int64)]
    item = torch.from_numpy(codec[sub["problem_id"].map(pid_to_idx).to_numpy()])
    score = sub["score"].astype("float32").fillna(0.0).to_numpy()
    return {
        "item_codec": item,
        "correct": torch.from_numpy(score.copy()),
        "hint_bin": torch.from_numpy((sub["hint_count"].to_numpy() > 0).astype(np.float32)),
        "saw_answer": torch.from_numpy(sub["saw_answer"].to_numpy(dtype=np.float32).copy()),
        "score_missing": torch.from_numpy(sub["score_missing"].to_numpy(dtype=bool).copy()),
    }


def _episodes(ev: pd.DataFrame, codec: np.ndarray, pid_to_idx: dict[int, int], cfg: dict,
              clusters: dict[int, int], answer_space: str = "binary") -> list[dict[str, Any]]:
    g = cfg["geometry"]
    out: list[dict[str, Any]] = []
    if answer_space == "categorical":
        labels = outcome_label(ev["support_code"], ev["score"], ev["score_missing"])
        ev = ev.copy()
        ev["outcome_code"] = labels["outcome_code"].to_numpy()
        ev["outcome_valid"] = labels["outcome_valid"].to_numpy()
    elif answer_space != "binary":
        raise ValueError(f"unknown answer_space {answer_space!r}")
    for learner, rows in ev.groupby("learner", sort=True):
        rows = rows.sort_values("t", kind="mergesort").reset_index(drop=True)
        if len(rows) < g["min_length"]:
            continue
        candidate_valid = ~rows["score_missing"].to_numpy(dtype=bool)
        target_valid = (rows["outcome_valid"].to_numpy(dtype=bool)
                        if answer_space == "categorical" else candidate_valid)
        target_valid_idx = np.flatnonzero(target_valid)
        if len(target_valid_idx) < g["n_target"]:
            continue
        target_idx = target_valid_idx[-g["n_target"]:]
        target_start = int(target_idx[0])
        pool_idx = recency_bounded_pool(target_start, g["pool_lookback"])
        pool_idx = pool_idx[candidate_valid[pool_idx]]
        if len(pool_idx) < g["candidate_w"]:
            continue
        pool_clusters = np.array([clusters.get(int(rows.iloc[i]["problem_id"]), -1) for i in pool_idx])
        cand_idx, n_clusters = stratified_candidates(pool_idx, pool_clusters, g["candidate_w"])
        if len(cand_idx) < g["candidate_w"]:
            continue
        prefix_idx = np.arange(0, int(pool_idx[0]), dtype=np.int64)
        out.append({
            "learner": str(learner), "rows": rows, "prefix_idx": prefix_idx,
            "cand_idx": cand_idx, "target_idx": target_idx, "n_clusters": n_clusters,
            "target_start": target_start, "length": len(rows),
            "prefix_pos_frac": float(np.median(cand_idx) / max(1, len(rows))),
            "candidate_target_distance": float(target_idx[0] - np.median(cand_idx)),
            "item_overlap": float(len(set(rows.iloc[cand_idx]["problem_id"]) &
                                      set(rows.iloc[target_idx]["problem_id"])) /
                                max(1, rows.iloc[target_idx]["problem_id"].nunique())),
            "prefix": _window(rows, prefix_idx, codec, pid_to_idx),
            "cand": _window(rows, cand_idx, codec, pid_to_idx),
            "target_codec": _window(rows, target_idx, codec, pid_to_idx)["item_codec"],
            "target_correct": _window(rows, target_idx, codec, pid_to_idx)["correct"],
            "target_label": torch.from_numpy(rows.iloc[target_idx]["outcome_code"].to_numpy(dtype=np.int64))
            if answer_space == "categorical" else _window(rows, target_idx, codec, pid_to_idx)["correct"],
        })
    return out


def _belief_table(filter_model, episodes: list[dict[str, Any]], device: str) -> tuple[dict[str, np.ndarray], dict]:
    learner_col, id_col, pid_col, t_col, mus, logs, traces = [], [], [], [], [], [], []
    with torch.no_grad():
        for ix, ep in enumerate(episodes):
            rows = ep["rows"]
            # The full sequence is safe: query_independent=True shifts item codecs by one,
            # so output position t is b_t=f(H_<t), exactly the planner boundary.
            full = _window(rows, np.arange(len(rows)), ep["_codec"], ep["_pid_to_idx"])
            item = full["item_codec"].unsqueeze(0).to(device)
            correct = full["correct"].unsqueeze(0).to(device)
            hint = full["hint_bin"].unsqueeze(0).to(device)
            saw = full["saw_answer"].unsqueeze(0).to(device)
            missing = full["score_missing"].unsqueeze(0).to(device)
            mu, log_sigma = filter_model(item, build_prev_feat_v2(correct, hint, saw, missing))
            mu, log_sigma = mu[0].cpu().numpy().astype(np.float32), log_sigma[0].cpu().numpy().astype(np.float32)
            learner_col.extend([ep["learner"]] * len(rows))
            id_col.extend(rows["id"].to_numpy(dtype=np.int64).tolist())
            pid_col.extend(rows["problem_id"].to_numpy(dtype=np.int64).tolist())
            t_col.extend(rows["t"].to_numpy(dtype=np.int64).tolist())
            mus.append(mu); logs.append(log_sigma)
            traces.append(float(np.exp(2.0 * log_sigma).sum(axis=-1).mean()))
            if (ix + 1) % 25 == 0:
                print(f"[pl3-b] belief {ix + 1}/{len(episodes)} learners rss={_rss_gb():.2f}GB", flush=True)
    # The stack uses heterogeneous sequence lengths, so concatenate the row-wise outputs.
    return {
        "learner": np.asarray(learner_col, dtype="U32"), "id": np.asarray(id_col, dtype=np.int64),
        "problem_id": np.asarray(pid_col, dtype=np.int64), "t": np.asarray(t_col, dtype=np.int64),
        "mu": np.concatenate(mus, axis=0), "log_sigma": np.concatenate(logs, axis=0),
    }, {"n_rows": len(learner_col), "n_learners": len(episodes),
        "sigma_trace_mean": float(np.mean(traces)), "sigma_trace_min": float(np.min(traces)),
        "sigma_trace_max": float(np.max(traces))}


def _canary_join(beliefs: dict[str, np.ndarray], manifest_path: pathlib.Path,
                 split_path: pathlib.Path, item_path: pathlib.Path) -> dict:
    independent = load_v2_events(manifest_path, split_path, item_path)
    independent = independent[independent["split"] == "val"]
    b = pd.DataFrame({k: beliefs[k] for k in ("learner", "id", "problem_id", "t")})
    if b.duplicated(["learner", "id"]).any():
        raise AssertionError("belief table has duplicate (learner,id) rows")
    j = b.merge(independent[["learner", "id", "problem_id", "t"]],
                on=["learner", "id"], how="left", suffixes=("_belief", "_manifest"),
                validate="one_to_one")
    if j["problem_id_manifest"].isna().any():
        raise AssertionError("belief rows missing from independent manifest")
    id_match = (j["problem_id_belief"].to_numpy() == j["problem_id_manifest"].to_numpy())
    t_match = (j["t_belief"].to_numpy() == j["t_manifest"].to_numpy())
    result = {"n_belief_rows": int(len(b)), "n_manifest_val_rows": int(len(independent)),
              "n_joined": int(len(j)), "problem_id_matches": int(id_match.sum()),
              "t_matches": int(t_match.sum()), "value_join_pass": bool(id_match.all() and t_match.all())}
    if not result["value_join_pass"]:
        raise AssertionError(f"belief value-join check failed: {result}")
    return result


def _run_policy(filter_model, heads, ep: dict[str, Any], cfg: dict, policy: str,
                flavor: str, device: str, seed: int,
                answer_space: str = "binary") -> tuple[list[dict], list[int]]:
    g = cfg["geometry"]
    noise = torch.randn(g["k"], 256, generator=torch.Generator().manual_seed(1000 + seed))
    def move_window(window: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {key: value.to(device) for key, value in window.items()}
    prefix = move_window(ep["prefix"])
    cand = move_window(ep["cand"])
    target_codec = ep["target_codec"].to(device)
    target_correct = ep["target_label"].to(device)
    picks: list[int] = []
    scorer = categorical_score_step if answer_space == "categorical" else None
    trace, _ = run_episode(
        filter_model, heads, prefix, cand, target_codec, target_correct,
        g["k"], device, policy, g["budget"], rng=np.random.default_rng(5000 + seed), noise=noise,
        pick_log=picks, v_flavor=flavor,
        predictor=lambda *args: _sim3_predictor(*args, answer_space=answer_space),
        scorer=scorer,
    )
    return trace, picks


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pl_3_fa_sim3_binary.yaml")
    ap.add_argument("--out-dir", default="results/pl_3_fa_sim3_binary")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--bucket", default="")
    ap.add_argument("--out-prefix", default="pl-3-fa-sim3-binary")
    ap.add_argument("--max-learners", type=int, default=0)
    ap.add_argument("--answer-space", choices=("binary", "categorical"), default="binary",
                    help="binary arm B (default) or native five-class arm A")
    args = ap.parse_args()
    cfg = yaml.safe_load((REPO / args.config).read_text())
    if args.max_learners:
        cfg["run"]["max_learners"] = args.max_learners
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    seed_everything(cfg["seed"])
    answer_space = args.answer_space
    logger = RunLogger(config={**cfg, "device": device, "out_prefix": args.out_prefix,
                               "answer_space": answer_space},
                       seed=cfg["seed"], data_version="FA-SIM3-val-2026-08-23", out_root=REPO / "runs")
    out_dir = REPO / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    s = cfg["source"]
    ev = load_v2_events(REPO / s["manifest"], REPO / s["learner_split"], REPO / s["item_split"])
    ev = ev[ev["split"] == s["split"]].copy()
    codec, pid_to_idx = load_production_codec(REPO / s["codec"], 64)
    if set(ev["problem_id"].unique()) - set(pid_to_idx):
        raise RuntimeError("FA-SIM3 codec coverage failed")
    filt, head, meta = load_sim3_stack(REPO / s["checkpoint"], device=device)
    clusters = fit_pid_clusters(codec, pid_to_idx, cfg["geometry"]["n_codec_clusters"], cfg["seed"])
    eps = _episodes(ev, codec, pid_to_idx, cfg, clusters, answer_space=answer_space)
    for ep in eps:
        ep["_codec"], ep["_pid_to_idx"] = codec, pid_to_idx
    if cfg["run"]["max_learners"]:
        eps = eps[:cfg["run"]["max_learners"]]
    logger.heartbeat(step=0, force=True, stage=f"prepared {len(eps)} episodes; device={device}; rss={_rss_gb():.2f}GB")
    beliefs, belief_meta = _belief_table(filt, eps, device)
    canary = _canary_join(beliefs, REPO / s["manifest"], REPO / s["learner_split"], REPO / s["item_split"])
    beliefs_path = out_dir / "beliefs.npz"
    np.savez(beliefs_path, **beliefs)
    npz_s3_key = f"{args.out_prefix}/beliefs.npz"
    _maybe_upload(beliefs_path, args.bucket, npz_s3_key)
    (out_dir / "belief_canary.json").write_text(json.dumps({"belief": belief_meta, "canary7": canary}, indent=2))
    _maybe_upload(out_dir / "belief_canary.json", args.bucket, f"{args.out_prefix}/belief_canary.json")

    def progress(stage: str, done: int) -> None:
        payload = {"stage": stage, "done": int(done), "total": int(n),
                   "run_id": logger.run_id, "rss_gb": _rss_gb(), "ts": time.time()}
        p = out_dir / "progress.json"
        p.write_text(json.dumps(payload, indent=2))
        _maybe_upload(p, args.bucket, f"{args.out_prefix}/progress.json")

    n = len(eps)
    result: dict[str, dict[str, Any]] = {}
    headroom_values = []
    # Reference first: unrestricted random's own prefix-to-budget movement on this exact
    # FA-SIM3 pool. It fixes the 25% bar before greedy/diagnostic baselines are inspected.
    for i, ep in enumerate(eps):
        trace, picks = _run_policy(filt, head, ep, cfg, "random", "jensen", device, i,
                                   answer_space=answer_space)
        headroom_values.append(float(trace[0]["logloss"] - trace[-1]["logloss"]))
        result.setdefault("random|jensen", {"ll": np.full(n, np.nan), "acc": np.full(n, np.nan), "v": np.full((n, cfg["geometry"]["budget"] + 1), np.nan), "llt": np.full((n, cfg["geometry"]["budget"] + 1), np.nan), "picks": []})
        result["random|jensen"]["ll"][i] = trace[-1]["logloss"]
        result["random|jensen"]["acc"][i] = trace[-1]["accuracy"]
        result["random|jensen"]["v"][i] = [x["V"] for x in trace]
        result["random|jensen"]["llt"][i] = [x["logloss"] for x in trace]
        result["random|jensen"]["picks"].append(picks)
        if (i + 1) % 10 == 0:
            logger.heartbeat(step=i + 1, force=True, stage=f"headroom random {i + 1}/{n}; rss={_rss_gb():.2f}GB")
            progress("headroom_random", i + 1)
    headroom, h_lo, h_hi = bootstrap_mean_ci(np.asarray(headroom_values), cfg["headroom"]["bootstrap_n"], cfg["headroom"]["ci_alpha"], 31)
    bar = max(0.0, headroom) * cfg["headroom"]["fraction"]
    (out_dir / "headroom.json").write_text(json.dumps({"values": headroom_values, "mean": headroom, "lo": h_lo, "hi": h_hi, "bar": bar}, indent=2))
    _maybe_upload(out_dir / "headroom.json", args.bucket, f"{args.out_prefix}/headroom.json")

    for flavor in FLAVORS:
        for policy in POLICIES:
            key = f"{policy}|{flavor}"
            if key in result:
                continue
            result[key] = {"ll": np.full(n, np.nan), "acc": np.full(n, np.nan), "v": np.full((n, cfg["geometry"]["budget"] + 1), np.nan), "llt": np.full((n, cfg["geometry"]["budget"] + 1), np.nan), "picks": []}
            for i, ep in enumerate(eps):
                trace, picks = _run_policy(filt, head, ep, cfg, policy, flavor, device, i,
                                           answer_space=answer_space)
                result[key]["ll"][i] = trace[-1]["logloss"]
                result[key]["acc"][i] = trace[-1]["accuracy"]
                result[key]["v"][i] = [x["V"] for x in trace]
                result[key]["llt"][i] = [x["logloss"] for x in trace]
                result[key]["picks"].append(picks)
                if (i + 1) % 10 == 0:
                    progress(f"{policy}|{flavor}", i + 1)
            logger.heartbeat(step=len(result), force=True, stage=f"completed {key}; rss={_rss_gb():.2f}GB")
            # Small resumable scalar checkpoint; no raw text is involved. If the instance
            # dies, at most the arm currently in flight is lost; completed arms are already
            # local (and uploaded when --bucket is given).
            partial_path = out_dir / f"partial_{policy}_{flavor}.npz"
            np.savez_compressed(partial_path, ll=result[key]["ll"], acc=result[key]["acc"],
                                v=result[key]["v"], llt=result[key]["llt"])
            _maybe_upload(partial_path, args.bucket, f"{args.out_prefix}/{partial_path.name}")

    final = {k: {name: (value.tolist() if isinstance(value, np.ndarray) else value) for name, value in v.items() if name != "picks"} for k, v in result.items()}
    np.savez_compressed(out_dir / "predictions.npz", **{f"{k.replace('|', '__')}__{name}": value for k, v in result.items() for name, value in v.items() if isinstance(value, np.ndarray)})
    (out_dir / "pick_logs.json").write_text(json.dumps({k: v["picks"] for k, v in result.items()}))
    metrics: dict[str, Any] = {
        "run_id": logger.run_id, "git_sha": logger.git_sha, "device": device,
        "answer_space": answer_space,
        "learners": n, "eligible_val_rows": int(len(ev)), "checkpoint_meta": meta,
        "belief_meta": belief_meta, "canary7": canary,
        "geometry": {"median_candidate_position_fraction": float(np.median([e["prefix_pos_frac"] for e in eps])), "median_candidate_target_distance": float(np.median([e["candidate_target_distance"] for e in eps])), "candidate_target_item_overlap": float(np.mean([e["item_overlap"] for e in eps]))},
        "headroom_random_prefix_minus_budget": {"mean": headroom, "lo": h_lo, "hi": h_hi, "bar": bar},
    }
    for flavor in FLAVORS:
        for base in ("random", "max_entropy", "max_epistemic"):
            d, lo, hi = bootstrap_mean_ci(result[f"{base}|{flavor}"]["ll"] - result[f"greedy_evsi|{flavor}"]["ll"], cfg["headroom"]["bootstrap_n"], cfg["headroom"]["ci_alpha"], 37)
            metrics[f"{flavor} greedy-minus-{base} logloss"] = {"mean": d, "lo": lo, "hi": hi, "fraction_headroom": d / headroom if headroom else float("nan")}
        metrics[f"{flavor} final_logloss"] = {p: float(np.mean(result[f"{p}|{flavor}"]["ll"])) for p in POLICIES}
        metrics[f"{flavor} final_accuracy"] = {p: float(np.mean(result[f"{p}|{flavor}"]["acc"])) for p in POLICIES}
    metrics["status"] = "complete"
    (out_dir / "summary.json").write_text(json.dumps(metrics, indent=2, default=str))
    for name in ("predictions.npz", "pick_logs.json", "summary.json"):
        _maybe_upload(out_dir / name, args.bucket, f"{args.out_prefix}/{name}")
    logger.log_metrics(**{"headroom": headroom, "headroom_lo": h_lo, "headroom_hi": h_hi, "bar": bar})
    logger.summary(status="ok", passed=True)
    report_name = "pl_3_fa_sim3_binary.md" if answer_space == "binary" else "pl_3_fa_sim3_categorical.md"
    report_title = ("PL-3 arm B — FA-SIM3 binary target policy comparison"
                    if answer_space == "binary" else
                    "PL-3 arm A — FA-SIM3 native five-class target policy comparison")
    gate_desc = ("Arm B is descriptive/comparative; the headroom is measured on the current FA-SIM3 belief before policy comparisons."
                 if answer_space == "binary" else
                 "Arm A is descriptive/comparative; the headroom is measured on the current FA-SIM3 belief before policy comparisons.")
    gate_report(REPO / "reports" / report_name, report_title,
                {"run_id": logger.run_id, "git_sha": logger.git_sha, "device": device, "seed": cfg["seed"], "checkpoint": s["checkpoint"]},
                metrics, True, gate_desc,
                notes=[("Targets and acquired observations are binary P(UNAIDED_CORRECT)."
                        if answer_space == "binary" else
                        "Targets are native five-class outcomes; acquired observations remain binary P(UNAIDED_CORRECT)."),
                       "Filter and OutcomeHead are frozen; no training or backpropagation.",
                       "The belief check joined every persisted row by learner/id/problem_id/t.",
                       "Large beliefs are retained on the compute instance (and S3 if --bucket is given); only summaries and scalar predictions are copied back."])
    print(json.dumps(metrics, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
