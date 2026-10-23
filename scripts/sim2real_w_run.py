#!/usr/bin/env python3
"""Runs the policy trained only inside the simulator (§8.3, Figure 11): warm-started,
learner-specific imagined menus, actor training on imagined answers only, then the real replay
exam.

The structural check lives in `sim2real_w.py`; this runner consumes its decisions, regenerates
the learner-specific corpus locally with the CodeWorkout generative simulator (refreshed outcome
head), trains the actor only on imagined answers with checkpoint selection on simulated
validation, and then opens the six-cell real held-out replay exam (`imagination_core`).
Generated text stays on the machine that produced it; numeric manifests and curves may be
uploaded to S3. Inputs: ``--handoff`` (train-prefix beliefs) and either ``--raw-dir`` or
``--precomputed-events`` (from `sim2real_w_prepare_events.py`). Outputs in the output
directory: structural_audit.json, real_exam.json and per-cell real_exam_start*_B*.json.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import pathlib
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import torch

REPO = pathlib.Path(__file__).resolve().parents[1]
for p in (REPO / "src", REPO / "scripts", REPO / "infra"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from cw_planner_suite import _build_manifest, _episode_summaries, _event_rows, _make_scenarios  # noqa: E402
from cw_sim_model_dict import build_cwsim_model_dict  # noqa: E402
from im_s3_dream import DreamConfig, DreamCorpusGenerator, load_cw_b_full_generation_model, make_cw_batched_generator  # noqa: E402
from rl_2 import PointerActorV2, _candidate_features, _frozen_filter_state, _point_logits, _trace_policy_v2  # noqa: E402
from imagination_core import Sim2RealConfig, evaluate_real_exam  # noqa: E402
from imagination_cw import (  # noqa: E402
    HANDOFF_SHA256,
    REGISTERED_CHECKPOINT_SHA256,
    encode_corpus_records,
    item_panel,
    load_train_handoff,
    sha256_file,
    train_sim_actor,
)
from sim2real_w import (  # noqa: E402
    FIT_LEARNERS,
    MENU_SIZE,
    N_CLASSES,
    STARTS,
    W_SEED,
    candidate_pool,
    read_split,
    real_event_rows,
    stable_seed,
    structural_audit,
)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def load_precomputed_events(path: pathlib.Path, device: str) -> dict[str, list[dict[str, Any]]]:
    """Load the numeric, locally prepared event handoff (no licensed source text)."""
    z = np.load(path, allow_pickle=False)
    meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    required = {"learner_idx", "episode_idx", "problem_id", "order", "label", "release_idx", "codec", "summary"}
    missing = required - set(z.files)
    if missing:
        raise ValueError(f"precomputed event handoff missing {sorted(missing)}")
    n = len(z["problem_id"])
    if any(len(z[k]) != n for k in ("learner_idx", "episode_idx", "order", "label", "release_idx")):
        raise ValueError("precomputed event arrays have inconsistent lengths")
    if z["codec"].shape[0] != n or z["summary"].shape[0] != n:
        raise ValueError("precomputed event tensor rows do not match metadata")
    learners = list(meta["learners"]); episodes = list(meta["episode_ids"]); releases = list(meta["releases"])
    # One shared tensor per field avoids allocating tens of thousands of independent
    # storages (the naive per-row ``torch.as_tensor`` path can exhaust a laptop's memory).
    codec_all = torch.as_tensor(z["codec"], dtype=torch.float32, device=device)
    summary_all = torch.as_tensor(z["summary"], dtype=torch.float32, device=device)
    out: dict[str, list[dict[str, Any]]] = {}
    for i in range(n):
        learner = str(learners[int(z["learner_idx"][i])])
        out.setdefault(learner, []).append({
            "order": int(z["order"][i]),
            "episode_id": str(episodes[int(z["episode_idx"][i])]),
            "problem_id": int(z["problem_id"][i]),
            "release": str(releases[int(z["release_idx"][i])]),
            "codec": codec_all[i],
            "summary": summary_all[i],
            "label": int(z["label"][i]),
        })
    for rows in out.values():
        rows.sort(key=lambda x: x["order"])
    if int(meta.get("n_rows", n)) != n or int(meta.get("n_learners", len(out))) != len(out):
        raise ValueError("precomputed event metadata count mismatch")
    return out


def scenarios_from_events(events: Mapping[str, Sequence[Mapping[str, Any]]], learners: Sequence[str]) -> dict[str, list[dict[str, Any]]]:
    """Build the same start/candidate/target partition without raw source tables."""
    scenarios: dict[str, list[dict[str, Any]]] = {}
    for start in STARTS:
        rows: list[dict[str, Any]] = []
        for learner in sorted(map(str, learners)):
            all_events = list(events.get(learner, []))
            if len(all_events) < start + 2:
                continue
            target_n = min(5, len(all_events) - start - 1)
            targets = all_events[-target_n:]
            middle = all_events[start:-target_n]
            if not middle:
                continue
            rows.append({
                "learner": learner,
                "semester": str(targets[0].get("release", "pooled")),
                "prefix_events": all_events[:start],
                "full_events": all_events[:-target_n],
                "candidates": middle[:50],
                "targets": targets,
                "menu_size": len(middle[:50]),
            })
        scenarios[f"val_start{start}"] = rows
    return scenarios


def _args_for_manifest(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        derived_val_predictions=None,
        raw_dir=args.raw_dir,
        code_map=args.code_map,
        diff_map=args.diff_map,
        codealone_meta=args.codealone_meta,
        codealone_states=args.codealone_states,
        diff_meta=args.diff_meta,
        diff_states=args.diff_states,
        learner_batch_size=8,
        epochs=1,
        lr=1e-3,
    )


def plan_w_episodes(
    *,
    manifest: Mapping[str, Any],
    summaries: Mapping[str, torch.Tensor],
    model: Mapping[str, Any],
    handoff: Mapping[str, Any],
    device: str,
    seed: int = W_SEED,
    events_override: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Create warm, learner-specific imagined menus before any text generation."""

    by_learner = dict(events_override) if events_override is not None else _event_rows(dict(manifest), dict(summaries), device)
    # The rl_2.py fit IDs are the first 543 sorted train-start-0 learners.  This is a
    # value join, not a positional join to the handoff NPZ.
    if events_override is None:
        stock = _make_scenarios(dict(manifest), dict(summaries), device)
        fit_ids = sorted({str(ep["learner"]) for ep in stock.get("train_start0", [])})[:FIT_LEARNERS]
    else:
        split = read_split(REPO / "splits/cw_learner_split.csv")
        fit_ids = sorted(str(learner) for learner, rows in by_learner.items()
                         if split.get(str(learner)) == "train" and len(rows) >= 2)[:FIT_LEARNERS]
    if len(fit_ids) != FIT_LEARNERS:
        raise AssertionError(f"expected {FIT_LEARNERS} rl_2 fit learners, got {len(fit_ids)}")

    planned: list[dict[str, Any]] = []
    for learner in fit_ids:
        all_events = list(by_learner.get(learner, []))
        feasible: list[int] = []
        pools: dict[int, tuple[list, list, list]] = {}
        for start in STARTS:
            try:
                partition = candidate_pool(all_events, start)
            except ValueError:
                continue
            if partition[1]:
                feasible.append(int(start)); pools[int(start)] = partition
        if not feasible:
            raise AssertionError(f"no feasible warm start for {learner}")
        rng = np.random.default_rng(stable_seed("start", learner, seed=seed))
        start = int(rng.choice(np.asarray(feasible, dtype=int)))
        prefix, pool, targets = pools[start]
        n_menu = min(MENU_SIZE, len(pool))
        menu_rng = np.random.default_rng(stable_seed("menu", learner, start, seed=seed))
        positions = menu_rng.choice(len(pool), size=n_menu, replace=False)
        bidx = int(handoff["last_index"][learner])
        bfull = torch.as_tensor(handoff["mu"][bidx], dtype=torch.float32, device=device)
        candidates: list[dict[str, Any]] = []
        for pos in positions:
            base = dict(pool[int(pos)])
            codec = base["codec"]
            logits = _point_logits(model, bfull, codec, device).reshape(-1)
            probs = torch.softmax(torch.as_tensor(logits), dim=-1).cpu().numpy()
            outcome_rng = np.random.default_rng(stable_seed("outcome", learner, base["problem_id"], start, seed=seed))
            outcome = int(outcome_rng.choice(len(probs), p=probs))
            base.update({
                "label": outcome,
                "outcome_code": outcome,
                "class_probs": probs.tolist(),
                "generated_code": None,
                "imagined": True,
            })
            candidates.append(base)
        prefix_ids = {str(x["episode_id"]) for x in prefix}
        candidate_ids = {str(x["episode_id"]) for x in candidates}
        target_ids = {str(x["episode_id"]) for x in targets}
        if prefix_ids & candidate_ids or prefix_ids & target_ids or candidate_ids & target_ids:
            raise AssertionError(f"prefix ∩ candidates ∩ targets != ∅ for {learner}")
        own = {int(x["problem_id"]) for x in pool}
        if not all(int(x["problem_id"]) in own for x in candidates):
            raise AssertionError(f"candidate outside learner's own real pool for {learner}")
        planned.append({
            "learner": learner,
            "start": start,
            "prefix_events": prefix,
            "full_events": all_events[:-min(5, len(all_events) - start - 1)],
            "candidates": candidates,
            "targets": targets,
            "b_full_mu": bfull,
            "b_full_log_sigma": torch.as_tensor(handoff["log_sigma"][bidx], dtype=torch.float32, device=device),
            "candidate_pool_size": len(pool),
            "menu_size": len(candidates),
        })
    return planned, fit_ids


def make_w_requests(planned: Sequence[Mapping[str, Any]], statements: Mapping[int, Mapping[str, Any]]) -> list[dict[str, Any]]:
    requests: list[dict[str, Any]] = []
    for ep in planned:
        for candidate in ep["candidates"]:
            pid = int(candidate["problem_id"])
            requests.append({
                "learner": str(ep["learner"]),
                "problem_id": pid,
                "source_split": "train",
                "statement": str(statements[pid]["statement"]),
                "class_probs": list(candidate["class_probs"]),
                "b_full_mu": ep["b_full_mu"].detach().cpu().numpy().tolist(),
            })
    return requests


def generate_w_corpus(
    *,
    planned: Sequence[Mapping[str, Any]],
    statements: Mapping[int, Mapping[str, Any]],
    base_checkpoint: pathlib.Path,
    head_checkpoint: pathlib.Path,
    handoff_sha: str,
    out_path: pathlib.Path,
    device: str,
    batch_size: int,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[tuple[str, int, int], torch.Tensor]]:
    requests = make_w_requests(planned, statements)
    expected = sum(len(ep["candidates"]) for ep in planned) * N_CLASSES
    generation_model = load_cw_b_full_generation_model(
        base_checkpoint, device=device, head_checkpoint=head_checkpoint, allow_download=False
    )
    generator = DreamCorpusGenerator(
        make_cw_batched_generator(generation_model, device=device),
        DreamConfig(seed=W_SEED, classes_per_row=None, batch_size=batch_size),
    )
    manifest = generator.generate(requests, out_path, handoff_sha)
    if int(manifest["n_records"]) != expected:
        raise AssertionError(f"corpus row count {manifest['n_records']} != expected {expected}")
    records = [json.loads(line) for line in out_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if any(not str(row.get("generated_code", "")).strip() for row in records):
        raise AssertionError("empty generated-code row")
    encoded = encode_corpus_records(records, generation_model, device)
    # Attach the selected generated outcome/content to the planned candidate rows.
    by_key = {(str(r["learner"]), int(r["problem_id"]), int(r["conditioned_class"])): r for r in records}
    for ep in planned:
        for candidate in ep["candidates"]:
            key = (str(ep["learner"]), int(candidate["problem_id"]), int(candidate["outcome_code"]))
            rec = by_key.get(key)
            if rec is None:
                raise AssertionError(f"corpus missing sampled outcome {key}")
            candidate["summary"] = encoded[key]
            candidate["generated_code"] = rec["generated_code"]
    return records, manifest, encoded


def split_sim_validation(episodes: Sequence[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Reserve a deterministic 40-learner simulated-validation slice within the fit corpus."""
    ids = sorted({str(ep["learner"]) for ep in episodes})
    val_ids = {ids[int(i)] for i in np.linspace(0, len(ids) - 1, min(40, len(ids)), dtype=int)}
    return [ep for ep in episodes if str(ep["learner"]) not in val_ids], [ep for ep in episodes if str(ep["learner"]) in val_ids]


def _real_exam_args(args: argparse.Namespace) -> argparse.Namespace:
    return _args_for_manifest(args)


def run_w_exam(*, model: Mapping[str, Any], actor: PointerActorV2, args: argparse.Namespace, cfg: Sim2RealConfig, device: str, out: pathlib.Path, train_eps: Sequence[dict[str, Any]], val_eps: Sequence[dict[str, Any]], events_override: Mapping[str, Sequence[Mapping[str, Any]]] | None = None, val_learners: Sequence[str] | None = None) -> dict[str, Any]:
    if events_override is None:
        manifest = _build_manifest(_real_exam_args(args))
        summaries = _episode_summaries(dict(model), manifest, device)
        scenarios = _make_scenarios(manifest, summaries, device)
    else:
        scenarios = scenarios_from_events(events_override, val_learners or sorted(events_override))
    policies = ("sim_actor", "random", "greedy_epig", "max_entropy", "difficulty_matched")
    cells: dict[str, Any] = {}
    exam_rows: list[dict[str, Any]] = []
    for start in STARTS:
        eps = scenarios.get(f"val_start{start}", [])
        if not eps:
            raise RuntimeError(f"no external validation learners at start={start}")
        for budget in cfg.budgets:
            by_policy: dict[str, dict[str, list[float]]] = {p: {} for p in policies}
            values: dict[str, list[float]] = {p: [] for p in policies}
            for i, ep in enumerate(eps):
                for policy in policies:
                    trace_policy = "rl_actor" if policy == "sim_actor" else policy
                    trace = _trace_policy_v2(dict(model), ep, trace_policy, int(budget), device, W_SEED + start * 1000 + budget * 10 + i, actor)
                    values[policy].append(float(trace["dlogit"][-1]))
                exam_rows.append({"source": "real_heldout_replay", "learner": str(ep["learner"]), "start": int(start), "budget": int(budget)})
            for policy in policies:
                by_policy[policy][str(W_SEED)] = values[policy]
            cell = evaluate_real_exam(
                cells=by_policy,
                training_rows=[{"source": "simulator", "generated_from": "b_full", "learner": e["learner"]} for e in train_eps],
                selection_rows=[{"source": "simulated_validation", "split": "sim_val", "generated_from": "b_full", "learner": e["learner"]} for e in val_eps],
                exam_rows=[r for r in exam_rows if r["start"] == start and r["budget"] == budget],
                config=cfg,
            )
            cell.update({"start": int(start), "budget": int(budget), "n_learners": len(eps), "primary": "sim_actor_vs_random"})
            (out / f"real_exam_start{start}_B{budget}.json").write_text(json.dumps(_jsonable(cell), indent=2) + "\n")
            cells[f"start{start}|B{budget}"] = cell
    result = {"status": "complete", "metric": "centered_point_logit_mse", "cells": cells, "n_exam_rows": len(exam_rows), "training_source": "simulator", "selection_source": "simulated_validation", "real_outcome_rows_in_training": 0, "real_rows_in_selection": 0, "starts": list(STARTS), "budgets": list(cfg.budgets), "primary": "sim_actor_vs_random", "blind_heuristics": "context_only"}
    (out / "real_exam.json").write_text(json.dumps(_jsonable(result), indent=2) + "\n")
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--handoff", default="artifacts/cw_sim_refreshed_head_train_prefix_beliefs.npz")
    ap.add_argument("--checkpoint", default="results/cw_sim_head_refresh/final.pt")
    ap.add_argument("--base-checkpoint", default="results/cw_sim_base_run/final.pt")
    ap.add_argument("--raw-dir")
    ap.add_argument("--precomputed-events")
    ap.add_argument("--structural-audit")
    ap.add_argument("--smoke-only", action="store_true")
    ap.add_argument("--out-dir", default="results/sim2real_w")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--reuse-corpus", action="store_true")
    for name in ("code-map", "diff-map", "codealone-meta", "codealone-states", "diff-meta", "diff-states"):
        ap.add_argument("--" + name, required=True)
    args = ap.parse_args()
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)
    out = (REPO / args.out_dir).resolve(); out.mkdir(parents=True, exist_ok=True)
    if not args.raw_dir and not args.precomputed_events:
        ap.error("one of --raw-dir or --precomputed-events is required")
    checkpoint = (REPO / args.checkpoint).resolve(); handoff_path = (REPO / args.handoff).resolve()
    if sha256_file(checkpoint) != REGISTERED_CHECKPOINT_SHA256:
        raise ValueError("refreshed-head checkpoint SHA mismatch")
    handoff = load_train_handoff(handoff_path, HANDOFF_SHA256)
    # Run the generation-free smoke on the same raw archive before model loading.
    split = read_split(REPO / "splits/cw_learner_split.csv")
    if args.precomputed_events:
        audit_path = pathlib.Path(args.structural_audit) if args.structural_audit else REPO / "results/sim2real_w_smoke/structural_audit.json"
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if audit.get("status") != "pass":
            raise RuntimeError("structural smoke failed; no generation permitted")
        events_override = load_precomputed_events((REPO / args.precomputed_events).resolve(), device)
    else:
        audit = structural_audit(real_event_rows(pathlib.Path(args.raw_dir)), split, smoke_learners=20)
        if audit["status"] != "pass":
            raise RuntimeError("structural smoke failed; no generation permitted")
        events_override = None
    (out / "structural_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    model = build_cwsim_model_dict(checkpoint, device)
    for value in model.values():
        if hasattr(value, "eval"): value.eval()
    if events_override is None:
        manifest = _build_manifest(_args_for_manifest(args)); summaries = _episode_summaries(dict(model), manifest, device)
    else:
        manifest, summaries = {}, {}
    statements, _ = item_panel()
    planned, fit_ids = plan_w_episodes(manifest=manifest, summaries=summaries, model=model, handoff=handoff, device=device, events_override=events_override)
    planned_for_generation = planned[:20] if args.smoke_only else planned
    corpus_path = out / "generations.jsonl"; manifest_path = out / "generations.jsonl.manifest.json"
    if args.reuse_corpus:
        from im_s3_dream import load_dream_corpus
        records = load_dream_corpus(corpus_path, ("train",)); corpus_manifest = json.loads(manifest_path.read_text())
        if int(corpus_manifest.get("n_records", -1)) != sum(len(ep["candidates"]) for ep in planned_for_generation) * N_CLASSES:
            raise ValueError("reused corpus row count does not match amended menu draw")
        generation_model = load_cw_b_full_generation_model(REPO / args.base_checkpoint, device=device, head_checkpoint=checkpoint, allow_download=False)
        encoded = encode_corpus_records(records, generation_model, device)
        by_key = {(str(r["learner"]), int(r["problem_id"]), int(r["conditioned_class"])): r for r in records}
        for ep in planned:
            for c in ep["candidates"]:
                key = (str(ep["learner"]), int(c["problem_id"]), int(c["outcome_code"])); c["summary"] = encoded[key]; c["generated_code"] = by_key[key]["generated_code"]
    else:
        records, corpus_manifest, encoded = generate_w_corpus(planned=planned_for_generation, statements=statements, base_checkpoint=REPO / args.base_checkpoint, head_checkpoint=checkpoint, handoff_sha=handoff["checkpoint_sha256"], out_path=corpus_path, device=device, batch_size=args.batch_size)
        manifest_path.write_text(json.dumps(corpus_manifest, indent=2) + "\n")
    if args.smoke_only:
        (out / "smoke_summary.json").write_text(json.dumps(_jsonable({"status": "pass", "n_learners": len(planned_for_generation), "n_records": len(records), "menu_cap": MENU_SIZE}), indent=2) + "\n")
        print(json.dumps(_jsonable({"status": "smoke_pass", "n_learners": len(planned_for_generation), "n_records": len(records)}), indent=2))
        return 0
    train_eps, val_eps = split_sim_validation(planned)
    cfg = Sim2RealConfig(seeds=(W_SEED,), budgets=(3, 5, 10), max_epochs=40, validation_patience=5, min_validation_improvement=0.001, n_boot=2000)
    actor, train_meta = train_sim_actor(model=model, train_eps=train_eps, val_eps=val_eps, seed=W_SEED, cfg=cfg, device=device, out_dir=out)
    (out / "simulator_manifest.json").write_text(json.dumps(_jsonable({"checkpoint_sha256": REGISTERED_CHECKPOINT_SHA256, "handoff_sha256": handoff["sha256"], "fit_learners": len(fit_ids), "n_train_episodes": len(train_eps), "n_simulated_validation_episodes": len(val_eps), "n_corpus_records": len(records), "corpus_manifest": corpus_manifest, "menu_cap": MENU_SIZE, "budgets": list(cfg.budgets), "training_source": "simulator", "selection_source": "simulated_validation", "real_outcome_rows_in_training": 0, "real_rows_in_selection": 0, "raw_text_uploaded": False}), indent=2) + "\n")
    val_learners = [str(k) for k, v in split.items() if v == "val"]
    exam = run_w_exam(model=model, actor=actor, args=args, cfg=cfg, device=device, out=out, train_eps=train_eps, val_eps=val_eps, events_override=events_override, val_learners=val_learners)
    print(json.dumps(_jsonable({"status": "complete", "train": train_meta, "exam": exam}), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
