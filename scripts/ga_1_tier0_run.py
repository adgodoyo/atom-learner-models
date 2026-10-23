#!/usr/bin/env python3
"""Zero-shot LLM selector (§8.4): the local Gemma backbone as one more policy in the replay of
`rl_2.py`, compared with the trained actor, blind comparators and a privileged upper bound.

Trajectories, common-random-number seeds, metrics and bootstrap come unchanged from
``scripts/rl_2.py`` and ``src/atom/simulator/cw_substrate.py``; the model loader, licensed
history join, local generation store and numeric progress come from ``scripts/ga_1_common.py``.
This file adds the selector loop (with and without history), the comparator panel (trained
actor, random, max-entropy, difficulty-matched, privileged oracle) and the aggregation.
Menu identifiers come from ``ga_1_tier0_prompt`` (a single-token A-Z/a-z alphabet, asserted per
state); choices are read with the constrained read-out of ``ga_1_tier0_readout``. The privileged
oracle is a labelled upper bound and is never described as deployable.

Inputs: one completed `rl_2.py` output directory per belief substrate (``--rl2-dir
SUBSTRATE=PATH``, with summary.json and policy_seed*.pt), the CW problem statements and raw
CW tables. Outputs in ``--out-dir`` (default results/ga_1_tier0): summary.json, numeric rows,
progress.json, and generations.jsonl (local only; optionally numeric files to S3 via --bucket).
Run: `python scripts/ga_1_tier0_run.py --rl2-dir ratchet-original=... [--pilot]`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
import time
from collections import Counter
from typing import Any, Sequence

import numpy as np
import torch

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from cw_planner_suite import STARTS  # noqa: E402
from ga_1_common import (  # noqa: E402
    EXPECTED_HF_ID,
    LocalGenerationStore,
    NumericProgress,
    load_local_gemma,
    load_problem_statements,
    load_raw_episode_text,
    release_backbone,
    stable_hash,
    stop_instance_if_requested,
    validate_no_external_api_surface,
)
from ga_1_tier0_prompt import (  # noqa: E402
    CHOICE_TAIL,
    IDENTIFIERS,
    assert_prompt_wellformed,
    assert_single_token,
    build_prompt,
)
from ga_1_tier0_readout import (  # noqa: E402
    identifier_token_ids,
    read_out_choices,
    summarise_diagnostics,
)
from rl_2 import (  # noqa: E402
    SEEDS,
    PointerActorV2,
    _bootstrap_seed_learner,
    _build_manifest,
    _dlogit,
    _episode_summaries,
    _frozen_filter_state,
    _kl,
    _load_bundle,
    _make_scenarios,
    _point_logits,
    _trace_policy_v2,
)

from atom.simulator.cw_substrate import load_substrate as _lane_p_load_substrate  # noqa: E402,F401

BUDGETS = (3, 8, 16)
SUBSTRATES = ("ratchet-original", "ratchet-cwsim/cold-head", "ratchet-cwsim/refreshed-head")
SELECTOR_POLICIES = ("llm_selector", "llm_selector_no_history")
# Blind deployable comparators, then the labelled privileged upper bound.
BLIND_POLICIES = ("random", "max_entropy", "difficulty_matched")
COMPARATOR_POLICIES = ("rl_actor",) + BLIND_POLICIES + ("privileged_oracle_upper_bound",)
# Only the privileged oracle is seed-invariant.
#
# `_trace_policy_v2` draws from its numpy CRN rng only in the `random` branch, but the
# harness also builds `eps = torch.randn(K_EVAL, latent_dim,
# generator=Generator().manual_seed(seed))`, the K-sample epistemic draws, and both
# `max_entropy` and `difficulty_matched` rank candidates on scores computed from `eps`, so
# they are seed-dependent by construction. A source-level check of which rng is consumed
# cannot establish seed invariance; the runtime re-verification below compares the traces
# themselves.
#
# The oracle's own branch touches neither `eps` nor `rng`: it scores every remaining
# candidate through `_frozen_filter_state`/`_point_logits`/`_dlogit` alone.  It is also
# the policy worth caching -- ~630 filter forwards per learner at B=16, which dominates
# replay cost -- so restricting the cache to it keeps nearly all of the saving.
SEED_INVARIANT_POLICIES = ("privileged_oracle_upper_bound",)
ALL_POLICIES = SELECTOR_POLICIES + COMPARATOR_POLICIES
METRICS = ("logloss", "dlogit", "pred_kl")
# How many cached traces to re-verify by recomputation on a later seed.  This is the
# load-bearing check on the caching premise.
VERIFY_BUDGET = 60
N_BOOT = 2000
PARSE_FLOOR = 0.99


# --------------------------------------------------------------------------- helpers


def _sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, pathlib.Path):
        return str(value)
    return value


def _mapping(values: Sequence[str], flag: str) -> dict[str, pathlib.Path]:
    out: dict[str, pathlib.Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"{flag} requires SUBSTRATE=PATH, got {value!r}")
        name, raw = value.split("=", 1)
        if name in out:
            raise ValueError(f"duplicate {flag} substrate {name!r}")
        path = pathlib.Path(raw)
        out[name] = path if path.is_absolute() else REPO / path
    unknown = set(out) - set(SUBSTRATES)
    if unknown:
        raise ValueError(f"{flag} names unknown substrates {sorted(unknown)}")
    return out


def _geometry_digest(scenarios: dict[str, list[dict[str, Any]]], starts: Sequence[int]) -> str:
    """Value-digest of the replay geometry (checks identity of the rows, not just counts)."""
    rows = []
    for start in starts:
        for ep in scenarios[f"val_start{start}"]:
            rows.append({
                "start": int(start),
                "learner": str(ep["learner"]),
                "prefix": [str(x["episode_id"]) for x in ep["prefix_events"]],
                "candidates": [[str(x["episode_id"]), int(x["problem_id"])]
                               for x in ep["candidates"]],
                "targets": [[str(x["episode_id"]), int(x["problem_id"]), int(t)]
                            for x, t in ((x, x["label"]) for x in ep["targets"])],
            })
    return stable_hash(json.dumps(rows, sort_keys=True, separators=(",", ":")), 64)


def _cohort(scenarios: dict[str, list[dict[str, Any]]], start: int,
            limit: int | None) -> list[dict[str, Any]]:
    episodes = scenarios[f"val_start{start}"]
    return episodes if limit is None else episodes[:limit]


def _load_rl2_reference(rl2_dirs: dict[str, pathlib.Path], starts: Sequence[int],
                        budgets: Sequence[int]) -> tuple[dict[str, Any], dict[str, Any]]:
    summaries: dict[str, Any] = {}
    manifest: dict[str, Any] = {}
    for substrate, directory in rl2_dirs.items():
        summary_path = directory / "summary.json"
        if not summary_path.exists():
            raise FileNotFoundError(f"rl_2 summary missing: {summary_path}")
        summary = json.loads(summary_path.read_text())
        if summary.get("status") != "complete" or summary.get("substrate") != substrate:
            raise ValueError(f"not a complete {substrate} rl_2 summary: {summary_path}")
        for start in starts:
            for budget in budgets:
                for policy in COMPARATOR_POLICIES:
                    key = f"start{start}|{policy}|B{budget}"
                    if key not in summary.get("cells", {}):
                        raise KeyError(f"rl_2 comparator cell missing: {substrate}/{key}")
        actor_hashes = {}
        for seed in SEEDS:
            actor_path = directory / f"policy_seed{seed}.pt"
            if not actor_path.exists():
                raise FileNotFoundError(f"rl_2 actor missing: {actor_path}")
            actor_hashes[str(seed)] = _sha256(actor_path)
        summaries[substrate] = summary
        manifest[substrate] = {"summary_path": str(summary_path),
                               "summary_sha256": _sha256(summary_path),
                               "actor_sha256": actor_hashes}
    return summaries, manifest


# ------------------------------------------------------------------------ generation


def _expected_rows(scenarios, starts, budgets, limit) -> int:
    per_mode = 0
    for start in starts:
        for ep in _cohort(scenarios, start, limit):
            per_mode += sum(min(b, len(ep["candidates"])) for b in budgets)
    return 2 * per_mode


def _generate_choices(args, scenarios, episode_text, statements, starts, budgets, limit,
                      progress: NumericProgress):
    """Greedy menu choices, generated once and value-joined across substrates and seeds.

    The choice depends only on (mode, learner, start, budget, step) -- the prompt sees no
    substrate and no actor -- so re-generating per substrate would burn GPU hours to
    reproduce identical text, and would not be an independent sample either way.
    """
    bundle = load_local_gemma(args.device)
    tokenizer = bundle["tokenizer"]
    ident_map = assert_single_token(tokenizer)
    ident_ids = identifier_token_ids(tokenizer, IDENTIFIERS)
    store = LocalGenerationStore(pathlib.Path(args.out_dir) / "generations.jsonl")
    saved = store.load(("mode", "learner", "start", "budget", "step"))
    choices: dict[tuple[str, int, int, str], list[str]] = {}
    parse_counts: Counter[str] = Counter()
    prompt_tokens: list[int] = []
    readout_diagnostics: list[dict[str, float]] = []
    total = _expected_rows(scenarios, starts, budgets, limit)
    done = 0
    progress.heartbeat("tier0_generate", done, total)

    for mode in ("history", "no_history"):
        for start in starts:
            episodes = _cohort(scenarios, start, limit)
            for budget in budgets:
                states = [{"ep": ep, "known": list(ep["prefix_events"]),
                           "remaining": list(range(len(ep["candidates"]))), "picks": []}
                          for ep in episodes]
                max_steps = max((min(budget, len(s["remaining"])) for s in states), default=0)
                for step in range(max_steps):
                    active = [s for s in states if s["remaining"]]
                    pending: list[tuple[dict, str, str, int]] = []
                    resolved: list[tuple[dict, str]] = []
                    for state in active:
                        ep = state["ep"]
                        candidates = [ep["candidates"][i] for i in state["remaining"]]
                        prompt = build_prompt(
                            state["known"], candidates, budget, step, episode_text, statements,
                            tokenizer, max_prompt_tokens=args.max_prompt_tokens,
                            with_history=(mode == "history"))
                        # Per-state check, on the RENDERED prompt: every menu item is
                        # addressable and nothing will be left-truncated away.
                        assert_prompt_wellformed(prompt, candidates, tokenizer,
                                                 args.max_prompt_tokens)
                        prompt_hash = stable_hash(prompt, 32)
                        key = (mode, str(ep["learner"]), str(start), str(budget), str(step))
                        if key in saved:
                            row = saved[key]
                            if str(row.get("prompt_hash")) != prompt_hash:
                                raise RuntimeError(f"tier-0 resume prompt drift for {key}")
                            saved_slot = IDENTIFIERS.index(str(row["generated_text"]))
                            resolved.append((state, saved_slot))
                        else:
                            n_tok = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
                            pending.append((state, prompt, prompt_hash, n_tok))
                    if pending:
                        base = done
                        slots, diags = read_out_choices(
                            bundle, [x[1] for x in pending],
                            [len(x[0]["remaining"]) for x in pending],
                            ident_ids,
                            batch_size=args.batch_size,
                            max_prompt_tokens=args.max_prompt_tokens,
                            token_budget=args.token_budget,
                            on_batch=lambda d, t, base=base: progress.maybe_heartbeat(
                                "tier0_generate", base + d, total, enforce_overrun=True))
                        for (state, _, prompt_hash, n_tok), slot, diag in zip(pending, slots, diags):
                            prompt_tokens.append(n_tok)
                            readout_diagnostics.append(diag)
                            store.append({"mode": mode, "learner": str(state["ep"]["learner"]),
                                          "start": int(start), "budget": int(budget),
                                          "step": int(step), "prompt_hash": prompt_hash,
                                          "prompt_tokens": n_tok,
                                          "menu_size": len(state["remaining"]),
                                          "model": EXPECTED_HF_ID,
                                          "generated_text": IDENTIFIERS[slot], **diag})
                            resolved.append((state, slot))
                    if len(resolved) != len(active):
                        raise RuntimeError("tier-0 selector lost an active replay row")
                    for state, slot in resolved:
                        if not 0 <= slot < len(state["remaining"]):
                            raise RuntimeError(f"read-out returned off-menu slot {slot}")
                        parse_counts["constrained_readout"] += 1
                        chosen = state["ep"]["candidates"][state["remaining"].pop(slot)]
                        state["known"].append(chosen)
                        state["picks"].append(str(chosen["episode_id"]))
                    done += len(active)
                    progress.maybe_heartbeat("tier0_generate", done, total, enforce_overrun=True,
                                             mode=mode, start=start, budget=budget, step=step)
                for state in states:
                    choices[(mode, int(start), int(budget), str(state["ep"]["learner"]))] = \
                        list(state["picks"])

    if done != total:
        raise RuntimeError(f"tier-0 generation accounting mismatch: {done=} {total=}")
    meta = {
        "model": "gemma4-e4b", "hf_id": EXPECTED_HF_ID,
        "checkpoint_revision": bundle["checkpoint_revision"], "decode": "greedy",
        "readout": "constrained argmax over live-menu identifier-token logits at CHOICE=",
        "readout_tail": CHOICE_TAIL, "alphabet": "".join(IDENTIFIERS),
        "identifier_token_ids": {k: v[0] for k, v in ident_map.items()},
        "max_prompt_tokens": args.max_prompt_tokens, "batch_size": args.batch_size,
        "token_budget": args.token_budget,
        "n_rows": total, "n_generated_this_run": len(prompt_tokens),
        "parse_counts": dict(parse_counts),
        "parse_rate": 1.0,
        "readout_diagnostics": summarise_diagnostics(readout_diagnostics),
        "fallback_rule": "none: the read-out cannot produce an off-menu answer",
        "prompt_tokens": {
            "min": int(min(prompt_tokens)) if prompt_tokens else None,
            "median": float(np.median(prompt_tokens)) if prompt_tokens else None,
            "max": int(max(prompt_tokens)) if prompt_tokens else None,
        },
    }
    release_backbone(bundle)
    return choices, meta


# --------------------------------------------------------------------------- scoring


def _score_fixed_picks(model, ep, pick_episode_ids: Sequence[str], device: str):
    """Score an externally chosen sequence with the frozen replay primitives of rl_2.py."""
    known = list(ep["prefix_events"])
    targets = list(ep["targets"])
    query = targets[0]["codec"]
    target_codecs = torch.stack([t["codec"] for t in targets])
    full_mu, _ = _frozen_filter_state(model, ep["full_events"], query, device)
    full_logits = _point_logits(model, full_mu, target_codecs, device)
    full_probs = torch.softmax(torch.from_numpy(full_logits), dim=-1).numpy()
    labels = np.asarray([t["label"] for t in targets], dtype=int)
    traces: dict[str, list[Any]] = {"logloss": [], "dlogit": [], "pred_kl": [],
                                    "picks": [], "pick_problems": []}
    remaining = {str(c["episode_id"]): c for c in ep["candidates"]}
    for step in range(len(pick_episode_ids) + 1):
        mu, _ = _frozen_filter_state(model, sorted(known, key=lambda x: x["order"]), query, device)
        logits = _point_logits(model, mu, target_codecs, device)
        probs = torch.softmax(torch.from_numpy(logits), dim=-1).numpy()
        traces["logloss"].append(
            float(-np.log(np.clip(probs[np.arange(len(labels)), labels], 1e-8, 1)).mean()))
        traces["dlogit"].append(_dlogit(logits, full_logits))
        traces["pred_kl"].append(_kl(probs, full_probs))
        if step == len(pick_episode_ids):
            break
        episode_id = str(pick_episode_ids[step])
        if episode_id not in remaining:
            raise KeyError(f"selector pick absent or already consumed: {episode_id}")
        chosen = remaining.pop(episode_id)
        known.append(chosen)
        traces["picks"].append(episode_id)
        traces["pick_problems"].append(int(chosen["problem_id"]))
    return traces


def _cell_arrays(rows: Sequence[dict[str, Any]]) -> dict[str, np.ndarray]:
    return {m: np.asarray([r[m][-1] for r in rows], dtype=np.float64) for m in METRICS}


def _partial_path(out, substrate, seed, start, budget) -> pathlib.Path:
    return out / f"partial_{substrate.replace('/', '_')}_seed{seed}_start{start}_B{budget}.npz"


def _save_partial(path, arrays) -> None:
    np.savez_compressed(path, **{f"{p}__{m}": v for p, ms in arrays.items()
                                 for m, v in ms.items()})


def _load_partial(path) -> dict[str, dict[str, np.ndarray]]:
    saved = np.load(path, allow_pickle=False)
    return {p: {m: np.asarray(saved[f"{p}__{m}"], dtype=np.float64) for m in METRICS}
            for p in ALL_POLICIES}


def _score_substrate(args, substrate, rl2_dir, rl2_summary, manifest, expected_geometry,
                     choices, starts, budgets, limit, progress, work_done, work_total):
    model, _, bundle_meta = _load_bundle(substrate, args.device)
    summaries = _episode_summaries(model, manifest, args.device)
    scenarios = _make_scenarios(manifest, summaries, args.device, 0, args.val_learners,
                                release_by_episode=None)
    digest = _geometry_digest(scenarios, starts)
    if digest != expected_geometry:
        raise RuntimeError(f"cross-substrate replay geometry drift for {substrate}")
    codec_dim = int(scenarios["val_start0"][0]["candidates"][0]["codec"].numel())
    actors = {}
    for seed in SEEDS:
        actor = PointerActorV2(codec_dim).to(args.device)
        actor.load_state_dict(torch.load(rl2_dir / f"policy_seed{seed}.pt",
                                         map_location=args.device, weights_only=False)["actor"])
        actor.eval()
        actors[seed] = actor

    seed_cells: dict[str, dict[str, dict[str, list[np.ndarray]]]] = {}
    numeric_rows: list[dict[str, Any]] = []
    out = REPO / args.out_dir
    # Per-substrate: the model is fixed inside this loop, so the cache is valid here and
    # is dropped when the substrate changes.
    trace_cache: dict[tuple[Any, ...], dict[str, Any]] = {}
    verify_cells: set[tuple[Any, ...]] = set()
    for seed in SEEDS:
        for start in starts:
            episodes = _cohort(scenarios, start, limit)
            for budget in budgets:
                partial = _partial_path(out, substrate, seed, start, budget)
                if args.resume and partial.exists():
                    arrays = _load_partial(partial)
                else:
                    rows = {p: [] for p in ALL_POLICIES}
                    for i, ep in enumerate(episodes):
                        learner = str(ep["learner"])
                        rows["llm_selector"].append(_score_fixed_picks(
                            model, ep, choices[("history", start, budget, learner)], args.device))
                        rows["llm_selector_no_history"].append(_score_fixed_picks(
                            model, ep, choices[("no_history", start, budget, learner)], args.device))
                        # The CRN seed formula of rl_2.py, unchanged.
                        replay_seed = seed + 100000 + start * 1000 + i
                        for policy in COMPARATOR_POLICIES:
                            ck = (policy, start, budget, i)
                            if policy in SEED_INVARIANT_POLICIES and ck in trace_cache:
                                trace = trace_cache[ck]
                                if ck in verify_cells:
                                    fresh = _trace_policy_v2(
                                        model, ep, policy, budget, args.device,
                                        replay_seed, None)
                                    for m in METRICS:
                                        if fresh[m] != trace[m]:
                                            raise RuntimeError(
                                                f"seed-invariance assertion FAILED for "
                                                f"{policy} at {ck}: cached trace differs "
                                                "across seeds")
                                    verify_cells.discard(ck)
                            else:
                                trace = _trace_policy_v2(
                                    model, ep, policy, budget, args.device, replay_seed,
                                    actors[seed] if policy == "rl_actor" else None)
                                if policy in SEED_INVARIANT_POLICIES:
                                    trace_cache[ck] = trace
                                    if len(verify_cells) < VERIFY_BUDGET:
                                        verify_cells.add(ck)
                            rows[policy].append(trace)
                        progress.maybe_heartbeat("tier0_replay", work_done + i + 1, work_total,
                                                 substrate=substrate, seed=seed, start=start,
                                                 budget=budget)
                    arrays = {p: _cell_arrays(r) for p, r in rows.items()}
                    _save_partial(partial, arrays)
                    progress.upload(partial)
                cell = seed_cells.setdefault(f"start{start}|B{budget}", {})
                for policy, metrics in arrays.items():
                    target = cell.setdefault(policy, {m: [] for m in METRICS})
                    for m, v in metrics.items():
                        target[m].append(np.asarray(v, dtype=np.float64))
                    for i, ep in enumerate(episodes):
                        numeric_rows.append({
                            "substrate": substrate, "seed": seed, "start": start,
                            "budget": budget, "policy": policy,
                            "learner_hash": int(stable_hash(str(ep["learner"]), 16), 16),
                            **{m: float(metrics[m][i]) for m in METRICS}})
                work_done += len(episodes)
                progress.heartbeat("tier0_replay", work_done, work_total, substrate=substrate,
                                   seed=seed, start=start, budget=budget)

    cells: dict[str, Any] = {}
    comparisons: dict[str, Any] = {}
    max_reference_error = 0.0
    for short_key, policies in seed_cells.items():
        start_text, budget_text = short_key.split("|")
        for policy, metrics in policies.items():
            key = f"{start_text}|{policy}|{budget_text}"
            cells[key] = {m: _bootstrap_seed_learner(v, args.seed + 1000 * mi + len(key),
                                                     n_boot=N_BOOT)
                          for mi, (m, v) in enumerate(metrics.items())}
            if policy in COMPARATOR_POLICIES:
                expected = rl2_summary["cells"][key]
                for m in METRICS:
                    max_reference_error = max(max_reference_error,
                                              abs(float(cells[key][m][0]) - float(expected[m][0])))
        pairs = [(s, c) for s in SELECTOR_POLICIES for c in COMPARATOR_POLICIES]
        pairs.append(("llm_selector", "llm_selector_no_history"))
        for a, b in pairs:
            key = f"{start_text}|{a}-minus-{b}|{budget_text}"
            comparisons[key] = {}
            for mi, m in enumerate(METRICS):
                paired = [x - y for x, y in zip(policies[a][m], policies[b][m])]
                comparisons[key][m] = _bootstrap_seed_learner(
                    paired, args.seed + 7000 + 1000 * mi + len(key), n_boot=N_BOOT)

    if not args.pilot and max_reference_error > args.rl2_reference_tolerance:
        raise RuntimeError(f"replayed comparators drifted from rl_2 by {max_reference_error:.6g} "
                           f"> {args.rl2_reference_tolerance:.6g}")
    del model, summaries, scenarios, actors
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return ({"cells": cells, "comparisons": comparisons, "bundle": bundle_meta,
             "rl2_reference_max_abs_point_error": max_reference_error,
             "rl2_reference_tolerance": args.rl2_reference_tolerance,
             "seed_invariant_cache": {"policies": list(SEED_INVARIANT_POLICIES),
                                      "cached_cells": len(trace_cache),
                                      "reverified_cells": VERIFY_BUDGET - len(verify_cells)}},
            numeric_rows, work_done)


def _write_numeric_rows(path, rows) -> None:
    substrates = {n: i for i, n in enumerate(sorted({r["substrate"] for r in rows}))}
    policies = {n: i for i, n in enumerate(ALL_POLICIES)}
    np.savez_compressed(
        path,
        substrate=np.asarray([substrates[r["substrate"]] for r in rows], dtype=np.int8),
        substrate_names=np.asarray(sorted(substrates), dtype=object).astype("U40"),
        policy_names=np.asarray(ALL_POLICIES, dtype=object).astype("U40"),
        seed=np.asarray([r["seed"] for r in rows], dtype=np.int64),
        start=np.asarray([r["start"] for r in rows], dtype=np.int16),
        budget=np.asarray([r["budget"] for r in rows], dtype=np.int16),
        policy=np.asarray([policies[r["policy"]] for r in rows], dtype=np.int8),
        learner_hash=np.asarray([r["learner_hash"] for r in rows], dtype=np.uint64),
        **{m: np.asarray([r[m] for r in rows], dtype=np.float64) for m in METRICS})


# ------------------------------------------------------------------------------ main


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rl2-dir", action="append", default=[], help="SUBSTRATE=PATH")
    ap.add_argument("--out-dir", default="results/ga_1_tier0")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--val-learners", type=int, default=110)
    ap.add_argument("--max-prompt-tokens", type=int, default=5120)
    ap.add_argument("--batch-size", type=int, default=16,
                    help="maximum ROWS per read-out batch; the binding limit is "
                         "--token-budget (rows x longest-prompt-in-batch)")
    ap.add_argument("--token-budget", type=int, default=18000,
                    help="padded-token budget per batch: what actually bounds activation "
                         "memory. Batch 8 x 5120 tokens = 40,960 exceeds a 22 GiB A10G.")
    ap.add_argument("--seed", type=int, default=SEEDS[0])
    ap.add_argument("--rl2-reference-tolerance", type=float, default=5e-5)
    ap.add_argument("--bucket", default="")
    ap.add_argument("--out-prefix", default="")
    ap.add_argument("--fuse-minutes", type=int, default=600)
    ap.add_argument("--overrun-minutes", type=int, default=0)
    ap.add_argument("--rearm-fuse", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--stop-on-complete", action="store_true")
    ap.add_argument("--raw-dir", required=True)
    ap.add_argument("--derived-val-predictions", default="")
    ap.add_argument("--pilot", action="store_true",
                    help="rate-measurement run: subset of learners, rl_2 reference check "
                         "reported but not enforced, results never quoted")
    ap.add_argument("--pilot-learners", type=int, default=6)
    ap.add_argument("--starts", default="")
    ap.add_argument("--budgets", default="")
    for name in ("code_map", "diff_map", "codealone_meta", "codealone_states",
                 "diff_meta", "diff_states"):
        ap.add_argument("--" + name.replace("_", "-"), required=True)
    args = ap.parse_args()

    if args.val_learners != 110:
        raise ValueError("tier 0 is fixed to the 110-learner external validation cohort")
    starts = tuple(int(x) for x in args.starts.split(",")) if args.starts else tuple(STARTS)
    budgets = tuple(int(x) for x in args.budgets.split(",")) if args.budgets else BUDGETS
    if not args.pilot and (starts != tuple(STARTS) or budgets != BUDGETS):
        raise ValueError("the registered full grid is starts (0,3,5,10) x budgets (3,8,16)")
    limit = args.pilot_learners if args.pilot else None

    out = REPO / args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    progress = NumericProgress(out, args.bucket, args.out_prefix, fuse_minutes=args.fuse_minutes,
                               overrun_minutes=args.overrun_minutes, rearm_fuse=args.rearm_fuse)
    validate_no_external_api_surface([pathlib.Path(__file__), REPO / "scripts/ga_1_common.py",
                                      REPO / "scripts/ga_1_tier0_prompt.py"])
    rl2_dirs = _mapping(args.rl2_dir, "--rl2-dir")
    if not args.pilot and set(rl2_dirs) != set(SUBSTRATES):
        raise ValueError(f"the full run needs all three substrates, got {sorted(rl2_dirs)}")
    rl2_summaries, rl2_manifest = _load_rl2_reference(rl2_dirs, starts, budgets)
    progress.heartbeat("tier0_start", 0, 1, substrates=sorted(rl2_dirs), pilot=args.pilot)
    t0 = time.time()

    manifest = _build_manifest(args)
    first = sorted(rl2_dirs)[0] if args.pilot else "ratchet-original"
    model, _, _ = _load_bundle(first, args.device)
    episode_summaries = _episode_summaries(model, manifest, args.device)
    scenarios = _make_scenarios(manifest, episode_summaries, args.device, 0, args.val_learners,
                                release_by_episode=None)
    geometry = _geometry_digest(scenarios, starts)
    per_substrate = sum(len(_cohort(scenarios, s, limit)) for s in starts) * len(budgets) * len(SEEDS)
    work_total = per_substrate * len(rl2_dirs)
    episode_text = load_raw_episode_text(args.raw_dir)
    statements = load_problem_statements()
    del model, episode_summaries
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    choices, generation_meta = _generate_choices(args, scenarios, episode_text, statements,
                                                 starts, budgets, limit, progress)
    del scenarios, episode_text
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if generation_meta["parse_rate"] < PARSE_FLOOR:
        raise RuntimeError(f"tier-0 parse-rate floor failed: "
                           f"{generation_meta['parse_rate']:.6f} < {PARSE_FLOOR}")

    substrate_results: dict[str, Any] = {}
    numeric_rows: list[dict[str, Any]] = []
    work_done = 0
    for substrate in [s for s in SUBSTRATES if s in rl2_dirs]:
        result, rows, work_done = _score_substrate(
            args, substrate, rl2_dirs[substrate], rl2_summaries[substrate], manifest, geometry,
            choices, starts, budgets, limit, progress, work_done, work_total)
        substrate_results[substrate] = result
        numeric_rows.extend(rows)

    numeric_path = out / "selector_rows.npz"
    _write_numeric_rows(numeric_path, numeric_rows)
    summary = {
        "status": "pilot_complete" if args.pilot else "complete",
        "lane": "T0",
        "claim": "direct replay value of a zero-shot LLM selector; the privileged oracle is "
                 "a labelled upper bound, never a deployable baseline",
        "model": generation_meta,
        "starts": list(starts), "budgets": list(budgets), "seeds": list(SEEDS),
        "val_learners": args.val_learners,
        "pilot_learners_per_cell": limit,
        "policies": list(ALL_POLICIES),
        "blind_comparators": list(BLIND_POLICIES),
        "primary_metric": "dlogit (centered point-logit distance to b_full; lower is better)",
        "guard_metrics": ["pred_kl", "logloss"],
        "geometry_sha256": geometry,
        "substrates": substrate_results,
        "source_boundary": {
            "rl_2": "imported as-is from scripts/rl_2.py",
            "cw_substrate": "imported as-is from src/atom/simulator/cw_substrate.py",
            "ga_1_common": "imported as-is from scripts/ga_1_common.py",
            "comparator_trajectory": "scripts.rl_2._trace_policy_v2",
            "ci": "scripts.rl_2._bootstrap_seed_learner, 2000 resamples, seed-then-learner",
            "external_api_calls": 0,
            "generated_text_uploaded_to_s3": False,
            "oracle_role": "privileged labelled upper bound only",
        },
        "wall_s": time.time() - t0,
    }
    (out / "summary.json").write_text(json.dumps(_jsonable(summary), indent=2, sort_keys=True) + "\n")
    (out / "manifest.json").write_text(json.dumps({
        "status": summary["status"], "rl2": rl2_manifest, "geometry_sha256": geometry,
        "numeric_sha256": _sha256(numeric_path),
        "generations_jsonl": "kept on the machine that ran the model; copy locally before shutdown; never uploaded",
    }, indent=2, sort_keys=True) + "\n")
    for name in ("selector_rows.npz", "summary.json", "manifest.json"):
        progress.upload(out / name)
    progress.heartbeat("tier0_complete", 1, 1, parse_rate=generation_meta["parse_rate"],
                       wall_s=summary["wall_s"])
    print(json.dumps({"status": summary["status"],
                      "parse_rate": generation_meta["parse_rate"],
                      "readout": generation_meta["readout_diagnostics"],
                      "prompt_tokens": generation_meta["prompt_tokens"],
                      "wall_s": summary["wall_s"]}, indent=2))
    stop_instance_if_requested(args.stop_on_complete)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
