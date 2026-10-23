#!/usr/bin/env python3
"""Evaluates the LLM selector fine-tuned on the simulator's pricing: the personalisation 2x2 and
the blind comparator panel (Figure 12).

Four LLM arms, all under the CRN identifier shuffle:

    {zero-shot, trained} x {history, no-history}

The zero-shot arms under the shuffle are the baseline row; the unshuffled zero-shot numbers of
`ga_1_tier0_run.py` are context only.

Primary pre-registered read: the trained model's history-vs-no-history gap exceeds the zero-shot
model's -- a difference-in-differences, learner-clustered, CI-clear:

    DiD = (trained_hist - trained_nohist) - (zeroshot_hist - zeroshot_nohist)

with the primary metric `dlogit` (lower is better), so a negative DiD means training widened the
history gap in the good direction. The zero-shot model reads the learner but does not act on it;
this asks whether training on the simulator's feedback turned reading into acting.

Blind comparators (random / max-entropy / difficulty-matched / privileged oracle) are re-run at
starts {3,10} across the three CRN seeds of `rl_2.py`, so their cell means are directly comparable
with that script's committed panel, and every recomputed blind cell is checked against it at
tolerance 5e-5 (see the comment at the check: breaches are recorded, not fatal). Both starts are in
`rl_2.STARTS`, so no cell has to be synthesised. The trained `rl_2.py` actor is quoted from its
committed cells at both starts as an unpaired reference; it is not re-run.

Pairing convention: the LLM arms have no seed dimension (one training run), so an LLM-vs-blind
pair uses the blind comparator's per-learner mean over the three CRN seeds. That reduces
comparator noise and never touches the LLM side.

Inputs: the adapter checkpoint from `ga_1_tier3_train.py`, the `rl_2.py` output directory for
the CodeWorkout generative simulator with its refreshed outcome head (``--rl2-dir``), the CW
statements and raw tables. Outputs in the output directory: cells_and_comparisons_partial.json
(written before the reference check) and eval_summary.json.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from typing import Any

import numpy as np
import torch

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from ga_1_common import (  # noqa: E402
    NumericProgress,
    load_local_gemma,
    load_problem_statements,
    load_raw_episode_text,
    stop_instance_if_requested,
    validate_no_external_api_surface,
)
from ga_1_tier0_run import METRICS, _score_fixed_picks  # noqa: E402
from ga_1_tier3_lora import attach_plain_lora, load_adapter_state  # noqa: E402
from ga_1_tier3_policy import identifier_ids_for, rollout_picks  # noqa: E402
from ga_1_tier3_states import (  # noqa: E402
    BUDGETS,
    EVAL_STARTS,
    assert_patch_is_inert,
    episode_key,
    make_scenarios_for_starts,
)
from rl_2 import (  # noqa: E402
    SEEDS,
    _bootstrap_seed_learner,
    _build_manifest,
    _episode_summaries,
    _load_bundle,
    _trace_policy_v2,
)

SUBSTRATE = "ratchet-cwsim/refreshed-head"
LLM_ARMS = ("zeroshot_history", "zeroshot_no_history",
            "trained_history", "trained_no_history")
BLIND = ("random", "max_entropy", "difficulty_matched", "privileged_oracle_upper_bound")
N_BOOT = 2000
RL2_TOLERANCE = 5e-5
# As in ga_1_tier0_run.py: only the oracle is seed-invariant. `eps`, the
# K-sample epistemic draws, is seeded from the CRN seed, and max_entropy /
# difficulty_matched rank on scores derived from it.
SEED_INVARIANT = ("privileged_oracle_upper_bound",)


def _jsonable(v):
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (np.integer, np.floating)):
        return v.item()
    return v


def _cell(rows):
    return {m: np.asarray([r[m][-1] for r in rows], dtype=np.float64) for m in METRICS}


def _boot(values: np.ndarray, seed: int):
    """Learner-clustered bootstrap. One training run ⇒ no seed level to cluster on."""
    return _bootstrap_seed_learner([np.asarray(values, dtype=np.float64)], seed,
                                   n_boot=N_BOOT)


def adapter_effect_canary(adapters, ckpt) -> dict:
    """Assert the trained adapter is a real, loadable, non-identity change.

    Without this, a silent load failure -- or an adapter whose `B` never left its zero
    initialisation -- makes every `trained_*` arm bit-identical to its `zeroshot_*` twin.
    The difference-in-differences then reads exactly 0.0 and gets written up as "no
    mechanism", which is a claim about the model that would actually be a fact about the
    loader: the forwards run and the losses are finite, so only a direct check shows
    whether the module moved.

    `B` is zero-initialised so the adapter is an exact no-op at step 0, which means a
    nonzero `B` norm is precisely the evidence that training happened at all.
    """
    state = ckpt["adapter"]
    b_ref = {n: float(state[f"{n}.lora_b"].float().norm()) for n in adapters}
    a_ref = {n: float(state[f"{n}.lora_a"].float().norm()) for n in adapters}
    nonzero = sum(1 for v in b_ref.values() if v > 0)
    if nonzero == 0:
        raise RuntimeError(
            "adapter check FAILED: every lora_b is still at its zero initialisation, so "
            "the trained arms would be identical to the zero-shot arms and the DiD would "
            "be a fact about the checkpoint, not about the model."
        )
    load_adapter_state(adapters, state)
    # bf16 modules against fp32 reference: compare relatively, not absolutely.
    worst = max(abs(float(adapters[n].lora_b.detach().float().norm()) - b_ref[n])
                / max(b_ref[n], 1e-6) for n in adapters)
    if worst > 0.02:
        raise RuntimeError(
            f"adapter check FAILED: load did not reproduce the checkpoint's B norms "
            f"(worst relative error {worst:.3g})"
        )
    return {"n_layers": len(adapters), "lora_b_nonzero_layers": nonzero,
            "lora_b_norm_mean": float(np.mean(list(b_ref.values()))),
            "lora_a_norm_mean": float(np.mean(list(a_ref.values()))),
            "load_worst_relative_error": worst}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="results/ga_1_tier3_eval")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--rl2-dir", default="results/rl_2/ratchet-cwsim_refreshed-head")
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--val-learners", type=int, default=110)
    ap.add_argument("--seed", type=int, default=20260901)
    ap.add_argument("--shuffle-seed", type=int, default=20260903)
    ap.add_argument("--max-prompt-tokens", type=int, default=5120)
    ap.add_argument("--batch-rows", type=int, default=12)
    ap.add_argument("--token-budget", type=int, default=16000)
    ap.add_argument("--pilot-learners", type=int, default=0)
    ap.add_argument("--bucket", default="")
    ap.add_argument("--out-prefix", default="")
    ap.add_argument("--fuse-minutes", type=int, default=900)
    ap.add_argument("--overrun-minutes", type=int, default=0)
    ap.add_argument("--rearm-fuse", action="store_true")
    ap.add_argument("--stop-on-complete", action="store_true")
    ap.add_argument("--raw-dir", required=True)
    ap.add_argument("--derived-val-predictions", default="")
    for n in ("code_map", "diff_map", "codealone_meta", "codealone_states",
              "diff_meta", "diff_states"):
        ap.add_argument("--" + n.replace("_", "-"), required=True)
    args = ap.parse_args()

    out = REPO / args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    progress = NumericProgress(out, args.bucket, args.out_prefix,
                               fuse_minutes=args.fuse_minutes,
                               overrun_minutes=args.overrun_minutes,
                               rearm_fuse=args.rearm_fuse)
    validate_no_external_api_surface([pathlib.Path(__file__),
                                      REPO / "scripts/ga_1_tier3_policy.py"])
    t0 = time.time()
    manifest = _build_manifest(args)
    model, _, bundle_meta = _load_bundle(SUBSTRATE, args.device)
    summaries = _episode_summaries(model, manifest, args.device)
    inert = assert_patch_is_inert(manifest, summaries, args.device, 0, args.val_learners)
    scen = make_scenarios_for_starts(manifest, summaries, args.device, EVAL_STARTS,
                                     0, args.val_learners)
    cohort = {s: (scen[f"val_start{s}"][:args.pilot_learners] if args.pilot_learners
                  else scen[f"val_start{s}"]) for s in EVAL_STARTS}

    statements = load_problem_statements()
    episode_text = load_raw_episode_text(args.raw_dir)
    bundle = load_local_gemma(args.device)
    token_ids = identifier_ids_for(bundle["tokenizer"])
    adapters = attach_plain_lora(bundle["model"])
    ckpt = torch.load(REPO / args.adapter, map_location="cpu", weights_only=False)
    adapter_effect = adapter_effect_canary(adapters, ckpt)
    print(f"[adapter] {json.dumps(adapter_effect)}", flush=True)

    picks: dict[tuple[str, int, int], dict[str, list[str]]] = {}
    total = len(LLM_ARMS) * len(EVAL_STARTS) * len(BUDGETS)
    done = 0
    for arm in LLM_ARMS:
        trained = arm.startswith("trained")
        with_history = arm.endswith("_history") and "no_history" not in arm
        if trained:
            load_adapter_state(adapters, ckpt["adapter"])
        else:
            for mod in adapters.values():
                with torch.no_grad():
                    mod.lora_b.zero_()          # exact base model: B=0 is a no-op
        bundle["model"].eval()
        for start in EVAL_STARTS:
            for budget in BUDGETS:
                picks[(arm, start, budget)] = rollout_picks(
                    bundle, cohort[start], budget, episode_text, statements, token_ids,
                    with_history=with_history, max_prompt_tokens=args.max_prompt_tokens,
                    batch_rows=args.batch_rows, token_budget=args.token_budget,
                    shuffle_seed=args.shuffle_seed, arm="eval",
                    on_batch=lambda d, t: progress.maybe_heartbeat(
                        "tier3_rollout", done, total, enforce_overrun=True, arm=arm))
                done += 1
                progress.heartbeat("tier3_rollout", done, total, arm=arm,
                                   start=start, budget=budget)

    # How often training changed nothing at the level that actually gets scored.  A
    # nonzero `B` proves the weights moved; it does not prove the greedy ARGMAX moved.
    # Reported, never enforced: identical picks with a nonzero adapter is a real null,
    # whereas identical picks with a zero adapter is a bug -- and the adapter check above
    # is what separates the two.
    pick_agreement = {}
    for start in EVAL_STARTS:
        for budget in BUDGETS:
            for base_arm, tr_arm in (("zeroshot_history", "trained_history"),
                                     ("zeroshot_no_history", "trained_no_history")):
                a, b = picks[(base_arm, start, budget)], picks[(tr_arm, start, budget)]
                same = sum(1 for k in a if a[k] == b.get(k))
                pick_agreement[f"{tr_arm}|start{start}|B{budget}"] = {
                    "identical_episodes": same, "n_episodes": len(a),
                    "fraction": round(same / max(1, len(a)), 4)}
    print(f"[picks] identical fractions: "
          f"{json.dumps({k: v['fraction'] for k, v in pick_agreement.items()})}", flush=True)

    cells: dict[str, Any] = {}
    arrays: dict[tuple[str, int, int], dict[str, np.ndarray]] = {}
    blind_cells: dict[tuple[str, int, int], list[dict[str, np.ndarray]]] = {}
    for start in EVAL_STARTS:
        eps = cohort[start]
        for budget in BUDGETS:
            for arm in LLM_ARMS:
                rows = [_score_fixed_picks(model, ep,
                                           picks[(arm, start, budget)][episode_key(ep)],
                                           args.device) for ep in eps]
                arrays[(arm, start, budget)] = _cell(rows)
            for policy in BLIND:
                per_seed = []
                cache: dict[int, Any] = {}
                for sd in SEEDS:
                    rows = []
                    for i, ep in enumerate(eps):
                        if policy in SEED_INVARIANT and i in cache:
                            rows.append(cache[i])
                            continue
                        tr = _trace_policy_v2(model, ep, policy, budget, args.device,
                                              sd + 100000 + start * 1000 + i, None)
                        if policy in SEED_INVARIANT:
                            cache[i] = tr
                        rows.append(tr)
                    per_seed.append(_cell(rows))
                blind_cells[(policy, start, budget)] = per_seed
                # Paired against the LLM arms via the per-learner mean over CRN seeds.
                arrays[(policy, start, budget)] = {
                    m: np.mean([c[m] for c in per_seed], axis=0) for m in METRICS}
            for name in LLM_ARMS:
                key = f"start{start}|{name}|B{budget}"
                cells[key] = {m: _boot(arrays[(name, start, budget)][m], args.seed + len(key))
                              for m in METRICS}
            for name in BLIND:
                key = f"start{start}|{name}|B{budget}"
                per_seed = blind_cells[(name, start, budget)]
                cells[key] = {m: _bootstrap_seed_learner([c[m] for c in per_seed],
                                                         args.seed + len(key), n_boot=N_BOOT)
                              for m in METRICS}
            progress.heartbeat("tier3_score", 1, 1, start=start, budget=budget)

    comparisons: dict[str, Any] = {}
    for start in EVAL_STARTS:
        for budget in BUDGETS:
            def paired(a, b, m):
                return arrays[(a, start, budget)][m] - arrays[(b, start, budget)][m]
            for a in ("trained_history", "trained_no_history", "zeroshot_history"):
                for b in list(BLIND) + (["zeroshot_history"] if a.startswith("trained") else []):
                    if a == b:
                        continue
                    key = f"start{start}|{a}-minus-{b}|B{budget}"
                    comparisons[key] = {m: _boot(paired(a, b, m), args.seed + 7000 + len(key))
                                        for m in METRICS}
            # the two history gaps and the difference between them
            for tag, hi, lo in (("trained_history_gap", "trained_history", "trained_no_history"),
                                ("zeroshot_history_gap", "zeroshot_history", "zeroshot_no_history")):
                key = f"start{start}|{tag}|B{budget}"
                comparisons[key] = {m: _boot(paired(hi, lo, m), args.seed + 9000 + len(key))
                                    for m in METRICS}
            key = f"start{start}|DiD_trained_minus_zeroshot_history_gap|B{budget}"
            comparisons[key] = {}
            for m in METRICS:
                did = (paired("trained_history", "trained_no_history", m)
                       - paired("zeroshot_history", "zeroshot_no_history", m))
                comparisons[key][m] = _boot(did, args.seed + 11000 + len(key))

    # Save now, before the reference check against rl_2.py.  This is the expensive result
    # -- every LLM rollout, every blind-comparator trace -- and a failure in the check
    # below (or anything after it) must not discard it: evaluation checkpoints too.
    (out / "cells_and_comparisons_partial.json").write_text(
        json.dumps(_jsonable({"cells": cells, "comparisons": comparisons}),
                  indent=2, sort_keys=True) + "\n")
    progress.upload(out / "cells_and_comparisons_partial.json")

    # ---- instrument check: recomputed blind cells vs the committed rl_2.py panel ----
    #
    # Deviation from the zero-shot runner, where a breach raises (there the same GPU
    # pipeline reran on the same stack and agreed to 0.000000). Here the same numbers are
    # re-derived on a fresh machine, and a standalone CPU-only recomputation that never
    # touches the LLM (`scripts/ga_1_tier3_diag_rl2_drift.py`) shows a narrow, structural
    # effect: virtually every cell carries a small (1e-6 to 3e-5) logloss-only floating
    # point gap, consistent with an ordinary compute-path difference (CPU vs the
    # substrate's original GPU numerics, non-associative float summation, or a library
    # version difference) rather than a construction bug. It compounds specifically in
    # `privileged_oracle_upper_bound` at the longest budget (16 sequential,
    # foresight-conditioned steps), where the standalone check reproduced the same worst
    # cell at a comparable magnitude (0.00091 CPU vs 0.00112 GPU). Cohort sizes, the
    # substrate checkpoint sha256 and `n_learners=110` all match the committed panel,
    # excluding a cohort or checkpoint mismatch. The check therefore always runs and its
    # full per-cell breakdown is always recorded, but a breach does not abort the run.
    rl2 = json.loads((REPO / args.rl2_dir / "summary.json").read_text())
    max_ref_err = 0.0
    checked = 0
    worst_cells: list[dict[str, Any]] = []
    for start in EVAL_STARTS:
        for budget in BUDGETS:
            for name in BLIND:
                key = f"start{start}|{name}|B{budget}"
                if key not in rl2.get("cells", {}):
                    continue
                for m in METRICS:
                    err = abs(float(cells[key][m][0]) - float(rl2["cells"][key][m][0]))
                    max_ref_err = max(max_ref_err, err)
                    checked += 1
                    if err > RL2_TOLERANCE:
                        worst_cells.append({"key": key, "metric": m, "abs_error": err})
    worst_cells.sort(key=lambda r: -r["abs_error"])
    rl2_agreement = {
        "comparisons": checked, "max_abs_point_error": max_ref_err,
        "tolerance": RL2_TOLERANCE,
        "pass": bool(checked and max_ref_err <= RL2_TOLERANCE),
        "cells_exceeding_tolerance": len(worst_cells),
        "worst_cells": worst_cells[:10],
    }
    if not rl2_agreement["pass"]:
        print(f"[warn] blind comparators drifted from rl_2 by {max_ref_err:.3g} "
              f"> {RL2_TOLERANCE:.3g} in {len(worst_cells)}/{checked} comparisons -- "
              f"recorded in eval_summary.json, not fatal (see source comment)", flush=True)

    actor_reference = {}
    for start in EVAL_STARTS:
        for budget in BUDGETS:
            k = f"start{start}|rl_actor|B{budget}"
            if k in rl2.get("cells", {}):
                actor_reference[k] = rl2["cells"][k]

    summary = {
        "status": "pilot_complete" if args.pilot_learners else "complete",
        "lane": "T0", "rung": "A", "substrate": SUBSTRATE,
        "substrate_sha256": bundle_meta.get("checkpoint_sha256"),
        "single_training_run": True,
        "adapter": str(args.adapter), "adapter_epoch": ckpt.get("epoch"),
        "eval_starts": list(EVAL_STARTS), "budgets": list(BUDGETS),
        "val_learners": args.val_learners, "pilot_learners": args.pilot_learners,
        "shuffle": "identifier<->candidate CRN-permuted per state, training and eval",
        "shuffle_seed": args.shuffle_seed,
        "arms": list(LLM_ARMS), "blind_comparators": list(BLIND),
        "primary_metric": "dlogit (lower is better)",
        "guard_metrics": ["pred_kl", "logloss"],
        "primary_mechanism_read": "DiD_trained_minus_zeroshot_history_gap, CI-clear negative",
        "canaries": {"starts_narrowing_inert": inert,
                     "adapter_effect": adapter_effect,
                     "trained_vs_zeroshot_pick_agreement": pick_agreement,
                     "rl2_blind_reference": rl2_agreement},
        "cells": cells, "comparisons": comparisons,
        "rl2_actor_reference_start3_only_unpaired": actor_reference,
        "disclosures": [
            "SINGLE TRAINING RUN: one training seed (pre-registered); no seed-level "
            "clustering is possible and none is claimed.",
            "The rl_2.py actor is quoted from committed cells at BOTH evaluation starts as an "
            "UNPAIRED reference; it is not re-run.",
            "Blind comparators are computed over the three rl_2.py CRN seeds; LLM-vs-blind "
            "pairs use the blind arm's per-learner mean across those seeds.",
            "Zero-shot arms are evaluated under the same shuffle and form the "
            "pre-registered zero-shot baseline row.",
        ],
        "wall_s": time.time() - t0,
    }
    (out / "eval_summary.json").write_text(json.dumps(_jsonable(summary), indent=2,
                                                      sort_keys=True) + "\n")
    progress.upload(out / "eval_summary.json")
    progress.heartbeat("tier3_eval_complete", 1, 1)
    print(json.dumps({"status": summary["status"], "wall_h": summary["wall_s"] / 3600},
                     indent=2))
    stop_instance_if_requested(args.stop_on_complete)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
