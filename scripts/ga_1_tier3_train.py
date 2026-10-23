#!/usr/bin/env python3
"""Fine-tunes the LLM selector on the simulator's pricing by exact-advantage fine-tuning (Figure 12).

The setting supplies exact, zero-cost action values: at a replay state `s`, asking menu item `j`
reveals the recorded outcome, so `r(s,j) = d(b_s, ref) - d(b_{s.j}, ref)` is computable for every
item. No critic, no reward model, no sampling -- the policy target is available in closed form,
so the update is a supervised KL to it:

    loss = KL(p* || pi_theta) + beta * KL(pi_theta || pi_base)

with `p*(j|s) ∝ exp(A(s,j)/tau)`, A z-scored within the state, tau=1.0, beta=0.1.

Pre-registered parameters (fixed here, not tuned): one training seed; epoch cap 4, patience 2,
best-internal-validation checkpoint; the full ~15K-state draw at starts {3,10}; identifier
shuffle in training and evaluation. Every table this produces is stamped "single training run".
Checks before training: the start-narrowing patch is inert, the reward's sign is right on real
replay, and prompts contain no reference/distance/future information.

Inputs: the CodeWorkout generative simulator with its refreshed outcome head (substrate), CW
statements and raw tables, the local Gemma backbone. Outputs in the output directory:
adapter_best.pt, epochs_partial.json, train_summary.json and caches (optionally uploaded to S3
via --bucket).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
import time

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
from ga_1_tier0_run import _score_fixed_picks  # noqa: E402
from ga_1_tier3_lora import (  # noqa: E402
    LORA_ALPHA,
    LORA_R,
    adapter_state_dict,
    adapters_disabled,
    attach_plain_lora,
    trainable_parameters,
)
from ga_1_tier3_policy import (  # noqa: E402
    identifier_ids_for,
    menu_logits,
    render_state,
    rollout_picks,
)
from ga_1_tier3_states import (  # noqa: E402
    BUDGETS,
    SPLIT_EXPECTED,
    TRAIN_STARTS,
    assert_patch_is_inert,
    crn_permutation,
    draw_training_states,
    episode_key,
    exact_advantages,
    fit_internal_split,
    make_scenarios_for_starts,
    soft_targets,
)
from rl_2 import _build_manifest, _episode_summaries, _load_bundle  # noqa: E402

TAU = 1.0
BETA = 0.1
LR = 1e-4
GRAD_CLIP = 1.0
EPOCH_CAP = 4
PATIENCE = 2
MIN_IMPROVEMENT = 0.001
SUBSTRATE = "ratchet-cwsim/refreshed-head"


def _sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


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


# ------------------------------------------------------------------ advantages


def compute_advantage_cache(model, states, device, progress, out_path: pathlib.Path):
    """One exact-advantage vector per state.  Small filter only -- no backbone."""
    if out_path.exists():
        z = np.load(out_path, allow_pickle=True)
        cached = list(z["adv"])
        if len(cached) == len(states):
            return [np.asarray(a, dtype=np.float64) for a in cached]
        raise RuntimeError(f"advantage cache has {len(cached)} rows, expected {len(states)}")
    adv = []
    for i, st in enumerate(states):
        adv.append(exact_advantages(model, st, device))
        if i % 200 == 0:
            progress.maybe_heartbeat("tier3_advantages", i, len(states))
    np.savez_compressed(out_path, adv=np.array(adv, dtype=object))
    progress.heartbeat("tier3_advantages", len(states), len(states))
    return adv


def reward_polarity_canary(model, states, advantages, device, n_probe=100, seed=20260903):
    """Assert the reward's SIGN end-to-end on real replay.

    For each probed state, replay the best-advantage action and the worst-advantage
    action and require the best to land strictly nearer the reference. A reward whose
    sign is inverted would train the policy to move away from `b_full`, and every
    downstream number would still look plausible.
    """
    rng = np.random.default_rng(seed)
    idx = [int(i) for i in rng.permutation(len(states))[:n_probe]]
    ok = 0
    checked = 0
    for i in idx:
        st, a = states[i], advantages[i]
        if len(a) < 2:
            continue
        ep = st["episode"]
        # `_score_fixed_picks` prepends the episode's own prefix and resolves every id
        # against the CANDIDATE menu, so only candidate picks may be passed. Passing the
        # prefix ids too raises KeyError.
        known = [ep["candidates"][k]["episode_id"] for k in st["known_candidate_idx"]]
        best = st["remaining_idx"][int(np.argmax(a))]
        worst = st["remaining_idx"][int(np.argmin(a))]
        tb = _score_fixed_picks(model, ep, known + [ep["candidates"][best]["episode_id"]], device)
        tw = _score_fixed_picks(model, ep, known + [ep["candidates"][worst]["episode_id"]], device)
        checked += 1
        if tb["dlogit"][-1] < tw["dlogit"][-1]:
            ok += 1
    rate = ok / max(1, checked)
    return {"probed": checked, "correct": ok, "rate": rate, "pass": rate >= 0.95}


# --------------------------------------------------------------------- prompts


def build_state_prompts(states, advantages, episode_text, statements, tokenizer,
                        max_prompt_tokens, shuffle_seed, with_history=True):
    """Render every training state once, with its CRN permutation and aligned target."""
    rows = []
    for st, a in zip(states, advantages):
        ep = st["episode"]
        cands = [ep["candidates"][i] for i in st["remaining_idx"]]
        if len(cands) < 2:
            continue
        perm = crn_permutation(len(cands), "train", st["learner"], st["start"],
                               st["budget"], st["step"], seed=shuffle_seed)
        known = list(ep["prefix_events"]) + [ep["candidates"][i] for i in st["known_candidate_idx"]]
        prompt, _ = render_state(known, cands, st["budget"], st["step"], episode_text,
                                 statements, tokenizer, max_prompt_tokens, with_history, perm)
        p_star = soft_targets(a, tau=TAU)[np.asarray(perm)]   # align target to the menu order
        rows.append({"prompt": prompt, "n": len(cands), "p_star": p_star,
                     "learner": st["learner"], "start": st["start"],
                     "budget": st["budget"], "step": st["step"]})
    return rows


def _batch_shape_report(rows, tokenizer, args, n_probe=400, seed=20260903):
    """What a micro-batch actually costs, measured rather than assumed.

    Reasoning from `max_prompt_tokens` alone assumes every prompt sits near the 5,120
    cap, so that a 4-row chunk would exceed a 16,000-token budget and split into two
    forward/backward passes.  That is a claim about the length DISTRIBUTION; this
    reports the distribution and the resulting sub-batches per optimizer step, so
    sizing decisions are made on the data instead of on the cap.
    """
    from ga_1_tier0_readout import _length_batches

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(rows))[:min(n_probe, len(rows))]
    sample = [rows[int(i)]["prompt"] for i in idx]
    lens = [len(tokenizer(p, add_special_tokens=False)["input_ids"]) for p in sample]
    splits, padded = [], []
    for pos in range(0, len(sample) - args.micro_batch + 1, args.micro_batch):
        chunk = sample[pos:pos + args.micro_batch]
        _, batches = _length_batches(chunk, tokenizer, args.micro_batch,
                                     args.max_prompt_tokens, args.token_budget)
        splits.append(len(batches))
        chunk_lens = [len(tokenizer(p, add_special_tokens=False)["input_ids"]) for p in chunk]
        padded.append(sum(len(b) * max(min(chunk_lens[i], args.max_prompt_tokens) for i in b)
                          for b in batches))
    pct = [50, 90, 99, 100]
    return {
        "n_sampled": len(sample),
        "prompt_tokens": {f"p{q}": int(np.percentile(lens, q)) for q in pct},
        "prompt_tokens_mean": float(np.mean(lens)),
        "max_prompt_tokens": args.max_prompt_tokens,
        "micro_batch": args.micro_batch,
        "token_budget": args.token_budget,
        "sub_batches_per_step_mean": float(np.mean(splits)) if splits else None,
        "sub_batches_per_step_max": int(np.max(splits)) if splits else None,
        "padded_tokens_per_step_mean": float(np.mean(padded)) if padded else None,
    }


def leakage_canary(rows, n_probe=100, seed=20260903):
    """The prompt must contain no reference, no distance, no future event."""
    rng = np.random.default_rng(seed)
    banned = ("b_full", "reference", "dlogit", "advantage", "reward", "pred_kl")
    bad = []
    for i in rng.permutation(len(rows))[:n_probe]:
        text = rows[int(i)]["prompt"].lower()
        hits = [b for b in banned if b in text]
        if hits:
            bad.append((int(i), hits))
    return {"probed": min(n_probe, len(rows)), "violations": bad, "pass": not bad}


# -------------------------------------------------------------------- training


def _upload_adapter(path: pathlib.Path, args) -> None:
    """Ship the adapter to S3 without going through the numeric-suffix allowlist.

    `NumericProgress.upload` rejects `.pt`: it guards the licence boundary, which is
    about retained *text*, and LoRA weights are parameters.  Failures are swallowed deliberately --
    the checkpoint is already durable on disk by the time this runs, and a telemetry
    error must never be able to kill a training run.
    """
    if not args.bucket:
        return
    try:
        import boto3

        key = f"{args.out_prefix.rstrip('/')}/{path.name}" if args.out_prefix else path.name
        boto3.client("s3").upload_file(str(path), args.bucket, key)
    except Exception as exc:  # noqa: BLE001 - telemetry must not kill training
        print(f"[warn] adapter upload failed: {exc}", flush=True)


def base_policy_cache(bundle, rows, token_ids, args, adapters, progress, out_path):
    """`pi_base` per state, computed once with the adapter disabled and reused."""
    if out_path.exists():
        z = np.load(out_path, allow_pickle=True)
        cached = [np.asarray(x, dtype=np.float64) for x in z["p"]]
        if len(cached) == len(rows):
            return cached
        raise RuntimeError("base-policy cache length mismatch")
    prompts = [r["prompt"] for r in rows]
    sizes = [r["n"] for r in rows]
    with adapters_disabled(adapters):
        logits = menu_logits(bundle, prompts, sizes, token_ids, args.max_prompt_tokens,
                             args.batch_rows, args.token_budget, grad=False,
                             on_batch=lambda d, t: progress.maybe_heartbeat(
                                 "tier3_pi_base", d, t, enforce_overrun=True))
    out = [torch.softmax(lg.float(), dim=-1).cpu().numpy().astype(np.float64) for lg in logits]
    np.savez_compressed(out_path, p=np.array(out, dtype=object))
    return out


def eaft_loss(logits: torch.Tensor, p_star: torch.Tensor, p_base: torch.Tensor):
    """KL(p* || pi) + beta * KL(pi || pi_base), both in nats."""
    logp = torch.log_softmax(logits.float(), dim=-1)
    p = logp.exp()
    kl_target = torch.sum(p_star * (torch.log(p_star.clamp_min(1e-12)) - logp))
    kl_anchor = torch.sum(p * (logp - torch.log(p_base.clamp_min(1e-12))))
    return kl_target + BETA * kl_anchor, float(kl_target.item()), float(kl_anchor.item())


def internal_validation(bundle, internal_eps, episode_text, statements, token_ids, args,
                        model, progress, shuffle_seed):
    """The pre-registered selection proxy of rl_2.py, driven by the LLM policy: dlogit + pred_kl."""
    vals = []
    for budget in BUDGETS:
        picks = rollout_picks(bundle, internal_eps, budget, episode_text, statements,
                              token_ids, with_history=True,
                              max_prompt_tokens=args.max_prompt_tokens,
                              batch_rows=args.batch_rows, token_budget=args.token_budget,
                              shuffle_seed=shuffle_seed, arm="internal_val",
                              on_batch=lambda d, t: progress.maybe_heartbeat(
                                  "tier3_internal_val", d, t))
        for ep in internal_eps:
            tr = _score_fixed_picks(model, ep, picks[episode_key(ep)], args.device)
            vals.append(float(tr["dlogit"][-1] + tr["pred_kl"][-1]))
    return float(np.mean(vals)) if vals else float("inf")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="results/ga_1_tier3")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=20260901)
    ap.add_argument("--shuffle-seed", type=int, default=20260903)
    ap.add_argument("--max-prompt-tokens", type=int, default=5120)
    ap.add_argument("--batch-rows", type=int, default=8)
    ap.add_argument("--token-budget", type=int, default=12000)
    ap.add_argument("--micro-batch", type=int, default=2)
    ap.add_argument("--epoch-cap", type=int, default=EPOCH_CAP)
    ap.add_argument("--max-states", type=int, default=0, help="pilot only; 0 = full draw")
    ap.add_argument("--bucket", default="")
    ap.add_argument("--out-prefix", default="")
    ap.add_argument("--fuse-minutes", type=int, default=900)
    ap.add_argument("--overrun-minutes", type=int, default=0)
    ap.add_argument("--rearm-fuse", action="store_true")
    ap.add_argument("--stop-on-complete", action="store_true")
    ap.add_argument("--pilot", action="store_true")
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
                                      REPO / "scripts/ga_1_tier3_policy.py",
                                      REPO / "scripts/ga_1_tier3_states.py",
                                      REPO / "scripts/ga_1_common.py"])
    t0 = time.time()
    progress.heartbeat("tier3_start", 0, 1, substrate=SUBSTRATE, pilot=args.pilot)

    manifest = _build_manifest(args)
    model, _, bundle_meta = _load_bundle(SUBSTRATE, args.device)
    summaries = _episode_summaries(model, manifest, args.device)
    inert = assert_patch_is_inert(manifest, summaries, args.device, 0, 0)
    scen = make_scenarios_for_starts(manifest, summaries, args.device, TRAIN_STARTS, 0, 0)
    train_eps = [e for s in TRAIN_STARTS for e in scen[f"train_start{s}"]]
    fit, internal, split_meta = fit_internal_split(train_eps)
    if not args.pilot:
        if split_meta["fit_learners"] != SPLIT_EXPECTED["fit"]:
            raise RuntimeError(f"fit split changed: {split_meta}")
        if split_meta["internal_val_learners"] != SPLIT_EXPECTED["internal"]:
            raise RuntimeError(f"internal split changed: {split_meta}")
    fit_ids = {str(e["learner"]) for e in fit}
    states = draw_training_states(scen, fit_ids, TRAIN_STARTS, BUDGETS, args.shuffle_seed)
    if args.max_states:
        states = states[:args.max_states]
    progress.heartbeat("tier3_states", len(states), len(states), **split_meta)

    adv = compute_advantage_cache(model, states, args.device, progress,
                                  out / "advantages.npz")
    polarity = reward_polarity_canary(model, states, adv, args.device)
    if not polarity["pass"]:
        raise RuntimeError(f"reward-polarity check FAILED: {polarity}")

    statements = load_problem_statements()
    episode_text = load_raw_episode_text(args.raw_dir)
    bundle = load_local_gemma(args.device)
    tokenizer = bundle["tokenizer"]
    token_ids = identifier_ids_for(tokenizer)
    rows = build_state_prompts(states, adv, episode_text, statements, tokenizer,
                               args.max_prompt_tokens, args.shuffle_seed)
    leak = leakage_canary(rows)
    if not leak["pass"]:
        raise RuntimeError(f"leakage check FAILED: {leak['violations'][:3]}")
    shape = _batch_shape_report(rows, tokenizer, args)
    print(f"[shape] {json.dumps(shape)}", flush=True)

    adapters = attach_plain_lora(bundle["model"], r=LORA_R, alpha=LORA_ALPHA, seed=args.seed)
    n_trainable = sum(p.numel() for p in trainable_parameters(bundle["model"]))
    if hasattr(bundle["model"], "gradient_checkpointing_enable"):
        bundle["model"].gradient_checkpointing_enable()
    p_base = base_policy_cache(bundle, rows, token_ids, args, adapters, progress,
                               out / "pi_base.npz")

    opt = torch.optim.Adam(trainable_parameters(bundle["model"]), lr=LR)
    rng = np.random.default_rng(args.seed)
    internal_eps = [e for e in internal]
    best = {"score": float("inf"), "epoch": 0, "state": None}
    epoch_rows, stale = [], 0
    stopping = "cap"
    ckpt = out / "adapter_best.pt"

    def _write_ckpt(sel: dict) -> None:
        """Persist the selected adapter the moment it exists.

        Holding the only copy in memory until every epoch finished would let a mid-loop
        failure destroy the work of every epoch already completed.
        """
        torch.save({"adapter": sel["state"], "epoch": sel["epoch"], "score": sel["score"],
                    "architecture": {"lora_r": LORA_R, "lora_alpha": LORA_ALPHA,
                                     "targets": "text q/v", "backbone": bundle["hf_id"]},
                    "tau": TAU, "beta": BETA, "lr": LR, "seed": args.seed,
                    "shuffle_seed": args.shuffle_seed, "substrate": SUBSTRATE,
                    "substrate_sha256": bundle_meta.get("checkpoint_sha256"),
                    "n_states": len(rows), "split": split_meta,
                    "single_training_run": True}, ckpt)
        _upload_adapter(ckpt, args)

    for epoch in range(1, args.epoch_cap + 1):
        bundle["model"].train()
        order = rng.permutation(len(rows))
        losses, kls, anchors = [], [], []
        for pos in range(0, len(order), args.micro_batch):
            chunk = [int(i) for i in order[pos:pos + args.micro_batch]]
            prompts = [rows[i]["prompt"] for i in chunk]
            sizes = [rows[i]["n"] for i in chunk]
            logits = menu_logits(bundle, prompts, sizes, token_ids, args.max_prompt_tokens,
                                 args.micro_batch, args.token_budget, grad=True)
            opt.zero_grad(set_to_none=True)
            total = 0.0
            for k, i in enumerate(chunk):
                ps = torch.tensor(rows[i]["p_star"], dtype=torch.float32,
                                  device=logits[k].device)
                pb = torch.tensor(p_base[i], dtype=torch.float32, device=logits[k].device)
                loss, kt, ka = eaft_loss(logits[k], ps, pb)
                total = total + loss
                kls.append(kt)
                anchors.append(ka)
            total = total / max(1, len(chunk))
            total.backward()
            torch.nn.utils.clip_grad_norm_(trainable_parameters(bundle["model"]), GRAD_CLIP)
            opt.step()
            losses.append(float(total.item()))
            progress.maybe_heartbeat(f"tier3_train_e{epoch}", pos + len(chunk), len(order),
                                     enforce_overrun=True, epoch=epoch)
        bundle["model"].eval()
        score = internal_validation(bundle, internal_eps, episode_text, statements,
                                    token_ids, args, model, progress, args.shuffle_seed)
        epoch_rows.append({"epoch": epoch, "score": score,
                           "loss_mean": float(np.mean(losses)) if losses else None,
                           "kl_target_mean": float(np.mean(kls)) if kls else None,
                           "kl_anchor_mean": float(np.mean(anchors)) if anchors else None})
        progress.heartbeat("tier3_epoch", epoch, args.epoch_cap, score=score)
        if best["score"] - score > MIN_IMPROVEMENT:
            best = {"score": score, "epoch": epoch,
                    "state": adapter_state_dict(adapters)}
            _write_ckpt(best)
            stale = 0
        else:
            stale += 1
        # Written AFTER the best-vs-current comparison above: writing it before would
        # report the PREVIOUS epoch's best, so a reader mid-run (or after a crash) would
        # see a stale verdict while `adapter_best.pt` already carried the true one.
        (out / "epochs_partial.json").write_text(
            json.dumps(_jsonable({"epochs": epoch_rows, "best_epoch": best["epoch"],
                                  "best_score": best["score"]}), indent=2) + "\n")
        progress.upload(out / "epochs_partial.json")
        if stale >= PATIENCE:
            stopping = "validation_patience"
            break
    if best["state"] is None:
        best = {"score": epoch_rows[-1]["score"], "epoch": epoch_rows[-1]["epoch"],
                "state": adapter_state_dict(adapters)}
        stopping = "no_improvement_last_kept"

    _write_ckpt(best)
    summary = {
        "status": "pilot_complete" if args.pilot else "complete",
        "lane": "T0", "rung": "A", "substrate": SUBSTRATE,
        "substrate_sha256": bundle_meta.get("checkpoint_sha256"),
        "single_training_run": True, "seed": args.seed, "shuffle_seed": args.shuffle_seed,
        "train_starts": list(TRAIN_STARTS), "budgets": list(BUDGETS),
        "split": split_meta, "n_states_drawn": len(states), "n_states_trained": len(rows),
        "hyperparameters": {"tau": TAU, "beta": BETA, "lr": LR, "grad_clip": GRAD_CLIP,
                            "lora_r": LORA_R, "lora_alpha": LORA_ALPHA,
                            "epoch_cap": args.epoch_cap, "patience": PATIENCE},
        "n_trainable_params": n_trainable,
        "canaries": {"starts_narrowing_inert": inert, "reward_polarity": polarity,
                     "leakage": {"probed": leak["probed"], "pass": leak["pass"]}},
        "epochs": epoch_rows, "stopping_reason": stopping,
        "selected_epoch": best["epoch"], "selected_score": best["score"],
        "adapter_sha256": _sha256(ckpt), "wall_s": time.time() - t0,
    }
    (out / "train_summary.json").write_text(json.dumps(_jsonable(summary), indent=2,
                                                       sort_keys=True) + "\n")
    for name in ("train_summary.json",):
        progress.upload(out / name)
    progress.heartbeat("tier3_train_complete", 1, 1, selected_epoch=best["epoch"])
    print(json.dumps({"status": summary["status"], "selected_epoch": best["epoch"],
                      "selected_score": best["score"], "stopping": stopping,
                      "wall_h": summary["wall_s"] / 3600}, indent=2))
    stop_instance_if_requested(args.stop_on_complete)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
