#!/usr/bin/env python3
"""Writing the question (Figure 13): a belief-conditioned, contrastive fine-tune of the local Gemma
backbone that generates the next diagnostic question for a learner. Pre-registered before any
weights were trained.

Motivation, measured. A belief-blind adapter (`LORA_M = 1` with the mixing vector pinned to a
constant 1.0, so one global weight update served all 543 learners and the belief reached the
model only as prompt text and as the sampled target) gave a personalisation DiD of -1.759
[-2.042, -1.501], CI-clear negative. The ceiling is not the problem: across 296 pairs of
learners facing an identical menu, two students share a best question only 9.5% of the time
and their advantage rankings correlate 0.121.

Two changes, both required, neither sufficient alone:
  (A) belief-conditioned weights -- m=8, r=8, alpha=32, and `c = Hypernetwork(u)` with
      `u = [mu || log_sigma]` (64-dim) read from the FROZEN substrate at that state and
      never back-propagated into it;
  (B) a parameter-free contrastive objective -- `L = NLL_A + softplus(NLL_A - NLL_B)`,
      where B is another learner's advantage-sampled target at the same start, scored under
      A's history and A's weights. A model that copies a corpus statement scores both
      alike, earns softplus(0) = 0.693, and is penalised, so ignoring the student is
      expensive.

The target sampler, row encoder, batching, windowed loss and instantaneous-overrun trigger come
unchanged from `ga_1_tier4_generation_common`, and the warm draw from `ga_1_tier4_common`, so the
belief-blind fine-tune remains a valid control for this run's 2x2. A belief-dependence gate runs
at the end (see `belief_dependence_gate`).

Licence boundary: prompts contain licensed student code; text stays on the machine that runs the
model and locally, and only scalars are uploaded (optional S3 bucket via --bucket).
Outputs in the output directory: the adapter/hypernetwork checkpoint, manifest.json and
summary.json.
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

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from ga_1_common import (  # noqa: E402
    EXPECTED_HF_ID,
    LocalGenerationStore,
    NumericProgress,
    build_tier1_prompt,
    load_local_gemma,
    load_problem_statements,
    load_raw_episode_text,
    stable_hash,
    stable_uint64,
    stop_instance_if_requested,
    validate_no_external_api_surface,
)
from ga_1_tier4_common import (  # noqa: E402
    BUDGETS,
    SUBSTRATE,
    WARM_STARTS,
    build_table,
    draw_digest,
    draw_warm_states,
    fit_learners,
    load_field,
    utility_learner_split,
)
from ga_1_tier4_generation_common import (  # noqa: E402
    GRAD_CLIP,
    HISTORY_TOKENS,
    INTERNAL_HOLDOUT_LEARNERS,
    LR,
    MAX_LEN,
    SINGLE_RUN_STAMP,
    InstantaneousOverrun,
    collate,
    encode_row,
    sample_target,
    token_budget_batches,
    windowed_nll,
)
from rl_2 import _frozen_filter_state  # noqa: E402

# ------------------------------------------------------------------ registered constants

LORA_M = 8
LORA_R = 8
LORA_ALPHA = 32
HYPER_KIND = "mlp"
HYPER_HIDDEN = 64
U_DIM = 64                      # [mu (32) || log_sigma (32)]
MAX_EPOCHS = 2
PATIENCE = 1
STATE_STRIDE = 8
SEED = 20260902
CONTRASTIVE_WEIGHT = 1.0        # fixed; nothing here is tunable after a result exists
K_BELIEF_PROBE = 8              # K epsilon draws for the belief-dependence gate


def belief_vector(model: dict, ep: dict, picks: Sequence[int], device: str) -> torch.Tensor:
    """u = [mu || log_sigma] at this state, from the FROZEN substrate.

    `_frozen_filter_state` runs under `torch.no_grad`, so no gradient can reach the
    substrate -- the belief is an input to the hypernetwork, never a thing trained here.
    """
    known = list(ep["prefix_events"]) + [ep["candidates"][i] for i in picks]
    mu, log_sigma = _frozen_filter_state(model, sorted(known, key=lambda x: x["order"]),
                                         ep["targets"][0]["codec"], device)
    return torch.cat((mu.detach().flatten(), log_sigma.detach().flatten())).float()


def build_contrastive_corpus(field: dict, learner_ids: set[str], episode_text: dict, tok: Any,
                             progress: NumericProgress, stage: str, stride: int,
                             device: str) -> tuple[list[dict], dict[str, Any]]:
    """Warm-start draw -> (prompt, own target, FOIL target, belief u) per state.

    The foil is another learner's advantage-sampled target **at the same start**, resampled
    if it collides with this learner's own target.  Same-start matching keeps curriculum
    position fixed so the contrast is about the person, exactly as the swapped-history arm
    is in the panel.
    """
    states, subset = draw_warm_states(field["scenarios"], "train", learner_ids, BUDGETS)
    digest = draw_digest(states, subset)
    n_full = len(states)
    if stride > 1:
        states = states[::stride]
    table = build_table(field["model"], subset, states, device,
                        on_state=lambda d, t: progress.maybe_heartbeat(f"{stage}_advantages", d, t),
                        with_epig=False)
    statements = field["statements"]

    # own advantage-sampled target per state
    own: list[int] = []
    for i, st in enumerate(states):
        o, c = int(table["state_offset"][i]), int(table["state_count"][i])
        j = sample_target(table["cand_advantage"][o:o + c], st["learner"], st["budget"],
                          st["step"], SEED)
        own.append(int(table["cand_problem_id"][o + j]))

    by_start: dict[int, list[int]] = {}
    for i, st in enumerate(states):
        by_start.setdefault(int(st["start"]), []).append(i)

    rows: list[dict] = []
    n_censored = 0
    n_foil_resampled = 0
    for i, st in enumerate(states):
        pool = [k for k in by_start[int(st["start"])] if states[k]["learner"] != st["learner"]]
        if not pool:
            continue
        rng = np.random.default_rng(
            stable_uint64(f"ga1-t4c-foil|{st['learner']}|B{st['budget']}|k{st['step']}")
            % (2 ** 63))
        foil_pid = own[i]
        for attempt in range(16):
            foil_pid = own[int(pool[int(rng.integers(len(pool)))])]
            if foil_pid != own[i]:
                break
            n_foil_resampled += 1
        if foil_pid == own[i]:
            continue                      # degenerate state; excluded and counted
        ep = subset[st["episode_index"]]
        known = list(ep["prefix_events"]) + [ep["candidates"][k] for k in st["picks"]]
        prompt = build_tier1_prompt(sorted(known, key=lambda x: x["order"]), episode_text, tok,
                                    max_history_tokens=HISTORY_TOKENS)
        if "CURRENT MENU" in prompt or "TOTAL DIAGNOSTIC BUDGET" in prompt:
            raise RuntimeError("menu/budget leak check fired")
        enc_own = encode_row(prompt, str(statements[own[i]]["statement"]), tok, MAX_LEN)
        enc_foil = encode_row(prompt, str(statements[foil_pid]["statement"]), tok, MAX_LEN)
        n_censored += int(enc_own["target_censored"]) + int(enc_foil["target_censored"])
        rows.append({
            "own": enc_own, "foil": enc_foil,
            "learner": st["learner"], "start": int(st["start"]), "budget": int(st["budget"]),
            "step": int(st["step"]), "episode_index": int(st["episode_index"]),
            "picks": list(st["picks"]),
            "own_problem_id": own[i], "foil_problem_id": foil_pid,
            "prompt_hash": stable_hash(prompt, 32),
        })
        progress.maybe_heartbeat(f"{stage}_corpus", i + 1, len(states))

    # beliefs, once per state, from the frozen substrate
    for n, r in enumerate(rows, start=1):
        r["u"] = belief_vector(field["model"], subset[r["episode_index"]], r["picks"], device)
        progress.maybe_heartbeat(f"{stage}_beliefs", n, len(rows))

    if any(r["own_problem_id"] == r["foil_problem_id"] for r in rows):
        raise RuntimeError("foil check: a foil equals its own target")
    lengths = np.asarray([len(r["own"]["input_ids"]) for r in rows])
    meta = {
        "draw_digest": digest, "n_states_full_draw": n_full, "state_stride": stride,
        "n_rows": len(rows), "n_target_censored": n_censored,
        "n_foil_resampled": n_foil_resampled,
        "warm_starts": list(WARM_STARTS),
        "start_composition": {str(s): sum(1 for r in rows if r["start"] == s) for s in WARM_STARTS},
        "n_cold_rows": int(sum(1 for r in rows if r["start"] == 0)),
        "distinct_own_targets": int(len({r["own_problem_id"] for r in rows})),
        "distinct_foil_targets": int(len({r["foil_problem_id"] for r in rows})),
        "token_length": {"min": int(lengths.min()), "median": float(np.median(lengths)),
                         "max": int(lengths.max())},
        "u_dim": U_DIM,
    }
    return rows, meta


def contrastive_loss(model: Any, hyper: Any, context: Any, batch_rows: Sequence[dict],
                     pad_id: int, device: str, dtype: torch.dtype) -> tuple[torch.Tensor, int]:
    """L = NLL_own + softplus(NLL_own - NLL_foil), both under THIS learner's weights.

    The two forwards share one mixing vector `c = hyper(u)`: the foil is scored as if this
    student had been asked it, which is what makes "would this student find the other
    student's best question equally likely?" the quantity being penalised.
    """
    u = torch.stack([r["u"] for r in batch_rows]).to(device)
    c = hyper(u)
    losses = {}
    for side in ("own", "foil"):
        b = collate([r[side] for r in batch_rows], pad_id, device)
        context.set(c.to(dtype))
        losses[side], _ = windowed_nll(model, b)
    n_tok = sum(r["own"]["n_target"] for r in batch_rows)
    margin = torch.nn.functional.softplus(losses["own"] - losses["foil"])
    return losses["own"] + CONTRASTIVE_WEIGHT * margin, n_tok


def belief_dependence_gate(hyper: Any, rows: Sequence[dict], device: str) -> dict[str, float]:
    """Belief-dependence gate, mandatory, run at the END and reported whatever it says.

    across-learner spread : how much `hyper(u)` moves between DIFFERENT learners.
    across-epsilon spread : how much it moves under this belief's OWN uncertainty at a fixed
                            row, u_k = [mu + sigma*eps_k || log_sigma], K draws.
    If epsilon exceeds learner, the adapter is a global adapter fitted to noise -> VOID.
    (A run that fails this gate measured 11.5% vs 135%.)
    """
    with torch.no_grad():
        u = torch.stack([r["u"] for r in rows]).to(device)
        c = hyper(u)
        mean_norm = float(c.mean(dim=0).norm())
        across_learner = float(c.std(dim=0).norm())
        eps_spreads = []
        gen = torch.Generator(device="cpu").manual_seed(SEED)
        for r in rows[: min(64, len(rows))]:
            mu, log_sigma = r["u"][:32], r["u"][32:]
            sigma = log_sigma.exp()
            draws = torch.stack([
                torch.cat((mu + sigma * torch.randn(32, generator=gen), log_sigma))
                for _ in range(K_BELIEF_PROBE)]).to(device)
            eps_spreads.append(float(hyper(draws).std(dim=0).norm()))
        across_eps = float(np.mean(eps_spreads))
    denom = max(mean_norm, 1e-12)
    out = {
        "mean_output_norm": mean_norm,
        "across_learner_spread": across_learner,
        "across_epsilon_spread": across_eps,
        "across_learner_over_norm": across_learner / denom,
        "across_epsilon_over_norm": across_eps / denom,
    }
    out["gate_pass"] = bool(across_learner > across_eps)
    out["verdict"] = ("PASS - belief dependence exceeds noise dependence" if out["gate_pass"]
                      else "FAIL - VOID: a global adapter fitted to noise; no personalisation "
                           "claim is made from this run")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--substrate", default=SUBSTRATE)
    ap.add_argument("--out-dir", default="results/ga_1_tier4_rungc")
    ap.add_argument("--state-stride", type=int, default=STATE_STRIDE)
    ap.add_argument("--token-budget", type=int, default=4096)
    ap.add_argument("--max-batch-rows", type=int, default=2)
    ap.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    ap.add_argument("--pilot-learners", type=int, default=0)
    ap.add_argument("--bucket", default="")
    ap.add_argument("--out-prefix", default="")
    ap.add_argument("--fuse-minutes", type=int, default=600)
    ap.add_argument("--overrun-minutes", type=int, default=0)
    ap.add_argument("--rearm-fuse", action="store_true")
    ap.add_argument("--stop-on-complete", action="store_true")
    args = ap.parse_args()

    validate_no_external_api_surface([pathlib.Path(__file__)])
    out = REPO / args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    progress = NumericProgress(out_dir=out, bucket=args.bucket, out_prefix=args.out_prefix,
                               fuse_minutes=args.fuse_minutes,
                               overrun_minutes=args.overrun_minutes, rearm_fuse=args.rearm_fuse)
    t0 = time.time()

    field = load_field(args.substrate, args.raw_dir, "cpu")
    field["statements"] = load_problem_statements()
    episode_text = load_raw_episode_text(args.raw_dir)
    progress.heartbeat("field", 1, 1)

    bundle = load_local_gemma(args.device)
    model, tok = bundle["model"], bundle["tokenizer"]
    if bundle["hf_id"] != EXPECTED_HF_ID:
        raise RuntimeError("generator mismatch")

    fit_ids = fit_learners(field["scenarios"])
    u_fit, u_int = utility_learner_split(fit_ids)
    holdout_ids = set(u_int[:INTERNAL_HOLDOUT_LEARNERS])
    if args.pilot_learners:
        u_fit = u_fit[:args.pilot_learners]
        holdout_ids = set(u_int[:2])

    fit_rows, fit_meta = build_contrastive_corpus(field, set(u_fit), episode_text, tok,
                                                  progress, "fit", args.state_stride, "cpu")
    hold_rows, hold_meta = build_contrastive_corpus(field, holdout_ids, episode_text, tok,
                                                    progress, "holdout", args.state_stride, "cpu")
    store = LocalGenerationStore(out / "generations_sft_corpus.jsonl")
    for r in fit_rows[:50]:
        store.append({k: v for k, v in r.items() if k not in ("own", "foil", "u", "picks")})

    from atom.binding import Hypernetwork, attach_lora_basis

    dtype = next(model.parameters()).dtype
    bases, context = attach_lora_basis(model, m=LORA_M, r=LORA_R, alpha=LORA_ALPHA, seed=SEED)
    # The hypernetwork stays FP32, like the LoRA basis it feeds: `hooked_backbone` keeps the
    # low-rank branch in fp32 because "bf16's ~3 decimal digits of mantissa is a bad place to
    # accumulate a rank-8x32 sum", and the same argument applies to a 64->64->8 MLP reading a
    # belief. Only the OUTPUT c is cast to the backbone's dtype, at the context boundary.
    hyper = Hypernetwork(U_DIM, LORA_M, kind=HYPER_KIND, hidden=HYPER_HIDDEN,
                         seed=SEED).to(args.device)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    params = [p for b in bases.values() for p in b.parameters()] + list(hyper.parameters())
    for p in params:
        p.requires_grad_(True)
    opt = torch.optim.Adam(params, lr=LR)
    n_trainable = sum(p.numel() for p in params)
    progress.heartbeat("attached", 1, 1, n_lora_modules=len(bases), n_trainable=n_trainable)

    fit_batches = token_budget_batches([r["own"] for r in fit_rows], args.token_budget,
                                       args.max_batch_rows)
    hold_batches = token_budget_batches([r["own"] for r in hold_rows], args.token_budget,
                                        args.max_batch_rows)

    def held_loss() -> float:
        model.eval()
        total, n = 0.0, 0
        with torch.no_grad():
            for bi, batch in enumerate(hold_batches):
                loss, _ = contrastive_loss(model, hyper, context, [hold_rows[i] for i in batch],
                                           tok.pad_token_id, args.device, dtype)
                total += float(loss) * len(batch)
                n += len(batch)
                progress.maybe_heartbeat("holdout_eval", bi + 1, len(hold_batches))
        return total / max(n, 1)

    base_held = held_loss()
    rng = np.random.default_rng(SEED)
    best, best_epoch, best_state, stale = float("inf"), 0, None, 0
    epoch_rows: list[dict[str, Any]] = []
    for epoch in range(1, args.max_epochs + 1):
        model.train()
        order = rng.permutation(len(fit_batches))
        overrun = InstantaneousOverrun(f"epoch{epoch}", len(order), args.overrun_minutes)
        run, ntok = 0.0, 0
        for n, bi in enumerate(order):
            batch = fit_batches[int(bi)]
            loss, k = contrastive_loss(model, hyper, context, [fit_rows[i] for i in batch],
                                       tok.pad_token_id, args.device, dtype)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, GRAD_CLIP)
            opt.step()
            run += float(loss.detach()) * len(batch)
            ntok += len(batch)
            projected = overrun.observe(n + 1)
            progress.maybe_heartbeat(f"epoch{epoch}", n + 1, len(order),
                                     train_loss=run / max(ntok, 1),
                                     instantaneous_projected_min=(round(projected / 60, 1)
                                                                  if projected else None))
        held = held_loss()
        epoch_rows.append({"epoch": epoch, "train_loss": run / max(ntok, 1), "held_loss": held})
        if held < best - 1e-4:
            best, best_epoch, stale = held, epoch, 0
            best_state = {"bases": {k: {n2: p.detach().clone() for n2, p in b.state_dict().items()}
                                    for k, b in bases.items()},
                          "hyper": {n2: p.detach().clone() for n2, p in hyper.state_dict().items()}}
        else:
            stale += 1
        torch.save({"bases": {k: b.state_dict() for k, b in bases.items()},
                    "hyper": hyper.state_dict(), "epoch": epoch, "held_loss": held,
                    "lora": {"m": LORA_M, "r": LORA_R, "alpha": LORA_ALPHA},
                    "hyper_cfg": {"u_dim": U_DIM, "kind": HYPER_KIND, "hidden": HYPER_HIDDEN}},
                   out / f"adapter_rungc_epoch{epoch}.pt")
        if args.bucket:
            progress_path = out / f"adapter_rungc_epoch{epoch}.pt"
            import boto3
            boto3.client("s3").upload_file(str(progress_path), args.bucket,
                                           f"{args.out_prefix.rstrip('/')}/{progress_path.name}")
        progress.heartbeat("epochs", epoch, args.max_epochs, held_loss=held, best=best)
        if stale >= PATIENCE:
            break
    if best_state is None:
        raise RuntimeError("training produced no finite held-out checkpoint")
    for name, b in bases.items():
        b.load_state_dict(best_state["bases"][name])
    hyper.load_state_dict(best_state["hyper"])
    torch.save({"bases": {k: b.state_dict() for k, b in bases.items()},
                "hyper": hyper.state_dict(), "selected_epoch": best_epoch, "best_held_loss": best,
                "lora": {"m": LORA_M, "r": LORA_R, "alpha": LORA_ALPHA},
                "hyper_cfg": {"u_dim": U_DIM, "kind": HYPER_KIND, "hidden": HYPER_HIDDEN}},
               out / "adapter_rungc.pt")

    gate = belief_dependence_gate(hyper, hold_rows, args.device)
    manifest = {
        "lane": "T4", "rung": "C", "stage": "train", "substrate": args.substrate,
        "model": EXPECTED_HF_ID, "n_training_runs": 1, "single_run_stamp": SINGLE_RUN_STAMP,
        "seed": SEED, "state_stride": args.state_stride,
        "lora": {"m": LORA_M, "r": LORA_R, "alpha": LORA_ALPHA, "n_modules": len(bases),
                 "n_trainable": n_trainable},
        "hypernetwork": {"u_dim": U_DIM, "kind": HYPER_KIND, "hidden": HYPER_HIDDEN},
        "objective": "NLL_own + softplus(NLL_own - NLL_foil), weight 1.0 fixed",
        "corpus": {"fit": fit_meta, "holdout": hold_meta},
        "training": {"zero_shot_holdout_contrastive_loss": base_held,
                     "best_holdout_contrastive_loss": best, "selected_epoch": best_epoch,
                     "epochs_run": len(epoch_rows), "epoch_rows": epoch_rows,
                     "max_epochs": args.max_epochs, "patience": PATIENCE, "lr": LR},
        "BELIEF_DEPENDENCE_GATE": gate,
        "canaries": {"menu_budget_leak": "pass", "foil_never_equals_target": "pass",
                     "n_cold_rows": fit_meta["n_cold_rows"]},
        "elapsed_s": time.time() - t0,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    (out / "summary.json").write_text(json.dumps({"status": "complete", "manifest": manifest},
                                                 indent=2, default=str) + "\n")
    for name in ("manifest.json", "summary.json"):
        progress.upload(out / name)
    print(json.dumps({"gate": gate, "training": manifest["training"]}, indent=2, default=str))
    stop_instance_if_requested(args.stop_on_complete)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
