#!/usr/bin/env python3
"""Trains and evaluates the CW planner actor on three beliefs (§8.2): a pointer actor trained on
replay per belief substrate, selected on validation, three seeds, against EPIG and blind baselines.

A substrate (the CodeWorkout classifier `ratchet-original`, the CodeWorkout generative simulator
`ratchet-cwsim/cold-head`, and its refreshed outcome head `ratchet-cwsim/refreshed-head`) is
obtained only through ``atom.simulator.cw_substrate.load_substrate(name)``, which is the single
authority for the filter, outcome head, codec and belief handoff; the runner takes no checkpoint
paths. The actor's deployed inputs are current-belief summaries, candidate predictive features,
step, budget and remaining-menu size; ``b_full`` is restricted to the reward and the discarded
privileged critic. Replay geometry and primitives are reused from ``cw_planner_suite.py``.
Outputs per substrate in ``--out-dir`` (default results/rl_2): policy_seed*.pt, summary.json,
epoch logs, figures, and cross_substrate.json.
Run: `python scripts/rl_2.py --substrates ratchet-original ... --raw-dir ... --code-map ...`.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import pathlib
import sys
from collections import Counter
from typing import Any

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from atom.data.cw_outcome_label import N_OUTCOME  # noqa: E402
from cw_planner_suite import (  # noqa: E402
    BUDGETS, STARTS, POLICIES, K_EVAL, N_BOOT, _build_manifest, _build_model,
    _episode_summaries, _filter_state, _point_logits, _candidate_scores,
    _dlogit, _kl, _center, _spearman, _jsonable, _sha256, run_preflight,
    _make_scenarios, menu_distribution,
)

SEEDS = (20260901, 20260902, 20260903)
PILOT_BUDGETS = (3, 5, 8)
MAX_EPOCHS = 40
EARLY_STOP_PATIENCE = 5
MIN_VALIDATION_IMPROVEMENT = 0.001
INTERNAL_VAL_MAX_LEARNERS = 40


def _arch(model: dict) -> dict:
    """Read architecture metadata from either substrate-bundle spelling."""
    meta = model.get("checkpoint_meta", {})
    if isinstance(meta, dict) and isinstance(meta.get("architecture"), dict):
        return meta["architecture"]
    if isinstance(model.get("architecture"), dict):
        return model["architecture"]
    return {}


def _frozen_filter_state(model: dict, events: list[dict], query: torch.Tensor,
                         device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the frozen substrate without constructing an actor-training graph."""
    with torch.no_grad():
        return _filter_state(model, events, query, device)


def _load_bundle(name: str, device: str) -> tuple[dict, Any, dict]:
    """Load one CW substrate bundle, never a raw checkpoint path.

    The documented loader is allowed to return either a mapping or a small object.
    This adapter accepts the equivalent spellings while requiring a replay model with
    ``attempt_encoder``, ``filt`` and ``outcome_head``.  The dense value-join handoff is
    verified when the bundle exposes it; no fallback path is accepted.
    """
    try:
        from atom.simulator.cw_substrate import load_substrate  # type: ignore
    except ImportError as exc:  # pragma: no cover - deliberate pre-handoff failure
        raise RuntimeError("CW substrate loader is unavailable: cannot start rl_2") from exc
    try:
        raw = load_substrate(name, device=device)
    except TypeError:
        raw = load_substrate(name)
    if isinstance(raw, dict):
        bundle = raw
    else:
        bundle = {k: getattr(raw, k) for k in dir(raw) if not k.startswith("_") and not callable(getattr(raw, k))}
        for k in ("model", "model_dict", "beliefs", "beliefs_path", "manifest", "metadata"):
            if hasattr(raw, k):
                bundle[k] = getattr(raw, k)
    model = bundle.get("model") or bundle.get("model_dict")
    if model is None and {"attempt_encoder", "filt", "outcome_head"}.issubset(bundle):
        model = bundle
    if not isinstance(model, dict) or not {"attempt_encoder", "filt", "outcome_head"}.issubset(model):
        raise TypeError("load_substrate(name) must expose a replay model dict with attempt_encoder/filt/outcome_head")
    beliefs = bundle.get("beliefs_path") or bundle.get("belief_path") or bundle.get("beliefs")
    if beliefs is None:
        raise ValueError(f"substrate {name} bundle has no dense val-prefix belief handoff")
    if isinstance(beliefs, (str, pathlib.Path)):
        bp = pathlib.Path(beliefs)
        if not bp.is_absolute():
            bp = REPO / bp
        if not bp.exists():
            raise FileNotFoundError(bp)
        # The handoff verifier is imported from the audited harness of cw_planner_suite.py.  A supplied
        # digest is mandatory when the loader publishes one.
        from cw_planner_suite import _verify_belief_handoff
        expected = str(bundle.get("beliefs_sha256", bundle.get("belief_sha256", "")))
        handoff = _verify_belief_handoff(bp, expected)
    elif isinstance(beliefs, np.lib.npyio.NpzFile):
        mu = np.asarray(beliefs["mu"]); ls = np.asarray(beliefs["log_sigma"])
        if mu.ndim != 2 or ls.shape != mu.shape or not np.isfinite(mu).all() or not np.isfinite(ls).all():
            raise ValueError("in-memory substrate belief handoff failed shape/finiteness check")
        handoff = {"n_rows": len(mu), "latent_dim": mu.shape[1], "finite": True,
                   "sha256": str(bundle.get("beliefs_sha256", "in-memory"))}
    else:
        raise TypeError("substrate beliefs must be a path or NPZ value-join artifact")
    for value in model.values():
        if hasattr(value, "eval"):
            value.eval()
    meta = dict(bundle.get("metadata") or {})
    meta.update({"name": name, "handoff": handoff,
                 "checkpoint_sha256": model.get("checkpoint_sha256", bundle.get("checkpoint_sha256", "unknown")),
                 "bundle_keys": sorted(bundle.keys())})
    return model, beliefs, meta


def _candidate_features(model: dict, mu: torch.Tensor, log_sigma: torch.Tensor,
                        candidates: list[dict], targets: list[dict], eps: torch.Tensor,
                        device: str) -> tuple[np.ndarray, np.ndarray]:
    """Return deployable predictive features and policy score records."""
    cc = torch.stack([c["codec"] for c in candidates]).to(device)
    logits = _point_logits(model, mu, cc, device)
    probs = torch.softmax(torch.from_numpy(logits), dim=-1).numpy()
    ent = -(probs * np.log(np.clip(probs, 1e-8, 1.0))).sum(axis=1)
    # K-spread is computed from the current belief only.  It is a feature, never a target.
    k = eps.shape[0]
    u = mu.unsqueeze(0) + log_sigma.exp().unsqueeze(0) * eps
    items = cc.unsqueeze(0).expand(k, len(candidates), -1).reshape(k * len(candidates), -1)
    uf = u.unsqueeze(1).expand(k, len(candidates), -1).reshape(k * len(candidates), -1)
    hidden = int(_arch(model).get("attempt_hidden", 64))
    prefix = torch.zeros((len(items), hidden), dtype=torch.float32, device=device)
    with torch.no_grad():
        ps = torch.softmax(model["outcome_head"](items, uf, prefix=prefix), dim=-1)
    ps = ps.reshape(k, len(candidates), N_OUTCOME).cpu().numpy()
    spread = ps.std(axis=0).mean(axis=1)
    feats = np.concatenate((probs, ent[:, None], spread[:, None]), axis=1)
    return feats.astype(np.float32), {i: {"entropy": float(ent[i]), "difficulty": float(probs[i, 3]),
                                          "spread": float(spread[i])} for i in range(len(candidates))}


class PointerActorV2(nn.Module):
    """Pointer actor with current predictive distribution, entropy and K-spread."""
    def __init__(self, codec_dim: int, state_dim: int = 8, hidden: int = 128):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(codec_dim + N_OUTCOME + 2, hidden), nn.Tanh(),
                                 nn.Linear(hidden, hidden), nn.Tanh())
        self.state = nn.Sequential(nn.Linear(state_dim, hidden), nn.Tanh())
        self.out = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.Tanh(), nn.Linear(hidden, 1))

    def forward(self, state: torch.Tensor, codecs: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        s = self.state(state).expand(codecs.shape[0], -1)
        e = self.enc(torch.cat((codecs, features), dim=1))
        return self.out(torch.cat((s, e), dim=1)).squeeze(1)


class PrivilegedCriticV2(nn.Module):
    def __init__(self, state_dim: int = 8, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_dim * 2, hidden), nn.Tanh(),
                                 nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1))

    def forward(self, state: torch.Tensor, full_state: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat((state, full_state), dim=-1)).squeeze(-1)


def _state(mu: torch.Tensor, ls: torch.Tensor, step: int, budget: int, remaining: int) -> torch.Tensor:
    return torch.stack((mu.mean(), mu.std(), ls.mean(), ls.std(),
                        torch.tensor(step / 30.0, device=mu.device),
                        torch.tensor(budget / 30.0, device=mu.device),
                        torch.tensor(remaining / 50.0, device=mu.device),
                        torch.tensor(1.0, device=mu.device)))


def _train_epoch(model: dict, actor: PointerActorV2, critic: PrivilegedCriticV2,
                 episodes: list[dict], budget: int, rng: np.random.Generator,
                 device: str, seed: int) -> list[float]:
    losses = []
    for idx in rng.permutation(len(episodes)):
        ep = episodes[int(idx)]
        if not ep["candidates"]:
            continue
        query = ep["targets"][0]["codec"]
        target_codecs = torch.stack([t["codec"] for t in ep["targets"]])
        # The substrate is frozen during actor training.  Detach its belief state at
        # this boundary so autograd cannot enter the eval-mode cuDNN GRU (which would
        # otherwise attempt a backward pass through a frozen recurrent module).
        full_mu, full_ls = _frozen_filter_state(model, ep["full_events"], query, device)
        full_state = _state(full_mu, full_ls, 0, budget, len(ep["candidates"]))
        eps = torch.randn(K_EVAL, full_mu.numel(), generator=torch.Generator(device="cpu").manual_seed(seed + int(idx))).to(device)
        known = list(ep["prefix_events"]); remaining = list(range(len(ep["candidates"])))
        logps, vals, rewards = [], [], []
        for step in range(min(budget, len(remaining))):
            mu, ls = _frozen_filter_state(model, sorted(known, key=lambda x: x["order"]), query, device)
            state = _state(mu, ls, step, budget, len(remaining))
            cands = [ep["candidates"][i] for i in remaining]
            feats, _ = _candidate_features(model, mu, ls, cands, ep["targets"], eps, device)
            cc = torch.stack([c["codec"] for c in cands]).to(device)
            ff = torch.from_numpy(feats).to(device)
            dist = torch.distributions.Categorical(logits=actor(state, cc, ff))
            j = dist.sample(); chosen_idx = remaining[int(j.item())]
            before = _dlogit(_point_logits(model, mu, target_codecs, device),
                             _point_logits(model, full_mu, target_codecs, device))
            known.append(ep["candidates"][chosen_idx]); remaining.remove(chosen_idx)
            mu2, _ = _frozen_filter_state(model, sorted(known, key=lambda x: x["order"]), query, device)
            after = _dlogit(_point_logits(model, mu2, target_codecs, device),
                            _point_logits(model, full_mu, target_codecs, device))
            logps.append(dist.log_prob(j)); vals.append(critic(state, full_state)); rewards.append(before - after)
        if not rewards:
            continue
        returns, g = [], 0.0
        for r in rewards[::-1]:
            g += float(r); returns.append(g)
        returns = torch.tensor(returns[::-1], dtype=torch.float32, device=device)
        vals_t = torch.stack(vals); adv = returns - vals_t.detach()
        loss = -(torch.stack(logps) * adv).mean() + 0.5 * F.mse_loss(vals_t, returns)
        # Average the episode contribution before the per-budget optimizer step; without
        # this normalization the larger fit cohort would make the update magnitude depend
        # on cohort size rather than the registered budget-balanced schedule.
        (loss / max(1, len(episodes))).backward()
        losses.append(float(loss.item()))
    return losses


def _fit_actor(model: dict, train_eps: list[dict], device: str, seed: int,
               epochs: int, pilot: bool, out: pathlib.Path) -> tuple[PointerActorV2, dict]:
    # Deterministic 80/20 learner split for validation checkpoint selection.
    learners = sorted({str(e["learner"]) for e in train_eps})
    cut = max(1, int(0.8 * len(learners))); fit_ids = set(learners[:cut])
    fit = [e for e in train_eps if str(e["learner"]) in fit_ids]
    internal_all = [x for x in learners if x not in fit_ids]
    # Keep checkpoint selection cheap and deterministic on the full cohort: select a
    # spread of at most 40 held-out learners, while the final external evaluation still
    # uses all 110 validation learners.  This is an internal selection set, not a claim
    # estimate, and its cap is pre-registered.
    if len(internal_all) > INTERNAL_VAL_MAX_LEARNERS:
        picks = np.linspace(0, len(internal_all) - 1, INTERNAL_VAL_MAX_LEARNERS, dtype=int)
        internal_ids = {internal_all[int(i)] for i in picks}
    else:
        internal_ids = set(internal_all)
    internal = [e for e in train_eps if str(e["learner"]) in internal_ids]
    codec_dim = int(fit[0]["candidates"][0]["codec"].numel())
    actor, critic = PointerActorV2(codec_dim).to(device), PrivilegedCriticV2().to(device)
    opt = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=3e-4)
    rng = np.random.default_rng(seed)
    # Pilot remains the fixed three-epoch feasibility gate.  Production uses the
    # pre-registered rule: cap at 40, stop only on the validation score after five
    # consecutive sub-threshold improvements.  Actor loss is deliberately ignored.
    epochs_cap = min(3, epochs) if pilot else min(MAX_EPOCHS, epochs)
    budget_grid = PILOT_BUDGETS if pilot else BUDGETS
    best_score, best_epoch, best_state = float("inf"), 0, None
    epoch_rows = []; stale_epochs = 0; stopping_reason = "cap"
    for epoch in range(1, epochs_cap + 1):
        # Balanced schedule: partition the fit cohort across budgets for this epoch,
        # rotating the assignment each epoch.  Thus every episode is seen exactly once
        # per epoch and every budget receives an equal (up to one) number of episodes,
        # while avoiding an 8x duplicate full-cohort pass, which would be infeasible on
        # a single GPU.
        losses = []
        perm = rng.permutation(len(fit))
        n_budgets = len(budget_grid)
        for budget_i, budget in enumerate(budget_grid):
            offset = (epoch - 1) % n_budgets
            budget_fit = [fit[int(i)] for pos, i in enumerate(perm)
                          if (pos + offset) % n_budgets == budget_i]
            if not budget_fit:
                continue
            opt.zero_grad(set_to_none=True)
            ls = _train_epoch(model, actor, critic, budget_fit, budget, rng, device,
                              seed + epoch * 10000 + budget)
            # One optimizer step per budget keeps all grid values equally represented;
            # rotating the partition prevents a learner from being tied to one budget.
            torch.nn.utils.clip_grad_norm_(list(actor.parameters()) + list(critic.parameters()), 1.0)
            opt.step(); losses.extend(ls)
        if internal:
            vals = []
            for i, ep in enumerate(internal):
                for b in budget_grid:
                    row = _trace_policy_v2(model, ep, "rl_actor", b, device,
                                           seed + 500000 + epoch * 1000 + i, actor)
                    vals.append(float(row["dlogit"][-1] + row["pred_kl"][-1]))
            score = float(np.mean(vals)) if vals else float("inf")
        else:
            score = float("inf")
        epoch_rows.append({"epoch": epoch, "score": score, "loss_mean": float(np.mean(losses)) if losses else None})
        if np.isfinite(score):
            improvement = best_score - score
            if improvement > MIN_VALIDATION_IMPROVEMENT:
                best_score, best_epoch = score, epoch
                best_state = {"actor": copy.deepcopy(actor.state_dict()), "critic": copy.deepcopy(critic.state_dict())}
                stale_epochs = 0
            else:
                stale_epochs += 1
        (out / f"seed{seed}_epoch{epoch}.json").write_text(json.dumps(epoch_rows[-1], indent=2) + "\n")
        if not pilot and stale_epochs >= EARLY_STOP_PATIENCE:
            stopping_reason = "validation_patience"
            break
    if best_state is None:
        raise RuntimeError(f"seed {seed} has no finite validation checkpoint")
    actor.load_state_dict(best_state["actor"])
    torch.save({"actor": best_state["actor"], "critic": best_state["critic"],
                "selected_epoch": best_epoch, "validation_score": best_score,
                "deployable_inputs": ["belief_summary", "remaining_menu_codecs", "predictive_distribution",
                                       "predictive_entropy", "k_spread", "step", "budget"],
                "reference_used_at_deployment": False}, out / f"policy_seed{seed}.pt")
    epochs_run = len(epoch_rows)
    return actor, {"epochs_run": epochs_run, "stopping_reason": stopping_reason,
                   "validation_patience": EARLY_STOP_PATIENCE,
                   "min_validation_improvement": MIN_VALIDATION_IMPROVEMENT,
                   "budget_schedule": "rotating_episode_partition",
                   "fit_learners": len(fit_ids),
                   "internal_val_learners": len(internal_ids),
                   "internal_val_cap": INTERNAL_VAL_MAX_LEARNERS,
                   "selected_epoch": best_epoch, "validation_score": best_score,
                   "epoch_rows": epoch_rows}


def _trace_policy_v2(model: dict, ep: dict, policy: str, budget: int, device: str,
                     seed: int, actor: PointerActorV2 | None = None) -> dict[str, Any]:
    rng = np.random.default_rng(seed); known = list(ep["prefix_events"])
    candidates = list(ep["candidates"]); targets = list(ep["targets"]); query = targets[0]["codec"]
    target_codecs = torch.stack([t["codec"] for t in targets])
    eps = torch.randn(K_EVAL, _arch(model).get("latent_dim", 32),
                      generator=torch.Generator(device="cpu").manual_seed(seed)).to(device)
    full_mu, _ = _frozen_filter_state(model, ep["full_events"], query, device)
    full_logits = _point_logits(model, full_mu, target_codecs, device)
    full_probs = torch.softmax(torch.from_numpy(full_logits), dim=-1).numpy()
    labels = np.asarray([t["label"] for t in targets], dtype=int)
    traces = {"logloss": [], "dlogit": [], "pred_kl": [], "picks": [], "pick_problems": []}
    remaining = list(range(len(candidates)))
    for step in range(min(budget, len(candidates)) + 1):
        mu, ls = _frozen_filter_state(model, sorted(known, key=lambda x: x["order"]), query, device)
        logits = _point_logits(model, mu, target_codecs, device); probs = torch.softmax(torch.from_numpy(logits), dim=-1).numpy()
        traces["logloss"].append(float(-np.log(np.clip(probs[np.arange(len(labels)), labels], 1e-8, 1)).mean()))
        traces["dlogit"].append(_dlogit(logits, full_logits)); traces["pred_kl"].append(_kl(probs, full_probs))
        if step >= budget or not remaining: break
        cands = [candidates[i] for i in remaining]; feats, scored = _candidate_features(model, mu, ls, cands, targets, eps, device)
        # EPIG/entropy/difficulty remain the audited blind policies.  Their scores are
        # obtained from the same CRN draws as the feature vector.
        score = _candidate_scores(model, mu, cands, targets, eps, device)
        if policy == "random": j = int(rng.integers(len(remaining)))
        elif policy == "greedy_epig": j = max(range(len(remaining)), key=lambda x: (score[x]["epig"], -x))
        elif policy == "max_entropy": j = max(range(len(remaining)), key=lambda x: (score[x]["entropy"], -x))
        elif policy == "difficulty_matched":
            j = int(np.argmin(np.abs(np.asarray([score[x]["difficulty"] for x in range(len(remaining))]) - np.median([score[x]["difficulty"] for x in range(len(remaining))]))))
        elif policy == "privileged_oracle_upper_bound":
            vals = []
            for x in range(len(remaining)):
                ev2 = sorted(known + [cands[x]], key=lambda z: z["order"]); mu2, _ = _frozen_filter_state(model, ev2, query, device)
                vals.append(_dlogit(_point_logits(model, mu2, target_codecs, device), full_logits))
            j = int(np.argmin(vals))
        elif policy == "rl_actor":
            if actor is None: raise ValueError("rl_actor requires actor")
            state = _state(mu, ls, step, budget, len(remaining)).to(device)
            cc = torch.stack([c["codec"] for c in cands]).to(device); ff = torch.from_numpy(feats).to(device)
            with torch.no_grad(): j = int(actor(state, cc, ff).argmax().item())
        else: raise ValueError(policy)
        chosen = remaining.pop(j); known.append(candidates[chosen]); traces["picks"].append(candidates[chosen]["episode_id"]); traces["pick_problems"].append(int(candidates[chosen]["problem_id"]))
    return traces


def _bootstrap_seed_learner(values: list[np.ndarray], seed: int, n_boot: int = N_BOOT) -> list[float]:
    """Two-level seed-then-learner bootstrap, returning mean/lo/hi."""
    arr = [np.asarray(v, dtype=float) for v in values if len(v)]
    if not arr: return [float("nan")] * 3
    rng = np.random.default_rng(seed); means = []
    for _ in range(n_boot):
        si = rng.integers(len(arr), size=len(arr)); sm = []
        for i in si:
            x = arr[int(i)]; sm.append(float(x[rng.integers(len(x), size=len(x))].mean()))
        means.append(float(np.mean(sm)))
    # The point estimate is the equally weighted mean of seed means, matching the
    # resampling unit.  (Starts 3/5/10 can have one or two fewer learners, so a flattened
    # row mean would otherwise give those seeds a tiny implicit weight advantage.)
    return [float(np.mean([x.mean() for x in arr])), float(np.quantile(means, .025)), float(np.quantile(means, .975))]


def _save_figure_family(out: pathlib.Path, summary: dict[str, Any], substrate: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    for start in STARTS:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
        for ax, metric, title in zip(axes, ("logloss", "dlogit", "pred_kl"), ("Target log-loss", "Centered-logit distance", "Predictive KL")):
            for policy in POLICIES:
                xs=[]; ys=[]; lo=[]; hi=[]
                for b in BUDGETS:
                    cell=summary["cells"].get(f"start{start}|{policy}|B{b}")
                    if cell is None: continue
                    xs.append(b); ys.append(cell[metric][0]); lo.append(cell[metric][1]); hi.append(cell[metric][2])
                if xs:
                    ax.plot(xs, ys, marker="o", label=policy.replace("_", " "))
                    ax.fill_between(xs, lo, hi, alpha=.12)
            ax.set_title(title); ax.set_xlabel("budget"); ax.grid(alpha=.2)
        axes[0].legend(fontsize=7); fig.suptitle(f"{substrate} — start {start}")
        fig.savefig(out / f"figure_{substrate.replace('/', '_')}_start{start}.png", dpi=160); plt.close(fig)


def run_substrate(name: str, args: argparse.Namespace) -> dict[str, Any]:
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)
    out = REPO / args.out_dir / name.replace("/", "_"); out.mkdir(parents=True, exist_ok=True)
    model, _, meta = _load_bundle(name, device)
    manifest_args = argparse.Namespace(**vars(args)); manifest = _build_manifest(manifest_args)
    summaries = _episode_summaries(model, manifest, device)
    # The substrate bundle is verified above; its dense beliefs are joined by the same raw
    # episode IDs used by the audited harness of cw_planner_suite.py.  Scenarios are built from the
    # bundle's filter model, never from a second checkpoint path.
    scenarios = _make_scenarios(manifest, summaries, device, args.train_learners, args.val_learners,
                                release_by_episode=None)
    menu = menu_distribution(scenarios); (out / "menu_distribution.json").write_text(json.dumps(_jsonable(menu), indent=2) + "\n")
    preflight = run_preflight(model, scenarios, device, out / "preflight.json", args.seed)
    if not preflight["V2_gauge"]["pass"]: raise RuntimeError(f"{name}: V2 gauge failed")
    train_meta={}; seed_cells={}
    seeds = SEEDS[:args.n_seeds]
    for seed in seeds:
        policy_path = out / f"policy_seed{seed}.pt"
        if args.resume and policy_path.exists():
            # A stopped worker can resume without retraining a completed seed.  The
            # checkpoint is the actor's standalone state; the external validation cells
            # below are independently resumable from their partial NPZs.
            codec_dim = int(scenarios["train_start0"][0]["candidates"][0]["codec"].numel())
            actor = PointerActorV2(codec_dim).to(device)
            saved = torch.load(policy_path, map_location=device, weights_only=False)
            actor.load_state_dict(saved["actor"])
            fit_meta = {"resumed": True, "selected_epoch": saved.get("selected_epoch"),
                        "validation_score": saved.get("validation_score")}
        else:
            actor, fit_meta = _fit_actor(model, scenarios["train_start0"], device, seed, args.epochs, args.pilot, out)
        train_meta[str(seed)] = fit_meta
        budgets = PILOT_BUDGETS if args.pilot else BUDGETS
        for start in STARTS:
            eps = scenarios.get(f"val_start{start}", [])
            for policy in POLICIES:
                for budget in budgets:
                    key=f"start{start}|{policy}|B{budget}|seed{seed}"
                    partial_path = out / f"partial_{key.replace('|','_')}.npz"
                    cell_key = f"start{start}|{policy}|B{budget}"
                    if args.resume and partial_path.exists():
                        saved = np.load(partial_path, allow_pickle=True)
                        seed_cells.setdefault(cell_key, {"logloss":[],"dlogit":[],"pred_kl":[]})
                        for m in ("logloss", "dlogit", "pred_kl"):
                            seed_cells[cell_key][m].append(np.asarray(saved[m]))
                        (out / "progress.json").write_text(json.dumps({"status":"running","substrate":name,"seed":seed,"cell":key,"resumed":True})+"\n")
                        continue
                    rows=[_trace_policy_v2(model, ep, policy, budget, device, seed + 100000 + start*1000+i, actor) for i,ep in enumerate(eps)]
                    ll=np.asarray([r["logloss"][-1] for r in rows]); dl=np.asarray([r["dlogit"][-1] for r in rows]); pk=np.asarray([r["pred_kl"][-1] for r in rows])
                    np.savez_compressed(partial_path, logloss=ll,dlogit=dl,pred_kl=pk,picks=np.asarray([r["picks"] for r in rows],dtype=object))
                    seed_cells.setdefault(cell_key, {"logloss":[],"dlogit":[],"pred_kl":[]})
                    for m,x in (("logloss",ll),("dlogit",dl),("pred_kl",pk)): seed_cells[f"start{start}|{policy}|B{budget}"][m].append(x)
                    (out / "progress.json").write_text(json.dumps({"status":"running","substrate":name,"seed":seed,"cell":key})+"\n")
    summary={"status":"complete","substrate":name,"device":device,"budgets":list(PILOT_BUDGETS if args.pilot else BUDGETS),"policies":list(POLICIES),"starts":list(STARTS),"menu_distribution":menu,"preflight":preflight,"bundle":meta,"train":train_meta,"cells":{}}
    for key, vals in seed_cells.items():
        summary["cells"][key]={m:_bootstrap_seed_learner(v, args.seed + i*1000 + len(key)) for i,(m,v) in enumerate(vals.items())}
    _save_figure_family(out, summary, name)
    costs={"primary":{str(b):float(.01*(1.15**b-1)/.15) for b in BUDGETS},"sensitivity":{str(b):float(.005*(1.10**b-1)/.10) for b in BUDGETS}}
    summary["cost_curve"]=costs; (out / "summary.json").write_text(json.dumps(_jsonable(summary), indent=2)+"\n"); (out / "progress.json").write_text(json.dumps({"status":"complete","substrate":name})+"\n")
    return summary


def main() -> int:
    ap=argparse.ArgumentParser()
    ap.add_argument("--substrates", nargs="+", default=["ratchet-original"])
    ap.add_argument("--device", default="auto"); ap.add_argument("--out-dir", default="results/rl_2")
    ap.add_argument("--pilot", action="store_true"); ap.add_argument("--n-seeds", type=int, default=3)
    ap.add_argument("--resume", action="store_true", help="resume completed seeds and partial cells")
    ap.add_argument("--epochs", type=int, default=MAX_EPOCHS); ap.add_argument("--seed", type=int, default=SEEDS[0])
    ap.add_argument("--train-learners", type=int, default=686); ap.add_argument("--val-learners", type=int, default=110)
    ap.add_argument("--derived-val-predictions", default="")
    for name in ("raw_dir","code_map","diff_map","codealone_meta","codealone_states","diff_meta","diff_states"):
        ap.add_argument("--"+name.replace("_","-"), required=name != "raw_dir")
    args=ap.parse_args(); outputs={}
    for substrate in args.substrates:
        outputs[substrate]=run_substrate(substrate,args)
    # Cross-substrate table is intentionally numeric and compact; figure families live
    # beside each substrate output.  The report appends interpretation only after this
    # table exists, preserving the preregistration above.
    root=REPO / args.out_dir; root.mkdir(parents=True,exist_ok=True)
    table={}
    for name,s in outputs.items():
        table[name]={k:v for k,v in s["cells"].items() if k.endswith("|B8") or k.endswith("|B16")}
    (root/"cross_substrate.json").write_text(json.dumps(_jsonable(table),indent=2)+"\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
