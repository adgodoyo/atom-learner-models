#!/usr/bin/env python
"""Budget-conditioned question-selection replay on CodeWorkout (the planner suite).

Runs a pre-flight check of the belief metric (V1 placebo, V2 gauge, V3 drift, V4
future gap), trains a pointer actor with a privileged critic, then replays every policy
(trained actor, random, greedy EPIG, max-entropy, difficulty-matched and a privileged
oracle upper bound) over start offsets and question budgets on validation learners.
The actor sees only the current belief summary, remaining-menu codecs and difficulty,
step and budget; the full-history belief shapes the training reward and feeds the
critic, which is not used at deployment.  ``privileged_oracle_upper_bound`` is a labelled
upper bound, never a baseline.
Inputs: the CodeWorkout classifier checkpoint (`--checkpoint`), its value-joined prefix
beliefs (`--beliefs`, from `cw_classifier_belief_handoff.py`) and the codec caches.
Outputs in `--out-dir`: menu_distribution.json, preflight.json, policy.pt, per-cell
partial_*.npz, progress.json and summary.json.  Raw code stays on the machine running it.
Run: `python scripts/cw_planner_suite.py --checkpoint ... --beliefs ... [--pilot]`.
Serves the §8.2 headline grid and Figure 10.
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
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from atom.data.cw_outcome_label import N_OUTCOME, OUTCOME_CLASSES  # noqa: E402
from atom.planner.epig import (  # noqa: E402
    categorical_entropy,
    sample_based_epig_categorical_categorical,
)
from atom.simulator.sampler_semantics import CURRENT_SAMPLER_SEMANTICS  # noqa: E402
from cw_planner_common import (  # noqa: E402
    _build_manifest,
    _build_model,
    _episode_summaries,
    _filter_state,
    _predict,
)

SEED = 20260829
BUDGETS = (3, 5, 8, 12, 16, 20, 25, 30)
STARTS = (0, 3, 5, 10)
POLICIES = ("rl_actor", "random", "greedy_epig", "max_entropy", "difficulty_matched",
            "privileged_oracle_upper_bound")
PRIMARY_POLICIES = ("rl_actor", "random", "greedy_epig", "max_entropy", "difficulty_matched")
K_EVAL = 8
N_BOOT = 2000


def _jsonable(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.integer, np.floating)):
        return x.item()
    return x


def _sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _center(x: np.ndarray) -> np.ndarray:
    return x - x.mean(axis=-1, keepdims=True)


def _dlogit(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((_center(a) - _center(b)) ** 2))


def _kl(p: np.ndarray, q: np.ndarray) -> float:
    p = np.clip(p, 1e-8, 1.0)
    q = np.clip(q, 1e-8, 1.0)
    return float(np.mean(np.sum(p * (np.log(p) - np.log(q)), axis=-1)))


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    ra = pd.Series(a).rank().to_numpy()
    rb = pd.Series(b).rank().to_numpy()
    return float(np.corrcoef(ra, rb)[0, 1])


def _bootstrap_mean(x: np.ndarray, seed: int, n_boot: int = N_BOOT) -> list[float]:
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return [float("nan")] * 3
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    means = x[idx].mean(axis=1)
    return [float(x.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def _verify_belief_handoff(path: pathlib.Path, expected_sha: str = "") -> dict[str, Any]:
    """Check the belief handoff file (hash, fields, join keys, finite shapes)."""
    got = _sha256(path)
    if expected_sha and got != expected_sha:
        raise ValueError(f"belief handoff SHA mismatch: expected {expected_sha}, got {got}")
    z = np.load(path, allow_pickle=False)
    names = set(z.files)
    required = {"mu", "log_sigma"}
    if not required.issubset(names):
        raise ValueError(f"belief handoff missing {sorted(required - names)}; fields={sorted(names)}")
    join_alternatives = ({"learner", "episode_id"}, {"learner", "t"}, {"learner", "episode_id", "t"})
    if not any(x.issubset(names) for x in join_alternatives):
        raise ValueError("belief handoff has no learner/episode value-join keys")
    mu, ls = np.asarray(z["mu"]), np.asarray(z["log_sigma"])
    if mu.ndim != 2 or ls.shape != mu.shape or not np.isfinite(mu).all() or not np.isfinite(ls).all():
        raise ValueError("belief handoff mu/log_sigma shape or finiteness check failed")
    return {"path": str(path), "sha256": got, "fields": sorted(names), "n_rows": int(len(mu)),
            "latent_dim": int(mu.shape[1]), "finite": True}


def _event_rows(manifest: dict, summaries: dict[str, torch.Tensor], device: str,
                release_by_episode: dict[str, str] | None = None) -> dict[str, list[dict]]:
    attempts = manifest["attempts"]
    order_cols = [c for c in ("learner", "timestamp", "t", "code_state_id") if c in attempts.columns]
    attempts = attempts.sort_values(order_cols, kind="mergesort")
    first = attempts.groupby("episode_id", sort=False).first()
    codec = manifest["problem_codec"]
    pid = manifest["pid_to_idx"]
    out: dict[str, list[dict]] = {}
    for _, row in manifest["episodes"].sort_values(["learner", "start_timestamp"], kind="mergesort").iterrows():
        # ``build_episode_level_table`` preserves the raw integer episode id, while
        # ``build_attempt_batch`` returns ids as integer keys.  The handoff's value-join
        # uses their canonical string form.  Keep the join by value, but do not let a
        # dtype-only mismatch silently erase every scenario (the raw-data path caught
        # this because the derived-predictions path already carries string ids).
        eid_raw = row.episode_id
        eid = str(eid_raw)
        if eid_raw not in first.index:
            continue
        if eid_raw not in summaries and eid not in summaries:
            continue
        summary = summaries.get(eid_raw, summaries.get(eid))
        a = first.loc[eid_raw]
        item_id = int(row.problem_id)
        out.setdefault(str(row.learner), []).append({
            "order": int(len(out.get(str(row.learner), []))),
            "episode_id": eid,
            "problem_id": item_id,
            "release": str((release_by_episode or {}).get(eid, a.get("release", "pooled"))),
            "codec": torch.as_tensor(codec[pid[item_id]], dtype=torch.float32, device=device),
            "summary": summary,
            "label": int(a.outcome_code),
        })
    return out


def _make_scenarios(manifest: dict, summaries: dict[str, torch.Tensor], device: str,
                    max_train: int = 0, max_val: int = 0,
                    release_by_episode: dict[str, str] | None = None) -> dict[str, list[dict]]:
    by_learner = _event_rows(manifest, summaries, device, release_by_episode)
    train = set(map(str, manifest.get("train_learners", set())))
    val = set(map(str, manifest.get("val_learners", set())))
    scenarios: dict[str, list[dict]] = {}
    for split, learners, limit in (("train", train, max_train), ("val", val, max_val)):
        for start in STARTS:
            key = f"{split}_start{start}"
            rows = []
            for learner in sorted(learners):
                all_events = by_learner.get(learner, [])
                if len(all_events) < start + 2:
                    continue
                target_n = min(5, len(all_events) - start - 1)
                targets = all_events[-target_n:]
                middle = all_events[start:-target_n]
                if not middle:
                    continue
                candidates = middle[:50]
                rows.append({
                    "learner": learner,
                    "semester": str(targets[0].get("release", "pooled")),
                    "prefix_events": all_events[:start],
                    "full_events": all_events[:-target_n],
                    "candidates": candidates,
                    "targets": targets,
                    "menu_size_uncapped": len(middle),
                    "menu_size": len(candidates),
                })
                if limit and len(rows) >= limit:
                    break
            scenarios[key] = rows
    return scenarios


def menu_distribution(scenarios: dict[str, list[dict]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, eps in scenarios.items():
        vals = np.asarray([e["menu_size_uncapped"] for e in eps], dtype=float)
        capped = np.asarray([e["menu_size"] for e in eps], dtype=float)
        if len(vals):
            q = np.quantile(vals, [0, .1, .25, .5, .75, .9, 1]).tolist()
            qc = np.quantile(capped, [0, .1, .25, .5, .75, .9, 1]).tolist()
        else:
            q = qc = []
        out[key] = {"n_learners": len(eps), "uncapped_quantiles_min_p10_p25_median_p75_p90_max": q,
                    "capped_at_50_quantiles": qc,
                    "semester_counts": dict(Counter(e["semester"] for e in eps))}
    return out


def _point_logits(model: dict, mu: torch.Tensor, items: torch.Tensor, device: str) -> np.ndarray:
    with torch.no_grad():
        if items.ndim == 1:
            items = items.unsqueeze(0)
        u = mu.to(device).expand(len(items), -1)
        hidden = int(model["checkpoint_meta"].get("architecture", {}).get("attempt_hidden", 64))
        # The CodeWorkout classifier's outcome head carries a prefix projection.  Planner
        # queries have no within-episode attempt prefix, so a neutral zero vector is
        # supplied.
        prefix = torch.zeros((len(items), hidden), dtype=torch.float32, device=device)
        logits = model["outcome_head"](items.to(device), u, prefix=prefix)
    return logits.detach().cpu().numpy().astype(np.float64)


def _belief_logits(model: dict, events: list[dict], query: torch.Tensor,
                   probe: torch.Tensor, device: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu, ls = _filter_state(model, sorted(events, key=lambda x: x["order"]), query, device)
    logits = _point_logits(model, mu, probe, device)
    probs = torch.softmax(torch.from_numpy(logits), dim=-1).numpy()
    return mu, logits, probs


def run_preflight(model: dict, scenarios: dict[str, list[dict]], device: str, out_path: pathlib.Path,
                  seed: int = SEED) -> dict[str, Any]:
    """Execute V1--V4 on the CW val cohort before any actor training."""
    eps_all = scenarios.get("val_start0", [])
    if len(eps_all) < 80:
        raise RuntimeError(f"CW pre-flight requires >=80 val learners, got {len(eps_all)}")
    # Fixed probe panel: the target items of the lexicographically first validation
    # learner.  The panel is fixed before any policy score is computed and is identical
    # for pooled and semester-stratified tests.
    probe = torch.stack([x["codec"] for x in eps_all[0]["targets"]])
    own_close: list[float] = []
    placebo_close: list[float] = []
    logit_gap: list[float] = []
    pred_gap: list[float] = []
    drift_full: list[float] = []
    drift_trim: list[float] = []
    rng = np.random.default_rng(seed)
    per_sem: dict[str, dict[str, list[float]]] = {}
    for i, ep in enumerate(eps_all):
        query = ep["targets"][0]["codec"]
        _, lf, _ = _belief_logits(model, ep["full_events"], query, probe, device)
        _, l0, _ = _belief_logits(model, ep["prefix_events"], query, probe, device)
        trim_n = max(1, len(ep["full_events"]) - max(1, len(ep["full_events"]) // 4))
        _, ltrim, _ = _belief_logits(model, ep["full_events"][:trim_n], query, probe, device)
        d0 = _dlogit(l0, lf)
        other = eps_all[int(rng.integers(len(eps_all)))]
        own_s, placebo_s, lg_s, pg_s = [], [], [], []
        for k in (3, 5, 10):
            own_events = ep["prefix_events"] + ep["candidates"][:min(k, len(ep["candidates"]))]
            # Placebo is the other learner's own history, not this learner's labels.
            p_events = other["prefix_events"] + other["candidates"][:min(k, len(other["candidates"]))]
            _, lo, po = _belief_logits(model, own_events, query, probe, device)
            _, lp, pp = _belief_logits(model, p_events, query, probe, device)
            do = _dlogit(lo, lf); dp = _dlogit(lp, lf)
            pf = torch.softmax(torch.from_numpy(lf), dim=-1).numpy()
            own_s.append(d0 - do); placebo_s.append(d0 - dp)
            lg_s.append(do); pg_s.append(_kl(po, pf))
            own_close.append(d0 - do); placebo_close.append(d0 - dp)
            logit_gap.append(do); pred_gap.append(_kl(po, pf))
            drift_full.append(do); drift_trim.append(_dlogit(lo, ltrim))
        s = per_sem.setdefault(ep["semester"], {"own": [], "placebo": []})
        s["own"].extend(own_s); s["placebo"].extend(placebo_s)
    own_a, plc_a = np.asarray(own_close), np.asarray(placebo_close)
    ratio = float(np.median(plc_a) / max(np.median(own_a), 1e-9))
    shifted = lf + np.linspace(0.1, 1.0, len(lf))[:, None]
    gauge_center = _dlogit(lf, shifted)
    gauge_unc = abs(float(np.mean((lf - l0) ** 2) - np.mean((shifted - l0) ** 2)))
    v1 = {"median_own_closure": float(np.median(own_a)), "median_placebo_closure": float(np.median(plc_a)),
          "ratio": ratio, "kill_threshold_ratio": .90, "pass": bool(ratio < .90)}
    v2 = {"centered_shift_delta": gauge_center, "uncentered_shift_delta": gauge_unc,
          "centered_tolerance": 1e-8, "uncentered_minimum": 1e-6,
          "pass": bool(gauge_center <= 1e-8 and gauge_unc > 1e-6)}
    v3rho = _spearman(np.asarray(drift_full), np.asarray(drift_trim))
    v3 = {"spearman_prefix_gap_full_vs_trim": v3rho, "kill_threshold": .80,
          "pass": bool(np.isfinite(v3rho) and v3rho >= .80)}
    v4rho = _spearman(-np.asarray(logit_gap), -np.asarray(pred_gap))
    v4 = {"spearman_logit_vs_predictive_gap": v4rho, "kill_threshold": .20,
          "pass": bool(np.isfinite(v4rho) and v4rho >= .20)}
    out: dict[str, Any] = {"status": "complete", "metric": "centered_point_logit_mse",
                           "n_learners": len(eps_all), "k_values": [3, 5, 10],
                           "probe_items": int(len(probe)), "sampler_semantics": CURRENT_SAMPLER_SEMANTICS,
                           "V1_placebo": v1, "V2_gauge": v2, "V3_drift": v3, "V4_future_gap": v4,
                           "per_semester": {k: {"n": len(v["own"]),
                                                  "median_own_closure": float(np.median(v["own"])),
                                                  "median_placebo_closure": float(np.median(v["placebo"]))}
                                            for k, v in per_sem.items()}}
    out["all_pass"] = bool(v1["pass"] and v2["pass"] and v3["pass"] and v4["pass"])
    if not v1["pass"] or not v4["pass"]:
        out["fallback"] = "probability_space_predictive_KL"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(_jsonable(out), indent=2) + "\n")
    return out


class PointerActor(nn.Module):
    def __init__(self, codec_dim: int, state_dim: int = 8, hidden: int = 96):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(codec_dim + 2, hidden), nn.Tanh(), nn.Linear(hidden, hidden))
        self.state = nn.Sequential(nn.Linear(state_dim, hidden), nn.Tanh())
        self.out = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.Tanh(), nn.Linear(hidden, 1))

    def forward(self, state: torch.Tensor, codecs: torch.Tensor, difficulty: torch.Tensor) -> torch.Tensor:
        s = self.state(state).expand(codecs.shape[0], -1)
        p = difficulty.clamp(1e-6, 1 - 1e-6)
        ent = -(p * p.log() + (1 - p) * (1 - p).log()).unsqueeze(1)
        e = self.enc(torch.cat((codecs, p.unsqueeze(1), ent), dim=1))
        return self.out(torch.cat((s, e), dim=1)).squeeze(1)


class PrivilegedCritic(nn.Module):
    def __init__(self, state_dim: int = 8, hidden: int = 96):
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


def _difficulty(model: dict, mu: torch.Tensor, candidates: list[dict], device: str) -> np.ndarray:
    codecs = torch.stack([c["codec"] for c in candidates])
    logits = _point_logits(model, mu, codecs, device)
    return torch.softmax(torch.from_numpy(logits), dim=-1).numpy()[:, 3]  # CLEAN_FULL probability


def _candidate_scores(model: dict, mu: torch.Tensor, candidates: list[dict], targets: list[dict],
                      eps: torch.Tensor, device: str) -> dict[int, dict[str, float]]:
    target_codecs = torch.stack([t["codec"] for t in targets])
    out: dict[int, dict[str, float]] = {}
    for i, c in enumerate(candidates):
        pred = _predict(model, mu, torch.zeros_like(mu), c["codec"], target_codecs, eps, device)
        cp = pred["candidate"].mean(dim=0)
        out[i] = {"epig": float(sample_based_epig_categorical_categorical(pred["candidate"], pred["targets"]).mean().item()),
                  "entropy": float(categorical_entropy(cp).item()),
                  "difficulty": float(cp[3].item())}
    return out


def _trace_policy(model: dict, ep: dict, policy: str, budget: int, device: str,
                  seed: int, actor: PointerActor | None = None) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    known = list(ep["prefix_events"])
    candidates = list(ep["candidates"])
    targets = list(ep["targets"])
    query = targets[0]["codec"]
    target_codecs = torch.stack([t["codec"] for t in targets])
    eps = torch.randn(K_EVAL, model["checkpoint_meta"]["architecture"].get("latent_dim", 32),
                      generator=torch.Generator(device="cpu").manual_seed(seed)).to(device)
    full_mu, _ = _filter_state(model, ep["full_events"], query, device)
    full_logits = _point_logits(model, full_mu, target_codecs, device)
    full_probs = torch.softmax(torch.from_numpy(full_logits), dim=-1).numpy()
    labels = np.asarray([t["label"] for t in targets], dtype=int)
    traces = {"logloss": [], "dlogit": [], "pred_kl": [], "picks": [], "pick_problems": []}
    remaining = list(range(len(candidates)))
    for step in range(min(budget, len(candidates)) + 1):
        mu, _ = _filter_state(model, sorted(known, key=lambda x: x["order"]), query, device)
        logits = _point_logits(model, mu, target_codecs, device)
        probs = torch.softmax(torch.from_numpy(logits), dim=-1).numpy()
        traces["logloss"].append(float(-np.log(np.clip(probs[np.arange(len(labels)), labels], 1e-8, 1)).mean()))
        traces["dlogit"].append(_dlogit(logits, full_logits))
        traces["pred_kl"].append(_kl(probs, full_probs))
        if step >= budget or not remaining:
            break
        score = _candidate_scores(model, mu, [candidates[i] for i in remaining], targets, eps, device)
        if policy == "random":
            j = int(rng.integers(len(remaining)))
        elif policy == "greedy_epig":
            j = max(range(len(remaining)), key=lambda x: (score[x]["epig"], -x))
        elif policy == "max_entropy":
            j = max(range(len(remaining)), key=lambda x: (score[x]["entropy"], -x))
        elif policy == "difficulty_matched":
            vals = np.asarray([score[x]["difficulty"] for x in range(len(remaining))])
            j = int(np.argmin(np.abs(vals - np.median(vals))))
        elif policy == "privileged_oracle_upper_bound":
            # This is deliberately privileged: b_full is used only here to form the labelled
            # upper bound, never in actor features or critic input at deployment.
            vals = []
            for x in range(len(remaining)):
                ev2 = sorted(known + [candidates[remaining[x]]], key=lambda z: z["order"])
                mu2, _ = _filter_state(model, ev2, query, device)
                vals.append(_dlogit(_point_logits(model, mu2, target_codecs, device), full_logits))
            j = int(np.argmin(vals))
        elif policy == "rl_actor":
            if actor is None:
                raise ValueError("rl_actor requires trained actor")
            state = _state(mu, torch.zeros_like(mu), step, budget, len(remaining)).to(device)
            cc = torch.stack([candidates[i]["codec"] for i in remaining]).to(device)
            dd = torch.tensor([score[x]["difficulty"] for x in range(len(remaining))], dtype=torch.float32, device=device)
            with torch.no_grad():
                j = int(actor(state, cc, dd).argmax().item())
        else:
            raise ValueError(policy)
        chosen = remaining.pop(j)
        known.append(candidates[chosen])
        traces["picks"].append(candidates[chosen]["episode_id"])
        traces["pick_problems"].append(int(candidates[chosen]["problem_id"]))
    return traces


def _train_actor(model: dict, train_eps: list[dict], device: str, seed: int,
                 epochs: int, budget_choices: tuple[int, ...]) -> tuple[PointerActor, PrivilegedCritic, dict]:
    torch.manual_seed(seed)
    latent = int(model["checkpoint_meta"]["architecture"].get("latent_dim", 32))
    codec_dim = int(train_eps[0]["candidates"][0]["codec"].numel())
    actor, critic = PointerActor(codec_dim).to(device), PrivilegedCritic().to(device)
    opt = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=3e-4)
    rng = np.random.default_rng(seed)
    losses = []
    for _ in range(epochs):
        for idx in rng.permutation(len(train_eps)):
            ep = train_eps[int(idx)]
            budget = int(rng.choice(budget_choices))
            if not ep["candidates"]:
                continue
            query = ep["targets"][0]["codec"]
            full_mu, _ = _filter_state(model, ep["full_events"], query, device)
            full_state = _state(full_mu, torch.zeros_like(full_mu), 0, budget, len(ep["candidates"]))
            known = list(ep["prefix_events"]); remaining = list(range(len(ep["candidates"])))
            logps, vals, rewards = [], [], []
            for step in range(min(budget, len(remaining))):
                mu, ls = _filter_state(model, sorted(known, key=lambda x: x["order"]), query, device)
                state = _state(mu, ls, step, budget, len(remaining))
                cands = [ep["candidates"][i] for i in remaining]
                diff = torch.tensor(_difficulty(model, mu, cands, device), dtype=torch.float32, device=device)
                cc = torch.stack([c["codec"] for c in cands]).to(device)
                dist = torch.distributions.Categorical(logits=actor(state, cc, diff))
                j = dist.sample(); chosen_idx = remaining[int(j.item())]
                before = _dlogit(_point_logits(model, mu, torch.stack([t["codec"] for t in ep["targets"]]), device),
                                 _point_logits(model, full_mu, torch.stack([t["codec"] for t in ep["targets"]]), device))
                known.append(ep["candidates"][chosen_idx]); remaining.remove(chosen_idx)
                mu2, _ = _filter_state(model, sorted(known, key=lambda x: x["order"]), query, device)
                after = _dlogit(_point_logits(model, mu2, torch.stack([t["codec"] for t in ep["targets"]]), device),
                                _point_logits(model, full_mu, torch.stack([t["codec"] for t in ep["targets"]]), device))
                logps.append(dist.log_prob(j)); vals.append(critic(state, full_state)); rewards.append(before - after)
            if not rewards:
                continue
            returns, g = [], 0.0
            for r in rewards[::-1]:
                g += float(r); returns.append(g)
            returns = torch.tensor(returns[::-1], dtype=torch.float32, device=device)
            vals_t = torch.stack(vals); adv = returns - vals_t.detach()
            loss = -(torch.stack(logps) * adv).mean() + 0.5 * F.mse_loss(vals_t, returns)
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(list(actor.parameters()) + list(critic.parameters()), 1.0); opt.step()
            losses.append(float(loss.item()))
    return actor, critic, {"epochs": epochs, "episodes": len(train_eps) * epochs,
                           "loss_last": losses[-1] if losses else None,
                           "loss_mean": float(np.mean(losses)) if losses else None}


def run_suite(args: argparse.Namespace) -> dict[str, Any]:
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    out = REPO / args.out_dir; out.mkdir(parents=True, exist_ok=True)
    if not pathlib.Path(args.checkpoint).exists():
        raise FileNotFoundError(f"CodeWorkout classifier checkpoint not found: {args.checkpoint}")
    if not pathlib.Path(args.beliefs).exists():
        raise FileNotFoundError(f"CodeWorkout classifier belief handoff not found: {args.beliefs}")
    handoff = _verify_belief_handoff(pathlib.Path(args.beliefs), args.beliefs_sha256)
    manifest_args = argparse.Namespace(**vars(args))
    manifest = _build_manifest(manifest_args)
    model = _build_model(device, pathlib.Path(args.checkpoint))
    summaries = _episode_summaries(model, manifest, device)
    handoff_z = np.load(args.beliefs, allow_pickle=False)
    release_by_episode = ({str(e): str(r) for e, r in zip(handoff_z["episode_id"], handoff_z["release"])}
                          if "release" in handoff_z.files and "episode_id" in handoff_z.files else {})
    scenarios = _make_scenarios(manifest, summaries, device, args.train_learners, args.val_learners,
                                release_by_episode=release_by_episode)
    menu = menu_distribution(scenarios)
    (out / "menu_distribution.json").write_text(json.dumps(_jsonable(menu), indent=2) + "\n")
    preflight = run_preflight(model, scenarios, device, out / "preflight.json", args.seed)
    if not preflight["V2_gauge"]["pass"]:
        raise RuntimeError("CW planner pre-flight V2 gauge implementation failure")
    if args.preflight_only:
        return {"status": "preflight_complete", "device": device, "menu_distribution": menu,
                "preflight": preflight, "belief_handoff": handoff,
                "checkpoint_sha256": _sha256(pathlib.Path(args.checkpoint))}
    actor, critic, train_meta = _train_actor(model, scenarios["train_start0"], device, args.seed,
                                              args.epochs, tuple((3, 5, 8) if args.pilot else BUDGETS))
    torch.save({"actor": actor.state_dict(), "critic": critic.state_dict(), "train": train_meta,
                "deployable_inputs": ["belief_summary", "remaining_menu_codecs", "difficulty", "step", "budget"],
                "reference_used_at_deployment": False}, out / "policy.pt")
    budgets = (3, 5, 8) if args.pilot else BUDGETS
    summary: dict[str, Any] = {"status": "running", "device": device, "budgets": list(budgets),
                               "policies": list(POLICIES), "starts": list(STARTS),
                               "menu_distribution": menu, "preflight": preflight,
                               "train": train_meta, "belief_handoff": handoff,
                               "checkpoint_sha256": _sha256(pathlib.Path(args.checkpoint)), "cells": {}}
    for start in STARTS:
        eps = scenarios.get(f"val_start{start}", [])
        for policy in POLICIES:
            for budget in budgets:
                rows = [_trace_policy(model, ep, policy, budget, device,
                                       args.seed + 10000 + start * 1000 + i, actor=actor)
                        for i, ep in enumerate(eps)]
                if not rows:
                    continue
                # Menus are intentionally replay-constrained and have different
                # lengths.  Each learner contributes its terminal value at
                # min(B, |menu|), while the per-step traces remain available in
                # the row object and the pick list is persisted below.
                ll = np.asarray([r["logloss"][-1] for r in rows], dtype=float)
                dl = np.asarray([r["dlogit"][-1] for r in rows], dtype=float)
                pk = np.asarray([r["pred_kl"][-1] for r in rows], dtype=float)
                trace_len = max(len(r["logloss"]) for r in rows)
                ll_trace = np.full((len(rows), trace_len), np.nan, dtype=float)
                dl_trace = np.full_like(ll_trace, np.nan)
                pk_trace = np.full_like(ll_trace, np.nan)
                for ri, row in enumerate(rows):
                    n = len(row["logloss"])
                    ll_trace[ri, :n] = row["logloss"]
                    dl_trace[ri, :n] = row["dlogit"]
                    pk_trace[ri, :n] = row["pred_kl"]
                key = f"start{start}|{policy}|B{budget}"
                np.savez_compressed(out / f"partial_{key.replace('|', '_')}.npz",
                                    logloss=ll, dlogit=dl, pred_kl=pk,
                                    logloss_trace=ll_trace, dlogit_trace=dl_trace,
                                    pred_kl_trace=pk_trace,
                                    picks=np.asarray([r["picks"] for r in rows], dtype=object))
                final = {"logloss": _bootstrap_mean(ll, args.seed + 20 + budget),
                         "dlogit": _bootstrap_mean(dl, args.seed + 30 + budget),
                         "pred_kl": _bootstrap_mean(pk, args.seed + 40 + budget),
                         "n_learners": len(rows),
                         "semester_counts": dict(Counter(e["semester"] for e in eps))}
                summary["cells"][key] = final
                (out / "progress.json").write_text(json.dumps({"status": "running", "cell": key,
                                                                  "n": len(rows)}) + "\n")
    # Menu-overlap-controlled pick divergence is a descriptive post-pass over saved rows.
    div: dict[str, Any] = {}
    for start in STARTS:
        eps = scenarios.get(f"val_start{start}", [])
        menu_sets = [set(e["problem_id"] for e in ep["candidates"]) for ep in eps]
        for policy in POLICIES:
            for budget in budgets:
                p = out / f"partial_start{start}_{policy}_B{budget}.npz"
                if not p.exists(): continue
                z = np.load(p, allow_pickle=True); picks = z["picks"]
                vals, overlaps = [], []
                for i in range(len(picks)):
                    for j in range(i):
                        a, b = set(picks[i]), set(picks[j]); u = a | b
                        vals.append(1.0 - len(a & b) / len(u) if u else 0.0)
                        m = menu_sets[i] & menu_sets[j]; overlaps.append(len(m) / max(1, len(menu_sets[i] | menu_sets[j])))
                div[f"start{start}|{policy}|B{budget}"] = {
                    "pick_jaccard_divergence_mean": float(np.mean(vals)) if vals else float("nan"),
                    "menu_problem_overlap_mean": float(np.mean(overlaps)) if overlaps else float("nan"),
                    "n_pairs": len(vals),
                }
    summary["pick_divergence_menu_control"] = div
    summary["status"] = "complete"
    (out / "summary.json").write_text(json.dumps(_jsonable(summary), indent=2) + "\n")
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    for name in ("code_map", "diff_map", "codealone_meta", "codealone_states", "diff_meta", "diff_states"):
        ap.add_argument("--" + name.replace("_", "-"), required=True)
    ap.add_argument("--raw-dir", default=None)
    ap.add_argument("--derived-val-predictions", default="")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--beliefs", required=True)
    ap.add_argument("--beliefs-sha256", default="")
    ap.add_argument("--out-dir", default="results/cw_planner_suite")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--train-learners", type=int, default=200)
    ap.add_argument("--val-learners", type=int, default=110)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--preflight-only", action="store_true",
                    help="build the frozen CW manifest and run V1--V4, then stop before actor training")
    args = ap.parse_args()
    if args.pilot:
        # Keep the full validation cohort for the mandatory pre-flight and pilot
        # evaluation; only the actor-training side is reduced for feasibility.
        # A 20-learner validation set would fail the pre-flight's minimum of 80
        # learners before the pilot could test anything (the pre-flight uses n=110).
        args.train_learners = min(args.train_learners, 20); args.epochs = 3
    run_suite(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
