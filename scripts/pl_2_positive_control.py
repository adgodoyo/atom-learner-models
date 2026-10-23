#!/usr/bin/env python
"""Planted-world positive control for the planner (§4.5): greedy EPIG must beat random in a world
built so that winning is the right answer, otherwise the implementation is broken.

Why it is needed. A negative planner result on real data is only interpretable if the
implementation has been shown capable of winning where the correct answer is known.

The design, and why each part is load-bearing. A planted latent theta ~ N(0, I_2) per learner;
items answered with probability sigmoid(a.theta + b). The candidate pool holds three families in
equal thirds, and two of them carry exactly zero information about theta:

  * DIAGNOSTIC (||a|| = 2)  -- informative; the correct answer key.
  * TRAP (a = 0, b = 0)     -- p == 0.5 for every learner: maximum predictive entropy, zero
                               information. A raw-uncertainty (BALD-flavour) score ranks these
                               top; a decision-aware score must score them 0.
  * EASY (a = 0, b = 2.2)   -- p == 0.900 for every learner: low entropy, zero information.
                               Present so "avoid the trap" cannot be passed by simply
                               preferring low-entropy items.

The trap family is the point: it is the coin flip everyone is equally unsure about (the
aleatoric trap), and here it is known which items are which, so the claim is checkable.

Everything runs through the production code path -- `evsi.run_episode`,
`evsi.candidate_evsi_scores`, `replay.predict_on_targets` -- with no control-specific branch
inside them. Only the belief and the likelihood are swapped, through the same `filter_model` /
`heads` duck-typed interface a real checkpoint presents, so a pass is evidence about the code
that runs on real data.

Two arms, and the contrast between them is the diagnosis:
  ARM 1 ORACLE    -- exact-Bayes posterior (grid), moment-matched to the diagonal Gaussian
                     `sample_u` requires. Bayes-consistent by construction. Answers: are the
                     estimator and harness correct?
  ARM 2 AMORTISED -- the real `S1Filter` + `FiLMHeads`, trained on this same world, then run
                     through the identical harness with identical pools and CRN. Answers: do
                     K draws from an amortised Gaussian belief carry real epistemic information
                     about the person? Arm 1 passing and arm 2 failing would localise a
                     real-data negative result to the belief rather than the estimator.

The gate is pre-registered in `configs/pl_2_positive_control.yaml`, frozen before the run.
Output: `reports/pl_2_positive_control.md` (`_pilot` suffix with --pilot).
Run: `python scripts/pl_2_positive_control.py [--pilot]` (CPU).
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import argparse  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

from atom.eval.metrics import bootstrap_mean_ci  # noqa: E402
from atom.eval.report import gate_report  # noqa: E402
from atom.filter.episode import build_prev_feat  # noqa: E402
from atom.filter.s1 import S1Filter  # noqa: E402
from atom.heads.s1_heads import FiLMHeads  # noqa: E402
from atom.planner.evsi import run_episode  # noqa: E402
from atom.planner.replay import predict_on_targets  # noqa: E402
from atom.planner.synthetic import (  # noqa: E402
    FAMILIES,
    OracleFilter,
    OracleHeads,
    assert_query_independent,
    build_grid,
    grid_exact_predictive,
    make_item_bank,
    true_prob,
)
from atom.utils.logging import RunLogger  # noqa: E402
from atom.utils.seeds import seed_everything  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[1]
DEVICE = "cpu"  # tiny tensors; MPS transfer overhead dominates and CPU keeps this reproducible


def build_world(cfg: dict, rng: np.random.Generator):
    """Item banks. Candidate items are family-mixed; prefix and target items are drawn from
    a SEPARATE diagnostic bank, so targets are always reducible-uncertainty items and no
    candidate is ever also a target (menu/target overlap = 0 by construction; replay
    geometry is always reported)."""
    w, ep = cfg["world"], cfg["episode"]
    cand_bank, cand_family = make_item_bank(
        n_diagnostic=ep["n_cand_diagnostic"], n_trap=ep["n_cand_trap"], n_easy=ep["n_cand_easy"],
        codec_dim=w["codec_dim"], diag_loading=w["diag_loading"], easy_b=w["easy_b"],
        nuisance_std=w["nuisance_std"], rng=rng)
    # Prefix + target bank: diagnostic only.
    n_pt = ep["prefix_n"] + ep["n_target"]
    pt_bank, _ = make_item_bank(n_diagnostic=n_pt, n_trap=0, n_easy=0, codec_dim=w["codec_dim"],
                                diag_loading=w["diag_loading"], easy_b=w["easy_b"],
                                nuisance_std=w["nuisance_std"], rng=rng)
    return cand_bank, cand_family, pt_bank


def make_episode(cfg: dict, cand_bank, pt_bank, rng: np.random.Generator):
    """One learner: planted theta, then FROZEN logged outcomes for every item they will
    ever be asked or scored on (answers come from the environment, once)."""
    ep = cfg["episode"]
    theta = rng.normal(size=cfg["world"]["latent_dim"])
    prefix_items = np.arange(ep["prefix_n"])
    target_items = np.arange(ep["prefix_n"], ep["prefix_n"] + ep["n_target"])
    cand_items = np.arange(cand_bank.shape[0])

    p_pt = true_prob(pt_bank, theta)
    y_pt = (rng.random(pt_bank.shape[0]) < p_pt).astype(np.float32)
    p_c = true_prob(cand_bank, theta)
    y_c = (rng.random(cand_bank.shape[0]) < p_c).astype(np.float32)

    return {
        "theta": theta,
        "prefix_codec": torch.tensor(pt_bank[prefix_items], dtype=torch.float32),
        "prefix_y": torch.tensor(y_pt[prefix_items]),
        "cand_codec": torch.tensor(cand_bank[cand_items], dtype=torch.float32),
        "cand_y": torch.tensor(y_c[cand_items]),
        "target_codec": torch.tensor(pt_bank[target_items], dtype=torch.float32),
        "target_y": torch.tensor(y_pt[target_items]),
        "target_p_true": p_pt[target_items],
    }


def run_arm(cfg: dict, filter_model, heads, episodes: list[dict], cand_family: np.ndarray,
            logger: RunLogger, arm: str) -> dict:
    """Every policy on every learner, sharing the IDENTICAL pool, targets and CRN noise
    (every arm shares its subsetting/menu, and random is always present to price the
    harness itself)."""
    ep = cfg["episode"]
    k, budget = ep["k"], ep["budget"]
    policies = list(ep["policies"])
    n_cand = cand_family.shape[0]

    traces = {p: [] for p in policies}
    picks = {p: [] for p in policies}
    t0 = time.time()

    for i, e in enumerate(episodes):
        gen = torch.Generator().manual_seed(1000 + i)
        noise = torch.randn(k, filter_model.latent_dim if hasattr(filter_model, "latent_dim")
                            else cfg["world"]["latent_dim"], generator=gen)
        prefix = {"item_codec": e["prefix_codec"], "correct": e["prefix_y"],
                  "hint_bin": torch.zeros_like(e["prefix_y"]),
                  "saw_answer": torch.zeros_like(e["prefix_y"])}
        cand = {"item_codec": e["cand_codec"], "correct": e["cand_y"],
                "hint_bin": torch.zeros_like(e["cand_y"]),
                "saw_answer": torch.zeros_like(e["cand_y"])}

        for p in policies:
            rng = np.random.default_rng(5000 + i)  # same draw for every policy that needs one
            pick_log: list[int] = []
            trace, _ = run_episode(filter_model, heads, prefix, cand, e["target_codec"],
                                   e["target_y"], k, DEVICE, p, budget, rng=rng, noise=noise,
                                   pick_log=pick_log, v_flavor=ep["v_flavor"])
            traces[p].append(trace)
            picks[p].append(pick_log)

        if (i + 1) % 10 == 0 or i == 0:
            rate = (time.time() - t0) / (i + 1)
            logger.heartbeat(step=i + 1, force=True,
                             stage=f"{arm}: {i+1}/{len(episodes)} learners, {rate:.2f}s each")

    out = {"policies": policies, "wall_s": time.time() - t0}
    for p in policies:
        ll = np.array([tr[-1]["logloss"] for tr in traces[p]])
        acc = np.array([tr[-1]["accuracy"] for tr in traces[p]])
        ll0 = np.array([tr[0]["logloss"] for tr in traces[p]])
        v_curve = np.array([[s["V"] for s in tr] for tr in traces[p]])
        ll_curve = np.array([[s["logloss"] for s in tr] for tr in traces[p]])
        flat = np.array([f for pl in picks[p] for f in pl], dtype=int)
        fam = cand_family[flat] if flat.size else np.array([], dtype=int)
        out[p] = {
            "logloss_final": float(ll.mean()), "logloss_start": float(ll0.mean()),
            "accuracy_final": float(acc.mean()),
            "logloss_per_learner": ll, "logloss_curve": ll_curve.mean(axis=0),
            "v_curve": v_curve.mean(axis=0),
            "pick_family_counts": {FAMILIES[f]: int((fam == f).sum()) for f in range(len(FAMILIES))},
            "n_picks": int(fam.size),
            "diagnostic_pick_rate": float((fam == 0).mean()) if fam.size else float("nan"),
            "trap_pick_rate": float((fam == 1).mean()) if fam.size else float("nan"),
            "easy_pick_rate": float((fam == 2).mean()) if fam.size else float("nan"),
        }
    out["pool_base_rate_diagnostic"] = float((cand_family == 0).mean())
    out["n_cand"] = n_cand
    return out


def paired_delta(a: np.ndarray, b: np.ndarray, cfg: dict, seed: int) -> tuple[float, float, float]:
    """base_ll - policy_ll, POSITIVE = policy better (the sign convention of the real-data
    replay runs, kept so they read the same way)."""
    g = cfg["gate"]
    return bootstrap_mean_ci(b - a, g["bootstrap_n"], g["ci_alpha"], seed)


def _stream_batch(bank: np.ndarray, n: int, seq_len: int, latent_dim: int,
                  rng: np.random.Generator):
    """n NATURAL chronological streams from the planted world: random items, outcomes drawn
    from the true model. This is the distribution a filter really sees -- never
    planner-chosen subsets -- so the subsetting shift between training streams and
    planner-chosen subsets is present in arm 2 exactly as it is on FA, rather than being
    trained away."""
    thetas = rng.normal(size=(n, latent_dim))
    item_idx = rng.integers(0, bank.shape[0], size=(n, seq_len))
    items = bank[item_idx]
    codec = torch.tensor(items, dtype=torch.float32)
    p = 1.0 / (1.0 + np.exp(-(np.einsum("ntl,nl->nt", items[..., 0:2], thetas) + items[..., 2])))
    y = torch.tensor((rng.random(p.shape) < p).astype(np.float32))
    return codec, y, torch.tensor(thetas, dtype=torch.float32), torch.tensor(p, dtype=torch.float32)


def train_amortised(cfg: dict, cand_bank, pt_bank, rng: np.random.Generator, logger: RunLogger,
                    n_train: int = 8000, n_val: int = 1500, epochs: int = 400, batch: int = 512):
    """ARM 2: train the REAL `S1Filter` + `FiLMHeads` on this same planted world.

    **Trained to convergence on purpose, and the convergence is MEASURED.** Arm 2's whole
    value is that a null here means "an amortised belief does not carry the signal", not "we
    undertrained it" -- an undertrained model would manufacture a spurious negative
    result. Three guards, all reported:

      * a held-out VAL BCE, not the train BCE (which can fall while the belief stays flat);
      * the ORACLE's own BCE on that identical val set -- the ACHIEVABLE FLOOR. "How far
        short does the amortised model fall" then has a denominator instead of being an
        opinion, and if it sits at the floor the training question is settled;
      * BELIEF-vs-THETA recovery: R^2 of the planted theta regressed on the amortised mu at
        the end of each stream. This is a semantic check (assert that the quantity is what
        it claims to be) -- a belief that does not track the latent cannot
        carry epistemic information about the person no matter what its loss says, and a
        loss-only check cannot tell those two states apart.
    """
    w, ep = cfg["world"], cfg["episode"]
    seq_len = ep["prefix_n"] + ep["budget"] + 4
    bank = np.concatenate([cand_bank, pt_bank], axis=0)
    ld = w["latent_dim"]

    codec, y, _, _ = _stream_batch(bank, n_train, seq_len, ld, rng)
    v_codec, v_y, v_theta, v_p = _stream_batch(bank, n_val, seq_len, ld, rng)

    filter_model = S1Filter(codec_dim=w["codec_dim"], episode_hidden=64, filter_hidden=64,
                            latent_dim=ld)
    heads = FiLMHeads(codec_dim=w["codec_dim"], latent_dim=ld, hidden=64)
    opt = torch.optim.Adam(list(filter_model.parameters()) + list(heads.parameters()), lr=3e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    bce = torch.nn.BCEWithLogitsLoss()

    prev_feat = build_prev_feat(y, torch.zeros_like(y), torch.zeros_like(y))
    v_prev = build_prev_feat(v_y, torch.zeros_like(v_y), torch.zeros_like(v_y))

    def val_bce() -> float:
        with torch.no_grad():
            mu, ls = filter_model(v_codec, v_prev)
            u = mu.unsqueeze(0) + ls.exp().unsqueeze(0) * torch.randn(8, *mu.shape)
            logits, _ = heads(v_codec, u)
            p = torch.sigmoid(logits).mean(0).clamp(1e-6, 1 - 1e-6)
            return float(-(v_y * p.log() + (1 - v_y) * (1 - p).log()).mean().item())

    best, best_state, since = float("inf"), None, 0
    for epoch in range(epochs):
        perm = torch.randperm(n_train)
        for s in range(0, n_train, batch):
            idx = perm[s:s + batch]
            opt.zero_grad()
            mu, log_sigma = filter_model(codec[idx], prev_feat[idx])
            u = mu.unsqueeze(0) + log_sigma.exp().unsqueeze(0) * torch.randn(1, *mu.shape)
            logits, _ = heads(codec[idx], u)
            loss = bce(logits[0], y[idx])
            kl = 0.5 * (mu ** 2 + (2 * log_sigma).exp() - 1 - 2 * log_sigma).sum(-1).mean()
            (loss + 0.01 * kl).backward()
            opt.step()
        sched.step()
        if (epoch + 1) % 10 == 0:
            vb = val_bce()
            if vb < best - 1e-4:
                best, since = vb, 0
                best_state = ({k: v.clone() for k, v in filter_model.state_dict().items()},
                              {k: v.clone() for k, v in heads.state_dict().items()})
            else:
                since += 1
            logger.heartbeat(step=epoch + 1, force=True,
                             stage=f"amortised: epoch {epoch+1}/{epochs} val_bce={vb:.4f} best={best:.4f}")
            if since >= 6:          # ~60 epochs with no val improvement
                break

    if best_state is not None:
        filter_model.load_state_dict(best_state[0])
        heads.load_state_dict(best_state[1])
    filter_model.eval()
    heads.eval()
    filter_model.latent_dim = ld

    # --- convergence + belief-quality diagnostics (all reported, none optional) ----------
    with torch.no_grad():
        mu, _ = filter_model(v_codec, v_prev)
        mu_end = mu[:, -1]                                   # belief after the full stream
    # R^2 of planted theta on the amortised belief (linear, with intercept).
    x = torch.cat([mu_end, torch.ones(mu_end.shape[0], 1)], dim=1).double()
    tt = v_theta.double()
    coef = torch.linalg.lstsq(x, tt).solution
    resid = tt - x @ coef
    r2 = float(1.0 - (resid.var(dim=0).sum() / tt.var(dim=0).sum()).item())

    # The achievable floor: the ORACLE's own BCE on the identical val streams.
    grid, log_prior = build_grid(w["grid_half_width"], w["grid_n_per_dim"])
    oracle = OracleFilter(grid, log_prior)
    with torch.no_grad():
        o_w = oracle.posterior_weights(v_codec, v_prev)       # [N, T, G]
        # The ORACLE's own belief-vs-theta R^2 on the identical streams. Without this the
        # amortised R^2 has no denominator: after only ~20 events the EXACT posterior mean
        # is itself far from theta, so a low number may be the world's irreducible
        # uncertainty rather than a defect in the amortised belief.
        o_mu_end = torch.einsum("ng,gl->nl", o_w[:, -1], oracle.grid)
        z = torch.einsum("ntl,gl->ntg", v_codec[..., 0:2].double(), oracle.grid) \
            + v_codec[..., 2].double().unsqueeze(-1)
        o_p = (o_w * torch.sigmoid(z)).sum(-1).clamp(1e-6, 1 - 1e-6)
        vy = v_y.double()
        oracle_bce = float(-(vy * o_p.log() + (1 - vy) * (1 - o_p).log()).mean().item())
        # Bayes-irreducible floor: score the TRUE probabilities themselves.
        tp = v_p.double().clamp(1e-6, 1 - 1e-6)
        bayes_bce = float(-(vy * tp.log() + (1 - vy) * (1 - tp).log()).mean().item())

    xo = torch.cat([o_mu_end, torch.ones(o_mu_end.shape[0], 1, dtype=o_mu_end.dtype)], dim=1)
    coef_o = torch.linalg.lstsq(xo, tt).solution
    r2_oracle = float(1.0 - ((tt - xo @ coef_o).var(dim=0).sum() / tt.var(dim=0).sum()).item())

    return filter_model, heads, {
        "val_bce": val_bce(), "oracle_val_bce": oracle_bce, "true_model_val_bce": bayes_bce,
        "belief_theta_r2": r2, "oracle_belief_theta_r2": r2_oracle, "epochs_run": epoch + 1,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true", help="5 learners, timing only")
    ap.add_argument("--skip-amortised", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load((REPO / "configs" / "pl_2_positive_control.yaml").read_text())
    if args.pilot:
        cfg["episode"]["n_learners"] = cfg["pilot"]["n_learners"]

    seed_everything(cfg["seed"])
    logger = RunLogger(config={**cfg, "pilot": bool(args.pilot)}, seed=cfg["seed"],
                       data_version="synthetic-planted-latents-PL2")
    logger.heartbeat(step=0, force=True, stage="building world")

    rng = np.random.default_rng(cfg["seed"])
    cand_bank, cand_family, pt_bank = build_world(cfg, rng)
    episodes = [make_episode(cfg, cand_bank, pt_bank, rng)
                for _ in range(cfg["episode"]["n_learners"])]

    grid, log_prior = build_grid(cfg["world"]["grid_half_width"], cfg["world"]["grid_n_per_dim"])
    oracle, oracle_heads = OracleFilter(grid, log_prior), OracleHeads()
    oracle.latent_dim = cfg["world"]["latent_dim"]

    # --- Validity canaries, BEFORE any result is computed -------------------------------
    e0 = episodes[0]
    t = cfg["episode"]["prefix_n"]
    ic = torch.cat([e0["prefix_codec"], e0["cand_codec"][:1]], dim=0).unsqueeze(0)
    cr = torch.cat([e0["prefix_y"], e0["cand_y"][:1]], dim=0).unsqueeze(0)
    z = torch.zeros_like(cr)
    peek = assert_query_independent(oracle, ic, build_prev_feat(cr, z, z), t, e0["cand_codec"][5])

    exact = grid_exact_predictive(oracle, e0["prefix_codec"], e0["prefix_y"], e0["target_codec"])
    gen = torch.Generator().manual_seed(1000)
    noise0 = torch.randn(cfg["episode"]["k"], cfg["world"]["latent_dim"], generator=gen)
    approx = predict_on_targets(oracle, oracle_heads, e0["prefix_codec"], e0["prefix_y"],
                                torch.zeros_like(e0["prefix_y"]), torch.zeros_like(e0["prefix_y"]),
                                e0["target_codec"], cfg["episode"]["k"], DEVICE, noise=noise0)
    mm_dev = float((approx["mean_prob"] - exact.float()).abs().max().item())

    # Grid convergence: the exact posterior must not depend on grid coarseness.
    g2, lp2 = build_grid(cfg["world"]["grid_half_width"], 81)
    fine = OracleFilter(g2, lp2)
    exact_fine = grid_exact_predictive(fine, e0["prefix_codec"], e0["prefix_y"], e0["target_codec"])
    grid_dev = float((exact - exact_fine).abs().max().item())

    logger.heartbeat(step=0, force=True,
                     stage=f"canaries: peek={peek:.2e} moment_match={mm_dev:.4f} grid={grid_dev:.2e}")

    # --- ARM 1: oracle belief ------------------------------------------------------------
    oracle_res = run_arm(cfg, oracle, oracle_heads, episodes, cand_family, logger, "oracle")

    # --- ARM 2: amortised belief ---------------------------------------------------------
    amort_res, amort_diag = None, None
    if not args.skip_amortised:
        logger.heartbeat(step=0, force=True, stage="training amortised filter (arm 2)")
        fm, hd, amort_diag = train_amortised(cfg, cand_bank, pt_bank,
                                            np.random.default_rng(cfg["seed"] + 77), logger)
        amort_res = run_arm(cfg, fm, hd, episodes, cand_family, logger, "amortised")

    # --- Gate ----------------------------------------------------------------------------
    g = cfg["gate"]
    metrics: dict = {
        "n_learners": cfg["episode"]["n_learners"],
        "pool size / base rate diagnostic": f"{oracle_res['n_cand']} / {oracle_res['pool_base_rate_diagnostic']:.3f}",
        "canary: oracle belief query-independence max|delta|": f"{peek:.3e}",
        "canary: moment-match max|dev| vs exact grid posterior": f"{mm_dev:.4f}",
        "canary: grid convergence 61 vs 81 max|dev|": f"{grid_dev:.3e}",
    }
    notes: list[str] = []

    def summarise(res: dict, tag: str) -> tuple[bool, bool]:
        d, lo, hi = paired_delta(res["greedy_evsi"]["logloss_per_learner"],
                                 res["random"]["logloss_per_learner"], cfg, seed=7)
        metrics[f"[{tag}] greedy log-loss (full budget)"] = f"{res['greedy_evsi']['logloss_final']:.4f}"
        metrics[f"[{tag}] random log-loss (full budget)"] = f"{res['random']['logloss_final']:.4f}"
        metrics[f"[{tag}] max_entropy log-loss"] = f"{res['max_entropy']['logloss_final']:.4f}"
        metrics[f"[{tag}] max_epistemic log-loss"] = f"{res['max_epistemic']['logloss_final']:.4f}"
        metrics[f"[{tag}] G1 greedy - random (paired, +=greedy better)"] = \
            f"{d:+.4f} [{lo:+.4f}, {hi:+.4f}]"
        metrics[f"[{tag}] G2 greedy diagnostic-pick rate"] = \
            f"{res['greedy_evsi']['diagnostic_pick_rate']:.3f} (pool {res['pool_base_rate_diagnostic']:.3f})"
        metrics[f"[{tag}] G3 max_entropy pick mix (diag/trap/easy)"] = \
            (f"{res['max_entropy']['diagnostic_pick_rate']:.3f}/"
             f"{res['max_entropy']['trap_pick_rate']:.3f}/{res['max_entropy']['easy_pick_rate']:.3f}")
        metrics[f"[{tag}] G3 max_epistemic pick mix (diag/trap/easy)"] = \
            (f"{res['max_epistemic']['diagnostic_pick_rate']:.3f}/"
             f"{res['max_epistemic']['trap_pick_rate']:.3f}/{res['max_epistemic']['easy_pick_rate']:.3f}")
        metrics[f"[{tag}] greedy pick mix (diag/trap/easy)"] = \
            (f"{res['greedy_evsi']['diagnostic_pick_rate']:.3f}/"
             f"{res['greedy_evsi']['trap_pick_rate']:.3f}/{res['greedy_evsi']['easy_pick_rate']:.3f}")
        metrics[f"[{tag}] greedy log-loss curve (0..budget)"] = \
            " -> ".join(f"{v:.4f}" for v in res["greedy_evsi"]["logloss_curve"])
        metrics[f"[{tag}] random log-loss curve (0..budget)"] = \
            " -> ".join(f"{v:.4f}" for v in res["random"]["logloss_curve"])
        metrics[f"[{tag}] wall (s)"] = f"{res['wall_s']:.1f}"
        g1 = (d >= g["min_logloss_improvement_vs_random"]) and (lo > 0 if g["require_ci_clear_of_zero"] else True)
        g2 = res["greedy_evsi"]["diagnostic_pick_rate"] >= g["min_diagnostic_pick_rate"]
        return g1, g2

    g1_o, g2_o = summarise(oracle_res, "ORACLE")
    passed = g1_o and g2_o
    if amort_res is not None:
        g1_a, g2_a = summarise(amort_res, "AMORTISED")
        metrics["[AMORTISED] val BCE (held out)"] = f"{amort_diag['val_bce']:.4f}"
        metrics["[AMORTISED] ORACLE val BCE on the same streams (achievable floor)"] = \
            f"{amort_diag['oracle_val_bce']:.4f}"
        metrics["[AMORTISED] true-model val BCE (Bayes-irreducible floor)"] = \
            f"{amort_diag['true_model_val_bce']:.4f}"
        metrics["[AMORTISED] belief-vs-planted-theta R^2 (canary: does mu track the person?)"] = \
            f"{amort_diag['belief_theta_r2']:.4f} (ORACLE on same streams: " \
            f"{amort_diag['oracle_belief_theta_r2']:.4f} -- the achievable ceiling)"
        metrics["[AMORTISED] epochs run (early-stopped on val)"] = amort_diag["epochs_run"]
        notes.append(
            f"ARM 2 (amortised) is a DIAGNOSTIC, not part of the positive-control gate: G1 "
            f"{'pass' if g1_a else 'fail'}, G2 {'pass' if g2_a else 'fail'}. The gate is arm 1, "
            "because the control's question is whether the ESTIMATOR and HARNESS are correct. Arm 2 "
            "asks the separate question -- whether K "
            "draws from an AMORTISED Gaussian belief carry real epistemic information about "
            "the person -- and its contrast with arm 1 is what localises any failure to the "
            "belief rather than the estimator.")

    notes += [
        "PRE-REGISTERED before the run in `configs/pl_2_positive_control.yaml`: G1 = greedy "
        f"beats random on target log-loss by >= {g['min_logloss_improvement_vs_random']} nats with "
        "the 95% paired-bootstrap CI clear of zero; G2 = greedy's diagnostic-pick rate >= "
        f"{g['min_diagnostic_pick_rate']} against a pool base rate of 1/3. G3 (max-entropy's pick "
        "mix) is REPORTED either way and is never a pass/fail criterion.",
        "Two of the three candidate families carry EXACTLY zero information about the planted "
        "latent (trap: p == 0.5 for every learner; easy: p == 0.900 for every learner), verified "
        "in `tests/test_planner_synthetic.py` at atol=0 for the trap. So 'greedy should pick "
        "diagnostic items' is a known-correct answer key, not a preference.",
        "Everything ran through the SHIPPED harness (`evsi.run_episode`, "
        "`evsi.candidate_evsi_scores`, `replay.predict_on_targets`) with no control-specific branch "
        "inside any of them -- only `filter_model`/`heads` were swapped, through the same "
        "duck-typed interface a real checkpoint presents. A pass is therefore evidence about the "
        "code that will run on FA.",
        "Every policy saw the IDENTICAL pool, targets and common-random-numbers draw per learner "
        "and `random` is present to price the harness itself. Menu/target "
        "item overlap is 0 by construction (candidates and targets come from disjoint banks).",
        "PHRASING CAUTION (as in pl_1_epig_estimator.py): "
        "'the estimator is mathematically sound' and 'the belief it is applied to carries a real "
        "signal' are different claims. Arm 1 establishes the first on a Bayes-consistent belief. "
        "Only arm 2 speaks to the second, and only in this synthetic world -- neither arm licenses "
        "any claim about FA, which needs the real-data replay.",
        f"Cost: $0 (Mac, CPU, {oracle_res['wall_s']:.0f}s for arm 1). No cloud, no MPS contention.",
    ]

    meta = {
        "run_id": logger.run_id, "git_sha": logger.git_sha, "config_hash": logger.config_hash,
        "device": DEVICE, "seed": cfg["seed"],
        "world": f"theta~N(0,I_2), sigmoid(a.theta+b), diag||a||={cfg['world']['diag_loading']}, "
                 f"trap p=0.5, easy p=sigmoid({cfg['world']['easy_b']})",
        "episode": f"prefix {cfg['episode']['prefix_n']} | pool {oracle_res['n_cand']} "
                   f"(10 diag/10 trap/10 easy) | targets {cfg['episode']['n_target']} | "
                   f"budget {cfg['episode']['budget']} | K {cfg['episode']['k']}",
        "v_flavor": cfg["episode"]["v_flavor"],
    }
    name = "pl_2_positive_control" + ("_pilot" if args.pilot else "")
    gate_desc = (
        "Positive control: greedy-EPIG must beat random with a "
        "known-correct model, else the implementation is broken. G1 (log-loss margin + CI) AND "
        "G2 (diagnostic-pick rate) must both hold on ARM 1 (oracle belief).")
    text = gate_report(REPO / "reports" / f"{name}.md", "PL-2 - Synthetic positive control (planted latents)",
                       meta, metrics, passed, gate_desc, notes)

    np.savez(REPO / "artifacts" / f"{name}_curves.npz",
             **{f"oracle_{p}_logloss": oracle_res[p]["logloss_per_learner"] for p in oracle_res["policies"]},
             **{f"oracle_{p}_curve": oracle_res[p]["logloss_curve"] for p in oracle_res["policies"]},
             **{f"oracle_{p}_v": oracle_res[p]["v_curve"] for p in oracle_res["policies"]},
             **({f"amort_{p}_logloss": amort_res[p]["logloss_per_learner"] for p in amort_res["policies"]}
                if amort_res else {}),
             **({f"amort_{p}_curve": amort_res[p]["logloss_curve"] for p in amort_res["policies"]}
                if amort_res else {}),
             cand_family=cand_family)

    print(text)
    print(f"\nVERDICT: {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
