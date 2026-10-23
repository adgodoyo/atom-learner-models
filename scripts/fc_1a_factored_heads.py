#!/usr/bin/env python
"""Factored outcome head: trains a help-mode head and a conditional-correctness head on frozen
filters and compares their product against the original 5-class outcome head.

Head-only training on two frozen checkpoints (FA-SIM3, the mathematics simulator, and the `br4_r3`
variant); the filter receives no gradient, and this is checked by comparing its parameters
byte-for-byte before and after training. Per checkpoint, two new heads (both instances of
`atom.heads.fa_hierarchical_heads.OutcomeHead`, which is n_classes-agnostic):
  - a 4-way help-mode head p(m|z,e), m in {none, hint, reveal, both}; target is the raw
    `support_code` column (0/1/2/3), valid on every row (never missing in the FA v2 manifest);
  - a binary conditional-correctness head p(c|m=none,z,e); target is `score`, valid only where
    support_code==0 (unaided) and the score is known. The masked loss means help-mode rows never
    reach this head's gradient, so hint/reveal mass cannot leak into the correctness estimate.

The scored 5-class prediction is the product, in OUTCOME_CLASSES' index order
(`atom.data.fa_outcome_label`):
    p(UNAIDED_CORRECT) = p(m=none) * p(c=correct|m=none)
    p(UNAIDED_WRONG)   = p(m=none) * p(c=wrong|m=none)
    p(HINT_ONLY)       = p(m=hint)
    p(REVEAL_ONLY)     = p(m=reveal)
    p(HINT_AND_REVEAL) = p(m=both)
so it is comparable one-for-one with the original head's softmax, and every downstream panel
(wrong-detection, coupling probe, standard comparison panel) runs unmodified on either variant.

Pre-registered decision rule: a factored variant is adopted iff its bottom-tercile wrong-detection
is not significantly worse than the original FA-SIM3 head's on both AUC definitions
(`auc_unaided_wrong`, `auc_not_unaided_correct`; learner-clustered bootstrap, n_boot=1000; the
upper CI bound of the difference must not sit entirely below zero) and its coupling
(`fa_coupling_probe`'s `top_bottom_ratio` median) stays above 1.2 (the original `br4_r3` head
measured 4.28).

Inputs: the two checkpoints in CHECKPOINTS, the FA v2 manifest and splits, the production codec.
Outputs: `results/fc_1a/{name}_factored.pt`, `{name}_factored_coupling.json`, `summary.json`.
Run: `python scripts/fc_1a_factored_heads.py` (CPU or GPU; the filter forward runs once per
checkpoint under no_grad).  Appendix A (factored head).
"""

from __future__ import annotations

import copy
import hashlib
import json
import pathlib
import subprocess
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "infra"))
sys.path.insert(0, str(REPO / "scripts"))

from atom.binding import reparameterize_from_log_sigma  # noqa: E402
from atom.data.fa_manifest import load_v2_events  # noqa: E402
from atom.data.fa_outcome_label import N_OUTCOME, OUTCOME_CLASSES, outcome_label  # noqa: E402
from atom.eval.metrics import multiclass_ece, multiclass_nll  # noqa: E402
from atom.filter.batch import build_padded_batch_v2, load_production_codec  # noqa: E402
from atom.filter.episode_v2 import build_prev_feat_v2  # noqa: E402
from atom.filter.item_encoder_mlp import S1FilterV2ItemMLP  # noqa: E402
from atom.heads.fa_hierarchical_heads import OutcomeHead  # noqa: E402
from atom.simulator.sampler_semantics import CURRENT_SAMPLER_SEMANTICS  # noqa: E402
from atom.utils.seeds import seed_everything  # noqa: E402
from fa_filter_common import B4_CONFIG, CODEC_PATH, chunked_filter_forward  # noqa: E402
from fa_bootstrap_utils import _boot_mean, _case  # noqa: E402
from wd_1_wrong_detection_panel import (  # noqa: E402
    LATE_T_LO,
    TIMES_EXACT,
    TRAIT_CEILING_CI,
    TRAIT_CEILING_MID,
    UNAIDED_CORRECT,
    UNAIDED_WRONG,
    _auc,
    _ece_per_class,
    _learner_clustered_bootstrap_auc_diff,
    _reliability_bins,
)

K = 32  # committee size for eval-time scoring, as in wd_1_wrong_detection_panel.py
K_COUPLING_DIRS, K_COUPLING_DRAWS = 16, 32  # matching fa_coupling_probe.run()'s defaults
TRAIN_STEPS = 6000
TRAIN_BATCH = 2048
TRAIN_LR = 1e-3
HEAD_HIDDEN = 2560  # matches the ORIGINAL heads' own capacity (ck["outcome_head"]["trunk.weight"].shape[0])
EVAL_CHUNK = 4000
SEED = 20260827

CHECKPOINTS = {
    "fa_sim3": (REPO / "artifacts/checkpoints/fa_sim3_final.pt", "direct"),
    "br4_r3": (REPO / "results/fa_br4_r3_full/final.pt", "direct"),
}
# Coupling ratio previously measured for the original `br4_r3` head; quoted only as the
# comparison point in the report, not recomputed here.
BR4_R3_ORIGINAL_COUPLING = 4.28
COUPLING_SHIP_THRESHOLD = 1.2


def _sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_sha() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True,
                          text=True, check=False).stdout.strip() or "unknown"


def load_frozen_filter_and_original_head(path: pathlib.Path, sigma_param: str, device: str):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    assert ck["sampler_semantics"] == CURRENT_SAMPLER_SEMANTICS, (
        f"{path.name}: sampler_semantics={ck.get('sampler_semantics')!r} -- refusing a "
        "checkpoint trained with a different sampler")
    filt = S1FilterV2ItemMLP(**B4_CONFIG, sigma_parameterization=sigma_param).to(device)
    filt.load_state_dict(ck["filter"], strict=True)
    hidden = int(ck["outcome_head"]["trunk.weight"].shape[0])
    orig_head = OutcomeHead(codec_dim=B4_CONFIG["codec_dim"], latent_dim=B4_CONFIG["latent_dim"],
                            hidden=hidden).to(device)
    orig_head.load_state_dict(ck["outcome_head"], strict=True)
    filt.eval()
    orig_head.eval()
    for p in filt.parameters():
        p.requires_grad_(False)
    for p in orig_head.parameters():
        p.requires_grad_(False)
    return filt, orig_head, ck.get("git_sha")


class FactoredCombinedHead(nn.Module):
    """Wraps `help_head` (4-way, p(m|z,e)) + `corr_head` (2-way, p(c|m=none,z,e)) into a
    single callable matching `OutcomeHead`'s own `forward(item_codec, u) -> logits[N,5]`
    contract, so every downstream eval path (the committee loop below, the coupling
    probe's `_case`/`_head`) runs UNCHANGED against either variant. Returns
    log(combined_probs) as "logits": since `softmax(log(p)) == p` exactly (softmax is
    shift-invariant and combined_probs already sums to 1 per row), every caller that
    applies its own softmax on top (both eval paths here do) recovers combined_probs
    bit-for-bit, not an approximation."""

    def __init__(self, help_head: OutcomeHead, corr_head: OutcomeHead):
        super().__init__()
        self.help_head = help_head
        self.corr_head = corr_head

    def forward(self, item_codec: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        p_m = torch.softmax(self.help_head(item_codec, u), dim=-1)  # [N,4]: none,hint,reveal,both
        p_c = torch.softmax(self.corr_head(item_codec, u), dim=-1)  # [N,2]: wrong,correct
        combined = torch.stack([
            p_m[:, 0] * p_c[:, 1],  # UNAIDED_CORRECT
            p_m[:, 0] * p_c[:, 0],  # UNAIDED_WRONG
            p_m[:, 1],              # HINT_ONLY
            p_m[:, 2],              # REVEAL_ONLY
            p_m[:, 3],              # HINT_AND_REVEAL
        ], dim=-1)
        return torch.log(combined.clamp(min=1e-12))


def build_fb(ev_split: pd.DataFrame, codec_matrix, pid_to_idx, device: str) -> dict:
    fb = build_padded_batch_v2(ev_split, codec_matrix, pid_to_idx, device="cpu", include_time_min=True)
    fb["prev_feat"] = build_prev_feat_v2(fb["correct"], fb["hint_bin"], fb["saw_answer"], fb["score_missing"])
    for k in ("item_codec", "prev_feat", "time_min", "valid_mask"):
        fb[k] = fb[k].to(device)
    return fb


def build_row_targets(ev_split: pd.DataFrame, fb: dict, device: str) -> dict:
    """Flat row-aligned tensors for every row in `ev_split`: (l_idx, t_idx) into `fb`'s
    padded [N_learners, T] layout, the 4-way help-mode target (ALL rows valid -- raw
    `support_code`, never missing), and the binary correctness target (valid only on
    unaided rows with a known score)."""
    learner_to_idx = {u: i for i, u in enumerate(fb["learners"])}
    l_idx = torch.tensor([learner_to_idx[u] for u in ev_split["learner"]], dtype=torch.long, device=device)
    t_idx = torch.tensor(ev_split["t"].astype("int64").to_numpy(), dtype=torch.long, device=device)
    m_target = torch.tensor(ev_split["support_code"].astype("int64").to_numpy(), dtype=torch.long, device=device)
    unaided = (ev_split["support_code"].to_numpy() == 0)
    score_known = ~ev_split["score_missing"].to_numpy()
    c_valid_np = unaided & score_known
    c_target_np = np.zeros(len(ev_split), dtype=np.int64)
    c_target_np[c_valid_np] = (ev_split["score"].to_numpy()[c_valid_np] == 1).astype(np.int64)
    return {
        "l_idx": l_idx, "t_idx": t_idx, "m_target": m_target,
        "c_target": torch.tensor(c_target_np, dtype=torch.long, device=device),
        "c_valid": torch.tensor(c_valid_np, dtype=torch.bool, device=device),
        "n_rows": len(ev_split),
    }


def train_factored_heads(mu_all: torch.Tensor, log_sigma_all: torch.Tensor, item_codec_all: torch.Tensor,
                         row: dict, device: str, seed: int) -> tuple[OutcomeHead, OutcomeHead, list[float]]:
    help_head = OutcomeHead(codec_dim=B4_CONFIG["codec_dim"], latent_dim=B4_CONFIG["latent_dim"],
                            hidden=HEAD_HIDDEN, n_classes=4).to(device)
    corr_head = OutcomeHead(codec_dim=B4_CONFIG["codec_dim"], latent_dim=B4_CONFIG["latent_dim"],
                            hidden=HEAD_HIDDEN, n_classes=2).to(device)
    optimizer = torch.optim.Adam(list(help_head.parameters()) + list(corr_head.parameters()), lr=TRAIN_LR)
    gen = torch.Generator(device=device).manual_seed(seed)
    rng = np.random.default_rng(seed)

    n_rows = row["n_rows"]
    l_idx, t_idx = row["l_idx"], row["t_idx"]
    m_target, c_target, c_valid = row["m_target"], row["c_target"], row["c_valid"]
    loss_history = []
    for step in range(TRAIN_STEPS):
        batch_idx = rng.integers(0, n_rows, size=TRAIN_BATCH)
        b = torch.tensor(batch_idx, dtype=torch.long, device=device)
        l_b, t_b = l_idx[b], t_idx[b]
        mu_b, ls_b = mu_all[l_b, t_b], log_sigma_all[l_b, t_b]
        ic_b = item_codec_all[l_b, t_b]
        u_b = reparameterize_from_log_sigma(mu_b, ls_b, k=1, generator=gen)[:, 0, :]

        m_logits = help_head(ic_b, u_b)
        m_loss = F.cross_entropy(m_logits, m_target[b])

        c_mask = c_valid[b]
        if bool(c_mask.any()):
            c_logits = corr_head(ic_b[c_mask], u_b[c_mask])
            c_loss = F.cross_entropy(c_logits, c_target[b][c_mask])
        else:
            c_loss = m_logits.sum() * 0.0

        loss = m_loss + c_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_history.append(float(loss.item()))
        if step % 1000 == 0 or step == TRAIN_STEPS - 1:
            print(f"    [fc-1a train] step={step}/{TRAIN_STEPS} loss={loss.item():.4f} "
                 f"m_loss={m_loss.item():.4f} c_loss={c_loss.item():.4f}", flush=True)
    return help_head, corr_head, loss_history


@torch.no_grad()
def committee_stats_5class(scorer, mu_sel: torch.Tensor, log_sigma_sel: torch.Tensor,
                           item_codec_row: torch.Tensor, k: int, gen: torch.Generator) -> dict:
    """Committee stats for ANY scorer with `scorer(item_codec, u) -> logits[N,5]` -- the
    original OutcomeHead directly, or a FactoredCombinedHead -- in one pass, matching the
    wrong-detection panel's loop (mean/median/mu-only-forward probs + P(UNAIDED_CORRECT)
    committee variance for G1/G2), chunked so that no single [N,K,latent] draw over a full
    val split is materialised (it does not fit in memory)."""
    n_rows = mu_sel.shape[0]
    mean_parts, median_parts, mu_parts, p0_var_parts = [], [], [], []
    for lo in range(0, n_rows, EVAL_CHUNK):
        hi = min(lo + EVAL_CHUNK, n_rows)
        mu_c, ls_c, ic_c = mu_sel[lo:hi], log_sigma_sel[lo:hi], item_codec_row[lo:hi]
        u_k = reparameterize_from_log_sigma(mu_c, ls_c, k=k, generator=gen)
        logits_k = torch.stack([scorer(ic_c, u_k[:, j, :]) for j in range(k)], dim=1)
        probs_k = torch.softmax(logits_k, dim=-1)
        mean_parts.append(probs_k.mean(dim=1).cpu())
        median_parts.append(probs_k.median(dim=1).values.cpu())
        p0_var_parts.append(probs_k[..., UNAIDED_CORRECT].var(dim=1, unbiased=True).cpu())
        mu_parts.append(torch.softmax(scorer(ic_c, mu_c), dim=-1).cpu())
    return {
        "probs_mean": torch.cat(mean_parts, dim=0).numpy(),
        "probs_median": torch.cat(median_parts, dim=0).numpy(),
        "probs_mu": torch.cat(mu_parts, dim=0).numpy(),
        "p0_var": torch.cat(p0_var_parts, dim=0).numpy(),
    }


def wrong_detection_and_standard_panel(name: str, stats: dict, outcome_target: np.ndarray,
                                       row_tercile: np.ndarray, t_np: np.ndarray) -> dict:
    """The four-part panel of wd_1_wrong_detection_panel.py (G1/G2 re-grade, wrong-detection
    AUC, bias decomposition + the three artifact-vs-real separating tests, mean-vs-median),
    using the same instruments (imported, not re-derived), plus the standard comparison panel (5-class accuracy, macro-F1, log-loss,
    per-class AUC, binary UC AUC, ECE), overall and bottom tercile."""
    probs_mean, probs_median, probs_mu, p0_var = (
        stats["probs_mean"], stats["probs_median"], stats["probs_mu"], stats["p0_var"])

    # -- (1) corrected G1/G2 re-grade --
    g1_mask, g2_mask = t_np == 0, t_np > LATE_T_LO
    g1_val = float(p0_var[g1_mask].mean()) if g1_mask.any() else float("nan")
    g2_val = float(p0_var[g2_mask].mean()) if g2_mask.any() else float("nan")

    # -- (2) wrong-detection panel --
    y_uw = (outcome_target == UNAIDED_WRONG).astype(int)
    s_uw = probs_mean[:, UNAIDED_WRONG]
    y_notcorrect = (outcome_target != UNAIDED_CORRECT).astype(int)
    s_notcorrect = 1.0 - probs_mean[:, UNAIDED_CORRECT]
    wrong_detection: dict = {"overall": {
        "auc_unaided_wrong": _auc(y_uw, s_uw), "auc_not_unaided_correct": _auc(y_notcorrect, s_notcorrect),
        "n": int(len(y_uw)),
    }}
    for terc in ("bottom", "middle", "top"):
        m = row_tercile == terc
        wrong_detection[terc] = {
            "auc_unaided_wrong": _auc(y_uw[m], s_uw[m]),
            "auc_not_unaided_correct": _auc(y_notcorrect[m], s_notcorrect[m]),
            "n": int(m.sum()),
            "ece_by_class": _ece_per_class(probs_mean[m], outcome_target[m]),
            "wrong_reliability_bins": _reliability_bins(s_notcorrect[m], y_notcorrect[m]),
        }

    # -- (3) bias decomposition, bottom tercile + the three separating tests --
    bottom = row_tercile == "bottom"
    pred_alloc = probs_mean[bottom].mean(axis=0)
    emp_alloc = np.array([(outcome_target[bottom] == c).mean() for c in range(N_OUTCOME)])
    bias_decomposition = {OUTCOME_CLASSES[c]: {
        "predicted_mean_prob": float(pred_alloc[c]), "empirical_frequency": float(emp_alloc[c]),
        "predicted_minus_empirical": float(pred_alloc[c] - emp_alloc[c]),
    } for c in range(N_OUTCOME)}

    y0 = (outcome_target == UNAIDED_CORRECT).astype(float)
    signed_bias_by_tercile, width_zeroed_bias_by_tercile = {}, {}
    for terc in ("bottom", "middle", "top"):
        m = row_tercile == terc
        signed_bias_by_tercile[terc] = float((probs_mean[m, UNAIDED_CORRECT] - y0[m]).mean())
        width_zeroed_bias_by_tercile[terc] = float((probs_mu[m, UNAIDED_CORRECT] - y0[m]).mean())

    bias_vs_t = {}
    for t_exact in TIMES_EXACT:
        m = bottom & (t_np == t_exact)
        bias_vs_t[str(t_exact)] = {
            "n": int(m.sum()),
            "signed_bias": float((probs_mean[m, UNAIDED_CORRECT] - y0[m]).mean()) if m.any() else None,
        }

    mean_vs_median_bias = {
        "mean_committee_bias": float((probs_mean[bottom, UNAIDED_CORRECT] - y0[bottom]).mean()),
        "median_committee_bias": float((probs_median[bottom, UNAIDED_CORRECT] - y0[bottom]).mean()),
    }

    # -- standard comparison panel --
    from sklearn.metrics import f1_score

    def _panel(mask: np.ndarray) -> dict:
        p, y = probs_mean[mask], outcome_target[mask]
        pred = p.argmax(axis=1)
        per_class_auc = {OUTCOME_CLASSES[c]: _auc((y == c).astype(int), p[:, c]) for c in range(N_OUTCOME)}
        return {
            "n": int(mask.sum()),
            "accuracy_5class": float((pred == y).mean()),
            "macro_f1": float(f1_score(y, pred, average="macro", labels=list(range(N_OUTCOME)), zero_division=0)),
            "log_loss_5class": multiclass_nll(p, y),
            "per_class_auc": per_class_auc,
            "binary_unaided_correct_auc": per_class_auc[OUTCOME_CLASSES[UNAIDED_CORRECT]],
            "ece_overall_top1": multiclass_ece(p, y, n_bins=10),
            "ece_per_class": _ece_per_class(p, y),
        }

    standard_panel = {"overall": _panel(np.ones(len(outcome_target), dtype=bool)),
                      "bottom_tercile": _panel(row_tercile == "bottom")}

    return {
        "g1_corrected": g1_val, "g2_corrected": g2_val,
        "trait_ceiling_mid": TRAIT_CEILING_MID, "trait_ceiling_ci": list(TRAIT_CEILING_CI),
        "wrong_detection": wrong_detection,
        "bias_decomposition_bottom_tercile": bias_decomposition,
        "signed_bias_by_tercile_natural": signed_bias_by_tercile,
        "signed_bias_by_tercile_width_zeroed": width_zeroed_bias_by_tercile,
        "bias_vs_t_bottom_tercile": bias_vs_t,
        "mean_vs_median_bias_bottom_tercile": mean_vs_median_bias,
        "standard_comparison_panel": standard_panel,
    }


def run_coupling_probe(name: str, wrapper_head: FactoredCombinedHead, filt: nn.Module,
                       n_learners: int, seed: int, out_path: pathlib.Path) -> dict:
    """fa_coupling_probe.py's own `run()` body, generalised to accept an already-built
    `model` dict (its own `_load()` assumes the OLD single-OutcomeHead checkpoint
    format, which a factored pair of heads does not match) -- `_case`/`_boot_mean` are
    imported unchanged from that module, so the geometry and verdict threshold are
    identical to the other coupling-probe runs.

    `fa_coupling_probe.py`'s `_case`/`_head` build plain CPU tensors throughout (no
    `device=` handling anywhere in that module -- `_load()` never calls `.to(device)`
    either), so `filt`/`wrapper_head` are moved to CPU here regardless of what device
    they trained/scored on upstream in `main()`; both are already `del`eted right after
    this call returns, so moving them in place is safe."""
    filt = filt.to("cpu")
    wrapper_head = wrapper_head.to("cpu")
    model = {"filter": filt, "head": wrapper_head}
    ev = load_v2_events(REPO / "artifacts/fa_manifest_v2.parquet", REPO / "splits/fa_learner_split_v2.csv",
                        REPO / "splits/fa_item_split_v2.csv")
    ev = ev[ev["split"] == "val"]
    codec, pid = load_production_codec(CODEC_PATH, B4_CONFIG["codec_dim"])
    groups = {u: g for u, g in ev.groupby("learner", sort=True)}
    eligible = [u for u, g in groups.items() if len(g) > 106
               and bool(outcome_label(g.sort_values("t")["support_code"], g.sort_values("t")["score"],
                                      g.sort_values("t")["score_missing"]).iloc[10]["outcome_valid"])
               and bool(outcome_label(g.sort_values("t")["support_code"], g.sort_values("t")["score"],
                                      g.sort_values("t")["score_missing"]).iloc[100]["outcome_valid"])]
    rng = np.random.default_rng(seed)
    rows = []
    for u in eligible[:n_learners]:
        g = groups[u]
        for t in (10, 100):
            rows.append(_case(model, g, codec, pid, t, K_COUPLING_DIRS, K_COUPLING_DRAWS, rng))
    top = np.array([r["top_tv"] for r in rows])
    bot = np.array([r["bottom_tv"] for r in rows])
    iso = np.array([r["iso_tv"] for r in rows])
    ratio = top / np.maximum(bot, 1e-12)
    top_iso = top / np.maximum(iso, 1e-12)
    out_obj = {
        "task": "FC-1a coupling probe (factored variant)", "checkpoint_name": name,
        "n_learners": len(eligible[:n_learners]), "n_rows": len(rows),
        "geometry": {"times": [10, 100], "n_dirs": K_COUPLING_DIRS, "k": K_COUPLING_DRAWS},
        "sensitivity": {
            "top_tv": _boot_mean(top, seed), "bottom_tv": _boot_mean(bot, seed + 1),
            "isotropic_tv": _boot_mean(iso, seed + 2), "top_bottom_ratio": _boot_mean(ratio, seed + 3),
            "top_isotropic_ratio": _boot_mean(top_iso, seed + 4),
        },
        "top_bottom_ratio_median": float(np.median(ratio)),
        "verdict": "DEAF_ALIGNMENT" if float(np.median(ratio)) <= 1.2 else "COUPLED_ALIGNMENT",
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out_obj, indent=2, default=str))
    print(f"    [coupling] {name}: top_bottom_ratio_median={out_obj['top_bottom_ratio_median']:.4f} "
         f"verdict={out_obj['verdict']}", flush=True)
    return out_obj


def main() -> int:
    seed_everything(SEED)
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[fc-1a] device={device} git_sha={_git_sha()}", flush=True)

    ev = load_v2_events(REPO / "artifacts/fa_manifest_v2.parquet", REPO / "splits/fa_learner_split_v2.csv",
                        REPO / "splits/fa_item_split_v2.csv")
    ev_train = ev[ev["split"] == "train"].reset_index(drop=True)
    ev_val = ev[ev["split"] == "val"].reset_index(drop=True)
    codec_matrix, pid_to_idx = load_production_codec(CODEC_PATH, B4_CONFIG["codec_dim"])
    print(f"[fc-1a] train rows={len(ev_train)} learners={ev_train['learner'].nunique()}; "
         f"val rows={len(ev_val)} learners={ev_val['learner'].nunique()}", flush=True)

    # -- ability tercile, CHECKPOINT-INDEPENDENT (true labels only), shared across every
    # variant, as in wd_1_wrong_detection_panel.py. --
    labels_val = outcome_label(ev_val["support_code"], ev_val["score"], ev_val["score_missing"])
    valid_val = labels_val["outcome_valid"].to_numpy()
    rows_val = ev_val.iloc[np.flatnonzero(valid_val)].reset_index(drop=True)
    outcome_target = labels_val["outcome_code"].to_numpy()[valid_val]
    learner_np = rows_val["learner"].astype(str).to_numpy()
    t_np = rows_val["t"].astype("int64").to_numpy()
    ability = (pd.Series(outcome_target == UNAIDED_CORRECT, index=learner_np).groupby(level=0).mean())
    tercile_map = pd.qcut(ability.rank(method="first"), 3, labels=["bottom", "middle", "top"])
    row_tercile = pd.Series(learner_np).map(tercile_map).to_numpy()
    print(f"[fc-1a] {len(rows_val)} val rows, {len(ability)} learners, "
         f"tercile counts: {pd.Series(row_tercile).value_counts().to_dict()}", flush=True)

    results: dict = {}
    raw_probs: dict = {}  # name -> {"original": probs[N,5], "factored": probs[N,5]} (numpy, not JSON-serialised)
    for name, (path, sigma_param) in CHECKPOINTS.items():
        print(f"[fc-1a] === {name} ===", flush=True)
        filt, orig_head, git_sha = load_frozen_filter_and_original_head(path, sigma_param, device)
        filt_forward = lambda ic, pf, tm, vm: filt(ic, pf)  # noqa: E731

        train_fb = build_fb(ev_train, codec_matrix, pid_to_idx, device)
        val_fb = build_fb(ev_val, codec_matrix, pid_to_idx, device)
        with torch.no_grad():
            mu_train_all, ls_train_all = chunked_filter_forward(filt_forward, train_fb)
            mu_val_all, ls_val_all = chunked_filter_forward(filt_forward, val_fb)
        row_train = build_row_targets(ev_train, train_fb, device)

        # -- check: filter parameters receive no gradient. Snapshot before training and
        # assert byte-identical after (a post-hoc check, not just the absence of an
        # optimiser for them). --
        filt_state_before = {k: v.clone() for k, v in filt.state_dict().items()}

        print(f"  [fc-1a] training factored heads on {name}'s frozen filter "
             f"({TRAIN_STEPS} steps, batch={TRAIN_BATCH})...", flush=True)
        help_head, corr_head, loss_hist = train_factored_heads(
            mu_train_all, ls_train_all, train_fb["item_codec"], row_train, device, seed=SEED)
        help_head.eval()
        corr_head.eval()

        filt_state_after = filt.state_dict()
        filter_unchanged = all(torch.equal(filt_state_before[k], filt_state_after[k]) for k in filt_state_before)
        assert filter_unchanged, f"{name}: filter parameters changed during head-only training -- frozen-filter check FAILED"
        print(f"  [fc-1a] frozen-filter check PASS: filter parameters byte-identical after head-only training", flush=True)

        combined_head = FactoredCombinedHead(help_head, corr_head)

        # -- save the factored checkpoint (filter unchanged, two new heads) --
        ckpt_out = {
            "filter": filt.state_dict(), "help_head": help_head.state_dict(), "corr_head": corr_head.state_dict(),
            "architecture": {"class": "S1FilterV2ItemMLP/FactoredHeads",
                             "help_head_n_classes": 4, "corr_head_n_classes": 2, "head_hidden": HEAD_HIDDEN,
                             "sigma_parameterization": sigma_param, **B4_CONFIG},
            "codec_path": str(CODEC_PATH), "codec_sha256": _sha256(CODEC_PATH),
            "sampler_semantics": CURRENT_SAMPLER_SEMANTICS, "git_sha": _git_sha(), "seed": SEED,
            "warm_start_from": str(path), "warm_start_git_sha": git_sha,
            "train_steps": TRAIN_STEPS, "train_batch": TRAIN_BATCH, "train_lr": TRAIN_LR,
            "loss_first": loss_hist[0], "loss_last": loss_hist[-1],
            "filter_gradient_free_verified": filter_unchanged,
            "stage": "fc_1a_step_zero_factored_heads",
        }
        out_dir = REPO / "results/fc_1a"
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(ckpt_out, out_dir / f"{name}_factored.pt")
        print(f"  [fc-1a] wrote {out_dir}/{name}_factored.pt", flush=True)

        # -- val-split committee scoring, both variants --
        learner_to_idx_val = {u: i for i, u in enumerate(val_fb["learners"])}
        l_idx_val = torch.tensor([learner_to_idx_val[u] for u in rows_val["learner"]], dtype=torch.long, device=device)
        t_idx_val = torch.tensor(rows_val["t"].astype("int64").to_numpy(), dtype=torch.long, device=device)
        mu_sel = mu_val_all[l_idx_val, t_idx_val]
        ls_sel = ls_val_all[l_idx_val, t_idx_val]
        item_codec_row_val = val_fb["item_codec"][l_idx_val, t_idx_val]

        gen_orig = torch.Generator(device=device).manual_seed(SEED)
        gen_fact = torch.Generator(device=device).manual_seed(SEED)
        stats_orig = committee_stats_5class(orig_head, mu_sel, ls_sel, item_codec_row_val, K, gen_orig)
        stats_factored = committee_stats_5class(combined_head, mu_sel, ls_sel, item_codec_row_val, K, gen_fact)
        probs_orig, probs_factored = stats_orig["probs_mean"], stats_factored["probs_mean"]

        panel_orig = wrong_detection_and_standard_panel(f"{name}_original", stats_orig, outcome_target,
                                                        row_tercile, t_np)
        panel_factored = wrong_detection_and_standard_panel(f"{name}_factored", stats_factored, outcome_target,
                                                             row_tercile, t_np)

        # -- arithmetic re-read baseline: p(c=correct|m=none) = p(UC)/(p(UC)+p(UW)) from
        # the ORIGINAL head, restricted to actually-unaided val rows, vs the FACTORED
        # corr_head's OWN conditional prediction on the same rows. This isolates
        # whether retraining a dedicated correctness head beats pure re-reading. --
        unaided_mask = (outcome_target == UNAIDED_CORRECT) | (outcome_target == UNAIDED_WRONG)
        y_correct = (outcome_target[unaided_mask] == UNAIDED_CORRECT).astype(int)
        denom = probs_orig[unaided_mask, UNAIDED_CORRECT] + probs_orig[unaided_mask, UNAIDED_WRONG]
        reread_p_correct = probs_orig[unaided_mask, UNAIDED_CORRECT] / np.clip(denom, 1e-12, None)

        with torch.no_grad():
            gen_corr = torch.Generator(device=device).manual_seed(SEED + 2)
            n_rows = mu_sel.shape[0]
            corr_parts = []
            for lo in range(0, n_rows, EVAL_CHUNK):
                hi = min(lo + EVAL_CHUNK, n_rows)
                u_k = reparameterize_from_log_sigma(mu_sel[lo:hi], ls_sel[lo:hi], k=K, generator=gen_corr)
                probs_k = torch.stack([torch.softmax(corr_head(item_codec_row_val[lo:hi], u_k[:, j, :]), dim=-1)
                                       for j in range(K)], dim=0)
                corr_parts.append(probs_k.mean(dim=0).cpu())
            factored_corr_probs = torch.cat(corr_parts, dim=0).numpy()
        factored_p_correct = factored_corr_probs[unaided_mask, 1]

        reread_vs_factored = {
            "n_unaided_val_rows": int(unaided_mask.sum()),
            "arithmetic_reread": {"auc": _auc(y_correct, reread_p_correct),
                                  "log_loss": multiclass_nll(np.stack([1 - reread_p_correct, reread_p_correct], axis=1), y_correct)},
            "factored_corr_head": {"auc": _auc(y_correct, factored_p_correct),
                                   "log_loss": multiclass_nll(np.stack([1 - factored_p_correct, factored_p_correct], axis=1), y_correct)},
            "note": "arithmetic_reread is derived (no training) from the ORIGINAL head's own "
                    "p(UC)/(p(UC)+p(UW)); factored_corr_head is the newly-trained conditional head. "
                    "Both scored on the SAME actually-unaided val rows.",
        }

        # -- coupling probe, factored variant only (original variants' coupling is
        # already measured -- fa_sim3.json / br4_r3.json in results/fa_coupling_probe/) --
        coupling = run_coupling_probe(f"{name}_factored", combined_head, filt, n_learners=50,
                                      seed=SEED, out_path=REPO / f"results/fc_1a/{name}_factored_coupling.json")

        results[name] = {
            "checkpoint_path": str(path), "warm_start_git_sha": git_sha, "sigma_parameterization": sigma_param,
            "loss_first": loss_hist[0], "loss_last": loss_hist[-1],
            "filter_gradient_free_verified": filter_unchanged,
            "original": panel_orig, "factored": panel_factored,
            "conditional_correctness_reread_vs_factored": reread_vs_factored,
            "coupling_factored": coupling,
        }
        raw_probs[name] = {"original": probs_orig, "factored": probs_factored}
        print(f"[fc-1a] {name}: original G1={panel_orig['g1_corrected']:.5f} G2={panel_orig['g2_corrected']:.5f} "
             f"AUC(UW,bottom)={panel_orig['wrong_detection']['bottom']['auc_unaided_wrong']:.4f} || "
             f"factored G1={panel_factored['g1_corrected']:.5f} G2={panel_factored['g2_corrected']:.5f} "
             f"AUC(UW,bottom)={panel_factored['wrong_detection']['bottom']['auc_unaided_wrong']:.4f}", flush=True)

        del filt, orig_head, help_head, corr_head, combined_head
        if device == "cuda":
            torch.cuda.empty_cache()

    # -- pre-registered decision rule (module docstring), applied --
    # Bottom-tercile rows are IDENTICAL across every variant (same val cohort, same
    # tercile assignment computed once from true labels, same l_idx_val/t_idx_val
    # ordering for every scorer this run) -- the paired bootstrap needs exactly that
    # row alignment.
    bm = row_tercile == "bottom"
    y_uw = (outcome_target == UNAIDED_WRONG).astype(int)
    y_nc = (outcome_target != UNAIDED_CORRECT).astype(int)

    def _ship_diff(probs_a: np.ndarray, probs_b: np.ndarray, seed: int) -> dict:
        """AUC(a) - AUC(b), learner-clustered bootstrap, bottom tercile -- the conservative
        rule of wd_1_wrong_detection_panel.py: adopted iff the CI does not show `a` significantly worse."""
        s_uw_a, s_uw_b = probs_a[:, UNAIDED_WRONG], probs_b[:, UNAIDED_WRONG]
        s_nc_a, s_nc_b = 1.0 - probs_a[:, UNAIDED_CORRECT], 1.0 - probs_b[:, UNAIDED_CORRECT]
        point_uw, lo_uw, hi_uw = _learner_clustered_bootstrap_auc_diff(
            y_uw[bm], s_uw_a[bm], y_uw[bm], s_uw_b[bm], learner_np[bm], n_boot=1000, seed=seed)
        point_nc, lo_nc, hi_nc = _learner_clustered_bootstrap_auc_diff(
            y_nc[bm], s_nc_a[bm], y_nc[bm], s_nc_b[bm], learner_np[bm], n_boot=1000, seed=seed + 1)
        return {"auc_unaided_wrong_diff": {"point": point_uw, "ci95": [lo_uw, hi_uw]},
               "auc_not_unaided_correct_diff": {"point": point_nc, "ci95": [lo_nc, hi_nc]},
               "not_significantly_worse": bool(hi_uw >= 0.0 and hi_nc >= 0.0)}

    fa_orig_probs = raw_probs["fa_sim3"]["original"]
    diff_r3_factored_vs_fa_orig = _ship_diff(raw_probs["br4_r3"]["factored"], fa_orig_probs, seed=SEED + 10)
    diff_fa_factored_vs_fa_orig = _ship_diff(raw_probs["fa_sim3"]["factored"], fa_orig_probs, seed=SEED + 12)

    coupling_r3_factored = results["br4_r3"]["coupling_factored"]["top_bottom_ratio_median"]
    coupling_fa_factored = results["fa_sim3"]["coupling_factored"]["top_bottom_ratio_median"]
    r3_wd_ok = diff_r3_factored_vs_fa_orig["not_significantly_worse"]
    r3_coupling_ok = coupling_r3_factored > COUPLING_SHIP_THRESHOLD
    ship_r3_factored = bool(r3_wd_ok and r3_coupling_ok)
    fa_wd_ok = diff_fa_factored_vs_fa_orig["not_significantly_worse"]
    fa_coupling_ok = coupling_fa_factored > COUPLING_SHIP_THRESHOLD
    ship_fa_factored = bool(fa_wd_ok and fa_coupling_ok)
    any_ship = bool(ship_r3_factored or ship_fa_factored)

    ship_decision = {
        "rule": "the factored-R3 variant ships iff bottom-tercile wrong-detection (vs FA-SIM3-ORIGINAL, "
                "learner-clustered bootstrap, n_boot=1000) CI does not show it significantly worse on BOTH "
                "AUC definitions AND coupling top_bottom_ratio_median > 1.2 (the original br4_r3 head "
                f"measured {BR4_R3_ORIGINAL_COUPLING}); a factored-FA-SIM3 win against FA-SIM3-original is "
                "also adoptable, reported as such.",
        "r3_factored": {"wrong_detection_diff_vs_fa_sim3_original": diff_r3_factored_vs_fa_orig,
                       "coupling_top_bottom_ratio_median": coupling_r3_factored,
                       "coupling_threshold": COUPLING_SHIP_THRESHOLD,
                       "coupling_pass": r3_coupling_ok, "wrong_detection_pass": r3_wd_ok, "ships": ship_r3_factored},
        "fa_sim3_factored": {"wrong_detection_diff_vs_fa_sim3_original": diff_fa_factored_vs_fa_orig,
                            "coupling_top_bottom_ratio_median": coupling_fa_factored,
                            "coupling_threshold": COUPLING_SHIP_THRESHOLD,
                            "coupling_pass": fa_coupling_ok, "wrong_detection_pass": fa_wd_ok, "ships": ship_fa_factored},
        "any_ship": any_ship,
        "verdict": (
            "ADOPTED -- at least one factored variant meets the decision rule."
            if any_ship else
            "NOT ADOPTED -- the decision rule failed for both factored variants; a retrain of the filter "
            "with the factored heads would be the next step."
        ),
    }
    print(f"[fc-1a] SHIP DECISION: {ship_decision['verdict']}", flush=True)

    summary = {
        "task": "FC-1a", "k": K, "train_steps": TRAIN_STEPS, "train_batch": TRAIN_BATCH, "train_lr": TRAIN_LR,
        "head_hidden": HEAD_HIDDEN, "git_sha": _git_sha(), "checkpoints": results, "ship_decision": ship_decision,
    }
    out_dir = REPO / "results/fc_1a"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"[fc-1a] wrote {out_dir}/summary.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
