"""Shared helpers for `fc_1a_factored_heads` (Appendix A, factored head).

`_rank` (average ranks with ties), `_corr` (Pearson correlation, None when undefined) and
`_boot_mean` (mean with a 2.5-97.5% percentile bootstrap interval). `_case` scores one
(learner, t) case: it runs the filter on the learner's history up to t, then measures the
total-variation change in the head's outcome distribution when the belief mean is moved by
+/- one sigma along the highest-sigma, lowest-sigma and random isotropic directions, the
K-sample predictive spread, and the error on the observed outcome. Note that
`sigma_sensitivity_spearman` pairs sigma in dimension-index order with the sensitivities in
descending-sigma order (`all_tv` is computed over `order`), so it is not a per-dimension
correlation.
"""
from __future__ import annotations
import pathlib
import sys
import numpy as np
import pandas as pd
import torch
REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
from atom.data.fa_outcome_label import outcome_label  # noqa: E402
from atom.filter.episode_v2 import build_prev_feat_v2  # noqa: E402


def _rank(x: np.ndarray) -> np.ndarray:
    o = np.argsort(x, kind="mergesort"); sx = x[o]; r = np.empty(len(x), dtype=float); i = 0
    while i < len(x):
        j = i + 1
        while j < len(x) and sx[j] == sx[i]: j += 1
        r[o[i:j]] = 0.5 * (i + j - 1) + 1.0; i = j
    return r


def _corr(x: np.ndarray, y: np.ndarray) -> float | None:
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0: return None
    return float(np.corrcoef(x, y)[0, 1])


def _boot_mean(x: np.ndarray, seed: int, n_boot: int = 2000) -> dict[str, float]:
    rng = np.random.default_rng(seed); vals = np.asarray([x[rng.integers(0, len(x), len(x))].mean() for _ in range(n_boot)])
    return {"mean": float(x.mean()), "lo": float(np.quantile(vals, .025)), "hi": float(np.quantile(vals, .975))}


def _head(head: torch.nn.Module, item: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    items = item.unsqueeze(0).expand(len(u), -1)
    return torch.softmax(head(items, u), dim=-1)


@torch.no_grad()
def _case(model: dict, g: pd.DataFrame, codec: np.ndarray, pid: dict[int, int], t: int,
          n_dirs: int, k: int, rng: np.random.Generator) -> dict:
    g = g.sort_values("t", kind="mergesort").reset_index(drop=True)
    labels = outcome_label(g["support_code"], g["score"], g["score_missing"])
    target_label = int(labels.iloc[t]["outcome_code"])
    if target_label < 0: raise ValueError("invalid coupling target")
    item = torch.as_tensor(codec[g["problem_id"].map(pid).to_numpy()], dtype=torch.float32)
    score = torch.as_tensor(g["score"].fillna(0).to_numpy(dtype=np.float32)[:t]).unsqueeze(0)
    missing = torch.as_tensor(g["score_missing"].to_numpy(dtype=bool)[:t].copy()).unsqueeze(0)
    hint = torch.as_tensor((g["hint_count"].to_numpy()[:t] > 0).astype(np.float32)).unsqueeze(0)
    reveal = torch.as_tensor(g["saw_answer"].to_numpy(dtype=np.float32)[:t]).unsqueeze(0)
    seq = item[:t].unsqueeze(0)
    mu, ls = model["filter"](torch.cat([seq, item[t].view(1, 1, -1)], dim=1),
                              build_prev_feat_v2(torch.cat([score, torch.zeros(1, 1)], dim=1),
                                                 torch.cat([hint, torch.zeros(1, 1)], dim=1),
                                                 torch.cat([reveal, torch.zeros(1, 1)], dim=1),
                                                 torch.cat([missing, torch.zeros(1, 1, dtype=torch.bool)], dim=1)))
    mu, ls = mu[:, -1], ls[:, -1]
    sig = ls.exp()[0]
    order = torch.argsort(sig, descending=True); top = order[:n_dirs]; bot = order[-n_dirs:]
    eye = torch.eye(256)
    def tv(ix: torch.Tensor) -> np.ndarray:
        plus = mu + sig[ix].unsqueeze(1) * eye[ix]; minus = mu - sig[ix].unsqueeze(1) * eye[ix]
        return (0.5 * (_head(model["head"], item[t], plus) - _head(model["head"], item[t], minus)).abs().sum(1)).numpy()
    top_tv, bot_tv, all_tv = tv(top), tv(bot), tv(order)
    iso = torch.as_tensor(rng.normal(size=(n_dirs, 256)), dtype=torch.float32); iso /= iso.norm(dim=1, keepdim=True).clamp_min(1e-8); iso = mu + iso * sig.norm() / np.sqrt(256)
    iso2 = 2 * mu - iso
    iso_tv = (0.5 * (_head(model["head"], item[t], iso) - _head(model["head"], item[t], iso2)).abs().sum(1)).numpy()
    eps = torch.as_tensor(rng.normal(size=(k, 256)), dtype=torch.float32)
    probs = _head(model["head"], item[t], mu + sig * eps); mp = probs.mean(0)
    spread = float(probs.std(0, unbiased=False).mean()); error = 1.0 - float(mp[target_label])
    return {"learner": str(g.learner.iloc[0]), "t": t, "top_tv": float(top_tv.mean()), "bottom_tv": float(bot_tv.mean()),
            "iso_tv": float(iso_tv.mean()), "top_bottom": float(top_tv.mean() / max(bot_tv.mean(), 1e-12)),
            "spread": spread, "error": error, "sigma_mean": float(sig.mean()),
            "sigma_sensitivity_spearman": _corr(_rank(sig.numpy()), _rank(all_tv))}
