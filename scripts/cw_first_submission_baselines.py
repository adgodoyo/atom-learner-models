#!/usr/bin/env python
"""CodeWorkout first-submission correctness: ATOM against two simple baselines (thesis Table 9).
Computes P(passes all tests) AUC for ATOM (class 3 of its saved 4-class validation posterior),
an item-difficulty + learner-ability baseline, and an ID-only recurrent network (DKT-style GRU).
Inputs: results/cw_sigma_precision_0305a22/aws/results/val_predictions.npz, splits/, ../Datasets/.
Output: results/cw_first_submission_baselines/summary.json (aggregates only, no per-row data).
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import pandas as pd

SCRIPTS_DIR = pathlib.Path(__file__).resolve().parent
DEFAULT_REPO = SCRIPTS_DIR.parent

PRED_RELPATH = "results/cw_sigma_precision_0305a22/aws/results/val_predictions.npz"
FULL_PASS = 3            # 4-class outcome: 0 compile error, 1 zero, 2 partial, 3 all tests pass
DIFFICULTY_PRIOR = 20.0  # pseudo-count shrinking each problem's pass rate to the global rate
ABILITY_PRIOR = 5.0      # pseudo-count shrinking each learner's running pass rate to the global rate
N_BOOT = 2000
SEED = 0

# ID-only GRU (DKT-style) hyper-parameters
DKT_DIM = 64
DKT_EPOCHS = 40
DKT_BATCH = 32
DKT_LR = 1e-3


# ----------------------------------------------------------------------------- data


def held_out_items(repo: pathlib.Path) -> list[int]:
    """The seven problems never seen in training (frozen item split)."""
    s = pd.read_csv(repo / "splits" / "cw_s19_item_split.csv")
    return sorted(int(p) for p in s.loc[s["item_split"] == "heldout", "problem_id"])


def first_rows(df: pd.DataFrame, learner: str, item: str, order: str) -> pd.DataFrame:
    """Each learner's earliest attempt at each problem, in causal order."""
    df = df.sort_values([learner, order], kind="mergesort")
    df = df[~df.duplicated([learner, item])]
    return df.reset_index(drop=True)


def load_atom_rows(repo: pathlib.Path) -> pd.DataFrame:
    """ATOM's saved validation predictions, one row per attempt."""
    d = np.load(repo / PRED_RELPATH)
    return pd.DataFrame({
        "learner": d["learner"].astype(str),
        "problem_id": d["item"].astype(int),
        "t": d["attempt_history_index"].astype(int),
        "prior_episodes": d["prior_episode_count"].astype(int),
        "y": (d["y_true"] == FULL_PASS).astype(int),
        "p_atom": d["y_pred"][:, FULL_PASS].astype(float),
    })


def load_corpus_attempts(repo: pathlib.Path, raw_dir: str | None) -> pd.DataFrame:
    """All S19+F19 attempts with the 4-class label and the learner split attached."""
    sys.path.insert(0, str(repo / "src"))
    from atom.data.cw_manifest import load_attempts_with_episodes
    from atom.data.cw_outcome_label import outcome_label

    a = load_attempts_with_episodes(raw_dir=raw_dir)
    lab = outcome_label(a["compile_result"], a["score_frac"])
    a = a.assign(y=(lab["outcome_code"].to_numpy() == FULL_PASS).astype(int))
    split = pd.read_csv(repo / "splits" / "cw_learner_split.csv")[["learner", "split"]]
    a = a.merge(split, on="learner", how="left", validate="many_to_one")
    if a["split"].isna().any():
        raise ValueError("attempts from learners missing in splits/cw_learner_split.csv")
    return a[["learner", "problem_id", "t", "y", "split"]]


# ----------------------------------------------------------------------------- metrics


def auc(y, s) -> float:
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y)
    return float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else float("nan")


def clustered_auc_ci(y, s, learners, n_boot: int = N_BOOT, seed: int = SEED) -> list[float]:
    """Percentile bootstrap over learners (rows of one learner move together)."""
    y, s, learners = np.asarray(y), np.asarray(s), np.asarray(learners)
    uniq = np.unique(learners)
    rows = {u: np.where(learners == u)[0] for u in uniq}
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        idx = np.concatenate([rows[u] for u in uniq[rng.integers(0, len(uniq), len(uniq))]])
        v = auc(y[idx], s[idx])
        if not np.isnan(v):
            draws.append(v)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return [float(lo), float(hi)]


def scored(y, p, learners, ci: bool = True) -> dict:
    from sklearn.metrics import log_loss
    y, p = np.asarray(y), np.clip(np.asarray(p, dtype=float), 1e-7, 1 - 1e-7)
    out = {"auc": auc(y, p), "n": int(len(y)), "n_learners": int(len(np.unique(learners))),
           "base_rate": float(y.mean()),
           "accuracy": float(((p >= 0.5) == y).mean()), "log_loss": float(log_loss(y, p, labels=[0, 1]))}
    if ci:
        out["ci95"] = clustered_auc_ci(y, p, learners)
    return out


def macro_auc(df: pd.DataFrame, group: str, score: str, min_rows: int = 1) -> dict:
    """Mean of per-group AUCs over groups with both outcomes and at least `min_rows` rows."""
    vals = [auc(g["y"], g[score]) for _, g in df.groupby(group)
            if len(g) >= min_rows and 0 < g["y"].sum() < len(g)]
    return {"auc": float(np.mean(vals)), "n_groups": len(vals), "min_rows": min_rows,
            "iqr": [float(np.percentile(vals, 25)), float(np.percentile(vals, 75))]}


# ----------------------------------------------------------------------------- two-scalar baseline


def logit(p):
    return np.log(p / (1 - p))


def item_difficulty(ref_first: pd.DataFrame, exclude=()) -> tuple[dict, float]:
    """Shrunken per-problem pass rate on reference learners' first submissions."""
    g = float(ref_first["y"].mean())
    ref = ref_first[~ref_first["problem_id"].isin(exclude)]
    st = ref.groupby("problem_id")["y"].agg(["sum", "count"])
    table = ((st["sum"] + DIFFICULTY_PRIOR * g) / (st["count"] + DIFFICULTY_PRIOR)).to_dict()
    return table, g


def causal_ability(eval_first: pd.DataFrame, g: float) -> np.ndarray:
    """Running pass rate over the learner's earlier first submissions only (rows in causal order)."""
    by = eval_first.groupby("learner", sort=False)["y"]
    prev_sum = by.cumsum() - eval_first["y"]
    prev_n = by.cumcount()
    return ((prev_sum + ABILITY_PRIOR * g) / (prev_n + ABILITY_PRIOR)).to_numpy()


def two_scalar_predictions(eval_first: pd.DataFrame, ref_first: pd.DataFrame, exclude=()) -> pd.DataFrame:
    """Difficulty, ability and their additive-logit combination for each evaluation row.

    The combination is logit(p) = logit(difficulty) + logit(ability) - logit(global rate):
    two scalars, nothing fitted beyond the two shrunken rates.
    """
    table, g = item_difficulty(ref_first, exclude)
    d = eval_first["problem_id"].map(table).fillna(g).to_numpy()
    a = causal_ability(eval_first, g)
    both = 1 / (1 + np.exp(-(logit(d) + logit(a) - logit(g))))
    return pd.DataFrame({"p_global": np.full(len(d), g), "p_difficulty": d,
                         "p_ability": a, "p_two_scalar": both}, index=eval_first.index)


# ----------------------------------------------------------------------------- ID-only GRU


def _sequences(first: pd.DataFrame, qid: dict, holdout: set) -> list[tuple]:
    out = []
    for _, g in first.groupby("learner", sort=False):
        out.append((g["problem_id"].map(qid).to_numpy(), g["y"].to_numpy(),
                    g["problem_id"].isin(holdout).to_numpy()))
    return out


def run_dkt(train_first: pd.DataFrame, val_first: pd.DataFrame, holdout: set,
            exclude_holdout: bool, seed: int) -> np.ndarray:
    """Train a GRU over (problem id, pass) sequences; return P(pass) for every val row.

    Input at step k is the embedding of (problem_{k-1}, pass_{k-1}); the prediction for step k
    reads the GRU state together with an embedding of problem_k, so a problem whose embedding
    was never trained still receives a prediction carried by the learner state.
    """
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    items = sorted(set(train_first["problem_id"]) | set(val_first["problem_id"]))
    qid = {p: i for i, p in enumerate(items)}
    n_items = len(items)

    class GRUKT(nn.Module):
        def __init__(self):
            super().__init__()
            self.inp = nn.Embedding(2 * n_items + 1, DKT_DIM)  # last index = start token
            self.gru = nn.GRU(DKT_DIM, DKT_DIM, batch_first=True)
            self.item = nn.Embedding(n_items, DKT_DIM)
            self.head = nn.Sequential(nn.Linear(2 * DKT_DIM, DKT_DIM), nn.ReLU(), nn.Linear(DKT_DIM, 1))

        def forward(self, q, y):
            start = torch.full_like(q[:, :1], 2 * n_items)
            h, _ = self.gru(self.inp(torch.cat([start, (2 * q + y)[:, :-1]], 1)))
            return self.head(torch.cat([h, self.item(q)], -1)).squeeze(-1)

    def pad(batch):
        L = max(len(b[0]) for b in batch)
        q = torch.zeros(len(batch), L, dtype=torch.long)
        y = torch.zeros(len(batch), L, dtype=torch.long)
        m = torch.zeros(len(batch), L, dtype=torch.bool)
        for k, (qq, yy, _) in enumerate(batch):
            q[k, :len(qq)], y[k, :len(qq)], m[k, :len(qq)] = torch.tensor(qq), torch.tensor(yy), True
        return q, y, m

    train = train_first[~train_first["problem_id"].isin(holdout)] if exclude_holdout else train_first
    tr = _sequences(train, qid, holdout)
    model = GRUKT()
    opt = torch.optim.Adam(model.parameters(), lr=DKT_LR)
    loss_fn = nn.BCEWithLogitsLoss()
    for _ in range(DKT_EPOCHS):
        perm = rng.permutation(len(tr))
        for s in range(0, len(tr), DKT_BATCH):
            q, y, m = pad([tr[j] for j in perm[s:s + DKT_BATCH]])
            loss = loss_fn(model(q, y)[m], y[m].float())
            opt.zero_grad()
            loss.backward()
            opt.step()

    model.eval()
    preds = []
    with torch.no_grad():
        for seq in _sequences(val_first, qid, holdout):
            q, y, _ = pad([seq])
            preds.append(torch.sigmoid(model(q, y))[0].numpy())
    return np.concatenate(preds).astype(float)


# ----------------------------------------------------------------------------- main


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--repo-root", default=str(DEFAULT_REPO))
    ap.add_argument("--raw-dir", default=None, help="CodeWorkout release dir (default ../Datasets/CodeWorkout_CMU)")
    ap.add_argument("--out-dir", default=None, help="default <repo>/results/cw_first_submission_baselines")
    ap.add_argument("--dkt-seeds", default="0,1,2,3,4")
    ap.add_argument("--skip-dkt", action="store_true")
    args = ap.parse_args()

    repo = pathlib.Path(args.repo_root).resolve()
    out_dir = pathlib.Path(args.out_dir) if args.out_dir else repo / "results" / "cw_first_submission_baselines"
    holdout = held_out_items(repo)
    H = set(holdout)
    summary: dict = {"metric": "AUC of P(passes all tests) on first submissions",
                     "held_out_items": holdout, "n_boot": N_BOOT, "bootstrap": "learner-clustered percentile",
                     "seed": SEED}

    # --- ATOM, from its saved validation predictions
    atom_all = load_atom_rows(repo)
    atom_first = first_rows(atom_all, "learner", "problem_id", "t")
    unseen = atom_first["problem_id"].isin(H).to_numpy()
    later = (atom_all["t"] > 0).to_numpy()  # every attempt except the learner's very first
    summary["atom"] = {
        "first_submission": scored(atom_first["y"], atom_first["p_atom"], atom_first["learner"]),
        "first_submission_seen_problems": scored(atom_first["y"][~unseen], atom_first["p_atom"][~unseen],
                                                 atom_first["learner"][~unseen]),
        "first_submission_held_out_problems": scored(atom_first["y"][unseen], atom_first["p_atom"][unseen],
                                                     atom_first["learner"][unseen]),
        "all_attempts": scored(atom_all["y"], atom_all["p_atom"], atom_all["learner"], ci=False),
        "attempts_after_first": scored(atom_all["y"][later], atom_all["p_atom"][later],
                                       atom_all["learner"][later], ci=False),
        "within_problem_macro_first_submission": macro_auc(atom_first, "problem_id", "p_atom"),
        "within_learner_macro_first_submission": macro_auc(atom_first, "learner", "p_atom", min_rows=10),
    }

    # --- reference learners for item statistics: every learner outside validation
    corpus = load_corpus_attempts(repo, args.raw_dir)
    corpus_first = first_rows(corpus, "learner", "problem_id", "t")
    ref_first = corpus_first[corpus_first["split"] != "val"]
    train_first = corpus_first[corpus_first["split"] == "train"]

    # --- two-scalar baseline on ATOM's own first-submission rows
    b = two_scalar_predictions(atom_first, ref_first)
    lr = atom_first["learner"]
    summary["two_scalar"] = {
        "reference_learners": "all non-validation learners", "n_reference_first_submissions": int(len(ref_first)),
        "difficulty_prior": DIFFICULTY_PRIOR, "ability_prior": ABILITY_PRIOR,
        "global_rate": scored(atom_first["y"], b["p_global"], lr, ci=False),
        "difficulty_only": scored(atom_first["y"], b["p_difficulty"], lr),
        "ability_only": scored(atom_first["y"], b["p_ability"], lr),
        "difficulty_plus_ability": scored(atom_first["y"], b["p_two_scalar"], lr),
    }
    bt = two_scalar_predictions(atom_first, train_first)
    summary["two_scalar"]["difficulty_plus_ability_train_learners_only"] = scored(
        atom_first["y"], bt["p_two_scalar"], lr, ci=False)
    bh = two_scalar_predictions(atom_first, ref_first, exclude=H)
    held = atom_first[unseen].reset_index(drop=True)
    _, g = item_difficulty(ref_first)
    summary["two_scalar"]["held_out_problems"] = {
        "difficulty_only": scored(atom_first["y"][unseen], bh["p_difficulty"][unseen], lr[unseen], ci=False),
        # Thesis value: ability as a running pass rate over the learner's earlier
        # first submissions to the seven held-out problems only.
        "ability_only_held_out_history": scored(held["y"], causal_ability(held, g), held["learner"]),
        # Same estimator with the learner's full first-submission history.
        "ability_only_full_history": scored(atom_first["y"][unseen], b["p_ability"][unseen], lr[unseen]),
    }

    # --- ID-only GRU, trained on the training learners, scored on validation first submissions
    if not args.skip_dkt:
        val_first = corpus_first[corpus_first["split"] == "val"].reset_index(drop=True)
        vu = val_first["problem_id"].isin(H).to_numpy()
        on_atom = val_first.merge(atom_first[["learner", "problem_id", "t"]], how="left",
                                  on=["learner", "problem_id", "t"], indicator=True)["_merge"].eq("both").to_numpy()
        runs = []
        for seed in [int(s) for s in args.dkt_seeds.split(",")]:
            p_h = run_dkt(train_first, val_first, H, exclude_holdout=True, seed=seed)
            p_l = run_dkt(train_first, val_first, H, exclude_holdout=False, seed=seed)
            r = {"seed": seed,
                 "seen_problems": auc(val_first["y"][~vu], p_h[~vu]),
                 "seen_problems_on_atom_rows": auc(val_first["y"][~vu & on_atom], p_h[~vu & on_atom]),
                 "held_out_problems_never_trained": auc(val_first["y"][vu], p_h[vu]),
                 "held_out_problems_trained_on_all_50": auc(val_first["y"][vu], p_l[vu])}
            runs.append(r)
            print("GRU seed", seed, {k: round(v, 4) for k, v in r.items() if k != "seed"}, flush=True)
        keys = [k for k in runs[0] if k != "seed"]
        summary["id_only_gru"] = {
            "history": "each learner's first submissions in causal order",
            "train_learners": "training split only", "epochs": DKT_EPOCHS, "dim": DKT_DIM,
            "batch": DKT_BATCH, "lr": DKT_LR,
            "n_seen_rows": int((~vu).sum()), "n_seen_rows_on_atom_rows": int((~vu & on_atom).sum()),
            "n_held_out_rows": int(vu.sum()), "n_learners": int(val_first["learner"].nunique()),
            "per_seed": runs,
            "mean": {k: float(np.mean([r[k] for r in runs])) for k in keys},
            "min": {k: float(np.min([r[k] for r in runs])) for k in keys},
            "max": {k: float(np.max([r[k] for r in runs])) for k in keys},
        }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    a, t = summary["atom"], summary["two_scalar"]
    print(f"ATOM first submission     {a['first_submission']['auc']:.4f}  n={a['first_submission']['n']}")
    print(f"ATOM within-problem macro {a['within_problem_macro_first_submission']['auc']:.4f}")
    print(f"difficulty + ability      {t['difficulty_plus_ability']['auc']:.4f}")
    print(f"ability, held-out items   {t['held_out_problems']['ability_only_held_out_history']['auc']:.4f}"
          f" (full history {t['held_out_problems']['ability_only_full_history']['auc']:.4f})")
    print("wrote", out_dir / "summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
