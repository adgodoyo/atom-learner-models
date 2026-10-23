#!/usr/bin/env python
"""Re-score the AKT, IRT and Majority baselines on FA-SIM3's own 4,977-row evaluation court.

The published baseline numbers use different populations (AKT: every scoreable position of
the val chunks; IRT: the eval half of each val learner's history), so they are not directly
comparable with FA-SIM3. This script puts all predictors on the identical 4,977 outcome-valid
rows of FA-SIM3's evaluation bundle, scores them against the same label, and saves every
per-row prediction.

Inputs: `results/fa_sim3_full_run/{predictions.csv.gz,belief_diagnostics.npz}`, the v2 FA
manifest and splits, and `configs/{irt,akt_baseline,data}.yaml`. Outputs:
`results/fa_court_baselines/{panel.json,court_predictions.csv.gz}`.
Run: `python scripts/fa_01p_court_baselines.py` (MPS/CUDA for AKT, CPU for IRT; about 20 min,
dominated by the AKT retrain). Serves Table 8 (FA locating).

Label. `y = (true_class == 0)` = UNAIDED_CORRECT, the same vector for every arm. On this court
it coincides with the platform `score` on 4,972 of 4,977 rows (FA forfeits credit on
hint/reveal, so an aided row is a 0 either way).

IRT is made causal. The published IRT protocol calibrates theta on the first 50% of each val
learner's history and evaluates on the second half, but 2,664 of the 4,977 court rows (53.5%)
lie in that first half, because FA-SIM3's evaluation bundle over-samples cold-start rows; the
published protocol would therefore calibrate on its own evaluation rows. Here, for each court
row (learner, item, t), theta is MAP-estimated from that learner's labelled events with t' < t
only, against train-frozen item parameters: the same information set as the query-independent
belief b_t = f(H_<t). Rows with no prior labelled event fall back to the prior mean theta = 0
(cold start, reported). All causal thetas are fitted in one `fit_2pl` call by treating each
court row as its own "person" whose observations are its causal prefix, with the same
optimiser, epochs, learning rate and Gaussian prior as the published calibration stage: a
change of information set, not of estimator. The published-protocol number is also reported,
on the 2,313 court rows that fall in the eval half, so the change can be audited.

AKT is causal by construction (pyKT's monotonic attention and `mask[:, 0] = False`), so only
row selection changes. The model is retrained identically to `scripts/fa_01_akt_v2.py` (same
config, seed, chunking and exclusions); its per-position predictions are mapped back to
(learner, problem_id, t) keys by re-running `chunk_learner_history`, the same deterministic call
`build_akt_batch_windowed` makes internally, so alignment is guaranteed rather than assumed.
Coverage caveat: a court row at position 0 of its chunk has no in-chunk history and is never
scored under pyKT's convention, so AKT's court is a subset; the other arms are reported both
on the full court and on AKT's subset.
"""

from __future__ import annotations

import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from atom.baselines.akt_data import (  # noqa: E402
    build_akt_batch_windowed,
    build_concept_map,
    build_pid_map,
    check_coverage,
    chunk_learner_history,
)
from atom.data.fa import load_problems, load_skills  # noqa: E402
from atom.data.fa_manifest import load_v2_events  # noqa: E402
from atom.eval.protocol import calib_eval_split  # noqa: E402
from atom.models.irt import fit_2pl, predict_2pl  # noqa: E402
from atom.utils.seeds import seed_everything  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[1]
RUN = REPO / "results" / "fa_sim3_full_run"
OUT = REPO / "results" / "fa_court_baselines"
KEY = ["learner", "problem_id", "t"]

# Bootstrap settings copied from scripts/fa_sim3_kt_panel.py, so ATOM's CIs here are the same
# quantity as the ones that script reports.
NBOOT = 1000
BOOT_SEED = 20260824


# ----------------------------------------------------------------- metrics
def cluster_boot(pack: np.ndarray, learners: np.ndarray, stat, nboot=NBOOT, seed=BOOT_SEED):
    """Learner-clustered bootstrap: resample LEARNERS with replacement, not rows."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(learners)
    idx_by_learner = {l: np.flatnonzero(learners == l) for l in uniq}
    point = stat(pack)
    draws = np.empty(nboot)
    for b in range(nboot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        rows = np.concatenate([idx_by_learner[l] for l in pick])
        draws[b] = stat(pack[rows])
    lo, hi = np.nanpercentile(draws, [2.5, 97.5])
    return float(point), float(lo), float(hi)


def _ece(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    out = 0.0
    for i in range(bins):
        m = (p >= edges[i]) & (p < edges[i + 1] if i < bins - 1 else p <= 1.0)
        if m.sum():
            out += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(out)


def score_arm(y: np.ndarray, p: np.ndarray, learners: np.ndarray, thr: float = 0.5,
              boot: bool = True) -> dict:
    """Every cell the thesis table needs, for one predictor on one court."""
    yhat = (p >= thr).astype(int)
    correct_rows, wrong_rows = y == 1, y == 0
    d = {
        "n": int(len(y)),
        "n_learners": int(len(np.unique(learners))),
        "base_rate": round(float(y.mean()), 4),
        "auc": round(float(roc_auc_score(y, p)), 4) if y.min() != y.max() else None,
        "accuracy": round(float((yhat == y).mean()), 4),
        # the two blank cells: accuracy conditioned on what the student actually did
        "acc_when_student_correct": round(float((yhat[correct_rows] == 1).mean()), 4),
        "acc_when_student_wrong": round(float((yhat[wrong_rows] == 0).mean()), 4),
        "n_student_correct": int(correct_rows.sum()),
        "n_student_wrong": int(wrong_rows.sum()),
        "balanced_accuracy": None,
        "ece": round(_ece(y, p), 4),
        "brier": round(float(np.mean((p - y) ** 2)), 4),
        "pred_positive_rate": round(float(yhat.mean()), 4),
    }
    d["balanced_accuracy"] = round(
        0.5 * (d["acc_when_student_correct"] + d["acc_when_student_wrong"]), 4)
    if boot:
        pack = np.stack([y, p], axis=1)
        _, lo, hi = cluster_boot(
            pack, learners,
            lambda a: roc_auc_score(a[:, 0], a[:, 1]) if a[:, 0].min() != a[:, 0].max() else np.nan)
        d["auc_ci95"] = [round(lo, 4), round(hi, 4)]
        for cell, sel in (("acc_when_student_correct", 1.0), ("acc_when_student_wrong", 0.0)):
            sub = pack[y == sel]
            subl = learners[y == sel]
            tgt = 1 if sel == 1.0 else 0
            _, lo, hi = cluster_boot(
                sub, subl, lambda a, t=tgt, th=thr: float(((a[:, 1] >= th).astype(int) == t).mean()))
            d[cell + "_ci95"] = [round(lo, 4), round(hi, 4)]
    return d


# ----------------------------------------------------------------- main
def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    irt_cfg = yaml.safe_load((REPO / "configs" / "irt.yaml").read_text())
    akt_cfg = yaml.safe_load((REPO / "configs" / "akt_baseline.yaml").read_text())
    v2cfg = yaml.safe_load((REPO / "configs" / "data.yaml").read_text())["fa"]["manifest_v2"]
    seed_everything(irt_cfg["seed"])

    # -- 1. the court, exactly as the paper-comparability panel defines it -------------------
    pred = pd.read_csv(RUN / "predictions.csv.gz")
    diag = np.load(RUN / "belief_diagnostics.npz")
    court = pred[KEY + ["true_class"]].copy()
    court["y"] = (court["true_class"] == 0).astype(int)
    for arm in ("personalised", "shuffled_u"):
        court[f"p_atom_{arm}"] = diag[f"outcome_prob_meanK_full_{arm}"][:, 0]
    assert len(court) == 4977, len(court)
    print(f"[court] {len(court)} rows · {court.learner.nunique()} learners · "
          f"base rate {court.y.mean():.4f}", flush=True)

    # -- 2. the v2 event stream (identical loader/splits to both baseline scripts) -----------
    ev = load_v2_events(REPO / v2cfg["manifest_out"], REPO / v2cfg["learner_split_out"],
                        REPO / v2cfg["item_split_out"])
    ev = ev[ev["split"].isin(["train", "val"])].copy()
    all_pids = sorted(ev["problem_id"].unique())
    item_of = {p: i for i, p in enumerate(all_pids)}
    ev["item"] = ev["problem_id"].map(item_of)
    train_labeled = ev[(ev["split"] == "train") & (~ev["score_missing"])]
    val_labeled = ev[(ev["split"] == "val") & (~ev["score_missing"])]

    # assert the court really is a subset of the val stream, keyed exactly
    merged = court.merge(val_labeled[KEY + ["item", "score", "support_code"]], on=KEY, how="left")
    assert len(merged) == len(court) and merged["item"].notna().all(), "court not a val subset"
    court["item"] = merged["item"].astype(int).to_numpy()
    court["support_code"] = merged["support_code"].astype(int).to_numpy()
    agree = float((merged["score"].astype(float) == court["y"]).mean())
    print(f"[court] y == platform score on {agree:.4%} of rows", flush=True)

    # -- 3. Majority: the TRAIN labelled correct rate, exactly as fa_01_majority_irt_v2 ------
    maj_const = float(train_labeled["score"].astype("float32").mean())
    court["p_majority"] = maj_const
    print(f"[majority] constant = {maj_const:.4f}", flush=True)

    # -- 4. IRT item params on TRAIN (identical call to fa_01_majority_irt_v2) ---------------
    tr_learners = sorted(train_labeled["learner"].unique())
    tperson = {u: i for i, u in enumerate(tr_learners)}
    item_params = fit_2pl(
        torch.tensor(train_labeled["learner"].map(tperson).values, dtype=torch.long),
        torch.tensor(train_labeled["item"].values, dtype=torch.long),
        torch.tensor(train_labeled["score"].astype("int64").values, dtype=torch.long),
        n_persons=len(tr_learners), n_items=len(all_pids),
        epochs=irt_cfg["train_epochs"], lr=irt_cfg["train_lr"],
        l2_theta=irt_cfg["l2_theta"], l2_log_a=irt_cfg["l2_log_a"], l2_b=irt_cfg["l2_b"],
        seed=irt_cfg["seed"])
    print(f"[irt] item params fit on {len(train_labeled):,} TRAIN rows", flush=True)

    # -- 5. CAUSAL IRT theta: one theta per court row, from that row's strict prefix ---------
    # Each court row becomes its own "person"; its observations are that learner's labelled
    # val events with t' < t. One fit_2pl call, same optimiser/prior/epochs as the published
    # calibration stage -- a change of information set, not of estimator.
    val_by_learner = {u: g.sort_values("t", kind="mergesort")
                      for u, g in val_labeled.groupby("learner", sort=False)}
    rows_person, rows_item, rows_y = [], [], []
    n_prefix = np.zeros(len(court), dtype=int)
    for ridx, (lrn, t_row) in enumerate(zip(court["learner"].to_numpy(), court["t"].to_numpy())):
        g = val_by_learner.get(lrn)
        if g is None:
            continue
        pre = g[g["t"] < t_row]
        if not len(pre):
            continue
        n_prefix[ridx] = len(pre)
        rows_person.append(np.full(len(pre), ridx, dtype=np.int64))
        rows_item.append(pre["item"].to_numpy())
        rows_y.append(pre["score"].astype("int64").to_numpy())
    cp = torch.tensor(np.concatenate(rows_person), dtype=torch.long)
    ci = torch.tensor(np.concatenate(rows_item), dtype=torch.long)
    cy = torch.tensor(np.concatenate(rows_y), dtype=torch.long)
    print(f"[irt] causal theta: {len(court)} thetas from {len(cp):,} prefix observations "
          f"(median prefix {int(np.median(n_prefix))}, {int((n_prefix == 0).sum())} cold-start)",
          flush=True)
    causal_params = fit_2pl(cp, ci, cy, n_persons=len(court), n_items=len(all_pids),
                            epochs=irt_cfg["calib_epochs"], lr=irt_cfg["calib_lr"],
                            l2_theta=irt_cfg["l2_theta"], fixed_log_a=item_params.log_a,
                            fixed_b=item_params.b, seed=irt_cfg["seed"])
    court["p_irt_causal"] = predict_2pl(
        causal_params,
        torch.arange(len(court), dtype=torch.long),
        torch.tensor(court["item"].to_numpy(), dtype=torch.long)).numpy()
    court["n_prefix"] = n_prefix

    # -- 6. IRT under its PUBLISHED protocol, on the court rows that fall in the eval half ---
    val_ev = ev[ev["split"] == "val"]
    calib_rows, eval_rows = calib_eval_split(val_ev, irt_cfg["calib_frac"],
                                             irt_cfg["min_interactions"])
    calib_lab = calib_rows[~calib_rows["score_missing"]]
    eval_keys = set(map(tuple, eval_rows[KEY].to_numpy()))
    in_eval_half = np.array([tuple(k) in eval_keys for k in court[KEY].to_numpy()])
    pub_learners = sorted(set(calib_lab["learner"]) | set(eval_rows["learner"]))
    pperson = {u: i for i, u in enumerate(pub_learners)}
    pub_params = fit_2pl(
        torch.tensor(calib_lab["learner"].map(pperson).values, dtype=torch.long),
        torch.tensor(calib_lab["item"].values, dtype=torch.long),
        torch.tensor(calib_lab["score"].astype("int64").values, dtype=torch.long),
        n_persons=len(pub_learners), n_items=len(all_pids),
        epochs=irt_cfg["calib_epochs"], lr=irt_cfg["calib_lr"], l2_theta=irt_cfg["l2_theta"],
        fixed_log_a=item_params.log_a, fixed_b=item_params.b, seed=irt_cfg["seed"])
    court["p_irt_published"] = predict_2pl(
        pub_params,
        torch.tensor(court["learner"].map(pperson).fillna(0).astype(int).to_numpy(),
                     dtype=torch.long),
        torch.tensor(court["item"].to_numpy(), dtype=torch.long)).numpy()
    court["in_eval_half"] = in_eval_half
    print(f"[irt] published protocol: {int(in_eval_half.sum())} of {len(court)} court rows "
          f"lie in the eval half ({int((~in_eval_half).sum())} would be IN-SAMPLE)", flush=True)

    # -- 7. AKT: retrain bit-identically to fa_01_akt_v2, then scatter per-row predictions ---
    import pykt.models.akt as _akt_module
    from pykt.models.akt import AKT
    from atom.utils.logging import default_device

    device = default_device()
    akt_ev = ev[~ev["score_missing"]].copy()
    akt_ev["correct"] = akt_ev["score"].astype("int64")
    train_ev, val_ev_akt = akt_ev[akt_ev["split"] == "train"], akt_ev[akt_ev["split"] == "val"]
    skills, problems = load_skills(), load_problems()
    concept_map, pid_map = build_concept_map(skills), build_pid_map(problems["problem_id"])
    check_coverage(akt_ev["problem_id"].unique(), concept_map, pid_map)
    n_question, n_pid = len(set(concept_map.values())), len(set(pid_map.values()))
    max_seq_len = akt_cfg["max_seq_len"]
    train_batch = build_akt_batch_windowed(train_ev, concept_map, pid_map, max_seq_len, device="cpu")
    val_batch = build_akt_batch_windowed(val_ev_akt, concept_map, pid_map, max_seq_len, device="cpu")

    if str(_akt_module.device) != str(torch.device(device)):
        _akt_module.device = torch.device(device)
    torch.manual_seed(akt_cfg["seed"])
    model = AKT(n_question=n_question, n_pid=n_pid, **akt_cfg["model"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=akt_cfg["lr"],
                           weight_decay=akt_cfg["weight_decay"])
    n_train, bs, epochs = train_batch["concepts"].shape[0], akt_cfg["full_run_batch_size"], akt_cfg["epochs"]
    rng = torch.Generator().manual_seed(akt_cfg["seed"])
    print(f"[akt] training {epochs} epochs on {n_train} chunks, device={device}", flush=True)
    t_akt = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n_train, generator=rng)
        tot, cnt = 0.0, 0
        for bstart in range(0, n_train, bs):
            idx = perm[bstart:bstart + bs]
            c, p_, r = (train_batch[k][idx].to(device) for k in ("concepts", "pids", "responses"))
            m = train_batch["valid_mask"][idx].clone()
            m[:, 0] = False
            m = m.to(device)
            opt.zero_grad()
            preds, reg_loss = model(c, r, p_)
            pr = preds.clamp(1e-6, 1 - 1e-6)
            bce = -(r.float() * pr.log() + (1 - r.float()) * (1 - pr).log())
            loss = (bce * m.float()).sum() / m.float().sum().clamp(min=1) + reg_loss
            loss.backward()
            opt.step()
            tot += loss.item() * m.sum().item()
            cnt += m.sum().item()
        print(f"[akt] epoch {epoch}/{epochs} loss {tot / max(cnt, 1):.4f} "
              f"({time.time() - t_akt:.0f}s)", flush=True)
    model.eval()
    with torch.no_grad():
        val_preds, _ = model(val_batch["concepts"].to(device),
                             val_batch["responses"].to(device),
                             val_batch["pids"].to(device))
    val_preds = val_preds.detach().cpu().numpy()
    val_mask = val_batch["valid_mask"].clone().numpy()
    val_mask[:, 0] = False   # pyKT convention: position 0 has no in-chunk history

    # scatter (chunk, position) -> (learner, problem_id, t) via the SAME chunk_learner_history
    chunked = chunk_learner_history(val_ev_akt, max_seq_len).sort_values(
        ["chunk_id", "t"], kind="mergesort")
    chunked = chunked.assign(t_local=chunked.groupby("chunk_id", sort=False).cumcount())
    chunk_row = {c: i for i, c in enumerate(val_batch["learners"])}
    chunked["n_idx"] = chunked["chunk_id"].map(chunk_row)
    ok = chunked["n_idx"].notna() & (chunked["t_local"] < val_preds.shape[1])
    chunked = chunked[ok].copy()
    ni, ti = chunked["n_idx"].astype(int).to_numpy(), chunked["t_local"].to_numpy()
    chunked["p_akt"] = val_preds[ni, ti]
    chunked["akt_scored"] = val_mask[ni, ti]
    akt_rows = chunked.loc[chunked["akt_scored"], KEY + ["p_akt"]]
    court = court.merge(akt_rows, on=KEY, how="left")
    n_akt = int(court["p_akt"].notna().sum())
    print(f"[akt] {n_akt} of {len(court)} court rows scoreable "
          f"({len(court) - n_akt} at chunk position 0, never scored by pyKT convention) · "
          f"{time.time() - t_akt:.0f}s", flush=True)

    # -- 8. score everything ------------------------------------------------------------------
    y_all, lr_all = court["y"].to_numpy(), court["learner"].to_numpy()
    arms_full = {
        "ATOM (personalised)": "p_atom_personalised",
        "ATOM (shuffled-u control)": "p_atom_shuffled_u",
        "AKT v2 (skill tags)": "p_akt",
        "IRT-2PL v2 (causal theta)": "p_irt_causal",
        "Majority": "p_majority",
    }
    akt_mask = court["p_akt"].notna().to_numpy()
    results = {"full_court": {}, "akt_common_court": {}, "irt_published_protocol": {}}
    for name, col in arms_full.items():
        p = court[col].to_numpy()
        if name.startswith("AKT"):
            results["full_court"][name] = {
                "n": int(akt_mask.sum()),
                "note": "AKT cannot score the 4,977-row court in full; see akt_common_court"}
        else:
            results["full_court"][name] = score_arm(y_all, p, lr_all)
        results["akt_common_court"][name] = score_arm(
            y_all[akt_mask], p[akt_mask], lr_all[akt_mask])
    m = in_eval_half
    results["irt_published_protocol"] = {
        "note": "IRT under its own published calib/eval protocol, restricted to the "
                "court rows that fall in the eval half (the other 2,664 would be in-sample).",
        "IRT-2PL v2 (published protocol)": score_arm(
            y_all[m], court["p_irt_published"].to_numpy()[m], lr_all[m]),
        "IRT-2PL v2 (causal theta, same rows)": score_arm(
            y_all[m], court["p_irt_causal"].to_numpy()[m], lr_all[m]),
        "ATOM (personalised, same rows)": score_arm(
            y_all[m], court["p_atom_personalised"].to_numpy()[m], lr_all[m]),
    }

    payload = {
        "court": {"source": "results/fa_sim3_full_run/predictions.csv.gz",
                  "n_rows": int(len(court)), "n_learners": int(court.learner.nunique()),
                  "label": "y = (true_class == 0) = UNAIDED_CORRECT",
                  "base_rate": round(float(y_all.mean()), 4),
                  "y_equals_platform_score_frac": round(agree, 6)},
        "protocol_notes": {
            "irt_court_rows_in_calib_half": int((~in_eval_half).sum()),
            "irt_court_rows_in_eval_half": int(in_eval_half.sum()),
            "irt_causal_cold_start_rows": int((n_prefix == 0).sum()),
            "irt_causal_median_prefix": int(np.median(n_prefix)),
            "akt_scoreable_court_rows": n_akt,
            "akt_unscoreable_reason": "chunk position 0 -- no in-chunk history (pyKT convention)",
            "majority_constant": round(maj_const, 4),
            "threshold": 0.5, "nboot": NBOOT, "boot_seed": BOOT_SEED,
        },
        "results": results,
        "wall_s": round(time.time() - t_start, 1),
    }
    (OUT / "panel.json").write_text(json.dumps(payload, indent=1))
    keep = KEY + ["true_class", "y", "support_code", "n_prefix", "in_eval_half",
                  "p_atom_personalised", "p_atom_shuffled_u", "p_akt", "p_irt_causal",
                  "p_irt_published", "p_majority"]
    court[keep].to_csv(OUT / "court_predictions.csv.gz", index=False, compression="gzip")

    # -- 9. print the table -------------------------------------------------------------------
    def show(title, block):
        print(f"\n=== {title} ===")
        hdr = (f"{'arm':34s}{'n':>7s}{'AUC':>8s}{'acc':>8s}{'when correct':>14s}"
               f"{'when wrong':>13s}{'bal acc':>9s}{'ECE':>7s}")
        print(hdr)
        for k, v in block.items():
            if "auc" not in v:
                print(f"{k:34s}{v['n']:>7d}{'--':>8s}{'--':>8s}{'--':>14s}{'--':>13s}"
                      f"{'--':>9s}{'--':>7s}")
                continue
            print(f"{k:34s}{v['n']:>7d}{v['auc']:>8.4f}{v['accuracy']:>8.4f}"
                  f"{v['acc_when_student_correct']:>14.4f}{v['acc_when_student_wrong']:>13.4f}"
                  f"{v['balanced_accuracy']:>9.4f}{v['ece']:>7.4f}")

    show("A. FULL COURT (4,977 rows) -- the thesis table's population", results["full_court"])
    show(f"B. AKT-COMMON COURT ({int(akt_mask.sum())} rows) -- all four arms comparable",
         results["akt_common_court"])
    show(f"C. IRT PROTOCOL AUDIT ({int(m.sum())} eval-half rows)",
         {k: v for k, v in results["irt_published_protocol"].items() if k != "note"})
    print(f"\nwrote {OUT / 'panel.json'} and {OUT / 'court_predictions.csv.gz'}")
    print(f"total wall {time.time() - t_start:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
