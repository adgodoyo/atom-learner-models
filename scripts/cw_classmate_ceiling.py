"""Similarity of ATOM's generated code to a learner's real submission, against two learner-blind references.
Inputs: results/cw_sim_launch2/headline/ (generations_raw.jsonl, generation_numeric_rows.npz) and the CodeWorkout logs (for belief maturity).
References per row: the mean similarity to every same-problem classmate's real submission, and the leave-one-out medoid of those submissions.
Outputs: a private per-row table (results/cw_classmate_ceiling/rows_private.csv, never published), results/cw_classmate_ceiling/summary.json and paper/figures/data/fig09_cw_ceiling_bins.csv (aggregates only).
Thesis Section 7.3 and Figure 9.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

import numpy as np
import pandas as pd

REPO = pathlib.Path(os.environ.get("ATOM_REPO", pathlib.Path(__file__).resolve().parents[1]))
sys.path[:0] = [str(REPO / "src"), str(REPO / "scripts")]

HEADLINE = REPO / "results" / "cw_sim_launch2" / "headline"
ARM, MODE = "personalised", "ancestral"          # blind ancestral decoding, personalised belief
# Semantic scale anchor: mean codec cosine between real submissions to different problems.
# 0 = unrelated course Java, 1 = the learner's exact submission. Full-precision value of the
# published 0.8765 anchor.
SEM_FLOOR = 0.8764547420
NEAR_MATCH = 0.90                                 # panel (b) threshold
STRATA = [(0, 0, "cold start"), (1, 8, "1-8"), (9, 16, "9-16"), (17, 36, "17-36"), (37, 84, "37+")]
N_BOOT = 4000


# ----------------------------------------------------------------------------- per-row table
def char_similarity(a: str, b: str) -> float:
    from atom.eval.cw_generation import normalised_edit_distance
    return 1.0 - normalised_edit_distance(a, b)


def load_rows() -> pd.DataFrame:
    """One row per (learner, problem) first attempt in the headline panel."""
    recs = [json.loads(l) for l in open(HEADLINE / "generations_raw.jsonl")]
    num = np.load(HEADLINE / "generation_numeric_rows.npz")
    assert all(int(r["row_idx"]) == int(num["row_idx"][i]) for i, r in enumerate(recs))
    keep = [i for i, r in enumerate(recs) if r["arm"] == ARM and r["generation_mode"] == MODE]
    df = pd.DataFrame({
        "learner": [recs[i]["learner"] for i in keep],
        "problem_id": [int(recs[i]["problem_id"]) for i in keep],
        "code_state_id": [recs[i]["code_state_id"] for i in keep],
        "generated": [recs[i]["generated_code"] for i in keep],
        "real": [recs[i]["real_completion"] for i in keep],
        "codec_cos": num["codec_similarity"][keep],
    })
    assert not df.duplicated(["learner", "problem_id"]).any()
    return df


def attach_maturity(df: pd.DataFrame) -> pd.DataFrame:
    """Episodes already in the belief = index of this attempt's episode in the learner's history."""
    from atom.data.cw_manifest import load_attempts_with_episodes
    att = load_attempts_with_episodes()
    att["ep"] = att.groupby("learner", sort=False)["episode_id"].transform(lambda s: pd.factorize(s)[0])
    ep = att.set_index("code_state_id")["ep"]
    missing = ~df["code_state_id"].isin(ep.index)
    assert not missing.any(), f"{int(missing.sum())} rows without an episode"
    df["ep"] = ep.loc[df["code_state_id"]].to_numpy()
    return df


def reference_scores(S: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """S: within-problem similarity matrix between real submissions (diagonal ignored).
    Returns per row the classmate average and the similarity to the leave-one-out medoid:
    the submission, among the other learners', with the highest mean similarity to the rest
    of those other learners' submissions."""
    n = len(S)
    D = 1.0 - S
    np.fill_diagonal(D, 0.0)
    peer, medoid = np.empty(n), np.empty(n)
    for a in range(n):
        others = np.delete(np.arange(n), a)
        peer[a] = S[a, others].mean()
        centrality = D[np.ix_(others, others)].sum(1) / (n - 2)
        medoid[a] = S[a, others[int(np.argmin(centrality))]]
    return peer, medoid


def build_table(real_vectors: pathlib.Path | None) -> pd.DataFrame:
    df = attach_maturity(load_rows())
    df["char_atom"] = [char_similarity(g, r) for g, r in zip(df["generated"], df["real"])]
    df["sem_atom"] = (df["codec_cos"] - SEM_FLOOR) / (1.0 - SEM_FLOOR)
    vecs = None
    if real_vectors is not None:
        z = np.load(real_vectors, allow_pickle=True)
        vecs = dict(zip(z["code_state_id"].tolist(), z["vector"]))
    for col in ("char_peeravg", "char_medoid", "sem_peeravg", "sem_medoid"):
        df[col] = np.nan
    for _, g in df.groupby("problem_id"):
        real = g["real"].tolist()
        n = len(real)
        S = np.ones((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                S[i, j] = S[j, i] = char_similarity(real[i], real[j])
        p, m = reference_scores(S)
        df.loc[g.index, "char_peeravg"], df.loc[g.index, "char_medoid"] = p, m
        if vecs is not None:
            V = np.stack([vecs[c] for c in g["code_state_id"]]).astype(np.float64)
            V /= np.linalg.norm(V, axis=1, keepdims=True) + 1e-12
            Sn = (V @ V.T - SEM_FLOOR) / (1.0 - SEM_FLOOR)
            p, m = reference_scores(Sn)
            df.loc[g.index, "sem_peeravg"], df.loc[g.index, "sem_medoid"] = p, m
    order = {l: i for i, l in enumerate(dict.fromkeys(df["learner"]))}
    df["lr"] = df["learner"].map(order)
    return df.drop(columns=["generated", "real"])


# ----------------------------------------------------------------------------- aggregation
def _boot(rng, groups, uniq, statfn, n_boot=N_BOOT):
    out = np.empty(n_boot)
    for k in range(n_boot):
        rows = np.concatenate([groups[u] for u in rng.choice(uniq, size=len(uniq), replace=True)])
        out[k] = statfn(rows)
    return np.percentile(out, [2.5, 97.5])


def cluster_ci(rng, lr, statfn):
    """Learner-clustered 95% percentile bootstrap of statfn(row indices)."""
    uniq = np.unique(lr)
    groups = {u: np.where(lr == u)[0] for u in uniq}
    obs = statfn(np.arange(len(lr)))
    lo, hi = _boot(rng, groups, uniq, statfn)
    return float(obs), float(lo), float(hi)


def figure_bins(char: dict, sem: dict, seed: int = 0) -> pd.DataFrame:
    """Figure 9 series. char/sem: dicts of arrays ep, lr, atom, medoid, peeravg.
    Resampling order (one generator, per bin: ATOM share, medoid share, semantic gap) is fixed
    so that the published figure is reproduced exactly."""
    rng = np.random.default_rng(seed)
    out = []
    for lo_ep, hi_ep, label in STRATA:
        mc = (char["ep"] >= lo_ep) & (char["ep"] <= hi_ep)
        ms = (sem["ep"] >= lo_ep) & (sem["ep"] <= hi_ep)
        ac, mdc, lrc = char["atom"][mc], char["medoid"][mc], char["lr"][mc]
        a90 = cluster_ci(rng, lrc, lambda r: 100 * np.mean(ac[r] >= NEAR_MATCH))
        m90 = cluster_ci(rng, lrc, lambda r: 100 * np.mean(mdc[r] >= NEAR_MATCH))
        asem, psem, lrs = sem["atom"][ms], sem["peeravg"][ms], sem["lr"][ms]
        gap = cluster_ci(rng, lrs, lambda r: asem[r].mean() - psem[r].mean())
        out.append({
            "bin": label, "ep_min": lo_ep, "ep_max": hi_ep,
            "n_rows": int(mc.sum()), "n_learners": int(len(np.unique(lrc))),
            "sem_gap_atom_minus_peeravg": gap[0], "sem_gap_lo": gap[1], "sem_gap_hi": gap[2],
            "char_pct_ge90_atom": a90[0], "char_pct_ge90_atom_lo": a90[1], "char_pct_ge90_atom_hi": a90[2],
            "char_pct_ge90_medoid": m90[0], "char_pct_ge90_medoid_lo": m90[1], "char_pct_ge90_medoid_hi": m90[2],
        })
    return pd.DataFrame(out)


def pooled_summary(char: dict, sem: dict | None, seed: int = 1) -> dict:
    """Section 7.3 pooled contrasts, learner-clustered and paired within row."""
    rng = np.random.default_rng(seed)
    res = {"n_rows": int(len(char["atom"])), "n_learners": int(len(np.unique(char["lr"]))),
           "n_bootstrap": N_BOOT, "interval": "learner-clustered 95% percentile bootstrap"}
    for space, b in (("char", char), ("semantic", sem)):
        if b is None or np.isnan(b["peeravg"]).any():
            continue
        block = {k: float(np.mean(b[k])) for k in ("atom", "peeravg", "medoid")}
        for ref in ("peeravg", "medoid"):
            d = b["atom"] - b[ref]
            m, lo, hi = cluster_ci(rng, b["lr"], lambda r: d[r].mean())
            block[f"atom_minus_{ref}"] = {"mean": m, "lo": lo, "hi": hi}
        res[space] = block
    return res


def threshold_check(char: dict, thresholds=(0.80, 0.85, 0.90, 0.95)) -> dict:
    """Share >= threshold, ATOM vs medoid, per bin (point estimates): does the reversal move?"""
    out = {}
    for t in thresholds:
        rowsd = {}
        for lo_ep, hi_ep, label in STRATA:
            m = (char["ep"] >= lo_ep) & (char["ep"] <= hi_ep)
            rowsd[label] = {"atom": float(100 * np.mean(char["atom"][m] >= t)),
                            "medoid": float(100 * np.mean(char["medoid"][m] >= t))}
        out[f"{t:.2f}"] = rowsd
    return out


def write_aggregates(char: dict, sem: dict, out_root: pathlib.Path, extra: dict | None = None) -> None:
    bins = figure_bins(char, sem)
    fig_csv = out_root / "paper" / "figures" / "data" / "fig09_cw_ceiling_bins.csv"
    fig_csv.parent.mkdir(parents=True, exist_ok=True)
    bins.to_csv(fig_csv, index=False, float_format="%.6f")
    summ = pooled_summary(char, sem)
    summ["near_match_threshold_check"] = threshold_check(char)
    if extra:
        summ.update(extra)
    sj = out_root / "results" / "cw_classmate_ceiling" / "summary.json"
    sj.parent.mkdir(parents=True, exist_ok=True)
    sj.write_text(json.dumps(summ, indent=1))
    print(f"wrote {fig_csv}\nwrote {sj}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-root", type=pathlib.Path, default=REPO)
    ap.add_argument("--real-codec-vectors", type=pathlib.Path, default=None,
                    help="npz with code_state_id and vector: codec coordinates of each row's real "
                         "submission (frozen backbone, layer 15, problem-mean transform). Needed "
                         "for the semantic references; without it panel (a) is left empty.")
    args = ap.parse_args()
    df = build_table(args.real_codec_vectors)
    private = args.out_root / "results" / "cw_classmate_ceiling" / "rows_private.csv"
    private.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(private, index=False)          # per-row, local only
    print(f"wrote {private} (private, per-row)")
    char = {"ep": df["ep"].to_numpy(), "lr": df["lr"].to_numpy(), "atom": df["char_atom"].to_numpy(),
            "medoid": df["char_medoid"].to_numpy(), "peeravg": df["char_peeravg"].to_numpy()}
    sem = {"ep": df["ep"].to_numpy(), "lr": df["lr"].to_numpy(), "atom": df["sem_atom"].to_numpy(),
           "medoid": df["sem_medoid"].to_numpy(), "peeravg": df["sem_peeravg"].to_numpy()}
    write_aggregates(char, sem, args.out_root,
                     extra={"semantic_atom_mean": float(df["sem_atom"].mean()),
                            "semantic_references_available": bool(args.real_codec_vectors)})


if __name__ == "__main__":
    main()
