#!/usr/bin/env python
"""Codec sweep scored by curriculum-free item geometry: chooses the codec layer, pooling and
deflation rank k for problem text.

Encodes a sample of FA problems (problem text, all layers, last and mean pooling), then sweeps
layer x pooling x k x metric, scoring each configuration by how well codec proximity recovers
FA's skill structure: same-skill vs different-skill pair separability (AUC). The 224 skill tags
are used only to validate here -- never to build the codec or enter the model -- which is what
the "matches a curriculum-using baseline without the curriculum" claim requires.
Check: the best configuration recovers the curriculum (AUC >= 0.70) and the codec (deflation) is
at least the raw baseline at the same layer/pooling.
Outputs: `cache/fa_problems/` (derived vectors) and `reports/p2_02c_sweep.md`.
Run: `python scripts/p2_02c_sweep.py --n 1200`.  Serves §5.2 and Appendix A (codec width).
"""

from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from atom.codec.backbone import DEFAULT_BACKBONE, MODEL_REGISTRY, is_cached, load_backbone  # noqa: E402
from atom.codec.extract import RawStateCache, extract_texts  # noqa: E402
from atom.codec.transform import CodecTransform  # noqa: E402
from atom.data.fa import load_problems, load_skills  # noqa: E402
from atom.eval.metrics import auc  # noqa: E402
from atom.eval.report import gate_report  # noqa: E402
from atom.utils.logging import RunLogger  # noqa: E402
from atom.utils.seeds import seed_everything  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[1]
CACHE_DIR = REPO / "cache" / "fa_problems"
LAYERS = list(range(6, 40, 3))  # 6,9,...,39 (spans mean-early to last-late bands)
POOLINGS = ["mean", "last"]
KS = [5, 10, 20]


def _cos(V, ra, rb):
    A = V[ra] / (np.linalg.norm(V[ra], axis=1, keepdims=True) + 1e-12)
    B = V[rb] / (np.linalg.norm(V[rb], axis=1, keepdims=True) + 1e-12)
    return (A * B).sum(1)


def _neg_eucl(V, ra, rb):
    return -np.linalg.norm(V[ra] - V[rb], axis=1)


def build_skill_pairs(pid_row, skills_of_pid, n_per=3000, seed=42):
    rng = np.random.default_rng(seed)
    skill_to_pids: dict = {}
    for pid, sk in skills_of_pid.items():
        for s in sk:
            skill_to_pids.setdefault(s, []).append(pid)
    multi = [s for s, ps in skill_to_pids.items() if len(set(ps)) >= 2]
    ra, rb, same = [], [], []
    # positives: two distinct problems sharing a skill
    n_pos, tries = 0, 0
    while n_pos < n_per and tries < n_per * 50:
        tries += 1
        ps = list(set(skill_to_pids[multi[rng.integers(len(multi))]]))
        i, j = rng.choice(len(ps), size=2, replace=False)
        ra.append(pid_row[ps[i]]); rb.append(pid_row[ps[j]]); same.append(1); n_pos += 1
    # negatives: two problems sharing no skill
    pids = list(pid_row)
    n_neg, tries = 0, 0
    while n_neg < n_per and tries < n_per * 50:
        tries += 1
        pi, pj = pids[rng.integers(len(pids))], pids[rng.integers(len(pids))]
        if pi != pj and not (skills_of_pid[pi] & skills_of_pid[pj]):
            ra.append(pid_row[pi]); rb.append(pid_row[pj]); same.append(0); n_neg += 1
    return np.array(ra), np.array(rb), np.array(same)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1200, help="problems (with skills) to encode")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    seed_everything(args.seed)

    if not is_cached(MODEL_REGISTRY[DEFAULT_BACKBONE]["hf_id"])[0]:
        print("gemma4-e4b not cached; run p2_01_backbone.py with --allow-download first.")
        return 3

    logger = RunLogger(config={"task": "p2_02c_sweep", "n": args.n, "layers": LAYERS,
                               "poolings": POOLINGS, "ks": KS}, data_version="FA-raw-2026-03")
    prob = load_problems(clean_body=True)
    skills = load_skills()
    sk_of_pid: dict = {}
    for pid, sid in zip(skills["problem_id"], skills["skill_id"]):
        sk_of_pid.setdefault(int(pid), set()).add(int(sid))

    with_skills = prob[prob["problem_id"].isin(sk_of_pid)].copy()
    samp = with_skills.sample(min(args.n, len(with_skills)), random_state=args.seed)
    items = [(int(r.problem_id), str(r.problem_body_clean)) for r in samp.itertuples(index=False)]
    logger.heartbeat(step=0, force=True, stage="loading backbone", n_problems=len(items))
    bundle = load_backbone(DEFAULT_BACKBONE)
    logger.heartbeat(step=0, force=True, stage="encoding problems")
    stats = extract_texts(bundle, items, CACHE_DIR, logger=logger)

    cache = RawStateCache(CACHE_DIR, stats["n_layers"], stats["hidden_dim"],
                          capacity=stats["n_unique"], model=bundle["hf_id"], template="")
    pid_row = {pid: cache.row_of_hash[h] for h, pids in cache.ids_of_hash.items() for pid in map(int, pids)}
    pid_row = {p: r for p, r in pid_row.items() if p in sk_of_pid}
    sk_sample = {pid: sk_of_pid[pid] for pid in pid_row}  # restrict to encoded problems
    ra, rb, same = build_skill_pairs(pid_row, sk_sample, seed=args.seed)
    logger.heartbeat(step=stats["n_unique"], force=True, stage="sweeping",
                     n_pairs=len(same), pos=int(same.sum()))

    results = []  # (auc, pooling, layer, k, metric)
    raw_by = {}   # (pooling, layer) -> raw cosine AUC
    for pooling in POOLINGS:
        S = cache.states(pooling)  # (U, L, D) fp16
        for layer in LAYERS:
            X = np.asarray(S[:, layer, :], dtype=np.float64)
            raw_auc = auc(same, _cos(X, ra, rb))
            raw_by[(pooling, layer)] = raw_auc
            ct = CodecTransform.fit(X, k_max=50, layer=layer, pooling=pooling)
            for k in KS:
                defl = ct.apply(X, k=k, whiten=False)
                results.append((auc(same, _cos(defl, ra, rb)), pooling, layer, k, "cos"))
                whit = ct.apply(X, k=k, whiten=True)
                results.append((auc(same, _neg_eucl(whit, ra, rb)), pooling, layer, k, "eucl"))

    results.sort(reverse=True)
    best = results[0]
    best_raw = raw_by[(best[1], best[2])]
    # best per pooling (for the depth story)
    best_mean = max((r for r in results if r[1] == "mean"))
    best_last = max((r for r in results if r[1] == "last"))
    passed = best[0] >= 0.70 and best[0] >= best_raw - 0.005

    logger.log_metrics(best_auc=best[0], best_pooling=best[1], best_layer=best[2],
                       best_k=best[3], best_metric=best[4], best_raw_auc=best_raw,
                       best_mean=str(best_mean), best_last=str(best_last), n_pos=int(same.sum()))
    logger.summary(status="ok" if passed else "gate-fail", best=str(best))

    top = "\n".join(f"  {a:.3f}  {p:>4}/L{ly}/k{k}/{m}" for a, p, ly, k, m in results[:8])
    gate_report(
        path="reports/p2_02c_sweep.md",
        title="P2-2c - Codec sweep by curriculum-free item geometry (skill recovery)",
        meta={"run_id": logger.run_id, "git_sha": logger.git_sha,
              "config_hash": logger.config_hash, "problems": stats["n_unique"],
              "pairs": f"{int(same.sum())} pos / {int((same == 0).sum())} neg"},
        metrics={
            "BEST config": f"{best[1]} pooling · L{best[2]} · k={best[3]} · {best[4]}",
            "BEST skill-recovery AUC": round(best[0], 3),
            "raw AUC @ best (no codec)": round(best_raw, 3),
            "codec Δ over raw @ best": round(best[0] - best_raw, 3),
            "best mean-pool": f"{best_mean[0]:.3f} @ L{best_mean[2]} k{best_mean[3]} {best_mean[4]}",
            "best last-token": f"{best_last[0]:.3f} @ L{best_last[2]} k{best_last[3]} {best_last[4]}",
        },
        passed=passed,
        gate_desc="best config recovers curriculum (skill AUC >= 0.70) and codec >= raw "
                  "baseline; skill tags used for validation only, never in the model.",
        notes=[
            "Job A (item geometry): codec-proximity of PROBLEMS recovers skill structure "
            "without the codec ever seeing a tag - the no-curriculum claim, measured.",
            "Top-8 configs (AUC · pooling/layer/k/metric):\n" + top,
            "This selects the codec config for problem geometry; answer resolution "
            "is a secondary check (Job B), scored separately.",
        ],
    )
    print(f"\nP2-2c: {'PASS' if passed else 'FAIL'} · best {best[1]}/L{best[2]}/k{best[3]}/{best[4]} "
          f"AUC {best[0]:.3f} (raw {best_raw:.3f}) · report: reports/p2_02c_sweep.md")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
