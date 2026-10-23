#!/usr/bin/env python3
"""Export the CodeWorkout classifier's dense validation prefix beliefs for the planner.

Computes the belief (`mu`, `log_sigma`) at every distinct episode position for every
validation learner, reusing the manifest, model and belief path of the classifier's
evaluation (`infra/cw_fc_grid.py`, `scripts/cw_fc_grid_ratchet_eval.py`). The evaluation
itself exports only the rows tied to a scored first-attempt target; this script exports
one row per (validation learner, episode) so the planner can query any point of a
trajectory. Rows carry `learner` and `episode_id` and are joined to the episode table by
value, never by position. CPU forward pass only, no training.
Inputs: the classifier checkpoint (`--checkpoint`), raw CodeWorkout data and codec caches.
Outputs: `--out` (.npz) and `<stem>_manifest.json` with row counts and SHA-256 hashes.
Run: `python scripts/cw_classifier_belief_handoff.py --code-map ... --out beliefs.npz`.
Serves Table 9 / Figure 6 and the Chapter 8 planner runs on the CodeWorkout classifier.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
for p in (REPO / "src", REPO / "infra", REPO / "scripts", REPO):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from atom.filter.cw_filter_batch import build_attempt_batch, build_episode_batch  # noqa: E402
from atom.simulator.cw_4_train_loop import _distinct_episode_positions  # noqa: E402
from atom.simulator.fa_train_step import gather_belief_for_batch  # noqa: E402
from cw_classifier_model_dict import RATCHET_CODEC_DIM, build_ratchet_model_dict  # noqa: E402
from cw_planner_common import _build_manifest  # noqa: E402


class _Args:
    pass


def _sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def compute_dense_val_beliefs(model: dict, manifest: dict, device: str, batch_size: int = 8) -> dict:
    """Return row-aligned arrays over every (validation learner, episode) pair:
    `learner`, `episode_id`, `mu`, `log_sigma` and `release` (S19/F19, taken from the
    attempts table, for the composition breakdown)."""
    val_learners = sorted(manifest["val_learners"])
    rows_learner, rows_episode, rows_mu, rows_ls, rows_release = [], [], [], [], []
    release_of_learner_episode = {}
    for r in manifest["attempts"].itertuples():
        release_of_learner_episode.setdefault((str(r.learner), str(r.episode_id)), r.release)

    with torch.no_grad():
        for start in range(0, len(val_learners), batch_size):
            chosen = set(val_learners[start:start + batch_size])
            at = manifest["attempts"][manifest["attempts"]["learner"].isin(chosen)]
            ev = manifest["episodes"][manifest["episodes"]["learner"].isin(chosen)]
            if ev.empty:
                continue
            ab = build_attempt_batch(at, manifest["code_of_hash"], manifest["diff_of_hash"],
                                     RATCHET_CODEC_DIM, RATCHET_CODEC_DIM, device)
            prefix_states, summary = model["attempt"](ab["code_vec"], ab["diff_vec"],
                                                       ab["scalar_feats"], ab["lengths"])
            eb = build_episode_batch(ev, manifest["problem_codec"], manifest["pid_to_idx"],
                                     summary, ab["episode_ids"], device)
            li, ti = _distinct_episode_positions(ev, eb, ab["episode_ids"], device)
            mu_all, ls_all = gather_belief_for_batch(
                lambda ic, pf, tm=None, vm=None: model["filt"](ic, pf), eb, li, ti)
            mu_np, ls_np = mu_all.cpu().numpy(), ls_all.cpu().numpy()
            # `ab["episode_ids"]` carries the episode ids row-aligned with mu_all/ls_all;
            # the learner for each is recovered by value from the episodes table, never
            # assumed from batch order.
            episode_to_learner = dict(zip(ev["episode_id"].astype(str), ev["learner"].astype(str)))
            for i, episode_id in enumerate(ab["episode_ids"]):
                episode_id = str(episode_id)
                learner = episode_to_learner.get(episode_id)
                if learner is None:
                    continue
                rows_learner.append(learner)
                rows_episode.append(episode_id)
                rows_mu.append(mu_np[i])
                rows_ls.append(ls_np[i])
                rows_release.append(release_of_learner_episode.get((learner, episode_id), "UNKNOWN"))

    return {"learner": np.asarray(rows_learner), "episode_id": np.asarray(rows_episode),
            "mu": np.asarray(rows_mu, dtype=np.float32), "log_sigma": np.asarray(rows_ls, dtype=np.float32),
            "release": np.asarray(rows_release)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="results/cw_sigma_precision_0305a22/aws/results/final.pt")
    for name in ("raw_dir", "code_map", "diff_map", "codealone_meta", "codealone_states",
                "diff_meta", "diff_states"):
        ap.add_argument("--" + name.replace("_", "-"), required=name != "raw_dir")
    ap.add_argument("--out", required=True)
    ns = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    build_args = _Args()
    build_args.derived_val_predictions = None
    build_args.raw_dir = ns.raw_dir
    build_args.code_map, build_args.diff_map = ns.code_map, ns.diff_map
    build_args.codealone_meta, build_args.codealone_states = ns.codealone_meta, ns.codealone_states
    build_args.diff_meta, build_args.diff_states = ns.diff_meta, ns.diff_states
    build_args.learner_batch_size, build_args.epochs, build_args.lr = 8, 1, 1e-3

    manifest = _build_manifest(build_args)
    model = build_ratchet_model_dict(pathlib.Path(ns.checkpoint), device)
    for v in model.values():
        if hasattr(v, "eval"):
            v.eval()

    beliefs = compute_dense_val_beliefs(model, manifest, device)
    n = len(beliefs["learner"])
    n_learners = len(set(beliefs["learner"].tolist()))
    release_counts = {r: int((beliefs["release"] == r).sum()) for r in sorted(set(beliefs["release"].tolist()))}

    out_path = pathlib.Path(ns.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, learner=beliefs["learner"], episode_id=beliefs["episode_id"],
                        mu=beliefs["mu"], log_sigma=beliefs["log_sigma"], release=beliefs["release"],
                        checkpoint=str(ns.checkpoint), checkpoint_sha256=model["checkpoint_sha256"])
    result = {"n_rows": n, "n_val_learners": n_learners, "release_composition": release_counts,
             "checkpoint": str(ns.checkpoint), "checkpoint_sha256": model["checkpoint_sha256"],
             "file_sha256": _sha256(out_path), "mu_dim": beliefs["mu"].shape[1]}
    print(json.dumps(result, indent=2))
    (out_path.parent / (out_path.stem + "_manifest.json")).write_text(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
