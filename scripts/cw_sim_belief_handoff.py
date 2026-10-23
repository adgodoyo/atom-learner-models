#!/usr/bin/env python3
"""Export the CodeWorkout generative simulator's dense validation prefix beliefs for the planner.

Writes the generative simulator's (`final.pt` of its base training run) validation prefix
beliefs in the same value-join format as `cw_classifier_belief_handoff.py` (`learner`,
`episode_id`, `mu`, `log_sigma`, `release`), so the planner can consume either belief
identically. One row per (validation learner, distinct episode position); the manifest
and belief path are shared and only the model loader differs (`cw_sim_model_dict`, which
reads the checkpoint's bare-key schema). CPU forward pass only.
Inputs: `--checkpoint`, raw CodeWorkout data and codec caches.  Outputs: `--out` (.npz) and
`<stem>_manifest.json`.  Run: `python scripts/cw_sim_belief_handoff.py --code-map ... --out beliefs.npz`.
Serves Table 11 and the Chapter 8 planner runs on the generative simulator's belief.
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
from cw_classifier_belief_handoff import compute_dense_val_beliefs  # noqa: E402
from cw_sim_model_dict import CWSIM_CODEC_DIM, build_cwsim_grid_dict  # noqa: E402
from cw_planner_common import _build_manifest  # noqa: E402


class _Args:
    pass


def _sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="results/cw_sim_base_run/final.pt")
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
    model = build_cwsim_grid_dict(pathlib.Path(ns.checkpoint), device)
    for v in model.values():
        if hasattr(v, "eval"):
            v.eval()

    # compute_dense_val_beliefs (cw_classifier_belief_handoff.py) hardcodes
    # RATCHET_CODEC_DIM=64 for build_attempt_batch's code/diff dim, identical to
    # CWSIM_CODEC_DIM=64 here (both checkpoints share the codec rank), so it is reused
    # unmodified.
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
