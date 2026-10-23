#!/usr/bin/env python3
"""Prepares the numeric event handoff for the policy trained only inside the simulator (§8.3)
without moving licensed raw text.

The training worker needs chronological learner histories, codecs, attempt summaries and native
outcome labels, but not the source-code tables once those quantities are computed. This script
runs the frozen-filter forward locally and writes a compact NPZ/JSON pair; no statement text or
student code is stored.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import torch

REPO = pathlib.Path(__file__).resolve().parents[1]
for p in (REPO / "src", REPO / "scripts", REPO / "infra"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from cw_planner_suite import _build_manifest, _episode_summaries, _event_rows  # noqa: E402
from cw_sim_model_dict import build_cwsim_model_dict  # noqa: E402
from imagination_cw import sha256_file  # noqa: E402


def _args(ns: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        derived_val_predictions=None,
        raw_dir=ns.raw_dir,
        code_map=ns.code_map,
        diff_map=ns.diff_map,
        codealone_meta=ns.codealone_meta,
        codealone_states=ns.codealone_states,
        diff_meta=ns.diff_meta,
        diff_states=ns.diff_states,
        learner_batch_size=8,
        epochs=1,
        lr=1e-3,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", required=True)
    ap.add_argument("--checkpoint", required=True)
    for name in ("code-map", "diff-map", "codealone-meta", "codealone-states", "diff-meta", "diff-states"):
        ap.add_argument("--" + name, required=True)
    ap.add_argument("--out", default="artifacts/sim2real_w_events")
    ap.add_argument("--device", default="auto")
    ns = ap.parse_args()
    device = "cuda" if ns.device == "auto" and torch.cuda.is_available() else ("cpu" if ns.device == "auto" else ns.device)
    out = (REPO / ns.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    manifest = _build_manifest(_args(ns))
    model = build_cwsim_model_dict(pathlib.Path(ns.checkpoint).resolve(), device)
    for value in model.values():
        if hasattr(value, "eval"):
            value.eval()
    summaries = _episode_summaries(model, manifest, device)
    events = _event_rows(manifest, summaries, device)
    learners = sorted(events)
    episode_ids = sorted({str(e["episode_id"]) for rows in events.values() for e in rows})
    learner_idx = {x: i for i, x in enumerate(learners)}
    episode_idx = {x: i for i, x in enumerate(episode_ids)}
    releases = sorted({str(e.get("release", "pooled")) for rows in events.values() for e in rows})
    release_idx = {x: i for i, x in enumerate(releases)}
    rows = [{**e, "learner": learner} for learner in learners for e in sorted(events[learner], key=lambda x: x["order"])]
    if not rows:
        raise RuntimeError("event handoff is empty")
    np.savez_compressed(
        out.with_suffix(".npz"),
        learner_idx=np.asarray([learner_idx[str(e["learner"])] for e in rows], dtype=np.int32),
        episode_idx=np.asarray([episode_idx[str(e["episode_id"])] for e in rows], dtype=np.int32),
        problem_id=np.asarray([int(e["problem_id"]) for e in rows], dtype=np.int32),
        order=np.asarray([int(e["order"]) for e in rows], dtype=np.int32),
        label=np.asarray([int(e["label"]) for e in rows], dtype=np.int8),
        release_idx=np.asarray([release_idx[str(e.get("release", "pooled"))] for e in rows], dtype=np.int8),
        codec=np.stack([e["codec"].detach().cpu().numpy().astype(np.float32) for e in rows]),
        summary=np.stack([e["summary"].detach().cpu().numpy().astype(np.float32) for e in rows]),
    )
    meta = {
        "format": "sim2real-w-events-v1",
        "n_rows": len(rows),
        "n_learners": len(learners),
        "learners": learners,
        "episode_ids": episode_ids,
        "releases": releases,
        "checkpoint_sha256": sha256_file(pathlib.Path(ns.checkpoint).resolve()),
        "raw_text_stored": False,
        "licensed_source_transferred": False,
    }
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
