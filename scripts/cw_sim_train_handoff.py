#!/usr/bin/env python3
"""Write train-learner belief states from the CodeWorkout generative simulator.

Loads the CodeWorkout generative simulator with its refreshed outcome head
(`ratchet-cwsim/refreshed-head`), checks the checkpoint's SHA-256 before any forward
pass, builds the pooled S19/F19 manifest, restricts every attempt and episode to the
frozen learner split's `train` set (by learner value), and computes one belief
(mu, log_sigma) per (learner, episode_id). It is a separate entry point from the
validation handoff (`cw_sim_belief_handoff.py`) and refuses validation-prefix belief
files as inputs. It fails if split membership is not exact, if a validation or lockbox
learner appears in the output, or if the checkpoint hash differs from the expected one.

Inputs: the checkpoint, `splits/cw_learner_split.csv`, and the code-vector caches.
Outputs: `--out` (numeric NPZ: learner, episode_id, mu, log_sigma, release, split) and
`<out>_manifest.json`. Raw code never leaves the local workspace.
Run: `python scripts/cw_sim_train_handoff.py --out PATH --code-map ... --diff-states ...`.
Serves Table 11 and §8.3 (training-learner beliefs used by `sim2real_w_run.py`).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
from typing import Any

REPO = pathlib.Path(__file__).resolve().parents[1]
for p in (REPO / "src", REPO / "infra", REPO / "scripts", REPO):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

REGISTERED_S3_CHECKPOINT_SHA256 = (
    "5337abbbad40e0d1474ddce1e3bdc69cd1159a94e29722cedef70d8c56d7a56d"
)
VAL_HANDOFF_NAMES = {
    "cw_ratchet_val_prefix_beliefs.npz",
    "cw_sim_val_prefix_beliefs.npz",
    "cw_sim_refreshed_head_val_prefix_beliefs.npz",
}


class _Args:
    pass


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _build_args(ns: argparse.Namespace) -> _Args:
    args = _Args()
    args.derived_val_predictions = None
    args.raw_dir = ns.raw_dir
    args.code_map, args.diff_map = ns.code_map, ns.diff_map
    args.codealone_meta, args.codealone_states = ns.codealone_meta, ns.codealone_states
    args.diff_meta, args.diff_states = ns.diff_meta, ns.diff_states
    args.learner_batch_size, args.epochs, args.lr = ns.learner_batch_size, 1, 1e-3
    return args


def _assert_no_val_handoff_inputs(ns: argparse.Namespace) -> None:
    """Keep this training path structurally unable to consume val handoffs."""
    for value in vars(ns).values():
        if not isinstance(value, str):
            continue
        if pathlib.Path(value).name in VAL_HANDOFF_NAMES:
            raise ValueError("train handoff refuses val-prefix belief artifacts")


def train_manifest(manifest: dict[str, Any], split_path: pathlib.Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return train-only manifest plus by-value membership/composition metadata."""
    split = pd.read_csv(split_path, dtype={"learner": str, "split": str})
    required = {"learner", "split"}
    if not required.issubset(split.columns):
        raise ValueError(f"split file missing columns: {sorted(required - set(split.columns))}")
    train_expected = set(split.loc[split["split"] == "train", "learner"].astype(str))
    val_expected = set(split.loc[split["split"] == "val", "learner"].astype(str))
    lockbox_expected = set(split.loc[split["split"] == "lockbox", "learner"].astype(str))
    if train_expected & (val_expected | lockbox_expected):
        raise ValueError("learner split is not disjoint by value")
    manifest_train = set(map(str, manifest.get("train_learners", set())))
    if manifest_train != train_expected:
        raise ValueError(
            "manifest train learner set differs from frozen split by value: "
            f"missing={sorted(train_expected - manifest_train)[:3]}, "
            f"extra={sorted(manifest_train - train_expected)[:3]}"
        )
    attempts = manifest["attempts"].copy()
    episodes = manifest["episodes"].copy()
    # The pooled manifest may contain val/lockbox rows for codec construction, but
    # they are filtered out before any episode summary or belief forward below.
    attempts = attempts[attempts["learner"].astype(str).isin(train_expected)].reset_index(drop=True)
    episodes = episodes[episodes["learner"].astype(str).isin(train_expected)].reset_index(drop=True)
    observed = set(attempts["learner"].astype(str))
    if observed != train_expected:
        raise ValueError(
            "train histories do not cover frozen train split by value: "
            f"missing={sorted(train_expected - observed)[:8]}, "
            f"extra={sorted(observed - train_expected)[:8]}"
        )
    if set(episodes["learner"].astype(str)) - train_expected:
        raise ValueError("train handoff episode table contains val/lockbox learner")
    out = dict(manifest)
    out["attempts"], out["episodes"] = attempts, episodes
    out["train_learners"] = train_expected
    out["val_learners"] = set()
    metadata = {
        "split_path": str(split_path),
        "n_train_learners_expected": len(train_expected),
        "n_train_learners_observed": len(observed),
        "n_val_learners_excluded_by_value": len(val_expected),
        "n_lockbox_learners_excluded_by_value": len(lockbox_expected),
        "n_attempts_train": int(len(attempts)),
        "n_episodes_train": int(len(episodes)),
        "release_composition": {
            str(k): int(v)
            for k, v in attempts["release"].value_counts(dropna=False).to_dict().items()
        },
        "train_set_equals_split_by_value": observed == train_expected,
        "val_handoff_inputs_read": False,
    }
    return out, metadata


def produce_train_handoff(ns: argparse.Namespace) -> dict[str, Any]:
    _assert_no_val_handoff_inputs(ns)
    checkpoint = pathlib.Path(ns.checkpoint).resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    checkpoint_sha = sha256_file(checkpoint)
    if checkpoint_sha != ns.expected_checkpoint_sha256:
        raise ValueError(
            f"checkpoint sha256 mismatch before forward: expected {ns.expected_checkpoint_sha256}, "
            f"got {checkpoint_sha}"
        )

    # Keep heavy CW imports lazy so split-membership tests work without optional
    # HTML/data dependencies.  Import only after the frozen checkpoint has passed
    # its registered byte-level identity check; no model/data forward can precede it.
    from cw_classifier_belief_handoff import compute_dense_val_beliefs
    from cw_sim_model_dict import build_cwsim_grid_dict
    from cw_planner_common import _build_manifest

    manifest_all = _build_manifest(_build_args(ns))
    manifest, split_meta = train_manifest(manifest_all, pathlib.Path(ns.split_path))
    device = ns.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = build_cwsim_grid_dict(checkpoint, device)
    # Verify the loader's own recorded digest too; this catches a loader/path mix-up.
    if model.get("checkpoint_sha256") != checkpoint_sha:
        raise ValueError("model loader checkpoint digest disagrees with pre-forward digest")
    for value in model.values():
        if hasattr(value, "eval"):
            value.eval()

    # The val handoff function is reused only as a row-aligned primitive.  Its input
    # manifest is train-only and its output is checked below; no val NPZ is read.
    dense = compute_dense_val_beliefs(model, {**manifest, "val_learners": manifest["train_learners"]},
                                      device, batch_size=ns.learner_batch_size)
    out_path = pathlib.Path(ns.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    observed = set(map(str, dense["learner"].tolist()))
    expected = set(map(str, manifest["train_learners"]))
    if observed != expected:
        raise ValueError("output learner set is not exactly the train split by value")
    forbidden = set(map(str, dense["learner"].tolist())) & (
        set(pd.read_csv(ns.split_path, dtype={"learner": str}).query("split != 'train'")["learner"])
    )
    if forbidden:
        raise ValueError(f"val/lockbox learners leaked into output: {sorted(forbidden)[:5]}")
    np.savez_compressed(
        out_path,
        learner=dense["learner"],
        episode_id=dense["episode_id"],
        mu=dense["mu"],
        log_sigma=dense["log_sigma"],
        release=dense["release"],
        split=np.asarray(["train"] * len(dense["learner"])),
        checkpoint=str(checkpoint),
        checkpoint_sha256=checkpoint_sha,
    )
    result = {
        "handoff": "cw_sim_refreshed_head_train_prefix_beliefs",
        "format": "canary-#7-value-join",
        "s3_key": ns.s3_key,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha,
        "file_sha256": sha256_file(out_path),
        "mu_dim": int(dense["mu"].shape[1]),
        "n_rows": int(len(dense["learner"])),
        "n_train_learners": int(len(observed)),
        **split_meta,
    }
    manifest_path = out_path.parent / f"{out_path.stem}_manifest.json"
    manifest_path.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="results/cw_sim_head_refresh/final.pt")
    ap.add_argument("--expected-checkpoint-sha256", default=REGISTERED_S3_CHECKPOINT_SHA256)
    ap.add_argument("--split-path", default="splits/cw_learner_split.csv")
    ap.add_argument("--raw-dir", default=None)
    ap.add_argument("--code-map", required=True)
    ap.add_argument("--diff-map", required=True)
    ap.add_argument("--codealone-meta", required=True)
    ap.add_argument("--codealone-states", required=True)
    ap.add_argument("--diff-meta", required=True)
    ap.add_argument("--diff-states", required=True)
    ap.add_argument("--learner-batch-size", type=int, default=8)
    ap.add_argument("--device", default=None)
    ap.add_argument(
        "--s3-key",
        default="sim2real-licensed-20260902/"
        "cw_sim_refreshed_head_train_prefix_beliefs.npz",
    )
    ap.add_argument("--out", required=True)
    return 0 if produce_train_handoff(ap.parse_args()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
