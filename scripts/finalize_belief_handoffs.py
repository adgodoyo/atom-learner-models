#!/usr/bin/env python
"""Bring rebuilt planner belief handoffs into the exact form the thesis runs used, and check them.

The four handoffs (`artifacts/cw_*_prefix_beliefs.npz`) hold one belief per learner and episode,
so they are not published; REPRODUCE.md step 1f rebuilds them from the CodeWorkout data. The
planner scripts check each handoff against the SHA-256 of the file the thesis used. Two stored
strings differ for reasons unrelated to the beliefs: `checkpoint` is the path given on the command
line, and `checkpoint_sha256` is the hash of the rebuilt `.pt` file rather than of the training
checkpoint it is tensor-identical to (`checkpoints/PROVENANCE.md`). This script writes the
reference values of those two strings, re-saves each file and reports whether it now matches.

Inputs and outputs: the four files in `artifacts/`, rewritten in place.
Run: `python scripts/finalize_belief_handoffs.py` after step 1f.
"""
from __future__ import annotations

import hashlib
import pathlib
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]

# file -> (checkpoint string, training-checkpoint SHA-256, SHA-256 of the file the thesis used)
REFERENCE = {
    "artifacts/cw_ratchet_val_prefix_beliefs.npz": (
        "results/cw_sigma_precision_0305a22/aws/results/final.pt",
        "860a79d5c4b3080d614a3eeca60dfdde71f3c24230f2f88f2fcfb615a28b6b0e",
        "4375b0c93d7610fef94dc21d2db657a95aaeb5dc663d102f30fa5e7030a4ecde"),
    "artifacts/cw_sim_val_prefix_beliefs.npz": (
        "results/cw_sim_base_run/final.pt",
        "4e460240b34f53847d571b3ff88d7412416d607fb46e2b99ced9cdb6098e01f1",
        "4c6b5087cc4d2c8d6ec8029a45059ea246035067d5521ea96a1a6e63051bc4a4"),
    "artifacts/cw_sim_refreshed_head_val_prefix_beliefs.npz": (
        "results/cw_sim_head_refresh/final.pt",
        "e32b85e69722e56bf53f88ec0acb4ca214f0e259321323659de94368505f0022",
        "10ca4e9aea005b76346e1b18ae501a3d973417b562cb26fe1654647418125c03"),
    "artifacts/cw_sim_refreshed_head_train_prefix_beliefs.npz": (
        "results/cw_sim_head_refresh/final.pt",
        "e32b85e69722e56bf53f88ec0acb4ca214f0e259321323659de94368505f0022",
        "5a809c709109866750c43c12900ef3e719eb330b04f6853d7040536398393a38"),
}


def sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    bad = 0
    for rel, (ckpt, ckpt_sha, want) in REFERENCE.items():
        path = REPO / rel
        if not path.exists():
            print(f"missing {rel}: run REPRODUCE.md step 1f first")
            bad += 1
            continue
        with np.load(path, allow_pickle=False) as z:
            arrays = {k: z[k] for k in z.files}  # keeps the order the handoff script wrote
        arrays["checkpoint"] = np.array(ckpt)
        arrays["checkpoint_sha256"] = np.array(ckpt_sha)
        np.savez_compressed(path, **arrays)
        ok = sha256(path) == want
        print(f"{'ok   ' if ok else 'DIFF '} {rel}")
        if not ok:
            bad += 1
            print("      the beliefs differ from the thesis file (another platform or library build can"
                  " change the last bits); the planner scripts that pin this hash will refuse it")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
