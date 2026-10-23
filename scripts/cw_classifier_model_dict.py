"""Shared helpers for `cw_classifier_belief_handoff`.

Builds the CodeWorkout classifier's model dictionary (attempt encoder, filter and
outcome head) from a checkpoint, in the form `infra/cw_fc_grid.py`'s evaluation reads.
"""
from __future__ import annotations
import pathlib
import sys
REPO = pathlib.Path(__file__).resolve().parents[1]
for p in (REPO / "src", REPO / "infra", REPO / "scripts", REPO):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from cw_planner_common import _build_model
RATCHET_CODEC_DIM = 64


def build_ratchet_model_dict(checkpoint: pathlib.Path, device: str) -> dict:
    m = _build_model(device, checkpoint)
    # cw_fc_grid.evaluate_iid only reads model["attempt"/"filt"/"outcome"/
    # "factored"]; "compile"/"teacher" stay None, matching a flat non-factored
    # arm, which is what the classifier's single 4-class OutcomeHeadItemMLP is.
    return {"attempt": m["attempt_encoder"], "filt": m["filt"], "outcome": m["outcome_head"],
            "compile": None, "teacher": None, "factored": False,
            "checkpoint_sha256": m["checkpoint_sha256"]}
