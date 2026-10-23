"""Named CW substrate bundles for the planner experiments.

The planner consumes this module's single ``load_substrate(name)`` interface.  Each
name maps to a trained CodeWorkout model checkpoint and its dense validation-prefix
belief file, with SHA-256 digests recorded for both: ``ratchet-original`` is the
CodeWorkout classifier, ``ratchet-cwsim/cold-head`` the CodeWorkout generative simulator
with its original outcome head, and ``ratchet-cwsim/refreshed-head`` the same simulator
with its refreshed outcome head.  The adapter keeps these names stable while the files
stay local (a cloud launcher may replace the paths with its staged copies).
"""
from __future__ import annotations

import pathlib
import sys
from typing import Any

REPO = pathlib.Path(__file__).resolve().parents[3]
S1_CKPT = REPO / "results/cw_sigma_precision_0305a22/aws/results/final.pt"
S1_BELIEFS = REPO / "artifacts/cw_ratchet_val_prefix_beliefs.npz"
S2_CKPT = REPO / "results/cw_sim_base_run/final.pt"
S2_BELIEFS = REPO / "artifacts/cw_sim_val_prefix_beliefs.npz"
S3_CKPT = REPO / "results/cw_sim_head_refresh/final.pt"
S3_BELIEFS = REPO / "artifacts/cw_sim_refreshed_head_val_prefix_beliefs.npz"

_BUNDLE_META = {
    "ratchet-original": {
        "checkpoint": S1_CKPT,
        "beliefs": S1_BELIEFS,
        "source": "CodeWorkout classifier belief handoff",
        "checkpoint_sha256": "8d22b6914e041dfa88bf911b1a498fa0c87af30488b07f6e49a72adca731c544",
        "beliefs_sha256": "4375b0c93d7610fef94dc21d2db657a95aaeb5dc663d102f30fa5e7030a4ecde",
    },
    "ratchet-cwsim/cold-head": {
        "checkpoint": S2_CKPT,
        "beliefs": S2_BELIEFS,
        "source": "CodeWorkout generative simulator belief handoff (own outcome head)",
        "checkpoint_sha256": "3143d05122957e855b2067f9bd9c61cd7aabf414b380ffc5fc2e46a0ccf7c3e9",
        "beliefs_sha256": "4c6b5087cc4d2c8d6ec8029a45059ea246035067d5521ea96a1a6e63051bc4a4",
    },
    "ratchet-cwsim/refreshed-head": {
        "checkpoint": S3_CKPT,
        "beliefs": S3_BELIEFS,
        "source": "CodeWorkout generative simulator belief handoff (refreshed outcome head)",
        "checkpoint_sha256": "5337abbbad40e0d1474ddce1e3bdc69cd1159a94e29722cedef70d8c56d7a56d",
        "beliefs_sha256": "10ca4e9aea005b76346e1b18ae501a3d973417b562cb26fe1654647418125c03",
    },
}


def _scripts_on_path() -> None:
    scripts = str(REPO / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)


def load_substrate(name: str, device: str = "cpu") -> dict[str, Any]:
    """Return a replay model and dense value-join beliefs for one named substrate.

    No raw checkpoint path is accepted by callers.  A missing artifact fails loudly so
    a substrate cannot be silently substituted or partially evaluated.
    """
    if name not in _BUNDLE_META:
        raise KeyError(f"unknown CW substrate {name!r}; choose {sorted(_BUNDLE_META)}")
    spec = _BUNDLE_META[name]
    checkpoint = pathlib.Path(spec["checkpoint"])
    beliefs = pathlib.Path(spec["beliefs"])
    if not checkpoint.exists():
        raise FileNotFoundError(f"substrate {name} checkpoint missing: {checkpoint}")
    if not beliefs.exists():
        raise FileNotFoundError(f"substrate {name} belief handoff missing: {beliefs}")
    _scripts_on_path()
    if name == "ratchet-original":
        from cw_planner_common import _build_model
        model = _build_model(device, checkpoint)
        # The planner's audited primitives read architecture through checkpoint_meta.
        import torch
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model["checkpoint_meta"] = ckpt
    else:
        from cw_sim_model_dict import build_cwsim_model_dict
        model = build_cwsim_model_dict(checkpoint, device)
    return {
        "model": model,
        "beliefs_path": beliefs,
        "checkpoint_sha256": model.get("checkpoint_sha256", spec["checkpoint_sha256"]),
        "beliefs_sha256": spec["beliefs_sha256"],
        "metadata": {
            "name": name,
            "source": spec["source"],
            "checkpoint": str(checkpoint),
            "beliefs": str(beliefs),
            "checkpoint_sha256_recorded": spec["checkpoint_sha256"],
        },
    }


__all__ = ["load_substrate"]
