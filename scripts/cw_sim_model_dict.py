"""Shared helpers for loading the CodeWorkout generative simulator's checkpoint.

Used by `cw_sim_belief_handoff`, `cw_sim_train_handoff`, `atom.simulator.cw_substrate`,
`sim2real_w_prepare_events` and `sim2real_w_run`.  Builds the attempt encoder, filter and
outcome head from a bare-key `final.pt`, in the model-dict shapes used by the planner
helpers and by the `cw_fc_grid` evaluation.
"""
from __future__ import annotations
import pathlib
import sys
REPO = pathlib.Path(__file__).resolve().parents[1]
for p in (REPO / "src", REPO / "infra", REPO / "scripts", REPO):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
import torch  # noqa: E402
from atom.filter.cw_attempt_encoder import AttemptEncoder  # noqa: E402
from atom.simulator.cw_item_mlp import OutcomeHeadItemMLP, S1FilterV2ItemMLP  # noqa: E402
from atom.data.cw_outcome_label import N_OUTCOME  # noqa: E402


def _sha256(path: pathlib.Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def build_cwsim_model_dict(checkpoint: pathlib.Path, device: str) -> dict:
    """Load `final.pt` (bare-key schema) into the model-dict shape that
    `cw_planner_common._build_model` produces for the classifier (without
    `y_ever_head`), so downstream code accepts either checkpoint identically. Dims are read from the checkpoint's
    own `architecture` dict, never hardcoded, so a retrain with different dims
    cannot silently mismatch."""
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    d = ckpt["architecture"]
    attempt_encoder = AttemptEncoder(d["codec_dim"], d["codec_dim"], d["attempt_proj_dim"],
                                     d["attempt_hidden"], d["summary_dim"]).to(device)
    filt = S1FilterV2ItemMLP(codec_dim=d["codec_dim"], episode_hidden=d["episode_hidden"],
                             filter_hidden=d["filter_hidden"], latent_dim=d["latent_dim"],
                             prev_feat_dim=d["summary_dim"],
                             sigma_parameterization="precision_accumulating").to(device)
    outcome_head = OutcomeHeadItemMLP(codec_dim=d["codec_dim"], latent_dim=d["latent_dim"],
                                      hidden=d["attempt_hidden"], n_classes=N_OUTCOME,
                                      prefix_dim=d["attempt_hidden"]).to(device)
    attempt_encoder.load_state_dict(ckpt["attempt_encoder"])
    filt.load_state_dict(ckpt["filt"])
    outcome_head.load_state_dict(ckpt["outcome_head"])
    return {"attempt_encoder": attempt_encoder, "filt": filt,
            "filt_forward": lambda ic, pf, tm, vm: filt(ic, pf),
            "outcome_head": outcome_head,
            "checkpoint_sha256": _sha256(checkpoint), "checkpoint_meta": ckpt}


def build_cwsim_grid_dict(checkpoint: pathlib.Path, device: str) -> dict:
    """`cw_fc_grid_panel.py`-shape wrapper (mirrors `build_ratchet_model_dict` in
    `cw_classifier_model_dict`): `evaluate_iid`, `coupling_probe` and similar only
    read `model["attempt"/"filt"/"outcome"/"factored"]`."""
    m = build_cwsim_model_dict(checkpoint, device)
    return {"attempt": m["attempt_encoder"], "filt": m["filt"], "outcome": m["outcome_head"],
            "compile": None, "teacher": None, "factored": False,
            "checkpoint_sha256": m["checkpoint_sha256"], "checkpoint_meta": m["checkpoint_meta"]}
CWSIM_CODEC_DIM = 64  # ckpt["architecture"]["codec_dim"]; same codec rank as the CodeWorkout classifier
