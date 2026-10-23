"""Shared helpers for the generation evaluations of the CodeWorkout generative simulator.

Provides model loading and the held-out-item class posterior used by
`cw_sim_launch2.py` and `cw_sim_launch2_tiktoc.py` (Table 11). Every generation panel
uses the simulator with its refreshed outcome head (`ratchet-cwsim/refreshed-head`);
the base run's outcome head is not used for generation. The model is assembled from the
base run's trained backbone LoRA bases, hypernetwork, attempt encoder and filter (these
modules are byte-identical between the base and refreshed-head checkpoints, so either
file would do; the base run's `final.pt` is used because it also carries
`test_vector_head`, which the head refresh does not train), with `outcome_head` taken
from the refreshed-head checkpoint written by `scripts/cw_sim_head_refresh.py`. The two
checkpoints' architectures are checked for agreement before the model is built.
"""
from __future__ import annotations

import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
for _p in (REPO / "src", REPO / "infra", REPO / "scripts", REPO):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import torch  # noqa: E402

import cw_sim_core as cw5  # noqa: E402
from atom.codec.backbone import DEFAULT_BACKBONE  # noqa: E402
from atom.data.cw_outcome_label import N_OUTCOME  # noqa: E402
from atom.simulator.cw_item_mlp import OutcomeHeadItemMLP  # noqa: E402

HEAD_SUBSTRATE_NAME = "ratchet-cwsim/refreshed-head"
SIGMA_PARAMETERIZATION = "precision_accumulating"  # matches cw_sim_full_run.py's own constant


def load_generation_model(base_checkpoint: pathlib.Path, head_checkpoint: pathlib.Path,
                          device: str, allow_download: bool = True,
                          backbone: str = DEFAULT_BACKBONE) -> tuple[dict, dict]:
    """Returns `(m, provenance)`. `m` is a `cw5.build_model`-shaped dict with EVERY
    trained module loaded (`attempt_encoder`/`filt`/`outcome_head`/`hyper`/`bases`/
    `test_vector_head`), inference-ready (`.eval()` on everything). `outcome_head` comes
    from `head_checkpoint`; every other trained module comes from `base_checkpoint`.
    `provenance` records both checkpoints' paths and sha256 digests, so that every
    generation table can name the exact head and checkpoint it was produced with."""
    base_ckpt = torch.load(base_checkpoint, map_location="cpu", weights_only=False)
    head_ckpt = torch.load(head_checkpoint, map_location="cpu", weights_only=False)

    for key in ("latent_dim", "codec_dim", "episode_hidden", "filter_hidden", "attempt_hidden"):
        assert base_ckpt["architecture"][key] == head_ckpt["architecture"][key], (
            f"architecture mismatch between base and head checkpoints on {key!r}: "
            f"{base_ckpt['architecture'][key]!r} != {head_ckpt['architecture'][key]!r} -- "
            "refusing to build a model from mismatched substrates"
        )

    backbone_bundle = cw5.load_backbone(backbone, device=device, allow_download=allow_download)
    m = cw5.build_model(cw5.CW5_DIMS, backbone_bundle, seed=base_ckpt.get("seed", 0), device=device,
                        item_encoder="mlp", sigma_parameterization=SIGMA_PARAMETERIZATION)

    m["attempt_encoder"].load_state_dict(base_ckpt["attempt_encoder"])
    m["filt"].load_state_dict(base_ckpt["filt"])
    m["hyper"].load_state_dict(base_ckpt["hyper"])
    for name, basis in m["bases"].items():
        basis.load_state_dict(base_ckpt["basis"][name])

    m["test_vector_head"] = OutcomeHeadItemMLP(
        codec_dim=cw5.CW5_DIMS["codec_dim"], latent_dim=cw5.CW5_DIMS["latent_dim"],
        hidden=cw5.CW5_DIMS["attempt_hidden"], n_classes=base_ckpt.get("max_test_cases", 32),
        prefix_dim=cw5.CW5_DIMS["attempt_hidden"]).to(device)
    if "test_vector_head" in base_ckpt:
        m["test_vector_head"].load_state_dict(base_ckpt["test_vector_head"])

    # outcome_head: the refreshed head, not the base run's own outcome head (see the
    # module docstring).
    m["outcome_head"] = OutcomeHeadItemMLP(
        codec_dim=cw5.CW5_DIMS["codec_dim"], latent_dim=cw5.CW5_DIMS["latent_dim"],
        hidden=cw5.CW5_DIMS["attempt_hidden"], n_classes=N_OUTCOME,
        prefix_dim=cw5.CW5_DIMS["attempt_hidden"]).to(device)
    m["outcome_head"].load_state_dict(head_ckpt["outcome_head"])

    for v in m.values():
        if hasattr(v, "eval"):
            v.eval()

    provenance = {
        "substrate_name": HEAD_SUBSTRATE_NAME,
        "base_checkpoint": str(base_checkpoint), "base_checkpoint_sha256": cw5._sha256(base_checkpoint),
        "head_checkpoint": str(head_checkpoint), "head_checkpoint_sha256": cw5._sha256(head_checkpoint),
        "outcome_head_source": "head_checkpoint (refreshed, item 8)",
        "attempt_encoder_filt_hyper_bases_test_vector_head_source": "base_checkpoint (item 7/base run)",
    }
    return m, provenance


def held_out_class_posterior_batch(outcome_head, item_codec: torch.Tensor, mu: torch.Tensor,
                                   log_sigma: torch.Tensor, k: int, seed: int,
                                   device: str) -> "torch.Tensor":
    """`[N, N_OUTCOME]` K-marginalised class posterior for rows whose problem is ABSENT
    from the regular 43-item manifest codec (the 7 held-out items) --
    `item_codec`/`mu`/`log_sigma` are `[N, codec_dim]`/`[N, latent_dim]`/`[N,
    latent_dim]`. Matches the `p_class_meanK` formula of `joint_kc_scores_for_rows` in
    `cw_sim_core` exactly (`mean_k softmax(...)`, never `softmax(mean_k ...)` -- the two
    are not the same quantity under a nonlinear head). The prefix is fixed at zero, the
    convention for a first attempt: these rows are first attempts, and there is no
    within-episode state to condition on for a problem the learner has never touched."""
    n = item_codec.shape[0]
    prefix_dim = outcome_head.prefix_proj.in_features if outcome_head.prefix_proj is not None else None
    prefix = torch.zeros(n, prefix_dim, device=device) if prefix_dim else None
    n_classes = outcome_head.outcome.out_features
    gen = torch.Generator(device=device).manual_seed(seed)
    probs_sum = torch.zeros(n, n_classes, device=device)
    with torch.no_grad():
        for _ in range(k):
            eps = torch.randn(mu.shape, device=device, generator=gen)
            u_j = mu + torch.exp(log_sigma) * eps
            logits = outcome_head(item_codec, u_j, prefix=prefix)
            probs_sum += torch.softmax(logits, dim=-1)
    return probs_sum / k
