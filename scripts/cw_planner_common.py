"""Shared helpers for the CodeWorkout planner scripts.

Used by `cw_planner_suite`, `cw_classifier_belief_handoff`, `cw_sim_belief_handoff`,
`cw_sim_train_handoff` and `atom.simulator.cw_substrate`. Provides the validation
manifest (from raw data, or from retained derived validation predictions), loading of
the CodeWorkout classifier checkpoint, per-episode attempt summaries, the filter belief
before a query item, and K-sample outcome predictions for a candidate and target items.
"""
from __future__ import annotations
import argparse
import pathlib
import sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from cw_manifest_io import load_manifest
from cw_4g_codec_rank_item_mlp import rebuild_problem_codec_at_rank  # noqa: E402
from cw_classifier_common import CW5_DIMS, ITEM_ENCODER, RANK, K_FIXED, TRANSFORM_PATH, _sha256
from atom.codec.transform import CodecTransform  # noqa: E402
from atom.data.cw_manifest import build_codec_vector_lookup  # noqa: E402
from atom.data.cw_outcome_label import N_OUTCOME, OUTCOME_CLASSES  # noqa: E402
from atom.filter.cw_attempt_encoder import AttemptEncoder  # noqa: E402
from atom.filter.cw_filter_batch import build_attempt_batch  # noqa: E402
from atom.filter.episode_v2 import build_prev_item_codec  # noqa: E402
from atom.simulator.cw_item_mlp import OutcomeHeadItemMLP, S1FilterV2ItemMLP  # noqa: E402
from atom.simulator.sampler_semantics import CURRENT_SAMPLER_SEMANTICS  # noqa: E402
REPO = pathlib.Path(__file__).resolve().parents[1]


def _build_manifest(args: argparse.Namespace) -> dict:
    if args.derived_val_predictions:
        return _build_manifest_from_predictions(args)
    base = load_manifest(args)
    code_hashes = set(base["attempts"]["code_hash"].dropna())
    diff_hashes = set(base["attempts"].loc[base["attempts"]["diff_hash"].notna(), "diff_hash"])
    transform = CodecTransform.load(TRANSFORM_PATH)
    code = build_codec_vector_lookup(code_hashes, args.codealone_meta, args.codealone_states,
                                     transform, k=K_FIXED, rank=RANK)
    diff = build_codec_vector_lookup(diff_hashes, args.diff_meta, args.diff_states,
                                     transform, k=K_FIXED, rank=RANK)
    codec_artifact = REPO / "artifacts" / "cw_item_codec_dim64.npz"
    if codec_artifact.exists():
        item = np.load(codec_artifact, allow_pickle=True)
        problem = item["codec"].astype(np.float32)
        pid_to_idx = {int(p): i for i, p in enumerate(item["problem_id"])}
    else:
        problem, pid_to_idx = rebuild_problem_codec_at_rank(RANK, transform=transform)
    return {**base, "code_of_hash": code, "diff_of_hash": diff,
            "problem_codec": problem, "pid_to_idx": pid_to_idx}


def _build_manifest_from_predictions(args: argparse.Namespace) -> dict:
    """Build the planner's validation-only episode table without raw source code.

    The classifier's retained ``val_predictions.npz`` carries attempt/episode/learner/
    problem identities and native four-class labels.  The code/diff vectors come from
    the separately retained hash-keyed derived caches.  This is enough to replay the
    frozen AttemptEncoder/filter/OutcomeHead while keeping licensed source text off a
    remote host.  Limitation: the scalar channels are reconstructed from the native
    class label rather than read from the raw data.
    """
    p = np.load(args.derived_val_predictions, allow_pickle=False)
    required = {"row_id", "learner", "item", "y_true", "episode_id", "attempt_history_index", "prior_episode_count"}
    missing = required - set(p.files)
    if missing:
        raise ValueError(f"derived validation predictions missing keys: {sorted(missing)}")
    ids = p["row_id"].astype(str)
    y = p["y_true"].astype(np.int64)
    if np.any((y < 0) | (y >= N_OUTCOME)):
        raise ValueError("derived validation predictions contain invalid CW class labels")
    attempts = pd.DataFrame({
        "code_state_id": ids, "learner": p["learner"].astype(str),
        "problem_id": p["item"].astype(np.int64), "episode_id": p["episode_id"].astype(str),
        "t": p["attempt_history_index"].astype(np.int64),
        "prior_episode_count": p["prior_episode_count"].astype(np.int64),
        "outcome_code": y, "correct": (y == 3),
        "compile_result": np.where(y == 0, "Error", "Success"),
        "score_frac": np.choose(y, [0.0, 0.0, 0.5, 1.0]).astype(np.float32),
        "n_compile_errors": (y == 0).astype(np.float32),
    }).sort_values(["learner", "t", "code_state_id"], kind="mergesort").reset_index(drop=True)
    code_map, diff_map = __import__("atom.data.cw_manifest", fromlist=["load_hash_maps"]).load_hash_maps(
        args.code_map, args.diff_map)
    attempts = attempts.merge(code_map, on="code_state_id", how="left").merge(diff_map, on="code_state_id", how="left")
    if attempts["code_hash"].isna().any():
        raise ValueError("derived validation rows contain code ids absent from the hash map")
    episodes = attempts.groupby("episode_id", sort=False).agg(
        learner=("learner", "first"), problem_id=("problem_id", "first"),
        start_timestamp=("prior_episode_count", "first"), y_ever=("correct", "max"),
    ).reset_index()
    episodes["start_timestamp"] = pd.Timestamp("2020-01-01") + pd.to_timedelta(
        episodes["start_timestamp"].astype(int), unit="m")
    transform = CodecTransform.load(TRANSFORM_PATH)
    code_hashes = set(attempts["code_hash"])
    diff_hashes = set(attempts.loc[attempts["diff_hash"].notna(), "diff_hash"])
    code = build_codec_vector_lookup(code_hashes, args.codealone_meta, args.codealone_states,
                                     transform, k=K_FIXED, rank=RANK)
    diff = build_codec_vector_lookup(diff_hashes, args.diff_meta, args.diff_states,
                                     transform, k=K_FIXED, rank=RANK)
    codec_artifact = REPO / "artifacts" / "cw_item_codec_dim64.npz"
    if codec_artifact.exists():
        item = np.load(codec_artifact, allow_pickle=True)
        problem = item["codec"].astype(np.float32)
        pid_to_idx = {int(p): i for i, p in enumerate(item["problem_id"])}
    else:
        problem, pid_to_idx = rebuild_problem_codec_at_rank(RANK, transform=transform)
    val_learners = sorted(attempts["learner"].unique())
    return {"attempts": attempts, "episodes": episodes, "code_of_hash": code,
            "diff_of_hash": diff, "problem_codec": problem, "pid_to_idx": pid_to_idx,
            "val_learners": val_learners, "train_learners": [],
            "derived_source": str(args.derived_val_predictions)}


def _build_model(device: str, checkpoint: pathlib.Path) -> dict:
    d = CW5_DIMS
    attempt_encoder = AttemptEncoder(d["codec_dim"], d["codec_dim"], d["attempt_proj_dim"],
                                     d["attempt_hidden"], d["summary_dim"]).to(device)
    filt = S1FilterV2ItemMLP(d["codec_dim"], d["episode_hidden"], d["filter_hidden"],
                             d["latent_dim"], prev_feat_dim=d["summary_dim"],
                             sigma_parameterization="precision_accumulating").to(device)
    outcome_head = OutcomeHeadItemMLP(d["codec_dim"], d["latent_dim"], d["attempt_hidden"],
                                      n_classes=N_OUTCOME, prefix_dim=d["attempt_hidden"]).to(device)
    y_ever_head = OutcomeHeadItemMLP(d["codec_dim"], d["latent_dim"], d["attempt_hidden"],
                                     n_classes=1).to(device)
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    arch = ckpt.get("architecture", {})
    for key, expected in (("item_encoder", ITEM_ENCODER), ("sigma_parameterization", "precision_accumulating")):
        got = arch.get(key, ckpt.get(key))
        if got != expected:
            raise ValueError(f"checkpoint {key}={got!r}, expected {expected!r}")
    if ckpt.get("sampler_semantics") != CURRENT_SAMPLER_SEMANTICS:
        raise ValueError("standalone checkpoint sampler semantics do not match current replay")
    attempt_encoder.load_state_dict(ckpt["attempt_encoder_state_dict"])
    filt.load_state_dict(ckpt["filter_state_dict"])
    outcome_head.load_state_dict(ckpt["outcome_head_state_dict"])
    y_ever_head.load_state_dict(ckpt["y_ever_head_state_dict"])
    return {"attempt_encoder": attempt_encoder, "filt": filt,
            "filt_forward": lambda ic, pf, tm, vm: filt(ic, pf),
            "outcome_head": outcome_head, "y_ever_head": y_ever_head,
            "checkpoint_sha256": _sha256(checkpoint), "checkpoint_meta": ckpt}


@torch.no_grad()
def _episode_summaries(model: dict, manifest: dict, device: str) -> dict[str, torch.Tensor]:
    attempts = manifest["attempts"].sort_values(["learner", "t"], kind="mergesort").reset_index(drop=True)
    # The full held-out replay contains tens of thousands of episodes.  Building one
    # padded tensor for all of them can exceed a 40-GiB GPU even though the encoder is
    # small; this chunking changes no rows, ordering, or weights, and only bounds the
    # peak workspace.  Each summary stays on the same device, so downstream value joins
    # are byte-for-byte equivalent to a single-batch computation.
    episode_ids = attempts["episode_id"].drop_duplicates().tolist()
    summaries: dict[str, torch.Tensor] = {}
    chunk_size = 256
    for start in range(0, len(episode_ids), chunk_size):
        wanted = set(episode_ids[start : start + chunk_size])
        chunk = attempts[attempts["episode_id"].isin(wanted)].reset_index(drop=True)
        batch = build_attempt_batch(chunk, manifest["code_of_hash"], manifest["diff_of_hash"],
                                    CW5_DIMS["codec_dim"], CW5_DIMS["codec_dim"], device)
        with torch.no_grad():
            _, summary = model["attempt_encoder"](batch["code_vec"], batch["diff_vec"],
                                                   batch["scalar_feats"], batch["lengths"])
        summaries.update(dict(zip(batch["episode_ids"], summary)))
    return summaries


def _filter_state(model: dict, events: list[dict], query_codec: torch.Tensor,
                  device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Belief before ``query_codec`` after the chronologically sorted known events."""
    d = CW5_DIMS
    n = len(events)
    codecs = [e["codec"] for e in events] + [query_codec]
    summaries = [e["summary"] for e in events] + [torch.zeros(d["summary_dim"], device=device)]
    item = torch.stack(codecs).unsqueeze(0)
    summary = torch.stack(summaries).unsqueeze(0)
    prev = build_prev_item_codec(summary)
    tm = torch.zeros((1, n + 1), dtype=torch.float32, device=device)
    valid = torch.ones((1, n + 1), dtype=torch.bool, device=device)
    mu, log_sigma = model["filt_forward"](item, prev, tm, valid)
    return mu[0, n], log_sigma[0, n]


def _predict(model: dict, mu: torch.Tensor, log_sigma: torch.Tensor,
             candidate: torch.Tensor, targets: torch.Tensor, eps: torch.Tensor,
             device: str) -> dict[str, torch.Tensor]:
    k, latent = eps.shape
    u = mu.unsqueeze(0) + log_sigma.exp().unsqueeze(0) * eps
    all_items = torch.cat((candidate.unsqueeze(0), targets), dim=0)
    t = all_items.shape[0]
    item = all_items.unsqueeze(0).expand(k, t, -1).reshape(k * t, -1)
    u_flat = u.unsqueeze(1).expand(k, t, latent).reshape(k * t, latent)
    prefix = torch.zeros(k * t, CW5_DIMS["attempt_hidden"], device=device)
    probs = torch.softmax(model["outcome_head"](item, u_flat, prefix=prefix), dim=-1)
    probs = probs.reshape(k, t, N_OUTCOME)
    return {"candidate": probs[:, 0], "targets": probs[:, 1:]}
