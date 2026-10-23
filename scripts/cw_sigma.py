#!/usr/bin/env python
"""Train and evaluate the CodeWorkout classifier under the K-sample mixture likelihood.

Trains the attempt encoder, GRU filter and MLP outcome head with the same probability-
mixture objective as the mathematics simulator, then gates validation NLL, CLEAN_FULL AUC,
the between-learner share of mu, and whether trSigma and the Jensen gap fall from attempt
t=10 to t=100 on a fixed learner cohort.
Inputs: raw CodeWorkout data, code/diff hash maps and codec caches (`--code-map` etc.).
Outputs in `--out-dir`: post_train.pt, final.pt, val_predictions.npz,
fixed_cohort_curve.npz and summary.json; optionally synced to an S3 bucket (`--bucket`).
Run: `python scripts/cw_sigma.py --likelihood kmixture --k-train 8 --eval-k 8 ...`
(``single`` remains the CLI default likelihood).  Serves Table 9 and Figure 6.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import subprocess
import sys
import time
from typing import Any

SCRIPTS_DIR = pathlib.Path(__file__).resolve().parent
REPO = SCRIPTS_DIR.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(SCRIPTS_DIR))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from cw_manifest_io import load_manifest  # noqa: E402
from cw_4g_codec_rank_item_mlp import rebuild_problem_codec_at_rank  # noqa: E402
from cw_classifier_common import (  # noqa: E402
    BODY,
    CLEAN_FULL_IDX,
    CODEC_DIM,
    CW5_DIMS,
    ITEM_ENCODER,
    K_FIXED,
    LOG_SIGMA_MAX,
    LOG_SIGMA_MIN,
    RANK,
    TRANSFORM_PATH,
    _sha256,
    between_learner_variance_share,
    gather_mu_log_sigma_over_population,
)

from atom.codec.transform import CodecTransform  # noqa: E402
from atom.data.cw_manifest import build_codec_vector_lookup  # noqa: E402
from atom.eval.metrics import auc, bootstrap_mean_ci, multiclass_nll  # noqa: E402
from atom.simulator.cw_4_train_loop import evaluate_val, train_one_body  # noqa: E402
from atom.simulator.sampler_semantics import CURRENT_SAMPLER_SEMANTICS  # noqa: E402
from atom.utils.logging import RunLogger  # noqa: E402

REFERENCE_NLL = 0.8840
REFERENCE_AUC = 0.8044
SEED_FLOOR = 0.0036
REFERENCE_MU_SHARE = 0.4358
N_BOOT = 2000
T_LOW = 10
T_HIGH = 100


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO).decode().strip()
    except (OSError, subprocess.SubprocessError):
        return "nogit"


def categorical_jensen_gap(probs_k: torch.Tensor) -> torch.Tensor:
    """Categorical ``H(mean_k p_k) - mean_k H(p_k)`` for ``[N,K,C]`` probabilities."""
    if probs_k.ndim != 3:
        raise ValueError(f"expected probs_k [N,K,C], got shape={tuple(probs_k.shape)}")
    mean_probs = probs_k.mean(dim=1)
    total_entropy = -(mean_probs.clamp(min=1e-8).log() * mean_probs).sum(dim=-1)
    sample_entropy = -(probs_k.clamp(min=1e-8).log() * probs_k).sum(dim=-1).mean(dim=1)
    return (total_entropy - sample_entropy).clamp(min=0.0)


def fixed_history_pairs(
    val: dict[str, Any],
    val_attempts: pd.DataFrame,
    val_episodes: pd.DataFrame,
    mu: np.ndarray,
    log_sigma: np.ndarray,
    t_low: int = T_LOW,
    t_high: int = T_HIGH,
) -> pd.DataFrame:
    """Select the same learners at two exact histories, with a by-value source check.

    Beliefs are emitted at attempt-row granularity, so this first verifies that every
    duplicate attempt row within an episode carries the same belief and then retains one
    representative row.  It independently reconstructs episode position from the source
    episode table and joins by ``(learner, episode_id)``; array position is never used as
    an identity join.
    """
    n = len(val_attempts)
    if not (len(mu) == len(log_sigma) == len(val["attempt_id"]) == n):
        raise AssertionError("belief, persisted identity, and source attempt rows are not row-aligned")

    persisted = pd.DataFrame(
        {
            "row_pos": np.arange(n, dtype=np.int64),
            "attempt_id": val["attempt_id"],
            "learner": val["learner"],
            "episode_id": val["episode_id"],
            "problem_id": val["problem_id"],
            "prior_episode_count": val["prior_episode_count"],
        }
    )
    source_attempt = val_attempts[
        ["code_state_id", "learner", "episode_id", "problem_id", "t"]
    ].rename(columns={"code_state_id": "attempt_id", "learner": "learner_source",
                      "episode_id": "episode_id_source", "problem_id": "problem_id_source",
                      "t": "attempt_t_source"})
    if source_attempt["attempt_id"].duplicated().any():
        raise AssertionError("source code_state_id is not unique; by-value check is ambiguous")
    joined = persisted.merge(source_attempt, on="attempt_id", how="left", validate="one_to_one")
    if joined["learner_source"].isna().any():
        raise AssertionError("persisted validation attempt is absent from source table")
    for carried, source in (("learner", "learner_source"), ("episode_id", "episode_id_source"),
                            ("problem_id", "problem_id_source")):
        if not np.array_equal(joined[carried].to_numpy(), joined[source].to_numpy()):
            raise AssertionError(f"by-value attempt check failed for {carried}")

    episode_rows: list[dict[str, Any]] = []
    for (learner, episode_id), group in persisted.groupby(["learner", "episode_id"], sort=False):
        pos = group["row_pos"].to_numpy(dtype=np.int64)
        if not np.allclose(mu[pos], mu[pos[0]], rtol=0.0, atol=1e-7):
            raise AssertionError(f"mu differs within episode {episode_id!r}")
        if not np.allclose(log_sigma[pos], log_sigma[pos[0]], rtol=0.0, atol=1e-7):
            raise AssertionError(f"log_sigma differs within episode {episode_id!r}")
        if group["prior_episode_count"].nunique() != 1 or group["problem_id"].nunique() != 1:
            raise AssertionError(f"persisted episode metadata differs within episode {episode_id!r}")
        episode_rows.append(
            {
                "learner": learner,
                "episode_id": episode_id,
                "problem_id": group["problem_id"].iloc[0],
                "prior_episode_count": int(group["prior_episode_count"].iloc[0]),
                "row_pos": int(pos[0]),
            }
        )
    observed = pd.DataFrame(episode_rows)

    source_episode = val_episodes.sort_values(
        ["learner", "start_timestamp"], kind="mergesort"
    ).copy()
    source_episode["t_source"] = source_episode.groupby("learner", sort=False).cumcount()
    source_episode = source_episode[["learner", "episode_id", "problem_id", "t_source"]].rename(
        columns={"problem_id": "problem_id_source"}
    )
    checked = observed.merge(
        source_episode, on=["learner", "episode_id"], how="left", validate="one_to_one"
    )
    if checked["t_source"].isna().any():
        raise AssertionError("persisted episode is absent from source episode table")
    if not np.array_equal(
        checked["prior_episode_count"].to_numpy(), checked["t_source"].to_numpy(dtype=np.int64)
    ):
        raise AssertionError("by-value episode check failed for prior_episode_count")
    if not np.array_equal(
        checked["problem_id"].to_numpy(), checked["problem_id_source"].to_numpy()
    ):
        raise AssertionError("by-value episode check failed for problem_id")

    # CodeWorkout's source `t` is the causal attempt index (the corpus's native evidence
    # unit).  A cohort at episode 100 is impossible in this split (at most 86 episodes
    # per learner), whereas exact attempt t=10/t=100 retains 82 learners.  Select by
    # source identity, while carrying the independently checked filter-history position
    # so the cohort also shows that episode-level evidence increased.
    low = joined.loc[
        joined["attempt_t_source"] == t_low,
        ["learner", "row_pos", "prior_episode_count"],
    ].rename(
        columns={"row_pos": "row_low", "prior_episode_count": "episode_history_low"}
    )
    high = joined.loc[
        joined["attempt_t_source"] == t_high,
        ["learner", "row_pos", "prior_episode_count"],
    ].rename(
        columns={"row_pos": "row_high", "prior_episode_count": "episode_history_high"}
    )
    pairs = low.merge(high, on="learner", how="inner", validate="one_to_one").sort_values(
        "learner", kind="mergesort"
    ).reset_index(drop=True)
    if pairs.empty:
        raise AssertionError(f"no validation learner has both attempt t={t_low} and t={t_high}")
    if not bool((pairs["episode_history_high"] > pairs["episode_history_low"]).all()):
        raise AssertionError("attempt-history cohort does not strictly increase episode evidence")
    return pairs


@torch.no_grad()
def evaluate_fixed_curve(
    outcome_head: torch.nn.Module,
    problem_codec: np.ndarray,
    mu: np.ndarray,
    log_sigma: np.ndarray,
    pairs: pd.DataFrame,
    prefix_dim: int,
    k: int,
    seed: int,
    device: str,
) -> dict[str, np.ndarray]:
    """Evaluate trSigma and the Jensen gap on one fixed learner/item cohort with common random numbers."""
    if k < 2:
        raise ValueError("Jensen-gap evaluation requires k >= 2")
    codec = torch.as_tensor(problem_codec, dtype=torch.float32, device=device)
    n_learners = len(pairs)
    latent_dim = mu.shape[1]
    eps = torch.randn(
        n_learners, k, latent_dim,
        generator=torch.Generator(device="cpu").manual_seed(seed),
        dtype=torch.float32,
    ).to(device)
    out: dict[str, list[float]] = {
        "trace_low": [], "trace_high": [], "gap_low": [], "gap_high": [],
        "sigma_low": [], "sigma_high": [],
    }
    outcome_head.eval()
    for i, row in pairs.iterrows():
        row_low, row_high = int(row["row_low"]), int(row["row_high"])
        for suffix, pos in (("low", row_low), ("high", row_high)):
            mu_i = torch.as_tensor(mu[pos], dtype=torch.float32, device=device)
            log_sigma_i = torch.as_tensor(log_sigma[pos], dtype=torch.float32, device=device)
            sigma_i = log_sigma_i.exp()
            u_k = mu_i.unsqueeze(0) + sigma_i.unsqueeze(0) * eps[i]
            probs_k = []
            prefix = torch.zeros(len(codec), prefix_dim, dtype=codec.dtype, device=device)
            for j in range(k):
                u_probe = u_k[j].unsqueeze(0).expand(len(codec), -1)
                probs_k.append(torch.softmax(outcome_head(codec, u_probe, prefix=prefix), dim=-1))
            probs = torch.stack(probs_k, dim=1)  # [P,K,C]
            out[f"trace_{suffix}"].append(float((sigma_i.square()).sum().item()))
            out[f"sigma_{suffix}"].append(float(sigma_i.mean().item()))
            out[f"gap_{suffix}"].append(float(categorical_jensen_gap(probs).mean().item()))
    return {name: np.asarray(values, dtype=np.float64) for name, values in out.items()}


def paired_fall_ci(low: np.ndarray, high: np.ndarray, seed: int) -> dict[str, Any]:
    delta = np.asarray(low, dtype=np.float64) - np.asarray(high, dtype=np.float64)
    mean, ci_lo, ci_hi = bootstrap_mean_ci(delta, n_boot=N_BOOT, alpha=0.05, seed=seed)
    return {
        "mean_t10": float(np.mean(low)),
        "mean_t100": float(np.mean(high)),
        "mean_fall_t10_minus_t100": mean,
        "ci95": [ci_lo, ci_hi],
        "ci_clear_fall": bool(ci_lo > 0.0),
        "n_learners": int(len(delta)),
        "n_boot": N_BOOT,
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, pathlib.Path):
        return str(value)
    return value


def _write_json(path: pathlib.Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n")


def _display_path(path: pathlib.Path) -> str:
    resolved = path.resolve()
    return str(resolved.relative_to(REPO)) if REPO in resolved.parents else str(resolved)


def _arm_fuse(enabled: bool, projected_remaining_minutes: float) -> int | None:
    if not enabled:
        return None
    minutes = max(60, int(math.ceil(2.0 * max(1.0, projected_remaining_minutes))))
    subprocess.run(["sudo", "shutdown", "-c"], check=False)
    subprocess.run(["sudo", "shutdown", "-h", f"+{minutes}"], check=True)
    print(f"[cw-sigma] OS fuse armed for {minutes} minutes", flush=True)
    return minutes


def _sync_to_s3(bucket: str, prefix: str, out_dir: pathlib.Path, run_dir: pathlib.Path) -> None:
    if not bucket:
        return
    subprocess.run(
        ["aws", "s3", "sync", str(out_dir), f"s3://{bucket}/{prefix}/results/", "--only-show-errors"],
        check=True,
    )
    subprocess.run(
        ["aws", "s3", "sync", str(run_dir), f"s3://{bucket}/{prefix}/run/", "--only-show-errors"],
        check=True,
    )
    print(f"[cw-sigma] synced derived artifacts to bucket {bucket} under {prefix}/", flush=True)


def _checkpoint_base(model: dict[str, Any], args: argparse.Namespace, git_sha: str) -> dict[str, Any]:
    split_path = REPO / "splits" / "cw_learner_split.csv"
    return {
        "attempt_encoder_state_dict": model["attempt_encoder"].state_dict(),
        "filter_state_dict": model["filt"].state_dict(),
        "outcome_head_state_dict": model["outcome_head"].state_dict(),
        "y_ever_head_state_dict": model["y_ever_head"].state_dict(),
        "optimizer_state_dict": model["optimizer_state_dict"],
        "torch_rng_state": model["torch_rng_state"],
        "torch_cuda_rng_state_all": model["torch_cuda_rng_state_all"],
        "numpy_rng_state": model["numpy_rng_state"],
        "python_rng_state": model["python_rng_state"],
        "architecture": {**CW5_DIMS, "body": BODY, "item_encoder": ITEM_ENCODER},
        "codec_path": str(TRANSFORM_PATH.relative_to(REPO)),
        "codec_sha256": _sha256(TRANSFORM_PATH),
        "codec_rank": RANK,
        "codec_k": K_FIXED,
        "learner_split_path": str(split_path.relative_to(REPO)),
        "learner_split_sha256": _sha256(split_path),
        "item_split_sha256": None,
        "train_regime": "cw_4_train_loop.train_one_body standalone; no backbone/LoRA/hypernetwork",
        "seed": args.seed,
        "epochs": args.epochs,
        "learner_batch_size": args.learner_batch_size,
        "learning_rate": args.lr,
        "beta_kl": args.beta_kl,
        "free_bits": args.free_bits,
        "likelihood": args.likelihood,
        "sigma_parameterization": args.sigma_parameterization,
        "k_train": args.k_train,
        "k_eval": args.eval_k,
        "y_ever_likelihood": "historical single-sample auxiliary (hybrid objective disclosed)",
        "git_sha": git_sha,
        "sampler_semantics": CURRENT_SAMPLER_SEMANTICS,
    }


def _estimate() -> dict[str, Any]:
    return {
        "registered_minutes": 30,
        "recommended_fuse_minutes": 60,
        "instance_type": "g5.xlarge",
        "memory_arithmetic": {
            "host_manifest_and_vector_cache_gib": 6.0,
            "model_optimizer_and_eval_working_set_gib": 1.0,
            "host_total_with_2x_headroom_gib": 14.0,
            "g5_xlarge_host_ram_gib": 16,
            "gpu_working_set_gib": 1.0,
            "g5_xlarge_gpu_ram_gib": 24,
        },
        "note": "No backbone is loaded; host-side vector reconstruction is the dominant allocation.",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default=None)
    ap.add_argument("--code-map", required=False)
    ap.add_argument("--diff-map", required=False)
    ap.add_argument("--codealone-meta", required=False)
    ap.add_argument("--codealone-states", required=False)
    ap.add_argument("--diff-meta", required=False)
    ap.add_argument("--diff-states", required=False)
    ap.add_argument("--learner-batch-size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--beta-kl", type=float, default=0.01)
    ap.add_argument("--free-bits", type=float, default=0.05)
    ap.add_argument("--k-train", type=int, default=8)
    ap.add_argument("--eval-k", type=int, default=8)
    ap.add_argument("--likelihood", choices=("single", "kmixture"), default="single")
    ap.add_argument(
        "--sigma-parameterization",
        choices=("direct", "precision_accumulating"),
        default="direct",
    )
    ap.add_argument("--nll-reference", type=float, default=REFERENCE_NLL)
    ap.add_argument("--auc-reference", type=float, default=REFERENCE_AUC)
    ap.add_argument("--seed-floor", type=float, default=SEED_FLOOR)
    ap.add_argument("--mu-share-reference", type=float, default=REFERENCE_MU_SHARE)
    ap.add_argument("--out-dir", default="results/cw_sigma")
    ap.add_argument("--bucket", default="")
    ap.add_argument("--out-prefix", default="cw-sigma")
    ap.add_argument("--estimated-minutes", type=float, default=30.0)
    ap.add_argument("--self-terminate", action="store_true")
    ap.add_argument("--git-sha", default=None)
    ap.add_argument("--estimate-only", action="store_true")
    args = ap.parse_args()

    if args.estimate_only:
        print(json.dumps(_estimate(), indent=2))
        return 0
    required = ("code_map", "diff_map", "codealone_meta", "codealone_states", "diff_meta", "diff_states")
    missing = [name for name in required if not getattr(args, name)]
    if missing:
        ap.error("the following are required outside --estimate-only: " + ", ".join("--" + x.replace("_", "-") for x in missing))
    if args.k_train < 1 or args.eval_k < 2:
        ap.error("--k-train must be >=1 and --eval-k must be >=2")
    if args.likelihood == "kmixture" and args.k_train < 8:
        ap.error("kmixture training requires --k-train >= 8")
    if args.k_train != args.eval_k:
        ap.error("this evaluation requires --k-train == --eval-k (default value: 8)")

    git_sha = args.git_sha or _git_sha()
    t_start = time.time()
    fuse_enabled = bool(args.bucket or args.self_terminate)
    _arm_fuse(fuse_enabled, args.estimated_minutes)
    out_dir = REPO / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = RunLogger(
        config={
            "task": "cw_sigma", "body": BODY, "codec_dim": CODEC_DIM,
            "item_encoder": ITEM_ENCODER, "seed": args.seed, "epochs": args.epochs,
            "beta_kl": args.beta_kl, "free_bits": args.free_bits,
            "likelihood": args.likelihood, "k_train": args.k_train, "k_eval": args.eval_k,
            "sigma_parameterization": args.sigma_parameterization,
            "estimated_minutes": args.estimated_minutes,
            "sampler_semantics": CURRENT_SAMPLER_SEMANTICS,
        },
        data_version="CSEDM-S19+F19-challenge-train-pool+val",
        seed=args.seed,
    )
    # Minimal deployment bundles omit `.git`; the caller then supplies the exact commit
    # with --git-sha, and the run manifest records it instead of RunLogger's fallback
    # `nogit`.
    if args.git_sha:
        logger.git_sha = git_sha
        logger._write_manifest()

    def elapsed_minutes() -> float:
        return (time.time() - t_start) / 60.0

    def sync() -> None:
        _sync_to_s3(args.bucket, args.out_prefix, out_dir, logger.run_dir)

    def overrun(stage: str) -> bool:
        elapsed = elapsed_minutes()
        if elapsed <= 2.0 * args.estimated_minutes:
            return False
        trigger = {
            "task": "cw_sigma", "stage": stage, "elapsed_minutes": elapsed,
            "registered_estimated_minutes": args.estimated_minutes,
            "threshold_minutes": 2.0 * args.estimated_minutes, "git_sha": git_sha,
        }
        _write_json(out_dir / "overrun_trigger.json", trigger)
        logger.summary(status="overrun-trigger", **trigger)
        sync()
        if args.self_terminate:
            subprocess.run(["sudo", "shutdown", "-h", "now"], check=False)
        return True

    logger.heartbeat(step=0, force=True, stage="loading manifest")
    base_manifest = load_manifest(args)
    code_hashes = set(base_manifest["attempts"]["code_hash"].dropna())
    diff_hashes = set(base_manifest["attempts"].loc[
        base_manifest["attempts"]["diff_hash"].notna(), "diff_hash"
    ])
    transform = CodecTransform.load(TRANSFORM_PATH)
    code_of_hash = build_codec_vector_lookup(
        code_hashes, args.codealone_meta, args.codealone_states, transform, k=K_FIXED, rank=RANK
    )
    diff_of_hash = build_codec_vector_lookup(
        diff_hashes, args.diff_meta, args.diff_states, transform, k=K_FIXED, rank=RANK
    )
    problem_codec, pid_to_idx = rebuild_problem_codec_at_rank(RANK, transform=transform)
    if len(problem_codec) != 50 or len(pid_to_idx) != 50:
        raise AssertionError(f"fixed diagnostic cohort requires all 50 problems, got {len(problem_codec)}")
    if len(base_manifest["val_learners"]) != 110:
        raise AssertionError(
            f"frozen learner cohort requires 110 validation learners, got {len(base_manifest['val_learners'])}"
        )
    manifest = {
        **base_manifest,
        "code_of_hash": code_of_hash,
        "diff_of_hash": diff_of_hash,
        "problem_codec": problem_codec,
        "pid_to_idx": pid_to_idx,
    }
    logger.heartbeat(
        step=1, force=True, stage="manifest built", n_attempts=len(manifest["attempts"]),
        n_train_learners=len(manifest["train_learners"]), n_val_learners=len(manifest["val_learners"]),
    )
    _arm_fuse(fuse_enabled, max(1.0, args.estimated_minutes - elapsed_minutes()))
    sync()
    if overrun("manifest"):
        return 2

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    logger.heartbeat(step=2, force=True, stage="training", device=device)
    n_batches_per_epoch = max(
        1,
        (len(manifest["train_learners"]) + args.learner_batch_size - 1) // args.learner_batch_size,
    )
    progress_state = {"last_synced_epoch": 0}

    def training_progress(step: int, total: int, epoch: int, outcome_loss: float,
                          y_ever_loss: float) -> None:
        logger.heartbeat(step=step, total=total, stage="training", epoch=epoch,
                         outcome_loss=round(outcome_loss, 6), y_ever_loss=round(y_ever_loss, 6))
        epoch_ended = step % n_batches_per_epoch == 0
        if epoch_ended and epoch % 10 == 0 and progress_state["last_synced_epoch"] != epoch:
            logger.heartbeat(step=step, total=total, force=True, stage="training tranche retained", epoch=epoch)
            _arm_fuse(fuse_enabled, max(1.0, args.estimated_minutes - elapsed_minutes()))
            sync()
            progress_state["last_synced_epoch"] = epoch

    model = train_one_body(
        BODY, manifest, args, device, seed=args.seed, dims={"codec_dim": CODEC_DIM},
        item_encoder=ITEM_ENCODER, beta_kl=args.beta_kl, k=args.k_train,
        free_bits=args.free_bits, likelihood=args.likelihood, progress_callback=training_progress,
        sigma_parameterization=args.sigma_parameterization,
    )
    checkpoint_base = _checkpoint_base(model, args, git_sha)
    post_train_path = out_dir / "post_train.pt"
    torch.save({**checkpoint_base, "checkpoint_stage": "post_train", "elapsed_s": time.time() - t_start}, post_train_path)
    logger.heartbeat(
        step=3, force=True, stage="post-train checkpoint retained", n_steps=model["n_steps"],
        train_s=round(model["train_s"], 1), checkpoint=_display_path(post_train_path),
    )
    _arm_fuse(fuse_enabled, max(1.0, args.estimated_minutes - elapsed_minutes()))
    sync()
    if overrun("post_train"):
        return 2

    val = evaluate_val(
        model, manifest, device, latent_dim=CW5_DIMS["latent_dim"], codec_dim=CODEC_DIM,
        beta_kl=args.beta_kl, k=args.eval_k, free_bits=args.free_bits, likelihood=args.likelihood,
    )
    nll = multiclass_nll(val["probs_real"], val["target"])
    auc_clean_full = auc((val["target"] == CLEAN_FULL_IDX).astype(np.int64), val["probs_real"][:, CLEAN_FULL_IDX])
    val_attempts = manifest["attempts"][
        manifest["attempts"]["learner"].isin(manifest["val_learners"])
    ].reset_index(drop=True)
    val_episodes = manifest["episodes"][
        manifest["episodes"]["learner"].isin(manifest["val_learners"])
    ].reset_index(drop=True)
    mu, log_sigma = gather_mu_log_sigma_over_population(model, val_attempts, val_episodes, manifest, device)
    mean_sigma = float(np.exp(log_sigma).mean())
    off_both_clamps = bool(np.all((log_sigma > LOG_SIGMA_MIN) & (log_sigma < LOG_SIGMA_MAX)))
    mu_share = between_learner_variance_share(mu, np.asarray(val["learner"]))
    pairs = fixed_history_pairs(val, val_attempts, val_episodes, mu, log_sigma)
    curve = evaluate_fixed_curve(
        model["outcome_head"], problem_codec, mu, log_sigma, pairs,
        prefix_dim=CW5_DIMS["attempt_hidden"], k=args.eval_k, seed=args.seed + 9100, device=device,
    )
    trace = paired_fall_ci(curve["trace_low"], curve["trace_high"], seed=args.seed + 9200)
    gap = paired_fall_ci(curve["gap_low"], curve["gap_high"], seed=args.seed + 9300)
    sigma_curve = paired_fall_ci(curve["sigma_low"], curve["sigma_high"], seed=args.seed + 9400)

    gates = {
        "trace_ci_clear_fall": trace["ci_clear_fall"],
        "jensen_gap_ci_clear_fall": gap["ci_clear_fall"],
        "nll_within_seed_floor": bool(nll <= args.nll_reference + args.seed_floor),
        "auc_within_seed_floor": bool(auc_clean_full >= args.auc_reference - args.seed_floor),
        "mu_share_not_reduced": bool(mu_share >= args.mu_share_reference),
        "log_sigma_off_both_clamps": off_both_clamps,
    }
    passed = bool(all(gates.values()))
    metrics = {
        "nll": nll,
        "auc_clean_full": auc_clean_full,
        "mean_sigma": mean_sigma,
        "mu_between_learner_variance_share": mu_share,
        "log_sigma_min": float(log_sigma.min()),
        "log_sigma_max": float(log_sigma.max()),
        "fixed_cohort": {
            "history_unit": "source_causal_attempt_index",
            "t_low": T_LOW, "t_high": T_HIGH, "n_learners": len(pairs),
            "episode_history_low": {
                "min": int(pairs["episode_history_low"].min()),
                "mean": float(pairs["episode_history_low"].mean()),
                "max": int(pairs["episode_history_low"].max()),
            },
            "episode_history_high": {
                "min": int(pairs["episode_history_high"].min()),
                "mean": float(pairs["episode_history_high"].mean()),
                "max": int(pairs["episode_history_high"].max()),
            },
            "n_problem_probes": int(len(problem_codec)), "k_eval": args.eval_k,
            "trace_sigma": trace, "categorical_jensen_gap": gap, "mean_sigma": sigma_curve,
        },
        "thresholds": {
            "nll_max": args.nll_reference + args.seed_floor,
            "auc_min": args.auc_reference - args.seed_floor,
            "mu_share_min": args.mu_share_reference,
        },
        "gates": gates,
        "passed": passed,
    }

    np.savez_compressed(
        out_dir / "val_predictions.npz",
        row_id=np.asarray(val["attempt_id"], dtype=str),
        learner=np.asarray(val["learner"], dtype=str),
        item=np.asarray(val["problem_id"]),
        split=np.asarray(val["split_flag"], dtype=str),
        y_true=np.asarray(val["target"], dtype=np.int64),
        y_pred=np.asarray(val["probs_real"], dtype=np.float32),
        prior_episode_count=np.asarray(val["prior_episode_count"], dtype=np.int64),
        attempt_history_index=val_attempts["t"].to_numpy(dtype=np.int64),
        episode_id=np.asarray(val["episode_id"], dtype=str),
    )
    np.savez_compressed(
        out_dir / "fixed_cohort_curve.npz",
        learner=np.asarray(pairs["learner"], dtype=str),
        row_low=pairs["row_low"].to_numpy(dtype=np.int64),
        row_high=pairs["row_high"].to_numpy(dtype=np.int64),
        episode_history_low=pairs["episode_history_low"].to_numpy(dtype=np.int64),
        episode_history_high=pairs["episode_history_high"].to_numpy(dtype=np.int64),
        problem_id=np.asarray(sorted(pid_to_idx, key=pid_to_idx.get)),
        **curve,
    )
    final_path = out_dir / "final.pt"
    torch.save(
        {
            **checkpoint_base,
            "checkpoint_stage": "final",
            "headline_metrics": metrics,
            "prediction_path": _display_path(out_dir / "val_predictions.npz"),
            "fixed_cohort_curve_path": _display_path(out_dir / "fixed_cohort_curve.npz"),
            "elapsed_s": time.time() - t_start,
        },
        final_path,
    )
    summary = {
        "task": "cw_sigma", "status": "pass" if passed else "gate-fail", "git_sha": git_sha,
        "run_id": logger.run_id, "device": device, "config": logger.config,
        "metrics": metrics, "artifacts": {
            "post_train_checkpoint": _display_path(post_train_path),
            "final_checkpoint": _display_path(final_path),
            "predictions": _display_path(out_dir / "val_predictions.npz"),
            "fixed_cohort_curve": _display_path(out_dir / "fixed_cohort_curve.npz"),
        },
        "elapsed_s": time.time() - t_start,
    }
    _write_json(out_dir / "summary.json", summary)
    logger.log_metrics(**metrics)
    logger.summary(status=summary["status"], nll=nll, auc_clean_full=auc_clean_full,
                   mean_sigma=mean_sigma, mu_share=mu_share, trace_fall=trace, gap_fall=gap)
    logger.heartbeat(step=4, force=True, stage="final artifacts retained and gate evaluated", passed=passed)
    _arm_fuse(fuse_enabled, 5.0)
    sync()
    if overrun("final"):
        return 2
    print(json.dumps(_jsonable(summary), indent=2), flush=True)
    if args.self_terminate:
        if not args.bucket:
            raise RuntimeError("refusing --self-terminate without --bucket confirmation path")
        print("[cw-sigma] final derived artifacts synced; self-terminating", flush=True)
        subprocess.run(["sudo", "shutdown", "-h", "now"], check=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
