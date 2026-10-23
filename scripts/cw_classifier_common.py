"""Shared helpers for `cw_sigma`, `cw_sim_head_refresh` and `cw_sim_tv_head_refresh`.

Provides the CodeWorkout model constants (filter body, item encoder, codec rank and
width, belief clamp bounds), a between-learner variance share (ICC(1)) of the belief,
and a population-wide gather of the belief `(mu, log_sigma)` aligned to attempt rows.
"""
from __future__ import annotations
import hashlib
import pathlib
import sys
SCRIPTS_DIR = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR.parents[0] / "src"))
sys.path.insert(0, str(SCRIPTS_DIR))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from atom.data.cw_outcome_label import OUTCOME_CLASSES  # noqa: E402
from atom.filter.cw_filter_batch import build_attempt_batch, build_episode_batch  # noqa: E402
from atom.simulator.cw_4_train_loop import (  # noqa: E402
    CODEC_K,
    DEFAULT_DIMS as CW4_DEFAULT_DIMS,
    _distinct_episode_positions,
    run_one_seed_body,
)
from atom.simulator.fa_train_step import gather_belief_for_batch  # noqa: E402
REPO = pathlib.Path(__file__).resolve().parents[1]
BODY = "gru"
ITEM_ENCODER = "mlp"  # the MLP item encoder is used for every CodeWorkout run.
K_FIXED = CODEC_K
RANK = 69
CODEC_DIM = RANK - K_FIXED
CW5_DIMS = {**CW4_DEFAULT_DIMS, "codec_dim": CODEC_DIM}
TRANSFORM_PATH = REPO / "transforms" / "problem_mean_L15.npz"
CLEAN_FULL_IDX = OUTCOME_CLASSES.index("CLEAN_FULL")
LOG_SIGMA_MIN, LOG_SIGMA_MAX = -5.0, 2.0  # the same clamp bounds as the training run, so beliefs match.


def _sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def between_learner_variance_share(values: np.ndarray, learner_ids: np.ndarray) -> float:
    """One-way random-effects ICC(1) (Shrout & Fleiss 1979), per latent dimension,
    averaged across dimensions. The unbiased ANOVA estimator CAN return a negative
    share when within-group (within-learner) variance exceeds the between-group
    estimate. That is not a bug but a diagnostic signal: a negative share means mu
    varies more within a learner over time than between learners, i.e. the belief
    encodes session position rather than identity.

    `var_between = (MSB - MSW) / n_bar` (unbalanced-groups n_bar, standard formula);
    `share = var_between / (var_between + var_within)`. Computed at ATTEMPT-ROW
    granularity (an episode's mu is duplicated across its own attempts), so learners
    are weighted by attempt count; a per-episode (unduplicated) version would weight
    them by episode count and could differ."""
    df = pd.DataFrame(values)
    n_dims = values.shape[1]
    df["learner"] = learner_ids
    shares = []
    for col in range(n_dims):
        groups = df.groupby("learner")[col]
        k = groups.ngroups
        n = len(df)
        if k < 2 or n <= k:
            continue
        grand_mean = float(df[col].mean())
        n_i = groups.size().to_numpy().astype(np.float64)
        means_i = groups.mean().to_numpy()
        ssb = float((n_i * (means_i - grand_mean) ** 2).sum())
        msb = ssb / (k - 1)
        ssw = float(sum(((g - g.mean()) ** 2).sum() for _, g in groups))
        msw = ssw / (n - k)
        n_bar = (n - (n_i ** 2).sum() / n) / (k - 1)
        if n_bar <= 0:
            continue
        var_between = (msb - msw) / n_bar
        var_within = msw
        denom = var_between + var_within
        if denom == 0:
            continue
        shares.append(var_between / denom)
    return float(np.mean(shares)) if shares else float("nan")


@torch.no_grad()
def gather_mu_log_sigma_over_population(model: dict, attempts_batch: pd.DataFrame,
                                        episodes_batch: pd.DataFrame, manifest: dict,
                                        device: str) -> tuple[np.ndarray, np.ndarray]:
    """Re-run `forward_batch`'s construction (attempt encoder -> episode batch ->
    distinct episode positions -> belief gather) independently, so that
    `losses_for_cw_hierarchical_batch` keeps its 3-tuple return (the same convention
    as `y_ever_loss_for_batch`). Returns `(mu, log_sigma)`, each
    `[N_attempts, latent_dim]`, row-aligned with `attempts_batch` via
    `attempt_episode_row_idx`, exactly as `forward_batch` computes
    `u_attempt = u_ep[attempt_episode_row_idx]`."""
    attempt_batch = build_attempt_batch(attempts_batch, manifest["code_of_hash"], manifest["diff_of_hash"],
                                        CODEC_DIM, CODEC_DIM, device)
    _, summary = model["attempt_encoder"](attempt_batch["code_vec"], attempt_batch["diff_vec"],
                                          attempt_batch["scalar_feats"], attempt_batch["lengths"])
    ep_batch = build_episode_batch(episodes_batch, manifest["problem_codec"], manifest["pid_to_idx"],
                                   summary, attempt_batch["episode_ids"], device)
    ep_learner_idx, ep_t_idx = _distinct_episode_positions(episodes_batch, ep_batch,
                                                           attempt_batch["episode_ids"], device)
    mu_sel, log_sigma_sel = gather_belief_for_batch(model["filt_forward"], ep_batch, ep_learner_idx, ep_t_idx)
    mu_attempt = mu_sel[attempt_batch["attempt_episode_row_idx"]]
    log_sigma_attempt = log_sigma_sel[attempt_batch["attempt_episode_row_idx"]]
    return mu_attempt.cpu().numpy(), log_sigma_attempt.cpu().numpy()
