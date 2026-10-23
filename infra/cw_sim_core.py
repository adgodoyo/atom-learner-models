"""Shared helpers for training and evaluating the CodeWorkout generative simulator.

Used by `cw_sim_full_run`, `cw_sim_launch2`, `cw_sim_launch2_model`, `cw_sim_launch2_tiktoc`,
`cw_ext_1_generate`, `cw_ext_1_row_selection` and `cw_sim_smoke_local`. Provides: manifest
loading (CodeWorkout S19+F19 attempts, held-out items removed, dim-64 codec vectors); model
construction (attempt encoder, belief filter, outcome heads, hypernetwork over a shared LoRA
basis on a frozen backbone) and warm starts; the microbatched training step; four-arm
scoring of the joint (belief sample, outcome class) mixture NLL; greedy and batched
ancestral generation; resumable checkpointing; and `main()`, the CLI behind
`cw_sim_full_run` (thesis Table 11).
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import pathlib
import random
import subprocess
import sys
import time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
REPO = pathlib.Path(__file__).resolve().parents[1]
from atom.binding import (GlobalMixingVector, Hypernetwork, attach_lora_basis,  # noqa: E402
                          build_second_basis, reparameterize_from_log_sigma)
from atom.binding.derangement import (assert_learner_matched_derangement,  # noqa: E402
                                      learner_history_matched_derangement)
from atom.codec.backbone import load_backbone  # noqa: E402
from atom.codec.extract import encode_one  # noqa: E402
from atom.codec.transform import CodecTransform  # noqa: E402
from atom.data.cw_challenge_split import HELD_OUT_ITEMS  # noqa: E402
from atom.data.cw_manifest import (attach_code_diff_hashes, attach_prior_attempt_code,  # noqa: E402
                                   build_codec_vector_lookup, build_episode_level_table,
                                   load_attempts_with_episodes, load_hash_maps)
from atom.data.cw_outcome_label import CW_OUTCOME_CLASS_TOKENS, N_OUTCOME, outcome_label  # noqa: E402
from atom.filter.cw_attempt_encoder import AttemptEncoder  # noqa: E402
from atom.filter.factory import build_filter  # noqa: E402
from atom.filter.losses import kl_anneal_weight  # noqa: E402
from atom.eval.cw_generation import (completed_row_indices, load_numeric_generation_rows,  # noqa: E402
                                     numeric_generation_row, save_numeric_generation_rows,
                                     summarise_generation_rows)
from atom.heads.fa_hierarchical_heads import OutcomeHead  # noqa: E402
from atom.simulator.cw_4_train_loop import (FACTORY_KW,  # noqa: E402
                                            _distinct_episode_positions, forward_batch,
                                            assert_row_identity_join_canary, row_identity)
from atom.simulator.cw_4_train_loop import DEFAULT_DIMS as CW4_DEFAULT_DIMS  # noqa: E402
from atom.simulator.cw_alt import (build_reserved_prompt_ids, collate, encode_example,  # noqa: E402
                                   forward_global_control, forward_personalised,
                                   joint_kc_mixture_nll, production_log_likelihood_sum_per_row,
                                   production_nll_per_row, production_topk_per_row)
from atom.simulator.cw_item_mlp import OutcomeHeadItemMLP, S1FilterV2ItemMLP  # noqa: E402
from atom.simulator.cw_train_step import (gather_belief_for_batch,  # noqa: E402
                                          gather_item_codec_for_batch,
                                          gather_prefix_for_attempts,
                                          grad_norm_at_belief_junction)
from atom.simulator.joint_set_train_step import (TeacherPosterior, joint_set_kmixture_nll,
                                                 teacher_posterior_kl)  # noqa: E402
from atom.simulator.fa_alt2 import pick_lambda_a  # noqa: E402
from atom.simulator.sampler_semantics import CURRENT_SAMPLER_SEMANTICS  # noqa: E402
from atom.utils.checkpoints import has_checkpoint, load_checkpoint, save_checkpoint  # noqa: E402
from atom.utils.epoch_sampler import ShuffledEpochSampler  # noqa: E402
from atom.utils.logging import default_device  # noqa: E402
from atom.utils.overrun_trigger import OverrunTrigger  # noqa: E402
from atom.utils.seeds import seed_everything  # noqa: E402
ARMS = ("personalised", "dw0", "shuffled_u", "u_mean")  # the global-basis arm is not scored here
HEADLINE_PROBLEM_IDS = (1, 3, 5, 12, 13, 17, 20, 21, 22, 24, 25, 46, 71)


def select_first_attempts_on_headline(attempts: pd.DataFrame, val_learners,
                                      headline_ids=HEADLINE_PROBLEM_IDS) -> pd.DataFrame:
    """One row per (val learner, headline problem) pair -- that learner's EARLIEST
    attempt at that problem, by `t` (the per-learner causal attempt index
    `cw_events.build_attempts` assigns; NOT `timestamp` directly, matching every other
    per-learner ordering in this codebase). A learner with zero attempts on any
    headline problem contributes zero rows (never a KeyError, never a synthetic row)."""
    val_learners = set(val_learners)
    df = attempts[attempts["learner"].isin(val_learners)
                  & attempts["problem_id"].astype(int).isin(set(headline_ids))]
    df = df.sort_values(["learner", "problem_id", "t"], kind="mergesort")
    return df.groupby(["learner", "problem_id"], sort=False).head(1).reset_index(drop=True)
LOG_SIGMA_MIN, LOG_SIGMA_MAX = -5.0, 2.0
MAX_LEN = 512  # chosen from the measured prompt-length distribution
MIN_COMPLETION_TOKENS = 8
CODEC_RANK = 69  # rank=69, k=5 -> codec_dim=64 (codec_dim = rank - k)
CW5_DIMS = {**CW4_DEFAULT_DIMS, "codec_dim": 64}  # the classifier's dims, with a 64-dim codec
PROBLEM_CODEC_DIM64_PATH = REPO / "artifacts" / "cw_item_codec_dim64.npz"


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO).decode().strip()
    except Exception:
        return "unknown"


def _sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_problem_statements() -> dict[int, str]:
    entries = json.loads((REPO / "artifacts" / "cw_problem_statements.json").read_text())
    return {int(e["problem_id"]): e["wrapped_prompt"] for e in entries}


def load_cw5_manifest(args) -> dict:
    """Load the attempt-level manifest with raw code text (`include_code=True`, needed for
    the production head's prompt/completion) and prior-attempt code.

    Enforces two things, both asserted rather than merely configured: the 7 held-out
    items are excluded from targets AND histories (`HELD_OUT_ITEMS` -- a target-only
    removal leaks, since the held-out outcome would still enter as history when
    predicting later events), and the codec vectors are built at `codec_dim=64`
    (rank=69, k=5), not the outcome-only classifier's 45."""
    frames = []
    for release in ["S19", "F19"]:
        from atom.data.cw_events import build_attempts, load_raw_tables
        main, code_states = load_raw_tables(release, base_dir=args.raw_dir, include_code=True)
        a = build_attempts(main, code_states)
        a["release"] = release
        frames.append(a)
    attempts = pd.concat(frames, ignore_index=True)
    from atom.data.cw_sessions import sessionize
    from atom.data.cw_manifest import TAU_SESSION_MINUTES
    attempts = sessionize(attempts, TAU_SESSION_MINUTES)

    split = pd.read_csv(REPO / "splits" / "cw_learner_split.csv")
    train_learners = set(split.loc[split["split"] == "train", "learner"])
    val_learners = set(split.loc[split["split"] == "val", "learner"])
    attempts = attempts[attempts["learner"].isin(train_learners | val_learners)].reset_index(drop=True)

    statements = load_problem_statements()
    attempts = attempts[attempts["problem_id"].isin(statements.keys())].reset_index(drop=True)
    n_before_holdout = attempts["problem_id"].nunique()
    attempts = attempts[~attempts["problem_id"].isin(HELD_OUT_ITEMS)].reset_index(drop=True)
    assert not (set(attempts["problem_id"].unique()) & set(HELD_OUT_ITEMS)), (
        "HELD_OUT_ITEMS leaked into the train/history pool -- the item holdout is "
        "binding for this script, not advisory")
    n_after_holdout = attempts["problem_id"].nunique()
    attempts = attach_prior_attempt_code(attempts)

    code_map, diff_map = load_hash_maps(args.code_map, args.diff_map)
    attempts, hash_stats = attach_code_diff_hashes(attempts, code_map, diff_map)

    labels = outcome_label(attempts["compile_result"], attempts["score_frac"])
    attempts = attempts.assign(outcome_code=labels["outcome_code"], outcome_valid=labels["outcome_valid"])
    attempts = attempts[attempts["outcome_valid"]].reset_index(drop=True)

    transform = CodecTransform.load(REPO / "transforms" / "problem_mean_L15.npz")
    code_hashes_needed = set(attempts["code_hash"].dropna())
    diff_hashes_needed = set(attempts.loc[attempts["diff_hash"].notna(), "diff_hash"])
    # rank=69 pinned explicitly (codec_dim = rank - k, k=5 fixed): this model uses
    # codec_dim=64, not the classifier's default of 45 (rank=50).
    code_of_hash = build_codec_vector_lookup(code_hashes_needed, args.codealone_meta,
                                             args.codealone_states, transform, k=5, rank=CODEC_RANK)
    diff_of_hash = build_codec_vector_lookup(diff_hashes_needed, args.diff_meta,
                                             args.diff_states, transform, k=5, rank=CODEC_RANK)
    for name, lookup in (("code_of_hash", code_of_hash), ("diff_of_hash", diff_of_hash)):
        if lookup:
            dim = next(iter(lookup.values())).shape[0]
            assert dim == 64, f"{name} built at dim={dim}, expected 64 (rank={CODEC_RANK}, k=5)"

    episodes = build_episode_level_table(attempts)

    item = np.load(PROBLEM_CODEC_DIM64_PATH, allow_pickle=True)
    problem_codec = item["codec"].astype(np.float32)
    assert problem_codec.shape[1] == 64, (
        f"{PROBLEM_CODEC_DIM64_PATH.name} is {problem_codec.shape[1]}-dim, expected 64")
    pid_to_idx = {int(p): i for i, p in enumerate(item["problem_id"])}
    # The artifact covers the full 50-item universe, a superset of the 43 items used
    # here -- extra unused rows are harmless. What matters is coverage: every item this model will actually index into
    # (i.e. every problem_id left in `attempts` after the holdout filter above) must
    # resolve, or `build_episode_batch` fails opaquely deep in a training step instead
    # of here, at load time, with a clear message.
    missing = set(attempts["problem_id"].unique()) - set(pid_to_idx.keys())
    assert not missing, f"{len(missing)} in-pool problem_id(s) absent from the dim64 problem codec: {missing}"

    return {
        "attempts": attempts, "episodes": episodes, "code_of_hash": code_of_hash,
        "diff_of_hash": diff_of_hash, "problem_codec": problem_codec, "pid_to_idx": pid_to_idx,
        "train_learners": train_learners, "val_learners": val_learners,
        "statements": statements, "hash_stats": hash_stats,
        "n_problems_before_holdout": n_before_holdout, "n_problems_after_holdout": n_after_holdout,
    }


def build_cw5_examples(attempts_batch: pd.DataFrame, statements: dict, tok) -> list[dict]:
    """One `cw_alt.encode_example` per row of `attempts_batch`, IN THE SAME ROW ORDER --
    load-bearing, since `forward_batch`'s `attempt_episode_row_idx`/`attempt_within_idx`
    index into this exact row order (`build_attempt_batch`'s own documented guarantee:
    "One entry per row of the INPUT attempts frame, same order"). Teacher-forced: `outcome_
    code` is this row's OBSERVED class (conditional imitation) -- ancestral generation
    (class sampled from p(c|u,q)) is handled by the generation passes, not here."""
    examples = []
    for row in attempts_batch.itertuples(index=False):
        statement = statements[int(row.problem_id)]
        prior_code = row.prior_code if isinstance(row.prior_code, str) else None
        examples.append(encode_example(statement, prior_code, row.code, tok, max_len=MAX_LEN,
                                       min_completion_tokens=MIN_COMPLETION_TOKENS,
                                       outcome_code=int(row.outcome_code)))
    return examples


def build_model(dims: dict, backbone_bundle: dict, seed: int, device: str,
                item_encoder: str = "mlp",
                sigma_parameterization: str = "direct") -> dict:
    """Fresh initialisation of every module (warm starts are applied afterwards by
    `load_warm_start_checkpoint` / `load_br4b_warm_start_checkpoint`).

    `item_encoder`: measured at 5 paired seeds with a row-identity join, `mlp - linear`
    on held-out items is **+0.0108 [+0.0017, +0.0183]** -- CI-clear of zero, about 3x
    the single-seed noise floor, which meets the pre-registered rule for adopting the
    MLP. `"mlp"` is therefore the default; `"linear"` is kept as the comparison arm.

    Every trainable module is moved to `device` explicitly (the same convention as
    `fa_sim3_full_run.py`) -- `attach_lora_basis`/`build_second_basis` place
    their new params on `model`'s own device already (`LoRAHookedLinear.forward`'s own
    device/dtype coercion is a robustness net, not a substitute for building on the right
    device to begin with)."""
    if item_encoder not in ("linear", "mlp"):
        raise ValueError(f"unknown item_encoder {item_encoder!r}; expected 'linear' or 'mlp'")
    if sigma_parameterization not in ("direct", "precision_accumulating"):
        raise ValueError(
            f"unknown sigma_parameterization {sigma_parameterization!r}; "
            "expected 'direct' or 'precision_accumulating'"
        )
    if sigma_parameterization != "direct" and item_encoder != "mlp":
        raise ValueError("precision_accumulating is wired only for the MLP/GRU filter")
    seed_everything(seed)
    d = dims
    attempt_encoder = AttemptEncoder(d["codec_dim"], d["codec_dim"], d["attempt_proj_dim"],
                                     d["attempt_hidden"], d["summary_dim"]).to(device)
    if item_encoder == "linear":
        filt, filt_forward = build_filter("gru", codec_dim=d["codec_dim"], episode_hidden=d["episode_hidden"],
                                          filter_hidden=d["filter_hidden"], latent_dim=d["latent_dim"],
                                          prev_feat_dim=d["summary_dim"], **FACTORY_KW)
        outcome_head = OutcomeHead(codec_dim=d["codec_dim"], latent_dim=d["latent_dim"],
                                   hidden=d["attempt_hidden"], n_classes=N_OUTCOME,
                                   prefix_dim=d["attempt_hidden"]).to(device)
        y_ever_head = OutcomeHead(codec_dim=d["codec_dim"], latent_dim=d["latent_dim"],
                                  hidden=d["attempt_hidden"], n_classes=1).to(device)
    else:
        filt = S1FilterV2ItemMLP(codec_dim=d["codec_dim"], episode_hidden=d["episode_hidden"],
                                 filter_hidden=d["filter_hidden"], latent_dim=d["latent_dim"],
                                 prev_feat_dim=d["summary_dim"],
                                 sigma_parameterization=sigma_parameterization)
        filt_forward = lambda ic, pf, tm, vm: filt(ic, pf)  # noqa: E731 -- matches build_filter's own convention
        outcome_head = OutcomeHeadItemMLP(codec_dim=d["codec_dim"], latent_dim=d["latent_dim"],
                                          hidden=d["attempt_hidden"], n_classes=N_OUTCOME,
                                          prefix_dim=d["attempt_hidden"]).to(device)
        y_ever_head = OutcomeHeadItemMLP(codec_dim=d["codec_dim"], latent_dim=d["latent_dim"],
                                         hidden=d["attempt_hidden"], n_classes=1).to(device)
    filt.to(device)

    model = backbone_bundle["model"]
    # Gradient checkpointing: an 8B-param frozen backbone leaves too little headroom on a
    # 24GB GPU to retain every layer's activations for backward without it (a measured
    # OOM on the FA simulator) -- load_backbone's own model.eval() call does not enable
    # it, so every caller must do so itself. Guarded by hasattr because the smoke test's
    # stub backbone (CausalStubLM) is a plain nn.Module with no HF checkpointing API and
    # no OOM risk at its tiny scale.
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
        model.config.use_cache = False
    bases, context = attach_lora_basis(model, m=16, r=8, seed=seed + 1)
    hyper = Hypernetwork(u_dim=d["latent_dim"], m=16, kind="mlp", seed=seed + 2).to(device)
    global_bases = build_second_basis(model, seed=seed + 3)               # global arm: built, not trained here
    global_mixing = GlobalMixingVector(m=16, seed=seed + 4).to(device)    # global arm: built, not trained here
    teacher = TeacherPosterior(codec_dim=d["codec_dim"], latent_dim=d["latent_dim"],
                               n_classes=N_OUTCOME, hidden=max(32, d["episode_hidden"] // 2)).to(device)

    return {
        "attempt_encoder": attempt_encoder, "filt": filt, "filt_forward": filt_forward,
        "outcome_head": outcome_head, "y_ever_head": y_ever_head, "hyper": hyper,
        "bases": bases, "context": context, "global_bases": global_bases,
        "global_mixing": global_mixing, "model": model, "tok": backbone_bundle["tokenizer"],
        "item_encoder": item_encoder,
        "sigma_parameterization": sigma_parameterization,
        "teacher": teacher,
    }


def load_warm_start_checkpoint(m: dict, checkpoint_path: pathlib.Path, dims: dict,
                               item_encoder: str, sigma_parameterization: str,
                               learner_split_sha256: str, problem_codec_sha256: str) -> dict:
    """Load only the belief stack from a standalone classifier checkpoint (`cw_sigma`).

    The backbone/LoRA/hypernetwork remain fresh by design.  All identity-bearing
    provenance is checked before any state is applied so a wrong codec, sampler, or
    architecture cannot silently become a warm start.
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    arch = ckpt.get("architecture") or {}
    for key, expected in dims.items():
        assert arch.get(key) == expected, (
            f"warm-start architecture mismatch for {key}: {arch.get(key)!r} != {expected!r}"
        )
    assert arch.get("body") == "gru", f"warm-start body mismatch: {arch.get('body')!r}"
    assert arch.get("item_encoder") == item_encoder, (
        f"warm-start item_encoder mismatch: {arch.get('item_encoder')!r} != {item_encoder!r}"
    )
    assert ckpt.get("sampler_semantics") == CURRENT_SAMPLER_SEMANTICS, (
        "warm-start sampler semantics mismatch"
    )
    assert ckpt.get("learner_split_sha256") == learner_split_sha256, (
        "warm-start learner split hash mismatch"
    )
    # The classifier records the committed transform hash rather than the dim-64 problem
    # codec hash used by this runner.  If a checkpoint carries the latter, enforce it;
    # otherwise the rank/dim architecture checks above are the binding codec guard.
    if ckpt.get("problem_codec_sha256") is not None:
        assert ckpt["problem_codec_sha256"] == problem_codec_sha256, (
            "warm-start problem codec hash mismatch"
        )
    assert ckpt.get("sigma_parameterization", sigma_parameterization) == sigma_parameterization, (
        "warm-start sigma parameterization mismatch"
    )
    required = {
        "attempt_encoder": "attempt_encoder_state_dict",
        "filt": "filter_state_dict",
        "outcome_head": "outcome_head_state_dict",
        "y_ever_head": "y_ever_head_state_dict",
    }
    for module_name, state_key in required.items():
        assert state_key in ckpt, f"warm-start checkpoint missing {state_key}"
        m[module_name].load_state_dict(ckpt[state_key], strict=True)
    return ckpt


def load_br4b_warm_start_checkpoint(m: dict, checkpoint_path: pathlib.Path, dims: dict,
                                    learner_split_sha256: str, problem_codec_sha256: str) -> dict:
    """Warm start from the CodeWorkout classifier: keep its encoder/filter, use direct sigma.

    The classifier checkpoint's precision head is deliberately not copied into the direct
    filter.  Common AttemptEncoder/GRU/μ-head parameters are loaded by name; the direct
    log-sigma head remains freshly initialised.
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    arch = ckpt.get("architecture") or {}
    for key in ("codec_dim", "latent_dim", "episode_hidden", "filter_hidden", "summary_dim"):
        if key in dims and arch.get(key) != dims[key]:
            raise AssertionError(f"classifier warm-start architecture mismatch for {key}: "
                                 f"{arch.get(key)!r} != {dims[key]!r}")
    if ckpt.get("sampler_semantics") != CURRENT_SAMPLER_SEMANTICS:
        raise AssertionError("classifier warm-start sampler semantics mismatch")
    if ckpt.get("learner_split_sha256") != learner_split_sha256:
        raise AssertionError("classifier warm-start learner split hash mismatch")
    if ckpt.get("problem_codec_sha256") not in (None, problem_codec_sha256):
        raise AssertionError("classifier warm-start problem codec hash mismatch")
    for module_name, state_key in (("attempt_encoder", "attempt_encoder_state_dict"),
                                   ("filt", "filter_state_dict")):
        if state_key not in ckpt:
            raise AssertionError(f"classifier warm-start checkpoint missing {state_key}")
        source = ckpt[state_key]
        target = m[module_name].state_dict()
        compatible = {k: v for k, v in source.items() if k in target and target[k].shape == v.shape}
        missing, unexpected = m[module_name].load_state_dict(compatible, strict=False)
        if module_name == "attempt_encoder" and missing:
            raise AssertionError(f"classifier AttemptEncoder warm start missing keys: {missing}")
        if module_name == "filt" and not any(k.startswith("knowledge_filter.mu_head") for k in compatible):
            raise AssertionError("classifier warm start did not load the filter μ head")
    return ckpt


def _length_bucketed_chunks(examples: list[dict], micro_batch_size: int) -> list[list[int]]:
    """Row indices (into `examples`/`attempts_batch`'s shared order) grouped into
    fixed-size chunks, sorted by token length first so each chunk pads to roughly its own
    rows' length rather than the whole batch's longest row -- free padding-waste
    reduction on top of the memory bound `micro_batch_size` itself provides."""
    order = sorted(range(len(examples)), key=lambda i: len(examples[i]["input_ids"]))
    return [order[i:i + micro_batch_size] for i in range(0, len(order), micro_batch_size)]


def build_joint_target_sets(attempts_batch: pd.DataFrame, episodes_batch: pd.DataFrame,
                            manifest: dict, attempt_batch: dict, prefix_states: torch.Tensor,
                            rng: np.random.Generator, device: str, min_targets: int = 10,
                            max_targets: int = 20) -> dict:
    """Construct one random causal prefix/target set per eligible learner.

    Each context is an episode and every target is a later episode's first attempt;
    target sets are padded only for batching.  The first-attempt row and its causal
    attempt-prefix state are selected by value from the already-built attempt batch.
    """
    ev = episodes_batch.sort_values(["learner", "start_timestamp"], kind="mergesort")
    at = attempts_batch.reset_index(drop=True)
    first_pos = at.groupby("episode_id", sort=False).head(1).index.to_numpy()
    first_pos_by_ep = {str(at.iloc[i].episode_id): int(i) for i in first_pos}
    ep_ids = list(attempt_batch["episode_ids"])
    ep_row = {str(e): i for i, e in enumerate(ep_ids)}
    rows = []
    for learner, g in ev.groupby("learner", sort=False):
        seq = g["episode_id"].tolist()
        eligible = [i for i in range(len(seq) - 1) if len(seq) - i - 1 >= min_targets]
        if not eligible:
            continue
        context_i = int(rng.choice(eligible))
        later = seq[context_i + 1:]
        n_target = int(rng.integers(min_targets, min(max_targets, len(later)) + 1))
        target_ids = rng.choice(np.asarray(later, dtype=object), size=n_target, replace=False).tolist()
        context_id = str(seq[context_i])
        if context_id not in ep_row or any(str(e) not in first_pos_by_ep for e in target_ids):
            continue
        rows.append((context_id, [str(e) for e in target_ids]))
    if not rows:
        return {"empty": True}
    n, t_max = len(rows), max(len(x[1]) for x in rows)
    codec_dim = manifest["problem_codec"].shape[1]
    target_codec = torch.zeros((n, t_max, codec_dim), dtype=torch.float32, device=device)
    target_class = torch.zeros((n, t_max), dtype=torch.long, device=device)
    target_mask = torch.zeros((n, t_max), dtype=torch.bool, device=device)
    target_prefix = torch.zeros((n, t_max, prefix_states.shape[-1]), dtype=prefix_states.dtype, device=device)
    context_ep_row = []
    first_attempt = at.iloc[first_pos]
    class_by_ep = {str(r.episode_id): int(r.outcome_code) for r in first_attempt.itertuples()}
    for i, (context_id, target_ids) in enumerate(rows):
        context_ep_row.append(ep_row[context_id])
        for j, target_id in enumerate(target_ids):
            row_i = first_pos_by_ep[target_id]
            problem_id = int(at.iloc[row_i].problem_id)
            target_codec[i, j] = torch.as_tensor(
                manifest["problem_codec"][manifest["pid_to_idx"][problem_id]], device=device)
            target_class[i, j] = class_by_ep[target_id]
            target_prefix[i, j] = prefix_states[attempt_batch["attempt_episode_row_idx"][row_i], 0]
            target_mask[i, j] = True
    return {"context_ep_row": torch.tensor(context_ep_row, dtype=torch.long, device=device),
            "target_codec": target_codec, "target_class": target_class,
            "target_mask": target_mask, "target_prefix": target_prefix}


def forward_joint_set(m: dict, attempts_batch: pd.DataFrame, episodes_batch: pd.DataFrame,
                      out: dict, manifest: dict, dims: dict, device: str,
                      target_rng: np.random.Generator, k: int = 8,
                      min_targets: int = 10, max_targets: int = 20) -> dict:
    """Joint-set (set-level) outcome and teacher-posterior KL for one learner batch."""
    ep_learner_idx, ep_t_idx = _distinct_episode_positions(
        episodes_batch, out["ep_batch"], out["attempt_batch"]["episode_ids"], device=device)
    sets = build_joint_target_sets(attempts_batch, episodes_batch, manifest,
                                   out["attempt_batch"], out["prefix_states"], target_rng, device,
                                   min_targets=min_targets, max_targets=max_targets)
    if sets.get("empty"):
        zero = out["l_outcome"] * 0.0
        return {"l_joint": zero, "kl_teacher": zero, "n_sets": 0, "target_sets": sets}
    mu_all, log_sigma_all = gather_belief_for_batch(m["filt_forward"], out["ep_batch"],
                                                    ep_learner_idx, ep_t_idx)
    mu = mu_all[sets["context_ep_row"]]
    log_sigma = log_sigma_all[sets["context_ep_row"]]
    context_codec_all = out["ep_batch"]["item_codec"][ep_learner_idx, ep_t_idx]
    context_codec = context_codec_all[sets["context_ep_row"]]
    teacher_mu, teacher_log_sigma = m["teacher"](
        sets["target_codec"], sets["target_class"], sets["target_mask"], context_codec=context_codec)
    u_k = reparameterize_from_log_sigma(mu, log_sigma, k=k).movedim(-2, 1)  # [B,K,L]
    teacher_u_k = reparameterize_from_log_sigma(teacher_mu, teacher_log_sigma, k=k).movedim(-2, 1)
    b, kk, t, d = sets["target_codec"].shape[0], u_k.shape[1], sets["target_codec"].shape[1], dims["latent_dim"]
    logits = []
    teacher_logits = []
    target_codec_flat = sets["target_codec"].reshape(b * t, -1)
    target_prefix_flat = sets["target_prefix"].reshape(b * t, -1)
    for j in range(kk):
        u_flat = u_k[:, j, :].unsqueeze(1).expand(-1, t, -1).reshape(b * t, d)
        logits.append(m["outcome_head"](target_codec_flat, u_flat, prefix=target_prefix_flat).reshape(b, t, -1))
        teacher_flat = teacher_u_k[:, j, :].unsqueeze(1).expand(-1, t, -1).reshape(b * t, d)
        teacher_logits.append(m["outcome_head"](target_codec_flat, teacher_flat,
                                                  prefix=target_prefix_flat).reshape(b, t, -1))
    logits_k = torch.stack(logits, dim=1)
    teacher_logits_k = torch.stack(teacher_logits, dim=1)
    l_student_per_set = joint_set_kmixture_nll(
        logits_k, sets["target_class"], sets["target_mask"])
    l_teacher_per_set = joint_set_kmixture_nll(
        teacher_logits_k, sets["target_class"], sets["target_mask"])
    # The shared primitive intentionally returns one NLL per target set.  The CW
    # trainer's outcome loss is a scalar, so reduce here before it reaches the
    # non-finite-loss guard and backward pass.
    l_student = l_student_per_set.mean()
    l_teacher = l_teacher_per_set.mean()
    l_joint = l_student + l_teacher
    kl_teacher = teacher_posterior_kl(
        teacher_mu, teacher_log_sigma, mu, log_sigma).mean()
    return {"l_joint": l_joint, "l_student": l_student, "l_teacher": l_teacher,
            "kl_teacher": kl_teacher, "n_sets": b, "target_sets": sets,
            "l_student_per_set": l_student_per_set, "l_teacher_per_set": l_teacher_per_set}


def forward_outcome(m: dict, attempts_batch: pd.DataFrame, episodes_batch: pd.DataFrame,
                    manifest: dict, dims: dict, device: str, free_bits: float) -> dict:
    """The cheap half of a training step: the classifier's `forward_batch`, with the
    `free_bits` keyword passed through. Small nets over codec vectors, no backbone -- safe to run over the FULL
    learner-batch regardless of how many attempts it contains; the backbone-heavy
    production forward is what needs microbatching, not this."""
    return forward_batch(m["attempt_encoder"], m["filt_forward"], m["outcome_head"], m["y_ever_head"],
                         attempts_batch, episodes_batch, manifest["code_of_hash"], manifest["diff_of_hash"],
                         manifest["problem_codec"], manifest["pid_to_idx"], device=device,
                         latent_dim=dims["latent_dim"], codec_dim=dims["codec_dim"], free_bits=free_bits)


def production_backward_microbatched(m: dict, attempts_batch: pd.DataFrame, manifest: dict,
                                     filter_batch: dict, ep_learner_idx: torch.Tensor,
                                     ep_t_idx: torch.Tensor, attempt_episode_row_idx: torch.Tensor,
                                     device: str, k: int, micro_batch_size: int, scale: float,
                                     row_chunk_size: int | None = None) -> float:
    """Chunks the production forward+backward by `micro_batch_size` attempts
    (length-bucketed), so the backbone never sees more than `micro_batch_size` rows in
    one forward regardless of how many attempts the outer learner-batch contains
    (measured: median 785 / p95 1,348 / max 2,777 attempts at the default 4-learner
    batch -- unbounded without chunking).

    The belief `u_ep` is computed ONCE (one filter forward, cheap) and sliced per chunk;
    each chunk backwards `scale * chunk_sum_nll / total_valid_rows`, so summing the
    chunks' `.backward()` calls accumulates to EXACTLY `scale` times the gradient
    `production_loss` on the unchunked batch would have produced -- no separate
    `/n_chunks` averaging needed (unlike FA-SIM3's fixed-size micro-batch convention;
    this one is exact under uneven chunk sizes because it is derived from the real
    valid-row counts). Returns the UNSCALED `l_prod` (what `production_loss` would report
    on the full batch) for logging, independent of whatever `scale` (e.g. `lambda_A`) was
    applied to the actual gradient.

    `retain_graph=True` on every chunk but the last: all chunks share the same upstream
    filter/hypernetwork graph through `u_ep`, which the caller's OWN separate `l_outcome`
    backward never touches (`production_forward_for_batch`'s design: a second, independent
    `filt_forward` call, not a shared graph with the outcome path).

    `row_chunk_size` (the same bound as on the eval path): threaded straight to
    `production_nll_per_row` -- bounds the peak memory
    of ITS OWN cross-entropy call (gemma4-e4b's huge vocab makes the full-sequence logits
    tensor the dominant term, independent of `micro_batch_size`) without changing the
    number of backbone forward passes this function makes. Default `None` computes the
    cross-entropy in one call."""
    examples = build_cw5_examples(attempts_batch, manifest["statements"], m["tok"])
    total_valid = sum(1 for e in examples if e["production_valid"])
    if total_valid == 0:
        return 0.0

    mu_sel, log_sigma_sel = gather_belief_for_batch(m["filt_forward"], filter_batch, ep_learner_idx, ep_t_idx)
    u_ep = reparameterize_from_log_sigma(mu_sel, log_sigma_sel, k=k)[:, 0, :]
    # Why u_ep is detached: with `retain_graph=True` on every chunk but the last, the
    # autograd engine keeps every saved tensor reachable from that backward call alive,
    # regardless of Python-level GC (on the real backbone this ran out of memory on a
    # 48GB card at step 1, even with an explicit del + gc.collect() + empty_cache per
    # chunk). Every chunk-but-the-last's call therefore keeps THAT CHUNK'S OWN
    # backbone-forward activations alive, not just the shared u_ep/filter ancestor it
    # exists to protect -- accumulating without bound across the ~100-350
    # chunks/learner-batch the real corpus produces (invisible at the smoke test's 1-2
    # chunk scale). So u_ep is detached into a fresh LEAF (`u_ep_leaf`) with no shared
    # history before the loop. Each chunk's
    # backward now targets a leaf with nothing upstream to retain, so the DEFAULT
    # retain_graph=False frees that chunk's entire backbone graph immediately after its
    # own backward -- gradient accumulates into `u_ep_leaf.grad` via autograd's normal
    # per-.backward()-call accumulation semantics, exactly like grad-accumulation across
    # minibatches anywhere else in this codebase. ONE final, cheap backward
    # (`u_ep.backward(gradient=u_ep_leaf.grad)`) re-attaches the accumulated gradient to
    # the real (small, backbone-free) filter/hypernetwork graph -- mathematically exact
    # (a chain-rule identity: u_ep_leaf has the same VALUES as u_ep, so the per-chunk
    # Jacobians are identical; this is the standard detach-and-reattach technique for
    # gradient-accumulation-with-one-shared-cheap-root, not an approximation).
    # `attempt_episode_row_idx` is a plain LongTensor (episode positions per attempt row,
    # no grad) -- safe to slice freely. Each chunk indexes `u_ep_leaf` DIRECTLY (one
    # index op per chunk, in the loop) rather than through a pre-sliced
    # `u_attempt_full = u_ep_leaf[attempt_episode_row_idx]` computed once outside the
    # loop: that intermediate would itself be a SHARED graph node every chunk's backward
    # passes through, so the first chunk's (default retain_graph=False) backward would
    # free it and the second chunk's backward would hit "Trying to backward through the
    # graph a second time" (the smoke test exercises this).
    u_ep_leaf = u_ep.detach().clone().requires_grad_(True)

    chunks = _length_bucketed_chunks(examples, micro_batch_size)
    total_loss = 0.0
    for idx in chunks:
        chunk_examples = [examples[j] for j in idx]
        encoded = collate(chunk_examples, pad_id=m["tok"].pad_token_id, device=device)
        chunk_idx_t = torch.tensor(idx, dtype=torch.long, device=device)
        u_chunk = u_ep_leaf[attempt_episode_row_idx[chunk_idx_t]]
        # The dominant memory term is gemma4-e4b's full-sequence, full-vocab lm_head
        # projection. Every row's prompt tokens carry label=-100 (masked
        # out of production_nll_per_row already), so restricting the projection to
        # [offset:] where offset = this chunk's OWN minimum prompt_end_index is an
        # IDENTITY for the loss (every row's completion starts at or after its own
        # prompt_end_index, which is >= offset by construction) -- never an
        # approximation. `production_nll_per_row` needs the matching label slice.
        offset = int(encoded["prompt_end_index"].min().item())
        seq_len = encoded["input_ids"].shape[1]
        out = forward_personalised(m["model"], m["hyper"], m["context"], u_chunk,
                                   encoded["input_ids"], encoded["attention_mask"],
                                   encoded["prompt_end_index"], logits_to_keep=seq_len - offset)
        per_row, row_has_any = production_nll_per_row(out["lm_logits"], encoded["labels"][:, offset:],
                                                       encoded["production_valid"],
                                                       row_chunk_size=row_chunk_size)
        chunk_sum = (per_row * row_has_any).sum()
        chunk_mean = chunk_sum / total_valid
        scaled = scale * chunk_mean
        if torch.is_tensor(scaled) and scaled.requires_grad:
            scaled.backward()  # default retain_graph=False -- this chunk's graph freed now
        total_loss += float(chunk_mean.detach().item())
        del out, per_row, row_has_any, chunk_sum, chunk_mean, scaled, encoded, u_chunk, chunk_idx_t
        if device == "cuda":
            torch.cuda.empty_cache()
    if u_ep_leaf.grad is not None and u_ep.requires_grad:
        u_ep.backward(gradient=u_ep_leaf.grad)
    return total_loss


def run_lambda_a_probe(m: dict, attempts_batch: pd.DataFrame, episodes_batch: pd.DataFrame,
                       manifest: dict, dims: dict, device: str, optimizer,
                       micro_batch_size: int, lambda_a_grid: tuple,
                       probe_log: dict | None = None) -> float:
    """Gradient-norm-balance `lambda_A` selection at the BELIEF JUNCTION
    (`grad_norm_at_belief_junction`): two backward passes on the SAME fixed
    batch, `filt.zero_grad()` between them, matching `pick_lambda_a`'s own established
    protocol -- microbatched on the production side, so the probe itself cannot
    OOM regardless of batch size."""
    optimizer.zero_grad()
    out1 = forward_outcome(m, attempts_batch, episodes_batch, manifest, dims, device, free_bits=0.05)
    out1["l_outcome"].backward()
    grad_norm_outcome = grad_norm_at_belief_junction(m["filt"])

    optimizer.zero_grad()
    out2 = forward_outcome(m, attempts_batch, episodes_batch, manifest, dims, device, free_bits=0.05)
    ep_learner_idx, ep_t_idx = _distinct_episode_positions(
        episodes_batch, out2["ep_batch"], out2["attempt_batch"]["episode_ids"], device=device)
    production_backward_microbatched(
        m, attempts_batch, manifest, out2["ep_batch"], ep_learner_idx, ep_t_idx,
        out2["attempt_batch"]["attempt_episode_row_idx"], device, k=1,
        micro_batch_size=micro_batch_size, scale=1.0)
    grad_norm_prod = grad_norm_at_belief_junction(m["filt"])

    optimizer.zero_grad()
    lambda_a = pick_lambda_a(grad_norm_prod, grad_norm_outcome, grid=lambda_a_grid)
    if probe_log is not None:
        probe_log.update({
            "grad_norm_prod": float(grad_norm_prod),
            "grad_norm_outcome": float(grad_norm_outcome),
            "lambda_a": float(lambda_a),
        })
    print(f"[cw-sim] lambda_A re-selected: {lambda_a} "
         f"(grad_norm_prod={grad_norm_prod:.4g}, grad_norm_outcome={grad_norm_outcome:.4g})", flush=True)
    return lambda_a


def assert_first_lambda_probe_has_production_gradient(grad_norm_prod: float) -> None:
    """Refuse a lambda probe that is still measuring LoRA's structural zero floor."""
    if not np.isfinite(grad_norm_prod) or grad_norm_prod <= 0.0:
        raise RuntimeError(
            "[cw-sim] FIRST_LAMBDA_PROBE_CANARY failed: grad_norm_prod must be finite and > 0 "
            f"after the warm-start step, got {grad_norm_prod!r}; refusing to select lambda_A"
        )


def build_class_conditioned_examples(attempts_batch: pd.DataFrame, manifest: dict, tok) -> list[list[dict]]:
    """One list of `N_OUTCOME` `cw_alt.encode_example` dicts per row of `attempts_batch`
    (same row order as `build_cw5_examples`), varying ONLY `outcome_code` -- statement/
    prior_code/completion (the learner's own actual code) held fixed across all four.
    Needed for the K-marginalised joint (k,c) estimator (`joint_kc_mixture_nll`), which
    scores the completion's likelihood under EVERY class, not just the observed one --
    unlike `build_cw5_examples` (training), which is teacher-forced on the OBSERVED
    class only."""
    out = []
    for row in attempts_batch.itertuples(index=False):
        statement = manifest["statements"][int(row.problem_id)]
        prior_code = row.prior_code if isinstance(row.prior_code, str) else None
        out.append([
            encode_example(statement, prior_code, row.code, tok, max_len=MAX_LEN,
                           min_completion_tokens=MIN_COMPLETION_TOKENS, outcome_code=c)
            for c in range(N_OUTCOME)
        ])
    return out


def common_completion_budget_mask(examples_per_row: list[list[dict]]) -> tuple[np.ndarray, np.ndarray]:
    """Common completion budget across class variants (the same requirement as for
    FA-SIM3, the mathematics simulator). The joint (k,c) estimator makes this BINDING:
    `joint_kc_mixture_nll` assumes ONE `n_valid_tok` per row, shared across all C
    classes. The assumption can fail here
    because CW's class-conditioning phrase (`CLASS_SUFFIX_TEMPLATE`) is a DIFFERENT
    LENGTH per class ("compile error" vs "compiles, all tests passed") -- a row sitting
    near the `MAX_LEN` truncation boundary can have a different amount of room left for
    the completion depending on which class token was inserted. A prompt-length
    measurement found only 0.39% of rows truncate at all at `MAX_LEN=512`, so
    this is expected to be rare, not structural -- but it must be MEASURED and excluded,
    never assumed away.

    A row is kept iff every one of its C variants is `production_valid` AND all C share
    the identical non-masked completion token count. Returns `(keep_mask, n_valid_tok)`,
    both length `len(examples_per_row)` -- `n_valid_tok[i]` is only meaningful where
    `keep_mask[i]` is True."""
    n = len(examples_per_row)
    keep = np.zeros(n, dtype=bool)
    n_valid_tok = np.zeros(n, dtype=np.int64)
    for i, variants in enumerate(examples_per_row):
        if not all(v["production_valid"] for v in variants):
            continue
        counts = [sum(1 for label in v["labels"] if label != -100) for v in variants]
        if len(set(counts)) == 1 and counts[0] > 0:
            keep[i] = True
            n_valid_tok[i] = counts[0]
    return keep, n_valid_tok


def joint_kc_scores_for_rows(m: dict, attempts_batch: pd.DataFrame, manifest: dict,
                             u_k: torch.Tensor, item_codec_sel: torch.Tensor,
                             prefix_sel: torch.Tensor, device: str,
                             micro_batch_size: int, arm: str = "personalised",
                             row_chunk_size: int | None = None) -> dict:
    """Scores the K-marginalised joint (k,c) mixture NLL (`joint_kc_mixture_nll`) for
    every row of `attempts_batch`, given a caller-supplied `u_k` [N,K,latent_dim] --
    WHICH `u` a row sees is `score_arm_for_batch`'s job (the 4-arm harness substitutes
    different `u`/basis here); this function scores whatever it is handed, for exactly ONE arm per
    call. `item_codec_sel` [N,codec_dim] and `prefix_sel` [N,prefix_dim] are the SAME
    two inputs training's `OutcomeHead` call takes (`losses_for_cw_hierarchical_batch`'s
    `item_codec_attempt`/`prefix_attempt`) -- `prefix` is not optional, `OutcomeHead`
    raises if it is constructed with `prefix_dim` set (this model always does) and called
    without one. Rows failing `common_completion_budget_mask` are DROPPED and counted,
    never silently included with a wrong `n_valid_tok`.

    Microbatched by `micro_batch_size`, length-bucketed by each row's class-0 prompt
    length (a proxy for the row's own length -- the four per-row variants differ only by
    the class-phrase's few tokens) -- mirrors `production_backward_microbatched`'s own
    memory bound, but eval is MORE expensive per row than training's production forward:
    `N_OUTCOME` classes x K samples x one backbone forward each, vs training's one
    forward per row. `elapsed_s`/`n_forward_calls` are returned so a caller can measure
    real per-row eval cost and size a full run from a measurement -- this IS the
    dominant eval cost.

    `arm` (`ARMS`): only `"dw0"` changes behaviour HERE -- the production forward's LoRA
    mixing vector `c` is set to ZERO directly (`context.set(zeros)`, bypassing
    `hyper(u)` entirely) rather than routed through the hypernetwork, ablating the
    MECHANISM (personalisation), not the belief. Every other arm (`personalised`/
    `shuffled_u`/`u_mean`) runs the identical `forward_personalised` call -- WHICH `u_k`
    they were handed (own belief / a learner-history-matched derangement's belief / the
    posterior mean) is entirely the CALLER's decision, made before this function is
    invoked; this function's own logic never depends on arm identity beyond the dw0
    production-forward substitution. The outcome head's class probabilities use `u_k`
    for EVERY arm including `dw0` -- dw0 ablates production personalisation only.

    Returns per-KEPT-row arrays (`mixture_nll`, `mixture_nll_raw`, `true_class_nll`,
    `true_class_nll_per_k` [K, N_kept] -- the per-K-sample true-class NLL BEFORE
    K-averaging, for the uncertainty-survives gate (K-spread non-degenerate is
    `true_class_nll_per_k.std(axis=0)`), `true_class_top1`, `true_class_exact_match`,
    `p_class_meanK` [N_kept, N_OUTCOME]) plus `kept_row_idx` (positional index into the
    INPUT `attempts_batch`), `n_dropped_budget`, `elapsed_s`, `n_forward_calls`,
    `n_rows_scored`. The K=1/4/8/16 STABILITY curve is `k_stability_curve`'s job -- call this
    function once per K with a freshly re-sampled `u_k`; there is no single-call
    shortcut, since that curve is about K as a Monte-Carlo sample COUNT, a different
    axis from the spread AT one fixed K this function already reports."""
    assert arm in ARMS, f"unknown arm {arm!r}; expected one of {ARMS}"
    t0 = time.time()
    examples_per_row = build_class_conditioned_examples(attempts_batch, manifest, m["tok"])
    keep_mask, n_valid_tok_all = common_completion_budget_mask(examples_per_row)
    n_dropped = int((~keep_mask).sum())
    kept_idx = np.nonzero(keep_mask)[0]
    empty = {
        "mixture_nll": np.zeros(0), "mixture_nll_raw": np.zeros(0), "true_class_nll": np.zeros(0),
        "true_class_top1": np.zeros(0), "true_class_exact_match": np.zeros(0),
        "p_class_meanK": np.zeros((0, N_OUTCOME)), "true_class_nll_per_k": np.zeros((u_k.shape[1], 0)),
        "kept_row_idx": kept_idx, "n_dropped_budget": n_dropped,
        "elapsed_s": time.time() - t0, "n_forward_calls": 0, "n_rows_scored": 0,
    }
    if len(kept_idx) == 0:
        return empty

    u_k_kept = u_k[kept_idx]
    item_codec_kept = item_codec_sel[kept_idx]
    prefix_kept = prefix_sel[kept_idx]
    n_kept, k_eff, _ = u_k_kept.shape
    n_valid_tok_kept = torch.tensor(n_valid_tok_all[kept_idx], dtype=torch.float32)
    true_class_kept = attempts_batch["outcome_code"].to_numpy()[kept_idx]

    order = sorted(range(n_kept), key=lambda i: len(examples_per_row[kept_idx[i]][0]["input_ids"]))
    chunks = [order[i:i + micro_batch_size] for i in range(0, n_kept, micro_batch_size)]

    log_p_seq_given_c_k = torch.empty(N_OUTCOME, k_eff, n_kept)
    top1 = torch.zeros(n_kept)
    exact = torch.zeros(n_kept)
    n_forward_calls = 0

    with torch.no_grad():
        log_p_class_kc = torch.empty(N_OUTCOME, k_eff, n_kept)
        for j in range(k_eff):
            logits_j = m["outcome_head"](item_codec_kept, u_k_kept[:, j, :], prefix=prefix_kept)
            log_p_class_kc[:, j, :] = torch.log_softmax(logits_j, dim=-1).transpose(0, 1).cpu()

        for chunk in chunks:
            chunk_t = torch.tensor(chunk, dtype=torch.long, device=device)
            for c in range(N_OUTCOME):
                chunk_examples = [examples_per_row[kept_idx[i]][c] for i in chunk]
                encoded = collate(chunk_examples, pad_id=m["tok"].pad_token_id, device=device)
                # Same completion-span restriction as
                # `production_backward_microbatched`: eval's N_OUTCOME x K x chunks
                # forward volume makes `--micro-batch-size 1` impractically slow (~97k
                # forwards/arm), and this restriction keeps a real micro-batch size
                # within memory.
                offset = int(encoded["prompt_end_index"].min().item())
                seq_len = encoded["input_ids"].shape[1]
                labels_sliced = encoded["labels"][:, offset:]
                # Dedup the dw0 arm's K identical backbone forwards (~K x faster on that
                # arm): dw0's production forward sets the LoRA mixing vector to a
                # hardcoded ZERO (`context.set(zeros_c)`), completely independent of
                # `j`/`u_k_kept[..., j, :]` -- every one of the k_eff forwards below
                # would be, byte for byte, the SAME computation on the SAME inputs. Compute
                # it once at the first j and broadcast to every other j, rather than
                # re-running the backbone k_eff times for an answer that cannot change.
                # `personalised`/`shuffled_u`/`u_mean` are unaffected -- their forward
                # genuinely depends on `u_k_kept[..., j, :]` per j.
                dw0_log_p_seq_j0 = None
                for j in range(k_eff):
                    if arm == "dw0" and dw0_log_p_seq_j0 is not None:
                        log_p_seq_given_c_k[c, j, chunk] = dw0_log_p_seq_j0
                        continue
                    if arm == "dw0":
                        zeros_c = torch.zeros(len(chunk), m["hyper"].m, device=device)
                        m["context"].set(zeros_c)
                        lm_logits = m["model"](input_ids=encoded["input_ids"],
                                              attention_mask=encoded["attention_mask"],
                                              output_hidden_states=False,
                                              logits_to_keep=seq_len - offset).logits
                    else:
                        out = forward_personalised(m["model"], m["hyper"], m["context"], u_k_kept[chunk_t, j, :],
                                                   encoded["input_ids"], encoded["attention_mask"],
                                                   encoded["prompt_end_index"], logits_to_keep=seq_len - offset)
                        lm_logits = out["lm_logits"]
                    n_forward_calls += 1
                    log_p_seq, _n_tok, _has_any = production_log_likelihood_sum_per_row(
                        lm_logits, labels_sliced, encoded["production_valid"],
                        row_chunk_size=row_chunk_size)
                    # ONE .cpu() transfer per forward call, not one per row -- MPS (and
                    # CUDA) synchronise on every host<->device transfer, so a Python loop
                    # calling `.cpu()` per-row turns a 13ms forward into a multi-second
                    # one at this call volume (4 classes x K x chunks). Measured on the
                    # smoke test: per-row transfers took the personalised arm's eval to
                    # 195s/80 rows -- the dominant cost of this function, not a
                    # micro-optimisation.
                    log_p_seq_cpu = log_p_seq.cpu()
                    log_p_seq_given_c_k[c, j, chunk] = log_p_seq_cpu
                    if arm == "dw0":
                        dw0_log_p_seq_j0 = log_p_seq_cpu
                    if j == 0:
                        top1_c, exact_c, _ = production_topk_per_row(
                            lm_logits, labels_sliced, encoded["production_valid"])
                        top1_c_cpu, exact_c_cpu = top1_c.cpu(), exact_c.cpu()
                        for local_i, global_i in enumerate(chunk):
                            if true_class_kept[global_i] == c:
                                top1[global_i] = top1_c_cpu[local_i]
                                exact[global_i] = exact_c_cpu[local_i]
                    # On the real backbone, memory grew slowly (~20GB->22GB) until a CUDA
                    # OOM after ~140 min -- invisible at the smoke test's handful-of-rows
                    # scale. This loop runs up to chunks x N_OUTCOME x k_eff forward passes
                    # (thousands at real scale) with no explicit release between them --
                    # under no_grad() nothing should be graph-retained, so this is read as
                    # allocator fragmentation from many differently-shaped logits tensors
                    # (PyTorch's own OOM message recommends exactly this), not a Python
                    # reference leak. Explicit del + a periodic empty_cache is cheap
                    # insurance either way.
                    del lm_logits
                    if arm != "dw0":
                        del out
                if device == "cuda":
                    torch.cuda.empty_cache()

    mixture_nll_raw, mixture_nll = joint_kc_mixture_nll(log_p_class_kc, log_p_seq_given_c_k, n_valid_tok_kept)

    true_class_t = torch.tensor(true_class_kept, dtype=torch.long)
    true_log_p_seq_k = torch.stack(
        [log_p_seq_given_c_k[true_class_t[i], :, i] for i in range(n_kept)], dim=1)  # [K, N_kept]
    true_log_p_seq = torch.logsumexp(true_log_p_seq_k, dim=0) - float(np.log(k_eff))
    true_class_nll = (-true_log_p_seq / n_valid_tok_kept.clamp(min=1)).numpy()
    true_class_nll_per_k = (-true_log_p_seq_k / n_valid_tok_kept.clamp(min=1)).numpy()

    return {
        "mixture_nll": mixture_nll.numpy(), "mixture_nll_raw": mixture_nll_raw.numpy(),
        "true_class_nll": true_class_nll, "true_class_top1": top1.numpy(),
        "true_class_exact_match": exact.numpy(),
        "p_class_meanK": log_p_class_kc.exp().mean(dim=1).transpose(0, 1).numpy(),
        "true_class_nll_per_k": true_class_nll_per_k,
        "kept_row_idx": kept_idx, "n_dropped_budget": n_dropped,
        "elapsed_s": time.time() - t0, "n_forward_calls": n_forward_calls, "n_rows_scored": n_kept,
    }


def _subset_scores(scores: dict, mask: np.ndarray) -> dict:
    """Restricts a `joint_kc_scores_for_rows`-shaped dict to `mask` (boolean, one entry
    per currently-kept row) -- extracted as its own function so the derangement-exclusion
    logic (`score_arm_for_batch`'s `shuffled_u` branch) is unit-testable without a real
    model/manifest. `true_class_nll_per_k` is [K, N] --
    masked on its SECOND axis, every other per-row array on its first (only) axis."""
    out = dict(scores)
    for key in ("mixture_nll", "mixture_nll_raw", "true_class_nll",
               "true_class_top1", "true_class_exact_match", "p_class_meanK"):
        out[key] = scores[key][mask]
    out["true_class_nll_per_k"] = scores["true_class_nll_per_k"][:, mask]
    out["kept_row_idx"] = scores["kept_row_idx"][mask]
    return out


def score_arm_for_batch(m: dict, attempts_batch: pd.DataFrame, episodes_batch: pd.DataFrame,
                        manifest: dict, dims: dict, device: str, k: int, micro_batch_size: int,
                        arm: str, rng: np.random.Generator | None = None,
                        eps: torch.Tensor | None = None,
                        row_chunk_size: int | None = None) -> dict:
    """One arm's K-marginalised joint (k,c) scores for a learner-batch's
    worth of attempts (`eval` mode -- `torch.no_grad()` throughout). `arm` in `ARMS`
    decides WHICH `u` a row is scored under (this function's own job, unlike
    `joint_kc_scores_for_rows`, which only decides HOW a given `u` is scored):

      - `personalised`: `u` sampled from the row's OWN belief (`gather_belief_for_batch`
        at the row's own (learner, t) position).
      - `shuffled_u`: `u` sampled from a LEARNER-HISTORY-MATCHED derangement
        -- another learner's belief, same episode-history-length bin, never
        the same learner even if that learner owns multiple rows in the bin. Episodes
        whose bin has fewer than 2 distinct learners CANNOT be deranged
        (`learner_history_matched_derangement`'s own `deranged_mask`) and are excluded
        from this arm's returned rows entirely -- never silently scored against
        themselves and passed off as a control.
      - `u_mean`: the posterior MEAN, no sampling (`k_eff` forced to 1).
      - `dw0`: the SAME real sampled `u` as `personalised` for the outcome head; the
        production forward's LoRA mixing is ablated (see `joint_kc_scores_for_rows`'s
        own `arm` handling) -- a mechanism ablation, not a belief ablation.

    Returns `joint_kc_scores_for_rows`'s own dict, with `kept_row_idx` further
    intersected against derangement success for `shuffled_u` (both budget-valid AND
    successfully deranged), plus `mu`/`log_sigma` (the row's OWN filter output, `[N_kept,
    latent_dim]` each -- always the OWN belief regardless of arm, since this is what the
    uncertainty-survives gate reads, not whichever belief the arm happened to sample
    from) and `n_dropped_derangement` (0 for every arm except `shuffled_u`).

    `eps` (common random numbers, as for FA-SIM3): optional pre-drawn
    standard-normal noise, shape `[n_episodes_in_batch, >=k, latent_dim]`, aligned to
    `out["ep_batch"]`'s own row order (asserted below, not assumed: a shape match is not
    a construct-validity guarantee, but it is the cheapest check that catches a caller
    passing the wrong batch's eps). Reused
    IDENTICALLY across every arm this function is called for on the same eval batch, so
    `personalised` and `shuffled_u` (and `dw0`, which shares `personalised`'s own belief)
    draw the SAME noise quantile -- the only thing that should differ between them is
    WHOSE belief `(mu, log_sigma)` is sampled, never the sampling noise itself. `eps=None`
    (default) draws independent noise per call; `u_mean` never
    samples (`k_eff` forced to 1, deterministic mean) so `eps` is irrelevant there."""
    assert arm in ARMS, f"unknown arm {arm!r}; expected one of {ARMS}"
    with torch.no_grad():
        out = forward_outcome(m, attempts_batch, episodes_batch, manifest, dims, device, free_bits=0.05)
        ep_learner_idx, ep_t_idx = _distinct_episode_positions(
            episodes_batch, out["ep_batch"], out["attempt_batch"]["episode_ids"], device=device)
        row_idx = out["attempt_batch"]["attempt_episode_row_idx"]

        mu_sel, log_sigma_sel = gather_belief_for_batch(m["filt_forward"], out["ep_batch"], ep_learner_idx, ep_t_idx)
        if eps is not None:
            assert eps.shape[0] == mu_sel.shape[0], (
                f"CRN eps has {eps.shape[0]} episode rows, this batch has {mu_sel.shape[0]} -- "
                "eps was drawn for a different batch (a shape match alone is not "
                "sufficient, but a mismatch here is a definite caller bug)")
            assert eps.shape[1] >= k, f"CRN eps only carries K={eps.shape[1]}, requested k={k}"
        item_codec_sel_ep = gather_item_codec_for_batch(out["ep_batch"], ep_learner_idx, ep_t_idx)

        sample_mu, sample_log_sigma = mu_sel, log_sigma_sel
        deranged_per_attempt = None
        if arm == "shuffled_u":
            learner_ids_np = ep_learner_idx.cpu().numpy()
            bin_ids_np = ep_t_idx.cpu().numpy()
            perm, deranged_mask = learner_history_matched_derangement(learner_ids_np, bin_ids_np, rng=rng)
            assert_learner_matched_derangement(perm, learner_ids_np, deranged_mask)
            perm_t = torch.as_tensor(perm, dtype=torch.long, device=device)
            bl_idx, bt_idx = ep_learner_idx[perm_t], ep_t_idx[perm_t]
            sample_mu, sample_log_sigma = gather_belief_for_batch(m["filt_forward"], out["ep_batch"], bl_idx, bt_idx)
            deranged_per_attempt = deranged_mask[row_idx.cpu().numpy()]

        k_eff = 1 if arm == "u_mean" else k
        # device-matched, not the CPU generator's own device: reparameterize_from_log_sigma
        # needs eps on the SAME device as mu/log_sigma -- the CPU-drawn shared eps
        # otherwise raises the first time this runs against a real (non-CPU) backbone.
        # Cheap (one small tensor .to() per call).
        eps_k = eps[:, :k_eff, :].to(sample_mu.device) if eps is not None else None
        u_ep_k = sample_mu.unsqueeze(1) if arm == "u_mean" else \
            reparameterize_from_log_sigma(sample_mu, sample_log_sigma, k=k_eff, eps=eps_k)

        u_k = u_ep_k[row_idx]
        item_codec_sel = item_codec_sel_ep[row_idx]
        prefix_sel = gather_prefix_for_attempts(out["prefix_states"], row_idx,
                                                out["attempt_batch"]["attempt_within_idx"])
        own_mu_sel, own_log_sigma_sel = mu_sel[row_idx], log_sigma_sel[row_idx]

        scores = joint_kc_scores_for_rows(m, attempts_batch, manifest, u_k, item_codec_sel, prefix_sel,
                                          device, micro_batch_size, arm=arm, row_chunk_size=row_chunk_size)

    kept = scores["kept_row_idx"]
    n_dropped_derangement = 0
    if arm == "shuffled_u" and len(kept) > 0:
        still_deranged = deranged_per_attempt[kept]
        n_dropped_derangement = int((~still_deranged).sum())
        scores = _subset_scores(scores, still_deranged)
        kept = scores["kept_row_idx"]

    scores["mu"] = own_mu_sel[kept].cpu().numpy()
    scores["log_sigma"] = own_log_sigma_sel[kept].cpu().numpy()
    scores["n_dropped_derangement"] = n_dropped_derangement
    scores["arm"] = arm
    return scores


def uncertainty_survives_gate(scores: dict) -> dict:
    """Pilot check that belief uncertainty reaches the backbone (the same check as on
    the FA simulator). Two independent conditions, both reported, neither inferred from
    the other:

    (i) the FILTER's own `log_sigma` output (`scores["log_sigma"]`, the row's OWN belief
    regardless of which arm was scored) never saturates either clamp bound
    (`LOG_SIGMA_MIN=-5.0`, `LOG_SIGMA_MAX=2.0`) -- a variational collapse (every row
    pinned at one bound) is the failure mode the fixed epoch sampler exists to prevent.

    (ii) the PRODUCTION-side K-spread (`scores["true_class_nll_per_k"].std(axis=0)`,
    nats/token) is non-degenerate on a real fraction of rows -- if sampling `u` from the
    SAME belief always produces the SAME production NLL, uncertainty never reaches the
    backbone even if the belief itself is non-degenerate ("uncertainty survives through
    the whole backbone, not just the belief")."""
    log_sigma = scores["log_sigma"]
    spread = scores["true_class_nll_per_k"].std(axis=0)
    log_sigma_off_bounds = bool(np.all((log_sigma > LOG_SIGMA_MIN) & (log_sigma < LOG_SIGMA_MAX)))
    spread_nonzero_frac = float((spread > 1e-6).mean()) if spread.size else 0.0
    return {
        "log_sigma_min_observed": float(log_sigma.min()) if log_sigma.size else float("nan"),
        "log_sigma_max_observed": float(log_sigma.max()) if log_sigma.size else float("nan"),
        "log_sigma_off_both_clamp_bounds": log_sigma_off_bounds,
        "k_spread_mean": float(spread.mean()) if spread.size else float("nan"),
        "k_spread_nonzero_frac": spread_nonzero_frac,
        "n_rows": int(log_sigma.shape[0]) if log_sigma.size else 0,
        "passed": log_sigma_off_bounds and spread_nonzero_frac > 0.99,
    }


def k_stability_curve(m: dict, attempts_batch: pd.DataFrame, episodes_batch: pd.DataFrame,
                      manifest: dict, dims: dict, device: str, micro_batch_size: int,
                      arm: str, ks: tuple = (1, 4, 8, 16),
                      rng: np.random.Generator | None = None,
                      eps: torch.Tensor | None = None, eps_seed: int = 0,
                      latent_dim: int = 0, resume_state: dict | None = None,
                      on_k_done=None) -> list[dict]:
    """K=1/4/8/16 stability curve: `score_arm_for_batch` re-run once per K (a FRESH
    re-sample each time -- there is no single-call shortcut, K only means anything as a
    Monte-Carlo sample count). Reports `mixture_nll.mean()`'s own convergence, not just
    its value at the largest K, so a caller can see WHERE it stabilises rather than
    trusting the last point alone.

    `eps` (common random numbers): the SAME shared eps drawn for the 4-arm scoring pass, sliced per K
    when `k <= eps.shape[1]`; for `k > eps.shape[1]` (the curve's own K can exceed the
    main pass's `--eval-k`, e.g. K=16 vs `--eval-k=8`) a FRESH larger draw is made from
    `eps_seed` -- mirrors `fa_sim3_full_run.py`'s own exact fallback (`eps_sweep = ...
    if k_sweep <= args.k_eval else torch.randn(...)`). `latent_dim` is only needed for
    that fallback branch's shape; unused when every `k` fits inside `eps`.

    `resume_state`/`on_k_done` (resumable unattended runs): if `resume_state` already
    has an entry for a given `k` (keyed by `str(k)`), that K is SKIPPED and its cached
    result reused rather than rescored -- eval resumability at K granularity, matching
    the 4-arm loop's own per-arm resumability. `on_k_done(k, result)` is called
    immediately after each K completes (fresh OR skipped), so a caller can persist
    progress incrementally rather than only at the end of the whole curve."""
    curve = []
    for k in ks:
        if resume_state is not None and str(k) in resume_state:
            result = resume_state[str(k)]
            print(f"[cw-sim] k_stability_curve: K={k} already scored (resumed), skipping", flush=True)
        else:
            eps_k = None
            if eps is not None:
                eps_k = eps[:, :k, :] if k <= eps.shape[1] else torch.randn(
                    eps.shape[0], k, latent_dim, generator=torch.Generator(device="cpu").manual_seed(eps_seed))
            scores = score_arm_for_batch(m, attempts_batch, episodes_batch, manifest, dims, device,
                                         k, micro_batch_size, arm, rng=rng, eps=eps_k)
            result = {
                "k": k, "n_rows_scored": scores["n_rows_scored"],
                "mixture_nll_mean": float(scores["mixture_nll"].mean()) if scores["n_rows_scored"] else float("nan"),
                "mixture_nll_std": float(scores["mixture_nll"].std()) if scores["n_rows_scored"] else float("nan"),
            }
        curve.append(result)
        if on_k_done is not None:
            on_k_done(k, result)
    return curve


def write_predictions_npz(scores_by_arm: dict[str, dict], attempts_batch: pd.DataFrame,
                          out_path: pathlib.Path) -> None:
    """Per-arm per-row predictions, kept so that any reported number can be recomputed.
    Row identity carried BY VALUE (`learner`/`problem_id`/`code_state_id`), never a bare
    positional index alone: metadata is carried from source tables, never re-derived
    through index arithmetic (a positional `problem_id` join is exactly the kind of
    error that silently corrupts a held-item slice). Different arms can have
    different `kept_row_idx` sets (budget/derangement exclusions vary), so identity is
    stored PER ARM, never assumed shared."""
    payload = {}
    for arm, scores in scores_by_arm.items():
        kept = scores["kept_row_idx"]
        rows = attempts_batch.iloc[kept]
        prefix = f"{arm}__"
        payload[prefix + "learner"] = rows["learner"].to_numpy()
        payload[prefix + "problem_id"] = rows["problem_id"].to_numpy()
        payload[prefix + "code_state_id"] = rows["code_state_id"].to_numpy()
        payload[prefix + "outcome_code"] = rows["outcome_code"].to_numpy()
        payload[prefix + "mixture_nll"] = scores["mixture_nll"]
        payload[prefix + "mixture_nll_raw"] = scores["mixture_nll_raw"]
        payload[prefix + "true_class_nll"] = scores["true_class_nll"]
        payload[prefix + "true_class_top1"] = scores["true_class_top1"]
        payload[prefix + "true_class_exact_match"] = scores["true_class_exact_match"]
        payload[prefix + "p_class_meanK"] = scores["p_class_meanK"]  # [N, N_OUTCOME]
        payload[prefix + "mu"] = scores["mu"]
        payload[prefix + "log_sigma"] = scores["log_sigma"]
        payload[prefix + "n_dropped_budget"] = np.array(scores["n_dropped_budget"])
        payload[prefix + "n_dropped_derangement"] = np.array(scores["n_dropped_derangement"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **payload)


def write_generations_jsonl(attempts_batch: pd.DataFrame, manifest: dict,
                            kept_row_idx: np.ndarray, out_path: pathlib.Path) -> None:
    """Write the raw licensed text for each scored row to a local file that is never
    uploaded: the problem statement, prior-attempt code and the learner's own actual
    completion are DataShop-licensed text (the same restriction as FA's own
    `generations.jsonl`); only derived artifacts are uploaded.

    Scope: persists the TRUE text per scored row (statement, prior_code, actual
    completion) for human inspection alongside the persisted predictions.
    Teacher-forced ARGMAX-PREDICTED text (decoded model output, beyond the aggregate
    top1/exact-match scalars `joint_kc_scores_for_rows` reports) is NOT captured here;
    that needs its own decode pass (see the ancestral generation passes below)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for i in kept_row_idx:
            row = attempts_batch.iloc[int(i)]
            record = {
                "learner": row["learner"], "problem_id": int(row["problem_id"]),
                "code_state_id": row["code_state_id"], "outcome_code": int(row["outcome_code"]),
                "statement": manifest["statements"][int(row["problem_id"])],
                "prior_code": row["prior_code"] if isinstance(row["prior_code"], str) else None,
                "true_completion": row["code"],
            }
            f.write(json.dumps(record) + "\n")


def greedy_generate(model, tok, context, c_vec: torch.Tensor, prompt_ids: list[int],
                    max_new_tokens: int, device: str, on_token=None) -> str:
    """Greedy ancestral generation for one row, with a KV cache.

    A full forward over the WHOLE growing sequence every step would cost O(T) recompute
    at each of T steps, i.e. O(T^2) total attention/MLP work for a generation of length
    T, and calling `torch.cuda.empty_cache()` after every token would synchronise the
    device and clear the allocator cache on every step; together these dominate the
    cost of any generation-heavy run.

    This function enables `use_cache=True` and threads `past_key_values` between steps,
    so each step after the first costs O(1) attention work (one new token against the
    cached K/V), not O(T); `empty_cache()` is called at most once, at the very end of
    the whole call, not per token. `context.set(c_vec)` is called
    ONCE before the loop, not per step -- `hooked_backbone.py`'s `MixingContext`
    broadcast is per-ROW, not per-token (`c.shape[0]` stays the batch size regardless of
    how many new tokens a given forward call carries), so a fixed personalisation vector
    for this whole generated sequence needs setting only once, whether the forward
    receives the full prompt (step 0) or a single cached-continuation token (every step
    after).

    Single-row only: multi-row batching lives in `greedy_generate_batch`. This function
    keeps its single-row signature for `run_ancestral_generation_pass` (8 arm/mode calls
    per row) and for the resumability test, which asserts exact call order/count via a
    monkeypatched `greedy_generate`. The KV cache is the dominant saving (it turns
    O(T^2) into O(T)); batching amortises fixed per-call overhead on top of that.

    `use_cache=True` is tried first; any backbone that does not support it (the smoke
    test's stub `CausalStubLM`, which has no `past_key_values` parameter at all) falls
    back to full recompute every step via a `try/except TypeError` -- the same pattern
    as the `logits_to_keep` optimisation below, so the stub-based smoke test still
    exercises the WIRING without needing cache support itself; results are numerically
    identical either way, only memory/speed differ on a real backbone."""
    context.set(c_vec)
    ids = list(prompt_ids)
    eos_id = tok.eos_token_id
    past_key_values = None
    use_cache_supported = True
    for step in range(max_new_tokens):
        with torch.no_grad():
            if use_cache_supported and past_key_values is not None:
                # every step after the first: ONE new token, the cache carries history.
                input_ids = torch.tensor([[ids[-1]]], device=device)
                attention_mask = torch.ones((1, len(ids)), device=device, dtype=torch.long)
                try:
                    out = model(input_ids=input_ids, attention_mask=attention_mask,
                               past_key_values=past_key_values, use_cache=True)
                except TypeError:
                    use_cache_supported = False
                    past_key_values = None
            if not use_cache_supported or past_key_values is None:
                if not use_cache_supported:
                    # fallback path: full recompute every step, no cache.
                    input_ids = torch.tensor([ids], device=device)
                    attention_mask = torch.ones_like(input_ids)
                    try:
                        out = model(input_ids=input_ids, attention_mask=attention_mask, logits_to_keep=1)
                    except TypeError:
                        out = model(input_ids=input_ids, attention_mask=attention_mask)
                else:
                    # step 0 of a cache-capable backbone: full prompt forward, seeds the cache.
                    input_ids = torch.tensor([ids], device=device)
                    attention_mask = torch.ones_like(input_ids)
                    try:
                        out = model(input_ids=input_ids, attention_mask=attention_mask,
                                   use_cache=True, logits_to_keep=1)
                    except TypeError:
                        try:
                            out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
                        except TypeError:
                            use_cache_supported = False
                            out = model(input_ids=input_ids, attention_mask=attention_mask)
            if use_cache_supported:
                past_key_values = getattr(out, "past_key_values", None)
                if past_key_values is None:
                    use_cache_supported = False
            logits = out.logits
            next_id = int(logits[0, -1].argmax())
        ids.append(next_id)
        del out, logits
        if on_token is not None:
            on_token()
        if eos_id is not None and next_id == eos_id:
            break
    if device == "cuda":
        torch.cuda.empty_cache()
    gen_ids = ids[len(prompt_ids):]
    return tok.decode(gen_ids, skip_special_tokens=True).strip()


def greedy_generate_batch(model, tok, context, c_vecs: torch.Tensor, prompt_ids_list: list[list[int]],
                          max_new_tokens: int, device: str, on_token=None) -> list[str]:
    """Row-batched greedy generation with a shared KV cache. Batching is what makes the
    downstream generation panels affordable: the single-row rate is ~55s/generation, too
    slow for the val first-attempt panel or the TIKTOC-transfer eval.

    Batches `len(prompt_ids_list)` independent generations (each its own `c_vec` row --
    `context.c`'s `(batch, m)` shape broadcasts per-row) through ONE shared KV-cache
    decode, LEFT-padded so every row's true last
    prompt token sits at the same (final) sequence position regardless of its own prompt
    length -- the standard convention this makes correct: at step 0 every row's next-
    token logits are simply `logits[:, -1, :]`, no per-row length bookkeeping needed.
    `position_ids` are computed once from the prompt attention mask (`cumsum(-1) - 1`,
    clamped so padding gets a harmless placeholder position) and then simply increment
    by one per row per step thereafter -- `transformers`' own left-pad-then-generate
    convention, not invented here.

    Per-row EOS: once a row's own argmax hits `eos_id`, its LATER steps' outputs are
    still computed (attention is batch-independent across rows -- a finished row cannot
    influence any other row's continuation, and vice versa) but are DISCARDED, never
    appended to that row's own returned token list -- matching `greedy_generate`'s exact
    single-row truncation semantics (the eos token itself IS appended once, then that
    row stops accumulating; `skip_special_tokens=True` at decode strips it, exactly as
    the single-row function relies on). The loop as a whole stops once every row has
    either hit its own EOS or reached `max_new_tokens`.

    BIT-IDENTICAL TO SINGLE-ROW is verified two ways, deliberately kept separate
    (`tests/test_cw_sim_greedy_generate_batch.py`): (a) a batch of SAME-length prompts
    (no padding at all) must match `greedy_generate` called once per row, exactly --
    this is the load-bearing proof that the batching/EOS-freezing/position-id logic
    itself introduces no bug. (b) a batch of DIFFERENT-length prompts (real padding)
    is measured, not asserted bit-identical -- a separate measurement
    (`tests/test_cw_batch_extract.py`) shows that batched bf16 attention over padded rows
    perturbs a padded row's own real-token hidden states by up to ~0.25 absolute
    (cosine similarity still 0.9998-1.0000), a genuine backbone numerical property, not
    a bug in this function; at the level of DISCRETE argmax token choices this can
    occasionally (not reliably never) flip a close-call token relative to single-row
    generation, so mixed-length-batch equality is a measured property (reported, not
    silently assumed), while same-length-batch equality is the actual correctness gate.

    `use_cache=True` + `position_ids` is tried first (the "modern" contract every real
    HF causal LM supports together); ANY `TypeError` (missing `position_ids`, missing
    `use_cache`/`past_key_values` -- the smoke test's `CausalStubLM` has neither) falls
    back to the single naive path used for the WHOLE rest of the call: full-recompute-
    every-step over `input_ids`/`attention_mask` alone (no `position_ids`, no cache) --
    still row-batched, still correct, just without the O(T) speedup, exactly mirroring
    `greedy_generate`'s own single-flag fallback philosophy rather than independently
    toggling two capabilities that in practice always travel together."""
    b = len(prompt_ids_list)
    eos_id = tok.eos_token_id
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else eos_id
    lengths = [len(p) for p in prompt_ids_list]
    max_len = max(lengths)

    input_ids = torch.full((b, max_len), pad_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros((b, max_len), dtype=torch.long, device=device)
    for i, p in enumerate(prompt_ids_list):
        input_ids[i, max_len - len(p):] = torch.tensor(p, dtype=torch.long, device=device)
        attention_mask[i, max_len - len(p):] = 1

    context.set(c_vecs)
    position_ids = attention_mask.long().cumsum(-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 1)

    gen_ids: list[list[int]] = [[] for _ in range(b)]
    finished = [False] * b
    past_key_values = None
    modern_supported = True

    for step in range(max_new_tokens):
        with torch.no_grad():
            if step == 0:
                try:
                    out = model(input_ids=input_ids, attention_mask=attention_mask,
                               position_ids=position_ids, use_cache=True)
                    past_key_values = getattr(out, "past_key_values", None)
                    modern_supported = past_key_values is not None
                except TypeError:
                    modern_supported = False
                    out = model(input_ids=input_ids, attention_mask=attention_mask)
            elif modern_supported:
                position_ids = torch.cat([position_ids, position_ids[:, -1:] + 1], dim=1)
                out = model(input_ids=input_ids[:, -1:], attention_mask=attention_mask,
                           position_ids=position_ids[:, -1:], past_key_values=past_key_values,
                           use_cache=True)
                past_key_values = getattr(out, "past_key_values", None)
            else:
                out = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = out.logits
            next_ids = logits[:, -1, :].argmax(dim=-1)  # [B]
        next_ids_list = next_ids.tolist()
        any_active = False
        for i in range(b):
            if finished[i]:
                continue
            gen_ids[i].append(next_ids_list[i])
            any_active = True
            if eos_id is not None and next_ids_list[i] == eos_id:
                finished[i] = True
        next_col = next_ids.unsqueeze(1)  # [B, 1]
        input_ids = torch.cat([input_ids, next_col], dim=1)
        attention_mask = torch.cat(
            [attention_mask, torch.ones((b, 1), dtype=torch.long, device=device)], dim=1)
        del out, logits
        if on_token is not None:
            on_token()
        if not any_active:
            break
    if device == "cuda":
        torch.cuda.empty_cache()
    return [tok.decode(g, skip_special_tokens=True).strip() for g in gen_ids]


def _simple_derangement(n: int, rng: np.random.Generator) -> np.ndarray:
    """A derangement (no fixed points) over `range(n)`. NOT learner-history-matched --
    that is `score_arm_for_batch`'s own, more rigorous, SCORING-time mechanism
    (`learner_history_matched_derangement`, bin-respecting). Ancestral generation is a
    diagnostic pass over a small subsample, so a plain derangement is used here
    rather than reproducing the bulk eval's exact instance.

    A random cyclic rotation by `shift` in `[1, n-1]`: `perm[i] = (i + shift) % n`. GUARANTEED
    no fixed points for any n>=2 (a rotation fixes an element only at shift=0 mod n, excluded
    by construction) -- not a uniform draw over all derangements, but that is not needed
    here. A "permute-then-repair-fixed-points" construction is avoided because it can
    flip a repaired pair right back to identity at n=2 (repairing index 0 by swapping
    with index 1, then "repairing" the now-stale fixed-point list's index 1 by swapping
    back)."""
    if n <= 1:
        return np.arange(n)
    shift = int(rng.integers(1, n))
    return (np.arange(n) + shift) % n


def run_ancestral_generation_pass(m: dict, eval_attempts_batch: pd.DataFrame, manifest: dict,
                                  scores_by_arm: dict, device: str, n_generation_rows: int,
                                  max_new_tokens: int, seed: int, on_progress=None,
                                  completed_row_idx: set[int] | None = None,
                                  on_token=None, arms: tuple[str, ...] = ARMS,
                                  modes: tuple[str, ...] = ("conditional_imitation", "ancestral")
                                  ) -> list[dict]:
    """Generation pass run IN-RUN on a small, evenly spaced subsample of the
    ALREADY-SCORED personalised-arm eval rows (one `greedy_generate` call per row, arm
    and mode; a full-corpus post-hoc pass would be far more expensive).

    Two generation modes per row per arm, mirroring `fa_sim3_full_run.py`'s own split:
      - `conditional_imitation`: generates conditioned on the row's OWN OBSERVED class.
      - `ancestral`: the class is SAMPLED from p(c|u,q) -- reusing the personalised arm's
        ALREADY-COMPUTED `p_class_meanK` (the K-marginalised class posterior under the
        row's own belief) rather than a second live `outcome_head` forward per arm.
        A simplification relative to FA-SIM3 (which recomputes p(c|u,q) per arm from that
        arm's own `u`): CW shares one class-posterior across all four arms' ancestral
        draws here, since this is a diagnostic instrument, not a scoring gate, and the
        class semantics ("what would this belief produce") is a property of the row's
        belief and item, not of which arm's production mechanism is being probed.

    `u` per arm mirrors `score_arm_for_batch`'s own arm semantics.  One row-specific
    epsilon is common to personalised and shuffled-u, removing avoidable Monte-Carlo
    noise from their contrast; dw0 is exactly zero and u_mean is the posterior mean.
    One ancestral class is sampled per row and reused across all arms, so the arm
    comparison is not confounded by different requested outcome classes.  Row-specific
    RNG seeds make skipping completed rows exactly resumable.

    Returns a list of JSON-serialisable records (never uploaded -- the same licence rule
    as `write_generations_jsonl`: this generates NEAR-COPIES of licensed code).

    Each record also carries `prior_code` and `prompt_text` (the DECODED prompt actually
    conditioned on): `prior_code` is what the copy-baseline metrics need (prior->current
    normalised edit distance, prior-equals-current-verbatim rate, on the generation
    subsample) and `prompt_text` is what the codec-similarity metric needs (it
    re-encodes `prompt ‖ text` through the backbone, matching
    `fa_sim3_eval_5e5f.py::codec_similarity_one`'s own convention). An optional
    `on_progress(n_done, n_total, elapsed_s, row_records)` callback is invoked after
    every ROW (i.e. every 8 records: 4 arms x 2 modes), so the overrun watchdog gets a
    progress signal DURING this loop, the most expensive phase of a generation run, in
    which the training loop's own heartbeat never fires on an eval-only `--resume` run.

    `arms`/`modes` are optional and default to the module's own `ARMS` and the two-mode
    set. A caller may restrict `arms` (e.g. launch2's `("personalised","shuffled_u","dw0")`,
    never `u_mean`) and extend `modes` with `"argmax"` -- the class conditioned on is
    then the row's OWN OBSERVED class (`conditional_imitation`), a SAMPLE from
    `p_class_meanK` (`ancestral`), or `p_class_meanK`'s own arg-max (`argmax`, the
    "most likely class" variant reported beside ancestral for the regime in which the
    outcome is not given -- the benchmark models this is compared against output one
    best guess, they do not sample)."""
    kept = scores_by_arm["personalised"]["kept_row_idx"]
    if len(kept) == 0:
        return []
    n_gen = min(n_generation_rows, len(kept))
    sel = np.linspace(0, len(kept) - 1, n_gen, dtype=int)

    mu_all = scores_by_arm["personalised"]["mu"]
    log_sigma_all = scores_by_arm["personalised"]["log_sigma"]
    p_class_all = scores_by_arm["personalised"]["p_class_meanK"]
    partner = _simple_derangement(len(kept), np.random.default_rng(seed + 301))

    records = []
    selected_global_rows = {int(kept[i]) for i in sel.tolist()}
    completed = set() if completed_row_idx is None else (set(completed_row_idx) & selected_global_rows)
    n_done = len(completed)
    pass_t0 = time.time()
    for local_i in sel.tolist():
        global_row = int(kept[local_i])
        if global_row in completed:
            continue
        plan = row_generation_plan(m, eval_attempts_batch, manifest, mu_all, log_sigma_all,
                                   p_class_all, partner, local_i, global_row, seed, device, modes)
        row_records = []

        for arm in arms:
            u = plan["u_by_arm"][arm]
            with torch.no_grad():
                c_vec = (torch.zeros(1, m["hyper"].m, device=device)
                         if arm == "dw0" else m["hyper"](u))

            for mode in modes:
                conditioned_c = plan["class_by_mode"][mode]
                gen = greedy_generate(m["model"], m["tok"], m["context"], c_vec,
                                      plan["prompt_ids_by_mode"][mode], max_new_tokens, device,
                                      on_token=on_token)
                row_records.append({
                    "row_idx": global_row, "arm": arm, "generation_mode": mode,
                    "learner": plan["learner"], "problem_id": plan["problem_id"],
                    "true_class": CW_OUTCOME_CLASS_TOKENS[plan["true_c"]],
                    "conditioned_class": CW_OUTCOME_CLASS_TOKENS[conditioned_c],
                    "true_class_code": plan["true_c"], "conditioned_class_code": conditioned_c,
                    "generated_code": gen, "real_completion": plan["real_completion"],
                    "prior_code": plan["prior_code"], "prompt_text": plan["prompt_text_by_mode"][mode],
                    "code_state_id": plan["code_state_id"],
                })
        records.extend(row_records)
        n_done += 1
        if on_progress is not None:
            on_progress(n_done, n_gen, time.time() - pass_t0, row_records)
        if device == "cuda":
            torch.cuda.empty_cache()
    return records


def row_generation_plan(m: dict, eval_attempts_batch: pd.DataFrame, manifest: dict,
                        mu_all: np.ndarray, log_sigma_all: np.ndarray, p_class_all: np.ndarray,
                        partner: np.ndarray, local_i: int, global_row: int, seed: int, device: str,
                        modes: tuple[str, ...]) -> dict:
    """The per-row math `run_ancestral_generation_pass` needs BEFORE any `greedy_generate`
    call -- factored out so a BATCHED caller (many rows' prompts/c_vec inputs assembled
    together, one `greedy_generate_batch` call per (arm,mode) cell rather than one
    `greedy_generate` call per row) computes the IDENTICAL RNG draws, sampled/argmax
    classes, and prompt token ids as the single-row path above -- a single source of
    truth for this math, not two copies that could silently drift (covered by the
    CRN/ancestral resumability test)."""
    gen_rng = torch.Generator(device=device).manual_seed(seed + 300 + global_row)
    row = eval_attempts_batch.iloc[global_row]
    true_c = int(row["outcome_code"])
    mu_row = torch.tensor(mu_all[local_i:local_i + 1], device=device, dtype=torch.float32)
    log_sigma_row = torch.tensor(log_sigma_all[local_i:local_i + 1], device=device, dtype=torch.float32)
    partner_i = int(partner[local_i])
    mu_partner = torch.tensor(mu_all[partner_i:partner_i + 1], device=device, dtype=torch.float32)
    log_sigma_partner = torch.tensor(log_sigma_all[partner_i:partner_i + 1], device=device, dtype=torch.float32)
    class_probs_row = torch.tensor(p_class_all[local_i], device=device, dtype=torch.float32)
    class_probs_row = class_probs_row / class_probs_row.sum().clamp(min=1e-8)
    eps = torch.randn(mu_row.shape, device=device, generator=gen_rng)
    u_by_arm = {
        "personalised": mu_row + torch.exp(log_sigma_row) * eps,
        "shuffled_u": mu_partner + torch.exp(log_sigma_partner) * eps,
        "u_mean": mu_row,
        "dw0": None,
    }
    sampled_c = int(torch.multinomial(class_probs_row, 1, generator=gen_rng).item())
    argmax_c = int(torch.argmax(class_probs_row).item())
    class_by_mode = {"conditional_imitation": true_c, "ancestral": sampled_c, "argmax": argmax_c}

    statement = manifest["statements"][int(row["problem_id"])]
    prior_code = row["prior_code"] if isinstance(row["prior_code"], str) else None
    # Decode without BOS/EOS artefacts before the codec scorer re-tokenises the
    # prompt; otherwise a literal decoded BOS plus a newly-added BOS would change
    # the external ruler's input. Outcome-class markers are ordinary content tokens.
    prompt_ids_by_mode = {}
    prompt_text_by_mode = {}
    for mode in modes:
        ids, _, _ = build_reserved_prompt_ids(
            statement, prior_code, m["tok"], MAX_LEN, MIN_COMPLETION_TOKENS,
            outcome_code=class_by_mode[mode])
        prompt_ids_by_mode[mode] = ids
        prompt_text_by_mode[mode] = m["tok"].decode(ids, skip_special_tokens=True)

    return {
        "row": row, "true_c": true_c, "u_by_arm": u_by_arm, "class_by_mode": class_by_mode,
        "prompt_ids_by_mode": prompt_ids_by_mode, "prompt_text_by_mode": prompt_text_by_mode,
        "prior_code": prior_code, "learner": str(row["learner"]),
        "problem_id": int(row["problem_id"]), "real_completion": str(row["code"]),
        "code_state_id": str(row["code_state_id"]),
    }


def run_batched_generation_pass(m: dict, eval_attempts_batch: pd.DataFrame, manifest: dict,
                                scores_by_arm: dict, device: str, n_generation_rows: int,
                                max_new_tokens: int, seed: int, arms: tuple[str, ...],
                                modes: tuple[str, ...], batch_size: int,
                                on_progress=None,
                                completed_row_idx: set[int] | None = None) -> list[dict]:
    """Row-batched generation pass. The launch2/TIKTOC generation volume (thousands of
    rows x up to 3 arms x 3 modes) is unaffordable at `run_ancestral_generation_pass`'s
    one-`greedy_generate`-call-per-(row,arm,mode) rate (measured ~23s/generation on the
    base run's smoke pass) -- this batches
    ROWS together within each (arm, mode) cell via `greedy_generate_batch`, one forward
    pass per decode step scoring `batch_size` rows at once instead of one.

    Per-row math (RNG draws for epsilon/ancestral-sample/argmax, prompt token ids) is
    IDENTICAL to `run_ancestral_generation_pass` -- both call the same
    `row_generation_plan`, so a batched and an unbatched pass given the SAME rows/seed
    request the SAME classes and build the SAME prompts; only the generation CALL
    differs (one `greedy_generate_batch` per (arm,mode,chunk) vs one `greedy_generate`
    per (row,arm,mode)). `greedy_generate_batch`'s test suite
    (`tests/test_cw_sim_greedy_generate_batch.py`) shows batched == sequential on a stub
    for same-length prompts, and close-but-not-bit-identical for padded variable-length
    prompts on a REAL backbone (attention-padding numerical noise, not a bug);
    `cw_sim_launch2_batch_equality_check.py` re-measures that gap on real rows from this
    pipeline rather than assuming the stub's guarantee transfers.

    `arms`/`modes` are REQUIRED (no default): the caller states its cell set explicitly.

    Resumability is per-ROW, not per-(row,arm,mode) cell (a coarser grain than
    `run_ancestral_generation_pass`'s own, a deliberate simplification): `completed_row_idx`
    rows are skipped ENTIRELY; a row interrupted mid-way through its arm/mode cells on a
    previous attempt is fully REDONE on resume, trading a little redundant compute for a
    much simpler resume boundary (batches span many rows at once, so a partial-row
    resume boundary inside a batch has no natural meaning here).

    `on_progress(arm, mode, n_rows_done_in_cell, n_rows_total, elapsed_s)` is called
    after every batched chunk (a different granularity from
    `run_ancestral_generation_pass`'s per-ROW callback -- document this at the call
    site, do not assume the two are interchangeable)."""
    kept = scores_by_arm["personalised"]["kept_row_idx"]
    if len(kept) == 0:
        return []
    n_gen = min(n_generation_rows, len(kept))
    sel = np.linspace(0, len(kept) - 1, n_gen, dtype=int)

    mu_all = scores_by_arm["personalised"]["mu"]
    log_sigma_all = scores_by_arm["personalised"]["log_sigma"]
    p_class_all = scores_by_arm["personalised"]["p_class_meanK"]
    partner = _simple_derangement(len(kept), np.random.default_rng(seed + 301))

    selected_global_rows = {int(kept[i]) for i in sel.tolist()}
    completed = set() if completed_row_idx is None else (set(completed_row_idx) & selected_global_rows)
    pending_local_by_row = {int(kept[i]): i for i in sel.tolist() if int(kept[i]) not in completed}

    plans = {
        global_row: row_generation_plan(m, eval_attempts_batch, manifest, mu_all, log_sigma_all,
                                        p_class_all, partner, local_i, global_row, seed, device, modes)
        for global_row, local_i in pending_local_by_row.items()
    }
    pending_rows = sorted(plans)
    records_by_row: dict[int, list[dict]] = {r: [] for r in pending_rows}
    pass_t0 = time.time()

    for arm in arms:
        for mode in modes:
            for start in range(0, len(pending_rows), max(1, batch_size)):
                chunk_rows = pending_rows[start:start + max(1, batch_size)]
                c_vecs = []
                prompt_ids_list = []
                with torch.no_grad():
                    for r in chunk_rows:
                        u = plans[r]["u_by_arm"][arm]
                        c_vec = (torch.zeros(1, m["hyper"].m, device=device)
                                 if arm == "dw0" else m["hyper"](u))
                        c_vecs.append(c_vec)
                        prompt_ids_list.append(plans[r]["prompt_ids_by_mode"][mode])
                    c_vecs_t = torch.cat(c_vecs, dim=0) if c_vecs else torch.zeros(0, m["hyper"].m, device=device)
                gens = greedy_generate_batch(m["model"], m["tok"], m["context"], c_vecs_t,
                                             prompt_ids_list, max_new_tokens, device)
                for r, gen in zip(chunk_rows, gens):
                    plan = plans[r]
                    conditioned_c = plan["class_by_mode"][mode]
                    records_by_row[r].append({
                        "row_idx": r, "arm": arm, "generation_mode": mode,
                        "learner": plan["learner"], "problem_id": plan["problem_id"],
                        "true_class": CW_OUTCOME_CLASS_TOKENS[plan["true_c"]],
                        "conditioned_class": CW_OUTCOME_CLASS_TOKENS[conditioned_c],
                        "true_class_code": plan["true_c"], "conditioned_class_code": conditioned_c,
                        "generated_code": gen, "real_completion": plan["real_completion"],
                        "prior_code": plan["prior_code"], "prompt_text": plan["prompt_text_by_mode"][mode],
                        "code_state_id": plan["code_state_id"],
                    })
                if device == "cuda":
                    torch.cuda.empty_cache()
                if on_progress is not None:
                    on_progress(arm, mode, min(start + len(chunk_rows), len(pending_rows)),
                               len(pending_rows), time.time() - pass_t0)

    records = []
    for r in pending_rows:
        records.extend(records_by_row[r])
    return records


def generation_codec_similarity_one(m: dict, transform: CodecTransform, device: str,
                                    record: dict, real_cache: dict,
                                    layer: int = 15, k_strip: int = 5,
                                    rank: int = CODEC_RANK, on_forward=None) -> float:
    """Codec cosine for one generated/reference pair under a neutral metric context.

    The trained adapter is zeroed for both sides, matching the existing FA generation
    scorer: the codec is an external ruler, not another personalised arm.  Reference
    vectors are cached by (row, conditioned class), which saves six of the eight real
    encodes for each completed row while keeping conditional and ancestral prompts
    distinct when their class token differs.
    """
    m["context"].set(torch.zeros(1, m["hyper"].m, device=device))
    bundle = {"model": m["model"], "tokenizer": m["tok"], "device": device}
    prefix = str(record["prompt_text"])
    real_full = f"{prefix} {str(record['real_completion']).strip()}"
    generated_full = f"{prefix} {str(record['generated_code']).strip()}"
    cache_key = (int(record["row_idx"]), int(record["conditioned_class_code"]))
    if cache_key not in real_cache:
        _, real_states = encode_one(bundle, prefix, real_full)
        real_coords = transform.apply(real_states[layer][None, :], k=k_strip,
                                      rank=rank, whiten=False)[0]
        real_cache[cache_key] = real_coords / (np.linalg.norm(real_coords) + 1e-12)
        if on_forward is not None:
            on_forward()
    _, generated_states = encode_one(bundle, prefix, generated_full)
    generated_coords = transform.apply(generated_states[layer][None, :], k=k_strip,
                                       rank=rank, whiten=False)[0]
    generated_vec = generated_coords / (np.linalg.norm(generated_coords) + 1e-12)
    if on_forward is not None:
        on_forward()
    if device == "cuda":
        torch.cuda.empty_cache()
    return float(np.dot(generated_vec, real_cache[cache_key]))


def ancestral_generation_metrics(records: list[dict]) -> dict:
    """Scalar summary of `run_ancestral_generation_pass`'s records -- the part that may be
    uploaded: generation-derived METRICS are computed where the run executes and only the
    scalars leave it; the text stays local. Exact-match rate and mean generated-code
    length, per (arm, generation_mode) cell."""
    cells: dict[str, dict] = {}
    for r in records:
        key = f"{r['arm']}__{r['generation_mode']}"
        cells.setdefault(key, {"n": 0, "n_exact": 0, "total_len": 0})
        c = cells[key]
        c["n"] += 1
        c["n_exact"] += int(r["generated_code"].strip() == r["real_completion"].strip())
        c["total_len"] += len(r["generated_code"])
    return {
        key: {
            "n": c["n"],
            "exact_match_rate": c["n_exact"] / c["n"] if c["n"] else float("nan"),
            "mean_generated_chars": c["total_len"] / c["n"] if c["n"] else float("nan"),
        }
        for key, c in cells.items()
    }


def assert_resume_compatible(ckpt: dict, architecture: dict, item_encoder: str,
                             learner_split_sha256: str, problem_codec_sha256: str,
                             sigma_parameterization: str | None = None) -> None:
    """A resumed run must be resuming the SAME config, not silently continuing under a
    drifted one (config drift otherwise goes undetected until a checkpoint is scored).
    Kept as its own function (rather than inlined in `main()`) so it is unit-testable
    without a real backbone/corpus."""
    assert ckpt["architecture"] == architecture, "resumed checkpoint's architecture does not match this run's config"
    assert ckpt["item_encoder"] == item_encoder, "resumed checkpoint's item_encoder does not match"
    assert ckpt["learner_split_sha256"] == learner_split_sha256, "resumed checkpoint's learner split has drifted"
    assert ckpt["problem_codec_sha256"] == problem_codec_sha256, "resumed checkpoint's problem codec has drifted"
    assert ckpt["held_out_items"] == list(HELD_OUT_ITEMS), "resumed checkpoint's item holdout has drifted"
    if sigma_parameterization is not None and "sigma_parameterization" in ckpt:
        assert ckpt["sigma_parameterization"] == sigma_parameterization, (
            "resumed checkpoint's sigma parameterization does not match"
        )
_ARM_SCORE_ARRAY_KEYS = ("mixture_nll", "mixture_nll_raw", "true_class_nll", "true_class_top1",
                        "true_class_exact_match", "p_class_meanK", "true_class_nll_per_k",
                        "kept_row_idx", "mu", "log_sigma")
_ARM_SCORE_SCALAR_KEYS = ("n_dropped_budget", "elapsed_s", "n_forward_calls", "n_rows_scored",
                         "n_dropped_derangement")


def _arm_scores_path(out_dir: pathlib.Path, arm: str) -> pathlib.Path:
    return out_dir / "eval_partial" / f"scores_{arm}.npz"


def save_arm_scores(scores: dict, out_dir: pathlib.Path, arm: str, bucket: str, out_prefix: str) -> None:
    """Persist ONE arm's scores the instant it finishes scoring, to local disk AND to the
    S3 bucket given with `--bucket` (when set) -- never held in memory until the whole
    4-arm loop completes, so a crash after arm 3 of 4 does not lose all four."""
    path = _arm_scores_path(out_dir, arm)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {k: scores[k] for k in _ARM_SCORE_ARRAY_KEYS}
    payload.update({k: np.array(scores[k]) for k in _ARM_SCORE_SCALAR_KEYS})
    np.savez(path, **payload)
    if bucket:
        import boto3
        boto3.client("s3").upload_file(str(path), bucket, f"{out_prefix}/eval_partial/{path.name}")
    print(f"[cw-sim] eval checkpoint: arm={arm} persisted ({path})", flush=True)


def load_arm_scores(path: pathlib.Path, arm: str) -> dict:
    data = np.load(path, allow_pickle=True)
    scores = {k: data[k] for k in _ARM_SCORE_ARRAY_KEYS}
    scores.update({k: data[k].item() for k in _ARM_SCORE_SCALAR_KEYS})
    scores["n_dropped_budget"] = int(scores["n_dropped_budget"])
    scores["n_forward_calls"] = int(scores["n_forward_calls"])
    scores["n_rows_scored"] = int(scores["n_rows_scored"])
    scores["n_dropped_derangement"] = int(scores["n_dropped_derangement"])
    scores["arm"] = arm
    return scores


def try_resume_arm_scores(out_dir: pathlib.Path, arm: str, bucket: str, out_prefix: str) -> dict | None:
    """Eval resumability: on `--resume`, an arm already scored (by this process or a
    prior, now-dead one on the same or a fresh machine) is loaded rather than re-scored.
    Pulls from the bucket first if not present locally, mirroring
    `_pull_latest_s3_checkpoint`."""
    path = _arm_scores_path(out_dir, arm)
    if not path.exists() and bucket:
        import boto3
        s3 = boto3.client("s3")
        key = f"{out_prefix}/eval_partial/{path.name}"
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            s3.download_file(bucket, key, str(path))
            print(f"[cw-sim] eval resume: pulled arm={arm} scores from bucket {bucket}, key {key}", flush=True)
        except Exception:
            return None
    if not path.exists():
        return None
    return load_arm_scores(path, arm)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default=None)
    ap.add_argument("--code-map", required=True)
    ap.add_argument("--diff-map", required=True)
    ap.add_argument("--codealone-meta", required=True)
    ap.add_argument("--codealone-states", required=True)
    ap.add_argument("--diff-meta", required=True)
    ap.add_argument("--diff-states", required=True)
    ap.add_argument("--backbone", default="gemma4-e4b")
    ap.add_argument("--item-encoder", default="mlp", choices=["linear", "mlp"])
    ap.add_argument("--sigma-parameterization", default="direct",
                    choices=["direct", "precision_accumulating"],
                    help="belief sigma head; precision_accumulating is the classifier (cw_sigma) warm-start regime")
    ap.add_argument("--warm-start-checkpoint", default="",
                    help="standalone classifier (cw_sigma) checkpoint; with --br4b-joint-set its "
                         "encoder/filter are loaded and the direct sigma head is kept fresh")
    ap.add_argument("--br4b-joint-set", action="store_true",
                    help="train the shared joint-set K-mixture objective with teacher-posterior KL")
    ap.add_argument("--target-set-k", type=int, default=8,
                    help="joint-set objective: K draws shared across each episode target set")
    ap.add_argument("--target-set-min", type=int, default=10)
    ap.add_argument("--target-set-max", type=int, default=20)
    ap.add_argument("--teacher-kl-weight", type=float, default=0.01,
                    help="joint-set objective: teacher-posterior KL ceiling")
    ap.add_argument("--learner-batch-size", type=int, default=4)
    ap.add_argument("--micro-batch-size", type=int, default=8,
                    help="max attempts per backbone forward pass in the production path "
                         "(length-bucketed, gradient-accumulated) -- bounds peak memory "
                         "independent of --learner-batch-size.")
    ap.add_argument("--max-steps", type=int, default=1000,
                    help="ceil(686 train learners / learner_batch_size) ~= one epoch per "
                         "~172 steps at the default batch size -- a pilot run sets "
                         "this small and explicit.")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--kl-max-weight", type=float, default=0.01)
    ap.add_argument("--kl-anneal-epochs", type=float, default=2.0,
                    help="linear KL ramp 0 -> --kl-max-weight over this many epochs "
                         "(epoch = step / ceil(n_train_learners / learner_batch_size)), "
                         "then held; 0 disables annealing (full weight from step 0).")
    ap.add_argument("--free-bits", type=float, default=0.05)
    ap.add_argument("--lambda-a-grid", default="0.1,0.3,1.0")
    ap.add_argument("--lambda-a-select-every", type=int, default=200)
    ap.add_argument("--out-dir", default="/tmp/cw_5_run")
    ap.add_argument("--bucket", default="")
    ap.add_argument("--out-prefix", default="cw-5-full-run")
    ap.add_argument("--checkpoint-every-s", type=float, default=900.0)
    ap.add_argument("--heartbeat-every-s", type=float, default=25.0)
    ap.add_argument("--fuse-minutes", type=float, default=240.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--git-sha", default=None)
    ap.add_argument("--self-terminate", action="store_true")
    ap.add_argument("--resume", action="store_true",
                    help="resume from the latest checkpoint in --out-dir (pulling it down "
                         "from --bucket first if --out-dir has none locally). Restores "
                         "model/optimizer/torch+numpy+python RNG state and continues from "
                         "the checkpointed step -- asserts the resumed config (architecture/ "
                         "item_encoder/split+codec hashes) matches this invocation's.")
    ap.add_argument("--eval-val-learners", type=int, default=20,
                    help="how many VAL learners to score (pilot scale by default -- "
                         "4 classes x K x forward is the dominant eval cost).")
    ap.add_argument("--eval-k", type=int, default=8,
                    help="K for the main 4-arm scoring pass (the K-stability curve below "
                         "always sweeps 1/4/8/16 regardless of this value).")
    ap.add_argument("--skip-eval", action="store_true",
                    help="training-only run, no eval pass (for a pure throughput/training "
                         "smoke run) -- an evaluation run must NOT pass this.")
    ap.add_argument("--skip-k-curve", action="store_true",
                    help="skip the K=1/4/8/16 stability curve (measured to converge by "
                         "K=8; K=16 alone costs ~2x the K=8 arm, so re-sweeping it on every "
                         "run is the largest single eval cost). --eval-k still controls the "
                         "main 4-arm scoring pass's own K.")
    ap.add_argument("--n-generation-rows", type=int, default=15,
                    help="ancestral generation: how many personalised-arm eval rows "
                         "to generate text for (x4 arms x2 modes each) -- kept small since "
                         "generation runs IN-RUN, one row at a time, not as a separate "
                         "post-hoc pass.")
    ap.add_argument("--max-new-tokens", type=int, default=400,
                    help="generation length cap for the ancestral-generation pass -- code "
                         "completions run longer than FA's short answers.")
    ap.add_argument("--extended-generation-metrics", action="store_true",
                    help="compute exact/NED/codec/copy metrics for both generation modes, "
                         "persist numeric rows after each completed source row, and report "
                         "the arm-symmetric all-four-arms-missed comparison. Raw and "
                         "generated text remains local-only.")
    ap.add_argument("--generation-n-boot", type=int, default=1000,
                    help="learner-clustered bootstrap replicates for final generation contrasts")
    ap.add_argument("--generation-flush-rows", type=int, default=1,
                    help="completed source rows between numeric partial uploads; default 1 "
                         "means a crash loses at most the currently-generating row")
    ap.add_argument("--codec-transform", default="transforms/problem_mean_L15.npz")
    ap.add_argument("--codec-layer", type=int, default=15)
    ap.add_argument("--codec-strip-k", type=int, default=5)
    ap.add_argument("--codec-rank", type=int, default=CODEC_RANK)
    ap.add_argument("--run-log-path", default="",
                    help="detached stdout/stderr file to retain as a final log upload")
    ap.add_argument("--estimated-minutes", type=float, default=None,
                    help="overrun trigger: this run's own wall-clock estimate, stated "
                         "before launch. Once the projected total "
                         "(measured rate x remaining units) exceeds 2x this, the run "
                         "self-terminates and writes overrun_trigger.json -- a broken-run "
                         "detector, not a budget. Defaults to half of --fuse-minutes if "
                         "not given explicitly (a caller should pass the real number).")
    args = ap.parse_args()
    if args.br4b_joint_set:
        if args.target_set_k < 8:
            ap.error("--br4b-joint-set requires --target-set-k >= 8")
        if not (10 <= args.target_set_min <= args.target_set_max <= 20):
            ap.error("joint-set target sets must satisfy 10 <= min <= max <= 20")
        if args.sigma_parameterization != "direct":
            ap.error("--br4b-joint-set requires the direct sigma parameterisation")
    if args.estimated_minutes is None:
        args.estimated_minutes = args.fuse_minutes / 2.0
        print(f"[cw-sim] --estimated-minutes not given, defaulting to --fuse-minutes/2 = "
             f"{args.estimated_minutes:.1f} min -- pass the real estimate explicitly", flush=True)

    # Boot fuse: an OS-level shutdown timer, armed BEFORE anything else runs, so a hang
    # during data loading / backbone loading / a stuck training loop is bounded even if
    # the script itself never reaches its own in-loop elapsed-time check below. The same
    # pattern as the FA simulator's runner; intended for a disposable cloud GPU machine.
    subprocess.run(["sudo", "shutdown", "-c"], check=False)
    subprocess.run(["sudo", "shutdown", "-h", f"+{int(args.fuse_minutes)}"], check=False)
    print(f"[cw-sim] boot fuse armed: shutdown in {args.fuse_minutes} min unless cancelled", flush=True)

    git_sha = args.git_sha or _git_sha()
    print(f"[cw-sim] git_sha={git_sha}", flush=True)
    seed_everything(args.seed)
    device = default_device()
    print(f"[cw-sim] device={device}", flush=True)

    manifest = load_cw5_manifest(args)
    print(f"[cw-sim] {len(manifest['attempts'])} outcome-valid attempts, "
         f"{manifest['n_problems_after_holdout']} problems (of {manifest['n_problems_before_holdout']} "
         f"before the {len(HELD_OUT_ITEMS)}-item holdout), "
         f"{len(manifest['train_learners'])} train / {len(manifest['val_learners'])} val learners",
         flush=True)

    backbone_bundle = load_backbone(args.backbone, device=device, allow_download=True)
    m = build_model(CW5_DIMS, backbone_bundle, args.seed, device, item_encoder=args.item_encoder,
                    sigma_parameterization=args.sigma_parameterization)
    print(f"[cw-sim] model built: codec_dim={CW5_DIMS['codec_dim']} item_encoder={args.item_encoder} "
         f"sigma_parameterization={args.sigma_parameterization} device={device}", flush=True)

    trainable = (list(m["attempt_encoder"].parameters()) + list(m["filt"].parameters())
                + list(m["outcome_head"].parameters()) + list(m["y_ever_head"].parameters())
                + list(m["teacher"].parameters())
                + list(m["hyper"].parameters())
                + [p for b in m["bases"].values() for p in b.parameters()])
    optimizer = torch.optim.Adam(trainable, lr=args.lr)
    lambda_a_grid = tuple(float(x) for x in args.lambda_a_grid.split(","))
    lambda_a = lambda_a_grid[0]

    learner_split_sha256 = _sha256(REPO / "splits" / "cw_learner_split.csv")
    problem_codec_sha256 = _sha256(PROBLEM_CODEC_DIM64_PATH)
    if args.warm_start_checkpoint:
        warm_path = pathlib.Path(args.warm_start_checkpoint)
        assert warm_path.exists(), f"warm-start checkpoint does not exist: {warm_path}"
        warm_ckpt = (load_br4b_warm_start_checkpoint(
            m, warm_path, CW5_DIMS, learner_split_sha256, problem_codec_sha256)
                     if args.br4b_joint_set else load_warm_start_checkpoint(
                         m, warm_path, CW5_DIMS, args.item_encoder, args.sigma_parameterization,
                         learner_split_sha256, problem_codec_sha256))
        print(f"[cw-sim] warm-start loaded: {warm_path} (seed={warm_ckpt.get('seed')}, "
              f"sigma_parameterization={warm_ckpt.get('sigma_parameterization', 'unspecified')})", flush=True)

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.jsonl"
    lambda_probe_path = out_dir / "lambda_a_probes.jsonl"
    heartbeat_path = out_dir / "heartbeat.log"

    # Overrun triggers: one instance per phase (training, eval, generation) since each
    # phase has its own units-of-progress and its own share of
    # --estimated-minutes is unknown -- checked against the WHOLE run's estimate at both
    # phases' own progress, which is conservative (a phase alone projecting >2x the total
    # run's estimate is already worth stopping for).
    train_trigger = OverrunTrigger(args.estimated_minutes, out_dir, args.bucket, args.out_prefix, "cw-5-train")
    eval_trigger = OverrunTrigger(args.estimated_minutes, out_dir, args.bucket, args.out_prefix, "cw-5-eval")
    generation_trigger = OverrunTrigger(
        args.estimated_minutes, out_dir, args.bucket, args.out_prefix, "cw-5-b3-generation")

    train_learners = sorted(manifest["train_learners"])
    n_batches_per_epoch = max(1, (len(train_learners) + args.learner_batch_size - 1) // args.learner_batch_size)

    def checkpoint_and_upload(step: int, is_final: bool) -> None:
        # atom.utils.checkpoints.save_checkpoint: temp-file + os.replace atomicity, local
        # keep-last-3 rotation + a never-rotated best.pt (is_best=is_final here -- no
        # validation metric drives checkpoint selection; the final checkpoint is "best").
        # RNG state is load-bearing: reparameterize_from_log_sigma draws from the GLOBAL
        # torch RNG during training (no explicit generator/eps is passed there), so exact
        # resume needs the saved RNG states.
        state = {
            "attempt_encoder": m["attempt_encoder"].state_dict(), "filt": m["filt"].state_dict(),
            "outcome_head": m["outcome_head"].state_dict(), "y_ever_head": m["y_ever_head"].state_dict(),
            "teacher": m["teacher"].state_dict(),
            "hyper": m["hyper"].state_dict(), "basis": {n: b.state_dict() for n, b in m["bases"].items()},
            "optimizer": optimizer.state_dict(),
            "torch_rng": torch.get_rng_state(), "numpy_rng": np.random.get_state(),
            "python_rng": random.getstate(),
            "architecture": CW5_DIMS, "item_encoder": args.item_encoder,
            "sigma_parameterization": args.sigma_parameterization, "lambda_a": lambda_a,
            "codec_path": str(PROBLEM_CODEC_DIM64_PATH.relative_to(REPO)),
            "problem_codec_sha256": problem_codec_sha256, "learner_split_sha256": learner_split_sha256,
            "held_out_items": list(HELD_OUT_ITEMS),
            "git_sha": git_sha, "seed": args.seed, "n_steps_requested": args.max_steps,
            "sampler_semantics": CURRENT_SAMPLER_SEMANTICS,
            "br4b_joint_set": bool(args.br4b_joint_set),
            "target_set_k": int(args.target_set_k),
        }
        local_path = save_checkpoint(state, out_dir, step=step, metric=None,
                                     is_best=is_final, keep_last=3)
        if args.bucket:
            import boto3
            key = f"{args.out_prefix}/checkpoints/{'final' if is_final else f'step{step:08d}'}.pt"
            boto3.client("s3").upload_file(str(local_path), args.bucket, key)
            print(f"[cw-sim] checkpoint uploaded to bucket {args.bucket}, key {key}", flush=True)

    start_step = 0
    if args.resume:
        if not has_checkpoint(out_dir) and args.bucket:
            import boto3
            _pull_latest_s3_checkpoint(boto3.client("s3"), args.bucket, args.out_prefix, out_dir)
        if not has_checkpoint(out_dir):
            print(f"[cw-sim] --resume given but no checkpoint found in {out_dir} or bucket {args.bucket}",
                 file=sys.stderr)
            return 1
        ckpt = load_checkpoint(out_dir, which="last", require_sampler_semantics=True)
        assert_resume_compatible(ckpt, architecture=CW5_DIMS, item_encoder=args.item_encoder,
                                 learner_split_sha256=learner_split_sha256,
                                 problem_codec_sha256=problem_codec_sha256,
                                 sigma_parameterization=args.sigma_parameterization)
        m["attempt_encoder"].load_state_dict(ckpt["attempt_encoder"])
        m["filt"].load_state_dict(ckpt["filt"])
        m["outcome_head"].load_state_dict(ckpt["outcome_head"])
        m["y_ever_head"].load_state_dict(ckpt["y_ever_head"])
        if args.br4b_joint_set:
            assert "teacher" in ckpt, "joint-set resume checkpoint missing teacher posterior"
            m["teacher"].load_state_dict(ckpt["teacher"])
        m["hyper"].load_state_dict(ckpt["hyper"])
        for name, basis in m["bases"].items():
            basis.load_state_dict(ckpt["basis"][name])
        optimizer.load_state_dict(ckpt["optimizer"])
        torch.set_rng_state(ckpt["torch_rng"])
        np.random.set_state(ckpt["numpy_rng"])
        random.setstate(ckpt["python_rng"])
        lambda_a = ckpt.get("lambda_a", lambda_a)
        start_step = ckpt["step"] + 1
        print(f"[cw-sim] resumed from step {ckpt['step']}, continuing at step {start_step}", flush=True)

    # Numeric-only provenance record.  Paths and SHA-256 digests are safe to retain;
    # none of the raw/vendor file contents are embedded.  This is written before the
    # long phase starts so a fuse/overrun exit still leaves a self-describing attempt.
    provenance_inputs = {
        "code_map": pathlib.Path(args.code_map),
        "diff_map": pathlib.Path(args.diff_map),
        "codealone_meta": pathlib.Path(args.codealone_meta),
        "codealone_states": pathlib.Path(args.codealone_states),
        "diff_meta": pathlib.Path(args.diff_meta),
        "diff_states": pathlib.Path(args.diff_states),
        "learner_split": REPO / "splits" / "cw_learner_split.csv",
        "problem_codec": PROBLEM_CODEC_DIM64_PATH,
        "codec_transform": REPO / args.codec_transform,
    }
    checkpoint_path = None
    checkpoint_index_path = out_dir / "checkpoints" / "index.json"
    if checkpoint_index_path.exists():
        checkpoint_index = json.loads(checkpoint_index_path.read_text())
        if checkpoint_index.get("history"):
            checkpoint_path = out_dir / "checkpoints" / checkpoint_index["history"][-1]["path"]
    run_manifest = {
        "task": "cw_5_b3_gen" if args.extended_generation_metrics else "cw_5_full_run",
        "git_sha": git_sha,
        "seed": args.seed,
        "item_encoder": args.item_encoder,
        "sampler_semantics": CURRENT_SAMPLER_SEMANTICS,
        "checkpoint_step": start_step - 1 if args.resume else None,
        "checkpoint_sha256": _sha256(checkpoint_path) if checkpoint_path and checkpoint_path.exists() else None,
        "input_sha256": {
            name: _sha256(path) for name, path in provenance_inputs.items() if path.is_file()
        },
        "generation": {
            "enabled": not args.skip_eval,
            "extended_numeric_metrics": args.extended_generation_metrics,
            "n_rows": args.n_generation_rows,
            "arms": list(ARMS),
            "modes": ["conditional_imitation", "ancestral"],
            "max_new_tokens": args.max_new_tokens,
            "codec_layer": args.codec_layer,
            "codec_strip_k": args.codec_strip_k,
            "codec_rank": args.codec_rank,
            "raw_text_uploaded": False,
        },
    }
    run_manifest_path = out_dir / "run_manifest.json"
    run_manifest_path.write_text(json.dumps(run_manifest, indent=2))
    if args.bucket:
        import boto3
        boto3.client("s3").upload_file(
            str(run_manifest_path), args.bucket, f"{args.out_prefix}/{run_manifest_path.name}")

    # ShuffledEpochSampler holds its own np.random.Generator (independent of the global
    # numpy RNG state restored above) -- deterministic given (seed, call sequence) alone,
    # so exact resume is a REPLAY: recreate it from the same seed and fast-forward past
    # `start_step` draws (discarded), rather than serialising its internal buffer/
    # bit-generator state. Cheap (pure numpy slicing, no filter/backbone forward) even
    # at thousands of steps.
    sampler = ShuffledEpochSampler(len(train_learners), np.random.default_rng(args.seed))
    for _ in range(start_step):
        sampler.next_batch(args.learner_batch_size)

    t0 = time.time()
    last_checkpoint_t = t0
    last_heartbeat_t = 0.0  # forces an immediate first heartbeat
    completed_steps = start_step
    first_lambda_probe_executed = False

    for step in range(start_step, args.max_steps):
        elapsed_min = (time.time() - t0) / 60.0
        if elapsed_min >= args.fuse_minutes:
            print(f"[cw-sim] FUSE: {elapsed_min:.1f} min >= --fuse-minutes={args.fuse_minutes}, stopping", flush=True)
            break

        batch_idx = sampler.next_batch(args.learner_batch_size)
        batch_learners = {train_learners[i] for i in batch_idx}
        attempts_batch = manifest["attempts"][manifest["attempts"]["learner"].isin(batch_learners)]
        episodes_batch = manifest["episodes"][manifest["episodes"]["learner"].isin(batch_learners)]
        if len(attempts_batch) == 0:
            continue

        if step > start_step and step % args.lambda_a_select_every == 0:
            probe_stats = {}
            lambda_a = run_lambda_a_probe(
                m, attempts_batch, episodes_batch, manifest, CW5_DIMS, device, optimizer,
                args.micro_batch_size, lambda_a_grid, probe_log=probe_stats)
            is_first_probe = not first_lambda_probe_executed
            if is_first_probe:
                assert_first_lambda_probe_has_production_gradient(probe_stats["grad_norm_prod"])
                first_lambda_probe_executed = True
            probe_record = {
                "step": step,
                "start_step": start_step,
                "first_executed_probe": is_first_probe,
                **probe_stats,
            }
            with open(lambda_probe_path, "a") as f:
                f.write(json.dumps(probe_record) + "\n")
            if args.bucket:
                import boto3
                boto3.client("s3").upload_file(
                    str(lambda_probe_path), args.bucket,
                    f"{args.out_prefix}/{lambda_probe_path.name}")
            if is_first_probe:
                print(f"[cw-sim] first valid lambda_A probe: lambda_A={lambda_a} "
                      f"(grad_norm_prod={probe_stats['grad_norm_prod']:.4g}, "
                      f"grad_norm_outcome={probe_stats['grad_norm_outcome']:.4g})", flush=True)

        epoch = step / n_batches_per_epoch
        kl_weight = kl_anneal_weight(epoch, args.kl_max_weight, args.kl_anneal_epochs)

        optimizer.zero_grad()
        out = forward_outcome(m, attempts_batch, episodes_batch, manifest, CW5_DIMS, device, args.free_bits)
        joint_sets = {"n_sets": 0}
        if args.br4b_joint_set:
            joint_sets = forward_joint_set(
                m, attempts_batch, episodes_batch, out, manifest, CW5_DIMS, device,
                target_rng=np.random.default_rng(args.seed + 50000 + step),
                k=args.target_set_k, min_targets=args.target_set_min, max_targets=args.target_set_max)
            out["l_outcome"] = joint_sets["l_joint"]
            out["kl"] = joint_sets["kl_teacher"]
        teacher_weight = kl_anneal_weight(
            epoch, args.teacher_kl_weight if args.br4b_joint_set else args.kl_max_weight,
            args.kl_anneal_epochs)
        outcome_total = out["l_outcome"] + out["l_y_ever"] + teacher_weight * out["kl"]
        if not torch.isfinite(outcome_total):
            raise RuntimeError(f"[cw-sim] TRIPWIRE non-finite outcome loss at step={step}: "
                               f"l_outcome={out['l_outcome'].item()} l_y_ever={out['l_y_ever'].item()} "
                               f"kl={out['kl'].item()}")
        # retain_graph=True: the production path's OWN separate filter-forward
        # (production_backward_microbatched -> gather_belief_for_batch) still needs
        # ep_batch's upstream graph (through attempt_encoder) after this call -- freeing
        # it here would break the second, independent filt_forward call downstream.
        # The LAST production chunk's backward (retain_graph=False) frees everything.
        outcome_total.backward(retain_graph=True)

        ep_learner_idx, ep_t_idx = _distinct_episode_positions(
            episodes_batch, out["ep_batch"], out["attempt_batch"]["episode_ids"], device=device)
        l_prod = production_backward_microbatched(
            m, attempts_batch, manifest, out["ep_batch"], ep_learner_idx, ep_t_idx,
            out["attempt_batch"]["attempt_episode_row_idx"], device, k=1,
            micro_batch_size=args.micro_batch_size, scale=lambda_a)
        if not np.isfinite(l_prod):
            raise RuntimeError(f"[cw-sim] TRIPWIRE non-finite production loss at step={step}: {l_prod}")

        non_finite_grad = [name for name, params in [
            ("attempt_encoder", list(m["attempt_encoder"].parameters())),
            ("filt", list(m["filt"].parameters())),
            ("outcome_head", list(m["outcome_head"].parameters())),
            ("y_ever_head", list(m["y_ever_head"].parameters())),
            ("hyper", list(m["hyper"].parameters())),
            ("bases", [p for b in m["bases"].values() for p in b.parameters()])]
            if any(param.grad is not None and not torch.isfinite(param.grad).all() for param in params)]
        if non_finite_grad:
            raise RuntimeError(f"[cw-sim] TRIPWIRE non-finite gradient at step={step} in: {non_finite_grad}")

        torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
        optimizer.step()
        completed_steps = step + 1

        record = {"step": step, "elapsed_s": time.time() - t0, "l_outcome": out["l_outcome"].item(),
                 "l_y_ever": out["l_y_ever"].item(), "l_prod": l_prod,
                 "kl": out["kl"].item(), "kl_weight": teacher_weight, "lambda_a": lambda_a,
                 "joint_set": bool(args.br4b_joint_set), "n_target_sets": int(joint_sets["n_sets"]),
                 "n_rows": len(attempts_batch)}
        with open(metrics_path, "a") as f:
            f.write(json.dumps(record) + "\n")

        now = time.time()
        if now - last_heartbeat_t >= args.heartbeat_every_s:
            heartbeat_path.write_text(json.dumps({"step": step, "t": now, "phase": "train",
                                                   "max_steps": args.max_steps}))
            last_heartbeat_t = now
            # Small reconnect-safe monitor record: an unattended run is monitored by
            # polling this JSON from the bucket, never the SSH stream or bulk artifacts.
            if args.bucket:
                import boto3
                boto3.client("s3").upload_file(
                    str(heartbeat_path), args.bucket, f"{args.out_prefix}/heartbeat.json")
            print(f"[cw-sim] step={step}/{args.max_steps} l_outcome={out['l_outcome'].item():.4f} "
                 f"l_prod={l_prod:.4f} kl={out['kl'].item():.4f} kl_weight={kl_weight:.4g}", flush=True)
            if train_trigger.check(units_done=(step - start_step + 1),
                                   units_total=(args.max_steps - start_step),
                                   elapsed_s=now - t0):
                checkpoint_and_upload(step, is_final=True)
                return 1

        if (time.time() - last_checkpoint_t) >= args.checkpoint_every_s:
            checkpoint_and_upload(step, is_final=False)
            last_checkpoint_t = time.time()

    exposure = sampler.exposure_counts
    print(f"[cw-sim] training done at step={completed_steps}/{args.max_steps} (fuse or completion). "
         f"learner exposure counts: min={exposure.min()} max={exposure.max()} mean={exposure.mean():.2f}",
         flush=True)

    final_step = max(completed_steps - 1, 0)
    checkpoint_and_upload(final_step, is_final=True)
    print(f"[cw-sim] final checkpoint saved under {out_dir}/checkpoints/ ({completed_steps} steps "
         f"completed of {args.max_steps} requested)", flush=True)

    if not args.skip_eval:
        # EVAL PASS (the smoke test runs the same code against a stub backbone): 4-arm
        # scoring on a VAL-learner subset, the uncertainty-survives gate, the
        # K=1/4/8/16 stability curve, the row-identity join canary, and the licence-split
        # persistence writers -- in the same run, not a separate eval job.
        eval_t0 = time.time()
        val_learners = sorted(manifest["val_learners"])
        eval_learners = set(val_learners[:args.eval_val_learners])
        eval_attempts_batch = manifest["attempts"][manifest["attempts"]["learner"].isin(eval_learners)]
        eval_episode_ids = set(eval_attempts_batch["episode_id"])
        eval_episodes_batch = manifest["episodes"][manifest["episodes"]["episode_id"].isin(eval_episode_ids)]
        print(f"[cw-sim] eval batch: {len(eval_learners)} val learners, "
             f"{len(eval_attempts_batch)} attempts, {len(eval_episodes_batch)} episodes", flush=True)

        # Common random numbers (as for FA-SIM3): one shared eps drawn ONCE,
        # reused identically across every arm below AND across the K-stability curve, so
        # paired-arm comparisons (personalised-vs-shuffled_u, personalised-vs-dw0) are not
        # spending power on avoidable Monte-Carlo noise. Sized to len(eval_episodes_batch)
        # episode rows -- score_arm_for_batch's own assertion catches a mismatch loudly if
        # that count turns out not to match `gather_belief_for_batch`'s actual output.
        crn_generator = torch.Generator(device="cpu").manual_seed(args.seed + 200)
        shared_eps = torch.randn(len(eval_episodes_batch), args.eval_k, CW5_DIMS["latent_dim"],
                                 generator=crn_generator)

        eval_rng = np.random.default_rng(args.seed + 1000)
        scores_by_arm = {}
        for arm in ARMS:
            resumed = try_resume_arm_scores(out_dir, arm, args.bucket, args.out_prefix) if args.resume else None
            # A saved arm is only valid to reuse if `kept_row_idx` (a POSITIONAL index
            # into THIS run's own eval_attempts_batch) is still in range -- an arm saved
            # under a LARGER --eval-val-learners population would silently mismatch a
            # smaller one here (write_predictions_npz's own `attempts_batch.iloc[kept]`
            # would then read the wrong rows without erroring: plausible size, wrong
            # membership). Discard and rescore fresh rather than trust a
            # population-mismatched cache.
            if resumed is not None and len(resumed["kept_row_idx"]) > 0:
                max_idx = int(resumed["kept_row_idx"].max())
                if max_idx >= len(eval_attempts_batch):
                    print(f"[cw-sim] eval arm={arm}: resumed scores reference row {max_idx} but this "
                         f"run's eval batch only has {len(eval_attempts_batch)} rows -- population "
                         "mismatch (saved under a different --eval-val-learners), discarding and "
                         "rescoring fresh rather than trusting a mismatched cache.", flush=True)
                    resumed = None
            if resumed is not None:
                scores_by_arm[arm] = resumed
                print(f"[cw-sim] eval arm={arm}: resumed from a prior checkpoint, skipping rescoring", flush=True)
            else:
                scores_by_arm[arm] = score_arm_for_batch(
                    m, eval_attempts_batch, eval_episodes_batch, manifest, CW5_DIMS, device,
                    k=args.eval_k, micro_batch_size=args.micro_batch_size, arm=arm, rng=eval_rng,
                    eps=shared_eps)
                save_arm_scores(scores_by_arm[arm], out_dir, arm, args.bucket, args.out_prefix)
            s = scores_by_arm[arm]
            n_scored = len(s["mixture_nll"])
            rate = s["elapsed_s"] / n_scored if n_scored else float("nan")
            print(f"[cw-sim] eval arm={arm} n_rows_scored={n_scored} "
                 f"n_dropped_budget={s['n_dropped_budget']} n_dropped_derangement={s['n_dropped_derangement']} "
                 f"mixture_nll_mean={float(s['mixture_nll'].mean()) if n_scored else float('nan'):.4f} "
                 f"n_forward_calls={s['n_forward_calls']} elapsed_s={s['elapsed_s']:.2f} "
                 f"({rate:.4f} s/row)", flush=True)
            eval_units_total = len(ARMS) if args.skip_k_curve else len(ARMS) + 4
            if eval_trigger.check(units_done=list(ARMS).index(arm) + 1, units_total=eval_units_total,
                                  elapsed_s=time.time() - eval_t0):
                return 1

        gate = uncertainty_survives_gate(scores_by_arm["personalised"])
        print(f"[cw-sim] uncertainty_survives_gate (personalised): {gate}", flush=True)

        if args.skip_k_curve:
            kcurve = []
            print("[cw-sim] --skip-k-curve: K=1/4/8/16 stability curve SKIPPED (measured to "
                 "converge by K=8; re-sweeping it is the dominant eval cost)", flush=True)
        else:
            # K-stability curve: per-K resumability mirrors the per-arm loop above, via a
            # small JSON progress file (the curve's own results are a handful of scalars
            # per K, not worth npz).
            kcurve_state_path = out_dir / "eval_partial" / "k_stability_curve.json"
            kcurve_resume_state = None
            if args.resume:
                if not kcurve_state_path.exists() and args.bucket:
                    import boto3
                    try:
                        kcurve_state_path.parent.mkdir(parents=True, exist_ok=True)
                        boto3.client("s3").download_file(
                            args.bucket, f"{args.out_prefix}/eval_partial/{kcurve_state_path.name}",
                            str(kcurve_state_path))
                    except Exception:
                        pass
                if kcurve_state_path.exists():
                    kcurve_resume_state = json.loads(kcurve_state_path.read_text())

            def _on_k_done(k, result):
                state = json.loads(kcurve_state_path.read_text()) if kcurve_state_path.exists() else {}
                state[str(k)] = result
                kcurve_state_path.parent.mkdir(parents=True, exist_ok=True)
                kcurve_state_path.write_text(json.dumps(state))
                if args.bucket:
                    import boto3
                    boto3.client("s3").upload_file(str(kcurve_state_path), args.bucket,
                                                   f"{args.out_prefix}/eval_partial/{kcurve_state_path.name}")
                print(f"[cw-sim] k_stability_curve: K={k} persisted", flush=True)
                if eval_trigger.check(units_done=len(ARMS) + list((1, 4, 8, 16)).index(k) + 1,
                                      units_total=len(ARMS) + 4, elapsed_s=time.time() - eval_t0):
                    raise SystemExit(1)  # unwind out of k_stability_curve; trigger already terminated the instance

            try:
                kcurve = k_stability_curve(m, eval_attempts_batch, eval_episodes_batch, manifest, CW5_DIMS,
                                           device, args.micro_batch_size, arm="personalised",
                                           ks=(1, 4, 8, 16), rng=eval_rng, eps=shared_eps, eps_seed=args.seed + 200,
                                           latent_dim=CW5_DIMS["latent_dim"], resume_state=kcurve_resume_state,
                                           on_k_done=_on_k_done)
            except SystemExit as e:
                return e.code
            print(f"[cw-sim] k_stability_curve (personalised, K=1,4,8,16): {kcurve}", flush=True)

        # Row-identity join canary, run against the real backbone: re-join a sampled
        # subset of every arm's persisted rows against the SOURCE attempts table by
        # value, not by position.
        #
        # Two details matter. (1) `identity` comes from the tested `row_identity()`
        # helper, not an inline dict: `assert_row_identity_join_canary` reads
        # `attempt_id`/`prior_same_problem_count`/`split_flag` unconditionally. (2) `df`
        # and `identity` must be the SAME row set in the SAME order (`rows`, the arm's
        # KEPT SUBSET): the canary samples positions from `len(df)` and reads
        # `identity[...][pos]`, so passing the full batch against a subset's identity
        # would either raise an IndexError or, worse, silently verify the wrong rows.
        for arm, s in scores_by_arm.items():
            kept = s["kept_row_idx"]
            if len(kept) == 0:
                continue
            rows = eval_attempts_batch.iloc[kept]
            identity = row_identity(rows, split_flag="val")
            assert_row_identity_join_canary(rows, identity)
        print("[cw-sim] row-identity join canary verified, all arms", flush=True)

        predictions_path = out_dir / "predictions.npz"
        write_predictions_npz(scores_by_arm, eval_attempts_batch, predictions_path)
        generations_path = out_dir / "generations.jsonl"
        write_generations_jsonl(eval_attempts_batch, manifest,
                                scores_by_arm["personalised"]["kept_row_idx"], generations_path)
        print(f"[cw-sim] persistence written: {predictions_path} ({predictions_path.stat().st_size} bytes, "
             f"safe to upload) + {generations_path} ({generations_path.stat().st_size} bytes, "
             "licensed text, local only -- NEVER uploaded below)", flush=True)

        ancestral_generations_path = out_dir / "ancestral_generations.jsonl"
        if args.extended_generation_metrics:
            # Extended generation metrics.  Every completed source row becomes eight numeric score
            # records and is uploaded immediately.  The corresponding text is appended
            # locally only and is explicitly expendable.
            generation_partial_dir = out_dir / "generation_partial"
            generation_partial_dir.mkdir(parents=True, exist_ok=True)
            generation_scores_partial = generation_partial_dir / "generation_scores_partial.npz"
            generation_progress_path = generation_partial_dir / "progress.json"
            generation_heartbeat_path = generation_partial_dir / "heartbeat.json"
            generation_summary_path = out_dir / "generation_summary.json"
            generation_scores_path = out_dir / "generation_scores.npz"
            s3 = None
            if args.bucket:
                import boto3
                s3 = boto3.client("s3")
            if args.resume and not generation_scores_partial.exists() and s3 is not None:
                try:
                    s3.download_file(
                        args.bucket,
                        f"{args.out_prefix}/generation_partial/{generation_scores_partial.name}",
                        str(generation_scores_partial),
                    )
                    print("[cw-sim] generation resume: pulled numeric partial from the bucket", flush=True)
                except Exception:
                    pass
            numeric_generation_rows = (
                load_numeric_generation_rows(generation_scores_partial)
                if generation_scores_partial.exists() else []
            )
            completed_generation = completed_row_indices(numeric_generation_rows)
            n_gen_target = min(
                args.n_generation_rows,
                len(scores_by_arm["personalised"]["kept_row_idx"]),
            )
            n_resumed = len(completed_generation)
            if n_resumed:
                print(f"[cw-sim] generation resume: {n_resumed}/{n_gen_target} source rows "
                      "already complete; raw text is not pulled because it is expendable", flush=True)

            codec_transform = CodecTransform.load(REPO / args.codec_transform)
            assert codec_transform.layer == args.codec_layer, (
                f"codec transform layer={codec_transform.layer}, requested {args.codec_layer}")
            real_codec_cache: dict = {}
            generation_t0 = time.time()
            generation_state = {
                "n_done": n_resumed,
                "n_total": n_gen_target,
                "last_heartbeat": 0.0,
                "fuse_minutes": int(args.fuse_minutes),
            }

            def _generation_heartbeat(force: bool = False) -> None:
                now = time.time()
                if not force and now - generation_state["last_heartbeat"] < args.heartbeat_every_s:
                    return
                session_done = max(0, generation_state["n_done"] - n_resumed)
                remaining_rows = max(0, generation_state["n_total"] - generation_state["n_done"])
                if session_done > 0:
                    seconds_per_row = (now - generation_t0) / session_done
                    projected_remaining_s = seconds_per_row * remaining_rows
                else:
                    seconds_per_row = None
                    projected_remaining_s = args.estimated_minutes * 60.0
                fuse_minutes = max(60, int(math.ceil(2.0 * projected_remaining_s / 60.0)))
                subprocess.run(["sudo", "shutdown", "-c"], check=False,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                subprocess.run(["sudo", "shutdown", "-h", f"+{fuse_minutes}"], check=False,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                generation_state["last_heartbeat"] = now
                generation_state["fuse_minutes"] = fuse_minutes
                heartbeat = {
                    "phase": "b3_generation",
                    "t": now,
                    "n_rows_done": generation_state["n_done"],
                    "n_rows_total": generation_state["n_total"],
                    "session_seconds_per_row": seconds_per_row,
                    "projected_remaining_s": projected_remaining_s,
                    "fuse_rearmed_minutes": fuse_minutes,
                }
                generation_heartbeat_path.write_text(json.dumps(heartbeat, indent=2))
                if s3 is not None:
                    s3.upload_file(
                        str(generation_heartbeat_path), args.bucket,
                        f"{args.out_prefix}/generation_partial/{generation_heartbeat_path.name}")
                print(f"[cw-sim] generation heartbeat: {generation_state['n_done']}/"
                      f"{generation_state['n_total']} rows; fuse re-armed +{fuse_minutes} min",
                      flush=True)

            def _upload_numeric_generation_partial() -> None:
                save_numeric_generation_rows(numeric_generation_rows, generation_scores_partial)
                progress = {
                    "phase": "b3_generation",
                    "n_numeric_records": len(numeric_generation_rows),
                    "n_completed_rows": len(completed_row_indices(numeric_generation_rows)),
                    "n_target_rows": n_gen_target,
                    "elapsed_s_this_session": time.time() - generation_t0,
                    "raw_text_uploaded": False,
                }
                generation_progress_path.write_text(json.dumps(progress, indent=2))
                if s3 is not None:
                    s3.upload_file(
                        str(generation_scores_partial), args.bucket,
                        f"{args.out_prefix}/generation_partial/{generation_scores_partial.name}")
                    s3.upload_file(
                        str(generation_progress_path), args.bucket,
                        f"{args.out_prefix}/generation_partial/{generation_progress_path.name}")

            raw_generation_file = open(ancestral_generations_path, "a")

            def _on_generation_row(n_done: int, n_total: int, elapsed_s: float,
                                   row_records: list[dict]) -> None:
                for record in row_records:
                    raw_generation_file.write(json.dumps(record) + "\n")
                raw_generation_file.flush()
                for record in row_records:
                    similarity = generation_codec_similarity_one(
                        m, codec_transform, device, record, real_codec_cache,
                        layer=args.codec_layer, k_strip=args.codec_strip_k,
                        rank=args.codec_rank, on_forward=_generation_heartbeat,
                    )
                    numeric_generation_rows.append(numeric_generation_row(record, similarity))
                generation_state["n_done"] = n_done
                if (n_done % args.generation_flush_rows == 0) or n_done == n_total:
                    _upload_numeric_generation_partial()
                _generation_heartbeat(force=True)
                session_done = n_done - n_resumed
                session_total = n_total - n_resumed
                if generation_trigger.check(
                        session_done, session_total, time.time() - generation_t0):
                    raise SystemExit(1)

            _generation_heartbeat(force=True)
            try:
                ancestral_records = run_ancestral_generation_pass(
                    m, eval_attempts_batch, manifest, scores_by_arm, device,
                    n_generation_rows=args.n_generation_rows,
                    max_new_tokens=args.max_new_tokens,
                    seed=args.seed,
                    on_progress=_on_generation_row,
                    completed_row_idx=completed_generation,
                    on_token=_generation_heartbeat,
                )
            except SystemExit as e:
                raw_generation_file.close()
                return e.code
            raw_generation_file.close()
            _upload_numeric_generation_partial()
            save_numeric_generation_rows(numeric_generation_rows, generation_scores_path)
            ancestral_metrics = summarise_generation_rows(
                numeric_generation_rows, n_boot=args.generation_n_boot, seed=args.seed)
            ancestral_metrics.update({
                "git_sha": git_sha,
                "checkpoint_source": "cw-5-b3-full-run/checkpoints/final.pt",
                "n_requested_generation_rows": args.n_generation_rows,
                "n_arms": len(ARMS),
                "n_modes": 2,
                "max_new_tokens": args.max_new_tokens,
                "codec_transform": args.codec_transform,
                "codec_layer": args.codec_layer,
                "codec_strip_k": args.codec_strip_k,
                "codec_rank": args.codec_rank,
                "elapsed_s_this_session": time.time() - generation_t0,
                "raw_text_uploaded": False,
            })
            generation_summary_path.write_text(json.dumps(ancestral_metrics, indent=2))
            if s3 is not None:
                s3.upload_file(str(generation_scores_path), args.bucket,
                               f"{args.out_prefix}/{generation_scores_path.name}")
                s3.upload_file(str(generation_summary_path), args.bucket,
                               f"{args.out_prefix}/{generation_summary_path.name}")
            print(f"[cw-sim] generation metrics complete: "
                  f"{ancestral_metrics['n_completed_generation_rows']} rows x 4 arms x 2 modes; "
                  "numeric scores/summary retained, raw text LOCAL ONLY", flush=True)
        else:
            # Small diagnostic generation path (exact-match and length only).
            ancestral_records = run_ancestral_generation_pass(
                m, eval_attempts_batch, manifest, scores_by_arm, device,
                n_generation_rows=args.n_generation_rows,
                max_new_tokens=args.max_new_tokens, seed=args.seed)
            with open(ancestral_generations_path, "w") as f:
                for r in ancestral_records:
                    f.write(json.dumps(r) + "\n")
            ancestral_metrics = ancestral_generation_metrics(ancestral_records)
            print(f"[cw-sim] ancestral generation: {len(ancestral_records)} records written to "
                  f"{ancestral_generations_path} (licensed text, local only -- NEVER uploaded). "
                  f"metrics={ancestral_metrics}", flush=True)

        eval_summary = {
            "n_eval_learners": len(eval_learners), "eval_k": args.eval_k,
            "eval_elapsed_s": time.time() - eval_t0,
            "uncertainty_survives_gate": gate, "k_stability_curve": kcurve,
            "ancestral_generation_metrics": ancestral_metrics,
            "arms": {arm: {"n_rows_scored": int(len(s["mixture_nll"])),
                          "mixture_nll_mean": float(s["mixture_nll"].mean()) if len(s["mixture_nll"]) else None,
                          "elapsed_s": s["elapsed_s"], "n_forward_calls": s["n_forward_calls"],
                          "n_dropped_budget": int(s["n_dropped_budget"]),
                          "n_dropped_derangement": int(s["n_dropped_derangement"])}
                     for arm, s in scores_by_arm.items()},
        }
        eval_summary_path = out_dir / "eval_summary.json"
        eval_summary_path.write_text(json.dumps(eval_summary, indent=2))
        if args.bucket:
            import boto3
            s3 = boto3.client("s3")
            s3.upload_file(str(predictions_path), args.bucket, f"{args.out_prefix}/predictions.npz")
            s3.upload_file(str(eval_summary_path), args.bucket, f"{args.out_prefix}/eval_summary.json")
            print(f"[cw-sim] predictions.npz + eval_summary.json uploaded to bucket {args.bucket}, "
                 f"prefix {args.out_prefix}/ -- generations.jsonl intentionally NOT uploaded "
                 "(licensed text); copy it over SSH before terminating.", flush=True)
        print(f"[cw-sim] eval pass done, elapsed_s={time.time()-eval_t0:.2f}", flush=True)

    if args.bucket:
        import boto3
        final_s3 = boto3.client("s3")
        for final_path in (run_manifest_path, metrics_path, lambda_probe_path, heartbeat_path):
            if final_path.exists():
                final_s3.upload_file(str(final_path), args.bucket,
                                     f"{args.out_prefix}/{final_path.name}")
        if args.run_log_path:
            run_log_path = pathlib.Path(args.run_log_path)
            if run_log_path.exists():
                final_s3.upload_file(str(run_log_path), args.bucket,
                                     f"{args.out_prefix}/run.log")
        print("[cw-sim] final manifest/metrics/heartbeat/log upload complete; raw generation "
              "text remains local-only", flush=True)

    if args.self_terminate:
        print("[cw-sim] --self-terminate: shutting down now.", flush=True)
        subprocess.run(["sudo", "shutdown", "-h", "now"], check=False)
    else:
        print("[cw-sim] --self-terminate not set -- machine stays up under the boot fuse "
             f"({args.fuse_minutes} min from launch) until manually terminated, so that "
             "generations.jsonl (licensed text, never uploaded) can be copied over SSH "
             "first.", flush=True)

    return 0


def _pull_latest_s3_checkpoint(s3_client, bucket: str, out_prefix: str, out_dir: pathlib.Path) -> None:
    """`--resume` on a fresh machine (not just a same-process kill/restart): `out_dir`
    (ephemeral local storage) has nothing yet, so pull the newest checkpoint down from
    the bucket first and write a matching local checkpoint index.

    `s3_client` is caller-supplied (the same convention as `fa_06_stage2_train.py::
    try_download_s3_checkpoint`) rather than constructed internally, so the function can
    be unit-tested with a fake client and without `boto3` installed.

    Key naming: `checkpoint_and_upload` (this file) uploads under `{'final' if is_final
    else f'step{step:08d}'}.pt` (no `ckpt_` prefix, and `final.pt` carries no step number
    at all), while `atom.utils.checkpoints.save_checkpoint` names the LOCAL file
    `ckpt_step{step:08d}.pt`. This function therefore matches the uploaded key names,
    prefers `final.pt` when present (it is the authoritative end-of-run checkpoint), else
    the highest-numbered `step*.pt`. The local index only needs `path` to point at
    whatever filename was actually downloaded (`load_checkpoint`'s `which="last"` reads
    `index["history"][-1]["path"]` directly, no naming convention assumed) plus the
    correct `step`, read from the checkpoint's own payload (`save_checkpoint`'s
    `payload = {"step": step, ...}`) rather than parsed from a filename that -- for
    `final.pt` -- does not contain it.
    """
    prefix = f"{out_prefix}/checkpoints/"
    paginator = s3_client.get_paginator("list_objects_v2")
    keys = [obj["Key"] for page in paginator.paginate(Bucket=bucket, Prefix=prefix)
           for obj in page.get("Contents", [])]
    pt_keys = [k for k in keys if k.endswith(".pt")]
    if not pt_keys:
        print(f"[cw-sim] --resume: no checkpoints found in bucket {bucket} under {prefix}", flush=True)
        return
    final_keys = [k for k in pt_keys if pathlib.Path(k).name == "final.pt"]
    step_keys = sorted((k for k in pt_keys if pathlib.Path(k).stem.startswith("step")),
                       key=lambda k: pathlib.Path(k).stem)
    latest_key = final_keys[0] if final_keys else step_keys[-1]
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    local_path = ckpt_dir / pathlib.Path(latest_key).name
    s3_client.download_file(bucket, latest_key, str(local_path))
    payload_step = int(torch.load(local_path, map_location="cpu", weights_only=False)["step"])
    index = {"history": [{"step": payload_step, "metric": None, "path": local_path.name}], "best": None}
    (ckpt_dir / "index.json").write_text(json.dumps(index, indent=2))
    print(f"[cw-sim] --resume: pulled {local_path.name} (step={payload_step}) from bucket {bucket}, key {latest_key}",
         flush=True)
