"""Shared helpers for `sim2real_w_prepare_events` and `sim2real_w_run` (§8.3, the policy trained
only inside the simulator): loading and validating the numeric train handoff of the CodeWorkout
generative simulator, the frozen CW item codec, encoding generated code into the frozen 64-D
code codec and attempt summary, and the simulator-only pilot/training loop around the
`rl_2.py` pointer actor.
"""
from __future__ import annotations
import copy
import hashlib
import json
import pathlib
import sys
from typing import Any, Mapping, Sequence
import numpy as np
import torch
REPO = pathlib.Path(__file__).resolve().parents[1]
for p in (REPO / "src", REPO / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from rl_2 import (  # noqa: E402
    PointerActorV2,
    _candidate_features,
    _dlogit,
    _frozen_filter_state,
    _kl,
    _point_logits,
    _state,
    _trace_policy_v2,
)
from imagination_core import Sim2RealConfig, train_pure_simulator
from atom.codec.extract import encode_one  # noqa: E402
from atom.codec.transform import CodecTransform  # noqa: E402
from atom.filter.cw_attempt_encoder import build_scalar_feats  # noqa: E402
REGISTERED_CHECKPOINT_SHA256 = "5337abbbad40e0d1474ddce1e3bdc69cd1159a94e29722cedef70d8c56d7a56d"
HANDOFF_SHA256 = "5a809c709109866750c43c12900ef3e719eb330b04f6853d7040536398393a38"
ITEM_CODEC = REPO / "artifacts/cw_item_codec_dim64.npz"


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_train_handoff(path: pathlib.Path, expected_sha: str = HANDOFF_SHA256) -> dict[str, Any]:
    """Load only the numeric train handoff and check train-split membership by value."""
    got = sha256_file(path)
    if expected_sha and got != expected_sha:
        raise ValueError(f"train handoff SHA mismatch: expected {expected_sha}, got {got}")
    z = np.load(path, allow_pickle=False)
    required = {"learner", "episode_id", "mu", "log_sigma", "release", "split"}
    if not required.issubset(z.files):
        raise ValueError(f"train handoff missing {sorted(required - set(z.files))}")
    split = np.asarray(z["split"]).astype(str)
    if not np.all(split == "train"):
        raise AssertionError("simulator-only train path received a non-train handoff row")
    if "checkpoint_sha256" not in z.files:
        raise ValueError("train handoff is missing checkpoint_sha256 provenance")
    checkpoint_sha = str(np.asarray(z["checkpoint_sha256"]).reshape(-1)[0])
    if checkpoint_sha != REGISTERED_CHECKPOINT_SHA256:
        raise ValueError(
            "train handoff checkpoint SHA does not match the registered refreshed-head "
            f"checkpoint: expected {REGISTERED_CHECKPOINT_SHA256}, got {checkpoint_sha}"
        )
    mu, ls = np.asarray(z["mu"], dtype=np.float32), np.asarray(z["log_sigma"], dtype=np.float32)
    if (
        mu.ndim != 2
        or ls.shape != mu.shape
        or not np.isfinite(mu).all()
        or not np.isfinite(ls).all()
    ):
        raise ValueError("train handoff mu/log_sigma shape or finiteness check failed")
    learners = np.asarray(z["learner"]).astype(str)
    if len(set(learners)) != 686:
        raise ValueError(f"expected 686 train learners by value, got {len(set(learners))}")
    # The dense exporter is sorted by learner/episode; the last row is the full-history
    # state.  This is a value join, never a positional join to a val artifact.
    last: dict[str, int] = {}
    for i, learner in enumerate(learners):
        last[learner] = i
    return {
        "sha256": got,
        "checkpoint_sha256": checkpoint_sha,
        "learner": learners,
        "episode_id": np.asarray(z["episode_id"]).astype(str),
        "mu": mu,
        "log_sigma": ls,
        "release": np.asarray(z["release"]).astype(str),
        "split": split,
        "last_index": last,
    }


def item_panel() -> tuple[dict[int, dict[str, Any]], np.ndarray]:
    statements = json.loads((REPO / "artifacts/cw_problem_statements.json").read_text())
    by_pid = {int(row["problem_id"]): row for row in statements}
    z = np.load(ITEM_CODEC, allow_pickle=True)
    codecs = np.asarray(z["codec"], dtype=np.float32)
    pids = np.asarray(z["problem_id"], dtype=np.int64)
    if codecs.ndim != 2 or codecs.shape[1] != 64 or len(pids) != len(codecs):
        raise ValueError("CW item codec artifact is not [N,64]")
    return by_pid, codecs[np.argsort(pids)]


def encode_corpus_records(
    records: Sequence[Mapping[str, Any]], generation_model: Mapping[str, Any], device: str
) -> dict[tuple[str, int, int], torch.Tensor]:
    """Encode generated code into the frozen 64-D code codec and AttemptEncoder summary."""
    transform = CodecTransform.load(REPO / "transforms/problem_mean_L15.npz")
    bundle = {
        "model": generation_model["model"],
        "tokenizer": generation_model["tok"],
        "device": device,
    }
    vec_of_code: dict[str, torch.Tensor] = {}
    # The CW backbone is LoRA-hooked and refuses an unset mixing context.  Codec
    # extraction is the registered neutral ruler (the same convention as the replay
    # generation scorer), so explicitly set the adapter vector to zero before every
    # frozen content forward.  This also prevents a stale per-learner generation vector
    # from leaking into the code representation.
    context = generation_model.get("context")
    hyper = generation_model.get("hyper")
    mixing_dim = getattr(hyper, "m", None)
    if context is None or mixing_dim is None:
        raise ValueError("generation model is missing the hooked context/hyper dimension")
    context.set(torch.zeros(1, int(mixing_dim), dtype=torch.float32, device=device))
    for row in records:
        code = str(row["generated_code"])
        if code not in vec_of_code:
            # Stream the large hidden-state tensor through the frozen transform and
            # retain only the 64-D codec vector.  Keeping every [layers, hidden] array
            # until a final np.stack exceeds 30 GiB of host memory on the full
            # 22k-row corpus.
            _, mean = encode_one(bundle, "", code)
            vector = transform.apply(
                np.asarray(mean[transform.layer], dtype=np.float64)[None, :], k=5, rank=69
            )[0].astype(np.float32)
            vec_of_code[code] = torch.as_tensor(vector, dtype=torch.float32, device=device)
    out: dict[tuple[str, int, int], torch.Tensor] = {}
    enc = generation_model["attempt_encoder"]
    enc.eval()
    for row in records:
        code_vec = vec_of_code[str(row["generated_code"])].view(1, 1, -1)
        diff_vec = torch.zeros_like(code_vec)
        c = int(row["conditioned_class"])
        scalar = build_scalar_feats(
            torch.tensor([[float(c != 0)]], device=device),
            torch.tensor([[float([0.0, 0.0, 0.5, 1.0][c])]], device=device),
            torch.tensor([[float(c == 0)]], device=device),
            torch.zeros((1, 1), device=device),
        )
        with torch.no_grad():
            _, summary = enc(
                code_vec, diff_vec, scalar, torch.ones(1, dtype=torch.long, device=device)
            )
        out[(str(row["learner"]), int(row["problem_id"]), c)] = summary[0].detach()
    return out


def train_sim_actor(
    *,
    model: Mapping[str, Any],
    train_eps: Sequence[dict[str, Any]],
    val_eps: Sequence[dict[str, Any]],
    seed: int,
    cfg: Sim2RealConfig,
    device: str,
    out_dir: pathlib.Path,
) -> tuple[PointerActorV2, dict[str, Any]]:
    if not train_eps or not val_eps:
        raise ValueError("simulator pilot needs non-empty train and simulated-validation episodes")
    actor = PointerActorV2(64).to(device)
    critic = __import__("rl_2", fromlist=["PrivilegedCriticV2"]).PrivilegedCriticV2().to(device)
    opt = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=3e-4)
    train_rows = [
        {
            "source": "simulator",
            "generated_from": "b_full",
            "learner": e["learner"],
            "episode_id": i,
        }
        for i, e in enumerate(train_eps)
    ]
    val_rows = [
        {
            "source": "simulated_validation",
            "split": "sim_val",
            "generated_from": "b_full",
            "learner": e["learner"],
            "episode_id": i,
        }
        for i, e in enumerate(val_eps)
    ]
    rng = np.random.default_rng(seed)

    def fit_epoch(_seed: int, epoch: int, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        del rows
        actor.train()
        critic.train()
        losses: list[float] = []
        for budget in cfg.budgets:
            opt.zero_grad(set_to_none=True)
            chosen_eps = [
                train_eps[i]
                for i in rng.permutation(len(train_eps))[
                    : max(1, len(train_eps) // len(cfg.budgets))
                ]
            ]
            for ei, ep in enumerate(chosen_eps):
                bfull = ep["b_full_mu"]
                full_logits = _point_logits(
                    model, bfull, torch.stack([x["codec"] for x in ep["targets"]]), device
                )
                # The warm-start variant supplies a real chronological warm prefix; the
                # cold-start episodes leave this empty, so that arm is unchanged.
                known: list[dict[str, Any]] = list(ep.get("prefix_events", []))
                remaining = list(range(len(ep["candidates"])))
                logps: list[torch.Tensor] = []
                vals: list[torch.Tensor] = []
                rewards: list[float] = []
                eps = torch.randn(
                    8,
                    32,
                    generator=torch.Generator(device="cpu").manual_seed(seed + epoch * 10000 + ei),
                ).to(device)
                for step in range(min(int(budget), len(remaining))):
                    mu, ls = _frozen_filter_state(model, known, ep["targets"][0]["codec"], device)
                    feats, _ = _candidate_features(
                        model,
                        mu,
                        ls,
                        [ep["candidates"][i] for i in remaining],
                        ep["targets"],
                        eps,
                        device,
                    )
                    cc = torch.stack([ep["candidates"][i]["codec"] for i in remaining]).to(device)
                    dist = torch.distributions.Categorical(
                        logits=actor(
                            _state(mu, ls, step, budget, len(remaining)),
                            cc,
                            torch.as_tensor(feats, device=device),
                        )
                    )
                    j = dist.sample()
                    chosen = remaining.pop(int(j.item()))
                    event = ep["candidates"][chosen]
                    before = _dlogit(
                        _point_logits(
                            model, mu, torch.stack([x["codec"] for x in ep["targets"]]), device
                        ),
                        full_logits,
                    )
                    known.append(event)
                    mu2, _ = _frozen_filter_state(model, known, ep["targets"][0]["codec"], device)
                    after = _dlogit(
                        _point_logits(
                            model, mu2, torch.stack([x["codec"] for x in ep["targets"]]), device
                        ),
                        full_logits,
                    )
                    logps.append(dist.log_prob(j))
                    vals.append(
                        critic(
                            _state(mu, ls, step, budget, len(remaining) + 1),
                            _state(bfull, ep["b_full_log_sigma"], 0, budget, len(ep["candidates"])),
                        )
                    )
                    rewards.append(before - after)
                if rewards:
                    ret: list[float] = []
                    g = 0.0
                    for reward in rewards[::-1]:
                        g += float(reward)
                        ret.append(g)
                    returns = torch.tensor(ret[::-1], dtype=torch.float32, device=device)
                    vt = torch.stack(vals)
                    adv = returns - vt.detach()
                    loss = -(torch.stack(logps) * adv).mean() + 0.5 * torch.nn.functional.mse_loss(
                        vt, returns
                    )
                    (loss / max(1, len(chosen_eps))).backward()
                    losses.append(float(loss.detach()))
            torch.nn.utils.clip_grad_norm_(
                list(actor.parameters()) + list(critic.parameters()), 1.0
            )
            opt.step()
        return {
            "actor": copy.deepcopy(actor.state_dict()),
            "critic": copy.deepcopy(critic.state_dict()),
            "loss": float(np.mean(losses)) if losses else float("nan"),
        }

    def score(state: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], epoch: int) -> float:
        del rows
        actor.load_state_dict(state["actor"])
        actor.eval()
        vals = []
        for i, ep in enumerate(val_eps):
            for budget in cfg.budgets:
                known: list[dict[str, Any]] = list(ep.get("prefix_events", []))
                remaining = list(range(len(ep["candidates"])))
                target_codecs = torch.stack([x["codec"] for x in ep["targets"]])
                full_logits = _point_logits(model, ep["b_full_mu"], target_codecs, device)
                eps = torch.randn(
                    8,
                    32,
                    generator=torch.Generator(device="cpu").manual_seed(
                        seed + 500000 + epoch * 100 + i
                    ),
                ).to(device)
                for step in range(min(int(budget), len(remaining))):
                    mu, ls = _frozen_filter_state(model, known, ep["targets"][0]["codec"], device)
                    feats, _ = _candidate_features(
                        model,
                        mu,
                        ls,
                        [ep["candidates"][j] for j in remaining],
                        ep["targets"],
                        eps,
                        device,
                    )
                    cc = torch.stack([ep["candidates"][j]["codec"] for j in remaining]).to(device)
                    with torch.no_grad():
                        j = int(
                            actor(
                                _state(mu, ls, step, budget, len(remaining)),
                                cc,
                                torch.as_tensor(feats, device=device),
                            ).argmax()
                        )
                    known.append(ep["candidates"][remaining.pop(j)])
                mu, _ = _frozen_filter_state(model, known, ep["targets"][0]["codec"], device)
                vals.append(
                    _dlogit(_point_logits(model, mu, target_codecs, device), full_logits)
                    + _kl(
                        torch.softmax(
                            torch.as_tensor(_point_logits(model, mu, target_codecs, device)), -1
                        ).numpy(),
                        torch.softmax(torch.as_tensor(full_logits), -1).numpy(),
                    )
                )
        return float(np.mean(vals))

    def save(state: Any, epoch: int, row: Mapping[str, Any]) -> None:
        torch.save(
            {
                "actor": state["actor"],
                "critic": state["critic"],
                "selected_epoch": int(epoch),
                "simulated_validation": dict(row),
                "reference_used_at_deployment": False,
            },
            out_dir / f"policy_seed{seed}_best.pt",
        )

    result = train_pure_simulator(
        train_rows=train_rows,
        validation_rows=val_rows,
        fit_epoch=fit_epoch,
        score_validation=score,
        save_checkpoint=save,
        config=cfg,
        seed=seed,
    )
    saved = torch.load(
        out_dir / f"policy_seed{seed}_best.pt", map_location=device, weights_only=False
    )
    actor.load_state_dict(saved["actor"])
    (out_dir / f"seed{seed}_epoch_curve.json").write_text(
        json.dumps(result, indent=2, default=str) + "\n"
    )
    return actor, result
