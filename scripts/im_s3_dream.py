#!/usr/bin/env python3
"""Shared helpers for the policy trained only inside the simulator (§8.3, Figure 11): the dream
corpus and the imagined-rollout environment.

A small, testable boundary around the CodeWorkout generative simulator. The simulated student is
fixed at ``b_full``; the observer's evolving belief is the only state an actor may see.
Provides: a batched, deterministic corpus generator conditioned on ``p(c | b_full, item)``, with
generated (licensed) text written only to a local path, never uploaded; corpus loading, indexing
and coverage checks; ``ImaginedRolloutEnv``; a 1:1 real/imagined mixer; a checkpoint selector
that accepts only real validation rows; the three launch guards (train-split leakage, outcome
marginals against head(b_full), and the random-policy floor against replay); and a loader that
rebuilds the CW generation path from numeric checkpoints. ``ga_1_tier2.py`` imports the same
environment. Run: `python scripts/im_s3_dream.py --smoke` (toy components, under a second).
"""
from __future__ import annotations

import copy
import hashlib
import json
import pathlib
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

STATIONARITY_DISCLOSURE = (
    "Stationarity within the diagnostic session is assumed: b_full defines the "
    "student-as-of-now, and imagined answers are drawn from that student — 'what "
    "would they do if asked today' — not from their historical selves. The measured "
    "size of the historical-drift caveat (full- vs trimmed-history reference rank "
    "agreement) is Spearman 0.7225, disclosed beside every "
    "curve; within-session learning from the asked questions themselves is the "
    "action-effect channel, explicitly outside this claim's scope."
)
BOUNDARY_DISCLOSURE = (
    "b_full plays the part the real student plays at deployment — it generates "
    "answers and defines success; it is never an input to the deployed policy."
)


@dataclass(frozen=True)
class DreamConfig:
    """Pre-registered constants for the corpus pass and its three guards."""

    seed: int = 20260902
    batch_size: int = 8
    max_new_tokens: int = 128
    classes_per_row: int | None = 2
    mix_ratio: float = 1.0
    marginal_z: float = 3.0
    marginal_tolerance_floor: float = 0.05
    random_floor_quantiles: tuple[float, float] = (0.05, 0.95)
    random_floor_margin: float = 0.0
    licence_class: str = "licensed_student_code"


def _local_path(path: str | pathlib.Path) -> pathlib.Path:
    """Return a local path and reject S3/HTTP destinations for raw generated text."""

    text = str(path)
    if text.startswith(("s3://", "http://", "https://")):
        raise ValueError("raw dream-corpus text is licence-class data and must remain local")
    out = pathlib.Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    return out


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalise_probs(values: Sequence[float], n_classes: int | None = None) -> np.ndarray:
    p = np.asarray(values, dtype=float)
    if p.ndim != 1 or (n_classes is not None and len(p) != n_classes):
        raise ValueError(f"class probabilities must be a 1-D vector, got shape {p.shape}")
    if not np.isfinite(p).all() or (p < 0).any() or float(p.sum()) <= 0:
        raise ValueError("class probabilities must be finite, non-negative, and non-zero")
    return p / p.sum()


def class_probabilities_from_b_full(
    outcome_head: Any,
    item_codec: Any,
    b_full_mu: Any,
    prefix: Any | None = None,
) -> np.ndarray:
    """Compute ``p(c | b_full, item)`` without sampling the observer's belief.

    Torch is imported lazily so the smoke tests and corpus bookkeeping remain cheap on
    machines that do not have the full backbone installed.
    """

    import torch

    with torch.no_grad():
        codec = torch.as_tensor(item_codec, dtype=torch.float32)
        mu = torch.as_tensor(b_full_mu, dtype=torch.float32)
        if codec.ndim == 1:
            codec = codec.unsqueeze(0)
        if mu.ndim == 1:
            mu = mu.unsqueeze(0)
        if prefix is None:
            prefix_dim = getattr(getattr(outcome_head, "prefix_proj", None), "in_features", None)
            prefix = torch.zeros((len(codec), prefix_dim), dtype=codec.dtype) if prefix_dim else None
        logits = outcome_head(codec, mu, prefix=prefix)
        return torch.softmax(logits, dim=-1)[0].detach().cpu().numpy()


class DreamCorpusGenerator:
    """Batched, deterministic corpus generator from frozen ``b_full`` rows.

    ``batch_generate`` receives a list of request dictionaries and returns one decoded
    answer per request.  Keeping this callback narrow lets the real CW model use
    ``greedy_generate_batch`` while tests use a deterministic local function.
    """

    def __init__(
        self,
        batch_generate: Callable[[list[dict[str, Any]]], Sequence[str]],
        config: DreamConfig | None = None,
        allowed_source_splits: Iterable[str] = ("train",),
    ) -> None:
        self.batch_generate = batch_generate
        self.config = config or DreamConfig()
        self.allowed_source_splits = frozenset(str(x) for x in allowed_source_splits)
        if self.config.batch_size < 1:
            raise ValueError("batch_size must be positive")

    def _prepare(self, rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        requests: list[dict[str, Any]] = []
        for i, row in enumerate(rows):
            source_split = str(row.get("source_split", row.get("split", "")))
            if source_split not in self.allowed_source_splits:
                raise AssertionError(
                    f"leakage canary: row {i} has source_split={source_split!r}; "
                    f"allowed={sorted(self.allowed_source_splits)}"
                )
            probs = row.get("class_probs")
            if probs is None:
                raise ValueError("each b_full row must provide class_probs or precompute them upstream")
            probs = _normalise_probs(probs)
            if self.config.classes_per_row is None:
                class_ids = list(range(len(probs)))
            else:
                if self.config.classes_per_row < 1:
                    raise ValueError("classes_per_row must be positive or None")
                class_ids = np.argsort(-probs, kind="stable")[: self.config.classes_per_row].tolist()
            for class_rank, conditioned_class in enumerate(class_ids):
                req = dict(row)
                req.update(
                    {
                        "row_index": i * max(1, len(class_ids)) + class_rank,
                        "source_row_index": i,
                        "class_probs": probs.tolist(),
                        "conditioned_class": int(conditioned_class),
                        "generated_from": "b_full",
                        "source_split": source_split,
                        "licence_class": self.config.licence_class,
                        "seed": self.config.seed,
                    }
                )
                requests.append(req)
        return requests

    def prepare_rows_from_b_full(self, rows: Sequence[Mapping[str, Any]], outcome_head: Any) -> list[dict[str, Any]]:
        """Attach ``p(c | b_full, item)`` to rows that carry an item codec and ``b_full_mu``."""

        prepared = []
        for row in rows:
            item = row.get("item_codec")
            mu = row.get("b_full_mu")
            if row.get("class_probs") is None and (item is None or mu is None):
                raise ValueError("a row without class_probs must provide item_codec and b_full_mu")
            copy_row = dict(row)
            if copy_row.get("class_probs") is None:
                copy_row["class_probs"] = class_probabilities_from_b_full(
                    outcome_head, item, mu, prefix=copy_row.get("prefix")
                ).tolist()
            prepared.append(copy_row)
        return prepared

    def generate(
        self,
        rows: Sequence[Mapping[str, Any]],
        out_path: str | pathlib.Path,
        checkpoint_sha256: str,
    ) -> dict[str, Any]:
        """Generate and atomically write a local JSONL corpus plus numeric manifest."""

        path = _local_path(out_path)
        requests = self._prepare(rows)
        records: list[dict[str, Any]] = []
        for start in range(0, len(requests), self.config.batch_size):
            chunk = requests[start : start + self.config.batch_size]
            answers = list(self.batch_generate(chunk))
            if len(answers) != len(chunk):
                raise ValueError(f"batch generator returned {len(answers)} answers for {len(chunk)} requests")
            for req, answer in zip(chunk, answers):
                records.append(
                    {
                        "schema_version": "im-s3-dream-v1",
                        "row_index": req["row_index"],
                        "source_row_index": req["source_row_index"],
                        "learner": str(req["learner"]),
                        "problem_id": int(req["problem_id"]),
                        "conditioned_class": int(req["conditioned_class"]),
                        "class_probs": req["class_probs"],
                        "source_split": req["source_split"],
                        "generated_from": "b_full",
                        "checkpoint_sha256": str(checkpoint_sha256),
                        "seed": self.config.seed,
                        "licence_class": self.config.licence_class,
                        "prompt_text": str(req.get("prompt_text", req.get("statement", ""))),
                        "generated_code": str(answer),
                    }
                )
        tmp = path.with_name(path.name + ".tmp")
        with tmp.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        tmp.replace(path)
        manifest = {
            "schema_version": "im-s3-dream-manifest-v1",
            "corpus_path": str(path),
            "n_records": len(records),
            "checkpoint_sha256": str(checkpoint_sha256),
            "seed": self.config.seed,
            "batch_size": self.config.batch_size,
            "allowed_source_splits": sorted(self.allowed_source_splits),
            "class_counts": {
                str(c): sum(int(r["conditioned_class"] == c) for r in records)
                for c in sorted({int(r["conditioned_class"]) for r in records})
            },
            "raw_text_uploaded": False,
            "licence_class": self.config.licence_class,
        }
        manifest_path = path.with_suffix(path.suffix + ".manifest.json")
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        return manifest


def load_dream_corpus(
    path: str | pathlib.Path, allowed_source_splits: Iterable[str] | None = None
) -> list[dict[str, Any]]:
    """Load local dream records and re-run the licence/leakage shape checks."""

    path = _local_path(path)
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    allowed = None if allowed_source_splits is None else frozenset(str(x) for x in allowed_source_splits)
    for record in records:
        if record.get("generated_from") != "b_full":
            raise AssertionError("dream record is not explicitly tied to frozen b_full")
        if record.get("licence_class") != "licensed_student_code":
            raise AssertionError("dream record has an unexpected licence class")
        if allowed is not None and str(record.get("source_split", "")) not in allowed:
            raise AssertionError(
                f"train/novel leakage canary: source_split={record.get('source_split')!r}; "
                f"allowed={sorted(allowed)}"
            )
    index_dream_corpus(records)
    return records


def index_dream_corpus(
    records: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, int, int], dict[str, Any]]:
    """Build the exact (learner, problem, conditioned-class) cache used by rollouts."""

    out: dict[tuple[str, int, int], dict[str, Any]] = {}
    for row in records:
        key = (str(row["learner"]), int(row["problem_id"]), int(row["conditioned_class"]))
        if key in out:
            raise AssertionError(f"dream corpus has duplicate cache key {key}")
        out[key] = dict(row)
    return out


def validate_dream_coverage(
    records: Sequence[Mapping[str, Any]],
    rollout_rows: Sequence[Mapping[str, Any]],
    n_classes: int,
) -> dict[str, Any]:
    """Fail fast unless every rollout (learner, problem) has every sampled class cached.

    The registered environment samples the full head distribution.  A top-K corpus is a
    useful cost-saving diagnostic, but it is not a valid imagined-rollout cache when a tail
    class still has non-zero probability.  Callers that need strict rollouts should pass
    ``DreamConfig(classes_per_row=None)`` and invoke this check before training.
    """

    if n_classes < 1:
        raise ValueError("n_classes must be positive")
    cache = index_dream_corpus(records)
    required: list[tuple[str, int, int]] = []
    for row in rollout_rows:
        learner, problem = str(row["learner"]), int(row["problem_id"])
        required.extend((learner, problem, c) for c in range(n_classes))
    missing = sorted(set(required) - set(cache))
    result = {"pass": not missing, "n_rollout_rows": len(rollout_rows),
              "n_required": len(set(required)), "n_cached": len(cache),
              "missing": missing[:50], "n_missing": len(missing), "n_classes": n_classes}
    if missing:
        raise AssertionError(
            f"dream corpus lacks {len(missing)} required class records; "
            "regenerate with classes_per_row=None for full-head sampling"
        )
    return result


class ImaginedRolloutEnv:
    """Environment whose student is frozen at ``b_full`` and whose observer is not."""

    def __init__(
        self,
        *,
        learner: str,
        candidates: Sequence[Mapping[str, Any]],
        initial_belief: Any,
        b_full: Any,
        predict_class: Callable[[Any, Mapping[str, Any]], Sequence[float]],
        update_belief: Callable[[Any, Mapping[str, Any]], Any],
        distance: Callable[[Any, Any], float],
        budget: int,
        seed: int,
        dream_records: Mapping[tuple[str, int, int], Mapping[str, Any]] | None = None,
    ) -> None:
        self.learner = str(learner)
        self.candidates = [dict(c) for c in candidates]
        self._initial = copy.deepcopy(initial_belief)
        self.b_full = copy.deepcopy(b_full)
        self.predict_class = predict_class
        self.update_belief = update_belief
        self.distance = distance
        self.budget = int(budget)
        self.seed = int(seed)
        self.dream_records = dream_records or {}
        if self.budget < 0:
            raise ValueError("budget must be non-negative")
        self.reset()

    def reset(self) -> dict[str, Any]:
        self.belief = copy.deepcopy(self._initial)
        self.remaining = list(range(len(self.candidates)))
        self.step_index = 0
        self.rng = np.random.default_rng(self.seed)
        self.outcome_history: list[dict[str, Any]] = []
        self.done = False
        return self.observation()

    def observation(self) -> dict[str, Any]:
        # Deliberately no b_full/reference field: this is the deployable actor boundary.
        return {"belief": copy.deepcopy(self.belief), "step": self.step_index,
                "remaining": tuple(self.remaining)}

    def step(self, action: int) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        if self.done:
            raise RuntimeError("rollout is already done")
        if action not in self.remaining:
            raise ValueError(f"action {action} is not an available candidate")
        candidate = self.candidates[action]
        probs = _normalise_probs(self.predict_class(self.b_full, candidate))
        outcome = int(self.rng.choice(len(probs), p=probs))
        key = (self.learner, int(candidate["problem_id"]), outcome)
        dream = self.dream_records.get(key)
        if self.dream_records and dream is None:
            raise KeyError(f"dream corpus missing required key {key}")
        before = float(self.distance(self.belief, self.b_full))
        event = {"learner": self.learner, "problem_id": int(candidate["problem_id"]),
                 "outcome_code": outcome, "generated_code": None if dream is None else dream["generated_code"],
                 "class_probs": probs.tolist(), "source": "imagined", "student_reference": "b_full"}
        self.outcome_history.append(dict(event))
        self.belief = self.update_belief(self.belief, event)
        after = float(self.distance(self.belief, self.b_full))
        self.remaining.remove(action)
        self.step_index += 1
        self.done = self.step_index >= self.budget or not self.remaining
        return self.observation(), before - after, self.done, {"event": event, "distance": after}


class OneToOneMixer:
    """Deterministically interleave one real replay pass and one imagined pass."""

    def __init__(self, ratio: float = 1.0) -> None:
        if ratio != 1.0:
            raise ValueError("imagination training is pre-registered at exactly 1:1 real:imagined mixing")
        self.ratio = ratio

    def mix(self, real: Sequence[Mapping[str, Any]], imagined: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        if len(real) != len(imagined):
            raise ValueError(f"1:1 mixer requires equal passes, got {len(real)} and {len(imagined)}")
        out: list[dict[str, Any]] = []
        for r, i in zip(real, imagined):
            rr, ii = dict(r), dict(i)
            rr["source"] = "real_replay"
            ii["source"] = "imagined"
            out.extend((rr, ii))
        return out


class RealOnlyCheckpointSelector:
    """Best-validation checkpoint selector that refuses imagined validation rows."""

    def __init__(self) -> None:
        self.best_score = float("inf")
        self.best_epoch: int | None = None
        self.best_state: Any = None
        self.curve: list[dict[str, Any]] = []

    def consider(self, epoch: int, score: float, state: Any, validation_rows: Sequence[Mapping[str, Any]]) -> bool:
        if any(str(row.get("source", "real_replay")) != "real_replay" for row in validation_rows):
            raise AssertionError("checkpoint selector may use real replay validation only")
        if any(str(row.get("split", "val")) not in {"val", "validation"} for row in validation_rows):
            raise AssertionError("checkpoint selector received a non-validation row")
        score = float(score)
        if not np.isfinite(score):
            raise ValueError("validation score must be finite")
        improved = score < self.best_score
        self.curve.append({"epoch": int(epoch), "score": score, "improved": improved})
        if improved:
            self.best_score = score
            self.best_epoch = int(epoch)
            self.best_state = copy.deepcopy(state)
        return improved


def leakage_canary(records: Sequence[Mapping[str, Any]], allowed_splits: Iterable[str] = ("train",)) -> dict[str, Any]:
    allowed = frozenset(str(x) for x in allowed_splits)
    bad = [i for i, r in enumerate(records) if str(r.get("source_split", "")) not in allowed]
    result = {"pass": not bad, "n_records": len(records), "bad_rows": bad,
              "allowed_source_splits": sorted(allowed)}
    if bad:
        raise AssertionError(f"train-split leakage canary failed on rows {bad[:5]}")
    return result


def marginal_guard(
    records: Sequence[Mapping[str, Any]],
    expected_probs: Sequence[float],
    config: DreamConfig | None = None,
    observed_field: str | None = None,
) -> dict[str, Any]:
    """Check observed outcome frequencies against head(``b_full``) marginals.

    Corpus records use ``conditioned_class`` because they contain one offline answer per
    requested class.  Rollout records use ``outcome_code`` because the environment samples
    that class from ``head(b_full, item)``.  The latter is the mechanical consistency guard
    required before imagination training; the field is inferred for backwards-compatible smoke callers but can
    be made explicit at launch.
    """

    cfg = config or DreamConfig()
    if not records:
        raise ValueError("marginal guard needs at least one generated record")
    expected = _normalise_probs(expected_probs)
    if observed_field is None:
        observed_field = "outcome_code" if any("outcome_code" in r for r in records) else "conditioned_class"
    if any(observed_field not in r for r in records):
        raise ValueError(f"marginal guard records are missing observed field {observed_field!r}")
    values = [int(r[observed_field]) for r in records]
    if any(v < 0 or v >= len(expected) for v in values):
        raise ValueError(f"observed class outside expected range 0..{len(expected)-1}")
    observed = np.bincount(values, minlength=len(expected)) / len(records)
    se = np.sqrt(expected * np.clip(1.0 - expected, 0.0, None) / len(records))
    tolerance = np.maximum(cfg.marginal_tolerance_floor, cfg.marginal_z * se)
    passed = bool(np.all(np.abs(observed - expected) <= tolerance))
    result = {"pass": passed, "n": len(records), "observed": observed.tolist(),
              "expected": expected.tolist(), "tolerance": tolerance.tolist(),
              "observed_field": observed_field,
              "rule": "|observed-expected| <= max(0.05, 3*binomial_se)"}
    if not passed:
        raise AssertionError(f"imagined marginal guard failed: {result}")
    return result


def rollout_marginal_guard(
    outcome_events: Sequence[Mapping[str, Any]], config: DreamConfig | None = None
) -> dict[str, Any]:
    """Apply the mechanical guard to sampled rollout outcomes, not conditioned corpus rows.

    The head can produce a different marginal for every candidate.  Pooling those expected
    vectors and comparing with the pooled sampled classes is the pre-registered aggregate
    check; callers should pass all CRN rollout events for the arm, not one five-step episode.
    """

    if not outcome_events:
        raise ValueError("rollout marginal guard needs at least one sampled event")
    vector_rows = [_normalise_probs(e["class_probs"]) for e in outcome_events]
    if len({row.shape[0] for row in vector_rows}) != 1:
        raise ValueError("rollout class-probability vectors must share one class count")
    vectors = np.stack(vector_rows)
    expected = vectors.mean(axis=0)
    return marginal_guard(outcome_events, expected, config=config, observed_field="outcome_code")


def run_all_guards(
    *,
    corpus_records: Sequence[Mapping[str, Any]],
    outcome_events: Sequence[Mapping[str, Any]],
    imagined_random_distances: Sequence[float],
    replay_random_distances: Sequence[float],
    allowed_source_splits: Iterable[str] = ("train",),
    config: DreamConfig | None = None,
    rollout_rows: Sequence[Mapping[str, Any]] | None = None,
    n_classes: int | None = None,
) -> dict[str, Any]:
    """Run the three imagination guards as one fail-closed launch gate.

    The first guard is evaluated on every generated corpus record; the second is evaluated on
    sampled rollout outcomes (not the offline top-K class rows); and the third compares the
    imagined RANDOM trajectory with the replay RANDOM reference range.  If rollout rows are
    supplied, full class-cache coverage is checked as an additional fail-fast condition. Any
    failed guard raises before a caller can treat the arm as launchable.
    """

    result = {
        "leakage": leakage_canary(corpus_records, allowed_splits=allowed_source_splits),
        "marginal": rollout_marginal_guard(outcome_events, config=config),
        "random_floor": random_policy_floor_guard(
            imagined_random_distances, replay_random_distances, config=config
        ),
    }
    if rollout_rows is not None:
        if n_classes is None:
            raise ValueError("n_classes is required when validating dream coverage")
        result["coverage"] = validate_dream_coverage(corpus_records, rollout_rows, n_classes)
    return result


def random_policy_floor_guard(
    imagined_distances: Sequence[float],
    replay_distances: Sequence[float],
    config: DreamConfig | None = None,
) -> dict[str, Any]:
    """Ensure imagined RANDOM final distances remain in replay's registered range."""

    cfg = config or DreamConfig()
    imagined = np.asarray(imagined_distances, dtype=float)
    replay = np.asarray(replay_distances, dtype=float)
    if imagined.size == 0 or replay.size == 0 or not np.isfinite(imagined).all() or not np.isfinite(replay).all():
        raise ValueError("random-policy floor guard requires finite non-empty distances")
    lo, hi = np.quantile(replay, cfg.random_floor_quantiles)
    margin = float(cfg.random_floor_margin)
    passed = bool(imagined.min() >= lo - margin and imagined.max() <= hi + margin)
    result = {"pass": passed, "imagined_min": float(imagined.min()), "imagined_max": float(imagined.max()),
              "replay_q05": float(lo), "replay_q95": float(hi), "margin": margin,
              "rule": "imagined RANDOM final distance lies in replay 5th–95th percentile range"}
    if not passed:
        raise AssertionError(f"imagined random-policy floor guard failed: {result}")
    return result


def make_cw_batched_generator(
    model: Mapping[str, Any], device: str = "cuda", max_new_tokens: int = DreamConfig().max_new_tokens
) -> Callable[[list[dict[str, Any]]], list[str]]:
    """Adapt the shipped CW model to the corpus generator's batch callback.

    The real run supplies request fields ``prompt_ids`` and ``b_full_mu``.  Prompt
    construction is intentionally outside this adapter so class markers and truncation
    continue to use CW's audited ``build_reserved_prompt_ids`` implementation.
    """

    import torch

    from infra.cw_sim_core import greedy_generate_batch

    def generate(requests: list[dict[str, Any]]) -> list[str]:
        if not requests:
            return []
        if "tok" not in model or "model" not in model or "context" not in model or "hyper" not in model:
            raise KeyError("CW generator model must expose model/tok/context/hyper")
        with torch.no_grad():
            mus = torch.as_tensor(np.stack([r["b_full_mu"] for r in requests]), dtype=torch.float32, device=device)
            c_vecs = model["hyper"](mus)
        prompts = []
        for r in requests:
            if "prompt_ids" in r:
                prompts.append(list(r["prompt_ids"]))
                continue
            from atom.simulator.cw_alt import build_reserved_prompt_ids

            ids, _, _ = build_reserved_prompt_ids(
                str(r["statement"]), r.get("prior_code"), model["tok"],
                max_len=512, min_completion_tokens=8,
                outcome_code=int(r["conditioned_class"]),
            )
            prompts.append(ids)
        return greedy_generate_batch(
            model["model"], model["tok"], model["context"], c_vecs, prompts,
            max_new_tokens=max_new_tokens, device=device,
        )

    return generate


def load_cw_b_full_generation_model(
    base_checkpoint: str | pathlib.Path,
    *,
    device: str = "cuda",
    backbone: str = "gemma4-e4b",
    head_checkpoint: str | pathlib.Path | None = None,
    allow_download: bool = False,
) -> dict[str, Any]:
    """Reconstruct the frozen CW generation path from numeric checkpoints.

    The CW checkpoints intentionally omit the 4B backbone.  This adapter loads it from
    the local HF cache, restores the ``b_full`` filter/hypernetwork/LoRA basis, and
    optionally swaps in the separately published refreshed outcome head.  It never
    accepts a raw-text or remote source, and leaves the caller responsible for copying
    the resulting ``generations.jsonl`` off the machine that ran it.
    """

    import torch

    from infra import cw_5_full_run as cw5

    base_path = pathlib.Path(base_checkpoint)
    if str(base_checkpoint).startswith(("s3://", "http://", "https://")):
        raise ValueError("generation checkpoints must be staged locally or through S3-to-instance")
    if not base_path.exists():
        raise FileNotFoundError(base_path)
    base = torch.load(base_path, map_location="cpu", weights_only=False)
    arch = dict(base.get("architecture") or cw5.CW5_DIMS)
    bundle = cw5.load_backbone(backbone, device=device, allow_download=allow_download)
    model = cw5.build_model(
        cw5.CW5_DIMS,
        bundle,
        int(base.get("seed", 20260830)),
        device,
        item_encoder="mlp",
        sigma_parameterization="precision_accumulating",
    )
    if arch.get("codec_dim") != cw5.CW5_DIMS["codec_dim"]:
        raise ValueError("b_full generation checkpoint codec dimension does not match CW planner")
    for module, key in ((model["attempt_encoder"], "attempt_encoder"),
                        (model["filt"], "filt")):
        if key not in base:
            raise ValueError(f"base generation checkpoint missing {key}")
        module.load_state_dict(base[key], strict=True)
    if head_checkpoint is None:
        head = base
    else:
        head_path = pathlib.Path(head_checkpoint)
        if str(head_checkpoint).startswith(("s3://", "http://", "https://")):
            raise ValueError("head checkpoint must be staged locally or through S3-to-instance")
        head = torch.load(head_path, map_location="cpu", weights_only=False)
    model["outcome_head"].load_state_dict(head["outcome_head"], strict=True)
    for key in ("hyper", "basis"):
        if key not in base:
            raise ValueError(f"base generation checkpoint missing {key}; cannot form b_full LoRA path")
    model["hyper"].load_state_dict(base["hyper"], strict=True)
    for name, state in base["basis"].items():
        if name not in model["bases"]:
            raise ValueError(f"base checkpoint basis {name!r} absent from reconstructed model")
        model["bases"][name].load_state_dict(state, strict=True)
    model["generation_checkpoint_sha256"] = _sha256(base_path)
    model["head_checkpoint_sha256"] = _sha256(pathlib.Path(head_checkpoint)) if head_checkpoint else model["generation_checkpoint_sha256"]
    for value in model.values():
        if hasattr(value, "eval"):
            value.eval()
    return model


def run_smoke() -> dict[str, Any]:
    """Exercise generation, guards, mixing, selector, and environment in <1 second."""

    import tempfile

    rows = [
        {"learner": "L0", "problem_id": i, "source_split": "train", "class_probs": [0.25, 0.25, 0.25, 0.25],
         "statement": f"toy problem {i}", "b_full_mu": [1.0, 0.0]}
        for i in range(8)
    ]
    generator = DreamCorpusGenerator(
        lambda batch: [f"answer-{r['problem_id']}-{r['conditioned_class']}" for r in batch],
        DreamConfig(classes_per_row=None),
    )
    with tempfile.TemporaryDirectory() as tmp:
        manifest = generator.generate(rows, pathlib.Path(tmp) / "generations.jsonl", "smoke-checkpoint")
        records = load_dream_corpus(pathlib.Path(tmp) / "generations.jsonl", ("train",))
        leak = leakage_canary(records)
        by_key = index_dream_corpus(records)
        env = ImaginedRolloutEnv(
            learner="L0", candidates=[{"problem_id": i} for i in range(8)], initial_belief=np.array([0.0]),
            b_full=np.array([1.0]), predict_class=lambda _b, _c: [0.25] * 4,
            update_belief=lambda b, e: b + np.array([1.0 if e["outcome_code"] == 0 else 0.0]),
            distance=lambda b, full: float(abs(float(full[0] - b[0]))), budget=2, seed=7,
            dream_records=by_key,
        )
        env.step(0)
        env.step(1)
        marg = rollout_marginal_guard(env.outcome_history)
        mixer = OneToOneMixer()
        mixed = mixer.mix([{"split": "train"}], [{"split": "train"}])
        selector = RealOnlyCheckpointSelector()
        selector.consider(1, 0.5, {"epoch": 1}, [{"split": "val", "source": "real_replay"}])
        floor = random_policy_floor_guard([0.3, 0.4], [0.2, 0.5, 0.4, 0.3])
        guards = run_all_guards(
            corpus_records=records,
            outcome_events=env.outcome_history,
            imagined_random_distances=[0.3, 0.4],
            replay_random_distances=[0.2, 0.5, 0.4, 0.3],
            rollout_rows=rows,
            n_classes=4,
        )
        return {"manifest": manifest, "leakage": leak, "marginal": marg, "mixed": len(mixed),
                "selected_epoch": selector.best_epoch, "random_floor": floor, "guards": guards,
                "env_done": env.done}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        print(json.dumps(run_smoke(), indent=2))
    else:
        parser.error("this module is a library with a build check only; run it with --smoke")
