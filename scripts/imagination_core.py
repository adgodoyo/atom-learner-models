"""Shared helpers for `sim2real_w_run` (§8.3, the policy trained only inside the simulator): the
simulator-only training contract. Provides the pre-registered constants, fail-closed validators
that keep real replay rows out of fitting and checkpoint selection, a best-simulated-validation
training loop, a seed-then-learner bootstrap, and the real held-out exam with its explicit
point-plus-CI transfer bar. Every exam summary carries the verbatim disclosures below.
"""
from __future__ import annotations
import copy
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence
import numpy as np
from im_s3_dream import BOUNDARY_DISCLOSURE, STATIONARITY_DISCLOSURE
STYLE_DISCLOSURE = (
    "Outcome-faithful/style-generic environment (belief content-swap probe 0.62–0.65; "
    "generated code personalised vs shuffled codec similarity Δ≈0.0005)."
)
VERBATIM_DISCLOSURES = (STYLE_DISCLOSURE, STATIONARITY_DISCLOSURE, BOUNDARY_DISCLOSURE)


@dataclass(frozen=True)
class Sim2RealConfig:
    """Pre-registered transfer-experiment constants; no values are selected after an exam result."""

    seeds: tuple[int, ...] = (20260902, 20260903, 20260904)
    budgets: tuple[int, ...] = (3, 5, 8, 12, 16, 20, 25, 30)
    max_epochs: int = 40
    validation_patience: int = 5
    min_validation_improvement: float = 0.001
    n_boot: int = 2000
    off_menu_allowed_in_sim: bool = True
    overrun_multiplier: float = 2.0
    gpu_instance_cap: int = 4
    training_source: str = "simulator"
    validation_source: str = "simulated_validation"
    exam_source: str = "real_heldout_replay"

    def __post_init__(self) -> None:
        if len(self.seeds) not in (1, 3):
            raise ValueError("the transfer experiment uses either one pre-registered seed (warm-start variant) or three (full arms)")
        if self.max_epochs < 1 or self.validation_patience < 1:
            raise ValueError("epoch and patience limits must be positive")
        if self.min_validation_improvement <= 0:
            raise ValueError("validation improvement tolerance must be positive")
        if self.overrun_multiplier <= 1:
            raise ValueError("overrun trigger must be greater than one")


def _source(row: Mapping[str, Any]) -> str:
    return str(row.get("source", row.get("data_source", ""))).strip().lower()


def _reject_real_rows(rows: Sequence[Mapping[str, Any]], *, stage: str) -> None:
    """Fail closed if replay data reaches simulator fitting or selection."""

    bad: list[tuple[int, str]] = []
    for i, row in enumerate(rows):
        source = _source(row)
        split = str(row.get("split", row.get("source_split", ""))).lower()
        marker = " ".join((source, split, str(row.get("episode_id", "")))).lower()
        if source in {"real", "real_replay", "real_heldout_replay", "replay", "validation"}:
            bad.append((i, source))
        if "replay" in marker or "heldout" in marker and source != "simulator":
            bad.append((i, source or split))
    if bad:
        raise AssertionError(
            f"simulator-only boundary violation at {stage}: real replay rows reached the "
            f"simulator-only stage (first rows={bad[:5]})"
        )


def validate_simulator_training_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Validate rows consumed by fitting; every row must be simulator-generated."""

    _reject_real_rows(rows, stage="training")
    bad = [i for i, row in enumerate(rows) if _source(row) != "simulator"]
    if bad:
        raise AssertionError(
            f"simulator-only training requires source='simulator'; bad rows={bad[:5]}"
        )
    if any(row.get("generated_from") != "b_full" for row in rows):
        raise AssertionError("simulator training rows must be generated from frozen b_full")
    return {"pass": True, "n_rows": len(rows), "source": "simulator", "real_rows": 0}


def validate_simulated_validation_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Validate rows used for checkpoint choice; real data is forbidden here too."""

    _reject_real_rows(rows, stage="checkpoint_selection")
    bad = [
        i
        for i, row in enumerate(rows)
        if _source(row) != "simulated_validation"
        or str(row.get("split", "")).lower() not in {"sim_val", "simulated_validation"}
    ]
    if bad:
        raise AssertionError(
            "checkpoint selection requires source='simulated_validation' and "
            f"split='sim_val'; bad rows={bad[:5]}"
        )
    if any(row.get("generated_from") != "b_full" for row in rows):
        raise AssertionError("simulated validation rows must be generated from frozen b_full")
    return {"pass": True, "n_rows": len(rows), "source": "simulated_validation", "real_rows": 0}


class SimOnlyCheckpointSelector:
    """Best-validation selector that refuses real replay rows by construction."""

    def __init__(self, min_improvement: float = Sim2RealConfig().min_validation_improvement) -> None:
        if min_improvement <= 0:
            raise ValueError("min_improvement must be positive")
        self.min_improvement = float(min_improvement)
        self.best_score = float("inf")
        self.best_epoch: int | None = None
        self.best_state: Any = None
        self.curve: list[dict[str, Any]] = []
        self.stale_epochs = 0

    def consider(
        self,
        epoch: int,
        score: float,
        state: Any,
        validation_rows: Sequence[Mapping[str, Any]],
    ) -> bool:
        validate_simulated_validation_rows(validation_rows)
        value = float(score)
        if not np.isfinite(value):
            raise ValueError("simulated validation score must be finite")
        improvement = self.best_score - value
        improved = improvement > self.min_improvement
        if improved:
            self.best_score = value
            self.best_epoch = int(epoch)
            self.best_state = copy.deepcopy(state)
            self.stale_epochs = 0
        else:
            self.stale_epochs += 1
        self.curve.append(
            {
                "epoch": int(epoch),
                "score": value,
                "improvement": float(improvement) if np.isfinite(improvement) else None,
                "improved": improved,
                "stale_epochs": self.stale_epochs,
                "source": "simulated_validation",
            }
        )
        return improved

    def should_stop(self, patience: int = Sim2RealConfig().validation_patience) -> bool:
        return self.stale_epochs >= int(patience)


def train_pure_simulator(
    *,
    train_rows: Sequence[Mapping[str, Any]],
    validation_rows: Sequence[Mapping[str, Any]],
    fit_epoch: Callable[[int, int, Sequence[Mapping[str, Any]]], Any],
    score_validation: Callable[[Any, Sequence[Mapping[str, Any]], int], float],
    save_checkpoint: Callable[[Any, int, Mapping[str, Any]], None] | None = None,
    config: Sim2RealConfig | None = None,
    seed: int,
) -> dict[str, Any]:
    """Run the generic pilot/full training contract without opening a replay path.

    ``fit_epoch`` receives simulator rows only.  ``score_validation`` receives
    simulated-validation rows only.  The callback shape keeps this boundary reusable by
    the existing pointer actor while making an accidental replay join impossible.
    """

    cfg = config or Sim2RealConfig()
    validate_simulator_training_rows(train_rows)
    validate_simulated_validation_rows(validation_rows)
    selector = SimOnlyCheckpointSelector(cfg.min_validation_improvement)
    epoch_rows: list[dict[str, Any]] = []
    for epoch in range(1, cfg.max_epochs + 1):
        state = fit_epoch(int(seed), epoch, train_rows)
        score = float(score_validation(state, validation_rows, epoch))
        selector.consider(epoch, score, state, validation_rows)
        row = dict(selector.curve[-1])
        row["seed"] = int(seed)
        epoch_rows.append(row)
        if save_checkpoint is not None:
            # Save only the selected state.  A final/last state is not a substitute for
            # best simulated validation and is deliberately not handed to the exam.
            if row["improved"]:
                save_checkpoint(selector.best_state, epoch, row)
        if selector.should_stop(cfg.validation_patience):
            break
    if selector.best_state is None or selector.best_epoch is None:
        raise RuntimeError("no finite simulated-validation checkpoint was selected")
    return {
        "seed": int(seed),
        "epochs_run": len(epoch_rows),
        "selected_epoch": selector.best_epoch,
        "selected_validation_score": selector.best_score,
        "stopping_reason": "validation_patience" if selector.should_stop(cfg.validation_patience) else "epoch_cap",
        "epoch_curve": epoch_rows,
        "training_source": "simulator",
        "selection_source": "simulated_validation",
        "real_rows_in_training": 0,
        "real_rows_in_selection": 0,
    }


def _cluster_bootstrap(values: Mapping[str, Sequence[float]], seed: int, n_boot: int) -> list[float]:
    """Two-level bootstrap: resample seeds, then learners within each seed."""

    arrays = [np.asarray(v, dtype=float) for _, v in sorted(values.items())]
    arrays = [x[np.isfinite(x)] for x in arrays if len(x) and np.isfinite(x).any()]
    if not arrays:
        return [float("nan")] * 3
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        picked = rng.integers(0, len(arrays), size=len(arrays))
        means = []
        for i in picked:
            x = arrays[int(i)]
            means.append(float(x[rng.integers(0, len(x), size=len(x))].mean()))
        draws[b] = float(np.mean(means))
    return [float(np.mean([x.mean() for x in arrays])), float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]


def evaluate_real_exam(
    *,
    cells: Mapping[str, Mapping[str, Sequence[float]]],
    training_rows: Sequence[Mapping[str, Any]],
    selection_rows: Sequence[Mapping[str, Any]],
    exam_rows: Sequence[Mapping[str, Any]],
    config: Sim2RealConfig | None = None,
) -> dict[str, Any]:
    """Evaluate a selected simulator actor on real held-out replay.

    ``cells`` is keyed by policy and contains learner-clustered arrays per seed.  The
    actor must beat random and every blind heuristic on the registered real exam metric
    (lower is better here).  The replay-trained actor is retained as a reference only.
    """

    cfg = config or Sim2RealConfig()
    validate_simulator_training_rows(training_rows)
    validate_simulated_validation_rows(selection_rows)
    bad_exam = [i for i, row in enumerate(exam_rows) if _source(row) != cfg.exam_source]
    if bad_exam:
        raise AssertionError(f"real exam requires source={cfg.exam_source!r}; bad rows={bad_exam[:5]}")
    required = {"sim_actor", "random", "greedy_epig", "max_entropy", "difficulty_matched"}
    missing = sorted(required - set(cells))
    if missing:
        raise ValueError(f"real exam is missing registered blind comparators: {missing}")
    summary: dict[str, Any] = {}
    for policy, by_seed in cells.items():
        summary[policy] = _cluster_bootstrap(by_seed, seed=cfg.seeds[0] + len(policy), n_boot=cfg.n_boot)
    actor_mean = summary["sim_actor"][0]
    # The bar is intentionally point+CI explicit: the simulator actor's upper CI must
    # be below each comparator's lower CI before a transfer claim is called positive.
    comparisons = {
        policy: {
            "beats": bool(summary["sim_actor"][2] < summary[policy][1]),
            "actor_minus_comparator": float(actor_mean - summary[policy][0]),
        }
        for policy in sorted(required - {"sim_actor"})
    }
    summary["replay_actor_reference"] = summary.get("replay_actor_reference")
    return {
        "status": "complete",
        "metric_direction": "lower_is_better",
        "bar": "sim_actor beats random and every blind heuristic on real held-out replay",
        "bar_pass": bool(all(x["beats"] for x in comparisons.values())),
        "policies": summary,
        "comparisons": comparisons,
        "real_rows_in_training": 0,
        "real_rows_in_selection": 0,
        "exam_source": cfg.exam_source,
        "disclosures": list(VERBATIM_DISCLOSURES),
    }
