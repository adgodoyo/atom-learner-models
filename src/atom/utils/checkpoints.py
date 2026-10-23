"""Checkpointing: rolling keep-last-N + best, resumable.

These primitives support a kill test: `kill -9` mid-run, resume from the newest
checkpoint, and the final metric must match an uninterrupted run. Everything
a run needs to resume (model, optimiser, step, RNG state, config hash) goes into the
`state` dict the caller builds; this module only handles durable rotation and lookup.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import torch


def _atomic_torch_save(payload: Any, path: Path) -> None:
    """Write via a temp file + os.replace so a kill mid-write leaves no torn .pt file --
    the reader only ever sees the old complete file or the new complete file, never a
    partial one (POSIX rename is atomic). A `kill -9` mid-checkpoint therefore cannot
    corrupt the checkpoint it lands on."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def _atomic_write_text(text: str, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def _ckpt_dir(run_dir: str | Path) -> Path:
    d = Path(run_dir) / "checkpoints"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _index_path(run_dir: str | Path) -> Path:
    return _ckpt_dir(run_dir) / "index.json"


def _load_index(run_dir: str | Path) -> dict[str, Any]:
    p = _index_path(run_dir)
    if p.exists():
        return json.loads(p.read_text())
    return {"history": [], "best": None}


def _save_index(run_dir: str | Path, index: dict[str, Any]) -> None:
    _atomic_write_text(json.dumps(index, indent=2), _index_path(run_dir))


def save_checkpoint(
    state: dict[str, Any],
    run_dir: str | Path,
    step: int,
    metric: float | None = None,
    is_best: bool = False,
    keep_last: int = 3,
) -> Path:
    """Save `state` at `step`; rotate to keep the newest `keep_last`; track best.

    `best.pt` is never rotated away. Returns the path of the step checkpoint written.
    """
    cdir = _ckpt_dir(run_dir)
    payload = {"step": step, "metric": metric, **state}
    path = cdir / f"ckpt_step{step:08d}.pt"
    _atomic_torch_save(payload, path)

    index = _load_index(run_dir)
    index["history"] = [h for h in index["history"] if h["step"] != step]
    index["history"].append({"step": step, "metric": metric, "path": path.name})
    index["history"].sort(key=lambda h: h["step"])

    if is_best:
        best_path = cdir / "best.pt"
        _atomic_torch_save(payload, best_path)
        index["best"] = {"step": step, "metric": metric, "path": best_path.name}

    # Rotate step checkpoints (never touch best.pt).
    steps_sorted = [h for h in index["history"]]
    if len(steps_sorted) > keep_last:
        for h in steps_sorted[:-keep_last]:
            stale = cdir / h["path"]
            if stale.exists():
                stale.unlink()
        index["history"] = steps_sorted[-keep_last:]

    _save_index(run_dir, index)
    return path


def latest_step(run_dir: str | Path) -> int | None:
    index = _load_index(run_dir)
    if not index["history"]:
        return None
    return index["history"][-1]["step"]


def load_checkpoint(
    run_dir: str | Path,
    which: str | int = "last",
    map_location: str | None = "cpu",
    require_sampler_semantics: bool = False,
) -> dict[str, Any]:
    """Load a checkpoint payload. `which` in {"last", "best", <step int>}.

    `require_sampler_semantics`: opt-in, default False so every non-simulator resume
    (filter-only, planner, binding) is unaffected. A simulator training script resuming or
    warm-starting from a checkpoint should pass True; it then refuses checkpoints whose
    belief samples were drawn under an older sampling convention -- see
    `atom.simulator.sampler_semantics` for what this refuses and why.
    """
    cdir = _ckpt_dir(run_dir)
    index = _load_index(run_dir)

    if which == "best":
        if not index.get("best"):
            raise FileNotFoundError(f"no best checkpoint in {cdir}")
        path = cdir / index["best"]["path"]
    elif which == "last":
        if not index["history"]:
            raise FileNotFoundError(f"no checkpoints in {cdir}")
        path = cdir / index["history"][-1]["path"]
    elif isinstance(which, int):
        path = cdir / f"ckpt_step{which:08d}.pt"
        if not path.exists():
            raise FileNotFoundError(f"no checkpoint at step {which} in {cdir}")
    else:
        raise ValueError(f"which must be 'last', 'best' or an int step; got {which!r}")

    payload = torch.load(path, map_location=map_location, weights_only=False)
    if require_sampler_semantics:
        from atom.simulator.sampler_semantics import require_current_sampler_semantics
        require_current_sampler_semantics(payload, context=str(path))
    return payload


def has_checkpoint(run_dir: str | Path) -> bool:
    return _index_path(run_dir).exists() and bool(_load_index(run_dir)["history"])
