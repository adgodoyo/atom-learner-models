"""Run logging: heartbeat, metrics stream, run manifest and tripwires.

A `RunLogger` gives every long-running process:
  1. a heartbeat (<= every 30 s) to stdout AND runs/<id>/heartbeat.log;
  2. a metrics stream, one JSON line per evaluation in runs/<id>/metrics.jsonl,
     each stamped with wall time, config hash, git SHA, data version, seed;
  3. a run manifest (runs/<id>/manifest.yaml) written at start -- a run without a
     manifest is invalid;
  4. inline tripwires (assert-and-halt) for NaN/inf and loss spikes.

Torch is optional: hardware fields degrade gracefully if it is not installed.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import socket
import subprocess
import sys
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


class TripwireError(RuntimeError):
    """Raised when an inline tripwire fires (NaN/inf, loss explosion, ...).

    The contract is assert-and-halt: a run that goes wrong must become visibly
    wrong within minutes, so we raise rather than warn.
    """


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def git_sha(short: bool = True) -> str:
    """Current git SHA, or 'nogit' if unavailable (e.g. before `git init`)."""
    cmd = ["git", "rev-parse", "--short", "HEAD"] if short else ["git", "rev-parse", "HEAD"]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        if out.returncode == 0:
            return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return "nogit"


def config_hash(config: Any, length: int = 10) -> str:
    """Stable short hash of a config dict, for tagging every artifact it produced."""
    blob = json.dumps(config, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:length]


def _hardware_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "hostname": socket.gethostname(),
        "cpu_count": os.cpu_count(),
    }
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        info["mps_available"] = bool(
            getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
        )
    except ImportError:
        info["torch"] = None
    return info


def default_device() -> str:
    """Pick the best available torch device string: mps (Apple), else cuda, else cpu."""
    try:
        import torch

        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


def rearm_shutdown_fuse(elapsed_s: float, current_step: int, total_steps: int,
                        min_fuse_minutes: int = 60) -> int:
    """Re-arm the OS shutdown fuse of an unattended cloud run from its own measured rate.

    A fuse set once at launch from a guessed duration can kill a run that is merely
    slower than expected; re-arming it from the measured pace avoids that. The fuse is
    `max(min_fuse_minutes, ceil(2 * projected_remaining_minutes))`, where the projection
    uses the mean seconds per step so far. The same rule is used by
    `infra/cw_5_full_run.py` with row counts; here it is expressed in steps so any
    step-based training loop can share it. Call it from the heartbeat block of a long
    training loop -- it is cheap (two `subprocess.run` calls) and idempotent.

    Returns the number of minutes armed (0 if `current_step <= 0`, i.e. no rate yet --
    the caller's own boot-time fuse covers that window instead)."""
    if current_step <= 0:
        return 0
    seconds_per_step = elapsed_s / current_step
    remaining_steps = max(0, total_steps - current_step)
    projected_remaining_min = seconds_per_step * remaining_steps / 60.0
    fuse_minutes = max(min_fuse_minutes, int(math.ceil(2.0 * projected_remaining_min)))
    subprocess.run(["sudo", "shutdown", "-c"], check=False,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run(["sudo", "shutdown", "-h", f"+{fuse_minutes}"], check=False,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return fuse_minutes


def _to_scalar(value: Any) -> float:
    """Coerce a python/numpy/torch scalar to float for finiteness checks."""
    try:
        import torch

        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                # For non-scalar tensors, reduce to a finiteness flag: NaN if any bad.
                return float("nan") if not bool(torch.isfinite(value).all()) else 0.0
            return float(value.item())
    except ImportError:
        pass
    return float(value)


class LossSpikeTripwire:
    """Halt if loss exceeds `factor` x its trailing median for `patience` steps.

    Defaults: loss > 3x its trailing median for 50 consecutive steps.
    """

    def __init__(self, factor: float = 3.0, window: int = 50, patience: int = 50,
                 min_history: int = 20):
        self.factor = factor
        self.patience = patience
        self.min_history = min_history
        self._hist: deque[float] = deque(maxlen=window)
        self._streak = 0

    def update(self, loss: float) -> float:
        """Feed one loss; raises TripwireError if the spike persists. Returns median."""
        loss = _to_scalar(loss)
        if math.isnan(loss) or math.isinf(loss):
            raise TripwireError(f"non-finite loss: {loss}")
        med = self._median()
        if len(self._hist) >= self.min_history and loss > self.factor * med:
            self._streak += 1
            if self._streak >= self.patience:
                raise TripwireError(
                    f"loss spike: {loss:.4g} > {self.factor}x median {med:.4g} "
                    f"for {self._streak} steps"
                )
        else:
            self._streak = 0
        self._hist.append(loss)
        return med

    def _median(self) -> float:
        if not self._hist:
            return float("inf")
        s = sorted(self._hist)
        n = len(s)
        mid = n // 2
        return s[mid] if n % 2 else 0.5 * (s[mid - 1] + s[mid])


class RunLogger:
    """One instance per run. Owns runs/<id>/ and enforces the logging contract above."""

    def __init__(
        self,
        run_id: str | None = None,
        config: dict[str, Any] | None = None,
        out_root: str | os.PathLike = "runs",
        data_version: str = "none",
        seed: int | None = None,
        heartbeat_every_s: float = 30.0,
        heartbeat_every_n: int | None = None,
        echo: bool = True,
    ):
        self.config = config or {}
        self.data_version = data_version
        self.seed = seed
        self.config_hash = config_hash(self.config)
        self.git_sha = git_sha()
        self.run_id = run_id or self._make_run_id()
        self.run_dir = Path(out_root) / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.heartbeat_every_s = heartbeat_every_s
        self.heartbeat_every_n = heartbeat_every_n
        self.echo = echo

        self._start = time.time()
        self._last_hb_time = 0.0
        self._last_hb_step = -(10**9)
        self._hb_path = self.run_dir / "heartbeat.log"
        self._metrics_path = self.run_dir / "metrics.jsonl"

        self._write_manifest()

    # -- construction helpers -------------------------------------------------
    def _make_run_id(self) -> str:
        ts = datetime.now().strftime("%Y%m%dT%H%M%S")
        return f"{ts}-{self.config_hash[:6]}"

    def _write_manifest(self) -> None:
        manifest = {
            "run_id": self.run_id,
            "created_at": _utc_now_iso(),
            "git_sha": self.git_sha,
            "config_hash": self.config_hash,
            "data_version": self.data_version,
            "seed": self.seed,
            "argv": sys.argv,
            "hardware": _hardware_info(),
            "config": self.config,
        }
        # Round-trip through JSON so arbitrary config/hardware values (e.g. torch's
        # TorchVersion, tuples, Paths) become YAML-safe primitives.
        safe = json.loads(json.dumps(manifest, default=str))
        with open(self.run_dir / "manifest.yaml", "w") as f:
            yaml.safe_dump(safe, f, sort_keys=False, default_flow_style=False)

    # -- heartbeat ------------------------------------------------------------
    def heartbeat(self, step: int, total: int | None = None, force: bool = False,
                  **fields: Any) -> None:
        now = time.time()
        due_time = (now - self._last_hb_time) >= self.heartbeat_every_s
        due_step = (
            self.heartbeat_every_n is not None
            and (step - self._last_hb_step) >= self.heartbeat_every_n
        )
        first = self._last_hb_time == 0.0
        if not (force or first or due_time or due_step):
            return

        elapsed = now - self._start
        parts = [f"step={step}"]
        if total:
            frac = step / total if total else 0.0
            eta = (elapsed / step * (total - step)) if step > 0 else float("nan")
            parts.append(f"{step}/{total} ({100 * frac:.1f}%)")
            parts.append(f"eta={_fmt_secs(eta)}")
        for k, v in fields.items():
            parts.append(f"{k}={_fmt_val(v)}")
        parts.append(f"elapsed={_fmt_secs(elapsed)}")
        line = "[hb] " + " ".join(parts)

        if self.echo:
            print(line, flush=True)
        with open(self._hb_path, "a") as f:
            f.write(line + "\n")

        self._last_hb_time = now
        self._last_hb_step = step

    # -- metrics stream -------------------------------------------------------
    def log_metrics(self, **fields: Any) -> dict[str, Any]:
        record = {
            "ts": _utc_now_iso(),
            "wall_s": round(time.time() - self._start, 3),
            "run_id": self.run_id,
            "git_sha": self.git_sha,
            "config_hash": self.config_hash,
            "data_version": self.data_version,
            "seed": self.seed,
            **{k: _jsonable(v) for k, v in fields.items()},
        }
        with open(self._metrics_path, "a") as f:
            f.write(json.dumps(record) + "\n")
        return record

    # -- tripwires ------------------------------------------------------------
    def tripwire_finite(self, name: str, value: Any) -> float:
        """Raise TripwireError if `value` is NaN/inf. Returns it as float otherwise."""
        v = _to_scalar(value)
        if math.isnan(v) or math.isinf(v):
            msg = f"tripwire: non-finite {name} = {value}"
            with open(self._hb_path, "a") as f:
                f.write("[tripwire] " + msg + "\n")
            raise TripwireError(msg)
        return v

    # -- end-of-run summary ---------------------------------------------------
    def summary(self, status: str = "ok", **fields: Any) -> str:
        elapsed = time.time() - self._start
        lines = [
            f"=== run {self.run_id} :: {status} ===",
            f"git={self.git_sha} config={self.config_hash} seed={self.seed} "
            f"elapsed={_fmt_secs(elapsed)}",
        ]
        for k, v in fields.items():
            lines.append(f"  {k}: {_fmt_val(v)}")
        block = "\n".join(lines)
        if self.echo:
            print(block, flush=True)
        with open(self._hb_path, "a") as f:
            f.write(block + "\n")
        return block


def _fmt_secs(s: float) -> str:
    if s != s or s in (float("inf"), float("-inf")):
        return "?"
    s = int(s)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m{sec:02d}s"
    if m:
        return f"{m}m{sec:02d}s"
    return f"{sec}s"


def _fmt_val(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)


def _jsonable(v: Any) -> Any:
    """Best-effort conversion of numpy/torch scalars to plain python for JSON."""
    try:
        import numpy as np

        if isinstance(v, (np.floating, np.integer)):
            return v.item()
        if isinstance(v, np.ndarray) and v.ndim == 0:
            return v.item()
    except ImportError:
        pass
    try:
        import torch

        if isinstance(v, torch.Tensor) and v.numel() == 1:
            return v.item()
    except ImportError:
        pass
    return v
