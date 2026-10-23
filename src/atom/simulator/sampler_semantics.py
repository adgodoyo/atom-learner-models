"""Sampler-semantics version marker for simulator checkpoints, and the loader guard that
checks it.

Every simulator checkpoint records how its belief samples were drawn. The guard refuses to
resume, warm-start, fine-tune or re-score a checkpoint whose marker is missing or differs
from the current one, so a model trained under one sampling convention is never silently
evaluated under another. The check lives on the loader rather than in documentation so
that every run enforces it.

Scope: simulator checkpoints only, i.e. anything trained through
`atom.simulator.fa_train_step` / `cw_train_step` / `fa_akt_matched_arm.py`, whose belief
samples pass through `reparameterize_from_log_sigma`. Filter-only, planner and binding
checkpoints never use that path and do not carry the marker.
"""

from __future__ import annotations

from typing import Any

# Bump this string on any future change to how a simulator checkpoint's belief samples are
# drawn (e.g. a different reparameterisation, a different K-axis convention at the save
# boundary) -- every checkpoint saved under the OLD string becomes unloadable for resume/
# warm-start/fine-tune the moment this changes, by design.
CURRENT_SAMPLER_SEMANTICS = "reparameterize_from_log_sigma_v1_2026-08-15"


def require_current_sampler_semantics(state: dict[str, Any], context: str = "") -> None:
    """Raise if `state` is missing the marker or carries a stale one. Call this before
    resuming, warm-starting, fine-tuning, or re-scoring ANY simulator checkpoint, so that
    checkpoints trained under an earlier sampling convention cannot be reused by mistake.
    """
    where = f" ({context})" if context else ""
    version = state.get("sampler_semantics")
    if version is None:
        raise ValueError(
            f"Refusing to load simulator checkpoint{where}: no 'sampler_semantics' field. "
            "This checkpoint predates the sampler correction and is presumed trained "
            "under the earlier sampler (a filter's raw log_sigma output was used as "
            "sigma directly, with no .exp()). Resuming, warm-starting, fine-tuning, or "
            "re-scoring it under the fixed sampler is not safe: a model trained at "
            "log_sigma~0 has only ever seen near-zero noise and will collapse when it "
            "suddenly receives sigma~1.0 noise, which will look like the fix broke the "
            "model rather than like what it is. Retrain instead.")
    if version != CURRENT_SAMPLER_SEMANTICS:
        raise ValueError(
            f"Refusing to load simulator checkpoint{where}: sampler_semantics={version!r} "
            f"does not match the current {CURRENT_SAMPLER_SEMANTICS!r}. Retrain rather "
            "than resume across a sampler-semantics change.")
