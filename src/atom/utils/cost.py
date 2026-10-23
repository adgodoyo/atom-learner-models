"""Cloud cost ledger helpers.

`ledger/cost_ledger.csv` gets one row per cloud session, written before the session
(estimate) and updated after it (actual, from the provider's billing data). The running
total is recomputed on every append. `estimate` applies a 1.5x safety factor, and
`STOP_AND_ASK_USD` is the projected single-run cost above which a launch needs human
approval.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

FIELDS = [
    "date",
    "phase",
    "instance",
    "hours",
    "usd_per_hour",
    "purpose",
    "est_usd",
    "actual_usd",
    "running_total_usd",
    "note",
]

STOP_AND_ASK_USD = 15.0  # projected single-run cost above this requires human approval


def estimate(est_hours: float, usd_per_hour: float, safety: float = 1.5) -> float:
    """Projected cost with a 1.5x safety factor."""
    return round(est_hours * usd_per_hour * safety, 2)


def init_ledger(path: str | Path) -> Path:
    """Create the ledger with a header row if it does not yet exist."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if not p.exists():
        with open(p, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=FIELDS).writeheader()
    return p


def _rows(path: str | Path) -> list[dict[str, str]]:
    p = Path(path)
    if not p.exists():
        return []
    with open(p, newline="") as f:
        return list(csv.DictReader(f))


def running_total(path: str | Path) -> float:
    total = 0.0
    for r in _rows(path):
        val = r.get("actual_usd") or r.get("est_usd") or "0"
        try:
            total += float(val)
        except ValueError:
            continue
    return round(total, 2)


def append_row(
    path: str | Path,
    date: str,
    phase: str,
    purpose: str,
    instance: str = "-",
    hours: float = 0.0,
    usd_per_hour: float = 0.0,
    est_usd: float | None = None,
    actual_usd: float | None = None,
    note: str = "",
) -> dict[str, Any]:
    """Append one ledger row and recompute the running total column."""
    init_ledger(path)
    if est_usd is None:
        est_usd = estimate(hours, usd_per_hour) if hours else 0.0
    prior = running_total(path)
    increment = actual_usd if actual_usd is not None else est_usd
    row = {
        "date": date,
        "phase": phase,
        "instance": instance,
        "hours": hours,
        "usd_per_hour": usd_per_hour,
        "purpose": purpose,
        "est_usd": est_usd,
        "actual_usd": "" if actual_usd is None else actual_usd,
        "running_total_usd": round(prior + (increment or 0.0), 2),
        "note": note,
    }
    with open(path, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=FIELDS).writerow(row)
    return row
