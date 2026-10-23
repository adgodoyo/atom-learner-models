"""Live overrun trigger for unattended cloud runs.

A cloud launch projects its own total duration from its own measured rate at each
progress check and, if that projection exceeds 2x its pre-registered estimate, writes a
report and shuts the instance down. It is a broken-run detector, not a budget: a run
projecting far beyond its estimate is presumptively mis-sized or stuck, and a fixed
threshold on the estimate alone cannot detect that.

Shared by the launch scripts (for example `infra/cw_5_full_run.py`) so there is one
implementation rather than one per script.
"""

from __future__ import annotations

import json
import pathlib
import subprocess


class OverrunTrigger:
    """Call `.check(units_done, units_total, elapsed_s)` periodically (once per heartbeat,
    once per eval arm, once per fold -- whatever progress unit the caller has at hand; only
    the RATIO `units_done / units_total` matters, not what the units are). The first call
    with `units_done > 0` records the measured rate and its resulting projection; every
    call re-checks that projection against `2 * estimated_minutes`. Fires AT MOST ONCE.

    On fire: writes a report to `out_dir/overrun_trigger.json` (+ S3 if `bucket` is given),
    then self-terminates the instance UNCONDITIONALLY -- not gated by `--self-terminate`,
    because a run projecting >2x its own pre-registered estimate is presumptively broken,
    and billing further is exactly the failure mode this exists to prevent."""

    def __init__(self, estimated_minutes: float, out_dir: pathlib.Path,
                bucket: str = "", out_prefix: str = "", label: str = "run"):
        self.estimated_minutes = estimated_minutes
        self.out_dir = out_dir
        self.bucket = bucket
        self.out_prefix = out_prefix
        self.label = label
        self.fired = False
        self.first_check: dict | None = None

    def check(self, units_done: float, units_total: float, elapsed_s: float) -> bool:
        """Returns True iff the trigger fires on THIS call (caller should stop and, for a
        real launch, let the instance terminate itself -- `_fire` already issued the
        shutdown). Returns False (never fires) for `units_done <= 0` or `units_total <= 0`
        -- there is nothing to project a rate from yet."""
        if self.fired or units_done <= 0 or units_total <= 0:
            return False
        projected_minutes = (elapsed_s / 60.0) * (units_total / units_done)
        if self.first_check is None:
            self.first_check = {
                "units_done": units_done, "units_total": units_total,
                "elapsed_s": elapsed_s, "projected_minutes": projected_minutes,
            }
        if projected_minutes > 2.0 * self.estimated_minutes:
            self._fire(units_done, units_total, elapsed_s, projected_minutes)
            return True
        return False

    def _fire(self, units_done: float, units_total: float, elapsed_s: float,
             projected_minutes: float) -> None:
        self.fired = True
        report = {
            "label": self.label, "estimated_minutes": self.estimated_minutes,
            "projected_minutes": projected_minutes,
            "overrun_ratio": projected_minutes / max(self.estimated_minutes, 1e-9),
            "units_done": units_done, "units_total": units_total, "elapsed_s": elapsed_s,
            "first_check": self.first_check,
        }
        print(f"[overrun-trigger] {self.label}: projected {projected_minutes:.1f} min "
             f"> 2x estimate ({self.estimated_minutes:.1f} min) -- SELF-TERMINATING", flush=True)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        report_path = self.out_dir / "overrun_trigger.json"
        report_path.write_text(json.dumps(report, indent=2))
        if self.bucket:
            try:
                import boto3
                boto3.client("s3").upload_file(
                    str(report_path), self.bucket, f"{self.out_prefix}/overrun_trigger.json")
            except Exception as e:  # pragma: no cover -- best-effort, never blocks shutdown
                print(f"[overrun-trigger] S3 upload of the report failed (non-fatal): {e}", flush=True)
        subprocess.run(["sudo", "shutdown", "-h", "now"], check=False)
