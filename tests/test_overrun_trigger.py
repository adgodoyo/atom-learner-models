"""Tests for the live overrun trigger (`atom.utils.overrun_trigger`): `.check()`'s pure projection
logic, and that `._fire()` writes its report and shuts down exactly once. `subprocess.run`
and `boto3` are mocked throughout -- this must never issue a real `sudo shutdown` from a
test run."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from atom.utils.overrun_trigger import OverrunTrigger


def test_check_does_not_fire_when_projection_is_under_the_estimate(tmp_path):
    trigger = OverrunTrigger(estimated_minutes=100.0, out_dir=tmp_path)
    with patch("atom.utils.overrun_trigger.subprocess.run") as run:
        # 10 of 100 units done in 5 minutes -> projected 50 min, well under 2x100=200.
        fired = trigger.check(units_done=10, units_total=100, elapsed_s=5 * 60.0)
    assert fired is False
    assert trigger.fired is False
    run.assert_not_called()


def test_check_fires_when_projection_exceeds_2x_estimate(tmp_path):
    trigger = OverrunTrigger(estimated_minutes=10.0, out_dir=tmp_path, label="unit-test")
    with patch("atom.utils.overrun_trigger.subprocess.run") as run:
        # 1 of 100 units done in 5 minutes -> projected 500 min, > 2*10=20.
        fired = trigger.check(units_done=1, units_total=100, elapsed_s=5 * 60.0)
    assert fired is True
    assert trigger.fired is True
    run.assert_called_once_with(["sudo", "shutdown", "-h", "now"], check=False)
    report = json.loads((tmp_path / "overrun_trigger.json").read_text())
    assert report["label"] == "unit-test"
    assert report["overrun_ratio"] > 2.0


def test_check_fires_at_most_once(tmp_path):
    trigger = OverrunTrigger(estimated_minutes=1.0, out_dir=tmp_path)
    with patch("atom.utils.overrun_trigger.subprocess.run") as run:
        first = trigger.check(units_done=1, units_total=100, elapsed_s=60.0)
        second = trigger.check(units_done=2, units_total=100, elapsed_s=120.0)
    assert first is True
    assert second is False  # already fired -- no double shutdown, no double report write
    run.assert_called_once()


@pytest.mark.parametrize("units_done,units_total", [(0, 100), (10, 0), (-1, 100)])
def test_check_is_a_noop_before_any_real_progress(tmp_path, units_done, units_total):
    trigger = OverrunTrigger(estimated_minutes=1.0, out_dir=tmp_path)
    with patch("atom.utils.overrun_trigger.subprocess.run") as run:
        fired = trigger.check(units_done=units_done, units_total=units_total, elapsed_s=1e9)
    assert fired is False
    run.assert_not_called()


def test_fire_with_bucket_set_still_writes_local_report_and_shuts_down(tmp_path):
    # boto3 is only ever installed on the cloud instance -- a local environment may have no boto3, so
    # the S3 upload path is exercised for graceful degradation here (`_fire`'s own
    # try/except), not for the exact boto3 call shape. `--bucket` given but no upload
    # possible must never block the local report write or the shutdown call.
    trigger = OverrunTrigger(estimated_minutes=1.0, out_dir=tmp_path, bucket="my-bucket",
                             out_prefix="cw-5-full-run", label="cw-5-eval")
    with patch("atom.utils.overrun_trigger.subprocess.run") as run:
        fired = trigger.check(units_done=1, units_total=1000, elapsed_s=60.0)
    assert fired is True
    run.assert_called_once_with(["sudo", "shutdown", "-h", "now"], check=False)
    report = json.loads((tmp_path / "overrun_trigger.json").read_text())
    assert report["label"] == "cw-5-eval"
