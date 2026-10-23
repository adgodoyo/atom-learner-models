from atom.utils.cost import append_row, estimate, init_ledger, running_total


def test_estimate_applies_safety_factor():
    assert estimate(4.0, 1.0, safety=1.5) == 6.0


def test_init_and_append_running_total(tmp_path):
    ledger = tmp_path / "cost_ledger.csv"
    init_ledger(ledger)
    assert ledger.exists()
    assert running_total(ledger) == 0.0

    append_row(ledger, date="2026-07-18", phase="P0", purpose="scaffold",
               est_usd=0.0, actual_usd=0.0)
    append_row(ledger, date="2026-07-21", phase="P3", purpose="AKT baseline",
               instance="g5.xlarge", hours=6.0, usd_per_hour=0.5, est_usd=4.5)
    assert running_total(ledger) == 4.5


def test_actual_supersedes_estimate_in_total(tmp_path):
    ledger = tmp_path / "cost_ledger.csv"
    append_row(ledger, date="2026-07-21", phase="P5", purpose="pilot",
               est_usd=5.0, actual_usd=3.2)
    assert running_total(ledger) == 3.2
