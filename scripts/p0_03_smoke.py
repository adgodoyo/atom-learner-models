#!/usr/bin/env python
"""Synthetic end-to-end pipeline smoke test, used by `make smoke`.

Runs `atom.smoke.run_smoke` on fake data with the settings in `configs/smoke.yaml` (if present)
and checks that it passes in under 5 minutes on a laptop CPU. Writes `reports/p0_03_smoke.md`
and a run directory. Run: `make smoke` or `python scripts/p0_03_smoke.py`.
"""

from __future__ import annotations

import pathlib
import sys

# Runnable directly (not only via the editable install).
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import yaml  # noqa: E402

from atom.smoke.run_smoke import run_smoke  # noqa: E402

CONFIG_PATH = pathlib.Path(__file__).resolve().parents[1] / "configs" / "smoke.yaml"


def main() -> int:
    config = {}
    if CONFIG_PATH.exists():
        config = yaml.safe_load(CONFIG_PATH.read_text()) or {}
    result = run_smoke(config=config)
    print()
    print(f"smoke: {'PASS' if result['passed'] else 'FAIL'} "
          f"(val_auc={result['val_auc']:.3f}, wall={result['wall_s']:.1f}s)")
    print(f"report: reports/p0_03_smoke.md · run: {result['run_dir']}")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
