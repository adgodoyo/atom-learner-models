#!/usr/bin/env python
"""Thesis Table 11, all-50-problem CodeBLEU: the row-weighted mean of the three generation strata.

Inputs (published aggregates): results/cw_ext_1/result.json (the 30 problems outside the headline
and held-out sets), results/cw_sim_launch2/headline/summary.json (13 headline problems) and
results/cw_sim_launch2/heldout/summary.json (7 problems held out of training); personalised arm,
argmax decoding, blind. Output: results/cw_codebleu_all50/summary.json. Run with no arguments.
"""
from __future__ import annotations

import json
import pathlib

REPO = pathlib.Path(__file__).resolve().parents[1]
CELL = "personalised__argmax"


def stratum(path: str) -> tuple[float, int]:
    cb = json.loads((REPO / path).read_text())["codebleu"]
    cells = cb["mean_by_arm_mode"]
    return float(cells[CELL]), int(cb["n_scored"]) // len(cells)  # rows per arm x mode cell


def main() -> None:
    ext = json.loads((REPO / "results" / "cw_ext_1" / "result.json").read_text())
    parts = {"other_30": (float(ext["thirty_problem_codebleu"]), int(ext["n_scored"])),
             "headline_13": stratum("results/cw_sim_launch2/headline/summary.json"),
             "heldout_7": stratum("results/cw_sim_launch2/heldout/summary.json")}
    rows = sum(n for _, n in parts.values())
    pooled = sum(m * n for m, n in parts.values()) / rows
    out = {"codebleu_all50": pooled, "rows": rows,
           "strata": {k: {"mean": m, "rows": n} for k, (m, n) in parts.items()}}
    d = REPO / "results" / "cw_codebleu_all50"
    d.mkdir(parents=True, exist_ok=True)
    (d / "summary.json").write_text(json.dumps(out, indent=1) + "\n")
    print(f"all-50 CodeBLEU {pooled:.4f} over {rows} rows " +
          ", ".join(f"{k} {m:.4f} (n={n})" for k, (m, n) in parts.items()))


if __name__ == "__main__":
    main()
