#!/usr/bin/env python3
"""Standalone CodeBLEU scorer for generated code, run under a separate Python 3.11 interpreter.

`atom.eval.codebleu` wraps the vendored CodeXGLUE CodeBLEU evaluator (unmodified), which is
incompatible with Python 3.12 (`fractions.Fraction(..., _normalize=False)` was removed) and
with `tree_sitter>=0.22` (the two-argument `Language(path, name)` constructor was removed).
The package targets Python >=3.11,<3.12, so this script runs in a small separate venv with
Python 3.11 and `tree_sitter==0.20.4` (no torch needed: scoring is string/AST comparison),
while the generation pass keeps its own CUDA environment.
Input (stdin): a JSON array of `{"id", "reference", "hypothesis"}`.
Output (stdout): a JSON array of `{"id", "codebleu", "ngram_match", "weighted_ngram_match",
"syntax_match", "dataflow_match"}`, one entry per row with a non-empty hypothesis (empty
ones are skipped, as in `cw_sim_launch2.score_records`). No code text is logged or written;
the caller (`cw_ext_1_generate.py`) must not print the texts either.
Run:  echo '[{"id": 0, "reference": "int a=1;", "hypothesis": "int a=2;"}]' \
          | <py311-venv>/bin/python3 infra/cw_ext_1_codebleu_score.py
Serves: thesis Table 11 (generation).
"""
from __future__ import annotations

import json
import pathlib
import contextlib
import io
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from atom.eval.codebleu import codebleu_available, codebleu_one  # noqa: E402


def main() -> int:
    rows = json.load(sys.stdin)
    real_stdout = sys.stdout
    # The vendored evaluator's own `dataflow_match.py` prints a diagnostic WARNING
    # straight to stdout on corpora with no extractable data-flow (a real, harmless
    # condition on a single-pair call, not an error), which would corrupt the JSON this
    # script must emit on stdout for its caller to parse. The vendored code is left
    # unmodified; instead every `codebleu_one` call is wrapped in
    # `redirect_stdout` to a throwaway buffer, and the ONLY thing ever written to the
    # real stdout is this script's own final JSON array, at the very end.
    with contextlib.redirect_stdout(io.StringIO()):
        if not codebleu_available():
            available = False
        else:
            available = True
    if not available:
        print(json.dumps({"error": "codebleu_available() is False in this interpreter"}), file=sys.stderr)
        return 1

    out = []
    for row in rows:
        hyp = str(row["hypothesis"]).strip()
        if not hyp:
            continue
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                cb = codebleu_one(str(row["reference"]), hyp)
        except Exception as e:
            print(f"[cw-ext-1-codebleu] row {row.get('id')} failed: {e}", file=sys.stderr)
            continue
        out.append({"id": row["id"], **cb})
    json.dump(out, real_stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
