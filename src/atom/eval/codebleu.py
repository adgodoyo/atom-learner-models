"""CodeBLEU wrapper for evaluating the CodeWorkout generative simulator.

Reuses, rather than forks, TIKTOC's vendored CodeBLEU implementation
(`Datasets/tiktoc-main/evaluator/CodeBLEU/`, itself an unmodified copy of Microsoft's
CodeXGLUE CodeBLEU), following the same approach as `scripts/cw_5_functional_equivalence.py`
for TIKTOC's compiler harness: import the vendored code directly (pure Python plus a
prebuilt tree-sitter grammar `.so`, with no torch/hydra/wandb dependencies) and never
reimplement it.

Platform note: the vendored `parser/my-languages.so` is an ELF x86-64 Linux shared
object. It loads, and the syntax/dataflow-match components run, on Linux x86_64, but not
on arm64 macOS, where `Language(...)` raises `OSError`. `codebleu_available()` lets a
caller detect this and skip syntax/dataflow-dependent assertions while still exercising
the pure-Python ngram/weighted-ngram path locally.
"""
from __future__ import annotations

import pathlib
import sys

TIKTOC_EVALUATOR = pathlib.Path(__file__).resolve().parents[3].parent / "Datasets" / "tiktoc-main" / "evaluator"

_import_error: Exception | None = None
try:
    if str(TIKTOC_EVALUATOR) not in sys.path:
        sys.path.insert(0, str(TIKTOC_EVALUATOR))
    from CodeBLEU.calc_code_bleu import get_codebleu as _get_codebleu  # noqa: E402
except Exception as e:  # pragma: no cover -- exercised by the "unavailable" branch
    _get_codebleu = None
    _import_error = e


def codebleu_available() -> bool:
    """False on a platform where the vendored tree-sitter `.so` cannot load (arm64
    macOS); True on x86_64 Linux."""
    if _get_codebleu is None:
        return False
    try:
        _get_codebleu([["public int f(){return 1;}"]], ["public int f(){return 1;}"], "java")
        return True
    except Exception:
        return False


def codebleu_one(reference: str, hypothesis: str, lang: str = "java") -> dict:
    """One (reference, hypothesis) pair -> `{codebleu, ngram_match, weighted_ngram_match,
    syntax_match, dataflow_match}`, all floats. Raises the ORIGINAL underlying exception
    (import or runtime) if the vendored evaluator is unusable on this platform; callers
    on macOS should check `codebleu_available()` first rather than catching broadly."""
    if _get_codebleu is None:
        raise RuntimeError(f"CodeBLEU evaluator unavailable on this platform: {_import_error}")
    score, (ngram, wngram, syntax, dataflow) = _get_codebleu([[reference]], [hypothesis], lang)
    return {
        "codebleu": float(score), "ngram_match": float(ngram), "weighted_ngram_match": float(wngram),
        "syntax_match": float(syntax), "dataflow_match": float(dataflow),
    }
