"""Test-session hooks.

1. Skip tests that need the gated Gemma 4 E4B tokenizer when it is not in the local Hugging Face
   cache. Accept the Gemma Terms of Use on the model page and log in (`huggingface-cli login`) to
   run them.
2. Four canaries assert bitwise equality, or a 1e-7 tolerance with no relative slack, between
   computations that differ only in batch shape. That holds on Apple silicon, where the thesis
   ran; on x86-64 the CPU matrix kernels can round the last bit differently by batch shape and
   by processor (2.98e-08 to 1.04e-07 observed), so off Apple silicon they are marked as
   expected failures, not removed.
"""
from __future__ import annotations

import functools
import inspect
import platform

import pytest

GATED_MODEL = "google/gemma-4-E4B"
BITWISE_EXACT_ON_APPLE_SILICON = {
    "tests/test_cw_attempt_encoder.py::test_prefix_truncation_invariance_canary_exact_zero",
    "tests/test_fa_alt2.py::test_forward_global_control_broadcasts_one_c_and_ignores_u_entirely",
    "tests/test_planner_replay_sim3.py::test_shared_belief_fast_path_matches_per_probe_to_float32_tolerance",
    "tests/test_planner_replay_v2.py::test_all_probe_rows_share_one_belief",
}


@functools.lru_cache(maxsize=1)
def _tokenizer_cached() -> bool:
    try:
        from transformers import AutoTokenizer

        AutoTokenizer.from_pretrained(GATED_MODEL, local_files_only=True)
        return True
    except Exception:
        return False


def _needs_tokenizer(item) -> bool:
    if "real_tokenizer" in getattr(item, "fixturenames", ()):
        return True
    try:
        src = inspect.getsource(item.function)
    except (OSError, TypeError, AttributeError):
        return False
    return GATED_MODEL in src and "AutoTokenizer" in src


def pytest_runtest_setup(item):
    if _needs_tokenizer(item) and not _tokenizer_cached():
        pytest.skip(f"{GATED_MODEL} tokenizer not in the local Hugging Face cache")


def pytest_collection_modifyitems(config, items):
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        return
    for item in items:
        if item.nodeid in BITWISE_EXACT_ON_APPLE_SILICON:
            item.add_marker(pytest.mark.xfail(
                reason="bitwise-exact on Apple silicon; last-bit rounding differs by batch shape on x86-64",
                strict=False))
