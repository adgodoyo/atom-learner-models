"""Shared helpers for the CodeWorkout filter-evaluation scripts.

Importing this module puts the release's `src/`, `scripts/` and `infra/` directories on
`sys.path`, so the scripts that import it can reach the `atom` package and each other.
"""
from __future__ import annotations
import pathlib
import sys
REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "infra"))
