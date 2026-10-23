"""Make `src/` importable for tests and scripts even without the editable install."""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "src"))
