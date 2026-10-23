"""Global seed control.

`seed_everything` is called once at the top of every run and the seed is written into
the run manifest. Torch is optional
here so this module imports cleanly in a minimal environment.
"""

from __future__ import annotations

import os
import random

import numpy as np


def seed_everything(seed: int, deterministic: bool = True) -> int:
    """Seed Python, NumPy and (if present) Torch. Returns the seed for logging."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # Apple Silicon MPS shares the CPU generator seed via manual_seed above.
        if deterministic:
            # Best-effort determinism; some ops have no deterministic kernel on MPS.
            torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:
        pass

    return seed


def torch_generator(seed: int):
    """A seeded torch.Generator for DataLoader/sampling reproducibility."""
    import torch

    g = torch.Generator()
    g.manual_seed(seed)
    return g
