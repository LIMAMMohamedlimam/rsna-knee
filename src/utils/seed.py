"""Seeding. Called at every entrypoint (CLAUDE.md §3.1)."""

from __future__ import annotations

import os
import random

import numpy as np


def set_seed(seed: int, deterministic: bool = True) -> int:
    """Seed python, numpy, and (if installed) torch. Returns the seed for logging."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:  # torch is an optional extra until Spec 04
        import torch
    except ImportError:
        pass
    else:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    try:  # langdetect is non-deterministic unless its factory is seeded
        from langdetect import DetectorFactory
    except ImportError:
        pass
    else:
        DetectorFactory.seed = 0

    return seed
