import os
import random
from typing import Optional

import numpy as np

try:
    import cupy as cp  # type: ignore
except Exception:  # CuPy might be unavailable
    cp = None


def set_global_seed(seed: Optional[int]):
    """
    Seed Python's RNG plus NumPy/CuPy generators (when available).
    """
    if seed is None:
        return
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    if cp is not None:
        try:
            cp.random.seed(seed)
        except Exception:
            pass
