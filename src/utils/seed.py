from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np


def set_global_seed(seed: Optional[int]) -> int:
    """
    Establece semillas reproducibles para los módulos que utilizamos.

    Devuelve la semilla efectiva (útil para logging).
    """
    if seed is None:
        seed = 42

    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    return seed
