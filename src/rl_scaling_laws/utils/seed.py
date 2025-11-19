"""Random seed utilities for reproducibility."""

import os
import random
from typing import Optional

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Set random seeds for reproducibility across all libraries.

    Args:
        seed: The random seed to use.
        deterministic: If True, enable deterministic CUDA operations (slower but reproducible).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = True


def get_random_seed() -> int:
    """Generate a random seed based on system entropy.

    Returns:
        A random integer seed.
    """
    return int.from_bytes(os.urandom(4), byteorder="big")
