"""Utility functions for reproducible execution and benchmarking."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int = 42, deterministic: bool = True) -> None:
    """Seed Python, NumPy, and PyTorch for reproducible CPU and GPU execution.

    Args:
        seed: Integer seed value to set across all random number generators.
        deterministic: Whether to configure PyTorch/cuDNN for deterministic execution.
            When True, cuDNN benchmark is disabled and deterministic algorithms are
            enforced.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch, "mps") and hasattr(torch.mps, "manual_seed"):
        try:
            torch.mps.manual_seed(seed)
        except Exception:
            pass

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
