import random

import numpy as np
import torch


def seed_everything(seed: int = 42) -> None:
    """Seed Python, NumPy, and PyTorch for reproducible CPU and CUDA execution."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False