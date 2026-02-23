import os
import random
import numpy as np
import torch


def set_global_seed(seed: int, deterministic: bool = False) -> None:
    """Set seeds for Python, NumPy, and PyTorch for reproducibility.

    Args:
        seed: The integer seed to use.
        deterministic: If True, configure CuDNN for deterministic behavior (slower).
    """

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Configure CuDNN determinism if requested
    try:
        import torch.backends.cudnn as cudnn
        print("inside try")
        cudnn.deterministic = deterministic
        cudnn.benchmark = not deterministic
    except Exception:
        print("inside cry")
        pass 