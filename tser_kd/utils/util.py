import random
import torch
import numpy as np


def setup_seed(seed: int = None) -> None:
    """Sets random seeds for Python, NumPy, and PyTorch to ensure reproducible results.

    If no seed is provided, a random 32-bit integer is generated.
    Additionally, PyTorch's CuDNN backend is configured for deterministic behavior
    and benchmarking is disabled to avoid nondeterminism.

    Args:
        seed: The seed value to use.
    """
    # Randomly generate a seed if not passed
    if seed is None:
        seed = random.randint(0, 2 ** 32 - 1)

    print(f"Random seed: {seed}")

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
