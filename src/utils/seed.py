import os
import random
import numpy as np
import torch

def set_seed(seed: int = 42, deterministic: bool = True, benchmark: bool = False) -> None:
    """
    Set random seeds for Python, NumPy, and PyTorch (CPU/CUDA) to achieve reproducible results.

    Args:
        seed (int): The desired random seed.
        deterministic (bool): If True, sets torch.backends.cudnn.deterministic = True 
            and torch.backends.cudnn.benchmark = False to enforce reproducibility at
            the expense of performance.
        benchmark (bool): If True, sets torch.backends.cudnn.benchmark = True, 
            which can improve performance but makes results less reproducible.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = benchmark
    torch.use_deterministic_algorithms(deterministic)
    