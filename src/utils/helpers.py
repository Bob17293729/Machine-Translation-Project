"""Helper utility functions."""

import random
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
import torch.nn as nn


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_parameters(model: nn.Module) -> int:
    """
    Count the number of trainable parameters in a model.

    Args:
        model: PyTorch model

    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(
    state: Dict[str, Any],
    filepath: str,
    is_best: bool = False,
    best_filepath: Optional[str] = None,
) -> None:
    """
    Save model checkpoint.

    Args:
        state: Dictionary containing model state, optimizer state, etc.
        filepath: Path to save checkpoint
        is_best: Whether this is the best model so far
        best_filepath: Path to save best model (if different from filepath)
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, filepath)

    if is_best and best_filepath:
        best_filepath = Path(best_filepath)
        best_filepath.parent.mkdir(parents=True, exist_ok=True)
        torch.save(state, best_filepath)


def load_checkpoint(
    filepath: str, model: nn.Module, optimizer: Optional[torch.optim.Optimizer] = None
) -> Dict[str, Any]:
    """
    Load model checkpoint.

    Args:
        filepath: Path to checkpoint file
        model: Model to load weights into
        optimizer: Optional optimizer to load state into

    Returns:
        Dictionary containing checkpoint information
    """
    checkpoint = torch.load(filepath, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint


def get_device() -> torch.device:
    """
    Get the appropriate device (CUDA if available, else CPU).

    Returns:
        torch.device object
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

