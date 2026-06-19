"""Deterministic seeding and device selection."""
from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed all RNGs used in the project for reproducible runs."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(prefer: str = "auto") -> torch.device:
    """Resolve a torch device. ``prefer`` is one of {auto, cpu, cuda}."""
    if prefer == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available() and prefer in ("auto", "cuda", "gpu"):
        return torch.device("cuda")
    return torch.device("cpu")
