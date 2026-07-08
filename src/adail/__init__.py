"""ADAIL: Adaptive Drift-Aware Incremental Learning.

A PyTorch research library for class-incremental medical image classification.
"""

from .config import ADAILConfig, DataConfig, OptimConfig
from .metrics import (
    accuracy,
    average_accuracy,
    average_forgetting,
    mean_absolute_error,
    per_class_accuracy,
    quadratic_weighted_kappa,
)
from .models import ADAILNet
from .trainer import ADAILTrainer
from .baselines import build_lo2ln_command, get_lo2ln_experiment

__all__ = [
    "ADAILConfig",
    "DataConfig",
    "OptimConfig",
    "ADAILNet",
    "ADAILTrainer",
    "build_lo2ln_command",
    "get_lo2ln_experiment",
    "accuracy",
    "average_accuracy",
    "average_forgetting",
    "mean_absolute_error",
    "per_class_accuracy",
    "quadratic_weighted_kappa",
]

__version__ = "0.3.0"
