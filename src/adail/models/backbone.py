from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn
from torchvision import models


class Identity(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return x


def build_resnet18_backbone(
    pretrained: bool = True,
    weights: models.ResNet18_Weights | None = None,
) -> tuple[nn.Module, int]:
    """Build a ResNet-18 feature extractor.

    Parameters
    ----------
    pretrained:
        If true, use ImageNet weights. Ignored when `weights` is explicitly given.
    weights:
        Torchvision weight enum. Set to `None` for random initialization.
    """

    if weights is None and pretrained:
        weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)
    feature_dim = model.fc.in_features
    model.fc = Identity()
    return model, feature_dim


class LambdaBackbone(nn.Module):
    """Wrap a callable as a simple backbone.

    Useful in tests and for users who already have a feature extractor function.
    """

    def __init__(self, fn: Callable[[torch.Tensor], torch.Tensor], feature_dim: int):
        super().__init__()
        self.fn = fn
        self.feature_dim = feature_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fn(x)
