from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class DriftAligner(nn.Module):
    """Learnable alignment map A_t for old-to-current latent representations."""

    def __init__(self, feature_dim: int, hidden_dim: int | None = None) -> None:
        super().__init__()
        hidden_dim = hidden_dim or feature_dim
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, feature_dim),
        )

    def forward(self, old_features: torch.Tensor) -> torch.Tensor:
        return self.net(old_features)


def drift_alignment_loss(
    current_features: torch.Tensor,
    old_features: torch.Tensor,
    aligner: nn.Module,
    normalize: bool = True,
) -> torch.Tensor:
    """Compute L2 drift alignment loss.

    Parameters
    ----------
    current_features:
        Features from the model currently being optimized.
    old_features:
        Features from a frozen previous model.
    aligner:
        Learnable mapping that projects old features into the current feature space.
    normalize:
        If true, apply L2 normalization before the MSE loss. This improves stability
        for medical images with high intensity variation.
    """

    projected_old = aligner(old_features.detach())
    if normalize:
        current_features = F.normalize(current_features, dim=1)
        projected_old = F.normalize(projected_old, dim=1)
    return F.mse_loss(current_features, projected_old)
