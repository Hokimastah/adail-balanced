from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class PrototypeBank(nn.Module):
    """Prototype storage and losses for ADAIL.

    The prototypes are registered as buffers, not trainable parameters. They are
    updated by exponential moving average after observing task data.
    """

    def __init__(
        self,
        num_classes: int,
        feature_dim: int,
        momentum: float = 0.9,
        margin: float = 1.0,
    ) -> None:
        super().__init__()
        if not 0 <= momentum < 1:
            raise ValueError("momentum must be in [0, 1)")
        self.momentum = momentum
        self.margin = margin
        self.register_buffer("vectors", torch.zeros(num_classes, feature_dim))
        self.register_buffer("initialized", torch.zeros(num_classes, dtype=torch.bool))

    def expand(self, new_num_classes: int) -> None:
        current = self.vectors.shape[0]
        if new_num_classes <= current:
            return
        extra_vectors = self.vectors.new_zeros(new_num_classes - current, self.vectors.shape[1])
        extra_flags = self.initialized.new_zeros(new_num_classes - current)
        self.vectors = torch.cat([self.vectors, extra_vectors], dim=0)
        self.initialized = torch.cat([self.initialized, extra_flags], dim=0)

    @torch.no_grad()
    def update(self, features: torch.Tensor, labels: torch.Tensor) -> None:
        """Update prototypes with class-wise EMA."""

        if features.numel() == 0:
            return
        features = features.detach()
        for cls in labels.unique().tolist():
            cls_int = int(cls)
            if cls_int < 0 or cls_int >= self.vectors.shape[0]:
                continue
            mask = labels == cls_int
            if not mask.any():
                continue
            mean_feature = features[mask].mean(dim=0)
            if not bool(self.initialized[cls_int]):
                self.vectors[cls_int].copy_(mean_feature)
                self.initialized[cls_int] = True
            else:
                self.vectors[cls_int].mul_(self.momentum).add_(mean_feature, alpha=1 - self.momentum)

    def compactness_loss(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Pull features toward their class prototypes."""

        if features.numel() == 0:
            return features.new_tensor(0.0)
        valid = (labels >= 0) & (labels < self.vectors.shape[0]) & self.initialized[labels.clamp_min(0)]
        if not valid.any():
            return features.new_tensor(0.0)
        features_valid = F.normalize(features[valid], dim=1)
        proto_valid = F.normalize(self.vectors[labels[valid]], dim=1)
        return F.mse_loss(features_valid, proto_valid)

    def separation_loss(self) -> torch.Tensor:
        """Push initialized class prototypes apart by a minimum margin."""

        active = self.vectors[self.initialized]
        if active.shape[0] < 2:
            return self.vectors.new_tensor(0.0)
        active = F.normalize(active, dim=1)
        distances = torch.cdist(active, active, p=2)
        mask = ~torch.eye(distances.shape[0], dtype=torch.bool, device=distances.device)
        penalties = F.relu(self.margin - distances[mask]) ** 2
        if penalties.numel() == 0:
            return self.vectors.new_tensor(0.0)
        return penalties.mean()
