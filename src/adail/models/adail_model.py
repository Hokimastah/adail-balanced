from __future__ import annotations

import copy

import torch
from torch import nn

from adail.modules import DriftAligner, PrototypeBank
from .backbone import build_resnet18_backbone


class ADAILNet(nn.Module):
    """Backbone, classifier, drift aligner, and prototype bank for ADAIL."""

    def __init__(
        self,
        num_classes: int,
        backbone: nn.Module | None = None,
        feature_dim: int | None = None,
        pretrained_backbone: bool = True,
        prototype_momentum: float = 0.9,
        prototype_margin: float = 1.0,
    ) -> None:
        super().__init__()
        if backbone is None:
            backbone, inferred_dim = build_resnet18_backbone(pretrained=pretrained_backbone)
            feature_dim = inferred_dim if feature_dim is None else feature_dim
        if feature_dim is None:
            raise ValueError("feature_dim must be provided when using a custom backbone")

        self.backbone = backbone
        self.feature_dim = feature_dim
        self.classifier = nn.Linear(feature_dim, num_classes)
        self.drift_aligner = DriftAligner(feature_dim)
        self.prototypes = PrototypeBank(
            num_classes=num_classes,
            feature_dim=feature_dim,
            momentum=prototype_momentum,
            margin=prototype_margin,
        )
        # Keep the frozen teacher out of PyTorch's module registry to avoid saving
        # a duplicated model in checkpoints or sending it to optimizers by accident.
        object.__setattr__(self, "_old_model", None)

    @property
    def old_model(self) -> "ADAILNet | None":
        return getattr(self, "_old_model", None)

    @property
    def num_classes(self) -> int:
        return self.classifier.out_features

    def forward(self, x: torch.Tensor, return_features: bool = False):
        features = self.backbone(x)
        if features.ndim > 2:
            features = torch.flatten(features, 1)
        logits = self.classifier(features)
        if return_features:
            return logits, features
        return logits

    @torch.no_grad()
    def freeze_old_model(self) -> None:
        """Snapshot the current model as frozen teacher for the next CIL task."""

        old = copy.deepcopy(self)
        object.__setattr__(old, "_old_model", None)
        old.eval()
        for param in old.parameters():
            param.requires_grad = False
        object.__setattr__(self, "_old_model", old)

    def clear_old_model(self) -> None:
        """Remove the frozen teacher reference."""

        object.__setattr__(self, "_old_model", None)

    def expand_classes(self, new_num_classes: int) -> None:
        """Expand classifier/prototypes while preserving old weights."""

        if new_num_classes <= self.num_classes:
            return
        old_classifier = self.classifier
        new_classifier = nn.Linear(self.feature_dim, new_num_classes).to(old_classifier.weight.device)
        with torch.no_grad():
            new_classifier.weight[: old_classifier.out_features].copy_(old_classifier.weight)
            new_classifier.bias[: old_classifier.out_features].copy_(old_classifier.bias)
        self.classifier = new_classifier
        self.prototypes.expand(new_num_classes)
