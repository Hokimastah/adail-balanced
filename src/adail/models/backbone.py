from __future__ import annotations

import copy
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


class SelectiveExpandableResNet18(nn.Module):
    """ResNet18 with frozen shared stages and expandable high-level branches.

    The stem through ``layer3`` is shared. A copy of ``layer4`` is introduced
    at every incremental task, while previous branches are frozen. New branch
    features enter the fixed 512-dimensional representation as gated residuals,
    avoiding the ever-growing prototype dimensionality of raw concatenation.
    """

    feature_dim = 512
    supports_expansion = True

    def __init__(
        self,
        pretrained: bool = True,
        weights: models.ResNet18_Weights | None = None,
        gate_init: float = -2.0,
        freeze_shared_after_base: bool = True,
    ) -> None:
        super().__init__()
        if weights is None and pretrained:
            weights = models.ResNet18_Weights.DEFAULT
        base = models.resnet18(weights=weights)
        self.shared = nn.Sequential(
            base.conv1,
            base.bn1,
            base.relu,
            base.maxpool,
            base.layer1,
            base.layer2,
            base.layer3,
        )
        self.branches = nn.ModuleList([base.layer4])
        self.projections = nn.ModuleList()
        self.residual_gates = nn.ParameterList()
        self.pool = base.avgpool
        self.gate_init = float(gate_init)
        self.freeze_shared_after_base = bool(freeze_shared_after_base)
        self._shared_frozen = False
        self._frozen_branch_count = 0
        self._frozen_projection_count = 0

    @property
    def num_branches(self) -> int:
        return len(self.branches)

    def _pooled(self, branch: nn.Module, shared: torch.Tensor) -> torch.Tensor:
        output = branch(shared)
        return torch.flatten(self.pool(output), 1)

    def forward_details(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        shared = self.shared(x)
        base_feature = self._pooled(self.branches[0], shared)
        fused = base_feature
        latest = base_feature
        for index in range(1, len(self.branches)):
            branch_feature = self._pooled(self.branches[index], shared)
            projected = self.projections[index - 1](branch_feature)
            gate = torch.sigmoid(self.residual_gates[index - 1])
            fused = fused + gate * projected
            latest = projected
        return fused, latest

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_details(x)[0]

    def expand(self) -> None:
        """Freeze previous representation and add one trainable layer4 branch."""

        source = copy.deepcopy(self.branches[-1])
        # The previous task may have ended in the stability phase, where the
        # latest branch was temporarily frozen. A newly copied branch must
        # always start trainable regardless of that transient phase state.
        source.requires_grad_(True)
        for branch in self.branches:
            branch.requires_grad_(False)
            branch.eval()
        for projection in self.projections:
            projection.requires_grad_(False)
            projection.eval()
        for gate in self.residual_gates:
            gate.requires_grad_(False)
        self._frozen_branch_count = len(self.branches)
        self._frozen_projection_count = len(self.projections)

        if self.freeze_shared_after_base:
            self.shared.requires_grad_(False)
            self.shared.eval()
            self._shared_frozen = True

        reference = next(source.parameters())
        projection = nn.Linear(
            self.feature_dim,
            self.feature_dim,
            bias=False,
            device=reference.device,
            dtype=reference.dtype,
        )
        nn.init.eye_(projection.weight)
        self.branches.append(source)
        self.projections.append(projection)
        self.residual_gates.append(nn.Parameter(reference.new_tensor(self.gate_init)))

    def ensure_num_branches(self, count: int) -> None:
        while self.num_branches < int(count):
            self.expand()

    def set_latest_branch_trainable(self, trainable: bool) -> None:
        self.branches[-1].requires_grad_(trainable)
        if len(self.branches) > 1:
            self.projections[-1].requires_grad_(trainable)
            self.residual_gates[-1].requires_grad_(trainable)
        if trainable:
            self.branches[-1].train(self.training)
            if len(self.branches) > 1:
                self.projections[-1].train(self.training)
        else:
            self.branches[-1].eval()
            if len(self.branches) > 1:
                self.projections[-1].eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if self._shared_frozen:
            self.shared.eval()
        for index in range(self._frozen_branch_count):
            self.branches[index].eval()
        for index in range(self._frozen_projection_count):
            self.projections[index].eval()
        return self


def build_selective_expandable_resnet18(
    pretrained: bool = True,
    weights: models.ResNet18_Weights | None = None,
    gate_init: float = -2.0,
    freeze_shared_after_base: bool = True,
) -> tuple[SelectiveExpandableResNet18, int]:
    backbone = SelectiveExpandableResNet18(
        pretrained=pretrained,
        weights=weights,
        gate_init=gate_init,
        freeze_shared_after_base=freeze_shared_after_base,
    )
    return backbone, backbone.feature_dim


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
