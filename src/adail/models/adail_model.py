from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import torch
from torch import nn
from torch.nn import functional as F

from adail.modules import DriftAligner, PrototypeBank

from .backbone import build_resnet18_backbone, build_selective_expandable_resnet18

if TYPE_CHECKING:
    from adail.config import ADAILConfig


class CosineClassifier(nn.Module):
    """Normalized classifier with an optional learned logit scale."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        scale: float = 10.0,
        learnable_scale: bool = True,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_normal_(self.weight, nonlinearity="linear")
        scale_tensor = torch.tensor(float(scale))
        if learnable_scale:
            self.logit_scale = nn.Parameter(scale_tensor)
        else:
            self.register_buffer("logit_scale", scale_tensor)

    def cosine(self, features: torch.Tensor) -> torch.Tensor:
        """Cosine similarity to every class weight, before the learned scale."""

        features = F.normalize(features, dim=1)
        weights = F.normalize(self.weight, dim=1)
        return F.linear(features, weights)

    def scale_logits(self, cosine: torch.Tensor) -> torch.Tensor:
        return self.logit_scale.clamp(1e-3, 100.0) * cosine

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.scale_logits(self.cosine(features))

    @torch.no_grad()
    def imprint(self, class_ids: list[int], vectors: torch.Tensor) -> None:
        if not class_ids:
            return
        class_index = torch.as_tensor(class_ids, dtype=torch.long, device=self.weight.device)
        self.weight.index_copy_(
            0,
            class_index,
            F.normalize(vectors.to(self.weight.device), dim=1),
        )

    def expand(self, new_num_classes: int) -> None:
        if new_num_classes <= self.out_features:
            return
        self.resize(new_num_classes)

    def resize(self, num_classes: int) -> None:
        """Grow (new rows randomly initialized) or shrink to `num_classes` rows."""

        num_classes = int(num_classes)
        if num_classes == self.out_features:
            return
        old_weight = self.weight
        new_weight = old_weight.new_empty(num_classes, self.in_features)
        nn.init.kaiming_normal_(new_weight, nonlinearity="linear")
        kept = min(num_classes, self.out_features)
        with torch.no_grad():
            new_weight[:kept].copy_(old_weight[:kept])
        self.weight = nn.Parameter(new_weight)
        self.out_features = num_classes


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
        classifier_type: str = "cosine",
        cosine_scale: float = 10.0,
        learnable_cosine_scale: bool = True,
        dynamic_expansion: bool = True,
        dynamic_gate_init: float = -2.0,
        freeze_shared_after_base: bool = True,
    ) -> None:
        super().__init__()
        if backbone is None:
            if dynamic_expansion:
                backbone, inferred_dim = build_selective_expandable_resnet18(
                    pretrained=pretrained_backbone,
                    gate_init=dynamic_gate_init,
                    freeze_shared_after_base=freeze_shared_after_base,
                )
            else:
                backbone, inferred_dim = build_resnet18_backbone(pretrained=pretrained_backbone)
            feature_dim = inferred_dim if feature_dim is None else feature_dim
        if feature_dim is None:
            raise ValueError("feature_dim must be provided when using a custom backbone")

        self.backbone = backbone
        self.feature_dim = feature_dim
        self.classifier_type = classifier_type
        if classifier_type == "cosine":
            self.classifier: nn.Module = CosineClassifier(
                feature_dim,
                num_classes,
                scale=cosine_scale,
                learnable_scale=learnable_cosine_scale,
            )
        elif classifier_type == "linear":
            self.classifier = nn.Linear(feature_dim, num_classes)
        else:
            raise ValueError("classifier_type must be 'cosine' or 'linear'")
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

    @classmethod
    def from_config(
        cls,
        config: ADAILConfig,
        *,
        backbone: nn.Module | None = None,
        pretrained_backbone: bool = True,
    ) -> ADAILNet:
        """Construct a model with every architecture option from a config."""

        return cls(
            num_classes=config.num_classes,
            backbone=backbone,
            feature_dim=config.feature_dim if backbone is not None else None,
            pretrained_backbone=pretrained_backbone,
            prototype_momentum=config.prototype_momentum,
            prototype_margin=config.prototype_margin,
            classifier_type=config.classifier_type,
            cosine_scale=config.cosine_scale,
            learnable_cosine_scale=config.learnable_cosine_scale,
            dynamic_expansion=config.dynamic_expansion,
            dynamic_gate_init=config.dynamic_gate_init,
            freeze_shared_after_base=config.freeze_shared_after_base,
        )

    @property
    def old_model(self) -> ADAILNet | None:
        return getattr(self, "_old_model", None)

    @property
    def num_classes(self) -> int:
        return self.classifier.out_features

    @property
    def num_branches(self) -> int:
        return int(getattr(self.backbone, "num_branches", 1))

    def forward_details(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if hasattr(self.backbone, "forward_details"):
            features, latest_features = self.backbone.forward_details(x)
        else:
            features = self.backbone(x)
            latest_features = features
        if features.ndim > 2:
            features = torch.flatten(features, 1)
        if latest_features.ndim > 2:
            latest_features = torch.flatten(latest_features, 1)
        logits, cosine_logits = self._classify(features)
        return {
            "logits": logits,
            # Unscaled cosine similarities for a cosine classifier (equal to
            # "logits" for a linear one); margins are measured in this space.
            "cosine_logits": cosine_logits,
            "features": features,
            "aux_logits": self.classifier(latest_features),
            "latest_features": latest_features,
        }

    def _classify(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(self.classifier, CosineClassifier):
            cosine = self.classifier.cosine(features)
            return self.classifier.scale_logits(cosine), cosine
        logits = self.classifier(features)
        return logits, logits

    def forward(self, x: torch.Tensor, return_features: bool = False):
        outputs = self.forward_details(x)
        logits, features = outputs["logits"], outputs["features"]
        if return_features:
            return logits, features
        return logits

    @torch.no_grad()
    def freeze_old_model(self) -> None:
        """Snapshot the current model as frozen teacher for the next CIL task."""

        # Discard a teacher from t-2 before copying the t-1 student. The teacher
        # is intentionally not a registered submodule, but deepcopy would still
        # recursively duplicate it if the reference were left in place.
        object.__setattr__(self, "_old_model", None)
        old = copy.deepcopy(self)
        object.__setattr__(old, "_old_model", None)
        old.eval()
        for param in old.parameters():
            param.requires_grad = False
        object.__setattr__(self, "_old_model", old)

    def clear_old_model(self) -> None:
        """Remove the frozen teacher reference."""

        object.__setattr__(self, "_old_model", None)

    def expand_representation(self) -> bool:
        """Add a high-level feature branch when the backbone supports it."""

        if not hasattr(self.backbone, "expand"):
            return False
        self.backbone.expand()
        return True

    def ensure_num_branches(self, count: int) -> None:
        if hasattr(self.backbone, "ensure_num_branches"):
            self.backbone.ensure_num_branches(count)

    def configure_phase(self, phase: str) -> None:
        """Toggle the newest representation for new/old alternating phases."""

        if phase not in {"new", "old"}:
            raise ValueError("phase must be 'new' or 'old'")
        if hasattr(self.backbone, "set_latest_branch_trainable"):
            self.backbone.set_latest_branch_trainable(phase == "new")

    @property
    def has_auxiliary_branch(self) -> bool:
        """Whether a distinct newest representation can support auxiliary CE."""

        return self.num_branches > 1 and hasattr(self.backbone, "forward_details")

    def mask_new_class_gradients(self, new_classes: list[int]) -> None:
        """Zero the gradient of new-class classifier rows.

        Zeroing the gradient alone does not freeze the rows under SGD with
        momentum or weight decay; the trainer restores them around the
        optimizer step instead (see ``ADAILTrainer._step_with_frozen_rows``).
        """

        weight = getattr(self.classifier, "weight", None)
        if weight is not None and weight.grad is not None and new_classes:
            weight.grad[new_classes] = 0
        bias = getattr(self.classifier, "bias", None)
        if bias is not None and bias.grad is not None and new_classes:
            bias.grad[new_classes] = 0

    @torch.no_grad()
    def imprint_classes(
        self,
        class_ids: list[int],
        features: torch.Tensor,
        labels: torch.Tensor,
    ) -> None:
        """Initialize class weights and prototypes from feature centroids."""

        vectors = []
        valid_ids = []
        for class_id in class_ids:
            mask = labels == int(class_id)
            if bool(mask.any()):
                valid_ids.append(int(class_id))
                vectors.append(features[mask].mean(dim=0))
        if not vectors:
            return
        stacked = torch.stack(vectors).to(next(self.parameters()).device)
        label_tensor = torch.as_tensor(valid_ids, device=stacked.device, dtype=torch.long)
        self.prototypes.update(stacked, label_tensor)
        if isinstance(self.classifier, CosineClassifier):
            self.classifier.imprint(valid_ids, stacked)
        else:
            self.classifier.weight.index_copy_(0, label_tensor, F.normalize(stacked, dim=1))
            if self.classifier.bias is not None:
                self.classifier.bias.index_fill_(0, label_tensor, 0.0)

    def expand_classes(self, new_num_classes: int) -> None:
        """Expand classifier/prototypes while preserving old weights."""

        if new_num_classes <= self.num_classes:
            return
        self.resize_classes(new_num_classes)

    def resize_classes(self, num_classes: int) -> None:
        """Grow or shrink the classifier and prototype bank.

        Rows of classes that exist before and after are kept; new rows are
        freshly initialized and new prototypes start uninitialized.
        """

        num_classes = int(num_classes)
        if num_classes < 1:
            raise ValueError("num_classes must be positive")
        if num_classes == self.num_classes:
            return
        if isinstance(self.classifier, CosineClassifier):
            self.classifier.resize(num_classes)
        else:
            old_classifier = self.classifier
            new_classifier = nn.Linear(
                self.feature_dim, num_classes, bias=old_classifier.bias is not None
            ).to(old_classifier.weight.device)
            kept = min(num_classes, old_classifier.out_features)
            with torch.no_grad():
                new_classifier.weight[:kept].copy_(old_classifier.weight[:kept])
                if old_classifier.bias is not None:
                    new_classifier.bias[:kept].copy_(old_classifier.bias[:kept])
            self.classifier = new_classifier
        self.prototypes.resize(num_classes)
