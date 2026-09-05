from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class OptimConfig:
    """Optimizer defaults following the proposal protocol: SGD + momentum."""

    lr: float = 0.01
    momentum: float = 0.9
    weight_decay: float = 5e-4
    nesterov: bool = False


@dataclass
class DataConfig:
    """Dataset and class-incremental scenario settings."""

    dataset_name: str = "generic"
    root: str | None = None
    scenario: str = "3-2"
    image_size: int = 224
    batch_size: int = 32
    num_workers: int = 4
    class_order: list[int] | None = None


@dataclass
class ADAILConfig:
    """Configuration for the enhanced ADAIL multi-loss objective.

    The defaults keep ADAIL's drift/prototype/ordinal components and enable the
    most useful ideas validated by LO2LN: cosine classification, imprinted class
    weights, class-balanced focal loss, logit distillation, margin ranking,
    balanced replay, alternating stability/plasticity phases, and selective
    high-level representation expansion.
    """

    num_classes: int
    feature_dim: int = 512
    memory_size: int = 2000
    prototype_momentum: float = 0.9
    prototype_margin: float = 1.0
    lambda_drift: float = 1.0
    lambda_proto: float = 0.2
    lambda_sep: float = 0.1
    lambda_ordinal: float = 0.1
    use_ordinal: bool = False
    mask_unseen_classes: bool = True

    # Cosine classifier and prototype-based initialization.
    classifier_type: str = "cosine"
    cosine_scale: float = 10.0
    learnable_cosine_scale: bool = True
    imprint_new_classes: bool = True

    # Class-balanced focal classification.
    use_class_balanced_focal: bool = True
    focal_beta: float = 0.99
    focal_gamma: float = 1.0
    old_focal_beta: float = 0.999
    old_focal_gamma: float = 0.5

    # LO2LN-inspired auxiliary, distillation, and margin objectives.
    lambda_aux: float = 0.5
    lambda_distill: float = 1.0
    distill_temperature: float = 2.0
    distill_memory_only: bool = True
    distill_confidence_threshold: float = 0.0
    lambda_ranking: float = 0.5
    ranking_margin: float = 0.2
    ranking_topk: int = 1
    ordinal_aware_margin: bool = True

    # Alternating plasticity (new) and stability (old) optimization.
    alternating_objectives: bool = True
    old_phase_frequency: int = 2
    old_phase_distill_multiplier: float = 1.5
    old_phase_ranking_multiplier: float = 1.5

    # Balanced current-data/replay batches.
    balanced_replay: bool = True
    replay_memory_fraction: float = 0.5
    sampler_seed: int = 1993

    # Selective expansion duplicates only the high-level ResNet block. Outputs
    # are fused back to feature_dim, so prototypes keep a fixed dimensionality.
    dynamic_expansion: bool = True
    dynamic_gate_init: float = -2.0
    freeze_shared_after_base: bool = True

    # Adaptive quota + representative/uncertain/diverse exemplar selection.
    memory_alpha_uncertainty: float = 1.0
    memory_beta_forgetting: float = 1.0
    memory_gamma_rarity: float = 1.0
    memory_selection: str = "hybrid"
    memory_representative_fraction: float = 0.5
    memory_uncertainty_fraction: float = 0.3
    optimizer: OptimConfig = field(default_factory=OptimConfig)
    data: DataConfig = field(default_factory=DataConfig)

    def __post_init__(self) -> None:
        if self.classifier_type not in {"cosine", "linear"}:
            raise ValueError("classifier_type must be 'cosine' or 'linear'")
        if not 0.0 <= self.replay_memory_fraction < 1.0:
            raise ValueError("replay_memory_fraction must be in [0, 1)")
        if self.old_phase_frequency < 1:
            raise ValueError("old_phase_frequency must be positive")
        if self.ranking_topk < 1:
            raise ValueError("ranking_topk must be positive")
        if self.memory_selection not in {"hybrid", "herding", "uncertainty", "random"}:
            raise ValueError("memory_selection must be hybrid, herding, uncertainty, or random")
        fractions = self.memory_representative_fraction + self.memory_uncertainty_fraction
        if not 0.0 <= self.memory_representative_fraction <= 1.0:
            raise ValueError("memory_representative_fraction must be in [0, 1]")
        if not 0.0 <= self.memory_uncertainty_fraction <= 1.0:
            raise ValueError("memory_uncertainty_fraction must be in [0, 1]")
        if fractions > 1.0:
            raise ValueError("representative + uncertainty fractions cannot exceed 1")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ADAILConfig:
        payload = dict(data)
        optim = payload.get("optimizer", {})
        if isinstance(optim, dict):
            payload["optimizer"] = OptimConfig(**optim)
        data_cfg = payload.get("data", {})
        if isinstance(data_cfg, dict):
            payload["data"] = DataConfig(**data_cfg)
        return cls(**payload)

    @classmethod
    def from_yaml(cls, path: str | Path) -> ADAILConfig:
        with Path(path).open("r", encoding="utf-8") as f:
            return cls.from_dict(yaml.safe_load(f))

    def to_yaml(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)
