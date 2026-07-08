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
    """Configuration for ADAIL training, memory, and multi-loss objective."""

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
    memory_alpha_uncertainty: float = 1.0
    memory_beta_forgetting: float = 1.0
    memory_gamma_rarity: float = 1.0
    optimizer: OptimConfig = field(default_factory=OptimConfig)
    data: DataConfig = field(default_factory=DataConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ADAILConfig":
        payload = dict(data)
        optim = payload.get("optimizer", {})
        if isinstance(optim, dict):
            payload["optimizer"] = OptimConfig(**optim)
        data_cfg = payload.get("data", {})
        if isinstance(data_cfg, dict):
            payload["data"] = DataConfig(**data_cfg)
        return cls(**payload)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ADAILConfig":
        with Path(path).open("r", encoding="utf-8") as f:
            return cls.from_dict(yaml.safe_load(f))

    def to_yaml(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)
