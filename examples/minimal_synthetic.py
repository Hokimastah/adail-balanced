"""Minimal ADAIL example with synthetic tensors.

Run:
    python examples/minimal_synthetic.py
"""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import TensorDataset

from adail import ADAILConfig, ADAILNet, ADAILTrainer
from adail.data import ClassIncrementalScenario


class TinyBackbone(nn.Module):
    def __init__(self, feature_dim: int = 16):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(3 * 16 * 16, feature_dim), nn.ReLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def make_dataset(num_classes: int = 5, n_per_class: int = 20):
    xs, ys = [], []
    for c in range(num_classes):
        xs.append(torch.randn(n_per_class, 3, 16, 16) + c * 0.2)
        ys.append(torch.full((n_per_class,), c, dtype=torch.long))
    return TensorDataset(torch.cat(xs), torch.cat(ys))


if __name__ == "__main__":
    num_classes = 5
    cfg = ADAILConfig(num_classes=num_classes, feature_dim=16, memory_size=50, use_ordinal=True)
    model = ADAILNet(
        num_classes=num_classes,
        backbone=TinyBackbone(feature_dim=16),
        feature_dim=16,
        pretrained_backbone=False,
    )
    trainer = ADAILTrainer(model, cfg)
    scenario = ClassIncrementalScenario.from_string(num_classes, "2-1")
    dataset = make_dataset(num_classes)
    history = trainer.fit_scenario(dataset, dataset, scenario, epochs_per_task=1, batch_size=16)
    print(history)
