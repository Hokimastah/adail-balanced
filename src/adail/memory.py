from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset


@dataclass
class Exemplar:
    image: torch.Tensor
    label: int
    score: float = 0.0


class MemoryDataset(Dataset):
    """A dataset view over stored exemplars."""

    def __init__(self, exemplars: list[Exemplar]) -> None:
        self.exemplars = exemplars

    def __len__(self) -> int:
        return len(self.exemplars)

    def __getitem__(self, index: int):
        item = self.exemplars[index]
        return item.image, torch.tensor(item.label, dtype=torch.long)


class ExemplarMemory:
    """Adaptive exemplar memory for class-incremental learning.

    The memory quota of each class is updated from three signals:
    uncertainty, forgetting rate, and class rarity.
    """

    def __init__(
        self,
        max_size: int,
        alpha_uncertainty: float = 1.0,
        beta_forgetting: float = 1.0,
        gamma_rarity: float = 1.0,
    ) -> None:
        if max_size < 0:
            raise ValueError("max_size must be non-negative")
        self.max_size = max_size
        self.alpha_uncertainty = alpha_uncertainty
        self.beta_forgetting = beta_forgetting
        self.gamma_rarity = gamma_rarity
        self.exemplars: dict[int, list[Exemplar]] = defaultdict(list)
        self.quotas: dict[int, int] = {}

    def __len__(self) -> int:
        return sum(len(items) for items in self.exemplars.values())

    def classes(self) -> list[int]:
        return sorted(self.exemplars.keys())

    def as_dataset(self) -> MemoryDataset:
        flat = [item for cls in self.classes() for item in self.exemplars[cls]]
        return MemoryDataset(flat)

    def state_dict(self) -> dict[str, object]:
        return {
            "max_size": self.max_size,
            "quotas": self.quotas,
            "exemplars": dict(self.exemplars),
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        self.max_size = int(state["max_size"])
        self.quotas = {int(k): int(v) for k, v in dict(state["quotas"]).items()}
        self.exemplars = defaultdict(list, {int(k): v for k, v in dict(state["exemplars"]).items()})

    @staticmethod
    def uncertainty_from_logits(logits: torch.Tensor) -> torch.Tensor:
        """Entropy-based uncertainty normalized to approximately [0, 1]."""

        probs = F.softmax(logits, dim=1)
        entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=1)
        max_entropy = float(np.log(max(2, logits.shape[1])))
        return entropy / max_entropy

    def compute_quotas(
        self,
        class_ids: list[int],
        uncertainty: dict[int, float] | None = None,
        forgetting: dict[int, float] | None = None,
        class_counts: dict[int, int] | None = None,
    ) -> dict[int, int]:
        """Compute adaptive per-class quotas.

        Missing uncertainty/forgetting values default to zero. Missing counts default
        to one to avoid division by zero.
        """

        if self.max_size == 0 or not class_ids:
            return {int(c): 0 for c in class_ids}

        uncertainty = uncertainty or {}
        forgetting = forgetting or {}
        class_counts = class_counts or {}
        counts = np.array([max(1, class_counts.get(int(c), 1)) for c in class_ids], dtype=float)
        rarity_values = counts.max() / counts
        rarity_values = rarity_values / max(rarity_values.max(), 1e-12)

        scores = []
        for i, c in enumerate(class_ids):
            c = int(c)
            score = (
                self.alpha_uncertainty * max(0.0, float(uncertainty.get(c, 0.0)))
                + self.beta_forgetting * max(0.0, float(forgetting.get(c, 0.0)))
                + self.gamma_rarity * float(rarity_values[i])
            )
            scores.append(max(score, 1e-6))
        scores_arr = np.array(scores, dtype=float)
        quotas_arr = np.floor(self.max_size * scores_arr / scores_arr.sum()).astype(int)

        # Distribute remainder to the largest fractional parts.
        remainder = self.max_size - int(quotas_arr.sum())
        fractions = (self.max_size * scores_arr / scores_arr.sum()) - quotas_arr
        order = np.argsort(-fractions)
        for idx in order[:remainder]:
            quotas_arr[idx] += 1
        return {int(c): int(q) for c, q in zip(class_ids, quotas_arr, strict=False)}

    def prune_to_quotas(self) -> None:
        for cls, quota in self.quotas.items():
            items = self.exemplars.get(cls, [])
            if len(items) > quota:
                self.exemplars[cls] = sorted(items, key=lambda x: x.score, reverse=True)[:quota]

    def update_from_tensors(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        logits: torch.Tensor | None = None,
        features: torch.Tensor | None = None,
        class_counts: dict[int, int] | None = None,
        forgetting: dict[int, float] | None = None,
    ) -> None:
        """Add candidates and adaptively prune memory.

        Images are stored on CPU to avoid holding GPU memory.
        Candidate score combines uncertainty and closeness to class centroid when
        features are provided.
        """

        if self.max_size == 0:
            return
        labels_cpu = labels.detach().cpu()
        class_ids = sorted(set(self.classes()) | {int(x) for x in labels_cpu.tolist()})

        uncertainty_values: dict[int, float] = {}
        sample_uncertainty: torch.Tensor | None = None
        if logits is not None:
            sample_uncertainty = self.uncertainty_from_logits(logits.detach()).cpu()
            for cls in class_ids:
                mask = labels_cpu == cls
                if mask.any():
                    uncertainty_values[cls] = float(sample_uncertainty[mask].mean())

        self.quotas = self.compute_quotas(
            class_ids=class_ids,
            uncertainty=uncertainty_values,
            forgetting=forgetting,
            class_counts=class_counts,
        )

        # Feature centroid closeness provides a simple herding-like signal.
        closeness = torch.zeros(len(labels_cpu), dtype=torch.float32)
        if features is not None:
            f_cpu = features.detach().cpu()
            for cls in class_ids:
                mask = labels_cpu == cls
                if mask.any():
                    centroid = f_cpu[mask].mean(dim=0, keepdim=True)
                    distance = torch.cdist(f_cpu[mask], centroid).flatten()
                    closeness[mask] = 1.0 / (1.0 + distance)

        for i, label in enumerate(labels_cpu.tolist()):
            u = float(sample_uncertainty[i]) if sample_uncertainty is not None else 0.0
            score = u + float(closeness[i])
            self.exemplars[int(label)].append(Exemplar(images[i].detach().cpu(), int(label), score))

        self.prune_to_quotas()
