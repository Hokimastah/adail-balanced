from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset


@dataclass
class Exemplar:
    # A stored `image` tensor is replayed as is. Without one, `source_index`
    # points into the training dataset and the image is reloaded (and
    # re-augmented) on access. Both may be set: the index then only identifies
    # the sample for later memory updates.
    image: torch.Tensor | None
    label: int
    score: float = 0.0
    feature: torch.Tensor | None = None
    uncertainty: float = 0.0
    source_index: int | None = None


class MemoryDataset(Dataset):
    """A dataset view over stored exemplars."""

    def __init__(self, exemplars: list[Exemplar], source: Dataset | None = None) -> None:
        self.exemplars = exemplars
        self.source = source
        self.targets = [int(item.label) for item in exemplars]
        self.source_indices = [item.source_index for item in exemplars]

    def __len__(self) -> int:
        return len(self.exemplars)

    def __getitem__(self, index: int):
        item = self.exemplars[index]
        if item.image is not None:
            image = item.image
        else:
            if item.source_index is None or self.source is None:
                raise RuntimeError(
                    "Exemplar refers to the training dataset by index, but no source "
                    "dataset is attached to the memory (set ExemplarMemory.source_dataset)."
                )
            image = self.source[item.source_index][0]
        return image, torch.tensor(item.label, dtype=torch.long)


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
        # Dataset that index-based exemplars (source_index) are reloaded from.
        # It is not serialized; reattach it after loading a checkpoint.
        self.source_dataset: Dataset | None = None

    def __len__(self) -> int:
        return sum(len(items) for items in self.exemplars.values())

    def classes(self) -> list[int]:
        return sorted(self.exemplars.keys())

    def as_dataset(self, source: Dataset | None = None) -> MemoryDataset:
        """Exemplars as a dataset.

        Index-based exemplars are reloaded from ``source`` when given (for
        example the training images under the evaluation transform), otherwise
        from ``source_dataset``.
        """

        flat = [item for cls in self.classes() for item in self.exemplars[cls]]
        return MemoryDataset(flat, source=source if source is not None else self.source_dataset)

    def state_dict(self) -> dict[str, object]:
        # Serialize only tensors and primitive containers so checkpoints can be
        # loaded with torch.load(weights_only=True) on PyTorch >= 2.6.
        serialized = {
            int(class_id): [
                {
                    "image": item.image,
                    "label": int(item.label),
                    "score": float(item.score),
                    "feature": item.feature,
                    "uncertainty": float(item.uncertainty),
                    "source_index": item.source_index,
                }
                for item in items
            ]
            for class_id, items in self.exemplars.items()
        }
        return {
            "max_size": self.max_size,
            "quotas": self.quotas,
            "exemplars": serialized,
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        self.max_size = int(state["max_size"])
        self.quotas = {int(k): int(v) for k, v in dict(state["quotas"]).items()}
        loaded: dict[int, list[Exemplar]] = {}
        for key, values in dict(state["exemplars"]).items():
            items: list[Exemplar] = []
            for value in values:
                if isinstance(value, Exemplar):  # Backward compatibility in memory.
                    items.append(value)
                else:
                    payload = dict(value)
                    source_index = payload.get("source_index")
                    items.append(
                        Exemplar(
                            image=payload.get("image"),
                            label=int(payload["label"]),
                            score=float(payload.get("score", 0.0)),
                            feature=payload.get("feature"),
                            uncertainty=float(payload.get("uncertainty", 0.0)),
                            source_index=None if source_index is None else int(source_index),
                        )
                    )
            loaded[int(key)] = items
        self.exemplars = defaultdict(list, loaded)

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
        total: int | None = None,
    ) -> dict[int, int]:
        """Compute adaptive per-class quotas.

        Missing uncertainty/forgetting values default to zero. Missing counts default
        to one to avoid division by zero. ``total`` caps the number of exemplars to
        distribute below ``max_size`` (e.g. a budget that grows with seen classes).
        """

        total = self.max_size if total is None else max(0, min(int(total), self.max_size))
        if total == 0 or not class_ids:
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
        quotas_arr = np.floor(total * scores_arr / scores_arr.sum()).astype(int)

        # Distribute remainder to the largest fractional parts.
        remainder = total - int(quotas_arr.sum())
        fractions = (total * scores_arr / scores_arr.sum()) - quotas_arr
        order = np.argsort(-fractions)
        for idx in order[:remainder]:
            quotas_arr[idx] += 1
        return {int(c): int(q) for c, q in zip(class_ids, quotas_arr, strict=False)}

    def prune_to_quotas(self) -> None:
        for cls, quota in self.quotas.items():
            items = self.exemplars.get(cls, [])
            if len(items) > quota:
                self.exemplars[cls] = sorted(items, key=lambda x: x.score, reverse=True)[:quota]

    @staticmethod
    def _herding_indices(features: torch.Tensor, count: int) -> list[int]:
        """Greedy iCaRL-style approximation of the normalized class mean."""

        if count <= 0 or features.shape[0] == 0:
            return []
        features = F.normalize(features.float(), dim=1)
        class_mean = features.mean(dim=0)
        running = torch.zeros_like(class_mean)
        selected: list[int] = []
        for step in range(min(count, features.shape[0])):
            candidate_means = (running[None, :] + features) / float(step + 1)
            distances = (candidate_means - class_mean[None, :]).pow(2).sum(dim=1)
            if selected:
                distances[selected] = torch.inf
            index = int(distances.argmin().item())
            selected.append(index)
            running = running + features[index]
        return selected

    @staticmethod
    def _diverse_indices(
        features: torch.Tensor,
        count: int,
        selected: list[int],
    ) -> list[int]:
        """Greedy k-center completion for within-class visual diversity."""

        output: list[int] = []
        available = [index for index in range(features.shape[0]) if index not in selected]
        if count <= 0 or not available:
            return output
        normalized = F.normalize(features.float(), dim=1)
        while available and len(output) < count:
            anchors = selected + output
            if anchors:
                distances = torch.cdist(normalized[available], normalized[anchors]).amin(dim=1)
            else:
                centroid = normalized.mean(dim=0, keepdim=True)
                distances = torch.cdist(normalized[available], centroid).squeeze(1)
            chosen_position = int(distances.argmax().item())
            output.append(available.pop(chosen_position))
        return output

    def _select_candidates(
        self,
        candidates: list[Exemplar],
        quota: int,
        selection: str,
        representative_fraction: float,
        uncertainty_fraction: float,
    ) -> list[Exemplar]:
        if quota <= 0:
            return []
        if len(candidates) <= quota:
            return candidates
        if selection == "random":
            rng = np.random.default_rng(1993 + int(candidates[0].label))
            indexes = rng.choice(len(candidates), size=quota, replace=False).tolist()
            return [candidates[int(index)] for index in indexes]
        if selection == "uncertainty":
            return sorted(candidates, key=lambda item: item.uncertainty, reverse=True)[:quota]

        has_features = all(item.feature is not None for item in candidates)
        if not has_features:
            return sorted(candidates, key=lambda item: item.score, reverse=True)[:quota]
        features = torch.stack([item.feature for item in candidates if item.feature is not None])
        representative_count = (
            quota if selection == "herding" else int(round(quota * representative_fraction))
        )
        representative_count = min(quota, max(0, representative_count))
        selected = self._herding_indices(features, representative_count)

        if selection == "hybrid":
            uncertain_count = min(quota - len(selected), int(round(quota * uncertainty_fraction)))
            remaining = [index for index in range(len(candidates)) if index not in selected]
            uncertainty_order = sorted(
                remaining,
                key=lambda index: candidates[index].uncertainty,
                reverse=True,
            )
            selected.extend(uncertainty_order[:uncertain_count])
            selected.extend(self._diverse_indices(features, quota - len(selected), selected))
        return [candidates[index] for index in selected[:quota]]

    def update_from_tensors(
        self,
        images: torch.Tensor | None,
        labels: torch.Tensor,
        logits: torch.Tensor | None = None,
        features: torch.Tensor | None = None,
        class_counts: dict[int, int] | None = None,
        forgetting: dict[int, float] | None = None,
        *,
        replace: bool = False,
        selection: str = "hybrid",
        representative_fraction: float = 0.5,
        uncertainty_fraction: float = 0.3,
        source_indices: Sequence[int | None] | None = None,
        capacity: int | None = None,
        materialize_from: Dataset | None = None,
    ) -> None:
        """Add candidates and adaptively prune memory.

        ``capacity`` limits how many exemplars are kept this time (never more than
        ``max_size``); None uses the whole ``max_size``.

        Images are stored on CPU to avoid holding GPU memory.
        Candidate score combines uncertainty and closeness to class centroid when
        features are provided.

        When ``source_indices`` gives a training-dataset index for a candidate,
        only that index is kept and the image is reloaded from
        ``source_dataset`` on replay; ``images`` may then be None. With
        ``materialize_from`` (a deterministic view of the same dataset), the
        selected exemplars instead get a stored tensor loaded from it, so they
        replay a fixed image while only the selected ones are ever held in RAM.
        """

        limit = self.max_size if capacity is None else max(0, min(int(capacity), self.max_size))
        if limit == 0:
            self.quotas = {}
            self.exemplars = defaultdict(list)
            return
        if source_indices is not None and len(source_indices) != len(labels):
            raise ValueError("source_indices must have one entry per label")
        labels_cpu = labels.detach().cpu().to(torch.long)
        incoming_classes = {int(x) for x in labels_cpu.tolist()}
        class_ids = sorted(incoming_classes if replace else set(self.classes()) | incoming_classes)

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
            total=limit,
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

        candidates_by_class: dict[int, list[Exemplar]] = defaultdict(list)
        if not replace:
            for class_id, items in self.exemplars.items():
                candidates_by_class[int(class_id)].extend(items)

        features_cpu = features.detach().cpu() if features is not None else None
        for i, label in enumerate(labels_cpu.tolist()):
            u = float(sample_uncertainty[i]) if sample_uncertainty is not None else 0.0
            score = u + float(closeness[i])
            feature = features_cpu[i].clone() if features_cpu is not None else None
            source_index = source_indices[i] if source_indices is not None else None
            if source_index is not None:
                image = None
            elif images is not None:
                image = images[i].detach().cpu()
            else:
                raise ValueError("each candidate needs either an image or a source index")
            candidates_by_class[int(label)].append(
                Exemplar(
                    image,
                    int(label),
                    score,
                    feature=feature,
                    uncertainty=u,
                    source_index=None if source_index is None else int(source_index),
                )
            )

        # If one class has fewer candidates than its adaptive quota, reuse the
        # remaining global capacity for other classes instead of silently
        # under-filling memory. The deterministic round-robin keeps the result
        # bounded and reproducible.
        effective_quotas = {
            class_id: min(self.quotas.get(class_id, 0), len(candidates_by_class[class_id]))
            for class_id in class_ids
        }
        fillable = min(limit, sum(len(candidates_by_class[c]) for c in class_ids))
        remainder = fillable - sum(effective_quotas.values())
        priority = sorted(
            class_ids,
            key=lambda class_id: (
                self.quotas.get(class_id, 0),
                len(candidates_by_class[class_id]),
                -class_id,
            ),
            reverse=True,
        )
        while remainder > 0:
            allocated = False
            for class_id in priority:
                if effective_quotas[class_id] < len(candidates_by_class[class_id]):
                    effective_quotas[class_id] += 1
                    remainder -= 1
                    allocated = True
                    if remainder == 0:
                        break
            if not allocated:
                break
        self.quotas = effective_quotas

        selected: dict[int, list[Exemplar]] = {}
        for class_id in class_ids:
            selected[class_id] = self._select_candidates(
                candidates_by_class.get(class_id, []),
                self.quotas.get(class_id, 0),
                selection,
                representative_fraction,
                uncertainty_fraction,
            )
        if materialize_from is not None:
            for items in selected.values():
                for item in items:
                    if item.image is None and item.source_index is not None:
                        item.image = materialize_from[item.source_index][0].detach().cpu()
        self.exemplars = defaultdict(list, selected)
