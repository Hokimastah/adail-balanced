from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterator, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler


class ReplayDataset(Dataset):
    """Current task data plus exemplar memory with explicit source flags."""

    def __init__(self, current: Dataset, memory: Dataset | None = None) -> None:
        self.current = current
        self.memory = memory
        self.current_size = len(current)
        self.memory_size = len(memory) if memory is not None else 0
        self.current_targets = [self._target(current, i) for i in range(self.current_size)]
        self.memory_targets = (
            [self._target(memory, i) for i in range(self.memory_size)] if memory is not None else []
        )
        self.targets = self.current_targets + self.memory_targets

    @staticmethod
    def _target(dataset: Dataset, index: int) -> int:
        if hasattr(dataset, "targets"):
            target = dataset.targets[index]
        else:
            target = dataset[index][1]
        return int(torch.as_tensor(target).reshape(-1)[0].item())

    def __len__(self) -> int:
        return self.current_size + self.memory_size

    def __getitem__(self, index: int):
        is_memory = index >= self.current_size
        if is_memory:
            if self.memory is None:
                raise IndexError(index)
            image, label = self.memory[index - self.current_size]
        else:
            image, label = self.current[index]
        label_scalar = int(torch.as_tensor(label).reshape(-1)[0].item())
        return image, torch.tensor(label_scalar, dtype=torch.long), torch.tensor(is_memory)


class BalancedReplayBatchSampler(Sampler[list[int]]):
    """Build class-balanced batches with a fixed current/memory proportion.

    Sampling with replacement is deliberate: scarce old exemplars are revisited
    at the same rate as abundant incoming samples without pretending that the
    full historical dataset remains available.
    """

    def __init__(
        self,
        current_labels: Sequence[int],
        memory_labels: Sequence[int],
        batch_size: int,
        memory_fraction: float = 0.5,
        seed: int = 1993,
        epoch_length: str = "current",
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if not 0.0 <= memory_fraction < 1.0:
            raise ValueError("memory_fraction must be in [0, 1)")
        if epoch_length not in {"current", "pool"}:
            raise ValueError("epoch_length must be 'current' or 'pool'")
        self.current_labels = [int(x) for x in current_labels]
        self.memory_labels = [int(x) for x in memory_labels]
        self.batch_size = int(batch_size)
        self.memory_fraction = float(memory_fraction)
        self.seed = int(seed)
        self.epoch_length = epoch_length
        self.epoch = 0

        requested_memory = int(round(self.batch_size * self.memory_fraction))
        self.memory_per_batch = min(self.batch_size - 1, requested_memory)
        if not self.memory_labels:
            self.memory_per_batch = 0
        self.current_per_batch = self.batch_size - self.memory_per_batch
        self._current_pools = self._make_pools(self.current_labels, offset=0)
        self._memory_pools = self._make_pools(
            self.memory_labels,
            offset=len(self.current_labels),
        )

    @staticmethod
    def _make_pools(labels: Sequence[int], offset: int) -> dict[int, np.ndarray]:
        pools: dict[int, list[int]] = defaultdict(list)
        for index, label in enumerate(labels):
            pools[int(label)].append(index + offset)
        return {key: np.asarray(value, dtype=np.int64) for key, value in pools.items()}

    @staticmethod
    def _balanced_draw(
        pools: dict[int, np.ndarray],
        count: int,
        rng: np.random.Generator,
    ) -> list[int]:
        if count <= 0 or not pools:
            return []
        classes = np.asarray(sorted(pools), dtype=np.int64)
        chosen_classes: list[int] = []
        while len(chosen_classes) < count:
            chosen_classes.extend(rng.permutation(classes).tolist())
        return [int(rng.choice(pools[int(class_id)])) for class_id in chosen_classes[:count]]

    def __len__(self) -> int:
        if self.epoch_length == "pool":
            # One pass over everything available for this task. Unlike the
            # legacy length, this does not grow when the current share of the
            # batch shrinks.
            total = len(self.current_labels) + len(self.memory_labels)
            return max(1, math.ceil(total / self.batch_size))
        return max(1, math.ceil(len(self.current_labels) / max(1, self.current_per_batch)))

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        for _ in range(len(self)):
            batch = self._balanced_draw(self._current_pools, self.current_per_batch, rng)
            batch.extend(self._balanced_draw(self._memory_pools, self.memory_per_batch, rng))
            rng.shuffle(batch)
            yield batch
