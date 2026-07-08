from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder

from .presets import get_preset
from .transforms import build_eval_transform, build_train_transform


class ImageFolderLoaders(NamedTuple):
    train: object
    val: object | None
    test: object


@dataclass(frozen=True)
class ClassIncrementalScenario:
    """Class splits for class-incremental training."""

    tasks: list[list[int]]

    @classmethod
    def from_num_classes(
        cls,
        num_classes: int,
        base_classes: int,
        increment: int,
        class_order: list[int] | None = None,
    ) -> "ClassIncrementalScenario":
        if base_classes <= 0 or increment <= 0:
            raise ValueError("base_classes and increment must be positive")
        order = class_order or list(range(num_classes))
        if sorted(order) != list(range(num_classes)):
            raise ValueError("class_order must be a permutation of 0..num_classes-1")
        tasks = [order[:base_classes]]
        start = base_classes
        while start < num_classes:
            tasks.append(order[start : start + increment])
            start += increment
        return cls(tasks)

    @classmethod
    def from_string(
        cls,
        num_classes: int,
        spec: str,
        class_order: list[int] | None = None,
    ) -> "ClassIncrementalScenario":
        """Parse scenario strings such as '4-1', '4-2', '3-1', or '3-2'."""

        try:
            base, inc = [int(x) for x in spec.split("-", maxsplit=1)]
        except Exception as exc:  # noqa: BLE001
            raise ValueError("scenario must look like '4-1' or '3-2'") from exc
        return cls.from_num_classes(num_classes, base, inc, class_order)

    @classmethod
    def from_preset(
        cls,
        dataset_name: str,
        scenario: str | None = None,
        class_order: list[int] | None = None,
    ) -> "ClassIncrementalScenario":
        preset = get_preset(dataset_name)
        spec = scenario or preset.default_scenarios[0]
        return cls.from_string(preset.num_classes, spec, class_order=class_order)

    def seen_classes_until(self, task_index: int) -> list[int]:
        seen: list[int] = []
        for task in self.tasks[: task_index + 1]:
            seen.extend(task)
        return seen


class DatasetSubsetByClass(Dataset):
    """Dataset wrapper that only exposes selected target classes."""

    def __init__(self, dataset: Dataset, classes: list[int]) -> None:
        self.dataset = dataset
        self.classes = set(int(c) for c in classes)
        self.indices = [i for i in range(len(dataset)) if self._target_at(i) in self.classes]
        self.targets = [self._target_at(i) for i in self.indices]

    def _target_at(self, idx: int) -> int:
        if hasattr(self.dataset, "targets"):
            return int(self.dataset.targets[idx])
        item = self.dataset[idx]
        return int(item[1])

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        return self.dataset[self.indices[index]]


def build_imagefolder_loaders(
    root: str | Path,
    image_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 4,
):
    """Build train/val/test ImageFolder datasets and DataLoaders.

    Expected structure:
    root/train/<class_name>/*.png
    root/test/<class_name>/*.png
    root/val/<class_name>/*.png  # optional
    """

    from torch.utils.data import DataLoader

    root = Path(root)
    train_dataset = ImageFolder(root / "train", transform=build_train_transform(image_size))
    test_dataset = ImageFolder(root / "test", transform=build_eval_transform(image_size))
    val_dataset = None
    if (root / "val").exists():
        val_dataset = ImageFolder(root / "val", transform=build_eval_transform(image_size))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return ImageFolderLoaders(train=train_loader, val=val_loader, test=test_loader)
