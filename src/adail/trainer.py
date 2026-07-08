from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from tqdm.auto import tqdm

from .config import ADAILConfig
from .data import ClassIncrementalScenario, DatasetSubsetByClass
from .memory import ExemplarMemory
from .metrics import (
    accuracy,
    average_accuracy,
    average_forgetting,
    mean_absolute_error,
    per_class_accuracy,
    quadratic_weighted_kappa,
)
from .models import ADAILNet
from .modules import drift_alignment_loss, ordinal_constraint_loss


@dataclass
class TaskResult:
    task_index: int
    new_classes: list[int]
    seen_classes: list[int]
    metrics: dict[str, float] = field(default_factory=dict)
    per_class_accuracy: dict[int, float] = field(default_factory=dict)
    memory_quotas: dict[int, int] = field(default_factory=dict)


class ADAILTrainer:
    """Class-incremental training loop matching the ADAIL proposal.

    Important protocol detail: after task 1, the trainer only uses the current
    task data plus the exemplar memory. It does not revisit the full historical
    training dataset, preserving the CIL setting described in the proposal.
    """

    def __init__(
        self,
        model: ADAILNet,
        config: ADAILConfig,
        device: str | torch.device = "cpu",
    ) -> None:
        self.model = model.to(device)
        self.config = config
        self.device = torch.device(device)
        self.memory = ExemplarMemory(
            max_size=config.memory_size,
            alpha_uncertainty=config.memory_alpha_uncertainty,
            beta_forgetting=config.memory_beta_forgetting,
            gamma_rarity=config.memory_gamma_rarity,
        )
        self.best_class_scores: dict[int, float] = {}
        self.current_class_scores: dict[int, float] = {}
        self.class_forgetting: dict[int, float] = {}
        self.observed_class_counts: dict[int, int] = {}
        self.current_seen_classes: list[int] = list(range(config.num_classes))

    def build_optimizer(self) -> torch.optim.Optimizer:
        opt = self.config.optimizer
        return torch.optim.SGD(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=opt.lr,
            momentum=opt.momentum,
            weight_decay=opt.weight_decay,
            nesterov=opt.nesterov,
        )

    @staticmethod
    def _target_at(dataset: Dataset, idx: int) -> int:
        if hasattr(dataset, "targets"):
            return int(dataset.targets[idx])
        item = dataset[idx]
        return int(item[1])

    def _count_classes(self, dataset: Dataset) -> dict[int, int]:
        counts: dict[int, int] = {}
        for i in range(len(dataset)):
            label = self._target_at(dataset, i)
            counts[label] = counts.get(label, 0) + 1
        return counts

    def _mask_logits(self, logits: torch.Tensor, seen_classes: list[int] | None = None) -> torch.Tensor:
        if not self.config.mask_unseen_classes:
            return logits
        seen = self.current_seen_classes if seen_classes is None else seen_classes
        if not seen or len(seen) >= logits.shape[1]:
            return logits
        masked = logits.clone()
        allowed = torch.zeros(logits.shape[1], dtype=torch.bool, device=logits.device)
        allowed[torch.tensor(seen, dtype=torch.long, device=logits.device)] = True
        masked[:, ~allowed] = torch.finfo(logits.dtype).min / 2
        return masked

    def fit_scenario(
        self,
        train_dataset: Dataset,
        test_dataset: Dataset,
        scenario: ClassIncrementalScenario,
        epochs_per_task: int = 1,
        batch_size: int = 32,
        num_workers: int = 0,
    ) -> dict[str, Any]:
        """Train ADAIL through all tasks in a class-incremental scenario."""

        task_accuracy_matrix: list[list[float]] = []
        task_results: list[TaskResult] = []

        for task_idx, task_classes in enumerate(scenario.tasks):
            seen_classes = scenario.seen_classes_until(task_idx)
            self.current_seen_classes = seen_classes
            current_task_dataset = DatasetSubsetByClass(train_dataset, task_classes)
            for cls, count in self._count_classes(current_task_dataset).items():
                self.observed_class_counts[cls] = self.observed_class_counts.get(cls, 0) + count

            train_dataset_available: Dataset = current_task_dataset
            if len(self.memory) > 0:
                train_dataset_available = ConcatDataset([current_task_dataset, self.memory.as_dataset()])

            if task_idx > 0:
                self.model.freeze_old_model()

            optimizer = self.build_optimizer()
            train_loader = DataLoader(
                train_dataset_available,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
            )
            epoch_logs: list[dict[str, float]] = []
            for _ in range(epochs_per_task):
                epoch_logs.append(self.train_epoch(train_loader, optimizer, seen_classes=seen_classes))

            # Prototype refresh uses only available information: new data + memory.
            refresh_dataset: Dataset = current_task_dataset
            if len(self.memory) > 0:
                refresh_dataset = ConcatDataset([current_task_dataset, self.memory.as_dataset()])
            refresh_loader = DataLoader(
                refresh_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
            )
            self.refresh_prototypes(refresh_loader)

            # Memory candidates are only the current task data, then old exemplars are pruned
            # according to uncertainty, forgetting, and class rarity.
            current_loader = DataLoader(
                current_task_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
            )
            self.update_memory(current_loader)

            row = []
            metrics_by_task: dict[str, float] = {}
            for eval_task_idx in range(task_idx + 1):
                eval_classes = scenario.tasks[eval_task_idx]
                eval_loader = DataLoader(
                    DatasetSubsetByClass(test_dataset, eval_classes),
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                )
                eval_metrics = self.evaluate(eval_loader, seen_classes=seen_classes)
                row.append(eval_metrics["accuracy"])
                for metric_name, metric_value in eval_metrics.items():
                    if isinstance(metric_value, float):
                        metrics_by_task[f"task_{eval_task_idx}_{metric_name}"] = metric_value
            task_accuracy_matrix.append(row)

            seen_eval_loader = DataLoader(
                DatasetSubsetByClass(test_dataset, seen_classes),
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
            )
            seen_metrics = self.evaluate(seen_eval_loader, seen_classes=seen_classes)
            class_scores = seen_metrics.get("per_class_accuracy", {})
            self._update_forgetting_estimates(class_scores)
            metrics_by_task.update(
                {
                    "seen_accuracy": float(seen_metrics["accuracy"]),
                    "seen_mae": float(seen_metrics["mae"]),
                    "seen_qwk": float(seen_metrics["qwk"]),
                    "last_loss": float(epoch_logs[-1]["loss"]) if epoch_logs else 0.0,
                }
            )

            task_results.append(
                TaskResult(
                    task_index=task_idx,
                    new_classes=list(task_classes),
                    seen_classes=seen_classes,
                    metrics=metrics_by_task,
                    per_class_accuracy={int(k): float(v) for k, v in class_scores.items()},
                    memory_quotas=dict(self.memory.quotas),
                )
            )

        return {
            "task_results": [
                {
                    "task_index": r.task_index,
                    "new_classes": r.new_classes,
                    "seen_classes": r.seen_classes,
                    "metrics": r.metrics,
                    "per_class_accuracy": r.per_class_accuracy,
                    "memory_quotas": r.memory_quotas,
                }
                for r in task_results
            ],
            "accuracy_matrix": task_accuracy_matrix,
            "average_accuracy": average_accuracy(task_accuracy_matrix),
            "average_forgetting": average_forgetting(task_accuracy_matrix),
            "class_forgetting": dict(self.class_forgetting),
            "observed_class_counts": dict(self.observed_class_counts),
        }

    def _update_forgetting_estimates(self, class_scores: dict[int, float]) -> None:
        self.current_class_scores = {int(k): float(v) for k, v in class_scores.items()}
        for cls, score in self.current_class_scores.items():
            best = self.best_class_scores.get(cls, score)
            self.class_forgetting[cls] = max(0.0, best - score)
            self.best_class_scores[cls] = max(best, score)

    def train_epoch(
        self,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        seen_classes: list[int] | None = None,
    ) -> dict[str, float]:
        self.model.train()
        if self.model.old_model is not None:
            self.model.old_model.eval()
        running: dict[str, float] = {
            "loss": 0.0,
            "ce": 0.0,
            "drift": 0.0,
            "proto": 0.0,
            "sep": 0.0,
            "ordinal": 0.0,
        }
        batches = 0
        ce_loss = nn.CrossEntropyLoss()
        for images, labels in tqdm(loader, desc="train", leave=False):
            images = images.to(self.device)
            labels = labels.to(self.device, dtype=torch.long)
            optimizer.zero_grad(set_to_none=True)
            logits_raw, features = self.model(images, return_features=True)
            logits = self._mask_logits(logits_raw, seen_classes)
            loss_ce = ce_loss(logits, labels)

            loss_drift = features.new_tensor(0.0)
            if self.model.old_model is not None:
                with torch.no_grad():
                    _, old_features = self.model.old_model(images, return_features=True)
                loss_drift = drift_alignment_loss(features, old_features, self.model.drift_aligner)

            loss_proto = self.model.prototypes.compactness_loss(features, labels)
            loss_sep = self.model.prototypes.separation_loss()
            loss_ord = features.new_tensor(0.0)
            if self.config.use_ordinal:
                loss_ord = ordinal_constraint_loss(logits, labels)

            total = (
                loss_ce
                + self.config.lambda_drift * loss_drift
                + self.config.lambda_proto * loss_proto
                + self.config.lambda_sep * loss_sep
                + self.config.lambda_ordinal * loss_ord
            )
            total.backward()
            optimizer.step()

            with torch.no_grad():
                self.model.prototypes.update(features.detach(), labels)

            running["loss"] += float(total.detach().cpu())
            running["ce"] += float(loss_ce.detach().cpu())
            running["drift"] += float(loss_drift.detach().cpu())
            running["proto"] += float(loss_proto.detach().cpu())
            running["sep"] += float(loss_sep.detach().cpu())
            running["ordinal"] += float(loss_ord.detach().cpu())
            batches += 1
        return {k: v / max(batches, 1) for k, v in running.items()}

    @torch.no_grad()
    def refresh_prototypes(self, loader: DataLoader) -> None:
        self.model.eval()
        for images, labels in tqdm(loader, desc="refresh-prototypes", leave=False):
            images_device = images.to(self.device)
            labels_device = labels.to(self.device, dtype=torch.long)
            _, features = self.model(images_device, return_features=True)
            self.model.prototypes.update(features, labels_device)

    @torch.no_grad()
    def update_memory(self, loader: DataLoader) -> None:
        self.model.eval()
        all_images: list[torch.Tensor] = []
        all_labels: list[torch.Tensor] = []
        all_logits: list[torch.Tensor] = []
        all_features: list[torch.Tensor] = []
        for images, labels in tqdm(loader, desc="update-memory", leave=False):
            images_device = images.to(self.device)
            labels_device = labels.to(self.device, dtype=torch.long)
            logits, features = self.model(images_device, return_features=True)
            logits = self._mask_logits(logits)
            all_images.append(images.cpu())
            all_labels.append(labels.cpu())
            all_logits.append(logits.cpu())
            all_features.append(features.cpu())

        if all_images:
            self.memory.update_from_tensors(
                images=torch.cat(all_images),
                labels=torch.cat(all_labels),
                logits=torch.cat(all_logits),
                features=torch.cat(all_features),
                class_counts=self.observed_class_counts,
                forgetting=self.class_forgetting,
            )

    @torch.no_grad()
    def evaluate(self, loader: DataLoader, seen_classes: list[int] | None = None) -> dict[str, Any]:
        self.model.eval()
        logits_all: list[torch.Tensor] = []
        labels_all: list[torch.Tensor] = []
        for images, labels in tqdm(loader, desc="eval", leave=False):
            images = images.to(self.device)
            labels = labels.to(self.device, dtype=torch.long)
            logits = self.model(images)
            logits = self._mask_logits(logits, seen_classes)
            logits_all.append(logits.cpu())
            labels_all.append(labels.cpu())
        if not logits_all:
            return {"accuracy": 0.0, "mae": 0.0, "qwk": 0.0, "per_class_accuracy": {}}
        logits = torch.cat(logits_all)
        labels = torch.cat(labels_all)
        preds = logits.argmax(dim=1)
        return {
            "accuracy": accuracy(logits, labels),
            "mae": mean_absolute_error(preds, labels),
            "qwk": quadratic_weighted_kappa(preds, labels, num_classes=self.model.num_classes),
            "per_class_accuracy": per_class_accuracy(preds, labels, classes=seen_classes),
        }

    def save_checkpoint(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": self.model.state_dict(),
                "config": self.config.to_dict(),
                "memory": self.memory.state_dict(),
                "best_class_scores": self.best_class_scores,
                "class_forgetting": self.class_forgetting,
                "observed_class_counts": self.observed_class_counts,
            },
            path,
        )

    def load_checkpoint(self, path: str | Path, map_location: str | torch.device | None = None) -> None:
        state = torch.load(path, map_location=map_location or self.device)
        self.model.load_state_dict(state["model"])
        self.memory.load_state_dict(state["memory"])
        self.best_class_scores = {int(k): float(v) for k, v in state.get("best_class_scores", {}).items()}
        self.class_forgetting = {int(k): float(v) for k, v in state.get("class_forgetting", {}).items()}
        self.observed_class_counts = {
            int(k): int(v) for k, v in state.get("observed_class_counts", {}).items()
        }
