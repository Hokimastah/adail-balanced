from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .config import ADAILConfig
from .data import (
    BalancedReplayBatchSampler,
    ClassIncrementalScenario,
    DatasetSubsetByClass,
    ReplayDataset,
)
from .memory import Exemplar, ExemplarMemory
from .metrics import (
    accuracy,
    average_accuracy,
    average_forgetting,
    mean_absolute_error,
    per_class_accuracy,
    quadratic_weighted_kappa,
)
from .models import ADAILNet
from .modules import (
    class_balanced_focal_loss,
    drift_alignment_loss,
    logit_distillation_loss,
    old_new_margin_ranking_loss,
    ordinal_constraint_loss,
)


@dataclass
class TaskResult:
    task_index: int
    new_classes: list[int]
    seen_classes: list[int]
    metrics: dict[str, float] = field(default_factory=dict)
    per_class_accuracy: dict[int, float] = field(default_factory=dict)
    memory_quotas: dict[int, int] = field(default_factory=dict)
    epoch_logs: list[dict[str, Any]] = field(default_factory=list)


class ADAILTrainer:
    """Enhanced ADAIL class-incremental trainer.

    The trainer combines ADAIL's drift-aware prototypes and adaptive quota with
    LO2LN-inspired representation expansion, alternating objectives, cosine
    classification, focal rebalancing, logits distillation, auxiliary learning,
    and old-vs-new margin ranking. Only current-task data and stored exemplars
    are available after the base task.
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
        parameters = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
        return torch.optim.SGD(
            parameters,
            lr=opt.lr,
            momentum=opt.momentum,
            weight_decay=opt.weight_decay,
            nesterov=opt.nesterov,
        )

    @staticmethod
    def _target_at(dataset: Dataset, idx: int) -> int:
        if hasattr(dataset, "targets"):
            target = dataset.targets[idx]
        else:
            target = dataset[idx][1]
        return int(torch.as_tensor(target).reshape(-1)[0].item())

    def _count_classes(self, dataset: Dataset) -> dict[int, int]:
        counts: dict[int, int] = {}
        for index in range(len(dataset)):
            label = self._target_at(dataset, index)
            counts[label] = counts.get(label, 0) + 1
        return counts

    @staticmethod
    def _unpack_batch(batch):
        if len(batch) == 3:
            images, labels, memory_flags = batch
        else:
            images, labels = batch
            memory_flags = torch.zeros(len(labels), dtype=torch.bool)
        return images, labels, memory_flags

    def _mask_logits(
        self,
        logits: torch.Tensor,
        seen_classes: list[int] | None = None,
    ) -> torch.Tensor:
        if not self.config.mask_unseen_classes:
            return logits
        seen = self.current_seen_classes if seen_classes is None else seen_classes
        if not seen or len(seen) >= logits.shape[1]:
            return logits
        masked = logits.clone()
        allowed = torch.zeros(logits.shape[1], dtype=torch.bool, device=logits.device)
        allowed[torch.as_tensor(seen, dtype=torch.long, device=logits.device)] = True
        masked[:, ~allowed] = torch.finfo(logits.dtype).min / 2
        return masked

    def _build_train_loader(
        self,
        current_dataset: Dataset,
        batch_size: int,
        num_workers: int,
    ) -> tuple[DataLoader, ReplayDataset]:
        memory_dataset = self.memory.as_dataset() if len(self.memory) else None
        replay_dataset = ReplayDataset(current_dataset, memory_dataset)
        if self.config.balanced_replay and replay_dataset.memory_size > 0:
            sampler = BalancedReplayBatchSampler(
                replay_dataset.current_targets,
                replay_dataset.memory_targets,
                batch_size=batch_size,
                memory_fraction=self.config.replay_memory_fraction,
                seed=self.config.sampler_seed,
            )
            loader = DataLoader(
                replay_dataset,
                batch_sampler=sampler,
                num_workers=num_workers,
            )
        else:
            loader = DataLoader(
                replay_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
            )
        return loader, replay_dataset

    @torch.no_grad()
    def imprint_new_classes(
        self,
        dataset: Dataset,
        class_ids: list[int],
        batch_size: int,
        num_workers: int,
    ) -> None:
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        features_all: list[torch.Tensor] = []
        labels_all: list[torch.Tensor] = []
        self.model.eval()
        for batch in loader:
            images, labels, _ = self._unpack_batch(batch)
            outputs = self.model.forward_details(images.to(self.device))
            features_all.append(outputs["features"].detach())
            labels_all.append(labels.to(self.device, dtype=torch.long))
        if features_all:
            self.model.imprint_classes(
                class_ids,
                torch.cat(features_all),
                torch.cat(labels_all),
            )

    def fit_scenario(
        self,
        train_dataset: Dataset,
        test_dataset: Dataset,
        scenario: ClassIncrementalScenario,
        epochs_per_task: int = 1,
        batch_size: int = 32,
        num_workers: int = 0,
    ) -> dict[str, Any]:
        """Train through every task and return serializable continual metrics."""

        task_accuracy_matrix: list[list[float]] = []
        task_results: list[TaskResult] = []

        for task_idx, task_classes in enumerate(scenario.tasks):
            seen_classes = scenario.seen_classes_until(task_idx)
            old_classes = scenario.seen_classes_until(task_idx - 1) if task_idx > 0 else []
            self.current_seen_classes = seen_classes
            current_task_dataset = DatasetSubsetByClass(train_dataset, task_classes)
            for class_id, count in self._count_classes(current_task_dataset).items():
                self.observed_class_counts[class_id] = (
                    self.observed_class_counts.get(class_id, 0) + count
                )

            # Snapshot before expansion so the teacher exactly represents t-1.
            if task_idx > 0:
                self.model.freeze_old_model()
                if self.config.dynamic_expansion:
                    self.model.expand_representation()

            if self.config.imprint_new_classes:
                self.imprint_new_classes(
                    current_task_dataset,
                    list(task_classes),
                    batch_size,
                    num_workers,
                )

            self.model.configure_phase("new")
            optimizer = self.build_optimizer()
            train_loader, replay_dataset = self._build_train_loader(
                current_task_dataset,
                batch_size,
                num_workers,
            )
            available_counts = self._count_classes(replay_dataset)
            epoch_logs: list[dict[str, Any]] = []

            for epoch in range(epochs_per_task):
                use_old_phase = (
                    self.config.alternating_objectives
                    and task_idx > 0
                    and (epoch + 1) % self.config.old_phase_frequency == 0
                )
                phase = "old" if use_old_phase else "new"
                print(
                    f"Task {task_idx + 1}/{len(scenario.tasks)} - "
                    f"Epoch {epoch + 1}/{epochs_per_task} - phase={phase}"
                )
                logs = self.train_epoch(
                    train_loader,
                    optimizer,
                    seen_classes=seen_classes,
                    old_classes=old_classes,
                    new_classes=list(task_classes),
                    class_counts=available_counts,
                    phase=phase,
                )
                logs["epoch"] = epoch + 1
                logs["phase"] = phase
                epoch_logs.append(logs)

            self.model.configure_phase("new")
            refresh_loader = DataLoader(
                replay_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
            )
            self.refresh_prototypes(refresh_loader)

            row: list[float] = []
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
                row.append(float(eval_metrics["accuracy"]))
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

            # Rebuild after evaluation so adaptive quotas use fresh forgetting.
            memory_before_update = self.memory.as_dataset() if len(self.memory) else None
            memory_candidates: Dataset = current_task_dataset
            if memory_before_update is not None:
                memory_candidates = ReplayDataset(current_task_dataset, memory_before_update)
            candidate_loader = DataLoader(
                memory_candidates,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
            )
            self.update_memory(candidate_loader, replace=True)

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
                    per_class_accuracy={
                        int(key): float(value) for key, value in class_scores.items()
                    },
                    memory_quotas=dict(self.memory.quotas),
                    epoch_logs=epoch_logs,
                )
            )

        return {
            "task_results": [
                {
                    "task_index": result.task_index,
                    "new_classes": result.new_classes,
                    "seen_classes": result.seen_classes,
                    "metrics": result.metrics,
                    "per_class_accuracy": result.per_class_accuracy,
                    "memory_quotas": result.memory_quotas,
                    "epoch_logs": result.epoch_logs,
                }
                for result in task_results
            ],
            "accuracy_matrix": task_accuracy_matrix,
            "average_accuracy": average_accuracy(task_accuracy_matrix),
            "average_forgetting": average_forgetting(task_accuracy_matrix),
            "class_forgetting": dict(self.class_forgetting),
            "observed_class_counts": dict(self.observed_class_counts),
            "num_representation_branches": self.model.num_branches,
        }

    def _update_forgetting_estimates(self, class_scores: dict[int, float]) -> None:
        self.current_class_scores = {int(key): float(value) for key, value in class_scores.items()}
        for class_id, score in self.current_class_scores.items():
            best = self.best_class_scores.get(class_id, score)
            self.class_forgetting[class_id] = max(0.0, best - score)
            self.best_class_scores[class_id] = max(best, score)

    def train_epoch(
        self,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        seen_classes: list[int],
        old_classes: list[int],
        new_classes: list[int],
        class_counts: dict[int, int],
        phase: str = "new",
    ) -> dict[str, float]:
        self.model.train()
        self.model.configure_phase(phase)
        if self.model.old_model is not None:
            self.model.old_model.eval()

        metric_names = [
            "loss",
            "classification",
            "auxiliary",
            "distillation",
            "ranking",
            "drift",
            "prototype",
            "separation",
            "ordinal",
            "accuracy",
        ]
        running = {name: 0.0 for name in metric_names}
        batches = 0

        beta = self.config.old_focal_beta if phase == "old" else self.config.focal_beta
        gamma = self.config.old_focal_gamma if phase == "old" else self.config.focal_gamma
        distill_multiplier = self.config.old_phase_distill_multiplier if phase == "old" else 1.0
        ranking_multiplier = self.config.old_phase_ranking_multiplier if phase == "old" else 1.0

        for batch in tqdm(loader, desc=f"train-{phase}", leave=False):
            images, labels, memory_flags = self._unpack_batch(batch)
            images = images.to(self.device)
            labels = labels.to(self.device, dtype=torch.long)
            memory_flags = memory_flags.to(self.device, dtype=torch.bool)
            optimizer.zero_grad(set_to_none=True)

            outputs = self.model.forward_details(images)
            logits_raw = outputs["logits"]
            logits = self._mask_logits(logits_raw, seen_classes)
            features = outputs["features"]

            if self.config.use_class_balanced_focal:
                loss_classification = class_balanced_focal_loss(
                    logits,
                    labels,
                    class_counts,
                    beta=beta,
                    gamma=gamma,
                )
            else:
                loss_classification = nn.functional.cross_entropy(logits, labels)

            zero = features.new_zeros(())
            loss_auxiliary = zero
            if (
                phase == "new"
                and old_classes
                and self.model.has_auxiliary_branch
                and self.config.lambda_aux > 0
            ):
                aux_logits = self._mask_logits(outputs["aux_logits"], seen_classes)
                loss_auxiliary = class_balanced_focal_loss(
                    aux_logits,
                    labels,
                    class_counts,
                    beta=self.config.focal_beta,
                    gamma=self.config.focal_gamma,
                )

            loss_drift = zero
            loss_distillation = zero
            if self.model.old_model is not None:
                with torch.no_grad():
                    teacher_outputs = self.model.old_model.forward_details(images)
                loss_drift = drift_alignment_loss(
                    features,
                    teacher_outputs["features"],
                    self.model.drift_aligner,
                )
                distill_mask = memory_flags if self.config.distill_memory_only else None
                loss_distillation = logit_distillation_loss(
                    logits_raw,
                    teacher_outputs["logits"],
                    old_classes,
                    temperature=self.config.distill_temperature,
                    sample_mask=distill_mask,
                    confidence_threshold=self.config.distill_confidence_threshold,
                )

            loss_ranking = old_new_margin_ranking_loss(
                logits_raw,
                labels,
                old_classes,
                new_classes,
                margin=self.config.ranking_margin,
                topk=self.config.ranking_topk,
                ordinal_aware=self.config.ordinal_aware_margin and self.config.use_ordinal,
            )
            loss_prototype = self.model.prototypes.compactness_loss(features, labels)
            loss_separation = self.model.prototypes.differentiable_separation_loss(
                features,
                labels,
                seen_classes,
                ordinal_aware=self.config.ordinal_aware_margin and self.config.use_ordinal,
            )
            loss_ordinal = (
                ordinal_constraint_loss(logits, labels) if self.config.use_ordinal else zero
            )

            total = (
                loss_classification
                + self.config.lambda_aux * loss_auxiliary
                + self.config.lambda_distill * distill_multiplier * loss_distillation
                + self.config.lambda_ranking * ranking_multiplier * loss_ranking
                + self.config.lambda_drift * loss_drift
                + self.config.lambda_proto * loss_prototype
                + self.config.lambda_sep * loss_separation
                + self.config.lambda_ordinal * loss_ordinal
            )
            total.backward()
            if phase == "old":
                self.model.mask_new_class_gradients(new_classes)
            optimizer.step()

            with torch.no_grad():
                self.model.prototypes.update(features.detach(), labels)
                batch_accuracy = (logits.argmax(dim=1) == labels).float().mean()

            values = {
                "loss": total,
                "classification": loss_classification,
                "auxiliary": loss_auxiliary,
                "distillation": loss_distillation,
                "ranking": loss_ranking,
                "drift": loss_drift,
                "prototype": loss_prototype,
                "separation": loss_separation,
                "ordinal": loss_ordinal,
                "accuracy": batch_accuracy,
            }
            for name, value in values.items():
                running[name] += float(value.detach().cpu())
            batches += 1

        return {name: value / max(batches, 1) for name, value in running.items()}

    @torch.no_grad()
    def refresh_prototypes(self, loader: DataLoader) -> None:
        self.model.eval()
        for batch in tqdm(loader, desc="refresh-prototypes", leave=False):
            images, labels, _ = self._unpack_batch(batch)
            outputs = self.model.forward_details(images.to(self.device))
            self.model.prototypes.update(
                outputs["features"],
                labels.to(self.device, dtype=torch.long),
            )

    @torch.no_grad()
    def update_memory(self, loader: DataLoader, *, replace: bool = True) -> None:
        self.model.eval()
        images_all: list[torch.Tensor] = []
        labels_all: list[torch.Tensor] = []
        logits_all: list[torch.Tensor] = []
        features_all: list[torch.Tensor] = []
        for batch in tqdm(loader, desc="update-memory", leave=False):
            images, labels, _ = self._unpack_batch(batch)
            images_device = images.to(self.device)
            labels_device = labels.to(self.device, dtype=torch.long)
            outputs = self.model.forward_details(images_device)
            logits = self._mask_logits(outputs["logits"], self.current_seen_classes)
            images_all.append(images.cpu())
            labels_all.append(labels_device.cpu())
            logits_all.append(logits.cpu())
            features_all.append(outputs["features"].cpu())

        if images_all:
            self.memory.update_from_tensors(
                images=torch.cat(images_all),
                labels=torch.cat(labels_all),
                logits=torch.cat(logits_all),
                features=torch.cat(features_all),
                class_counts=self.observed_class_counts,
                forgetting=self.class_forgetting,
                replace=replace,
                selection=self.config.memory_selection,
                representative_fraction=self.config.memory_representative_fraction,
                uncertainty_fraction=self.config.memory_uncertainty_fraction,
            )

    @torch.no_grad()
    def evaluate(
        self,
        loader: DataLoader,
        seen_classes: list[int] | None = None,
    ) -> dict[str, Any]:
        self.model.eval()
        logits_all: list[torch.Tensor] = []
        labels_all: list[torch.Tensor] = []
        for batch in tqdm(loader, desc="eval", leave=False):
            images, labels, _ = self._unpack_batch(batch)
            labels = labels.to(self.device, dtype=torch.long)
            logits = self._mask_logits(self.model(images.to(self.device)), seen_classes)
            logits_all.append(logits.cpu())
            labels_all.append(labels.cpu())
        if not logits_all:
            return {"accuracy": 0.0, "mae": 0.0, "qwk": 0.0, "per_class_accuracy": {}}
        logits = torch.cat(logits_all)
        labels = torch.cat(labels_all)
        predictions = logits.argmax(dim=1)
        return {
            "accuracy": accuracy(logits, labels),
            "mae": mean_absolute_error(predictions, labels),
            "qwk": quadratic_weighted_kappa(
                predictions,
                labels,
                num_classes=self.model.num_classes,
            ),
            "per_class_accuracy": per_class_accuracy(
                predictions,
                labels,
                classes=seen_classes,
            ),
        }

    def save_checkpoint(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "format_version": 2,
                "model": self.model.state_dict(),
                "num_branches": self.model.num_branches,
                "config": self.config.to_dict(),
                "memory": self.memory.state_dict(),
                "best_class_scores": self.best_class_scores,
                "class_forgetting": self.class_forgetting,
                "observed_class_counts": self.observed_class_counts,
                "current_seen_classes": self.current_seen_classes,
            },
            path,
        )

    def load_checkpoint(
        self,
        path: str | Path,
        map_location: str | torch.device | None = None,
    ) -> None:
        load_kwargs = {"map_location": map_location or self.device}
        if hasattr(torch.serialization, "safe_globals"):
            with torch.serialization.safe_globals([Exemplar]):
                state = torch.load(path, weights_only=True, **load_kwargs)
        else:
            state = torch.load(path, **load_kwargs)

        self.model.ensure_num_branches(int(state.get("num_branches", 1)))
        self.model.load_state_dict(state["model"])
        self.model.clear_old_model()
        self.memory.load_state_dict(state["memory"])
        self.best_class_scores = {
            int(key): float(value) for key, value in state.get("best_class_scores", {}).items()
        }
        self.class_forgetting = {
            int(key): float(value) for key, value in state.get("class_forgetting", {}).items()
        }
        self.observed_class_counts = {
            int(key): int(value) for key, value in state.get("observed_class_counts", {}).items()
        }
        self.current_seen_classes = [
            int(value) for value in state.get("current_seen_classes", range(self.model.num_classes))
        ]
