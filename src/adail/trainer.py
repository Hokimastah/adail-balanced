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
    average_incremental_accuracy,
    classification_metrics,
    mean_absolute_error,
    old_to_new_confusion,
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

    def _memory_capacity(self, num_seen_classes: int) -> int | None:
        """Exemplars to keep after a task; None means the whole memory_size."""

        if self.config.memory_budget == "global":
            return None
        per_class = self.config.memory_size // max(1, self.config.num_classes)
        return per_class * num_seen_classes

    def _step_with_frozen_rows(self, optimizer: torch.optim.Optimizer, rows: list[int]) -> None:
        """Run ``optimizer.step()`` leaving the given classifier rows untouched.

        Zeroing their gradient is not enough: SGD momentum and weight decay
        would still move them. Their values and momentum are restored instead,
        matching a parameter that sits out the step.
        """

        classifier = self.model.classifier
        params = [
            param
            for param in (getattr(classifier, "weight", None), getattr(classifier, "bias", None))
            if isinstance(param, torch.Tensor) and param.requires_grad
        ]
        if not rows or not params:
            optimizer.step()
            return
        index = torch.as_tensor(rows, dtype=torch.long, device=params[0].device)
        saved = []
        for param in params:
            buffer = optimizer.state.get(param, {}).get("momentum_buffer")
            saved.append(
                (
                    param,
                    param.detach()[index].clone(),
                    None if buffer is None else buffer[index].clone(),
                )
            )
        optimizer.step()
        with torch.no_grad():
            for param, values, buffer in saved:
                param[index] = values
                state_buffer = optimizer.state.get(param, {}).get("momentum_buffer")
                if state_buffer is not None:
                    state_buffer[index] = 0 if buffer is None else buffer

    def _replay_memory_fraction(self, replay_dataset: ReplayDataset) -> float:
        """Share of each balanced batch reserved for exemplars."""

        if self.config.replay_fraction_mode == "fixed":
            return self.config.replay_memory_fraction
        # Proportional: n_old / n_seen, so every seen class gets about the same
        # number of slots per batch.
        new_classes = set(replay_dataset.current_targets)
        old_classes = set(replay_dataset.memory_targets) - new_classes
        if not old_classes:
            return 0.0
        return len(old_classes) / (len(old_classes) + len(new_classes))

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
                memory_fraction=self._replay_memory_fraction(replay_dataset),
                seed=self.config.sampler_seed,
                epoch_length=self.config.replay_epoch_length,
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
        train_eval_dataset: Dataset | None = None,
    ) -> dict[str, Any]:
        """Train through every task and return serializable continual metrics.

        ``train_eval_dataset`` holds the same samples as ``train_dataset``, in the
        same order, but with the deterministic evaluation transform. When given,
        weight imprinting, the prototype refresh and exemplar scoring see clean
        images instead of randomly augmented ones (LO2LN also extracts herding
        features with its test transform).
        """

        if train_eval_dataset is not None and len(train_eval_dataset) != len(train_dataset):
            raise ValueError("train_eval_dataset must hold the same samples as train_dataset")
        task_accuracy_matrix: list[list[float]] = []
        task_results: list[TaskResult] = []
        if self.config.replay_reaugment:
            # Exemplars are kept as indices into train_dataset and reloaded
            # through its training transform each time they are replayed.
            self.memory.source_dataset = train_dataset
        # Index-based exemplars are scored through the clean view when available.
        scoring_source = train_eval_dataset

        for task_idx, task_classes in enumerate(scenario.tasks):
            seen_classes = scenario.seen_classes_until(task_idx)
            old_classes = scenario.seen_classes_until(task_idx - 1) if task_idx > 0 else []
            self.current_seen_classes = seen_classes
            current_task_dataset = DatasetSubsetByClass(train_dataset, task_classes)
            current_scoring_dataset: Dataset = current_task_dataset
            if train_eval_dataset is not None:
                current_scoring_dataset = DatasetSubsetByClass(train_eval_dataset, task_classes)
                if current_scoring_dataset.indices != current_task_dataset.indices:
                    raise ValueError(
                        "train_eval_dataset must list the same samples as train_dataset"
                    )
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
                    current_scoring_dataset,
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
            refresh_dataset: Dataset = replay_dataset
            if train_eval_dataset is not None:
                refresh_dataset = ReplayDataset(
                    current_scoring_dataset,
                    self.memory.as_dataset(source=scoring_source) if len(self.memory) else None,
                )
            refresh_loader = DataLoader(
                refresh_dataset,
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
            memory_before_update = (
                self.memory.as_dataset(source=scoring_source) if len(self.memory) else None
            )
            memory_candidates: Dataset = current_scoring_dataset
            if memory_before_update is not None:
                memory_candidates = ReplayDataset(current_scoring_dataset, memory_before_update)
            # Candidates are referenced by dataset index whenever that is enough
            # to recover them: with re-augmentation, or when the clean view can
            # reload the selected exemplars as fixed tensors. Only the selected
            # exemplars are then held in RAM, not every candidate image.
            # Same order as the unshuffled candidate loader: current task first,
            # then the previous exemplars.
            candidate_sources: list[int | None] | None = None
            if self.config.replay_reaugment or train_eval_dataset is not None:
                candidate_sources = list(current_task_dataset.indices)
                if memory_before_update is not None:
                    candidate_sources.extend(memory_before_update.source_indices)
            candidate_loader = DataLoader(
                memory_candidates,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
            )
            self.update_memory(
                candidate_loader,
                replace=True,
                source_indices=candidate_sources,
                capacity=self._memory_capacity(len(seen_classes)),
                materialize_from=None if self.config.replay_reaugment else train_eval_dataset,
            )

            metrics_by_task.update(
                {
                    "seen_accuracy": float(seen_metrics["accuracy"]),
                    "seen_mae": float(seen_metrics["mae"]),
                    "seen_qwk": float(seen_metrics["qwk"]),
                    "seen_precision": float(seen_metrics.get("precision", 0.0)),
                    "seen_recall": float(seen_metrics.get("recall", 0.0)),
                    # Macro recall over the seen classes.
                    "seen_balanced_accuracy": float(seen_metrics.get("recall", 0.0)),
                    "seen_f1": float(seen_metrics.get("f1", 0.0)),
                    "old_to_new_confusion": old_to_new_confusion(
                        seen_metrics.get("confusion_matrix", []), old_classes, list(task_classes)
                    ),
                    "confusion_matrix": seen_metrics.get("confusion_matrix", []),
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
            # Task-weighted; kept for compatibility with 0.4.0 histories.
            "average_accuracy": average_accuracy(task_accuracy_matrix),
            # Sample-weighted, as reported by iCaRL/LO2LN.
            "average_incremental_accuracy": average_incremental_accuracy(
                [result.metrics["seen_accuracy"] for result in task_results]
            ),
            "average_old_to_new_confusion": (
                sum(result.metrics["old_to_new_confusion"] for result in task_results[1:])
                / max(1, len(task_results) - 1)
            ),
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

            ranking_scores = (
                outputs["cosine_logits"] if self.config.ranking_space == "cosine" else logits_raw
            )
            loss_ranking = old_new_margin_ranking_loss(
                ranking_scores,
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
                # New-class classifier rows sit out the stability phase entirely.
                self._step_with_frozen_rows(optimizer, new_classes)
            else:
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
    def update_memory(
        self,
        loader: DataLoader,
        *,
        replace: bool = True,
        source_indices: list[int | None] | None = None,
        capacity: int | None = None,
        materialize_from: Dataset | None = None,
    ) -> None:
        self.model.eval()
        # With a source index for every candidate, the candidate images need not
        # be kept in RAM: exemplars are stored as indices, or reloaded from
        # `materialize_from` once selected.
        keep_images = source_indices is None or any(index is None for index in source_indices)
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
            if keep_images:
                images_all.append(images.cpu())
            labels_all.append(labels_device.cpu())
            logits_all.append(logits.cpu())
            features_all.append(outputs["features"].cpu())

        if labels_all:
            self.memory.update_from_tensors(
                images=torch.cat(images_all) if keep_images else None,
                labels=torch.cat(labels_all),
                logits=torch.cat(logits_all),
                features=torch.cat(features_all),
                class_counts=self.observed_class_counts,
                forgetting=self.class_forgetting,
                replace=replace,
                selection=self.config.memory_selection,
                representative_fraction=self.config.memory_representative_fraction,
                uncertainty_fraction=self.config.memory_uncertainty_fraction,
                source_indices=source_indices,
                capacity=capacity,
                materialize_from=materialize_from,
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
            return {
                "accuracy": 0.0,
                "mae": 0.0,
                "qwk": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "confusion_matrix": [],
                "per_class_accuracy": {},
            }
        logits = torch.cat(logits_all)
        labels = torch.cat(labels_all)
        predictions = logits.argmax(dim=1)
        cls_metrics = classification_metrics(predictions, labels, average="macro")
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(
            labels.numpy(),
            predictions.numpy(),
            labels=list(range(self.model.num_classes)),
        )
        return {
            "accuracy": accuracy(logits, labels),
            "mae": mean_absolute_error(predictions, labels),
            "qwk": quadratic_weighted_kappa(
                predictions,
                labels,
                num_classes=self.model.num_classes,
            ),
            "precision": cls_metrics["precision"],
            "recall": cls_metrics["recall"],
            "f1": cls_metrics["f1"],
            "confusion_matrix": cm.tolist(),
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
