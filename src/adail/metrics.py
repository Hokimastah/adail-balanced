from __future__ import annotations

import numpy as np
import torch


def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Top-1 accuracy for logits and integer labels."""

    if labels.numel() == 0:
        return 0.0
    preds = logits.argmax(dim=1)
    return float((preds == labels).float().mean().item())


def per_class_accuracy(
    logits_or_preds: torch.Tensor | np.ndarray,
    labels: torch.Tensor | np.ndarray,
    classes: list[int] | None = None,
) -> dict[int, float]:
    """Class-wise accuracy used to estimate per-class forgetting."""

    if isinstance(logits_or_preds, torch.Tensor):
        arr = logits_or_preds.detach().cpu()
        preds = arr.argmax(dim=1).numpy() if arr.ndim == 2 else arr.numpy()
    else:
        arr = np.asarray(logits_or_preds)
        preds = arr.argmax(axis=1) if arr.ndim == 2 else arr
    y = labels.detach().cpu().numpy() if isinstance(labels, torch.Tensor) else np.asarray(labels)
    y = y.astype(int).reshape(-1)
    preds = preds.astype(int).reshape(-1)
    if classes is None:
        classes = sorted(int(c) for c in np.unique(y).tolist())
    out: dict[int, float] = {}
    for cls in classes:
        mask = y == int(cls)
        out[int(cls)] = float(np.mean(preds[mask] == y[mask])) if mask.any() else 0.0
    return out


def mean_absolute_error(preds: np.ndarray | torch.Tensor, labels: np.ndarray | torch.Tensor) -> float:
    """Mean absolute error for ordinal predictions."""

    p = preds.detach().cpu().numpy() if isinstance(preds, torch.Tensor) else np.asarray(preds)
    y = labels.detach().cpu().numpy() if isinstance(labels, torch.Tensor) else np.asarray(labels)
    if len(y) == 0:
        return 0.0
    return float(np.mean(np.abs(p - y)))


def quadratic_weighted_kappa(
    preds: np.ndarray | torch.Tensor,
    labels: np.ndarray | torch.Tensor,
    num_classes: int | None = None,
) -> float:
    """Quadratic Weighted Kappa (QWK) for ordinal classification."""

    p = preds.detach().cpu().numpy() if isinstance(preds, torch.Tensor) else np.asarray(preds)
    y = labels.detach().cpu().numpy() if isinstance(labels, torch.Tensor) else np.asarray(labels)
    p = p.astype(int).reshape(-1)
    y = y.astype(int).reshape(-1)
    if p.size == 0:
        return 0.0
    if num_classes is None:
        num_classes = int(max(p.max(initial=0), y.max(initial=0)) + 1)
    observed = np.zeros((num_classes, num_classes), dtype=float)
    for yi, pi in zip(y, p, strict=False):
        if 0 <= yi < num_classes and 0 <= pi < num_classes:
            observed[yi, pi] += 1
    hist_y = observed.sum(axis=1)
    hist_p = observed.sum(axis=0)
    expected = np.outer(hist_y, hist_p) / max(observed.sum(), 1.0)
    weights = np.zeros_like(observed)
    denom = max((num_classes - 1) ** 2, 1)
    for i in range(num_classes):
        for j in range(num_classes):
            weights[i, j] = ((i - j) ** 2) / denom
    numerator = (weights * observed).sum()
    denominator = (weights * expected).sum()
    if denominator == 0:
        return 1.0 if numerator == 0 else 0.0
    return float(1.0 - numerator / denominator)


def average_accuracy(task_accuracy_matrix: list[list[float]]) -> float:
    """Average accuracy from a CIL task accuracy matrix."""

    vals = [score for row in task_accuracy_matrix for score in row]
    return float(np.mean(vals)) if vals else 0.0


def average_forgetting(task_accuracy_matrix: list[list[float]]) -> float:
    """Average forgetting for a class/task-incremental experiment."""

    if len(task_accuracy_matrix) <= 1:
        return 0.0
    final_row = task_accuracy_matrix[-1]
    forgetting_values: list[float] = []
    for task_idx in range(len(final_row) - 1):
        previous_scores = [row[task_idx] for row in task_accuracy_matrix[task_idx:-1] if len(row) > task_idx]
        if not previous_scores:
            continue
        forgetting_values.append(max(previous_scores) - final_row[task_idx])
    return float(np.mean(forgetting_values)) if forgetting_values else 0.0
