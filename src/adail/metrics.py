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


def mean_absolute_error(
    preds: np.ndarray | torch.Tensor, labels: np.ndarray | torch.Tensor
) -> float:
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


def average_incremental_accuracy(seen_accuracies: list[float]) -> float:
    """Mean over tasks of the accuracy on all test samples of the seen classes.

    This is the "average incremental accuracy" of iCaRL/LO2LN: every task's
    accuracy is weighted by test samples, not by task.
    """

    return float(np.mean(seen_accuracies)) if seen_accuracies else 0.0


def old_to_new_confusion(
    confusion: np.ndarray | list[list[int]],
    old_classes: list[int],
    new_classes: list[int],
) -> float:
    """Share of old-class test samples predicted as one of the new classes.

    Measures the bias towards the classes added in the current task.
    """

    cm = np.asarray(confusion)
    if cm.size == 0 or not old_classes or not new_classes:
        return 0.0
    old = [int(c) for c in old_classes]
    new = [int(c) for c in new_classes]
    total = cm[old].sum()
    return float(cm[np.ix_(old, new)].sum() / total) if total else 0.0


def average_accuracy(task_accuracy_matrix: list[list[float]]) -> float:
    """Mean over tasks of the unweighted mean of per-task-block accuracies.

    Every block of classes counts equally, whatever its number of test samples.
    This is not the iCaRL/LO2LN average incremental accuracy; use
    :func:`average_incremental_accuracy` to compare with LO2LN.
    """

    if not task_accuracy_matrix:
        return 0.0
    row_means = [float(np.mean(row)) for row in task_accuracy_matrix if row]
    return float(np.mean(row_means)) if row_means else 0.0


def average_forgetting(task_accuracy_matrix: list[list[float]]) -> float:
    """Average forgetting for a class/task-incremental experiment."""

    if len(task_accuracy_matrix) <= 1:
        return 0.0
    final_row = task_accuracy_matrix[-1]
    forgetting_values: list[float] = []
    for task_idx in range(len(final_row) - 1):
        previous_scores = [
            row[task_idx] for row in task_accuracy_matrix[task_idx:-1] if len(row) > task_idx
        ]
        if not previous_scores:
            continue
        forgetting_values.append(max(previous_scores) - final_row[task_idx])
    return float(np.mean(forgetting_values)) if forgetting_values else 0.0


def classification_metrics(
    preds: np.ndarray | torch.Tensor,
    labels: np.ndarray | torch.Tensor,
    average: str = "macro",
) -> dict[str, float]:
    """Calculate precision, recall, and F1-score."""
    from sklearn.metrics import f1_score, precision_score, recall_score

    p = preds.detach().cpu().numpy() if isinstance(preds, torch.Tensor) else np.asarray(preds)
    y = labels.detach().cpu().numpy() if isinstance(labels, torch.Tensor) else np.asarray(labels)
    p = p.astype(int).reshape(-1)
    y = y.astype(int).reshape(-1)
    if p.size == 0 or y.size == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    prec = float(precision_score(y, p, average=average, zero_division=0))
    rec = float(recall_score(y, p, average=average, zero_division=0))
    f1 = float(f1_score(y, p, average=average, zero_division=0))
    return {"precision": prec, "recall": rec, "f1": f1}
