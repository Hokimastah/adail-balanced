from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from torch.nn import functional as F


def effective_number_weights(
    class_counts: Mapping[int, int] | Sequence[int] | torch.Tensor,
    num_classes: int,
    beta: float = 0.99,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return class-balanced weights based on the effective number of samples.

    Classes with no currently available samples receive weight zero. Positive
    weights are normalized to have mean one, keeping the classification-loss
    scale comparable with ordinary cross entropy.
    """

    if isinstance(class_counts, Mapping):
        counts = torch.zeros(num_classes, dtype=dtype, device=device)
        for class_id, count in class_counts.items():
            if 0 <= int(class_id) < num_classes:
                counts[int(class_id)] = max(0, int(count))
    else:
        counts = torch.as_tensor(class_counts, dtype=dtype, device=device).flatten()
        if counts.numel() < num_classes:
            counts = F.pad(counts, (0, num_classes - counts.numel()))
        counts = counts[:num_classes].clamp_min(0)

    present = counts > 0
    weights = torch.zeros_like(counts)
    if not bool(present.any()):
        return weights
    if beta <= 0.0:
        weights[present] = 1.0
        return weights
    if not 0.0 < beta < 1.0:
        raise ValueError("beta must be in (0, 1), or <= 0 to disable balancing")

    effective = 1.0 - torch.pow(torch.as_tensor(beta, device=counts.device), counts[present])
    weights[present] = (1.0 - beta) / effective.clamp_min(1e-12)
    weights[present] *= present.sum() / weights[present].sum().clamp_min(1e-12)
    return weights


def class_balanced_focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    class_counts: Mapping[int, int] | Sequence[int] | torch.Tensor,
    *,
    beta: float = 0.99,
    gamma: float = 1.0,
) -> torch.Tensor:
    """Class-balanced focal loss used for imbalanced incremental batches."""

    if logits.ndim != 2:
        raise ValueError("logits must have shape [batch, classes]")
    targets = targets.to(device=logits.device, dtype=torch.long)
    weights = effective_number_weights(
        class_counts,
        logits.shape[1],
        beta,
        device=logits.device,
        dtype=logits.dtype,
    )
    log_probs = F.log_softmax(logits, dim=1)
    log_pt = log_probs.gather(1, targets[:, None]).squeeze(1)
    pt = log_pt.detach().exp()
    alpha = weights.gather(0, targets)
    return (-(1.0 - pt).pow(gamma) * log_pt * alpha).mean()


def logit_distillation_loss(
    current_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    old_classes: Sequence[int],
    *,
    temperature: float = 2.0,
    sample_mask: torch.Tensor | None = None,
    confidence_threshold: float = 0.0,
) -> torch.Tensor:
    """KL distillation restricted to classes known by the frozen teacher."""

    if not old_classes:
        return current_logits.new_zeros(())
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    class_index = torch.as_tensor(old_classes, dtype=torch.long, device=current_logits.device)
    student = current_logits.index_select(1, class_index)
    teacher = teacher_logits.detach().index_select(1, class_index)

    keep = torch.ones(student.shape[0], dtype=torch.bool, device=student.device)
    if sample_mask is not None:
        keep &= sample_mask.to(device=student.device, dtype=torch.bool)
    if confidence_threshold > 0:
        confidence = F.softmax(teacher / temperature, dim=1).amax(dim=1)
        keep &= confidence >= confidence_threshold
    if not bool(keep.any()):
        return current_logits.new_zeros(())

    student = student[keep]
    teacher = teacher[keep]
    return F.kl_div(
        F.log_softmax(student / temperature, dim=1),
        F.softmax(teacher / temperature, dim=1),
        reduction="batchmean",
    ) * (temperature**2)


def old_new_margin_ranking_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    old_classes: Sequence[int],
    new_classes: Sequence[int],
    *,
    margin: float = 0.2,
    topk: int = 1,
    ordinal_aware: bool = False,
) -> torch.Tensor:
    """Keep an old sample's true logit above its strongest new-class logits.

    With ``ordinal_aware=True``, adjacent disease grades use a smaller margin
    than distant grades, avoiding excessive separation of clinically adjacent
    ordinal classes.
    """

    if not old_classes or not new_classes:
        return logits.new_zeros(())
    targets = targets.to(device=logits.device, dtype=torch.long)
    old_index = torch.as_tensor(old_classes, dtype=torch.long, device=logits.device)
    new_index = torch.as_tensor(new_classes, dtype=torch.long, device=logits.device)
    old_mask = (targets[:, None] == old_index[None, :]).any(dim=1)
    if not bool(old_mask.any()):
        return logits.new_zeros(())

    old_targets = targets[old_mask]
    true_scores = logits[old_mask].gather(1, old_targets[:, None])
    candidate_scores = logits[old_mask].index_select(1, new_index)
    k = min(int(topk), candidate_scores.shape[1])
    new_scores, relative_index = candidate_scores.topk(k, dim=1)

    margins = logits.new_full(new_scores.shape, float(margin))
    if ordinal_aware and logits.shape[1] > 1:
        compared_classes = new_index[relative_index]
        ordinal_distance = (old_targets[:, None] - compared_classes).abs().to(logits.dtype)
        ordinal_distance /= float(logits.shape[1] - 1)
        margins = margin * (0.5 + 0.5 * ordinal_distance)

    return F.relu(new_scores - true_scores.expand_as(new_scores) + margins).mean()
