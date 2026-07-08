from __future__ import annotations

import torch
from torch.nn import functional as F


def ordinal_expected_value(logits: torch.Tensor) -> torch.Tensor:
    """Return expected class index from logits."""

    probs = F.softmax(logits, dim=1)
    class_positions = torch.arange(logits.shape[1], device=logits.device, dtype=probs.dtype)
    return probs @ class_positions


def ordinal_constraint_loss(logits: torch.Tensor, labels: torch.Tensor, normalize: bool = True) -> torch.Tensor:
    """MAE-based ordinal consistency loss.

    This is most useful when class labels are ordered by disease severity, e.g.
    EyePACS: No DR -> Mild -> Moderate -> Severe -> Proliferative.
    """

    pred = ordinal_expected_value(logits)
    target = labels.to(pred.dtype)
    loss = torch.abs(pred - target).mean()
    if normalize and logits.shape[1] > 1:
        loss = loss / float(logits.shape[1] - 1)
    return loss
