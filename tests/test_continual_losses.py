import torch

from adail.modules import (
    class_balanced_focal_loss,
    effective_number_weights,
    logit_distillation_loss,
    old_new_margin_ranking_loss,
)


def test_effective_number_upweights_minority_class():
    weights = effective_number_weights({0: 100, 1: 10}, num_classes=2, beta=0.99)
    assert weights[1] > weights[0]
    assert torch.isclose(weights.mean(), torch.tensor(1.0), atol=1e-5)


def test_class_balanced_focal_is_finite_and_differentiable():
    logits = torch.randn(6, 3, requires_grad=True)
    labels = torch.tensor([0, 0, 0, 1, 1, 2])
    loss = class_balanced_focal_loss(logits, labels, {0: 30, 1: 10, 2: 2})
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None


def test_distillation_respects_memory_mask():
    student = torch.randn(5, 4, requires_grad=True)
    teacher = torch.randn(5, 4)
    memory_mask = torch.tensor([True, False, True, False, True])
    loss = logit_distillation_loss(
        student,
        teacher,
        old_classes=[0, 1, 2],
        sample_mask=memory_mask,
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert student.grad is not None
    assert torch.count_nonzero(student.grad[~memory_mask]) == 0


def test_margin_ranking_uses_shape_safe_targets():
    logits = torch.tensor(
        [[0.1, 0.7, 0.8, 0.5], [0.8, 0.2, 0.1, 0.0]],
        requires_grad=True,
    )
    labels = torch.tensor([1, 0])
    loss = old_new_margin_ranking_loss(
        logits,
        labels,
        old_classes=[0, 1],
        new_classes=[2, 3],
        margin=0.2,
        topk=2,
        ordinal_aware=True,
    )
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None
