import torch

from adail.modules import DriftAligner, PrototypeBank, drift_alignment_loss, ordinal_constraint_loss


def test_drift_loss_is_scalar():
    aligner = DriftAligner(4)
    current = torch.randn(8, 4)
    old = torch.randn(8, 4)
    loss = drift_alignment_loss(current, old, aligner)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_prototype_losses():
    bank = PrototypeBank(num_classes=3, feature_dim=4)
    features = torch.randn(12, 4, requires_grad=True)
    labels = torch.tensor([0, 1, 2] * 4)
    bank.update(features, labels)
    assert bank.compactness_loss(features, labels).ndim == 0
    assert bank.separation_loss().ndim == 0
    differentiable = bank.differentiable_separation_loss(features, labels, [0, 1, 2])
    assert differentiable.ndim == 0
    assert differentiable.requires_grad
    differentiable.backward()
    assert features.grad is not None


def test_ordinal_loss():
    logits = torch.randn(6, 5)
    labels = torch.tensor([0, 1, 2, 3, 4, 2])
    loss = ordinal_constraint_loss(logits, labels)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
