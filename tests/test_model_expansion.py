import torch
from torch import nn

from adail import ADAILNet


class TinyBackbone(nn.Module):
    def forward(self, x):
        return x


def test_resize_classes_preserves_overlapping_classifier_and_prototypes():
    model = ADAILNet(
        num_classes=3,
        backbone=TinyBackbone(),
        feature_dim=4,
        pretrained_backbone=False,
    )
    with torch.no_grad():
        model.classifier.weight.copy_(torch.arange(12, dtype=torch.float32).reshape(3, 4))
        model.classifier.bias.copy_(torch.arange(3, dtype=torch.float32))
        model.prototypes.vectors.copy_(torch.arange(12, dtype=torch.float32).reshape(3, 4))
        model.prototypes.initialized[:] = True

    old_weight = model.classifier.weight.detach().clone()
    old_bias = model.classifier.bias.detach().clone()
    old_prototypes = model.prototypes.vectors.detach().clone()

    model.expand_classes(5)

    assert model.num_classes == 5
    assert torch.equal(model.classifier.weight[:3], old_weight)
    assert torch.equal(model.classifier.bias[:3], old_bias)
    assert torch.equal(model.prototypes.vectors[:3], old_prototypes)
    assert not model.prototypes.initialized[3:].any()

    model.resize_classes(2)

    assert model.num_classes == 2
    assert torch.equal(model.classifier.weight, old_weight[:2])
    assert torch.equal(model.classifier.bias, old_bias[:2])
    assert torch.equal(model.prototypes.vectors, old_prototypes[:2])
