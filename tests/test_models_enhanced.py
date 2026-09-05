import torch

from adail.models import CosineClassifier, SelectiveExpandableResNet18


def test_cosine_classifier_imprinting():
    classifier = CosineClassifier(4, 3, scale=5.0, learnable_scale=False)
    vector = torch.tensor([[0.0, 3.0, 4.0, 0.0]])
    classifier.imprint([2], vector)
    expected = torch.tensor([0.0, 0.6, 0.8, 0.0])
    assert torch.allclose(classifier.weight[2], expected)
    logits = classifier(torch.randn(2, 4))
    assert logits.shape == (2, 3)


def test_selective_resnet_expands_with_fixed_feature_dimension():
    backbone = SelectiveExpandableResNet18(pretrained=False)
    x = torch.randn(2, 3, 64, 64)
    before = backbone(x)
    backbone.expand()
    after, latest = backbone.forward_details(x)
    assert before.shape == after.shape == latest.shape == (2, 512)
    assert backbone.num_branches == 2
    assert not any(parameter.requires_grad for parameter in backbone.branches[0].parameters())
    assert all(parameter.requires_grad for parameter in backbone.branches[1].parameters())
