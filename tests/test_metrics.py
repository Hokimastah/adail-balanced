import numpy as np
import torch

from adail.metrics import accuracy, average_forgetting, quadratic_weighted_kappa


def test_accuracy():
    logits = torch.tensor([[1.0, 0.0], [0.1, 0.9]])
    labels = torch.tensor([0, 1])
    assert accuracy(logits, labels) == 1.0


def test_qwk_perfect():
    y = np.array([0, 1, 2, 3])
    assert quadratic_weighted_kappa(y, y, num_classes=4) == 1.0


def test_average_forgetting():
    matrix = [[0.8], [0.7, 0.9], [0.6, 0.85, 0.8]]
    assert average_forgetting(matrix) > 0
