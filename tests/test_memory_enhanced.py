import torch

from adail.memory import Exemplar, ExemplarMemory


def _contains_exemplar(value):
    if isinstance(value, Exemplar):
        return True
    if isinstance(value, dict):
        return any(_contains_exemplar(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_exemplar(item) for item in value)
    return False


def test_hybrid_memory_fills_capacity_and_serializes_safely():
    memory = ExemplarMemory(max_size=6)
    images = torch.randn(10, 3, 4, 4)
    labels = torch.tensor([0, 0] + [1] * 8)
    logits = torch.randn(10, 2)
    features = torch.randn(10, 5)
    memory.update_from_tensors(
        images,
        labels,
        logits,
        features,
        class_counts={0: 2, 1: 8},
        replace=True,
        selection="hybrid",
    )
    assert len(memory) == 6
    assert sum(memory.quotas.values()) == 6
    assert not _contains_exemplar(memory.state_dict())

    restored = ExemplarMemory(max_size=1)
    restored.load_state_dict(memory.state_dict())
    assert len(restored) == len(memory)
    assert restored.quotas == memory.quotas
