import torch
from torch.utils.data import Dataset

from adail.data import BalancedReplayBatchSampler, ReplayDataset


class MixedLabelDataset(Dataset):
    def __init__(self, labels, tensor_labels=False):
        self.labels = labels
        self.tensor_labels = tensor_labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        label = self.labels[index]
        if self.tensor_labels:
            label = torch.tensor([label])
        return torch.full((2,), float(index)), label


def test_replay_dataset_normalizes_mixed_label_types():
    replay = ReplayDataset(
        MixedLabelDataset([2, 2, 3]),
        MixedLabelDataset([0, 1], tensor_labels=True),
    )
    assert replay.targets == [2, 2, 3, 0, 1]
    for index in range(len(replay)):
        _, label, memory_flag = replay[index]
        assert label.dtype == torch.long
        assert label.ndim == 0
        assert memory_flag.dtype == torch.bool


def test_balanced_sampler_keeps_memory_fraction_and_classes():
    sampler = BalancedReplayBatchSampler(
        current_labels=[2, 2, 2, 3, 3, 3],
        memory_labels=[0, 0, 1, 1],
        batch_size=4,
        memory_fraction=0.5,
        seed=7,
    )
    batches = list(iter(sampler))
    assert batches
    for batch in batches:
        assert len(batch) == 4
        assert sum(index >= 6 for index in batch) == 2
        assert sum(index < 6 for index in batch) == 2
