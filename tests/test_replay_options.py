import pytest
import torch
from torch import nn
from torch.utils.data import Dataset, TensorDataset

from adail import ADAILConfig, ADAILNet, ADAILTrainer
from adail.data import BalancedReplayBatchSampler, ClassIncrementalScenario, ReplayDataset
from adail.memory import Exemplar, ExemplarMemory


class TinyBackbone(nn.Module):
    def __init__(self, feature_dim=8):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(12, feature_dim), nn.ReLU())

    def forward(self, images):
        return self.net(images)


class NoisyDataset(Dataset):
    """Returns a fresh random perturbation on every access, like a train transform."""

    def __init__(self, per_class=6, num_classes=3):
        generator = torch.Generator().manual_seed(5)
        self.base = []
        self.targets = []
        for class_id in range(num_classes):
            for _ in range(per_class):
                self.base.append(torch.randn(3, 2, 2, generator=generator) + class_id)
                self.targets.append(class_id)
        self.calls = 0

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        self.calls += 1
        return self.base[index] + 0.01 * torch.randn(3, 2, 2), self.targets[index]


def build_trainer(**overrides):
    options = dict(
        num_classes=3,
        feature_dim=8,
        memory_size=6,
        dynamic_expansion=False,
        alternating_objectives=False,
        memory_selection="random",
    )
    options.update(overrides)
    config = ADAILConfig(**options)
    model = ADAILNet.from_config(
        config, backbone=TinyBackbone(feature_dim=8), pretrained_backbone=False
    )
    return ADAILTrainer(model, config)


def test_config_rejects_unknown_replay_modes():
    with pytest.raises(ValueError):
        ADAILConfig(num_classes=3, replay_fraction_mode="auto")
    with pytest.raises(ValueError):
        ADAILConfig(num_classes=3, replay_epoch_length="memory")


def test_proportional_fraction_gives_each_seen_class_equal_share():
    trainer = build_trainer(replay_fraction_mode="proportional")
    current = TensorDataset(torch.zeros(10, 3, 2, 2), torch.full((10,), 6))
    memory = TensorDataset(torch.zeros(12, 3, 2, 2), torch.arange(6).repeat(2))
    fraction = trainer._replay_memory_fraction(ReplayDataset(current, memory))
    assert fraction == pytest.approx(6 / 7)

    sampler = BalancedReplayBatchSampler(
        [6] * 10, list(range(6)) * 2, batch_size=32, memory_fraction=fraction
    )
    # 27 memory slots over 6 old classes (4.5 each) vs 5 slots for the new class,
    # instead of 16 vs 2.7 with the fixed 0.5 split.
    assert sampler.memory_per_batch == 27
    assert sampler.current_per_batch == 5


def test_fixed_fraction_is_unchanged_by_default():
    trainer = build_trainer()
    current = TensorDataset(torch.zeros(4, 3, 2, 2), torch.full((4,), 2))
    memory = TensorDataset(torch.zeros(4, 3, 2, 2), torch.tensor([0, 0, 1, 1]))
    assert trainer._replay_memory_fraction(ReplayDataset(current, memory)) == 0.5


def test_pool_epoch_length_counts_current_and_memory_at_full_batch():
    current, memory = [1] * 100, [0] * 50
    legacy = BalancedReplayBatchSampler(current, memory, batch_size=32, memory_fraction=0.75)
    pool = BalancedReplayBatchSampler(
        current, memory, batch_size=32, memory_fraction=0.75, epoch_length="pool"
    )
    assert len(legacy) == 13  # ceil(100 / 8 current slots)
    assert len(pool) == 5  # ceil(150 / 32)
    assert all(len(batch) == 32 for batch in pool)


def test_index_exemplars_are_reloaded_from_source_and_survive_state_dict():
    source = NoisyDataset()
    memory = ExemplarMemory(max_size=4)
    memory.source_dataset = source
    labels = torch.tensor(source.targets)
    memory.update_from_tensors(
        images=None,
        labels=labels,
        replace=True,
        selection="random",
        source_indices=list(range(len(labels))),
    )
    assert len(memory) == 4
    assert all(item.image is None and item.source_index is not None
               for items in memory.exemplars.values() for item in items)

    dataset = memory.as_dataset()
    first, label = dataset[0]
    second, _ = dataset[0]
    assert int(label) == source.targets[dataset.source_indices[0]]
    assert not torch.equal(first, second)  # re-augmented on every access

    restored = ExemplarMemory(max_size=4)
    restored.load_state_dict(memory.state_dict())
    assert restored.as_dataset().source_indices == dataset.source_indices
    with pytest.raises(RuntimeError):
        restored.as_dataset()[0]  # no source attached yet
    restored.source_dataset = source
    restored.as_dataset()[0]


def test_update_requires_image_or_source_index():
    memory = ExemplarMemory(max_size=2)
    with pytest.raises(ValueError):
        memory.update_from_tensors(images=None, labels=torch.tensor([0, 1]), replace=True)


def test_trainer_with_all_replay_options_keeps_index_exemplars(tmp_path):
    trainer = build_trainer(
        replay_fraction_mode="proportional",
        replay_epoch_length="pool",
        replay_reaugment=True,
    )
    dataset = NoisyDataset()
    scenario = ClassIncrementalScenario.from_string(3, "1-1")
    history = trainer.fit_scenario(
        dataset, dataset, scenario, epochs_per_task=1, batch_size=4, num_workers=0
    )
    assert len(history["task_results"]) == 3
    assert len(trainer.memory) == 6
    items = [item for values in trainer.memory.exemplars.values() for item in values]
    assert all(item.image is None for item in items)
    assert all(dataset.targets[item.source_index] == item.label for item in items)

    checkpoint = tmp_path / "reaugment.pt"
    trainer.save_checkpoint(checkpoint)
    torch.load(checkpoint, weights_only=True)


def test_reaugment_off_still_stores_tensors():
    trainer = build_trainer()
    dataset = NoisyDataset()
    scenario = ClassIncrementalScenario.from_string(3, "2-1")
    trainer.fit_scenario(dataset, dataset, scenario, epochs_per_task=1, batch_size=4)
    items = [item for values in trainer.memory.exemplars.values() for item in values]
    assert items and all(isinstance(item.image, torch.Tensor) for item in items)
    assert all(item.source_index is None for item in items)
    assert isinstance(items[0], Exemplar)
