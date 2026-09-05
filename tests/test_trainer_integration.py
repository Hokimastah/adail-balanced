import torch
from torch import nn
from torch.utils.data import TensorDataset

from adail import ADAILConfig, ADAILNet, ADAILTrainer
from adail.data import ClassIncrementalScenario
from adail.memory import Exemplar


class TinyBackbone(nn.Module):
    def __init__(self, feature_dim=8):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(12, feature_dim), nn.ReLU())

    def forward(self, images):
        return self.net(images)


def make_dataset():
    images = []
    labels = []
    generator = torch.Generator().manual_seed(11)
    for class_id in range(3):
        images.append(torch.randn(4, 3, 2, 2, generator=generator) + class_id)
        labels.append(torch.full((4,), class_id, dtype=torch.long))
    return TensorDataset(torch.cat(images), torch.cat(labels))


def build_trainer():
    config = ADAILConfig(
        num_classes=3,
        feature_dim=8,
        memory_size=6,
        dynamic_expansion=False,
        old_phase_frequency=2,
    )
    model = ADAILNet.from_config(
        config,
        backbone=TinyBackbone(feature_dim=8),
        pretrained_backbone=False,
    )
    return ADAILTrainer(model, config), config


def test_two_task_training_and_safe_checkpoint(tmp_path):
    trainer, _ = build_trainer()
    dataset = make_dataset()
    scenario = ClassIncrementalScenario.from_string(3, "2-1")
    history = trainer.fit_scenario(
        dataset,
        dataset,
        scenario,
        epochs_per_task=2,
        batch_size=4,
        num_workers=0,
    )
    assert len(history["task_results"]) == 2
    assert history["task_results"][1]["epoch_logs"][1]["phase"] == "old"
    assert len(trainer.memory) == 6

    checkpoint = tmp_path / "enhanced.pt"
    trainer.save_checkpoint(checkpoint)
    raw = torch.load(checkpoint, weights_only=True)
    assert raw["format_version"] == 2

    restored, _ = build_trainer()
    restored.load_checkpoint(checkpoint)
    assert len(restored.memory) == len(trainer.memory)
    for key, value in trainer.model.state_dict().items():
        assert torch.equal(value, restored.model.state_dict()[key])


def test_legacy_exemplar_checkpoint_is_allowlisted(tmp_path):
    trainer, _ = build_trainer()
    legacy = {
        "model": trainer.model.state_dict(),
        "memory": {
            "max_size": 6,
            "quotas": {0: 1},
            "exemplars": {0: [Exemplar(torch.zeros(3, 2, 2), 0, 1.0)]},
        },
        "best_class_scores": {},
        "class_forgetting": {},
        "observed_class_counts": {},
    }
    checkpoint = tmp_path / "legacy.pt"
    torch.save(legacy, checkpoint)
    trainer.load_checkpoint(checkpoint)
    assert len(trainer.memory) == 1
