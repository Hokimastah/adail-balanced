"""Regression tests for the ADAIL 0.5.0 fixes."""

import pytest
import torch
from torch import nn
from torch.utils.data import Dataset

import adail.trainer as trainer_module
from adail import ADAILConfig, ADAILNet, ADAILTrainer, old_to_new_confusion
from adail.benchmarking import extract_adail_metrics
from adail.data import ClassIncrementalScenario


class TinyBackbone(nn.Module):
    def __init__(self, feature_dim=8):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(12, feature_dim), nn.ReLU())

    def forward(self, images):
        return self.net(images)


class ToyImages(Dataset):
    """Class-conditional images; `noise` mimics a random training transform."""

    def __init__(self, per_class=6, num_classes=3, noise=0.0):
        generator = torch.Generator().manual_seed(3)
        self.base = []
        self.targets = []
        for class_id in range(num_classes):
            for _ in range(per_class):
                self.base.append(torch.randn(3, 2, 2, generator=generator) + class_id)
                self.targets.append(class_id)
        self.noise = noise

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        image = self.base[index]
        if self.noise:
            image = image + self.noise * torch.randn_like(image)
        return image, self.targets[index]


def build(**overrides):
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
    model = ADAILNet.from_config(config, backbone=TinyBackbone(), pretrained_backbone=False)
    return ADAILTrainer(model, config)


def test_config_rejects_unknown_new_options():
    with pytest.raises(ValueError):
        ADAILConfig(num_classes=3, ranking_space="scaled")
    with pytest.raises(ValueError):
        ADAILConfig(num_classes=3, memory_budget="per_class")


def test_cosine_logits_are_unscaled_similarities():
    trainer = build()
    outputs = trainer.model.forward_details(torch.randn(5, 3, 2, 2))
    scale = trainer.model.classifier.logit_scale.clamp(1e-3, 100.0)
    assert torch.allclose(outputs["logits"], scale * outputs["cosine_logits"])
    assert outputs["cosine_logits"].abs().max() <= 1.0 + 1e-6

    linear = build(classifier_type="linear")
    outputs = linear.model.forward_details(torch.randn(5, 3, 2, 2))
    assert torch.equal(outputs["logits"], outputs["cosine_logits"])


@pytest.mark.parametrize("space", ["cosine", "logit"])
def test_ranking_loss_receives_scores_in_the_configured_space(monkeypatch, space):
    seen = []
    original = trainer_module.old_new_margin_ranking_loss

    def spy(scores, *args, **kwargs):
        seen.append(scores.detach().abs().max().item())
        return original(scores, *args, **kwargs)

    monkeypatch.setattr(trainer_module, "old_new_margin_ranking_loss", spy)
    trainer = build(ranking_space=space, cosine_scale=10.0)
    dataset = ToyImages()
    trainer.fit_scenario(
        dataset, dataset, ClassIncrementalScenario.from_string(3, "2-1"),
        epochs_per_task=1, batch_size=4,
    )
    if space == "cosine":
        assert max(seen) <= 1.0 + 1e-6
    else:
        assert max(seen) > 1.0


def test_frozen_rows_ignore_momentum_and_weight_decay():
    trainer = build()
    weight = trainer.model.classifier.weight
    optimizer = torch.optim.SGD([weight], lr=0.1, momentum=0.9, weight_decay=0.1)
    weight.grad = torch.ones_like(weight)
    optimizer.step()  # builds a non-zero momentum buffer for every row
    before = weight.detach().clone()
    buffer_before = optimizer.state[weight]["momentum_buffer"].clone()

    weight.grad = torch.ones_like(weight)
    trainer._step_with_frozen_rows(optimizer, [2])

    assert torch.equal(weight[2], before[2])
    assert torch.equal(optimizer.state[weight]["momentum_buffer"][2], buffer_before[2])
    assert not torch.equal(weight[0], before[0])


def test_new_class_row_does_not_move_during_old_phase_epochs():
    class Recorder(ADAILTrainer):
        imprinted = {}

        def imprint_new_classes(self, dataset, class_ids, *args, **kwargs):
            super().imprint_new_classes(dataset, class_ids, *args, **kwargs)
            for class_id in class_ids:
                self.imprinted[class_id] = self.model.classifier.weight[class_id].detach().clone()

    config = ADAILConfig(
        num_classes=3, feature_dim=8, memory_size=6, dynamic_expansion=False,
        alternating_objectives=True, old_phase_frequency=1, memory_selection="random",
    )
    model = ADAILNet.from_config(config, backbone=TinyBackbone(), pretrained_backbone=False)
    trainer = Recorder(model, config)
    dataset = ToyImages()
    history = trainer.fit_scenario(
        dataset, dataset, ClassIncrementalScenario.from_string(3, "2-1"),
        epochs_per_task=2, batch_size=4,
    )
    assert [log["phase"] for log in history["task_results"][1]["epoch_logs"]] == ["old", "old"]
    assert torch.equal(trainer.model.classifier.weight[2], trainer.imprinted[2])


def test_per_seen_class_budget_matches_fixed_memory_per_class():
    dataset = ToyImages()
    scenario = ClassIncrementalScenario.from_string(3, "1-1")

    per_class = build(memory_budget="per_seen_class")
    history = per_class.fit_scenario(dataset, dataset, scenario, epochs_per_task=1, batch_size=4)
    sizes = [sum(task["memory_quotas"].values()) for task in history["task_results"]]
    assert sizes == [2, 4, 6]  # (6 // 3) exemplars per seen class

    global_budget = build()
    history = global_budget.fit_scenario(dataset, dataset, scenario, epochs_per_task=1, batch_size=4)
    sizes = [sum(task["memory_quotas"].values()) for task in history["task_results"]]
    assert sizes == [6, 6, 6]


def test_train_eval_dataset_drives_imprinting_and_prototypes():
    noisy, clean = ToyImages(noise=5.0), ToyImages()
    trainer = build()
    # A single base task: imprinting and the refresh both see every sample once.
    trainer.fit_scenario(
        noisy, clean, ClassIncrementalScenario.from_string(3, "3-1"),
        epochs_per_task=0, batch_size=64, train_eval_dataset=clean,
    )
    trainer.model.eval()
    with torch.no_grad():
        clean_class0 = torch.stack([clean[i][0] for i in range(6)])
        expected = trainer.model.forward_details(clean_class0)["features"].mean(dim=0)
    assert torch.allclose(trainer.model.prototypes.vectors[0], expected, atol=1e-5)

    with pytest.raises(ValueError):
        build().fit_scenario(
            noisy, clean, ClassIncrementalScenario.from_string(3, "1-1"),
            epochs_per_task=0, train_eval_dataset=ToyImages(per_class=2),
        )


def test_clean_view_stores_only_selected_exemplars_as_fixed_tensors(monkeypatch):
    collected = []
    original = trainer_module.ExemplarMemory.update_from_tensors

    def spy(self, images, *args, **kwargs):
        collected.append(images)
        return original(self, images, *args, **kwargs)

    monkeypatch.setattr(trainer_module.ExemplarMemory, "update_from_tensors", spy)
    noisy, clean = ToyImages(noise=5.0), ToyImages()
    trainer = build()
    trainer.fit_scenario(
        noisy, clean, ClassIncrementalScenario.from_string(3, "1-1"),
        epochs_per_task=1, batch_size=4, train_eval_dataset=clean,
    )
    assert collected and all(images is None for images in collected)  # no candidate images
    items = [item for values in trainer.memory.exemplars.values() for item in values]
    assert len(items) == 6
    for item in items:
        assert torch.equal(item.image, clean[item.source_index][0])  # fixed, clean tensor


def test_history_reports_sample_weighted_accuracy_and_new_class_bias():
    trainer = build()
    dataset = ToyImages()
    history = trainer.fit_scenario(
        dataset, dataset, ClassIncrementalScenario.from_string(3, "1-1"),
        epochs_per_task=1, batch_size=4,
    )
    seen = [task["metrics"]["seen_accuracy"] for task in history["task_results"]]
    assert history["average_incremental_accuracy"] == pytest.approx(sum(seen) / len(seen))
    assert history["task_results"][0]["metrics"]["old_to_new_confusion"] == 0.0
    biases = [task["metrics"]["old_to_new_confusion"] for task in history["task_results"][1:]]
    assert history["average_old_to_new_confusion"] == pytest.approx(sum(biases) / 2)
    assert all(
        task["metrics"]["seen_balanced_accuracy"] == task["metrics"]["seen_recall"]
        for task in history["task_results"]
    )


def test_old_to_new_confusion():
    cm = [[8, 2, 0], [1, 5, 4], [0, 0, 10]]
    assert old_to_new_confusion(cm, [0, 1], [2]) == pytest.approx(4 / 20)
    assert old_to_new_confusion(cm, [], [2]) == 0.0


def test_comparison_uses_sample_weighted_accuracy():
    history = {
        "average_accuracy": 0.6,
        "task_results": [
            {"metrics": {"seen_accuracy": 0.9}},
            {"metrics": {"seen_accuracy": 0.5}},
        ],
    }
    assert extract_adail_metrics(history, "ham10000", "3-1").average_accuracy == pytest.approx(70.0)
    history["average_incremental_accuracy"] = 0.8
    assert extract_adail_metrics(history, "ham10000", "3-1").average_accuracy == pytest.approx(80.0)
