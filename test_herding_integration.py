#!/usr/bin/env python3
"""Comprehensive test of herding memory integration with ADAIL trainer."""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, 'src')

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset

from adail import ADAILConfig, ADAILNet, ADAILTrainer
from adail.data import ClassIncrementalScenario


class TinyBackbone(nn.Module):
    def __init__(self, feature_dim: int = 16):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(3 * 16 * 16, feature_dim), nn.ReLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def make_dataset(num_classes: int = 5, n_per_class: int = 20):
    xs, ys = [], []
    for c in range(num_classes):
        xs.append(torch.randn(n_per_class, 3, 16, 16) + c * 0.2)
        ys.append(torch.full((n_per_class,), c, dtype=torch.long))
    return TensorDataset(torch.cat(xs), torch.cat(ys))


def test_trainer_with_cache_dir():
    """Test that trainer accepts cache_dir parameter."""
    print("\n" + "="*60)
    print("TEST 1: Trainer with custom cache_dir")
    print("="*60)
    
    cache_dir = Path(tempfile.mkdtemp(prefix="adail_test_"))
    print(f"Cache directory: {cache_dir}")
    
    cfg = ADAILConfig(num_classes=5, feature_dim=16, memory_size=50, use_ordinal=True)
    model = ADAILNet(
        num_classes=5,
        backbone=TinyBackbone(feature_dim=16),
        feature_dim=16,
        pretrained_backbone=False,
    )
    
    # Create trainer with cache_dir
    trainer = ADAILTrainer(model, cfg, memory_cache_dir=cache_dir)
    print(f"✓ Trainer created with cache_dir: {trainer.memory.cache_dir}")
    
    scenario = ClassIncrementalScenario.from_string(5, "2-1")
    dataset = make_dataset(5)
    
    # Train for one task
    history = trainer.fit_scenario(
        dataset, dataset, scenario,
        epochs_per_task=1, batch_size=16
    )
    
    print(f"✓ Training completed")
    print(f"  - Average accuracy: {history['average_accuracy']:.3f}")
    print(f"  - Memory size: {len(trainer.memory)}")
    print(f"  - Exemplars in cache: {len(list(cache_dir.glob('exemplar_*.jpg')))}")
    
    # Cleanup
    import shutil
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    print(f"✓ Cleanup completed")


def test_exemplar_serialization():
    """Test exemplar export/import functionality."""
    print("\n" + "="*60)
    print("TEST 2: Exemplar export/import")
    print("="*60)
    
    from adail.memory import ExemplarMemory
    
    cache_dir = Path(tempfile.mkdtemp(prefix="adail_cache_"))
    export_dir = Path(tempfile.mkdtemp(prefix="adail_export_"))
    import_cache_dir = Path(tempfile.mkdtemp(prefix="adail_import_"))
    
    print(f"Cache directory: {cache_dir}")
    print(f"Export directory: {export_dir}")
    
    # Create and populate memory
    memory1 = ExemplarMemory(max_size=100, cache_dir=cache_dir)
    images = torch.rand(10, 3, 16, 16)
    labels = torch.tensor([0, 1, 0, 1, 2, 0, 1, 2, 0, 1])
    logits = torch.randn(10, 3)
    
    memory1.update_from_tensors(images, labels, logits=logits)
    print(f"✓ Memory populated: {len(memory1)} exemplars")
    print(f"  - Classes: {memory1.classes()}")
    
    # Export exemplars
    memory1.export_exemplars(export_dir)
    print(f"✓ Exemplars exported to {export_dir}")
    print(f"  - Files: {len(list(export_dir.glob('exemplar_*.jpg')))}")
    print(f"  - Metadata: {(export_dir / 'exemplars_metadata.json').exists()}")
    
    # Import exemplars to new memory
    memory2 = ExemplarMemory(max_size=100, cache_dir=import_cache_dir)
    memory2.import_exemplars(export_dir)
    print(f"✓ Exemplars imported: {len(memory2)} exemplars")
    print(f"  - Classes: {memory2.classes()}")
    
    # Verify consistency
    assert len(memory1) == len(memory2), "Memory sizes don't match"
    assert memory1.classes() == memory2.classes(), "Classes don't match"
    print(f"✓ Import/export consistency verified")
    
    # Cleanup
    import shutil
    for d in [cache_dir, export_dir, import_cache_dir]:
        if d.exists():
            shutil.rmtree(d)
    print(f"✓ Cleanup completed")


def test_checkpoint_with_memory():
    """Test checkpoint save/load with memory metadata."""
    print("\n" + "="*60)
    print("TEST 3: Checkpoint save/load with memory")
    print("="*60)
    
    checkpoint_dir = Path(tempfile.mkdtemp(prefix="adail_ckpt_"))
    cache_dir = Path(tempfile.mkdtemp(prefix="adail_cache_"))
    checkpoint_file = checkpoint_dir / "checkpoint.pt"
    
    print(f"Cache directory: {cache_dir}")
    print(f"Checkpoint file: {checkpoint_file}")
    
    # Create trainer and train
    cfg = ADAILConfig(num_classes=5, feature_dim=16, memory_size=50)
    model = ADAILNet(
        num_classes=5,
        backbone=TinyBackbone(feature_dim=16),
        feature_dim=16,
        pretrained_backbone=False,
    )
    trainer1 = ADAILTrainer(model, cfg, memory_cache_dir=cache_dir)
    
    scenario = ClassIncrementalScenario.from_string(5, "2-1")
    dataset = make_dataset(5)
    
    # Train first task only
    history = trainer1.fit_scenario(
        dataset, dataset, scenario,
        epochs_per_task=1, batch_size=16
    )
    print(f"✓ Initial training completed: {len(trainer1.memory)} exemplars in memory")
    
    # Save checkpoint
    trainer1.save_checkpoint(checkpoint_file)
    print(f"✓ Checkpoint saved")
    
    # Load checkpoint into new trainer
    trainer2 = ADAILTrainer(model, cfg, memory_cache_dir=cache_dir)
    trainer2.load_checkpoint(checkpoint_file)
    print(f"✓ Checkpoint loaded")
    
    # Verify metadata restored
    assert trainer2.memory.quotas == trainer1.memory.quotas, "Quotas don't match"
    print(f"✓ Memory quotas restored: {trainer2.memory.quotas}")
    
    # Cleanup
    import shutil
    for d in [checkpoint_dir, cache_dir]:
        if d.exists():
            shutil.rmtree(d)
    print(f"✓ Cleanup completed")


def test_memory_dataset_compatibility():
    """Test that MemoryDataset is compatible with regular datasets."""
    print("\n" + "="*60)
    print("TEST 4: MemoryDataset compatibility")
    print("="*60)
    
    from adail.memory import ExemplarMemory
    from torch.utils.data import ConcatDataset, DataLoader
    
    cache_dir = Path(tempfile.mkdtemp(prefix="adail_cache_"))
    
    # Create exemplar memory
    memory = ExemplarMemory(max_size=50, cache_dir=cache_dir)
    images = torch.rand(6, 3, 16, 16)
    labels = torch.tensor([0, 1, 0, 1, 0, 1])
    logits = torch.randn(6, 3)
    
    memory.update_from_tensors(images, labels, logits=logits)
    print(f"✓ Memory created: {len(memory)} exemplars")
    
    # Create regular dataset
    regular_dataset = TensorDataset(torch.rand(10, 3, 16, 16), torch.randint(0, 2, (10,)))
    print(f"✓ Regular dataset created: {len(regular_dataset)} samples")
    
    # Combine into ConcatDataset
    memory_dataset = memory.as_dataset()
    combined = ConcatDataset([regular_dataset, memory_dataset])
    print(f"✓ ConcatDataset created: {len(combined)} total samples")
    
    # Create DataLoader and iterate
    loader = DataLoader(combined, batch_size=4)
    for batch_idx, (imgs, lbls) in enumerate(loader):
        print(f"  Batch {batch_idx}: images shape {imgs.shape}, labels shape {lbls.shape}")
        if batch_idx >= 2:
            break
    
    print(f"✓ DataLoader iteration successful")
    
    # Cleanup
    import shutil
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    print(f"✓ Cleanup completed")


if __name__ == "__main__":
    try:
        test_trainer_with_cache_dir()
        test_exemplar_serialization()
        test_checkpoint_with_memory()
        test_memory_dataset_compatibility()
        
        print("\n" + "="*60)
        print("✓ ALL TESTS PASSED")
        print("="*60)
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
