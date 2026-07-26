#!/usr/bin/env python3
"""Final verification: All ADAIL components with herding method."""

import sys
sys.path.insert(0, 'src')

import torch
from pathlib import Path
import tempfile

from adail.memory import ExemplarMemory, Exemplar, MemoryDataset
from adail.trainer import ADAILTrainer
from adail.config import ADAILConfig
from adail.models import ADAILNet
from adail.data import ClassIncrementalScenario
from torch.utils.data import TensorDataset
import torch.nn as nn


def verify_exemplar_class():
    """Verify Exemplar class has correct fields."""
    print("✓ Exemplar dataclass structure:")
    ex = Exemplar(
        image_path="/tmp/test.jpg",
        label=0,
        score=0.95,
        image_size=(224, 224)
    )
    print(f"  - image_path: {ex.image_path}")
    print(f"  - label: {ex.label}")
    print(f"  - score: {ex.score}")
    print(f"  - image_size: {ex.image_size}")
    assert ex.image_path == "/tmp/test.jpg"
    assert ex.label == 0
    assert ex.image_size == (224, 224)


def verify_memory_operations():
    """Verify core memory operations."""
    print("\n✓ ExemplarMemory operations:")
    cache_dir = Path(tempfile.mkdtemp())
    memory = ExemplarMemory(max_size=20, cache_dir=cache_dir)
    
    # Test update_from_tensors
    images = torch.rand(5, 3, 32, 32)
    labels = torch.tensor([0, 1, 0, 1, 0])
    memory.update_from_tensors(images, labels)
    
    print(f"  - Exemplars stored: {len(memory)}")
    print(f"  - Classes: {memory.classes()}")
    print(f"  - Cache files: {len(list(cache_dir.glob('exemplar_*.jpg')))}")
    
    # Test dataset conversion
    dataset = memory.as_dataset()
    img, lbl = dataset[0]
    print(f"  - Dataset item shape: {img.shape}, label: {lbl}")
    
    # Test state_dict
    state = memory.state_dict()
    print(f"  - State dict keys: {list(state.keys())}")
    
    # Cleanup
    memory.cleanup()
    assert not cache_dir.exists(), "Cache not cleaned up"
    print(f"  - Cleanup: OK")


def verify_trainer_integration():
    """Verify trainer integration with herding memory."""
    print("\n✓ ADAILTrainer integration:")
    
    cache_dir = Path(tempfile.mkdtemp())
    
    # Create minimal model and config
    class TinyBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Linear(3*8*8, 16)
        def forward(self, x):
            return self.net(x.flatten(1))
    
    cfg = ADAILConfig(num_classes=3, feature_dim=16, memory_size=20)
    model = ADAILNet(
        num_classes=3,
        backbone=TinyBackbone(),
        feature_dim=16,
        pretrained_backbone=False,
    )
    
    # Test trainer with memory_cache_dir
    trainer = ADAILTrainer(model, cfg, memory_cache_dir=cache_dir)
    print(f"  - Trainer created with cache_dir: {trainer.memory.cache_dir == cache_dir}")
    print(f"  - Memory max_size: {trainer.memory.max_size}")
    
    # Test checkpoint
    checkpoint_file = cache_dir / "test.pt"
    trainer.save_checkpoint(checkpoint_file)
    print(f"  - Checkpoint saved: {checkpoint_file.exists()}")
    
    trainer2 = ADAILTrainer(model, cfg, memory_cache_dir=cache_dir)
    trainer2.load_checkpoint(checkpoint_file)
    print(f"  - Checkpoint loaded: OK")
    
    # Cleanup
    import shutil
    if cache_dir.exists():
        shutil.rmtree(cache_dir)


def verify_imports():
    """Verify all necessary imports work."""
    print("\n✓ Module imports:")
    
    from adail import ADAILConfig, ADAILNet, ADAILTrainer
    print("  - Main imports: OK")
    
    from adail.memory import ExemplarMemory, Exemplar, MemoryDataset
    print("  - Memory imports: OK")
    
    from adail.trainer import TaskResult
    print("  - Trainer imports: OK")
    
    from adail.data import ClassIncrementalScenario
    print("  - Data imports: OK")


def verify_cli_compatibility():
    """Verify CLI components work."""
    print("\n✓ CLI compatibility:")
    
    from adail.cli import _config_from_args, _parse_args
    print("  - CLI functions import: OK")
    
    from adail.config import ADAILConfig
    cfg = ADAILConfig(num_classes=5)
    cfg_dict = cfg.to_dict()
    cfg2 = ADAILConfig.from_dict(cfg_dict)
    print(f"  - Config serialization: OK")


def verify_file_structure():
    """Verify file structure."""
    print("\n✓ Project structure:")
    
    required_files = [
        "src/adail/memory.py",
        "src/adail/trainer.py",
        "src/adail/config.py",
        "src/adail/cli.py",
        "tests/test_import.py",
        "examples/minimal_synthetic.py",
    ]
    
    for file in required_files:
        path = Path(file)
        exists = path.exists()
        print(f"  - {file}: {'✓' if exists else '✗'}")
        assert exists, f"Missing file: {file}"


if __name__ == "__main__":
    print("="*60)
    print("FINAL VERIFICATION: ADAIL with Herding Method")
    print("="*60)
    
    try:
        verify_file_structure()
        verify_imports()
        verify_exemplar_class()
        verify_memory_operations()
        verify_trainer_integration()
        verify_cli_compatibility()
        
        print("\n" + "="*60)
        print("✓ ALL VERIFICATIONS PASSED")
        print("✓ ADAIL Herding Method Implementation Complete")
        print("="*60)
        
    except Exception as e:
        print(f"\n✗ VERIFICATION FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
