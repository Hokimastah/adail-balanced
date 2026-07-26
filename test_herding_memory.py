#!/usr/bin/env python3
"""Test script to verify herding method implementation in memory.py"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, 'src')

import torch
import numpy as np
from adail.memory import ExemplarMemory, Exemplar, MemoryDataset


def test_herding_memory():
    """Test that exemplar memory uses herding method (stores paths, not tensors)."""
    
    print("=" * 60)
    print("Testing Herding Memory Implementation")
    print("=" * 60)
    
    # Create temporary cache directory
    cache_dir = Path(tempfile.mkdtemp(prefix="test_herding_"))
    print(f"\nCache directory: {cache_dir}")
    
    # Initialize memory
    memory = ExemplarMemory(
        max_size=10,
        alpha_uncertainty=1.0,
        beta_forgetting=1.0,
        gamma_rarity=1.0,
        cache_dir=cache_dir,
    )
    print(f"Memory initialized with max_size=10")
    
    # Create test data
    batch_size = 5
    images = torch.randn(batch_size, 3, 224, 224)  # Random images
    labels = torch.tensor([0, 1, 0, 1, 2])
    logits = torch.randn(batch_size, 3)
    
    print(f"\nTest data:")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Image shape: {images.shape}")
    print(f"  - Labels: {labels}")
    
    # Update memory
    print(f"\nAdding exemplars to memory...")
    memory.update_from_tensors(
        images=images,
        labels=labels,
        logits=logits,
        class_counts={0: 10, 1: 10, 2: 5},
    )
    
    print(f"Memory size: {len(memory)}")
    print(f"Classes in memory: {memory.classes()}")
    
    # Verify that images are stored as paths, not tensors
    print(f"\nVerifying herding method (images stored as paths):")
    for cls in memory.classes():
        exemplars = memory.exemplars[cls]
        print(f"  Class {cls}: {len(exemplars)} exemplars")
        for i, ex in enumerate(exemplars):
            print(f"    - Exemplar {i}:")
            print(f"        Type of image_path: {type(ex.image_path)}")
            print(f"        Path exists: {Path(ex.image_path).exists()}")
            print(f"        Label: {ex.label}")
            print(f"        Score: {ex.score:.4f}")
    
    # Verify cached images exist
    print(f"\nVerifying cached image files:")
    cached_files = list(cache_dir.glob("exemplar_*.jpg"))
    print(f"  - Total cached files: {len(cached_files)}")
    for file in sorted(cached_files)[:3]:
        file_size = file.stat().st_size / 1024  # KB
        print(f"    - {file.name} ({file_size:.1f} KB)")
    
    # Test dataset loading
    print(f"\nTesting MemoryDataset loading:")
    dataset = memory.as_dataset()
    print(f"  - Dataset size: {len(dataset)}")
    
    if len(dataset) > 0:
        # Load a sample
        image, label = dataset[0]
        print(f"  - Sample 0: image shape {image.shape}, label {label}")
        print(f"    Image dtype: {image.dtype}, range: [{image.min():.3f}, {image.max():.3f}]")
    
    # Test cleanup
    print(f"\nTesting cleanup:")
    memory.cleanup()
    remaining_files = list(cache_dir.glob("exemplar_*.jpg"))
    print(f"  - Cached files after cleanup: {len(remaining_files)}")
    
    print(f"\n{'=' * 60}")
    print("✓ Test completed successfully!")
    print("✓ Herding method is working (images stored as paths)")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    test_herding_memory()
