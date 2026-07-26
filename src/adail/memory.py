from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import tempfile
import shutil

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


@dataclass
class Exemplar:
    image_path: str  # Path to stored image file (herding method)
    label: int
    score: float = 0.0
    image_size: tuple[int, int] = (224, 224)  # (height, width) - used to restore original dimensions


class MemoryDataset(Dataset):
    """A dataset view over stored exemplars.
    
    Uses the herding method by storing image paths instead of full tensors,
    reducing memory footprint during class-incremental learning.
    
    Images are restored to their original size when loaded.
    """

    def __init__(self, exemplars: list[Exemplar], transform=None) -> None:
        self.exemplars = exemplars
        self.transform = transform

    def __len__(self) -> int:
        return len(self.exemplars)

    def __getitem__(self, index: int):
        item = self.exemplars[index]
        # Load image from path on-demand
        image = Image.open(item.image_path).convert("RGB")
        
        # Restore original dimensions
        if hasattr(item, 'image_size') and item.image_size:
            image = transforms.Resize(item.image_size)(image)
        
        # Apply custom transform if provided
        if self.transform is not None:
            image = self.transform(image)
        else:
            # Default: convert to tensor only (no normalization or resizing)
            image = transforms.ToTensor()(image)
        
        return image, torch.tensor(item.label, dtype=torch.long)


class ExemplarMemory:
    """Adaptive exemplar memory for class-incremental learning using herding method.

    The memory quota of each class is updated from three signals:
    uncertainty, forgetting rate, and class rarity.
    
    Images are stored as files on disk rather than in tensors to reduce RAM usage
    (herding method for coreset selection).
    """

    def __init__(
        self,
        max_size: int,
        alpha_uncertainty: float = 1.0,
        beta_forgetting: float = 1.0,
        gamma_rarity: float = 1.0,
        cache_dir: str | Path | None = None,
    ) -> None:
        if max_size < 0:
            raise ValueError("max_size must be non-negative")
        self.max_size = max_size
        self.alpha_uncertainty = alpha_uncertainty
        self.beta_forgetting = beta_forgetting
        self.gamma_rarity = gamma_rarity
        self.exemplars: dict[int, list[Exemplar]] = defaultdict(list)
        self.quotas: dict[int, int] = {}
        
        # Setup cache directory for storing exemplar images
        if cache_dir is None:
            self.cache_dir = Path(tempfile.mkdtemp(prefix="adail_memory_"))
        else:
            self.cache_dir = Path(cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self._exemplar_counter = 0

    def __len__(self) -> int:
        return sum(len(items) for items in self.exemplars.values())

    def classes(self) -> list[int]:
        return sorted(self.exemplars.keys())

    def as_dataset(self) -> MemoryDataset:
        flat = [item for cls in self.classes() for item in self.exemplars[cls]]
        return MemoryDataset(flat)

    def _save_exemplar_image(self, image_tensor: torch.Tensor) -> tuple[str, tuple[int, int]]:
        """Save an exemplar image tensor to disk and return the path and size.
        
        Args:
            image_tensor: Image tensor on CPU with shape (C, H, W) and values in [0, 1]
            
        Returns:
            Tuple of (path to saved image file, (height, width) of original image)
        """
        # Ensure image is on CPU and detached
        image = image_tensor.detach().cpu()
        
        # Store original dimensions
        if image.dim() == 3:
            _, height, width = image.shape
            original_size = (height, width)
        else:
            original_size = (224, 224)
        
        # Convert tensor to PIL Image
        # Tensor is expected to be in [0, 1] range with shape (C, H, W)
        if image.dim() == 3 and image.shape[0] == 3:
            # Denormalize if needed (assume normalized with ImageNet stats)
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            image = image * std + mean
        
        # Clamp to valid range and convert to uint8
        image = torch.clamp(image, 0, 1)
        image = (image * 255).permute(1, 2, 0).numpy().astype(np.uint8)
        
        # Save to file
        pil_image = Image.fromarray(image)
        exemplar_id = self._exemplar_counter
        self._exemplar_counter += 1
        file_path = self.cache_dir / f"exemplar_{exemplar_id:06d}.jpg"
        pil_image.save(str(file_path), quality=95)
        
        return str(file_path), original_size

    def state_dict(self) -> dict[str, object]:
        """Save state. Note: exemplar paths are not saved; only metadata.
        
        For persistent storage, exemplar images should be saved separately.
        This method primarily saves quotas for reproducibility.
        """
        return {
            "max_size": self.max_size,
            "quotas": self.quotas,
            # exemplars are not saved as they're stored on disk
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        """Load state from dict.
        
        Note: exemplar images must be loaded separately as they're stored on disk.
        """
        self.max_size = int(state["max_size"])
        self.quotas = {int(k): int(v) for k, v in dict(state.get("quotas", {})).items()}

    @staticmethod
    def uncertainty_from_logits(logits: torch.Tensor) -> torch.Tensor:
        """Entropy-based uncertainty normalized to approximately [0, 1]."""

        probs = F.softmax(logits, dim=1)
        entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=1)
        max_entropy = float(np.log(max(2, logits.shape[1])))
        return entropy / max_entropy

    def compute_quotas(
        self,
        class_ids: list[int],
        uncertainty: dict[int, float] | None = None,
        forgetting: dict[int, float] | None = None,
        class_counts: dict[int, int] | None = None,
    ) -> dict[int, int]:
        """Compute adaptive per-class quotas.

        Missing uncertainty/forgetting values default to zero. Missing counts default
        to one to avoid division by zero.
        """

        if self.max_size == 0 or not class_ids:
            return {int(c): 0 for c in class_ids}

        uncertainty = uncertainty or {}
        forgetting = forgetting or {}
        class_counts = class_counts or {}
        counts = np.array([max(1, class_counts.get(int(c), 1)) for c in class_ids], dtype=float)
        rarity_values = counts.max() / counts
        rarity_values = rarity_values / max(rarity_values.max(), 1e-12)

        scores = []
        for i, c in enumerate(class_ids):
            c = int(c)
            score = (
                self.alpha_uncertainty * max(0.0, float(uncertainty.get(c, 0.0)))
                + self.beta_forgetting * max(0.0, float(forgetting.get(c, 0.0)))
                + self.gamma_rarity * float(rarity_values[i])
            )
            scores.append(max(score, 1e-6))
        scores_arr = np.array(scores, dtype=float)
        quotas_arr = np.floor(self.max_size * scores_arr / scores_arr.sum()).astype(int)

        # Distribute remainder to the largest fractional parts.
        remainder = self.max_size - int(quotas_arr.sum())
        fractions = (self.max_size * scores_arr / scores_arr.sum()) - quotas_arr
        order = np.argsort(-fractions)
        for idx in order[:remainder]:
            quotas_arr[idx] += 1
        return {int(c): int(q) for c, q in zip(class_ids, quotas_arr, strict=False)}

    def prune_to_quotas(self) -> None:
        for cls, quota in self.quotas.items():
            items = self.exemplars.get(cls, [])
            if len(items) > quota:
                # Keep top-scoring exemplars, delete the rest from disk
                sorted_items = sorted(items, key=lambda x: x.score, reverse=True)[:quota]
                deleted_items = sorted(items, key=lambda x: x.score, reverse=True)[quota:]
                
                # Remove deleted exemplars from disk
                for item in deleted_items:
                    try:
                        Path(item.image_path).unlink()
                    except (FileNotFoundError, OSError):
                        pass  # File may have already been deleted
                
                self.exemplars[cls] = sorted_items

    def cleanup(self) -> None:
        """Clean up temporary cache directory."""
        if self.cache_dir and self.cache_dir.exists():
            try:
                shutil.rmtree(self.cache_dir)
            except (OSError, PermissionError):
                pass  # Best effort cleanup

    def __del__(self) -> None:
        """Cleanup when object is destroyed."""
        self.cleanup()

    def export_exemplars(self, export_dir: str | Path) -> None:
        """Export exemplar metadata and images to a directory for persistent storage.
        
        Args:
            export_dir: Directory to export exemplars to. Images are copied, metadata is saved.
        """
        import json
        
        export_dir = Path(export_dir)
        export_dir.mkdir(parents=True, exist_ok=True)
        
        # Save exemplar metadata
        metadata = {
            "max_size": self.max_size,
            "quotas": self.quotas,
            "exemplars": {}
        }
        
        # Copy exemplar images and update paths
        for cls, exemplars in self.exemplars.items():
            metadata["exemplars"][str(cls)] = []
            for exemplar in exemplars:
                # Copy image to export directory
                src_path = Path(exemplar.image_path)
                filename = src_path.name
                dst_path = export_dir / filename
                shutil.copy2(str(src_path), str(dst_path))
                
                # Store metadata
                image_size = getattr(exemplar, 'image_size', (224, 224))
                metadata["exemplars"][str(cls)].append({
                    "image_file": filename,
                    "label": exemplar.label,
                    "score": exemplar.score,
                    "image_size": list(image_size),
                })
        
        # Save metadata as JSON
        with (export_dir / "exemplars_metadata.json").open("w") as f:
            json.dump(metadata, f, indent=2)

    def import_exemplars(self, import_dir: str | Path) -> None:
        """Import exemplar metadata and images from a directory.
        
        Args:
            import_dir: Directory to import exemplars from (must contain exemplars_metadata.json).
        """
        import json
        
        import_dir = Path(import_dir)
        metadata_file = import_dir / "exemplars_metadata.json"
        
        if not metadata_file.exists():
            raise FileNotFoundError(f"Exemplar metadata not found: {metadata_file}")
        
        with metadata_file.open("r") as f:
            metadata = json.load(f)
        
        self.max_size = metadata["max_size"]
        self.quotas = {int(k): v for k, v in metadata.get("quotas", {}).items()}
        
        # Copy exemplar images to cache directory and restore exemplars
        self.exemplars.clear()
        for cls_str, exemplars_list in metadata.get("exemplars", {}).items():
            cls = int(cls_str)
            self.exemplars[cls] = []
            
            for item in exemplars_list:
                src_path = import_dir / item["image_file"]
                if not src_path.exists():
                    raise FileNotFoundError(f"Exemplar image not found: {src_path}")
                
                # Copy to cache directory
                dst_path = self.cache_dir / src_path.name
                shutil.copy2(str(src_path), str(dst_path))
                
                # Restore exemplar with original metadata
                image_size = tuple(item.get("image_size", [224, 224]))
                exemplar = Exemplar(
                    image_path=str(dst_path),
                    label=item["label"],
                    score=item["score"],
                    image_size=image_size,
                )
                self.exemplars[cls].append(exemplar)

    def update_from_tensors(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        logits: torch.Tensor | None = None,
        features: torch.Tensor | None = None,
        class_counts: dict[int, int] | None = None,
        forgetting: dict[int, float] | None = None,
    ) -> None:
        """Add candidates and adaptively prune memory using herding method.

        Images are saved to disk to reduce RAM usage instead of storing tensors.
        Candidate score combines uncertainty and closeness to class centroid when
        features are provided.
        """

        if self.max_size == 0:
            return
        labels_cpu = labels.detach().cpu()
        class_ids = sorted(set(self.classes()) | {int(x) for x in labels_cpu.tolist()})

        uncertainty_values: dict[int, float] = {}
        sample_uncertainty: torch.Tensor | None = None
        if logits is not None:
            sample_uncertainty = self.uncertainty_from_logits(logits.detach()).cpu()
            for cls in class_ids:
                mask = labels_cpu == cls
                if mask.any():
                    uncertainty_values[cls] = float(sample_uncertainty[mask].mean())

        self.quotas = self.compute_quotas(
            class_ids=class_ids,
            uncertainty=uncertainty_values,
            forgetting=forgetting,
            class_counts=class_counts,
        )

        # Feature centroid closeness provides a simple herding-like signal.
        closeness = torch.zeros(len(labels_cpu), dtype=torch.float32)
        if features is not None:
            f_cpu = features.detach().cpu()
            for cls in class_ids:
                mask = labels_cpu == cls
                if mask.any():
                    centroid = f_cpu[mask].mean(dim=0, keepdim=True)
                    distance = torch.cdist(f_cpu[mask], centroid).flatten()
                    closeness[mask] = 1.0 / (1.0 + distance)

        for i, label in enumerate(labels_cpu.tolist()):
            u = float(sample_uncertainty[i]) if sample_uncertainty is not None else 0.0
            score = u + float(closeness[i])
            # Save image to disk instead of storing tensor (herding method)
            image_path, image_size = self._save_exemplar_image(images[i])
            self.exemplars[int(label)].append(Exemplar(image_path, int(label), score, image_size))

        self.prune_to_quotas()
