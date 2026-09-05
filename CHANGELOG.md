# Changelog

## 0.4.0 — 2026-08-19

### Added

- Selectively expandable ResNet-18 high-level branches with gated residual fusion.
- Alternating new/stability training phases.
- Cosine classifier, learned scale, and prototype weight imprinting.
- Class-balanced focal, logit distillation, auxiliary, and old/new margin ranking losses.
- Class-balanced replay batch sampler with explicit memory flags.
- Hybrid herding, uncertainty, and diversity exemplar selection.
- Per-epoch phase/loss logs and representation branch count in training history.
- `ADAILNet.from_config` to keep YAML and architecture construction synchronized.
- Unit and integration coverage for the enhanced components.

### Fixed

- Mixed `int`/tensor/array labels now collate consistently as `torch.long`.
- Prototype separation now backpropagates through live batch features.
- Text progress bars no longer require Jupyter widget models.
- Epoch and phase numbers are printed explicitly.
- New checkpoints are compatible with PyTorch 2.6 `weights_only=True` loading.
- Adaptive memory redistributes quota that cannot be filled by a sparse class.

### Compatibility

- Existing 0.3.0 YAML files remain valid through default values.
- The checkpoint loader accepts the legacy ADAIL `Exemplar` safe global.

