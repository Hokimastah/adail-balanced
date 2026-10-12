# Changelog

## 0.5.0 — 2026-10-13

### Fixed

- The old-vs-new margin ranking loss now compares raw cosine similarities
  (`ranking_space="cosine"`, the new default) like LO2LN. 0.4.0 applied the margin
  to logits already multiplied by the cosine scale (about 10), so a margin of 0.2
  was effectively 0.02. `ranking_space="logit"` restores the old behaviour.
- During the `old` phase the new-class classifier rows now really stay put: their
  values and momentum are restored after `optimizer.step()`. Zeroing their
  gradient, as 0.4.0 did, still let SGD momentum and weight decay move them.
- `history["average_accuracy"]` is task-weighted and is not the quantity LO2LN
  reports. The new `history["average_incremental_accuracy"]` is sample-weighted,
  and `adail compare-results` now uses it (rebuilt from `seen_accuracy` for 0.4.0
  histories).
- `ADAILNet.resize_classes` exists again (grow or shrink classifier and prototype
  bank); `expand_classes` uses it. Its test no longer assumes a classifier bias.
- Memory updates kept every candidate image in RAM (current task + memory; on
  HAM10000 about 3.6 GB per run when the nv class arrives). With
  `replay_reaugment` or `train_eval_dataset`, candidates are now referenced by
  dataset index and only the selected exemplars are loaded.
- Import ordering flagged by ruff in `metrics.py` and `trainer.py`.

### Added

- `replay_fraction_mode="proportional"`: balanced replay gives memory `n_old / n_seen`
  of each batch, so every seen class gets about the same number of slots instead of
  the new class always taking half.
- `replay_epoch_length="pool"`: an epoch is one pass over current data plus exemplars
  at the full batch size, instead of growing with the size of the newest class.
- `replay_reaugment=True`: exemplars are stored as indices into the training dataset
  and reloaded through its random training transform on every replay. Candidate
  images are then no longer held in RAM during memory updates. After loading a
  checkpoint, set `trainer.memory.source_dataset` before replaying.
- `memory_budget="per_seen_class"`: keep `(memory_size // num_classes) * n_seen`
  exemplars, the budget of LO2LN/iCaRL with fixed memory per class.
- `fit_scenario(..., train_eval_dataset=...)`: the training samples under the
  evaluation transform, used for weight imprinting, the prototype refresh and
  exemplar scoring instead of randomly augmented views.
- Per-task `seen_balanced_accuracy` and `old_to_new_confusion` (share of old-class
  test samples predicted as the task's new classes), and
  `history["average_old_to_new_confusion"]`.
- `ADAILNet.forward_details` returns `cosine_logits` (unscaled similarities).

### Removed

- `HERDING_METHOD_SUMMARY.md`, `INTEGRATION_AUDIT_COMPLETE.md`,
  `KAGGLE_DEPLOYMENT_GUIDE.md`, `test_herding_integration.py`,
  `test_herding_memory.py` and `verify_herding_implementation.py`. They described
  or exercised an on-disk exemplar cache (`cache_dir`, `memory_cache_dir`,
  `export_exemplars`, `cleanup`) that is no longer part of ADAIL and could not run.
- `update.md`, whose content is now `README.md`.

### Compatibility

- All new options except `ranking_space` default to the 0.4.0 behaviour. The A2
  arm (no ranking loss) stays bit-identical to ADAIL-A2
  (`ADAIL-A2/scripts/verify_equivalence.py`).

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

