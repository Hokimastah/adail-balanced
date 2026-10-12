# ADAIL Implementation Map

Dokumen ini memetakan proposal ADAIL dan peningkatan versi 0.4.0–0.5.0 ke file repository.

| Konsep | Lokasi implementasi | Status 0.5.0 |
|---|---|---|
| ResNet-18 pretrained ImageNet | `src/adail/models/backbone.py` | Backbone standar dan expandable |
| Selective high-level expansion | `src/adail/models/backbone.py` | Shared `layer1`–`layer3`, cabang `layer4`, gated fusion |
| Cosine classifier + imprinting | `src/adail/models/adail_model.py` | Learned scale dan centroid initialization |
| Frozen previous model | `src/adail/models/adail_model.py` | Snapshot teacher antar-task tanpa masuk checkpoint |
| Skenario Class-Incremental Learning | `src/adail/data/datasets.py` | `4-1`, `4-2`, `3-1`, `3-2`, custom order |
| Dataset CCH5000, HAM10000, EyePACS | `src/adail/data/presets.py` | Metadata preset + default scenarios |
| Mixed-label normalization | `src/adail/data/replay.py` | Label scalar `torch.long` untuk collate aman |
| Balanced replay | `src/adail/data/replay.py`, `src/adail/trainer.py` | Porsi memori tetap atau proporsional (`n_lama/n_terlihat`), epoch `current` atau `pool` |
| Re-augmentasi exemplar | `src/adail/memory.py`, `src/adail/trainer.py` | Exemplar sebagai indeks dataset (`replay_reaugment`) |
| View evaluasi data latih | `src/adail/trainer.py` | `train_eval_dataset` untuk imprinting, refresh prototipe, skor exemplar |
| Drift-Aware Feature Alignment | `src/adail/modules/drift.py` | Learnable `A_t` + normalized MSE |
| Class-balanced focal | `src/adail/modules/continual.py` | Effective-number weighting |
| Logit distillation | `src/adail/modules/continual.py` | Kelas lama, optional memory/confidence mask |
| Old-vs-new margin ranking | `src/adail/modules/continual.py`, `src/adail/trainer.py` | Margin pada similarity cosine (`ranking_space`), optional ordinal margin |
| Adaptive Memory Allocation | `src/adail/memory.py` | Uncertainty, forgetting, rarity, capacity redistribution, `memory_budget` global/per kelas terlihat |
| Hybrid exemplar selection | `src/adail/memory.py` | Herding, uncertainty, dan k-center diversity |
| Pembatasan akses data lama | `src/adail/trainer.py` | Task > 1 hanya current task + exemplar memory |
| Alternating objectives | `src/adail/trainer.py` | Fase `new`/`old`; row kelas baru dibekukan tepat (nilai + momentum) |
| Prototype Evolution | `src/adail/modules/prototype.py` | EMA prototype update |
| Differentiable separation | `src/adail/modules/prototype.py` | Live batch centroid memberi gradien ke backbone |
| Ordinal Constraint Learning | `src/adail/modules/ordinal.py` | Expected ordinal + MAE loss |
| MAE & QWK | `src/adail/metrics.py` | Evaluasi dataset ordinal |
| Average incremental accuracy, old→new confusion | `src/adail/metrics.py`, `src/adail/trainer.py` | Definisi LO2LN (berbobot sampel) dan bias ke kelas baru |
| Safe checkpoint | `src/adail/trainer.py`, `src/adail/memory.py` | `weights_only=True`, legacy Exemplar allowlist |
| Optimizer SGD + momentum | `src/adail/config.py`, `src/adail/trainer.py` | Default sesuai proposal |
| CLI siap pakai | `src/adail/cli.py` | Config diteruskan melalui `ADAILNet.from_config` |
| Config eksperimen | `configs/*.yaml` | Seluruh opsi enhanced ditulis eksplisit |
| LO2LN external adapter | `src/adail/baselines/lo2ln.py` | Command generator, tidak menyalin source LO2LN |
| Report ADAIL vs LO2LN | `src/adail/benchmarking/comparison.py` | Markdown/CSV report memakai average incremental accuracy |

## Protocol Alignment

Training mengikuti protocol berikut:

1. **Basis task**: model belajar kelas awal sesuai skenario, misalnya `4-1` berarti 4 kelas basis.
2. **Teacher snapshot**: model task sebelumnya dibekukan sebelum representasi diperluas.
3. **Incremental task**: kelas baru dilatih dengan data task berjalan dan exemplar memory; full historical dataset tidak dipanggil lagi.
4. **Alternating phase**: objective `new` dan `old` dipilih per epoch sesuai konfigurasi.
5. **Post-task update**: prototype direfresh, model dievaluasi, lalu quota dan exemplar memory dibangun ulang.

Metrik yang tersimpan di `history.json` meliputi accuracy matrix, average incremental accuracy (berbobot sampel), average accuracy (berbobot task), average forgetting, balanced accuracy, old→new confusion, class forgetting, memory quotas, MAE, QWK, log loss per epoch, phase, dan jumlah cabang representasi.
