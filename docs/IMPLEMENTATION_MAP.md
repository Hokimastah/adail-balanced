# ADAIL Implementation Map

Dokumen ini memetakan proposal ADAIL ke file repository.

| Konsep proposal | Lokasi implementasi | Status |
|---|---|---|
| ResNet-18 pretrained ImageNet backbone | `src/adail/models/backbone.py` | Siap pakai via Torchvision |
| Skenario Class-Incremental Learning | `src/adail/data/datasets.py` | `4-1`, `4-2`, `3-1`, `3-2`, custom class order |
| Dataset medis 2D CCH5000, HAM10000, EyePACS | `src/adail/data/presets.py` | Preset metadata + default scenarios |
| Drift-Aware Feature Alignment | `src/adail/modules/drift.py` | Learnable `A_t` + L2/MSE feature alignment |
| Frozen previous model | `src/adail/models/adail_model.py` | Snapshot teacher antar-task |
| Adaptive Memory Allocation | `src/adail/memory.py` | Quota berdasarkan uncertainty, forgetting, rarity |
| Pembatasan akses data lama | `src/adail/trainer.py` | Task > 1 hanya memakai current task + exemplar memory |
| Prototype Evolution | `src/adail/modules/prototype.py` | EMA prototype update |
| Prototype Compactness & Separation Loss | `src/adail/modules/prototype.py` | Compactness + margin separation |
| Ordinal Constraint Learning | `src/adail/modules/ordinal.py` | Expected ordinal + MAE loss |
| MAE & QWK untuk progresi penyakit | `src/adail/metrics.py` | Tersedia untuk evaluasi EyePACS/ordinal |
| Optimizer SGD + momentum | `src/adail/config.py`, `src/adail/trainer.py` | Default sesuai proposal |
| CLI siap pakai | `src/adail/cli.py` | `init-config`, `list-presets`, `train` |
| Config eksperimen proposal | `configs/*.yaml` | CCH5000, HAM10000, EyePACS |

## Protocol Alignment

Training mengikuti dua tahap proposal:

1. **Basis task**: model belajar kelas awal sesuai skenario, misalnya `4-1` berarti 4 kelas basis.
2. **Incremental task**: setiap task berikutnya melatih kelas baru dengan gabungan data task berjalan dan exemplar memory; full historical dataset tidak dipanggil lagi.

Metrik yang tersimpan di `history.json` setelah CLI training meliputi accuracy matrix, average accuracy, average forgetting, class forgetting, memory quotas, MAE, dan QWK.

| LO2LN external baseline adapter | `src/adail/baselines/lo2ln.py` | Command generator, tidak menyalin source LO2LN |
| Report ADAIL vs LO2LN | `src/adail/benchmarking/comparison.py`, `scripts/compare_adail_lo2ln.py` | Markdown/CSV comparison report |
| Panduan komparasi LO2LN | `docs/LO2LN_COMPARISON.md` | Protocol dan fairness checklist |
