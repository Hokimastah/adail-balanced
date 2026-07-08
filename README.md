# ADAIL: Adaptive Drift-Aware Incremental Learning

**ADAIL** adalah library PyTorch untuk **Class-Incremental Learning (CIL) pada klasifikasi citra medis**. Repository ini disusun agar siap diunggah ke GitHub, diinstal dengan `pip`, dan digunakan sebagai library Python maupun command-line tool.

> Status: implementasi riset. Library ini bukan alat diagnosis klinis dan harus divalidasi secara medis, statistik, dan regulatori sebelum dipakai dalam sistem kesehatan nyata.

---

## Inti Metode

ADAIL mengikuti rancangan proposal *Adaptive Drift-Aware Incremental Learning untuk Klasifikasi Citra Medis* dengan empat komponen utama:

1. **Drift-Aware Feature Alignment**  
   Menyelaraskan fitur model saat ini dengan fitur frozen teacher dari task sebelumnya untuk menekan *temporal feature drift*.

2. **Adaptive Memory Allocation**  
   Membagi kuota exemplar secara dinamis berdasarkan:
   - uncertainty,
   - forgetting rate,
   - class rarity.

3. **Prototype Evolution**  
   Mempertahankan prototipe kelas dengan pembaruan exponential moving average agar stabil, tetapi tetap adaptif.

4. **Ordinal Constraint Learning**  
   Menjaga struktur progresi penyakit, terutama untuk dataset ordinal seperti EyePACS: `No DR -> Mild -> Moderate -> Severe -> Proliferative`.

Total loss:

```text
L_total = L_ce
        + lambda_drift   * L_drift
        + lambda_proto   * L_proto
        + lambda_sep     * L_sep
        + lambda_ordinal * L_ordinal
```

---

## Instalasi

```bash
git clone https://github.com/your-username/ADAIL.git
cd ADAIL
python -m pip install -e .
```

Dengan dependency development:

```bash
python -m pip install -e ".[dev]"
pytest
```

---

## Struktur Dataset

ADAIL memakai format `torchvision.datasets.ImageFolder`:

```text
dataset_root/
  train/
    class_0/
      image_001.png
    class_1/
      image_002.png
  val/          # optional
    class_0/
    class_1/
  test/
    class_0/
    class_1/
```

Pastikan urutan folder/class sesuai label yang ingin dipakai. Untuk EyePACS ordinal, urutan yang disarankan:

```text
0: no_dr
1: mild
2: moderate
3: severe
4: proliferative
```

---

## Dataset Preset Proposal

Lihat preset:

```bash
adail list-presets
```

Preset yang tersedia:

| Preset | Jumlah kelas | Default scenario | Ordinal | Keterangan |
|---|---:|---|---|---|
| `cch5000` | 8 | `4-1`, `4-2` | Tidak | Histologi kolorektal, balanced |
| `ham10000` | 7 | `3-1`, `3-2` | Tidak | Dermatoskopi, long-tail imbalance |
| `eyepacs` | 5 | `3-1`, `3-2` | Ya | Retinopati diabetik, progresi stadium |

Config siap pakai tersedia di folder `configs/`:

```text
configs/cch5000_4_1.yaml
configs/cch5000_4_2.yaml
configs/ham10000_3_1.yaml
configs/ham10000_3_2.yaml
configs/eyepacs_3_1.yaml
configs/eyepacs_3_2.yaml
```

---

## Quick Start: CLI

Membuat config proposal-aligned:

```bash
adail init-config --dataset eyepacs --scenario 3-1 --output config.yaml
```

Training dengan YAML:

```bash
adail train \
  --config configs/eyepacs_3_1.yaml \
  --data /path/to/eyepacs_imagefolder \
  --output runs/eyepacs_3_1 \
  --epochs 5
```

Training langsung tanpa YAML:

```bash
adail train \
  --dataset ham10000 \
  --data /path/to/ham10000_imagefolder \
  --output runs/ham10000_3_2 \
  --scenario 3-2 \
  --epochs 5 \
  --batch-size 32 \
  --memory-size 2000
```

Output utama:

```text
runs/<experiment>/
  config.yaml
  history.json
  adail_last.pt
```

---

## Quick Start: Python API

```python
import torch
from adail import ADAILConfig, ADAILNet, ADAILTrainer
from adail.data import build_imagefolder_loaders, ClassIncrementalScenario

config = ADAILConfig(
    num_classes=5,
    memory_size=2000,
    use_ordinal=True,
    lambda_drift=1.0,
    lambda_proto=0.2,
    lambda_sep=0.1,
    lambda_ordinal=0.1,
)
config.data.scenario = "3-1"

loaders = build_imagefolder_loaders(
    root="/path/to/eyepacs_imagefolder",
    image_size=224,
    batch_size=32,
    num_workers=4,
)

scenario = ClassIncrementalScenario.from_string(
    num_classes=config.num_classes,
    spec=config.data.scenario,
)

model = ADAILNet(
    num_classes=config.num_classes,
    pretrained_backbone=True,
    prototype_momentum=config.prototype_momentum,
    prototype_margin=config.prototype_margin,
)

trainer = ADAILTrainer(
    model=model,
    config=config,
    device="cuda" if torch.cuda.is_available() else "cpu",
)

history = trainer.fit_scenario(
    train_dataset=loaders.train.dataset,
    test_dataset=loaders.test.dataset,
    scenario=scenario,
    epochs_per_task=5,
    batch_size=32,
)
print(history["average_accuracy"], history["average_forgetting"])
```

---

## Protocol CIL yang Diikuti

ADAIL memakai protokol class-incremental:

1. **Task basis** melatih kelas awal, misalnya `4-1` berarti 4 kelas pertama sebagai basis.
2. **Task inkremental** hanya menerima kelas baru pada task berjalan.
3. Untuk kelas lama, trainer hanya memakai **exemplar memory**, bukan seluruh dataset historis.
4. Pada setiap task, model menyimpan frozen teacher dari task sebelumnya untuk drift alignment.
5. Prototype dan memory diperbarui setelah task selesai.

---

## Metrik Evaluasi

Tersedia di `adail.metrics`:

- `accuracy`
- `average_accuracy`
- `average_forgetting`
- `per_class_accuracy`
- `mean_absolute_error`
- `quadratic_weighted_kappa`

`history.json` menyimpan:

- accuracy matrix per task,
- average accuracy,
- average forgetting,
- per-class forgetting,
- memory quota per task,
- MAE dan QWK untuk evaluasi ordinal.

---

## Membandingkan ADAIL dengan LO2LN

Repository resmi LO2LN dapat dipakai sebagai baseline eksternal:

```bash
git clone https://github.com/EvelynChee/LO2LN.git external/LO2LN
```

Buat command LO2LN yang sejajar dengan skenario proposal:

```bash
adail lo2ln-command \
  --lo2ln-root external/LO2LN \
  --dataset eyepacs \
  --scenario 3-1 \
  --data-path data/diabetic \
  --device 0 \
  --output-script scripts/run_lo2ln_eyepacs_3_1.sh
```

Buat tabel perbandingan dari hasil ADAIL dan angka LO2LN:

```bash
adail compare-results \
  --dataset eyepacs \
  --scenario 3-1 \
  --adail-history runs/adail_eyepacs_3_1/history.json \
  --output reports/compare_eyepacs_3_1.md
```

Panduan lengkap ada di `docs/LO2LN_COMPARISON.md`.

---

## Struktur Repository

```text
ADAIL/
  configs/                 # konfigurasi eksperimen proposal
  docs/                    # formula dan mapping implementasi
  examples/                # contoh penggunaan
  benchmarks/             # template eksperimen ADAIL vs LO2LN
  docs/                    # formula, mapping, dan panduan LO2LN comparison
  scripts/                 # helper benchmark/report
  src/adail/
    baselines/             # adapter command LO2LN eksternal
    benchmarking/          # pembuat tabel laporan perbandingan
    data/                  # ImageFolder, CIL scenario, preset dataset
    models/                # ADAILNet dan ResNet-18 backbone
    modules/               # drift, prototype, ordinal losses
    memory.py              # adaptive exemplar memory
    metrics.py             # metrik CIL dan ordinal
    trainer.py             # training loop CIL
    cli.py                 # command-line interface
  tests/                   # unit tests
```

---

## Contoh Synthetic

Untuk cek cepat tanpa dataset medis:

```bash
python examples/minimal_synthetic.py
```

---

## Catatan Reproduksibilitas

Default library mengikuti proposal:

- backbone ResNet-18 pretrained ImageNet,
- optimizer SGD dengan momentum,
- input 2D image classification,
- skenario CIL `4-1`, `4-2`, `3-1`, `3-2`,
- memory exemplar adaptif,
- evaluasi accuracy, forgetting, MAE, dan QWK.

Repository ini juga menyediakan adapter pembanding untuk **LO2LN** sebagai baseline eksternal. Kode LO2LN tidak disalin ke package ADAIL; adapter hanya membuat command resmi agar eksperimen dapat dijalankan fair dan hasilnya dapat dirangkum bersama ADAIL.
