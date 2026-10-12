# ADAIL: Adaptive Drift-Aware Incremental Learning

**ADAIL** adalah library PyTorch untuk **Class-Incremental Learning (CIL) pada klasifikasi citra medis**. Library ini dapat diinstal dengan `pip` dan dipakai sebagai library Python maupun command-line tool.

> Status: implementasi riset. Library ini bukan alat diagnosis klinis dan harus divalidasi secara medis, statistik, dan regulatori sebelum dipakai dalam sistem kesehatan nyata.

---

## Inti Metode

ADAIL mengikuti rancangan proposal *Adaptive Drift-Aware Incremental Learning untuk Klasifikasi Citra Medis* dan memperluasnya dengan strategi stabilitas/plastisitas yang terinspirasi LO2LN (tanpa menyalin kode LO2LN):

1. **Drift-Aware Feature Alignment** — menyelaraskan fitur model saat ini dengan fitur frozen teacher dari task sebelumnya.
2. **Adaptive Memory Allocation** — kuota exemplar per kelas dari uncertainty, forgetting rate, dan class rarity.
3. **Prototype Evolution** — prototipe kelas diperbarui dengan exponential moving average.
4. **Ordinal Constraint Learning** — menjaga urutan stadium penyakit pada dataset ordinal seperti EyePACS atau RetinaMNIST.
5. **Selective Representation Expansion** — salinan `layer4` ResNet-18 per task inkremental, digabung lewat gated residual berdimensi tetap.
6. **Alternating Stability/Plasticity Objectives** — fase `new` mempelajari kelas baru; fase `old` membekukan cabang terbaru dan row classifier kelas baru, lalu memperkuat distillation dan margin ranking.
7. **Balanced Classification and Replay** — cosine classifier, weight imprinting, class-balanced focal loss, dan batch sampler yang mencampur data baru dengan exemplar.
8. **Hybrid Exemplar Selection** — herding, uncertainty, dan k-center diversity.

Total loss:

```text
L_total = L_cb_focal
        + lambda_aux     * L_aux
        + lambda_kd      * L_distill
        + lambda_rank    * L_ranking      (margin pada similarity cosine)
        + lambda_drift   * L_drift
        + lambda_proto   * L_compact
        + lambda_sep     * L_sep_diff
        + lambda_ordinal * L_ordinal
```

Rumus lengkap ada di [`docs/FORMULAS.md`](docs/FORMULAS.md).

### Perubahan penting di 0.5.0

| Masalah di 0.4.0 | Perbaikan |
|---|---|
| Margin ranking diterapkan pada logit yang sudah dikali skala cosine (≈10), sehingga margin 0.2 efektif 0.02 | Margin diukur pada similarity cosine (`ranking_space="cosine"`, default) |
| Fase `old` hanya me-nol-kan gradien row kelas baru; momentum dan weight decay tetap menggesernya | Nilai dan momentum row dipulihkan setelah `optimizer.step()` |
| Kelas baru selalu mengisi 50% batch replay, berapa pun jumlah kelas lama | `replay_fraction_mode="proportional"` (opsional) |
| Panjang epoch ikut membesar bersama ukuran kelas baru | `replay_epoch_length="pool"` (opsional) |
| Exemplar disimpan sebagai satu tensor beraugmentasi yang tidak pernah berubah | `replay_reaugment=True` (opsional) |
| Memori penuh sejak task pertama, tidak sebanding dengan `fixed_memory` LO2LN | `memory_budget="per_seen_class"` (opsional) |
| Imprinting, refresh prototipe, dan skor exemplar memakai citra beraugmentasi acak | `fit_scenario(..., train_eval_dataset=...)` (opsional) |
| `history["average_accuracy"]` berbobot task, tetapi dibandingkan dengan angka LO2LN yang berbobot sampel | `history["average_incremental_accuracy"]`; `compare-results` memakai definisi ini |

Opsi bertanda *opsional* defaultnya mempertahankan perilaku 0.4.0, sehingga hasil lama tetap dapat direproduksi.

---

## Instalasi

```bash
git clone https://github.com/Hokimastah/ADAIL.git
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
  val/          # optional
  test/
    class_0/
```

Indeks kelas mengikuti urutan alfabetis nama folder. Untuk EyePACS ordinal, urutan yang disarankan:

```text
0: no_dr
1: mild
2: moderate
3: severe
4: proliferative
```

Bila membandingkan dengan LO2LN, perhatikan bahwa penomoran label LO2LN (dari CSV manifest-nya) bisa berbeda dari penomoran alfabetis ImageFolder; lihat [`docs/LO2LN_COMPARISON.md`](docs/LO2LN_COMPARISON.md).

---

## Dataset Preset

```bash
adail list-presets
```

| Preset | Jumlah kelas | Default scenario | Ordinal | Keterangan |
|---|---:|---|---|---|
| `cch5000` | 8 | `4-1`, `4-2` | Tidak | Histologi kolorektal, balanced |
| `ham10000` | 7 | `3-1`, `3-2` | Tidak | Dermatoskopi, long-tail imbalance |
| `eyepacs` | 5 | `3-1`, `3-2` | Ya | Retinopati diabetik, progresi stadium |

Config siap pakai tersedia di folder `configs/`.

---

## Quick Start: CLI

```bash
adail init-config --dataset eyepacs --scenario 3-1 --output config.yaml

adail train \
  --config configs/eyepacs_3_1.yaml \
  --data /path/to/eyepacs_imagefolder \
  --output runs/eyepacs_3_1 \
  --epochs 5
```

Output utama: `runs/<experiment>/config.yaml`, `history.json`, `adail_last.pt`.

---

## Quick Start: Python API

```python
import torch
from torchvision.datasets import ImageFolder

from adail import ADAILConfig, ADAILNet, ADAILTrainer
from adail.data import ClassIncrementalScenario, build_eval_transform, build_train_transform

config = ADAILConfig(
    num_classes=5,
    memory_size=500,
    use_ordinal=True,
    # Opsi 0.5.0 (opsional):
    replay_fraction_mode="proportional",
    replay_epoch_length="pool",
    replay_reaugment=True,
    memory_budget="per_seen_class",
)

root = "/path/to/eyepacs_imagefolder"
train_dataset = ImageFolder(f"{root}/train", transform=build_train_transform(224))
# Sampel yang sama dengan transform evaluasi, untuk imprinting/prototipe/skor exemplar.
train_eval_dataset = ImageFolder(f"{root}/train", transform=build_eval_transform(224))
test_dataset = ImageFolder(f"{root}/test", transform=build_eval_transform(224))

# Classifier dibuat untuk seluruh kelas sejak awal; kelas yang belum terlihat
# di-mask dari loss dan evaluasi.
model = ADAILNet.from_config(config, pretrained_backbone=True)
trainer = ADAILTrainer(model, config, device="cuda" if torch.cuda.is_available() else "cpu")
scenario = ClassIncrementalScenario.from_string(config.num_classes, "3-1")

history = trainer.fit_scenario(
    train_dataset=train_dataset,
    test_dataset=test_dataset,
    scenario=scenario,
    epochs_per_task=5,
    batch_size=32,
    train_eval_dataset=train_eval_dataset,
)
print(history["average_incremental_accuracy"], history["average_forgetting"])
```

---

## Protokol CIL

1. **Task basis** melatih kelas awal, misalnya `4-1` berarti 4 kelas pertama sebagai basis.
2. **Task inkremental** hanya menerima data kelas baru; kelas lama hanya hadir lewat **exemplar memory**.
3. Sebelum setiap task inkremental, model saat ini dibekukan sebagai teacher untuk drift alignment dan distillation.
4. Bila `dynamic_expansion=True`, cabang representasi baru ditambahkan dan fase `new`/`old` dioptimalkan bergantian.
5. Setelah task selesai: prototipe direfresh, model dievaluasi, lalu kuota dan exemplar memory dibangun ulang.

---

## Metrik Evaluasi

`history` (dan `history.json` dari CLI) menyimpan:

- `average_incremental_accuracy` — rata-rata atas task dari akurasi pada seluruh sampel test kelas terlihat (berbobot sampel; **ini yang dilaporkan LO2LN/iCaRL**);
- `average_accuracy` — rata-rata atas task dari rata-rata akurasi per blok task (berbobot task);
- `average_forgetting`, per-class forgetting, dan accuracy matrix;
- per task: `seen_accuracy`, `seen_balanced_accuracy`, macro precision/recall/F1, MAE, QWK, confusion matrix, `old_to_new_confusion`;
- `average_old_to_new_confusion` — bias rata-rata ke kelas yang baru ditambahkan;
- kuota memori per task dan log loss per epoch (termasuk nama fase `new`/`old`).

---

## Checkpoint

```python
trainer.save_checkpoint("adail_last.pt")
trainer.load_checkpoint("adail_last.pt")
```

Checkpoint hanya berisi tensor dan container primitif, sehingga dapat dimuat dengan `weights_only=True` (PyTorch 2.6+). Dengan `replay_reaugment=True`, exemplar disimpan sebagai indeks dataset; setelah memuat checkpoint, pasang kembali dataset latihnya:

```python
trainer.memory.source_dataset = train_dataset
```

Jangan memuat checkpoint dari sumber yang tidak dipercaya.

---

## Membandingkan ADAIL dengan LO2LN

```bash
git clone https://github.com/EvelynChee/LO2LN.git external/LO2LN

adail lo2ln-command --lo2ln-root external/LO2LN --dataset eyepacs --scenario 3-1 \
  --data-path data/diabetic --device 0 --output-script scripts/run_lo2ln_eyepacs_3_1.sh

adail compare-results --dataset eyepacs --scenario 3-1 \
  --adail-history runs/adail_eyepacs_3_1/history.json \
  --output reports/compare_eyepacs_3_1.md
```

Checklist kewajaran perbandingan (class order, split, memory budget, metrik, jadwal LR) ada di [`docs/LO2LN_COMPARISON.md`](docs/LO2LN_COMPARISON.md). Perbedaan desain yang tetap ada antara ADAIL dan LO2LN dirangkum di [`docs/LO2LN_ENHANCEMENTS.md`](docs/LO2LN_ENHANCEMENTS.md).

---

## Struktur Repository

```text
ADAIL/
  configs/                 # konfigurasi eksperimen
  docs/                    # formula, mapping implementasi, panduan LO2LN
  examples/                # contoh penggunaan
  benchmarks/              # template eksperimen ADAIL vs LO2LN
  scripts/                 # helper benchmark/report
  src/adail/
    baselines/             # adapter command LO2LN eksternal
    benchmarking/          # pembuat tabel laporan perbandingan
    data/                  # dataset, scenario, balanced replay
    models/                # ADAILNet dan expandable ResNet-18
    modules/               # drift, prototype, focal, KD, ranking, ordinal
    memory.py              # adaptive exemplar memory
    metrics.py             # metrik CIL dan ordinal
    trainer.py             # training loop CIL
    cli.py                 # command-line interface
  tests/                   # unit dan integration tests
```

---

## Contoh Synthetic

```bash
python examples/minimal_synthetic.py
```

---

## Catatan Reproduksibilitas

- Backbone ResNet-18 pretrained ImageNet, optimizer SGD + momentum, tanpa scheduler LR.
- `sampler_seed` mengendalikan batch balanced replay; seleksi exemplar `random` memakai seed per kelas yang tetap.
- Jalankan beberapa class order (dan idealnya beberapa seed), lalu laporkan rata-rata ± simpangan baku; selisih antar-varian yang lebih kecil dari simpangan bakunya bukan bukti peningkatan.
