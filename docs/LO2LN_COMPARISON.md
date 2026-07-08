# Membandingkan ADAIL dengan LO2LN

Dokumen ini menjelaskan cara menjalankan eksperimen pembanding antara **ADAIL** dan repository resmi **LO2LN**: `EvelynChee/LO2LN`.

## Ringkasan LO2LN yang Dipakai sebagai Baseline

Berdasarkan repository resminya, LO2LN adalah implementasi dari paper **“Leveraging Old Knowledge to Continually Learn New Classes in Medical Images”**. Implementasi utama metode mereka diarahkan ke `inclearn/models/der.py`, yaitu framework berbasis **Dynamically Expandable Representations**. Repository tersebut juga menyediakan baseline iCaRL, UCIR, PODNet, dan DER melalui folder `options/`.

LO2LN diuji pada tiga dataset yang sama dengan proposal ADAIL:

| Dataset | Nama folder LO2LN | Skenario utama |
|---|---|---|
| CCH5000 | `data/colorectal` | `4-1`, `4-2` |
| HAM10000 | `data/ham10000` | `3-1`, `3-2` |
| EyePACS | `data/diabetic` | `3-1` dilaporkan resmi; `3-2` bisa dicoba dengan increment 2 |

Nilai ringkasan yang dilaporkan di README LO2LN:

| Dataset | New classes per step | Average incremental accuracy |
|---|---:|---:|
| CCH5000 | 1 | 94.5 |
| CCH5000 | 2 | 94.6 |
| HAM10000 | 1 | 78.1 |
| HAM10000 | 2 | 82.0 |
| EyePACS | 1 | 81.9 |

## Catatan Lisensi

Source code LO2LN tidak disalin ke repository ADAIL. Repo ini hanya menyediakan **adapter command** dan **template benchmarking** agar kamu bisa menjalankan LO2LN sebagai baseline eksternal. Ini lebih aman untuk GitHub karena repository LO2LN tidak terlihat menyertakan file lisensi eksplisit pada listing utama.

## Menyiapkan LO2LN sebagai External Baseline

Dari root repository ADAIL:

```bash
git clone https://github.com/EvelynChee/LO2LN.git external/LO2LN
cd external/LO2LN
python -m pip install -r requirements.txt
```

LO2LN resminya diuji dengan Python 3.8, Ubuntu 20.04, dan CUDA 11.4. Untuk reproducibility, lebih baik jalankan LO2LN di environment terpisah dari ADAIL.

## Menjalankan ADAIL

Contoh EyePACS `3-1`:

```bash
adail train \
  --config configs/eyepacs_3_1.yaml \
  --data /path/to/eyepacs_imagefolder \
  --output runs/adail_eyepacs_3_1 \
  --epochs 5
```

Output ADAIL:

```text
runs/adail_eyepacs_3_1/history.json
```

## Membuat Command LO2LN yang Setara

Gunakan CLI bawaan ADAIL:

```bash
adail lo2ln-command \
  --lo2ln-root external/LO2LN \
  --dataset eyepacs \
  --scenario 3-1 \
  --data-path data/diabetic \
  --device 0 \
  --output-script scripts/run_lo2ln_eyepacs_3_1.sh
```

Output-nya akan berupa command seperti:

```bash
cd external/LO2LN && python3 -m inclearn --options options/der/derE3_diabetic.yaml options/data/diabetic_3orders.yaml --initial-increment 3 --increment 1 --fixed-memory --workers 4 --batch-size 128 --device 0 --label lo2ln_eyepacs_3_1 --data-path data/diabetic
```

Jalankan:

```bash
bash scripts/run_lo2ln_eyepacs_3_1.sh
```

## Membuat Tabel Perbandingan

Jika ADAIL sudah menghasilkan `history.json`, kamu bisa membuat laporan awal dengan nilai LO2LN dari README resmi:

```bash
adail compare-results \
  --dataset eyepacs \
  --scenario 3-1 \
  --adail-history runs/adail_eyepacs_3_1/history.json \
  --output reports/compare_eyepacs_3_1.md
```

Atau masukkan nilai hasil training LO2LN sendiri:

```bash
adail compare-results \
  --dataset ham10000 \
  --scenario 3-2 \
  --adail-history runs/adail_ham10000_3_2/history.json \
  --lo2ln-average-accuracy 82.0 \
  --lo2ln-average-forgetting 0.0 \
  --output reports/compare_ham10000_3_2.csv
```

## Fairness Checklist

Agar perbandingan kuat untuk TA/sidang:

1. Gunakan class order yang sama antara ADAIL dan LO2LN.
2. Gunakan split train/test yang sama.
3. Samakan memory budget.
4. Laporkan average incremental accuracy dan average forgetting.
5. Untuk EyePACS, tambahkan MAE dan QWK karena ADAIL punya ordinal constraint.
6. Jalankan minimal 3 class orders bila mengikuti protokol LO2LN.
7. Tulis jelas bahwa LO2LN memakai ekspansi representasi, sedangkan ADAIL menambahkan drift alignment, adaptive memory, prototype evolution, dan ordinal constraint.
