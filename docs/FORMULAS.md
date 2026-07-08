# ADAIL Formulas

Dokumen ini merangkum komponen matematis yang diimplementasikan dari proposal **Adaptive Drift-Aware Incremental Learning (ADAIL) untuk Klasifikasi Citra Medis**.

## 1. Drift-Aware Feature Alignment

Pada task inkremental `t`, model aktif menghasilkan fitur `f_t(x)`, sedangkan frozen teacher dari task sebelumnya menghasilkan `f_{t-1}(x)`. ADAIL menggunakan fungsi penyelarasan learnable `A_t`:

```text
L_drift = || normalize(f_t(x)) - normalize(A_t(f_{t-1}(x))) ||_2^2
```

Implementasi: `src/adail/modules/drift.py`.

## 2. Adaptive Memory Allocation

Kuota exemplar kelas `c` dihitung dari tiga sinyal proposal: uncertainty, forgetting rate, dan class rarity.

```text
score_c = alpha * U_c + beta * F_c + gamma * R_c
quota_c = floor(memory_size * score_c / sum_j score_j)
```

Keterangan:

- `U_c`: rata-rata entropy prediksi kelas `c`.
- `F_c`: penurunan akurasi kelas `c` dari nilai terbaik historisnya.
- `R_c`: nilai kelangkaan kelas, dihitung terbalik dari jumlah sampel kelas yang telah diamati.

Implementasi: `src/adail/memory.py` dan `ADAILTrainer.update_memory()`.

## 3. Prototype Evolution

Prototipe kelas `p_c` diperbarui dengan exponential moving average agar stabil, namun tetap adaptif terhadap distribusi fitur task baru.

```text
p_c <- m * p_c + (1 - m) * mean(f_i | y_i = c)
```

Implementasi: `src/adail/modules/prototype.py`.

## 4. Prototype Compactness dan Separation Loss

Compactness menjaga fitur dekat ke prototipe kelasnya.

```text
L_proto = mean || normalize(f_i) - normalize(p_{y_i}) ||_2^2
```

Separation memberi penalti jika dua prototipe terlalu dekat.

```text
L_sep = mean_{i != j} max(0, margin - || normalize(p_i) - normalize(p_j) ||_2)^2
```

Implementasi: `PrototypeBank.compactness_loss()` dan `PrototypeBank.separation_loss()`.

## 5. Ordinal Constraint Learning

Untuk dataset berurutan secara klinis seperti EyePACS, ADAIL memakai expected ordinal prediction:

```text
pred = sum_k softmax(logits)_k * k
L_ordinal = mean | pred - y |
```

Implementasi: `src/adail/modules/ordinal.py`.

## 6. Total Objective

```text
L_total = L_ce
        + lambda_drift   * L_drift
        + lambda_proto   * L_proto
        + lambda_sep     * L_sep
        + lambda_ordinal * L_ordinal
```

## 7. Evaluasi

Metrik yang tersedia:

- Average Accuracy
- Average Forgetting
- per-class forgetting estimate
- MAE
- Quadratic Weighted Kappa (QWK)
